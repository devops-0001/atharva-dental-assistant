import os
import time
import httpx
from fastapi import FastAPI, Query, Response
from pydantic import BaseModel
from typing import List, Dict, Any

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from prompt_templates import build_messages

RETRIEVER_URL = os.getenv("RETRIEVER_URL", "http://atharva-retriever.atharva-ml.svc.cluster.local:8001")
VLLM_URL      = os.getenv("VLLM_URL",      "http://atharva-vllm.atharva-ml.svc.cluster.local:8000")
MODEL_NAME    = os.getenv("MODEL_NAME",    "smollm2-135m-atharva")

MAX_CTX_SNIPPETS = int(os.getenv("MAX_CTX_SNIPPETS", "3"))
MAX_CTX_CHARS    = int(os.getenv("MAX_CTX_CHARS", "2400"))

app = FastAPI()

# -----------------------------
# Prometheus metrics
# -----------------------------
REQS = Counter("chat_requests_total", "Total Chat API requests", ["route"])
ERRS = Counter("chat_errors_total", "Total Chat API errors", ["stage"])
E2E_LAT = Histogram(
    "chat_end_to_end_latency_seconds",
    "End-to-end /chat latency in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13),
)
RAG_LAT = Histogram(
    "rag_retrieval_latency_seconds",
    "Retriever call latency in seconds",
    buckets=(0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3),
)
VLLM_LAT = Histogram(
    "vllm_request_latency_seconds",
    "vLLM chat/completions call latency in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13),
)

TOK_PROMPT = Gauge("chat_prompt_tokens", "Prompt tokens for the last completed /chat request")
TOK_COMPLETION = Gauge("chat_completion_tokens", "Completion tokens for the last completed /chat request")
TOK_TOTAL = Gauge("chat_total_tokens", "Total tokens for the last completed /chat request")


class ChatRequest(BaseModel):
    question: str
    k: int = 4
    max_tokens: int = 200
    temperature: float = 0.1
    debug: bool = False  # when true, include prompt/messages in response


def _label(meta: Dict[str, Any]) -> str:
    did = (meta or {}).get("doc_id")
    sec = (meta or {}).get("section")
    if not did:
        return "unknown"
    return f"{did}#{sec}" if sec and sec != "full" else did


def _collect_citations(hits: List[Dict[str, Any]]) -> List[str]:
    seen, out = set(), []
    for h in hits:
        lab = _label(h.get("meta"))
        if lab not in seen:
            seen.add(lab)
            out.append(lab)
    return out


def _normalize_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Drop recent_queries from grounding context (they’re noisy)
    filt = []
    for h in hits:
        did = ((h.get("meta") or {}).get("doc_id") or "").lower()
        if did.startswith("recent_queries.jsonl"):
            continue
        filt.append(h)

    # Prefer those with text first
    filt.sort(key=lambda h: (h.get("text") is None), reverse=False)

    # Dedup by label
    seen, dedup = set(), []
    for h in filt:
        lab = _label(h.get("meta"))
        if lab in seen:
            continue
        seen.add(lab)
        dedup.append(h)

    # Trim by count and char budget
    total = 0
    trimmed = []
    for h in dedup:
        txt = h.get("text") or (h.get("meta") or {}).get("text") or ""
        if len(trimmed) < MAX_CTX_SNIPPETS and total + len(txt) <= MAX_CTX_CHARS:
            trimmed.append(h)
            total += len(txt)
        if len(trimmed) >= MAX_CTX_SNIPPETS:
            break

    return trimmed


def _strip_existing_source(txt: str) -> str:
    lines = txt.rstrip().splitlines()
    kept = [ln for ln in lines if not ln.strip().lower().startswith("source:")]
    return "\n".join(kept).rstrip()


@app.get("/health")
def health():
    return {"ok": True, "retriever": RETRIEVER_URL, "vllm": VLLM_URL, "model": MODEL_NAME}


@app.get("/metrics")
def metrics():
    # Expose Prometheus metrics
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/dryrun")
def dryrun(q: str = Query(..., alias="question"), k: int = 4):
    """Build exactly what /chat would send to vLLM, but don’t call vLLM."""
    REQS.labels(route="/dryrun").inc()
    with httpx.Client(timeout=30) as cx:
        t_r0 = time.time()
        try:
            r = cx.post(f"{RETRIEVER_URL}/search", json={"query": q, "k": k})
            r.raise_for_status()
            raw_hits = r.json().get("hits", [])
        except Exception:
            ERRS.labels(stage="retriever").inc()
            raise
        finally:
            RAG_LAT.observe(time.time() - t_r0)

    ctx_hits   = _normalize_hits(raw_hits)
    citations  = _collect_citations(ctx_hits)
    messages   = build_messages(q, ctx_hits)

    # Also surface the precise snippets we used (label + text)
    used_snippets = []
    for h in ctx_hits:
        meta = h.get("meta") or {}
        used_snippets.append({
            "label": _label(meta),
            "text": h.get("text") or meta.get("text") or ""
        })

    return {
        "question": q,
        "citations": citations,
        "used_snippets": used_snippets,   # what the model will actually see
        "messages": messages,             # the exact OpenAI Chat payload
        "note": "This is a dry run; no LLM call was made."
    }


@app.post("/chat")
def chat(req: ChatRequest):
    REQS.labels(route="/chat").inc()
    t0 = time.time()

    # 1) retrieve
    with httpx.Client(timeout=30) as cx:
        t_r0 = time.time()
        try:
            r = cx.post(f"{RETRIEVER_URL}/search", json={"query": req.question, "k": req.k})
            r.raise_for_status()
            raw_hits = r.json().get("hits", [])
        except Exception:
            ERRS.labels(stage="retriever").inc()
            raise
        finally:
            RAG_LAT.observe(time.time() - t_r0)

    # 2) normalize + citations
    ctx_hits  = _normalize_hits(raw_hits)
    citations = _collect_citations(ctx_hits)

    # 3) build messages with actual snippet text
    messages = build_messages(req.question, ctx_hits)

    # 4) call vLLM (OpenAI-compatible)
    temperature = max(0.0, min(req.temperature, 0.5))
    max_tokens  = min(req.max_tokens, 256)
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": max_tokens,
        "stream": False,
    }

    with httpx.Client(timeout=120) as cx:
        t_llm0 = time.time()
        try:
            rr = cx.post(f"{VLLM_URL}/v1/chat/completions", json=payload)
            rr.raise_for_status()
            data = rr.json()
        except Exception:
            ERRS.labels(stage="vllm").inc()
            raise
        finally:
            VLLM_LAT.observe(time.time() - t_llm0)

    content = data["choices"][0]["message"]["content"]
    usage   = data.get("usage", {})
    dt      = time.time() - t0

    content = _strip_existing_source(content)
    content = content + ("\nSource: " + "; ".join(citations) if citations else "\nSource: (none)")

    # Update token gauges (best-effort)
    try:
        TOK_PROMPT.set(float(usage.get("prompt_tokens", 0) or 0))
        TOK_COMPLETION.set(float(usage.get("completion_tokens", 0) or 0))
        TOK_TOTAL.set(float(usage.get("total_tokens", 0) or 0))
    except Exception:
        # Avoid failing the request if usage is missing or malformed
        pass

    # Observe end-to-end latency
    E2E_LAT.observe(dt)

    resp = {
        "answer": content,
        "citations": citations,
        "latency_seconds": round(dt, 3),
        "usage": usage,
    }

    # 5) optional debug payload so you can inspect exactly what was sent
    if req.debug:
        used_snippets = []
        for h in ctx_hits:
            meta = h.get("meta") or {}
            used_snippets.append({
                "label": _label(meta),
                "text": h.get("text") or meta.get("text") or ""
            })
        resp["debug"] = {
            "messages": messages,         # exact system+user messages sent
            "used_snippets": used_snippets,
            "raw_hits": raw_hits[:10],    # original retriever output (trimmed)
            "payload_model": MODEL_NAME,
            "payload_temperature": temperature,
            "payload_max_tokens": max_tokens,
        }

    return resp
