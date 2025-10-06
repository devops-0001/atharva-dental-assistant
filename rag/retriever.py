import os
import json
from pathlib import Path
from typing import List, Optional, Tuple, Any

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

# --- Prometheus (only new dependency) ---
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

BACKEND    = os.getenv("BACKEND", "dense")  # "sparse" or "dense"
INDEX_PATH = Path(os.getenv("INDEX_PATH", "/mnt/project/atharva-dental-assistant/artifacts/rag/index.faiss"))
META_PATH  = Path(os.getenv("META_PATH",  "/mnt/project/atharva-dental-assistant/artifacts/rag/meta.json"))
MODEL_DIR  = os.getenv("MODEL_DIR")  # optional for dense
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

app = FastAPI(title=f"Atharva Retriever ({BACKEND})")

class SearchRequest(BaseModel):
    query: str
    k: int = 4

_ready_reason = "starting"
_model = None; _index = None; _meta: List[dict] = []
_vec = None; _X = None  # sparse objects

# ------------------ Prometheus metrics (added) ------------------
REQS_TOTAL = Counter("retriever_requests_total", "Total /search requests")
ERRS_TOTAL = Counter("retriever_errors_total", "Total retriever errors", ["stage"])
E2E_LAT = Histogram(
    "retriever_search_latency_seconds",
    "End-to-end /search latency (s)",
    buckets=(0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2),
)
# Dense sub-steps
ENC_LAT = Histogram(
    "retriever_dense_encode_latency_seconds",
    "SentenceTransformer.encode latency (s)",
    buckets=(0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2),
)
FAISS_LAT = Histogram(
    "retriever_dense_faiss_latency_seconds",
    "FAISS search latency (s)",
    buckets=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1),
)
# Sparse sub-steps
VEC_LAT = Histogram(
    "retriever_sparse_vectorize_latency_seconds",
    "TF-IDF vectorizer.transform latency (s)",
    buckets=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1),
)
DOT_LAT = Histogram(
    "retriever_sparse_dot_latency_seconds",
    "Sparse dot/matmul latency (s)",
    buckets=(0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1),
)
# Load-time & sizes
MODEL_LOAD_SEC = Gauge("retriever_model_load_seconds", "Dense model load time (s)")
INDEX_LOAD_SEC = Gauge("retriever_index_load_seconds", "FAISS index load time (s)")
SPARSE_VEC_LOAD_SEC = Gauge("retriever_sparse_vectorizer_load_seconds", "TF-IDF vectorizer load time (s)")
SPARSE_MAT_LOAD_SEC = Gauge("retriever_sparse_matrix_load_seconds", "TF-IDF matrix load time (s)")
INDEX_ITEMS = Gauge("retriever_index_items", "Items in index/matrix")
META_ITEMS = Gauge("retriever_meta_items", "Number of meta records")

# ------------------ Utils ------------------

def _normalize_meta_loaded(data: Any) -> List[dict]:
    """
    Accepts various shapes of meta.json and returns a list of entries.
    Supported:
      - list[dict]
      - {"items": [...]}  (common pattern)
      - {"hits": [...]}   (fallback)
    """
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
            # keep original behavior
        if "items" in data and isinstance(data["items"], list):
            return data["items"]
        if "hits" in data and isinstance(data["hits"], list):
            return data["hits"]
    raise ValueError("META_PATH must contain a list or a dict with 'items'/'hits'.")

def _parse_doc_and_section(path: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Parse labels from meta.path:
      - 'treatments.json#0' -> ('treatments.json', '0')
      - 'faq.md'            -> ('faq.md', None)
      - 'policies/emergency.md' -> ('policies/emergency.md', None)
    """
    if not path:
        return "unknown", None
    if "#" in path:
        d, s = path.split("#", 1)
        return d, s
    return path, None

def _extract_text(m: dict) -> Optional[str]:
    """Try common keys for stored chunk text."""
    return m.get("text") or m.get("chunk") or m.get("content")

def _enrich_hit(idx: int, score: float) -> dict:
    """
    Build a single enriched hit from meta[idx].
    """
    if idx < 0 or idx >= len(_meta):
        doc_id, section, path, typ, txt = "unknown", None, None, None, None
    else:
        m   = _meta[idx] or {}
        path = m.get("path")
        typ  = m.get("type")
        doc_id, section = _parse_doc_and_section(path)
        txt = _extract_text(m)

    hit = {
        "score": float(score),
        "meta": {
            "doc_id": doc_id,
            "section": section,
            "path": path,
            "type": typ,
        },
    }
    if txt:
        hit["text"] = txt
    return hit

# ------------------ Loaders (unchanged behavior; just timed gauges) ------------------

def _load_dense():
    global _model, _index, _meta
    try:
        import time as _t
        import faiss
        from sentence_transformers import SentenceTransformer

        t0 = _t.time()
        _model = SentenceTransformer(MODEL_DIR) if (MODEL_DIR and Path(MODEL_DIR).exists()) else SentenceTransformer(MODEL_NAME)
        MODEL_LOAD_SEC.set(_t.time() - t0)

        t1 = _t.time()
        _index = faiss.read_index(str(INDEX_PATH))
        INDEX_LOAD_SEC.set(_t.time() - t1)

        _meta = _normalize_meta_loaded(json.loads(META_PATH.read_text(encoding="utf-8")))
        META_ITEMS.set(len(_meta) if isinstance(_meta, list) else 0)

        # best-effort size (for FAISS)
        try:
            INDEX_ITEMS.set(int(getattr(_index, "ntotal", len(_meta))))
        except Exception:
            INDEX_ITEMS.set(len(_meta))

        return None
    except Exception as e:
        return f"dense load error: {e}"

def _load_sparse():
    global _vec, _X, _meta
    try:
        import time as _t
        import joblib
        from scipy import sparse

        vec_p = Path(os.getenv("VEC_PATH", "/mnt/project/atharva-dental-assistant/artifacts/rag/tfidf_vectorizer.joblib"))
        X_p   = Path(os.getenv("MAT_PATH", "/mnt/project/atharva-dental-assistant/artifacts/rag/tfidf_matrix.npz"))

        t0 = _t.time()
        _vec = joblib.load(vec_p)
        SPARSE_VEC_LOAD_SEC.set(_t.time() - t0)

        t1 = _t.time()
        _X = sparse.load_npz(X_p)  # assume rows L2-normalized; dot == cosine
        SPARSE_MAT_LOAD_SEC.set(_t.time() - t1)

        _meta = _normalize_meta_loaded(json.loads(META_PATH.read_text(encoding="utf-8")))
        META_ITEMS.set(len(_meta) if isinstance(_meta, list) else 0)

        try:
            INDEX_ITEMS.set(int(getattr(_X, "shape", (0, 0))[0]))
        except Exception:
            INDEX_ITEMS.set(len(_meta))

        return None
    except Exception as e:
        return f"sparse load error: {e}"

@app.on_event("startup")
def startup():
    global _ready_reason
    _ready_reason = _load_sparse() if BACKEND == "sparse" else _load_dense()

# ------------------ Endpoints (original behavior preserved) ------------------

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/ready")
def ready():
    return {"ready": _ready_reason is None, "reason": _ready_reason}

@app.post("/reload")
def reload_index():
    global _ready_reason
    _ready_reason = _load_sparse() if BACKEND == "sparse" else _load_dense()
    if _ready_reason is not None:
        raise HTTPException(status_code=503, detail=_ready_reason)
    return {"reloaded": True}

# --- /metrics added (Prometheus text format) ---
@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/search")
def search(req: SearchRequest):
    if _ready_reason is not None:
        raise HTTPException(status_code=503, detail=_ready_reason)

    import time as _t
    t0 = _t.time()
    REQS_TOTAL.inc()

    k = max(1, min(int(req.k), 20))

    if BACKEND == "sparse":
        try:
            import numpy as np
            t_vec0 = _t.time()
            q = _vec.transform([req.query])
        except Exception:
            ERRS_TOTAL.labels(stage="sparse_vectorize").inc()
            raise
        finally:
            VEC_LAT.observe(_t.time() - t_vec0)

        try:
            t_dot0 = _t.time()
            scores = (_X @ q.T).toarray().ravel()  # cosine since rows normalized
        except Exception:
            ERRS_TOTAL.labels(stage="sparse_dot").inc()
            raise
        finally:
            DOT_LAT.observe(_t.time() - t_dot0)

        if scores.size == 0:
            E2E_LAT.observe(_t.time() - t0)
            return {"hits": []}
        # get top-k indices by score desc
        k_eff = min(k, scores.size)
        top = np.argpartition(-scores, range(k_eff))[:k_eff]
        top = top[np.argsort(-scores[top])]
        hits = [
            _enrich_hit(int(i), float(scores[int(i)]))
            for i in top
            if scores[int(i)] > 0
        ]
        E2E_LAT.observe(_t.time() - t0)
        return {"hits": hits}

    # dense (unchanged behavior; with timing)
    try:
        import faiss
        import numpy as np
        t_enc0 = _t.time()
        v = _model.encode([req.query], normalize_embeddings=True)  # IP ~ cosine
    except Exception:
        ERRS_TOTAL.labels(stage="dense_encode").inc()
        raise
    finally:
        ENC_LAT.observe(_t.time() - t_enc0)

    try:
        t_faiss0 = _t.time()
        D, I = _index.search(v.astype("float32"), k)
    except Exception:
        ERRS_TOTAL.labels(stage="dense_faiss_search").inc()
        raise
    finally:
        FAISS_LAT.observe(_t.time() - t_faiss0)

    hits = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx == -1:
            continue
        hits.append(_enrich_hit(int(idx), float(score)))

    E2E_LAT.observe(_t.time() - t0)
    return {"hits": hits}
