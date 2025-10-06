import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Tuple, Dict, Any, List

from sentence_transformers import SentenceTransformer
import faiss


# -------- Helpers to render concise, model-friendly chunk text --------

def _render_treatment_item(it: Dict[str, Any]) -> str:
    """
    Render a single treatment JSON object into a compact, informative snippet.
    """
    keys_order = (
        "code", "name", "category",
        "duration_minutes", "visits", "price_band_inr",
        "indications", "contraindications",
        "steps", "aftercare", "risks"
    )
    lines: List[str] = []
    for k in keys_order:
        if k not in it:
            continue
        v = it[k]
        if isinstance(v, (list, tuple)):
            v = ", ".join(map(str, v))
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


def _render_markdown_snippet(text: str, max_lines: int = 8) -> str:
    """
    Take the heading and first few meaningful lines from markdown.
    """
    lines = [ln.rstrip() for ln in text.splitlines()]
    # keep non-empty lines; prefer headings/bullets first
    cleaned: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        cleaned.append(s)
        if len(cleaned) >= max_lines:
            break
    return "\n".join(cleaned)


def _render_recent_qa(obj: Dict[str, Any]) -> str:
    q = str(obj.get("q", "")).strip()
    a = str(obj.get("a", "")).strip()
    return f"Q: {q}\nA: {a}"


# -------- Corpus iterator producing (text_for_embedding, meta_dict) --------

def iter_docs(root: Path) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """
    Yields (text, meta) pairs. meta includes:
      - doc_id: file path relative to dataset root (e.g., 'policies/emergency.md', 'treatments.json')
      - section: semantic section id (e.g., 'TX-SCALE-01') or 'full' for whole docs, or timestamp for jsonl
      - path: doc_id#section (or doc_id when section == 'full')
      - type: md | json | jsonl
      - text: concise snippet for grounding (NOT the full document)
    The 'text' is also used as the embedding input.
    """
    # policies/*.md
    for md in (root / "policies").glob("*.md"):
        doc_id = f"policies/{md.name}"
        full = md.read_text(encoding="utf-8", errors="ignore")
        snippet = _render_markdown_snippet(full, max_lines=8)
        meta = {
            "doc_id": doc_id,
            "section": "full",
            "path": doc_id,
            "type": "md",
            "text": snippet,
        }
        yield snippet, meta

    # faq.md
    faq_p = (root / "faq.md")
    if faq_p.exists():
        faq_txt = faq_p.read_text(encoding="utf-8", errors="ignore")
        snippet = _render_markdown_snippet(faq_txt, max_lines=10)
        meta = {
            "doc_id": "faq.md",
            "section": "full",
            "path": "faq.md",
            "type": "md",
            "text": snippet,
        }
        yield snippet, meta

    # treatments.json: one chunk per treatment, section = code (semantic id)
    tr_p = (root / "treatments.json")
    if tr_p.exists():
        treatments = json.loads(tr_p.read_text(encoding="utf-8"))
        if isinstance(treatments, list):
            for it in treatments:
                # prefer semantic section id 'code' (e.g., TX-SCALE-01)
                code = it.get("code") or "item"
                snippet = _render_treatment_item(it)
                meta = {
                    "doc_id": "treatments.json",
                    "section": str(code),
                    "path": f"treatments.json#{code}",
                    "type": "json",
                    "text": snippet,
                }
                yield snippet, meta

    # recent_queries.jsonl: optional, include as weak signals (can downweight later)
    rq_p = (root / "recent_queries.jsonl")
    if rq_p.exists():
        for line in rq_p.read_text(encoding="utf-8", errors="ignore").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            snippet = _render_recent_qa(obj)
            ts = str(obj.get("ts", "na"))
            meta = {
                "doc_id": "recent_queries.jsonl",
                "section": ts,
                "path": f"recent_queries.jsonl:{ts}",
                "type": "jsonl",
                "text": snippet,
            }
            yield snippet, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="datasets/clinic")
    ap.add_argument("--outdir", required=True, help="artifacts/rag")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Build corpus
    texts: List[str] = []
    metas: List[Dict[str, Any]] = []
    for txt, meta in iter_docs(root):
        # keep text modest to avoid huge meta and keep embeddings focused
        txt_capped = txt.strip()[:1500]
        texts.append(txt_capped)
        # store the same snippet in meta (so retriever can return it directly)
        meta = dict(meta)
        meta["text"] = txt_capped
        metas.append(meta)

    # Embed and index
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    embs = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(embs)

    # Persist
    faiss.write_index(index, str(out / "index.faiss"))
    (out / "meta.json").write_text(
        json.dumps(metas, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Indexed {len(texts)} chunks → {out}")


if __name__ == "__main__":
    main()
