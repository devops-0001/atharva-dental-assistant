SYSTEM_PROMPT = (
  "You are Atharva Dental Clinic assistant based in Pune, India. "
  "Respond in concise steps, use INR as currency for any prices or costs, be safety-minded, first try to find the answer from the context provided here."
  "ask for missing info when necessary, and ALWAYS include a final 'Source:' line "
  "citing file#section for facts derived from context.\n"
  "If the question indicates emergency red flags (uncontrolled bleeding, facial swelling, high fever, trauma), "
  "urge immediate contact with the clinic's emergency number.\n"
)

def _label(meta: dict) -> str:
    did = (meta or {}).get("doc_id")
    sec = (meta or {}).get("section")
    if not did:
        return "unknown"
    return f"{did}#{sec}" if sec and sec != "full" else did

def _render_context_block(retrieved_hits: list[dict]) -> str:
    """
    Render only label + text, no Python dicts.
    """
    blocks: list[str] = []
    for h in retrieved_hits:
        meta = h.get("meta") or {}
        label = _label(meta)
        text = (h.get("text") or meta.get("text") or "").strip()
        if not text:
            continue
        blocks.append(f"### {label}\n{text}")
    return "\n\n".join(blocks).strip()

def build_messages(user_q: str, retrieved_hits: list[dict]) -> list[dict]:
    context_block = _render_context_block(retrieved_hits)
    system = SYSTEM_PROMPT + "\nContext snippets:\n" + (context_block if context_block else "(none)")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_q.strip()},
    ]
