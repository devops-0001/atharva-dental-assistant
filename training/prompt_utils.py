# training/prompt_utils.py
from typing import List, Dict

DEFAULT_SYSTEM = (
    "You are Atharva Dental Clinic assistant in Pune, India (INR). "
    "Always use INR (₹) as a currency for prices and cost ranges, "
    "Be concise, safety-minded, ask follow-ups if info is missing, "
    "Consider the context provided and derive answered based on that, "
    "and always include a final 'Source:' line citing file#section."
)

def to_chat(messages: List[Dict], default_system: str = DEFAULT_SYSTEM):
    """
    Ensure we have system→user→assistant message order for a single-turn sample.
    Our dataset already stores messages with roles. We enforce one assistant turn.
    """
    sys_seen = any(m["role"] == "system" for m in messages)
    msgs = []
    if not sys_seen:
        msgs.append({"role": "system", "content": default_system})
    msgs.extend(messages)
    # Basic guard: keep only first assistant answer for label masking
    out, assistant_added = [], False
    for m in msgs:
        if m["role"] == "assistant":
            if assistant_added:  # drop extra assistant turns for simplicity
                continue
            assistant_added = True
        out.append(m)
    return out

def simple_template(messages: List[Dict]) -> str:
    """
    Fallback formatting if tokenizer has no chat template.
    """
    lines = []
    for m in messages:
        role = m["role"]
        prefix = {"system":"[SYS]", "user":"[USER]", "assistant":"[ASSISTANT]"}.get(role, f"[{role.upper()}]")
        lines.append(f"{prefix}\n{m['content'].strip()}\n")
    # Ensure the string ends with assistant text (trainer expects labels on last turn)
    return "\n".join(lines).strip()
