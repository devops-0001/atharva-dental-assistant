import random, re
from pathlib import Path

def read_md(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def sys_prompt() -> str:
    return ("You are Atharva Dental Clinic assistant in Pune (INR). "
            "Be concise, safety-minded, include 'Source:' with file#section. "
            "If info is missing, ask follow-up questions.")
