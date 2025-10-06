import json, argparse, random, re
from pathlib import Path
from difflib import SequenceMatcher

# If you still want to keep common.read_md/normalize_ws, import them.
from common import read_md, normalize_ws  # noqa: F401

random.seed(42)

def make_system_prompt(clinic: str, currency: str) -> str:
    return (
        f"You are Atharva Dental Clinic assistant in {clinic}, India. "
        f"Respond in concise steps, use {currency} (₹) for any prices, be safety-minded, "
        f"ask for missing info when necessary, and ALWAYS include a final 'Source:' line "
        f"citing file#section for facts derived from context."
    )

def fmt_inr(x: int) -> str:
    return f"₹{x:,}"

def join_steps(items):
    """
    Return a plain, semicolon-separated list (no numbering).
    We'll format to numbered steps later in normalize_list_answer().
    """
    return "; ".join(s.strip() for s in items if s and str(s).strip())

_BULLET_PREFIX = re.compile(r'^\s*(?:[-*•]+|\d+[.)])\s*', re.IGNORECASE)

def _strip_bullet(s: str) -> str:
    return _BULLET_PREFIX.sub("", s).strip()

def _capitalize_first(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s

def normalize_list_answer(text: str) -> str:
    """
    If the answer is a list (separated by ';' or newlines), convert to numbered 1) ... lines.
    If it's already numbered/bulleted, strip existing bullets/numbers and renumber cleanly.
    Single-line answers are returned as-is (with trimmed whitespace).
    """
    if text is None:
        return ""
    raw = text.strip()

    # If it already looks like multiple lines or contains semicolons, treat as a list
    is_listy = (";" in raw) or ("\n" in raw)

    # Split on semicolons OR newlines, keep non-empty
    parts = [p for p in re.split(r"[;\n]+", raw) if p.strip()]
    # If splitting produced only one item and it doesn't start with bullets/numbers, return trimmed
    if len(parts) <= 1 and not _BULLET_PREFIX.match(raw):
        return raw

    # If it's single line but has bullets/numbers, treat as one part
    if len(parts) <= 1:
        parts = [raw]

    # Clean bullets/numbers and whitespace; capitalize first letter of each step
    cleaned = [_capitalize_first(_strip_bullet(p)) for p in parts if _strip_bullet(p)]

    # If after cleaning we only have one part, just return it
    if len(cleaned) <= 1 and not is_listy:
        return cleaned[0] if cleaned else raw

    # Number them
    return "\n".join(f"{i+1}) {p}" for i, p in enumerate(cleaned))

def add_paraphrases(q: str) -> list[str]:
    out = [q]
    ql = q.lower()

    m = re.match(r"how long does (.+?) take and how many visits\??", q, flags=re.I)
    if m:
        proc = m.group(1)
        out.append(f"{proc} — duration and number of visits?")
        out.append(f"What's the time per visit and total visits for {proc}?")

    if "what is the cost for " in ql:
        proc = q[q.lower().find("what is the cost for ")+len("what is the cost for "):].rstrip("?")
        out.append(f"Approximate price range for {proc}?")
        out.append(f"How much does {proc} typically cost?")

    if re.search(r"\baftercare\b", ql):
        out.append(q.replace("What are aftercare steps", "Post-treatment care steps"))

    uniq = []
    seen = set()
    for cand in out:
        if cand not in seen:
            seen.add(cand)
            uniq.append(cand)
    return uniq

def near_duplicate(a: str, b: str, threshold: float = 0.90) -> bool:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio() >= threshold

def emit_sample(system_prompt, q, a, source, ask_clarify: str | None = None):
    # Normalize to consistent numbered-steps style when list-like
    norm_a = normalize_list_answer(a)
    if ask_clarify:
        # Ensure the clarifying question starts on a new line and is properly capitalized
        clar = _capitalize_first(ask_clarify.strip())
        norm_a = f"{norm_a}\n{clar}"
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": q},
            {"role": "assistant", "content": f"{norm_a}\nSource: {source}"}
        ]
    }

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clinic", default="Pune")
    ap.add_argument("--currency", default="INR")
    ap.add_argument("--treatments", required=True)
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--faq", required=True)
    ap.add_argument("--recent", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max_per_treatment", type=int, default=7,
                    help="Upper bound of Q/A variants per treatment")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    train, val, evalq = [], [], []
    sys_p = make_system_prompt(args.clinic, args.currency)

    # ----------------------------
    # Treatments → richer Q/A set
    # ----------------------------
    treatments = json.loads(Path(args.treatments).read_text(encoding="utf-8"))
    for t in treatments:
        code = t.get("code", "TX-UNK")
        name = t["name"]
        dur = t.get("duration_minutes")
        visits = t.get("visits")
        low, high = t.get("price_band_inr", [None, None])
        aftercare = t.get("aftercare", [])
        risks = t.get("risks", [])
        indications = t.get("indications", [])
        contraind = t.get("contraindications", [])

        src = f"treatments.json#{code}"

        samples = []

        # 1) Duration + visits + price (+ 1–2 paraphrases)
        if dur and visits and low is not None and high is not None:
            q = f"How long does {name} take and how many visits?"
            a = join_steps([
                f"Typically {dur} minutes",
                f"About {visits} visit(s)",
                f"Price band: {fmt_inr(low)}–{fmt_inr(high)}"
            ])
            for pq in add_paraphrases(q)[:2]:
                samples.append(emit_sample(sys_p, pq, a, src))

        # 2) Aftercare (+ 1 paraphrase)
        if aftercare:
            q = f"What are aftercare steps for {name}?"
            a = join_steps(aftercare)
            for pq in add_paraphrases(q)[:2]:
                samples.append(emit_sample(sys_p, pq, a, src))

        # 3) Risks
        q = f"Any risks with {name}?"
        a = join_steps(risks) if risks else "Minimal risks when indicated by the clinician."
        samples.append(emit_sample(sys_p, q, a, src))

        # 4) Cost-only (+ 1 paraphrase)
        if low is not None and high is not None:
            q = f"What is the cost for {name}?"
            a = f"{fmt_inr(low)}–{fmt_inr(high)} depending on case complexity and materials."
            for pq in add_paraphrases(q)[:2]:
                samples.append(emit_sample(sys_p, pq, a, src))

        # 5) Pain/comfort expectation
        q = f"Is {name.lower()} painful?"
        if "Scaling" in name:
            a = "You may feel mild discomfort; anesthesia can be used for sensitive cases."
        elif "Root Canal" in name:
            a = "Local anesthesia is used; you may feel post-op soreness for 24–48h."
        else:
            a = "Local anesthesia minimizes pain; some soreness after the procedure is common."
        samples.append(emit_sample(sys_p, q, a, src))

        # 6) Eligibility/contraindications (if present)
        if contraind:
            q = f"Who should avoid or be cautious about {name}?"
            a = join_steps(contraind) + "; Consult your dentist to evaluate individual risks."
            samples.append(emit_sample(sys_p, q, a, src))

        # 7) When indicated
        if indications:
            q = f"When is {name} recommended?"
            a = join_steps([f"Indicated for: {', '.join(indications)}"])
            samples.append(emit_sample(sys_p, q, a, src))

        # 8) Clarifying-quote variant
        if low is not None and high is not None:
            q = f"Can I get a quick quote for {name}?"
            a = f"Typical range is {fmt_inr(low)}–{fmt_inr(high)}; exact estimate varies by tooth and complexity."
            ask = "Which tooth/area and any prior treatment? I can give a closer estimate."
            samples.append(emit_sample(sys_p, q, a, src, ask_clarify=ask))

        if args.max_per_treatment > 0:
            samples = samples[:args.max_per_treatment]

        train.extend(samples)

    # ----------------------------
    # Policies → Q/A
    # ----------------------------
    for p in args.policies:
        path = Path(p)
        if not path.exists():
            continue
        head = path.stem

        if "appointments" in head:
            train.append(emit_sample(
                sys_p,
                "What are clinic hours?",
                "Mon–Sat 9:30–18:30 IST; limited emergency slots on call.",
                f"{head}.md#hours"
            ))
            train.append(emit_sample(
                sys_p,
                "Do you accept walk-ins?",
                "Appointments preferred; limited same-day slots may be available. Call/WhatsApp to check wait time.",
                f"{head}.md#walkins"
            ))

        if "cancellations" in head:
            train.append(emit_sample(
                sys_p,
                "Cancellation policy?",
                "24h notice requested; same-day cancellations may incur a chair-time fee of ₹300.",
                f"{head}.md#policy"
            ))
            train.append(emit_sample(
                sys_p,
                "How to reschedule appointment?",
                "Call/WhatsApp at least 24h prior to reschedule; late changes may incur ₹300 fee.",
                f"{head}.md#reschedule"
            ))

        if "emergency" in head:
            train.append(emit_sample(
                sys_p,
                "What are dental red flags?",
                "Uncontrolled bleeding, facial swelling, high fever, trauma—seek urgent care/call immediately.",
                f"{head}.md#red-flags"
            ))
            train.append(emit_sample(
                sys_p,
                "Wisdom tooth pain with swelling—what to do?",
                join_steps([
                    "Avoid self-medicating antibiotics",
                    "Warm saline rinses",
                    "Seek urgent evaluation if fever, trismus, or spreading swelling"
                ]),
                f"{head}.md#wisdom-swelling"
            ))

        if "billing" in head:
            train.append(emit_sample(
                sys_p,
                "Payment methods accepted?",
                "UPI, cards, netbanking; Insurance reimbursement support available.",
                f"{head}.md#methods"
            ))

        if "sterilization" in head:
            train.append(emit_sample(
                sys_p,
                "How do you sterilize instruments?",
                "Class-B autoclave cycles, pouched instruments, surface disinfection between patients.",
                f"{head}.md#protocols"
            ))

    # ----------------------------
    # FAQs → a few seed items
    # ----------------------------
    if Path(args.faq).exists():
        _ = read_md(Path(args.faq))
        train.append(emit_sample(
            sys_p, "Is scaling painful?",
            "Mild discomfort; Local anesthesia for sensitive cases.",
            "faq.md#scaling-pain"
        ))
        train.append(emit_sample(
            sys_p, "Do whitening results last?",
            "Results typically last 6–12 months; Depends on diet and habits.",
            "faq.md#whitening-duration"
        ))
        train.append(emit_sample(
            sys_p, "Do you work on Sundays?",
            "Only emergency slots on call.",
            "faq.md#sunday-hours"
        ))

    # ----------------------------
    # Deduplicate near-identical questions
    # ----------------------------
    deduped = []
    seen_qs = []
    for ex in train:
        q = ex["messages"][1]["content"]
        if any(near_duplicate(q, s) for s in seen_qs):
            continue
        seen_qs.append(q)
        deduped.append(ex)
    train = deduped

    # ----------------------------
    # Shuffle + split (80/20)
    # ----------------------------
    random.shuffle(train)
    split = int(0.8 * len(train)) if len(train) else 0

    (Path(out) / "train.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in train[:split]),
        encoding="utf-8"
    )
    (Path(out) / "val.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in train[split:]),
        encoding="utf-8"
    )
    (Path(out) / "eval.jsonl").write_text(
        "\n".join(json.dumps(x, ensure_ascii=False) for x in evalq),
        encoding="utf-8"
    )

    print(f"Wrote {len(train[:split])} train, {len(train[split:])} val, {len(evalq)} eval to {out}")

if __name__ == "__main__":
    main()
