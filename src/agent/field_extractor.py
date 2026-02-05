from __future__ import annotations

import re
from typing import Dict

# Patterns
_DOLLAR_NUM = re.compile(r"\$\s*(\d{2,6})")
_BUDGET_AFTER = re.compile(r"\bbudget\b\D{0,20}(\d{2,6})")
_BUDGET_BEFORE = re.compile(r"\b(\d{2,6})\b\D{0,20}\bbudget\b")
_HAVE_BUDGET = re.compile(r"\bi have\s+(\d{2,6})\s+budget\b")


def _bucket_budget(n: int) -> str:
    if n < 50:
        return "<50"
    if 50 <= n <= 100:
        return "50-100"
    if 100 < n <= 300:
        return "100-300"
    if 300 < n <= 500:
        return "300-500"
    return "500-1000"


def extract_prefill(text: str) -> Dict[str, str]:
    """Best-effort extraction of common pre-quote fields from a free-text message.

    IMPORTANT:
    - Conservative: regex + rules only
    - Prefill only (never blocks the flow)
    """
    t = (text or "").strip()
    tl = t.lower()
    out: Dict[str, str] = {}

    # ---- Timeline ----
    if any(w in tl for w in ["today", "tomorrow", "within 24", "within_24h"]):
        out["timeline"] = "within_24h"
    elif any(w in tl for w in ["next week", "this week", "within 1 week", "7 days"]):
        out["timeline"] = "within_1_week"
    elif any(w in tl for w in ["two weeks", "within 2 weeks", "14 days"]):
        out["timeline"] = "within_2_weeks"

    # ---- Budget (multiple safe patterns) ----
    m = (
        _DOLLAR_NUM.search(t)
        or _HAVE_BUDGET.search(tl)
        or _BUDGET_AFTER.search(tl)
        or _BUDGET_BEFORE.search(tl)
    )
    if m:
        try:
            n = int(m.group(1))
            out["budget_range"] = _bucket_budget(n)
        except Exception:
            pass

    # ---- Location (still conservative) ----
    if "toronto" in tl:
        out["location"] = "Toronto"

    # ---- Urgency ----
    if any(w in tl for w in ["urgent", "asap", "emergency", "immediately", "right now"]):
        out["urgency"] = "urgent"
    elif any(w in tl for w in ["flexible", "no rush", "not urgent", "whenever"]):
        out["urgency"] = "flexible"

    return out
