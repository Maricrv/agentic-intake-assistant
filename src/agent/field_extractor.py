from __future__ import annotations

import re
from typing import Dict

_BUDGET_NUM = re.compile(r"\$\s*(\d{2,6})")


def extract_prefill(text: str) -> Dict[str, str]:
    """Best-effort extraction of common pre-quote fields from a free-text message.

    IMPORTANT:
    - This is intentionally conservative (rules/regex only).
    - It should never block the flow; it only *prefills* if it's confident.
    """
    t = (text or "").strip()
    tl = t.lower()
    out: Dict[str, str] = {}

    # ---- Timeline (light, conservative) ----
    if "within_24h" in tl or "within 24" in tl or "today" in tl or "tomorrow" in tl:
        out["timeline"] = "within_24h"
    elif "within_2_weeks" in tl or "within 2 weeks" in tl or "two weeks" in tl or "14 days" in tl:
        out["timeline"] = "within_2_weeks"
    elif "within_1_week" in tl or "within 1 week" in tl or "next week" in tl or "this week" in tl or "7 days" in tl:
        out["timeline"] = "within_1_week"

    # ---- Budget (extract first $number and bucket it) ----
    m = _BUDGET_NUM.search(t)
    if m:
        try:
            n = int(m.group(1))
            if n < 50:
                out["budget_range"] = "<50"
            elif 50 <= n <= 100:
                out["budget_range"] = "50-100"
            elif 100 < n <= 300:
                out["budget_range"] = "100-300"
            elif 300 < n <= 500:
                out["budget_range"] = "300-500"
            else:
                out["budget_range"] = "500-1000"
        except Exception:
            pass

    # ---- Location (very conservative: only obvious city mentions you choose to support) ----
    # You can expand this list safely later.
    if "toronto" in tl:
        out["location"] = "Toronto"

    # ---- Urgency ----
    if any(w in tl for w in ["urgent", "asap", "emergency", "immediately", "right now"]):
        out["urgency"] = "urgent"
    elif any(w in tl for w in ["flexible", "no rush", "not urgent", "whenever"]):
        out["urgency"] = "flexible"

    return out
