from __future__ import annotations

import re
from typing import Any, Dict, Optional


def norm_text(s: str) -> str:
    return (s or "").strip()


def norm_lc(s: str) -> str:
    return norm_text(s).lower()


def extract_first_int(text: str) -> Optional[int]:
    m = re.search(r"(\d+)", text or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def is_valid_service_type(text: str) -> bool:
    t = norm_lc(text)
    if not t:
        return False
    question_markers = ["what", "how", "price", "pricing", "cost", "charge", "rates", "hours", "address", "?"]
    if any(m in t for m in question_markers):
        return False
    if len(t) < 3:
        return False
    if t.isdigit():
        return False
    if t in {"yes", "no", "ok", "okay", "urgent", "flexible"}:
        return False
    return True


def normalize_value(kind: str, raw: str, config: Dict[str, Any]) -> str:
    """
    kind: "urgency" | "timeline" | "budget" | "service_type" | "text"
    """
    raw_clean = norm_text(raw)
    raw_lc = norm_lc(raw)

    if kind in ("text", "service_type"):
        return raw_clean if raw_clean else "not_provided"

    norms = (config or {}).get("normalizers", {})
    table = norms.get(kind, {}) or {}

    # Match by synonyms (exact)
    for canonical, synonyms in table.items():
        for s in (synonyms or []):
            if raw_lc == str(s).lower().strip():
                return canonical

    # Allow typing canonical
    if raw_clean in table:
        return raw_clean

    # Budget buckets
    if kind == "budget":
        n = extract_first_int(raw_lc)
        if n is None:
            return "not_provided"
        if n < 50:
            return "<50"
        if 50 <= n <= 100:
            return "50-100"
        if 100 < n <= 300:
            return "100-300"
        if 300 < n <= 500:
            return "300-500"
        return "500-1000"

    # Timeline parsing (supports "6 days", "in 2 weeks", "24h", "tomorrow")
    if kind == "timeline":
        t = raw_lc

        if "today" in t or "tomorrow" in t or "within_24h" in t or "within 24" in t or "24h" in t:
            return "within_24h"

        m = re.search(r"(\d+)\s*(day|days|d)\b", t)
        if m:
            try:
                days = int(m.group(1))
                if days <= 1:
                    return "within_24h"
                if days <= 7:
                    return "within_1_week"
                if days <= 14:
                    return "within_2_weeks"
            except Exception:
                pass

        if "week" in t:
            if "2" in t or "two" in t:
                return "within_2_weeks"
            return "within_1_week"

        if "within_1_week" in t:
            return "within_1_week"
        if "within_2_weeks" in t:
            return "within_2_weeks"

        return "not_provided"

    return "not_provided"


def normalize_constraints(raw: str, config: Dict[str, Any]) -> str:
    raw_clean = norm_text(raw)
    if not raw_clean:
        return ""

    raw_lc = norm_lc(raw_clean)

    # ignore "no..."
    if raw_lc.startswith("no"):
        return ""

    ignore = (config or {}).get("normalizers", {}).get("constraints_ignore", []) or []
    ignore_set = {str(x).lower().strip() for x in ignore}
    if raw_lc in ignore_set:
        return ""

    return raw_clean
