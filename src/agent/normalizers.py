from typing import Any, Dict

def _norm_text(s: str) -> str:
    return (s or "").strip()

def _norm_lc(s: str) -> str:
    return _norm_text(s).lower()

def normalize_value(kind: str, raw: str, config: Dict[str, Any]) -> str:
    """
    kind: "urgency" | "timeline" | "budget" | "service_type" | "text"
    """
    raw_clean = _norm_text(raw)
    raw_lc = _norm_lc(raw)

    if kind in ("text", "service_type"):
        return raw_clean

    norms = (config or {}).get("normalizers", {})
    table = norms.get(kind, {})

    # match by synonyms list
    for canonical, synonyms in table.items():
        for s in synonyms:
            if raw_lc == str(s).lower().strip():
                return canonical

    # fallback: if user already typed canonical
    if raw_clean in table:
        return raw_clean

    # special: numeric budget -> bucket
    if kind == "budget":
        if raw_lc.isdigit():
            n = int(raw_lc)
            if n < 50:
                return "<50"
            if 50 <= n <= 100:
                return "50-100"
            if 100 < n <= 300:
                return "100-300"
            return "not_provided"

    return "not_provided"

def normalize_constraints(raw: str, config: Dict[str, Any]) -> str:
    raw_clean = _norm_text(raw)
    raw_lc = _norm_lc(raw)
    ignore = set((config or {}).get("normalizers", {}).get("constraints_ignore", []))
    ignore = {str(x).lower().strip() for x in ignore}
    if not raw_clean:
        return ""
    if raw_lc in ignore:
        return ""  # ignore "yes/no/ok/none"
    return raw_clean
