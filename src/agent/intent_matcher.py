from typing import Any, Dict, List, Optional

def pick_intent(first_text: str, config: Dict[str, Any]) -> Dict[str, Any]:
    t = (first_text or "").lower().strip()
    intents: List[Dict[str, Any]] = config.get("intents", [])

    candidates: List[Dict[str, Any]] = []
    for it in intents:
        match = it.get("match", {}) or {}
        priority = int(it.get("priority", 0))

        if match.get("always") is True:
            candidates.append((priority, it))
            continue

        kws = [str(x).lower() for x in match.get("keywords_any", [])]
        if kws and any(k in t for k in kws):
            candidates.append((priority, it))
            continue

        starts = [str(x).lower() for x in match.get("starts_with_any", [])]
        if starts and any(t.startswith(s) for s in starts):
            candidates.append((priority, it))
            continue

    if not candidates:
        # fallback: first intent or a safe default
        return intents[0] if intents else {"id": "fallback_unknown", "flow": []}

    # highest priority wins
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1]
