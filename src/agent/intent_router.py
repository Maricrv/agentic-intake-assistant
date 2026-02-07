from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .normalizers import norm_lc


class IntentRouter:
    """
    Encapsulates intent selection + question/normalize helpers.
    Keeps agent.py smaller and config-driven.
    """

    def __init__(
        self,
        intent_config: Dict[str, Any],
        defaults: Dict[str, Any],
        log: Callable[[str], None],
    ):
        self.intent_config = intent_config or {}
        self.defaults = defaults or {}
        self._log = log

    def intents(self) -> List[Dict[str, Any]]:
        return self.intent_config.get("intents", []) or []

    def pick_intent(self, first_text: str) -> Dict[str, Any]:
        t = norm_lc(first_text)
        intents = self.intents()

        candidates: List[tuple[int, int, Dict[str, Any]]] = []
        always_intents: List[tuple[int, Dict[str, Any]]] = []

        for it in intents:
            match = it.get("match", {}) or {}
            priority = int(it.get("priority", 0))

            if match.get("always") is True:
                always_intents.append((priority, it))
                continue

            score = 0
            kws = [str(x).lower() for x in match.get("keywords_any", [])]
            for kw in kws:
                if kw and kw in t:
                    score += 1

            starts = [str(x).lower() for x in match.get("starts_with_any", [])]
            for s in starts:
                if s and t.startswith(s):
                    score += 2

            if score > 0:
                candidates.append((score, priority, it))

        if candidates:
            candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
            chosen = candidates[0][2]
            self._log(f"intent_selected: {chosen.get('id')} (rule_match)")
            return chosen

        for it in intents:
            if it.get("id") == "fallback_unknown":
                self._log("intent_selected: fallback_unknown (no_match)")
                return it

        if always_intents:
            always_intents.sort(key=lambda x: x[0], reverse=True)
            chosen = always_intents[0][1]
            self._log(f"intent_selected: {chosen.get('id')} (always)")
            return chosen

        self._log("intent_selected: fallback_unknown (no_intents)")
        return {"id": "fallback_unknown", "flow": []}

    def find_intent_by_id(self, intent_id: Optional[str]) -> Optional[Dict[str, Any]]:
        if not intent_id:
            return None
        for it in self.intents():
            if it.get("id") == intent_id:
                return it
        return None

    def fallback_intent(self) -> Dict[str, Any]:
        for it in self.intents():
            if it.get("id") == "fallback_unknown":
                return it
        if self.intents():
            return self.intents()[0]
        return {"id": "fallback_unknown", "flow": []}

    def opening_message(self, intent: Dict[str, Any]) -> str:
        msg = intent.get("opening_message") or self.defaults.get("opening_message")
        if not msg:
            msg = "I can help you create a service request. I’ll ask a few quick questions.\n"
        return str(msg)

    def question_for_field(self, intent: Dict[str, Any], field: str) -> str:
        for step in (intent.get("flow", []) or []):
            if step.get("field") == field and step.get("question"):
                return str(step["question"])

        # Neutral fallbacks
        if field == "issue_description":
            return "Tell me briefly what you need help with."
        if field == "service_type":
            return "What type of service is this?"
        if field == "location":
            return "What is your location (city/country)?"
        if field == "budget_range":
            return "Do you have an estimated budget? (example: <50, 50-100, 100-300, 300-500, 500-1000, not sure)"
        if field == "timeline":
            return "When do you want this addressed? (within_24h / within_1_week / within_2_weeks)"
        if field == "urgency":
            return "Is this urgent or flexible? (urgent/flexible)"
        return f"Please provide: {field}"

    def normalizer_for_field(self, intent: Dict[str, Any], field: str) -> str:
        for step in (intent.get("flow", []) or []):
            if step.get("field") == field:
                return step.get("normalize", "text")
        return "text"

    def required_fields_from_intent(self, intent: Dict[str, Any]) -> List[str]:
        rf = intent.get("required_fields")
        if isinstance(rf, list) and rf:
            return [str(x) for x in rf]

        return [
            str(step["field"])
            for step in (intent.get("flow", []) or [])
            if step.get("required") and step.get("field")
        ]

    def allowed_for_field(self, intent: Dict[str, Any], field: str) -> List[str]:
        fields = intent.get("fields", {}) or {}
        spec = fields.get(field, {}) or {}
        allowed = spec.get("allowed") or []
        return [str(x) for x in allowed]
