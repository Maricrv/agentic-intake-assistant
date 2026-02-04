from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .schema import IntakeResult
from .llm_client import LLMClient
from .field_corrector import FieldCorrector


@dataclass
class AgentState:
    name: str


S0 = AgentState("S0")
S1 = AgentState("S1")
S2 = AgentState("S2")
S3 = AgentState("S3")
S4 = AgentState("S4")
S5 = AgentState("S5")


class GenericIntakeAgent:
    MAX_FOLLOWUP_ROUNDS = 2
    MAX_EMPTY_TRIES_PER_FIELD_IN_RUN = 2

    def __init__(
        self,
        request_id: str = "req_local_000001",
        session_id: str = "sess_local",
        previous_state: dict | None = None,
        intent_config: Dict[str, Any] | None = None,
    ):
        self.result = IntakeResult(request_id=request_id)
        self.result.session.session_id = session_id

        self.intent_config: Dict[str, Any] = intent_config or {}

        llm_cfg = (self.intent_config.get("llm") or {})
        llm_enabled = bool(llm_cfg.get("enabled", False))
        llm_model = str(llm_cfg.get("model", "gpt-5"))
        llm_effort = str(llm_cfg.get("reasoning_effort", "low"))

        self.llm = LLMClient(enabled=llm_enabled, model=llm_model, reasoning_effort=llm_effort)
        self.corrector = FieldCorrector(self.llm)

        self.state = S0
        self.turns = 0
        self.memory = {
            "missing_fields": [],
            "collected": {},
            "attempts": {},
            "last_state": "S0",
            "last_intent_id": None,
        }

        if previous_state:
            self.memory.update(previous_state)

        self.memory.setdefault("missing_fields", [])
        self.memory.setdefault("collected", {})
        self.memory.setdefault("attempts", {})
        self.memory.setdefault("last_state", "S0")
        self.memory.setdefault("last_intent_id", None)

    def export_state(self) -> dict:
        self.memory["last_state"] = self.state.name
        return self.memory

    def _ask(self, question: str) -> str:
        self.turns += 1
        return input(question + "\n> ").strip()

    def _set_state(self, st: AgentState):
        self.state = st
        self.result.session.state = st.name

    def _norm_text(self, s: str) -> str:
        return (s or "").strip()

    def _norm_lc(self, s: str) -> str:
        return self._norm_text(s).lower()

    def _is_valid_service_type(self, text: str) -> bool:
        t = (text or "").lower().strip()
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

    def _pick_intent(self, first_text: str) -> Dict[str, Any]:
        t = self._norm_lc(first_text)
        intents: List[Dict[str, Any]] = self.intent_config.get("intents", []) or []

        candidates: List[tuple[int, Dict[str, Any]]] = []
        always_intents: List[tuple[int, Dict[str, Any]]] = []

        for it in intents:
            match = it.get("match", {}) or {}
            priority = int(it.get("priority", 0))

            if match.get("always") is True:
                always_intents.append((priority, it))
                continue

            kws = [str(x).lower() for x in match.get("keywords_any", [])]
            if kws and any(k in t for k in kws):
                candidates.append((priority, it))
                continue

            starts = [str(x).lower() for x in match.get("starts_with_any", [])]
            if starts and any(t.startswith(s) for s in starts):
                candidates.append((priority, it))
                continue

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        for it in intents:
            if it.get("id") == "service_request":
                return it

        if always_intents:
            always_intents.sort(key=lambda x: x[0], reverse=True)
            return always_intents[0][1]

        return {"id": "fallback_unknown", "flow": []}

    def _extract_first_int(self, text: str) -> Optional[int]:
        m = re.search(r"(\d+)", text or "")
        if not m:
            return None
        try:
            return int(m.group(1))
        except Exception:
            return None

    def _normalize_value(self, kind: str, raw: str) -> str:
        raw_clean = self._norm_text(raw)
        raw_lc = self._norm_lc(raw)

        if kind in ("text", "service_type"):
            return raw_clean if raw_clean else "not_provided"

        norms = (self.intent_config or {}).get("normalizers", {})
        table = norms.get(kind, {}) or {}

        # Match by synonyms (exact)
        for canonical, synonyms in table.items():
            for s in synonyms or []:
                if raw_lc == str(s).lower().strip():
                    return canonical

        # Allow typing canonical
        if raw_clean in table:
            return raw_clean

        # --- Budget: allow "yes, 20" / "$20" / "around 20" ---
        if kind == "budget":
            n = self._extract_first_int(raw_lc)
            if n is not None:
                if n < 50:
                    return "<50"
                if 50 <= n <= 100:
                    return "50-100"
                if 100 < n <= 300:
                    return "100-300"
            return "not_provided"

        # Timeline parsing
        if kind == "timeline":
            t = raw_lc
            if "today" in t or "tomorrow" in t or "within_24h" in t:
                return "within_24h"
            if ("hour" in t or "hrs" in t or "hr" in t) and any(ch.isdigit() for ch in t):
                return "within_24h"
            if "day" in t and any(ch.isdigit() for ch in t):
                return "within_1_week"
            if "week" in t:
                if "2" in t or "two" in t:
                    return "within_2_weeks"
                return "within_1_week"

        return "not_provided"

    def _normalize_constraints(self, raw: str) -> str:
        raw_clean = self._norm_text(raw)
        if not raw_clean:
            return ""

        raw_lc = self._norm_lc(raw_clean)

        # NEW: ignore "no ..." answers (no, no now, no constraints, etc.)
        if raw_lc.startswith("no"):
            return ""

        ignore = (self.intent_config or {}).get("normalizers", {}).get("constraints_ignore", []) or []
        ignore_set = {str(x).lower().strip() for x in ignore}
        if raw_lc in ignore_set:
            return ""

        return raw_clean

    def _question_for_field(self, intent: Dict[str, Any], field: str) -> str:
        for step in (intent.get("flow", []) or []):
            if step.get("field") == field and step.get("question"):
                return str(step["question"])
        if field == "service_type":
            return "What type of service are you looking for? (repair, installation, consultation, etc.)"
        if field == "location":
            return "What is your location (city/country)?"
        if field == "budget_range":
            return "Do you have a budget range? (example: <50, 50-100, 100-300, not sure)"
        if field == "timeline":
            return "When do you want this addressed? (within_24h / within_1_week / within_2_weeks)"
        if field == "urgency":
            return "Is this urgent or flexible? (urgent/flexible)"
        if field == "issue_description":
            return "Tell me briefly what you need help with today."
        return f"Please provide: {field}"

    def _required_fields_from_intent(self, intent: Dict[str, Any]) -> List[str]:
        return [
            str(step["field"])
            for step in (intent.get("flow", []) or [])
            if step.get("required") and step.get("field")
        ]

    def _normalizer_for_field(self, intent: Dict[str, Any], field: str) -> str:
        for step in (intent.get("flow", []) or []):
            if step.get("field") == field:
                return step.get("normalize", "text")
        return "text"

    def _apply_field(self, intent: Dict[str, Any], field: str, raw: str) -> None:
        normalizer = self._normalizer_for_field(intent, field)

        if field == "constraints":
            val = self._normalize_constraints(raw)
            if val:
                self.result.request.details.constraints.append(val)
            return

        if field == "issue_description":
            val = self._normalize_value(normalizer, raw)
            self.memory["collected"]["issue_description"] = val
            self.result.request.details.issue_description = val
            return

        if field == "service_type":
            val = self._normalize_value("service_type", raw)
            if val != "not_provided" and self._is_valid_service_type(val):
                self.memory["collected"]["service_type"] = val
                self.result.request.details.service_type = val
                self.result.request.summary = f"User is requesting: {val}"
            return

        if field == "urgency":
            self.result.request.details.urgency = self._normalize_value("urgency", raw)
            return

        if field == "timeline":
            self.result.request.details.timeline = self._normalize_value("timeline", raw)
            return

        if field == "location":
            corrected = self.corrector.maybe_correct_location_with_confirmation(raw)
            raw_to_use = corrected if corrected else raw
            val = self._normalize_value(normalizer, raw_to_use)
            self.result.request.details.location = val if val != "not_provided" else "not_provided"
            return

        if field == "budget_range":
            self.result.request.details.budget_range = self._normalize_value("budget", raw)
            return

        self.memory["collected"][field] = self._normalize_value(normalizer, raw)

    def _ask_and_apply_followups(self, intent: Dict[str, Any], missing_fields: List[str]) -> None:
        for field in missing_fields:
            self.memory.setdefault("attempts", {})
            self.memory["attempts"][field] = int(self.memory["attempts"].get(field, 0)) + 1

            raw = self._ask(self._question_for_field(intent, field))

            if not raw and self.memory["attempts"][field] >= self.MAX_EMPTY_TRIES_PER_FIELD_IN_RUN:
                print("(No problem — we can continue for now.)")
                continue

            self._apply_field(intent, field, raw)

    def _handoff_for_ready(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        defaults = self.intent_config.get("defaults", {}) or {}
        default_handoff = defaults.get("handoff", {}) or {
            "recommended_action": "route_human",
            "routing_hint": "human_review",
        }
        intent_handoff = intent.get("handoff", {}) or {}
        rec = (intent_handoff.get("recommended_action") or "").strip()
        if rec in {"route_human", "completed"}:
            return intent_handoff
        return default_handoff

    def run(self) -> IntakeResult:
        self._set_state(S0)
        print("I can help you prepare a service request. I’ll ask a few quick questions to understand your needs.\n")

        intents = self.intent_config.get("intents", []) or []
        pending = self.memory.get("missing_fields", []) or []
        last_intent_id = self.memory.get("last_intent_id")

        last_intent = None
        if last_intent_id and intents:
            for it in intents:
                if it.get("id") == last_intent_id:
                    last_intent = it
                    break
        if last_intent is None:
            for it in intents:
                if it.get("id") == "fallback_unknown":
                    last_intent = it
                    break
        if last_intent is None and intents:
            last_intent = intents[0]
        if last_intent is None:
            last_intent = {"id": "fallback_unknown", "flow": []}

        if pending:
            print("Continuing your previous request...\n")
            self._set_state(S1)

            self.result.request.intent_id = last_intent.get("id", "fallback_unknown")
            required_fields = self._required_fields_from_intent(last_intent)

            for field in pending:
                raw = self._ask(self._question_for_field(last_intent, field))
                self._apply_field(last_intent, field, raw)

            self._set_state(S4)
            missing_now = self._compute_missing_fields(required_fields=required_fields)

            rounds = 0
            while missing_now and rounds < self.MAX_FOLLOWUP_ROUNDS:
                rounds += 1
                print("\nI’m still missing a couple of details to complete your request.\n")
                self._ask_and_apply_followups(last_intent, missing_now)
                missing_now = self._compute_missing_fields(required_fields=required_fields)

            if missing_now:
                self._finalize(not_ready_fields=missing_now, next_qs=self._questions_for_missing(missing_now, last_intent))
            else:
                self.memory["missing_fields"] = []
                self.result.readiness.status = "ready"
                self.result.readiness.notes = "Request has sufficient information for human handling."
                h = self._handoff_for_ready(last_intent)
                self.result.handoff.recommended_action = h.get("recommended_action", "route_human")
                self.result.handoff.routing_hint = h.get("routing_hint", "human_review")
                self.result.handoff.next_questions = []

            self._set_state(S5)
            return self._done()

        self._set_state(S1)

        issue_q = "Tell me briefly what you need help with today."
        for it in intents:
            if it.get("id") == "service_request":
                issue_q = self._question_for_field(it, "issue_description")
                break

        first_text = self._ask(issue_q)
        self._apply_field({"flow": [{"field": "issue_description", "normalize": "text"}]}, "issue_description", first_text)

        intent = self._pick_intent(first_text)
        intent_id = intent.get("id", "fallback_unknown")
        self.memory["last_intent_id"] = intent_id
        self.result.request.intent_id = intent_id

        defaults = self.intent_config.get("defaults", {}) or {}
        self.result.request.request_type = "service_request"
        self.result.request.service_category = defaults.get("service_category", "generic_service")

        flow = intent.get("flow", []) or []
        required_fields: List[str] = []
        self._set_state(S2)

        for step in flow:
            field = step.get("field")
            question = step.get("question", "")
            required = bool(step.get("required", False))

            if required and field:
                required_fields.append(field)

            if field == "issue_description":
                continue
            if not field:
                continue

            raw = self._ask(str(question))
            self._apply_field(intent, field, raw)

        self._set_state(S4)
        missing = self._compute_missing_fields(required_fields=required_fields)

        rounds = 0
        while missing and rounds < self.MAX_FOLLOWUP_ROUNDS:
            rounds += 1
            print("\nI’m still missing a couple of details to complete your request.\n")
            self._ask_and_apply_followups(intent, missing)
            missing = self._compute_missing_fields(required_fields=required_fields)

        if missing:
            self._finalize(not_ready_fields=missing, next_qs=self._questions_for_missing(missing, intent))
        else:
            self.result.readiness.status = "ready"
            self.result.readiness.notes = "Request has sufficient information for human handling."
            h = self._handoff_for_ready(intent)
            self.result.handoff.recommended_action = h.get("recommended_action", "route_human")
            self.result.handoff.routing_hint = h.get("routing_hint", "human_review")
            self.result.handoff.next_questions = []

        self._set_state(S5)
        return self._done()

    def _compute_missing_fields(self, required_fields: Optional[List[str]] = None) -> List[str]:
        missing: List[str] = []
        d = self.result.request.details
        required_fields = required_fields or ["service_type", "location"]

        if "issue_description" in required_fields:
            issue = self.memory.get("collected", {}).get("issue_description", "")
            if not self._norm_text(issue) or issue == "not_provided":
                missing.append("issue_description")

        if "service_type" in required_fields:
            service_type = self.memory.get("collected", {}).get("service_type")
            if not self._is_valid_service_type(service_type):
                missing.append("service_type")

        if "location" in required_fields:
            if not d.location or d.location == "not_provided":
                missing.append("location")

        return missing

    def _questions_for_missing(self, missing: List[str], intent: Dict[str, Any] | None = None) -> List[str]:
        intent = intent or {"flow": []}
        return [self._question_for_field(intent, field) for field in missing]

    def _finalize(self, not_ready_fields: List[str], next_qs: List[str]):
        self.result.readiness.status = "not_ready"
        self.result.readiness.missing_fields = not_ready_fields
        self.result.readiness.notes = "More information is required to proceed."
        self.result.handoff.recommended_action = "ask_follow_up"
        self.result.handoff.routing_hint = "human_review"
        self.result.handoff.next_questions = next_qs
        self.memory["missing_fields"] = not_ready_fields

    def _done(self) -> IntakeResult:
        self.result.audit.conversation_turns = self.turns
        return self.result
