from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .schema import IntakeResult


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
    """
    Config-driven agentic intake agent.

    - Intents + flow + normalization live in intent_config (JSON).
    - Engine is generic: runs the selected intent flow and produces IntakeResult.
    """

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

        self.state = S0
        self.turns = 0
        self.memory = {
            "missing_fields": [],
            "collected": {},
            "last_state": "S0",
            "last_intent_id": None,
        }

        if previous_state:
            self.memory.update(previous_state)

    def export_state(self) -> dict:
        self.memory["last_state"] = self.state.name
        return self.memory

    # ---------- helpers ----------
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

        # If it looks like a general question, it is NOT a service type
        question_markers = ["what", "how", "price", "pricing", "cost", "charge", "rates", "hours", "address", "?"]
        if any(m in t for m in question_markers):
            return False

        # Too short / vague
        if len(t) < 3:
            return False

        # Avoid obvious junk answers
        if t.isdigit():
            return False
        if t in {"yes", "no", "ok", "okay", "urgent", "flexible"}:
            return False

        return True

    # ---------- config-driven pieces ----------
    def _pick_intent(self, first_text: str) -> Dict[str, Any]:
        """Pick best intent by match + highest priority."""
        t = self._norm_lc(first_text)
        intents: List[Dict[str, Any]] = self.intent_config.get("intents", []) or []

        candidates: List[tuple[int, Dict[str, Any]]] = []

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
            # Safe fallback
            # Prefer explicit fallback_unknown if present
            for it in intents:
                if it.get("id") == "fallback_unknown":
                    return it
            return intents[0] if intents else {"id": "fallback_unknown", "flow": []}

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def _normalize_value(self, kind: str, raw: str) -> str:
        """
        kind: urgency | timeline | budget | service_type | text
        Returns canonical value or 'not_provided'.
        """
        raw_clean = self._norm_text(raw)
        raw_lc = self._norm_lc(raw)

        if kind in ("text", "service_type"):
            return raw_clean if raw_clean else "not_provided"

        norms = (self.intent_config or {}).get("normalizers", {})
        table = norms.get(kind, {}) or {}

        # Match by synonyms
        for canonical, synonyms in table.items():
            for s in synonyms or []:
                if raw_lc == str(s).lower().strip():
                    return canonical

        # If user typed canonical already (rare but fine)
        if raw_clean in table:
            return raw_clean

        # Special numeric handling for budget
        if kind == "budget" and raw_lc.isdigit():
            n = int(raw_lc)
            if n < 50:
                return "<50"
            if 50 <= n <= 100:
                return "50-100"
            if 100 < n <= 300:
                return "100-300"
            return "not_provided"

        return "not_provided"

    def _normalize_constraints(self, raw: str) -> str:
        raw_clean = self._norm_text(raw)
        if not raw_clean:
            return ""

        ignore = (self.intent_config or {}).get("normalizers", {}).get("constraints_ignore", []) or []
        ignore_set = {str(x).lower().strip() for x in ignore}
        if self._norm_lc(raw_clean) in ignore_set:
            return ""
        return raw_clean

    def _question_for_field(self, intent: Dict[str, Any], field: str) -> str:
        """Find question in intent flow for a specific field, fallback to a generic prompt."""
        for step in (intent.get("flow", []) or []):
            if step.get("field") == field and step.get("question"):
                return str(step["question"])
        # fallbacks
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
        return f"Please provide: {field}"

    # ---------- state handlers ----------
    def run(self) -> IntakeResult:
        self._set_state(S0)
        print("I can help you prepare a service request. I’ll ask a few quick questions to understand your needs.\n")

        # If we have pending missing fields from a previous run, continue intake
        pending = self.memory.get("missing_fields", []) or []
        last_intent_id = self.memory.get("last_intent_id")

        # If we know last intent, reuse it; else default fallback
        intents = self.intent_config.get("intents", []) or []
        last_intent = None
        if last_intent_id and intents:
            for it in intents:
                if it.get("id") == last_intent_id:
                    last_intent = it
                    break
        if last_intent is None:
            # fallback_unknown if exists
            for it in intents:
                if it.get("id") == "service_request":
                    last_intent = it
                    break
            if last_intent is None:
                last_intent = {"id": "fallback_unknown", "flow": []}

        if pending:
            print("Continuing your previous request...\n")
            self._set_state(S1)

            # Ask only missing fields (using config questions)
            for field in list(pending):
                q = self._question_for_field(last_intent, field)
                raw = self._ask(q)

                if field == "constraints":
                    val = self._normalize_constraints(raw)
                    if val:
                        self.result.request.details.constraints.append(val)
                    pending.remove(field)
                    continue

                if field == "service_type":
                    val = self._normalize_value("service_type", raw)
                    if val != "not_provided" and self._is_valid_service_type(val):
                        self.memory["collected"]["service_type"] = val
                        self.result.request.summary = f"User is requesting: {val}"
                        pending.remove(field)
                    continue

                if field == "urgency":
                    self.result.request.details.urgency = self._normalize_value("urgency", raw)
                    pending.remove(field)
                elif field == "timeline":
                    self.result.request.details.timeline = self._normalize_value("timeline", raw)
                    pending.remove(field)
                elif field == "location":
                    self.result.request.details.location = self._normalize_value("text", raw)
                    if not self.result.request.details.location:
                        self.result.request.details.location = "not_provided"
                    pending.remove(field)
                elif field == "budget_range":
                    self.result.request.details.budget_range = self._normalize_value("budget", raw)
                    pending.remove(field)
                else:
                    # store anything unknown into memory
                    self.memory["collected"][field] = self._normalize_value("text", raw)
                    pending.remove(field)

            # Recompute readiness
            self._set_state(S4)
            missing_now = self._compute_missing_fields()

            # Constraint check: budget not a fit (safe)
            budget_raw = self.result.request.details.budget_range or ""
            budget_lc = budget_raw.lower().strip()
            if "free" in budget_lc or budget_lc == "<10":
                self.result.readiness.status = "not_a_fit"
                self.result.readiness.notes = "Budget constraint is not compatible with typical service delivery."
                self.result.handoff.recommended_action = "route_human"
                self.result.handoff.routing_hint = "not_a_fit_review"
                self.result.handoff.next_questions = []
                self._set_state(S5)
                return self._done()

            if missing_now:
                self.memory["missing_fields"] = missing_now
                self._finalize(not_ready_fields=missing_now, next_qs=self._questions_for_missing(missing_now, last_intent))
            else:
                self.memory["missing_fields"] = []
                self.result.readiness.status = "ready"
                self.result.readiness.notes = "Request has sufficient information for human handling."
                self.result.handoff.recommended_action = "route_human"
                self.result.handoff.routing_hint = "human_review"
                self.result.handoff.next_questions = []

            self._set_state(S5)
            return self._done()

        # ---------- Normal run (config-driven) ----------
        self._set_state(S1)
        first = self._ask("Tell me briefly what you need help with today.")

        intent = self._pick_intent(first)
        intent_id = intent.get("id", "fallback_unknown")
        self.memory["last_intent_id"] = intent_id

        # Handle informational intent quickly
        if intent_id == "general_question":
            self.result.request.request_type = "general_question"
            self.result.request.service_category = "informational"
            self.result.request.summary = f"User asked a general question: {first}"

            self.result.readiness.status = "not_ready"
            self.result.readiness.notes = "Informational request (not an intake)."

            handoff = intent.get("handoff", {}) or {}
            self.result.handoff.recommended_action = handoff.get("recommended_action", "completed")
            self.result.handoff.routing_hint = handoff.get("routing_hint", "informational")
            self.result.handoff.next_questions = []

            print("\nI can help with service requests by preparing an intake. For general info, please check the service page or contact support.\n")
            self._set_state(S5)
            return self._done()

        # Setup defaults from config
        defaults = self.intent_config.get("defaults", {}) or {}
        self.result.request.request_type = "service_request"
        self.result.request.service_category = defaults.get("service_category", "generic_service")

        # Run intent flow
        flow = intent.get("flow", []) or []
        required_fields: List[str] = []

        self._set_state(S2)

        for step in flow:
            field = step.get("field")
            question = step.get("question", "")
            required = bool(step.get("required", False))
            normalizer = step.get("normalize", "text")

            if required and field:
                required_fields.append(field)

            raw = self._ask(str(question))

            if field == "constraints":
                val = self._normalize_constraints(raw)
                if val:
                    self.result.request.details.constraints.append(val)
                continue

            if field == "service_type":
                # allow reuse of the first user message if it already looks like a service type
                raw_service = raw
                if (not raw_service) and self._is_valid_service_type(first):
                    raw_service = first

                val = self._normalize_value("service_type", raw_service)
                if val != "not_provided" and self._is_valid_service_type(val):
                    self.memory["collected"]["service_type"] = val
                    self.result.request.summary = f"User is requesting: {val}"
                continue

            # Standard fields
            if field == "urgency":
                self.result.request.details.urgency = self._normalize_value("urgency", raw)
            elif field == "timeline":
                self.result.request.details.timeline = self._normalize_value("timeline", raw)
            elif field == "location":
                val = self._normalize_value("text", raw)
                self.result.request.details.location = val if val != "not_provided" else "not_provided"
            elif field == "budget_range":
                self.result.request.details.budget_range = self._normalize_value("budget", raw)
            else:
                # future: custom fields go to memory
                if field:
                    self.memory["collected"][field] = self._normalize_value("text", raw)

        # Readiness: check required fields
        self._set_state(S4)
        missing = self._compute_missing_fields(required_fields=required_fields)

        if missing:
            self._finalize(not_ready_fields=missing, next_qs=self._questions_for_missing(missing, intent))
        else:
            self.result.readiness.status = "ready"
            self.result.readiness.notes = "Request has sufficient information for human handling."

            handoff = intent.get("handoff", {}) or defaults.get("handoff", {}) or {}
            self.result.handoff.recommended_action = handoff.get("recommended_action", "route_human")
            self.result.handoff.routing_hint = handoff.get("routing_hint", "human_review")
            self.result.handoff.next_questions = []

        self._set_state(S5)
        return self._done()

    def _compute_missing_fields(self, required_fields: Optional[List[str]] = None) -> List[str]:
        missing: List[str] = []
        d = self.result.request.details
        required_fields = required_fields or ["service_type", "location"]  # safe default

        if "service_type" in required_fields:
            service_type = self.memory.get("collected", {}).get("service_type")
            if not self._is_valid_service_type(service_type):
                missing.append("service_type")

        if "location" in required_fields:
            if not d.location or d.location == "not_provided":
                missing.append("location")

        if "budget_range" in required_fields:
            if not d.budget_range or d.budget_range == "not_provided":
                missing.append("budget_range")

        if "timeline" in required_fields:
            if not d.timeline or d.timeline == "not_provided":
                missing.append("timeline")

        return missing

    def _questions_for_missing(self, missing: List[str], intent: Dict[str, Any] | None = None) -> List[str]:
        intent = intent or {"flow": []}
        qs: List[str] = []
        for field in missing:
            qs.append(self._question_for_field(intent, field))
        return qs

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
