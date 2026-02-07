from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .schema import IntakeResult
from .llm_client import LLMClient
from .field_corrector import FieldCorrector

from .normalizers import norm_text
from .intent_router import IntentRouter
from .field_handlers import FieldHandlers


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
        self.defaults: Dict[str, Any] = (self.intent_config.get("defaults") or {})

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

        # Ensure sources always present
        if not isinstance(self.result.request.sources, dict) or not self.result.request.sources:
            self.result.request.sources = {"prefill": False, "llm_used": []}
        else:
            self.result.request.sources.setdefault("prefill", False)
            self.result.request.sources.setdefault("llm_used", [])

        # Delegates
        self.router = IntentRouter(self.intent_config, self.defaults, self._log)
        self.fields = FieldHandlers(
            intent_config=self.intent_config,
            llm=self.llm,
            corrector=self.corrector,
            result=self.result,
            memory=self.memory,
            log=self._log,
        )

    def export_state(self) -> dict:
        self.memory["last_state"] = self.state.name
        return self.memory

    def _ask(self, question: str) -> str:
        self.turns += 1
        return input(question + "\n> ").strip()

    def _set_state(self, st: AgentState):
        self.state = st
        self.result.session.state = st.name

    def _log(self, msg: str) -> None:
        self.result.request.decision_log.append(msg)

    def _ask_and_apply_followups(self, intent: Dict[str, Any], missing_fields: List[str]) -> None:
        for field in missing_fields:
            self.memory.setdefault("attempts", {})
            self.memory["attempts"][field] = int(self.memory["attempts"].get(field, 0)) + 1

            raw = self._ask(self.router.question_for_field(intent, field))

            if not raw and self.memory["attempts"][field] >= self.MAX_EMPTY_TRIES_PER_FIELD_IN_RUN:
                print("(No problem — we can continue for now.)")
                continue

            kind = self.router.normalizer_for_field(intent, field)
            allowed = self.router.allowed_for_field(intent, field) if field == "service_type" else []
            self.fields.apply_field(intent, field, raw, kind, allowed)

    def _handoff_for_ready(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        default_handoff = self.defaults.get("handoff", {}) or {
            "recommended_action": "route_human",
            "routing_hint": "human_review",
        }
        intent_handoff = intent.get("handoff", {}) or {}
        rec = (intent_handoff.get("recommended_action") or "").strip()
        if rec in {"route_human", "completed"}:
            return intent_handoff
        return default_handoff

    def _compute_missing_fields(self, required_fields: Optional[List[str]] = None) -> List[str]:
        missing: List[str] = []
        d = self.result.request.details
        if required_fields is None:
            required_fields = ["issue_description", "service_type", "location"]


        for f in required_fields:
            if f == "issue_description":
                if not norm_text(d.issue_description) or d.issue_description == "not_provided":
                    missing.append(f)
                continue

            if f == "service_type":
                if not norm_text(d.service_type) or d.service_type == "not_provided":
                    missing.append(f)
                continue

            if f == "location":
                if not norm_text(d.location) or d.location == "not_provided":
                    missing.append(f)
                continue

            v = d.extra_fields.get(f)
            if isinstance(v, str):
                if not norm_text(v) or v == "not_provided":
                    missing.append(f)
            elif isinstance(v, list):
                if len(v) == 0:
                    missing.append(f)
            elif v is None:
                missing.append(f)

        return missing

    def _finalize(self, not_ready_fields: List[str], next_qs: List[str], intent: Dict[str, Any]):
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

    def run(self) -> IntakeResult:
        self._set_state(S0)

        intents = self.router.intents()
        pending = self.memory.get("missing_fields", []) or []
        last_intent_id = self.memory.get("last_intent_id")

        # Resume intent selection
        last_intent = self.router.find_intent_by_id(last_intent_id) or self.router.fallback_intent()

        # Resume pending session
        if pending:
            print("Continuing your previous request...\n")
            self._set_state(S1)

            self.result.request.intent_id = last_intent.get("id", "fallback_unknown")
            required_fields = self.router.required_fields_from_intent(last_intent)

            for field in pending:
                raw = self._ask(self.router.question_for_field(last_intent, field))
                kind = self.router.normalizer_for_field(last_intent, field)
                allowed = self.router.allowed_for_field(last_intent, field) if field == "service_type" else []
                self.fields.apply_field(last_intent, field, raw, kind, allowed)

            self._set_state(S4)
            missing_now = self._compute_missing_fields(required_fields=required_fields)

            rounds = 0
            while missing_now and rounds < self.MAX_FOLLOWUP_ROUNDS:
                rounds += 1
                print("\nI’m still missing a couple of details to complete your request.\n")
                self._ask_and_apply_followups(last_intent, missing_now)
                missing_now = self._compute_missing_fields(required_fields=required_fields)

            if missing_now:
                next_qs = [self.router.question_for_field(last_intent, f) for f in missing_now]
                self._finalize(missing_now, next_qs, last_intent)
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

        # New session
        self._set_state(S1)

        # Single opening message (default)
        print(self.router.opening_message({"opening_message": self.defaults.get("opening_message")}))

        first_text = self._ask(self.router.question_for_field({"flow": []}, "issue_description"))
        self.fields.apply_field(
            intent={"flow": [{"field": "issue_description", "normalize": "text"}], "label": "Service request"},
            field="issue_description",
            raw=first_text,
            normalizer_kind="text",
            allowed=[],
        )

        # Prefill (optional)
        prefill = self.fields.extract_prefill_safe(first_text)
        if prefill:
            self._log("prefill: extracted_fields_from_first_message")
            for k, v in prefill.items():
                self.fields.apply_prefill(str(k), str(v))

        # Pick intent
        intent = self.router.pick_intent(first_text)
        intent_id = intent.get("id", "fallback_unknown")
        self.memory["last_intent_id"] = intent_id
        self.result.request.intent_id = intent_id

        # Apply request metadata
        self.result.request.request_type = str(intent.get("request_type") or self.defaults.get("request_type") or self.result.request.request_type)
        self.result.request.service_category = str(intent.get("service_category") or self.defaults.get("service_category") or self.result.request.service_category)

        # Intent-specific opening message (if you want it). If you prefer ONLY one welcome, comment this line.
        print(self.router.opening_message(intent))

        # Summary label
        self.result.request.summary = str(intent.get("label") or "Service request")

        # Run flow
        flow = intent.get("flow", []) or []
        required_fields = self.router.required_fields_from_intent(intent)

        self._set_state(S2)

        for step in flow:
            field = step.get("field")
            question = step.get("question", "")
            if field == "issue_description" or not field:
                continue

            # Skip if already filled (details or extra_fields)
            d = self.result.request.details
            already = False
            if hasattr(d, field):
                cur = getattr(d, field)
                if isinstance(cur, str) and cur and cur != "not_provided":
                    already = True
                elif isinstance(cur, list) and len(cur) > 0:
                    already = True
            else:
                if field in d.extra_fields and norm_text(str(d.extra_fields.get(field) or "")):
                    already = True

            if already:
                continue

            q = str(question) if question else self.router.question_for_field(intent, field)
            raw = self._ask(q)
            kind = self.router.normalizer_for_field(intent, field)
            allowed = self.router.allowed_for_field(intent, field) if field == "service_type" else []
            self.fields.apply_field(intent, field, raw, kind, allowed)

        # Final readiness
        self._set_state(S4)
        missing = self._compute_missing_fields(required_fields=required_fields)

        rounds = 0
        while missing and rounds < self.MAX_FOLLOWUP_ROUNDS:
            rounds += 1
            print("\nI’m still missing a couple of details to complete your request.\n")
            self._ask_and_apply_followups(intent, missing)
            missing = self._compute_missing_fields(required_fields=required_fields)

        if missing:
            next_qs = [self.router.question_for_field(intent, f) for f in missing]
            self._finalize(missing, next_qs, intent)
        else:
            self.result.readiness.status = "ready"
            self.result.readiness.notes = "Request has sufficient information for human handling."
            h = self._handoff_for_ready(intent)
            self.result.handoff.recommended_action = h.get("recommended_action", "route_human")
            self.result.handoff.routing_hint = h.get("routing_hint", "human_review")
            self.result.handoff.next_questions = []

        self._set_state(S5)
        return self._done()
