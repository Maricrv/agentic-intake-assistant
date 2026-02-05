from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .schema import IntakeResult
from .llm_client import LLMClient
from .field_corrector import FieldCorrector
from .field_extractor import extract_prefill

from .normalizers import (
    norm_text,
    norm_lc,
    normalize_value,
    normalize_constraints,
    is_valid_service_type,
)
from .consistency import keep_existing_on_conflict


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

        # traceability (schema also sets defaults, but safe to ensure here)
        if not isinstance(self.result.request.sources, dict) or not self.result.request.sources:
            self.result.request.sources = {"prefill": False, "llm_used": []}
        else:
            self.result.request.sources.setdefault("prefill", False)
            self.result.request.sources.setdefault("llm_used", [])

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

    def _pick_intent(self, first_text: str) -> Dict[str, Any]:
        t = norm_lc(first_text)
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
            chosen = candidates[0][1]
            self._log(f"intent_selected: {chosen.get('id')} (rule_match)")
            return chosen

        for it in intents:
            if it.get("id") == "pre_quote_request":
                self._log("intent_selected: pre_quote_request (default)")
                return it

        if always_intents:
            always_intents.sort(key=lambda x: x[0], reverse=True)
            chosen = always_intents[0][1]
            self._log(f"intent_selected: {chosen.get('id')} (always)")
            return chosen

        self._log("intent_selected: fallback_unknown (no_match)")
        return {"id": "fallback_unknown", "flow": []}

    def _infer_service_type_from_text(self, text: str) -> str:
        tl = norm_lc(text)
        table = (self.intent_config or {}).get("normalizers", {}).get("service_type", {}) or {}

        candidates: List[tuple[int, str, str]] = []
        for canonical, synonyms in table.items():
            for s in (synonyms or []):
                phrase = str(s).lower().strip()
                if phrase and phrase in tl:
                    candidates.append((len(phrase), canonical, phrase))

        if not candidates:
            return "not_provided"

        candidates.sort(key=lambda x: x[0], reverse=True)
        _, canonical, phrase = candidates[0]
        self._log(f"service_type_inferred: {canonical} (matched_phrase='{phrase}')")
        return canonical

    def _normalizer_for_field(self, intent: Dict[str, Any], field: str) -> str:
        for step in (intent.get("flow", []) or []):
            if step.get("field") == field:
                return step.get("normalize", "text")
        return "text"

    def _question_for_field(self, intent: Dict[str, Any], field: str) -> str:
        for step in (intent.get("flow", []) or []):
            if step.get("field") == field and step.get("question"):
                return str(step["question"])
        if field == "service_type":
            return "What type of service is this? (repair / installation / maintenance / consultation)"
        if field == "location":
            return "What is your location (city/country)?"
        if field == "budget_range":
            return "Do you have an estimated budget? (example: <50, 50-100, 100-300, 300-500, 500-1000, not sure)"
        if field == "timeline":
            return "When do you want this addressed? (within_24h / within_1_week / within_2_weeks)"
        if field == "urgency":
            return "Is this urgent or flexible? (urgent/flexible)"
        if field == "issue_description":
            return "Tell me briefly what you need help with so we can prepare a pre-quotation."
        return f"Please provide: {field}"

    def _required_fields_from_intent(self, intent: Dict[str, Any]) -> List[str]:
        return [
            str(step["field"])
            for step in (intent.get("flow", []) or [])
            if step.get("required") and step.get("field")
        ]

    def _apply_prefill(self, field: str, value: str) -> None:
        value = norm_text(value)
        if not value:
            return

        d = self.result.request.details
        self.result.request.sources["prefill"] = True

        if field == "location":
            norm = normalize_value("text", value, self.intent_config)
            if norm != "not_provided":
                d.location = norm
                self._log(f"prefill: location='{norm}'")
            return

        if field == "timeline":
            norm = normalize_value("timeline", value, self.intent_config)
            if norm != "not_provided":
                d.timeline = norm
                self._log(f"prefill: timeline='{norm}'")
            return

        if field == "budget_range":
            norm = normalize_value("budget", value, self.intent_config)
            if norm != "not_provided":
                d.budget_range = norm
                self._log(f"prefill: budget_range='{norm}'")
            return

        if field == "urgency":
            norm = normalize_value("urgency", value, self.intent_config)
            if norm != "not_provided":
                d.urgency = norm
                self._log(f"prefill: urgency='{norm}'")
            return

    def _apply_field(self, intent: Dict[str, Any], field: str, raw: str) -> None:
        if not norm_text(raw):
            return

        d = self.result.request.details
        kind = self._normalizer_for_field(intent, field)

        # ---------------- constraints ----------------
        if field == "constraints":
            val = normalize_constraints(raw, self.intent_config)
            if val:
                d.constraints.append(val)
                self._log(f"user_set: constraints += '{val}'")
            return

        # ---------------- issue_description ----------------
        if field == "issue_description":
            val = normalize_value("text", raw, self.intent_config)
            self.memory["collected"]["issue_description"] = val
            d.issue_description = val
            self._log("user_set: issue_description")
            return

        # ---------------- service_type (LLM-assisted) ----------------
        if field == "service_type":
            # Start with normalizer (cheap)
            val = normalize_value("service_type", raw, self.intent_config)
            allowed = ["repair", "installation", "maintenance", "consultation"]

            # If not in allowed, ask LLM to map and confirm
            if val != "not_provided" and val.lower() not in allowed:
                resp = self.llm.suggest_service_type_correction(val, allowed)
                if resp:
                    proposed = (resp.text or "").strip()
                    ans = input(f'I think you meant "{proposed}". Use that? (y/n)\n> ').strip().lower()
                    if ans in {"y", "yes"}:
                        val = proposed
                        if "service_type_correction" not in self.result.request.sources.get("llm_used", []):
                            self.result.request.sources["llm_used"].append("service_type_correction")
                        self._log(f"llm_suggestion_accepted: service_type='{val}'")
                    else:
                        self._log(f"llm_suggestion_rejected: service_type='{proposed}'")

            if val == "not_provided" or not is_valid_service_type(val):
                return

            # Conflict strategy: keep existing if conflict
            res = keep_existing_on_conflict(
                "service_type",
                d.service_type,
                val,
                self.result.readiness.inconsistencies,
                self._log,
            )
            if res.applied:
                self.memory["collected"]["service_type"] = val
                d.service_type = val
                self.result.request.summary = f"Pre-quote for: {val}"
                self._log(f"user_set: service_type='{val}'")
            return

        # ---------------- urgency ----------------
        if field == "urgency":
            val = normalize_value("urgency", raw, self.intent_config)
            if val == "not_provided":
                return

            res = keep_existing_on_conflict(
                "urgency",
                d.urgency,
                val,
                self.result.readiness.inconsistencies,
                self._log,
            )
            if res.applied:
                d.urgency = val
                self._log(f"user_set: urgency='{val}'")
            return

        # ---------------- timeline ----------------
        if field == "timeline":
            val = normalize_value("timeline", raw, self.intent_config)
            if val == "not_provided":
                return

            res = keep_existing_on_conflict(
                "timeline",
                d.timeline,
                val,
                self.result.readiness.inconsistencies,
                self._log,
            )
            if res.applied:
                d.timeline = val
                self._log(f"user_set: timeline='{val}'")
            return

        # ---------------- location (LLM-assisted) ----------------
        if field == "location":
            corrected = self.corrector.maybe_correct_location_with_confirmation(raw)
            raw_to_use = corrected if corrected else raw

            if corrected:
                if "location_correction" not in self.result.request.sources.get("llm_used", []):
                    self.result.request.sources["llm_used"].append("location_correction")
                self._log(f"llm_suggestion_accepted: location='{corrected}'")

            val = normalize_value(kind, raw_to_use, self.intent_config)
            if val == "not_provided":
                return

            res = keep_existing_on_conflict(
                "location",
                d.location,
                val,
                self.result.readiness.inconsistencies,
                self._log,
            )
            if res.applied:
                d.location = val
                self._log(f"user_set: location='{val}'")
            return

        # ---------------- budget_range ----------------
        if field == "budget_range":
            val = normalize_value("budget", raw, self.intent_config)
            if val == "not_provided":
                return

            res = keep_existing_on_conflict(
                "budget",
                d.budget_range,
                val,
                self.result.readiness.inconsistencies,
                self._log,
            )
            if res.applied:
                d.budget_range = val
                self._log(f"user_set: budget_range='{val}'")
            return

        # fallback
        self.memory["collected"][field] = normalize_value(kind, raw, self.intent_config)

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
        print("I can help you prepare a pre-quotation request. I’ll ask a few quick questions to understand your needs.\n")

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

        # Resume pending session
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
                self._finalize(
                    not_ready_fields=missing_now,
                    next_qs=self._questions_for_missing(missing_now, last_intent),
                )
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

        issue_q = "Tell me briefly what you need help with so we can prepare a pre-quotation."
        for it in intents:
            if it.get("id") == "pre_quote_request":
                issue_q = self._question_for_field(it, "issue_description")
                break

        first_text = self._ask(issue_q)
        self._apply_field({"flow": [{"field": "issue_description", "normalize": "text"}]}, "issue_description", first_text)

        # Prefill
        prefill = extract_prefill(first_text)
        if prefill:
            self._log("prefill: extracted_fields_from_first_message")
            for k, v in prefill.items():
                self._apply_prefill(k, v)

        # Intent
        intent = self._pick_intent(first_text)
        intent_id = intent.get("id", "fallback_unknown")
        self.memory["last_intent_id"] = intent_id
        self.result.request.intent_id = intent_id

        defaults = self.intent_config.get("defaults", {}) or {}
        self.result.request.request_type = "pre_quote"
        self.result.request.service_category = (
            intent.get("service_category") or defaults.get("service_category", "technical_services")
        )

        # Service type inference from first message (if missing)
        if self.result.request.details.service_type in ("", "not_provided"):
            inferred = self._infer_service_type_from_text(first_text)
            if inferred != "not_provided":
                self.memory["collected"]["service_type"] = inferred
                self.result.request.details.service_type = inferred
                self.result.request.summary = f"Pre-quote for: {inferred}"

        flow = intent.get("flow", []) or []
        required_fields: List[str] = []
        self._set_state(S2)

        for step in flow:
            field = step.get("field")
            question = step.get("question", "")
            required = bool(step.get("required", False))

            if required and field:
                required_fields.append(field)

            if field == "issue_description" or not field:
                continue

            d = self.result.request.details

            already = False
            if field == "service_type" and d.service_type != "not_provided":
                already = True
            elif field == "urgency" and d.urgency != "not_provided":
                already = True
            elif field == "timeline" and d.timeline != "not_provided":
                already = True
            elif field == "location" and d.location != "not_provided":
                already = True
            elif field == "budget_range" and d.budget_range != "not_provided":
                already = True
            elif field == "constraints" and len(d.constraints) > 0:
                already = True

            if already:
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
        required_fields = required_fields or ["issue_description", "service_type", "location"]

        if "issue_description" in required_fields:
            issue = self.memory.get("collected", {}).get("issue_description", "")
            if not norm_text(issue) or issue == "not_provided":
                missing.append("issue_description")

        if "service_type" in required_fields:
            service_type = self.memory.get("collected", {}).get("service_type") or d.service_type
            if not is_valid_service_type(service_type):
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
