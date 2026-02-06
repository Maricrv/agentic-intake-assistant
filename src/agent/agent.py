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

    # ----------------------------
    # Intent selection
    # ----------------------------
    def _pick_intent(self, first_text: str) -> Dict[str, Any]:
        t = norm_lc(first_text)
        intents: List[Dict[str, Any]] = self.intent_config.get("intents", []) or []

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

    # ----------------------------
    # Config helpers
    # ----------------------------
    def _opening_message(self, intent: Dict[str, Any]) -> str:
        msg = intent.get("opening_message") or self.defaults.get("opening_message")
        if not msg:
            msg = "I can help you create a service request. I’ll ask a few quick questions.\n"
        return str(msg)

    def _question_for_field(self, intent: Dict[str, Any], field: str) -> str:
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

    def _normalizer_for_field(self, intent: Dict[str, Any], field: str) -> str:
        for step in (intent.get("flow", []) or []):
            if step.get("field") == field:
                return step.get("normalize", "text")
        return "text"

    def _required_fields_from_intent(self, intent: Dict[str, Any]) -> List[str]:
        rf = intent.get("required_fields")
        if isinstance(rf, list) and rf:
            return [str(x) for x in rf]

        return [
            str(step["field"])
            for step in (intent.get("flow", []) or [])
            if step.get("required") and step.get("field")
        ]

    def _allowed_for_field(self, intent: Dict[str, Any], field: str) -> List[str]:
        fields = intent.get("fields", {}) or {}
        spec = fields.get(field, {}) or {}
        allowed = spec.get("allowed") or []
        return [str(x) for x in allowed]

    # ----------------------------
    # Prefill handling
    # ----------------------------
    def _extract_prefill_safe(self, text: str) -> Dict[str, Any]:
        try:
            data = extract_prefill(text, self.intent_config)  # type: ignore
        except TypeError:
            data = extract_prefill(text)  # type: ignore
        return data or {}

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

        if field == "service_type":
            norm = normalize_value("service_type", value, self.intent_config)
            if norm != "not_provided":
                d.service_type = norm
                self._log(f"prefill: service_type='{norm}'")
            return

        # Anything else -> extra_fields
        d.extra_fields[field] = normalize_value("text", value, self.intent_config)
        self._log(f"prefill: extra_fields['{field}']")

    # ----------------------------
    # Apply field values
    # ----------------------------
    def _apply_field(self, intent: Dict[str, Any], field: str, raw: str) -> None:
        if not norm_text(raw):
            return

        d = self.result.request.details
        kind = self._normalizer_for_field(intent, field)

        # constraints -> list
        if field == "constraints":
            val = normalize_constraints(raw, self.intent_config)
            if val:
                d.constraints.append(val)
                self._log(f"user_set: constraints += '{val}'")
            return

        # issue_description
        if field == "issue_description":
            val = normalize_value("text", raw, self.intent_config)
            self.memory["collected"]["issue_description"] = val
            d.issue_description = val
            self._log("user_set: issue_description")
            return

        # location with confirmation
        if field == "location":
            corrected = self.corrector.maybe_correct_location_with_confirmation(raw)
            raw_to_use = corrected if corrected else raw

            if corrected and corrected != raw:
                if "location_correction" not in self.result.request.sources.get("llm_used", []):
                    self.result.request.sources["llm_used"].append("location_correction")
                self._log(f"llm_suggestion_accepted: location='{corrected}'")

            val = normalize_value("text", raw_to_use, self.intent_config)
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

        # service_type (allowed per intent + optional LLM)
        if field == "service_type":
            val = normalize_value("service_type", raw, self.intent_config)
            allowed = self._allowed_for_field(intent, "service_type")

            if allowed:
                allowed_lc = {a.lower(): a for a in allowed}
                if val != "not_provided" and val.lower() not in allowed_lc:
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

                if val.lower() in allowed_lc:
                    val = allowed_lc[val.lower()]

            if val == "not_provided":
                return

            # Keep old validator only if you still want strict IT types.
            # For multi-domain, if allowed exists it is enough.
            if not allowed and not is_valid_service_type(val):
                # If your old validator is too strict, accept raw
                val = norm_text(raw)

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
                label = str(intent.get("label") or "Service request")
                self.result.request.summary = f"{label}: {val}"
                self._log(f"user_set: service_type='{val}'")
            return

        # If kind says urgency/timeline/budget, normalize properly and store:
        if kind == "urgency":
            val = normalize_value("urgency", raw, self.intent_config)
            if val == "not_provided":
                return
            if field == "urgency":
                d.urgency = val
                self._log(f"user_set: urgency='{val}'")
            else:
                d.extra_fields[field] = val
                self._log(f"user_set: extra_fields['{field}']='{val}'")
            return

        if kind == "timeline":
            val = normalize_value("timeline", raw, self.intent_config)
            if val == "not_provided":
                return
            if field == "timeline":
                d.timeline = val
                self._log(f"user_set: timeline='{val}'")
            else:
                d.extra_fields[field] = val
                self._log(f"user_set: extra_fields['{field}']='{val}'")
            return

        if kind == "budget":
            val = normalize_value("budget", raw, self.intent_config)
            if val == "not_provided":
                return
            if field == "budget_range":
                d.budget_range = val
                self._log(f"user_set: budget_range='{val}'")
            else:
                d.extra_fields[field] = val
                self._log(f"user_set: extra_fields['{field}']='{val}'")
            return

        # Default: store as text into extra_fields (domain-specific)
        val = normalize_value(kind, raw, self.intent_config)
        self.memory["collected"][field] = val
        d.extra_fields[field] = val
        self._log(f"user_set: extra_fields['{field}']='{val}'")

    # ----------------------------
    # Followups and handoff
    # ----------------------------
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
        default_handoff = self.defaults.get("handoff", {}) or {
            "recommended_action": "route_human",
            "routing_hint": "human_review",
        }
        intent_handoff = intent.get("handoff", {}) or {}
        rec = (intent_handoff.get("recommended_action") or "").strip()
        if rec in {"route_human", "completed"}:
            return intent_handoff
        return default_handoff

    # ----------------------------
    # Main run
    # ----------------------------
    def run(self) -> IntakeResult:
        self._set_state(S0)

        intents = self.intent_config.get("intents", []) or []
        pending = self.memory.get("missing_fields", []) or []
        last_intent_id = self.memory.get("last_intent_id")

        # Resume intent
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

        # Print default opening message (neutral)
        print(self._opening_message({"opening_message": self.defaults.get("opening_message")}))

        first_text = self._ask(self._question_for_field({"flow": []}, "issue_description"))
        self._apply_field({"flow": [{"field": "issue_description", "normalize": "text"}]}, "issue_description", first_text)

        # Prefill
        prefill = self._extract_prefill_safe(first_text)
        if prefill:
            self._log("prefill: extracted_fields_from_first_message")
            for k, v in prefill.items():
                self._apply_prefill(str(k), str(v))

        # Intent
        intent = self._pick_intent(first_text)
        intent_id = intent.get("id", "fallback_unknown")
        self.memory["last_intent_id"] = intent_id
        self.result.request.intent_id = intent_id

        # Apply request metadata from config (✅ generic)
        self.result.request.request_type = str(intent.get("request_type") or self.defaults.get("request_type") or self.result.request.request_type)
        self.result.request.service_category = str(intent.get("service_category") or self.defaults.get("service_category") or self.result.request.service_category)

        # Print intent opening message (optional)
        print(self._opening_message(intent))

        # Summary base label
        self.result.request.summary = str(intent.get("label") or "Service request")

        # Run flow
        flow = intent.get("flow", []) or []
        required_fields = self._required_fields_from_intent(intent)

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
                if field in d.extra_fields and norm_text(str(d.extra_fields.get(field))):
                    already = True

            if already:
                continue

            raw = self._ask(str(question) if question else self._question_for_field(intent, field))
            self._apply_field(intent, field, raw)

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

    # ----------------------------
    # Missing fields / finalize / done
    # ----------------------------
    def _compute_missing_fields(self, required_fields: Optional[List[str]] = None) -> List[str]:
        missing: List[str] = []
        d = self.result.request.details

        required_fields = required_fields or ["issue_description", "service_type", "location"]

        for f in required_fields:
            if f == "issue_description":
                issue = d.issue_description
                if not norm_text(issue) or issue == "not_provided":
                    missing.append(f)
                continue

            if f == "service_type":
                st = d.service_type
                if not norm_text(st) or st == "not_provided":
                    missing.append(f)
                continue

            if f == "location":
                loc = d.location
                if not norm_text(loc) or loc == "not_provided":
                    missing.append(f)
                continue

            # Any other required field: check extra_fields
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
