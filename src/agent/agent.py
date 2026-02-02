from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

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
    Domain-agnostic agentic intake agent.
    Runs a state-based conversation and produces an IntakeResult output.
    """
    def _classify_intent(self, text: str) -> str:
        t = (text or "").lower().strip()
        if not t:
            return "unknown"

        service_keywords = ["repair", "install", "installation", "fix", "quote", "book", "support", "help", "service"]
        info_starters = ["what is", "how do", "how does", "price list", "hours", "location", "address"]

        if any(k in t for k in service_keywords):
            return "service_request"
        if any(t.startswith(s) for s in info_starters):
            return "general_question"
        if t in {"hi", "hello", "hey", "hola"}:
            return "unknown"
        return "unknown"

    def __init__(self, request_id: str = "req_local_000001", session_id: str = "sess_local", previous_state: dict | None = None):

        self.result = IntakeResult(request_id=request_id)
        self.result.session.session_id = session_id
        self.state = S0
        self.turns = 0
        self.memory = {
            "missing_fields": [],
            "collected": {},
            "last_state": "S0"
        }

        if previous_state:
            self.memory.update(previous_state)

    
    def export_state(self) -> dict:
        # Minimal session memory snapshot
        self.memory["last_state"] = self.state.name
        return self.memory


    # ---------- helpers ----------
    def _ask(self, question: str) -> str:
        self.turns += 1
        return input(question + "\n> ").strip()

    def _set_state(self, st: AgentState):
        self.state = st
        self.result.session.state = st.name


    def _is_valid_service_type(self, text: str) -> bool:
        t = (text or "").lower().strip()
        if not t:
            return False

        # If it looks like a general question, it is NOT a service type
        question_markers = ["what", "how", "price", "pricing", "cost", "charge", "rates", "hours", "address", "?"]
        if any(m in t for m in question_markers):
            return False

        # Too short / vague (optional rule)
        if len(t) < 3:
            return False

        return True



    # ---------- state handlers ----------
    def run(self) -> IntakeResult:
        self._set_state(S0)
        print("I can help you prepare a service request. I’ll ask a few quick questions to understand your needs.\n")

        # If we have pending missing fields from a previous run, continue intake
        pending = self.memory.get("missing_fields", [])
        if pending:
            print("Continuing your previous request...\n")

            # Resume: ask only what is missing
            if "service_type" in pending:
                self._set_state(S1)
                service = self._ask("What type of service are you looking for? (repair, installation, consultation, etc.)")

                if service and self._is_valid_service_type(service):
                    self.result.request.summary = f"User is requesting: {service}"
                    self.memory["collected"]["service_type"] = service
                    pending = [f for f in pending if f != "service_type"]
                else:
                    # keep it missing if invalid
                    print("Thanks — I need the service type (e.g., repair, installation, consultation).")

            if "location" in pending:
                self._set_state(S3)
                location = self._ask("What is your location (city/country)?")
                if location:
                    self.result.request.details.location = location
                    self.memory["collected"]["location"] = location
                    pending = [f for f in pending if f != "location"]

            if "budget_range" in pending:
                self._set_state(S3)
                budget = self._ask("Do you have a budget range in mind?")
                if budget:
                    self.result.request.details.budget_range = budget
                    self.memory["collected"]["budget_range"] = budget
                    pending = [f for f in pending if f != "budget_range"]

            # Recompute readiness after collecting missing fields
            self._set_state(S4)
            missing_now = self._compute_missing_fields()
            
            # Constraint check: budget not a fit
            budget = self.result.request.details.budget_range.lower()
            if "free" in budget or budget.strip() == "<10":
                self.result.readiness.status = "not_a_fit"
                self.result.readiness.notes = "Budget constraint is not compatible with typical service delivery."
                self.result.handoff.recommended_action = "route_human"
                self.result.handoff.routing_hint = "not_a_fit_review"
                self.result.handoff.next_questions = []
                self._set_state(S5)
                return self._done()

            
            if missing_now:
                self.memory["missing_fields"] = missing_now
                self._finalize(not_ready_fields=missing_now, next_qs=self._questions_for_missing(missing_now))
            else:
                self.memory["missing_fields"] = []
                self.result.readiness.status = "ready"
                self.result.readiness.notes = "Request has sufficient information for human handling."
                self.result.handoff.recommended_action = "route_human"
                self.result.handoff.routing_hint = "human_review"
                self.result.handoff.next_questions = []

            self._set_state(S5)
            return self._done()

        # S1 Intent Clarification
        self._set_state(S1)
        first = self._ask("Tell me briefly what you need help with today.")
        intent = self._classify_intent(first)

        # If the first answer itself is a valid service_type, reuse it
        if intent == "service_request" and self._is_valid_service_type(first):
            service = first
        else:
            service = self._ask(
                "What type of service are you looking for? (repair, installation, consultation, etc.)"
            )

        # Validate before storing
        if service and self._is_valid_service_type(service):
            self.result.request.summary = f"User is requesting: {service}"
            self.memory["collected"]["service_type"] = service
        else:
            self._finalize(
                not_ready_fields=["service_type"],
                next_qs=["What type of service are you looking for? (repair, installation, consultation, etc.)"]
            )
            self._set_state(S5)
            return self._done()

        # If the first answer itself is a valid service_type, reuse it
        if intent == "service_request" and self._is_valid_service_type(first):
            service = first
        else:
            service = self._ask(
                "What type of service are you looking for? (repair, installation, consultation, etc.)"
            )

        if intent == "general_question":
            self.result.request.request_type = "general_question"
            self.result.request.service_category = "informational"

            self.result.request.summary = f"User asked a general question: {first}"

            self.result.readiness.status = "not_ready"
            self.result.readiness.notes = "Informational request (not an intake)."

            self.result.handoff.recommended_action = "completed"
            self.result.handoff.routing_hint = "informational"
            self.result.handoff.next_questions = []

            print("\nI can help with service requests by preparing an intake. For general info, please check the service page or contact support.\n")
            self._set_state(S5)
            return self._done()


        if intent == "unknown":
            # treat as not enough info: follow-up question
            if not first:
                self.result.request.summary = "User started intake but provided no initial message."
            else:
                self.result.request.summary = f"User intent unclear: {first}"

            self._finalize(
                not_ready_fields=["service_type"],
                next_qs=["What type of service are you looking for? (repair, installation, consultation, etc.)"]
            )
            self._set_state(S5)
            return self._done()

        # service_request
        service = self._ask("What type of service are you looking for? (example: repair, consultation, installation)")
        
        if service and self._is_valid_service_type(service):
            self.result.request.summary = f"User is requesting: {service}"
            self.memory["collected"]["service_type"] = service
            pending = [f for f in pending if f != "service_type"]
        else:
            # keep it missing if invalid
            print("Thanks — I need the service type (e.g., repair, installation, consultation).")

        
        if not service:
            self.result.request.summary = "User started service request but did not provide service type."
            self._finalize(not_ready_fields=["service_type"], next_qs=["What type of service are you looking for?"])
            self._set_state(S5)
            return self._done()

        self.result.request.summary = f"User is requesting: {service}"
        self.result.request.service_category = "generic_service"


        # S2 Context Collection
        self._set_state(S2)
        urgency = self._ask("Is this urgent or flexible? (urgent/flexible)")
        urgency_norm = urgency.lower()
        if urgency_norm in ("urgent", "flexible"):
            self.result.request.details.urgency = urgency_norm
        elif urgency_norm:
            self.result.request.details.urgency = "not_provided"

        timeline = self._ask("When do you want this addressed? (within_24h / within_1_week / within_2_weeks)")
        timeline_norm = timeline.strip()
        allowed_tl = {"within_24h", "within_1_week", "within_2_weeks"}
        if timeline_norm in allowed_tl:
            self.result.request.details.timeline = timeline_norm

        # S3 Constraint Evaluation
        self._set_state(S3)
        location = self._ask("What is your location (city/country)?")
        if location:
            self.result.request.details.location = location

        budget = self._ask("Do you have a budget range? (example: <50, 50-100, 100-300, not sure)")
        if budget:
            self.result.request.details.budget_range = budget

        extra = self._ask("Any constraints or notes? (optional)")
        if extra:
            self.result.request.details.constraints.append(extra)

        # S4 Readiness Assessment
        self._set_state(S4)
        missing = self._compute_missing_fields()

        if missing:
            next_qs = self._questions_for_missing(missing)
            self._finalize(not_ready_fields=missing, next_qs=next_qs)
        else:
            self.result.readiness.status = "ready"
            self.result.readiness.notes = "Request has sufficient information for human handling."
            self.result.handoff.recommended_action = "route_human"
            self.result.handoff.routing_hint = "human_review"
            self.result.handoff.next_questions = []

        # S5 Summary & Handoff
        self._set_state(S5)
        return self._done()

    def _compute_missing_fields(self) -> List[str]:
        missing = []
        d = self.result.request.details

        # service_type is stored in memory (not in details)
        service_type = self.memory.get("collected", {}).get("service_type")
        if not self._is_valid_service_type(service_type):
            missing.append("service_type")

        if d.location == "not_provided":
            missing.append("location")

        if d.budget_range == "not_provided":
            missing.append("budget_range")

        return missing


    def _questions_for_missing(self, missing: List[str]) -> List[str]:
        qs = []
        if "location" in missing:
            qs.append("What is your location (city/country)?")
        if "budget_range" in missing:
            qs.append("Do you have a budget range in mind?")
        if "service_type" in missing:
            qs.append("What type of service are you looking for?")
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
