from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class Channel:
    source: str = "cli"
    user_id: str = "local_user"
    username: str = "local"
    timestamp_utc: str = field(default_factory=utc_now_iso)


@dataclass
class Session:
    session_id: str = "sess_local"
    language: str = "en"
    state: str = "S0"


@dataclass
class RequestDetails:
    issue_description: str = "not_provided"
    service_type: str = "not_provided"       # repair | installation | maintenance | consultation
    urgency: str = "not_provided"            # urgent | flexible | not_provided
    timeline: str = "not_provided"           # within_24h | within_1_week | within_2_weeks | not_provided
    location: str = "not_provided"           # free text
    budget_range: str = "not_provided"       # "<50" | "50-100" | "100-300" | "300-500" | "500-1000" | not_provided
    constraints: List[str] = field(default_factory=list)
    attachments: List[str] = field(default_factory=list)


@dataclass
class Request:
    request_type: str = "pre_quote"
    service_category: str = "technical_services"
    intent_id: str = "fallback_unknown"
    summary: str = ""
    details: RequestDetails = field(default_factory=RequestDetails)

    # ✅ Product-grade traceability
    decision_log: List[str] = field(default_factory=list)
    sources: Dict[str, Any] = field(default_factory=dict)  # e.g., {"prefill": true, "llm_used": ["location_correction"]}


@dataclass
class Readiness:
    status: str = "not_ready"  # ready | not_ready | not_a_fit
    missing_fields: List[str] = field(default_factory=list)
    inconsistencies: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class Handoff:
    recommended_action: str = "ask_follow_up"  # ask_follow_up | route_human | completed
    next_questions: List[str] = field(default_factory=list)
    routing_hint: str = "human_review"


@dataclass
class Audit:
    conversation_turns: int = 0
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    created_at_utc: str = field(default_factory=utc_now_iso)


@dataclass
class IntakeResult:
    schema_version: str = "1.2"
    request_id: str = "req_local_000001"
    channel: Channel = field(default_factory=Channel)
    session: Session = field(default_factory=Session)
    request: Request = field(default_factory=Request)
    readiness: Readiness = field(default_factory=Readiness)
    handoff: Handoff = field(default_factory=Handoff)
    audit: Audit = field(default_factory=Audit)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
