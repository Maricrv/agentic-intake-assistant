from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


@dataclass
class Channel:
    source: str = "cli"
    user_id: str = "local_user"
    username: str = "local"
    timestamp_utc: str = field(default_factory=_utc_now_iso)


@dataclass
class Session:
    session_id: str = "sess_local_001"
    language: str = "en"
    state: str = "S0"


@dataclass
class RequestDetails:
    # Core generic fields (work for ANY domain)
    issue_description: str = "not_provided"
    service_type: str = "not_provided"
    urgency: str = "not_provided"
    timeline: str = "not_provided"
    location: str = "not_provided"
    budget_range: str = "not_provided"

    constraints: List[str] = field(default_factory=list)
    attachments: List[dict] = field(default_factory=list)

    # ✅ Any domain-specific fields go here (tax_year, insurance_type, etc.)
    extra_fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Request:
    request_type: str = "service_request"
    service_category: str = "general_services"
    intent_id: str = "fallback_unknown"
    summary: str = "Service request"
    details: RequestDetails = field(default_factory=RequestDetails)

    decision_log: List[str] = field(default_factory=list)

    # Keep dict for backward compatibility with your current code
    sources: Dict[str, Any] = field(default_factory=lambda: {"prefill": False, "llm_used": []})


@dataclass
class Readiness:
    status: str = "not_ready"  # ready / needs_followup / not_ready
    missing_fields: List[str] = field(default_factory=list)
    inconsistencies: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class Handoff:
    recommended_action: str = "ask_follow_up"  # route_human / ask_follow_up / completed
    next_questions: List[str] = field(default_factory=list)
    routing_hint: str = "human_review"


@dataclass
class Audit:
    conversation_turns: int = 0
    tool_calls: List[str] = field(default_factory=list)
    created_at_utc: str = field(default_factory=_utc_now_iso)


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

    # ✅ This keeps your CLI working (cli.py calls result.to_dict())
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
