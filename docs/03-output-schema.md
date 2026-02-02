# Output Schema Definition  
## Generic Agentic Intake Assistant

---

## Purpose of This Document

This document defines the **standard output contract** produced by the agent.

The output must be:
- consistent
- system-agnostic
- easy to store (DB / Sheets)
- easy to integrate (API / CRM / ticketing systems)

---

## Output Principles

- The agent must always produce an output, even if the request is not ready.
- The output must separate:
  - what the user provided
  - what the agent inferred/decided
  - what is missing
- The output must support human-in-the-loop review.

---

## Output Object (High-Level)

The agent produces a single JSON-like object called `intake_result`.

### Top-Level Fields

- `request_id`
- `channel`
- `session`
- `request`
- `readiness`
- `handoff`
- `audit`

---

## JSON Schema (Version 1)

> This is a practical schema for implementation (not a formal JSON Schema spec).

```json
{
  "schema_version": "1.0",
  "request_id": "req_2026_000001",
  "channel": {
    "source": "telegram",
    "user_id": "7643595593",
    "username": "Mariela",
    "timestamp_utc": "2026-02-01T16:30:00Z"
  },
  "session": {
    "session_id": "sess_7643595593_20260201",
    "language": "en",
    "state": "S5"
  },
  "request": {
    "request_type": "service_request",
    "service_category": "generic_service",
    "summary": "User wants help with a service request",
    "details": {
      "urgency": "flexible",
      "timeline": "within_2_weeks",
      "location": "not_provided",
      "budget_range": "not_provided",
      "constraints": []
    }
  },
  "readiness": {
    "status": "not_ready",
    "missing_fields": ["location", "budget_range"],
    "inconsistencies": [],
    "notes": "Need location and budget range to proceed."
  },
  "handoff": {
    "recommended_action": "ask_follow_up",
    "next_questions": [
      "What is your location (city / country)?",
      "Do you have a budget range in mind?"
    ],
    "routing_hint": "human_review"
  },
  "audit": {
    "conversation_turns": 4,
    "tool_calls": [],
    "created_at_utc": "2026-02-01T16:30:05Z"
  }
}
