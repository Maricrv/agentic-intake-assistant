# Decision Logic  
## Generic Agentic Intake Assistant

---

## Purpose

This document defines the decision rules the agent uses to:
- classify intent
- decide what to ask next
- evaluate readiness
- stop gracefully when needed

---

## Intent Classification (S1)

### Supported intents
- `service_request` (continue intake)
- `general_question` (provide brief guidance + stop)
- `unknown` (ask clarification)

### Rule (v1)
- If user message contains a service keyword (e.g., "repair", "install", "help", "fix", "book", "quote") → `service_request`
- If user message is a greeting only or vague ("hi", "hello", "help") → `unknown`
- If user asks informational questions without requesting service ("what is...", "how does...", "price list") → `general_question`

---

## Constraint Evaluation (S3)

### Budget rule (demo)
- If budget is "<10" or "free only" → likely `not_a_fit` for most services

### Timeline rule (demo)
- If urgency is "urgent" AND timeline is not provided → request follow-up
- If user needs "within_24h" but also says "no availability" → inconsistency

---

## Readiness Evaluation (S4)

### Minimal required fields (demo)
- location
- budget_range

### Status output
- If required fields present → `ready`
- If missing required fields → `not_ready`
- If constraints incompatible → `not_a_fit`

---

## Stopping Conditions

The agent must stop when:
- user intent is `general_question` (after brief helpful response)
- constraints are incompatible (`not_a_fit`)
- user provides no input after clarification attempts (optional future)

---
