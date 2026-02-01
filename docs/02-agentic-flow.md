# Agentic Flow Definition  
## Generic Agentic Intake Assistant

---

## Purpose of This Document

This document defines the **agentic behavior** of the system by describing:

- agent states
- decision points
- transitions between states
- stopping conditions

The goal is to ensure the agent behaves consistently, predictably, and explainably.

---

## High-Level Agent Flow

The agent operates as a **state-based conversational system**.

Each state:
- has a clear purpose
- determines what information to request
- decides whether to advance, clarify, or stop

---

## Agent States Overview

| State ID | State Name | Description |
|--------|-----------|------------|
| S0 | Initialization | Establishes context and expectations |
| S1 | Intent Clarification | Determines if the request is valid and relevant |
| S2 | Context Collection | Gathers essential contextual information |
| S3 | Constraint Evaluation | Evaluates feasibility and constraints |
| S4 | Readiness Assessment | Determines if the request is ready |
| S5 | Structured Summary & Handoff | Produces final structured output |

---

## State S0 – Initialization

### Objective
Set expectations and define the interaction scope.

### Agent Behavior
- Greet the user
- Explain that the agent will ask a few questions
- Clarify that the goal is to prepare a request

### Example Output
> “I can help you prepare a service request. I’ll ask a few quick questions to understand your needs.”

### Transition
- Always transitions to **S1**

---

## State S1 – Intent Clarification

### Objective
Determine whether the user has a valid service request.

### Information Collected
- Type of request
- Whether the user is seeking a service or general information

### Example Questions
- “What type of service are you looking for?”
- “Is this a request for support, consultation, or something else?”

### Decision Logic
- Valid service request → **S2**
- Unclear intent → ask clarification (stay in S1)
- Invalid / irrelevant intent → stop politely

---

## State S2 – Context Collection

### Objective
Gather high-level context required for processing.

### Information Collected
- urgency (urgent / flexible)
- timeline
- general scope (small / medium / large)

### Example Questions
- “Is this request urgent?”
- “When would you like this to be addressed?”

### Decision Logic
- Required context collected → **S3**
- Missing key context → ask follow-up (stay in S2)

---

## State S3 – Constraint Evaluation

### Objective
Evaluate constraints that affect feasibility.

### Information Collected
- availability
- budget range (if applicable)
- operational limitations

### Example Questions
- “Do you have a budget range in mind?”
- “Are there any constraints we should be aware of?”

### Decision Logic
- Constraints acceptable → **S4**
- Constraints incompatible → stop with explanation
- Constraints unclear → ask clarification (stay in S3)

---

## State S4 – Readiness Assessment

### Objective
Determine whether enough information has been collected.

### Internal Evaluation
The agent checks:
- required fields completeness
- consistency of answers
- feasibility based on constraints

### Decision Logic
- Request is ready → **S5**
- Request not ready → targeted follow-up (return to S2 or S3)

---

## State S5 – Structured Summary & Handoff

### Objective
Produce a structured, system-ready summary.

### Output Structure (High Level)
- request type
- urgency
- timeline
- constraints
- readiness status
- missing information (if any)

### Example Output
> “Your request has been prepared and is ready for review.”

### End State
- Conversation ends
- Output is stored or forwarded to downstream systems

---

## Stopping Conditions

The agent must stop when:
- the request is completed
- the request is invalid
- constraints make fulfillment impossible
- the user chooses not to continue

---

## Design Principles

- Human-in-the-loop is always preserved
- No autonomous final decisions
- Explainability over optimization
- Configurable for different domains

---

## Notes

This agentic flow is intentionally **domain-agnostic** and can be adapted to:
- Small businesses
- Service Desk platforms
- CRM and ERP systems
