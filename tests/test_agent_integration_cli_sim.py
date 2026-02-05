import builtins
from src.agent.agent import GenericIntakeAgent


def _run_with_inputs(monkeypatch, answers):
    """
    Patch builtins.input so the agent consumes predefined answers.
    If answers run out, return "" to avoid StopIteration (safe for followups).
    """
    it = iter(answers)

    def fake_input(prompt=""):
        try:
            return next(it)
        except StopIteration:
            return ""

    monkeypatch.setattr(builtins, "input", fake_input)


def test_agent_simulated_inputs_happy_path(monkeypatch):
    # LLM disabled for deterministic tests
    intent_config = {
        "llm": {"enabled": False, "model": "gpt-5", "reasoning_effort": "low"},
        "defaults": {"service_category": "technical_services"},
        "normalizers": {
            "urgency": {
                "urgent": ["urgent", "asap", "immediately"],
                "flexible": ["flexible", "not urgent"],
            },
            "service_type": {
                "repair": ["repair", "fix"],
                "installation": ["install", "installation"],
                "maintenance": ["maintenance", "maintain"],
                "consultation": ["consultation", "consult"],
            },
            "constraints_ignore": ["no", "none", "n/a", "na", "nope"],
        },
        "intents": [
            {
                "id": "pre_quote_request",
                "priority": 10,
                "service_category": "technical_services",
                "match": {"always": True},
                "flow": [
                    {
                        "field": "issue_description",
                        "required": True,
                        "question": "Tell me briefly what you need help with so we can prepare a pre-quotation.",
                        "normalize": "text",
                    },
                    {
                        "field": "service_type",
                        "required": True,
                        "question": "What type of service is this? (repair / installation / maintenance / consultation)",
                        "normalize": "service_type",
                    },
                    {
                        "field": "urgency",
                        "required": False,
                        "question": "Is this urgent or flexible? (urgent/flexible)",
                        "normalize": "urgency",
                    },
                    {
                        "field": "location",
                        "required": True,
                        "question": "What is your location (city/country)?",
                        "normalize": "text",
                    },
                    {
                        "field": "budget_range",
                        "required": False,
                        "question": "Do you have an estimated budget? (example: <50, 50-100, 100-300, 300-500, 500-1000, not sure)",
                        "normalize": "budget",
                    },
                    {
                        "field": "constraints",
                        "required": False,
                        "question": "Any constraints or notes? (building access hours, parking, safety rules, etc.)",
                        "normalize": "text",
                    },
                ],
                "handoff": {"recommended_action": "route_human", "routing_hint": "human_review"},
            }
        ],
    }

    # IMPORTANT: first message WITHOUT the keyword "repair"
    # so the agent does NOT infer service_type and it WILL ask for it.
    answers = [
        "Need help with a laptop in Toronto next week, budget $800.",  # issue_description
        "repair",                                                     # service_type
        "urgent",                                                     # urgency
        "Toronto",                                                    # location
        "800",                                                        # budget_range
        "no",                                                         # constraints
    ]
    _run_with_inputs(monkeypatch, answers)

    agent = GenericIntakeAgent(session_id="sess_test_001", intent_config=intent_config)
    result = agent.run()
    d = result.request.details

    assert result.request.intent_id == "pre_quote_request"
    assert d.service_type == "repair"
    assert d.urgency == "urgent"
    assert d.location == "Toronto"
    assert d.budget_range == "500-1000"
    assert d.constraints == []  # "no" is ignored
    assert result.readiness.status == "ready"


def test_agent_missing_location_triggers_followup(monkeypatch):
    intent_config = {
        "llm": {"enabled": False},
        "intents": [
            {
                "id": "pre_quote_request",
                "priority": 10,
                "match": {"always": True},
                "flow": [
                    {"field": "issue_description", "required": True, "question": "Describe.", "normalize": "text"},
                    {"field": "location", "required": True, "question": "What is your location (city/country)?", "normalize": "text"},
                ],
            }
        ],
    }

    # We answer issue_description, then leave location empty.
    # Because followups may repeat, our fake_input returns "" after answers run out.
    answers = [
        "Need repair next week.",  # issue_description
        "",                        # location (empty)
        "",                        # location followup (empty)
        "",                        # extra empty just in case
    ]
    _run_with_inputs(monkeypatch, answers)

    agent = GenericIntakeAgent(session_id="sess_test_002", intent_config=intent_config)
    result = agent.run()

    assert result.readiness.status == "not_ready"
    assert "location" in result.readiness.missing_fields
    assert result.handoff.recommended_action == "ask_follow_up"
