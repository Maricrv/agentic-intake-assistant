import pytest

from src.agent.normalizers import normalize_value, normalize_constraints, is_valid_service_type


CONFIG = {
    "normalizers": {
        "urgency": {
            "urgent": ["urgent", "asap", "immediately"],
            "flexible": ["flexible", "not urgent"],
        },
        "timeline": {
            "within_24h": ["within_24h"],
            "within_1_week": ["within_1_week"],
            "within_2_weeks": ["within_2_weeks"],
        },
        "constraints_ignore": ["none", "n/a", "na", "no", "nope"],
        # service_type synonyms (optional, used only if you normalize service_type by table)
        "service_type": {
            "repair": ["repair", "fix", "repar", "reparar"],
            "installation": ["install", "installation"],
            "maintenance": ["maintenance", "maintain"],
            "consultation": ["consultation", "consult"],
        },
    }
}


def test_budget_normalization():
    assert normalize_value("budget", "$800", CONFIG) == "500-1000"
    assert normalize_value("budget", "i told you 800", CONFIG) == "500-1000"
    assert normalize_value("budget", "40", CONFIG) == "<50"
    assert normalize_value("budget", "not sure", CONFIG) == "not_provided"


def test_timeline_normalization():
    assert normalize_value("timeline", "next week", CONFIG) == "within_1_week"
    assert normalize_value("timeline", "6 days", CONFIG) == "within_1_week"
    assert normalize_value("timeline", "14 days", CONFIG) == "within_2_weeks"
    assert normalize_value("timeline", "tomorrow", CONFIG) == "within_24h"


def test_constraints_normalization():
    assert normalize_constraints("no", CONFIG) == ""
    assert normalize_constraints("N/A", CONFIG) == ""
    assert normalize_constraints("open until 6pm", CONFIG) == "open until 6pm"


def test_service_type_validity_basic():
    assert is_valid_service_type("repair") is True
    assert is_valid_service_type("installation") is True
    assert is_valid_service_type("urgent") is False
    assert is_valid_service_type("what is the price?") is False


def test_service_type_typo_without_llm():
    # Without LLM correction, typos might be accepted as "valid" by is_valid_service_type.
    # If you want to enforce closed-set validation, you should validate against an allowed list in agent.py.
    # This test documents the current behavior and expectations:
    assert is_valid_service_type("rapair") is True  # typo still passes generic validator
