from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional, Any


@dataclass
class LLMResponse:
    text: str


class LLMClient:
    """
    OpenAI-backed LLM client (Responses API).

    Config (intent_config["llm"]):
      - enabled: bool
      - model: str
      - reasoning_effort: "low" | "medium" | "high"
    """

    def __init__(self, enabled: bool = False, model: str = "gpt-5", reasoning_effort: str = "low"):
        self.enabled = bool(enabled)
        self.model = model or "gpt-5"
        self.reasoning_effort = reasoning_effort or "low"

        # NOTE: Use Any to avoid hard dependency on OpenAI types.
        self._client: Optional[Any] = None

        if self.enabled:
            # Lazy import: avoids crashing when openai isn't installed but LLM is disabled.
            try:
                from openai import OpenAI  # type: ignore
            except ModuleNotFoundError as e:
                raise ModuleNotFoundError(
                    "Missing dependency 'openai'. Install it (e.g., pip install openai) "
                    "or disable LLM in configs."
                ) from e

            self._client = OpenAI()

    def _call_text(self, instructions: str, user_input: str) -> str:
        """
        Returns model output_text or "" on any error.
        Keeps the system resilient even if SDK/model settings are not available.
        """
        if not self._client:
            return ""

        try:
            resp = self._client.responses.create(
                model=self.model,
                reasoning={"effort": self.reasoning_effort},
                instructions=instructions,
                input=user_input,
            )
            return (resp.output_text or "").strip()
        except Exception:
            return ""

    # -----------------------------
    # Location correction (primary)
    # -----------------------------
    def suggest_correction_location(self, value: str) -> Optional[LLMResponse]:
        """
        Returns a single corrected location string or None.
        Conservative: returns None if not confident.

        Expected output format preference:
          - City
          - City, Country
        Prefer shortest unambiguous form.
        """
        if not self.enabled:
            return None

        value = (value or "").strip()
        if not value:
            return None

        instructions = (
            "You are helping clean a user's location field.\n"
            "Return a normalized location in ONE of these formats:\n"
            "  - City\n"
            "  - City, Country\n"
            "Prefer the shortest unambiguous form.\n\n"
            "Return ONLY a JSON object with keys:\n"
            "  - suggestion: string (corrected location)\n"
            "  - confidence: number from 0 to 1\n"
            "If you are not confident (confidence < 0.75) or you cannot improve it, return:\n"
            '{"suggestion":"NO_SUGGESTION","confidence":0}\n'
            "Do NOT add any extra text."
        )

        user_input = f'Raw location: "{value}"'
        text = self._call_text(instructions, user_input)
        if not text:
            return None

        try:
            data = json.loads(text)
        except Exception:
            return None

        suggestion = str(data.get("suggestion", "")).strip()
        try:
            confidence = float(data.get("confidence", 0) or 0)
        except Exception:
            confidence = 0.0

        if confidence < 0.75:
            return None
        if not suggestion or suggestion.upper() == "NO_SUGGESTION":
            return None
        if suggestion.lower() == value.lower():
            return None

        return LLMResponse(text=suggestion)

    # -----------------------------------------
    # Backward-compatible alias (your code uses this)
    # -----------------------------------------
    def suggest_location_correction(self, value: str) -> Optional[LLMResponse]:
        return self.suggest_correction_location(value)

    # -----------------------------------------
    # Service type correction (closed set)
    # -----------------------------------------
    def suggest_service_type_correction(self, value: str, allowed: list[str]) -> Optional[LLMResponse]:
        """
        Suggest a corrected service_type from a closed set (allowed).
        Returns None if not confident.
        """
        if not self.enabled:
            return None

        value = (value or "").strip()
        if not value:
            return None

        instructions = (
            "You are helping map a user's service type into a closed set.\n"
            "Choose exactly ONE from the allowed list.\n"
            "Return ONLY JSON with keys:\n"
            "  - suggestion: string (must be one of allowed)\n"
            "  - confidence: number 0..1\n"
            "If you are not confident (confidence < 0.75), return:\n"
            '{"suggestion":"NO_SUGGESTION","confidence":0}\n'
            "Do NOT add any extra text."
        )

        user_input = f"Allowed: {allowed}\nUser entered: \"{value}\""
        text = self._call_text(instructions, user_input)
        if not text:
            return None

        try:
            data = json.loads(text)
        except Exception:
            return None

        suggestion = str(data.get("suggestion", "")).strip()
        try:
            confidence = float(data.get("confidence", 0) or 0)
        except Exception:
            confidence = 0.0

        if confidence < 0.75:
            return None
        if not suggestion or suggestion.upper() == "NO_SUGGESTION":
            return None
        if suggestion not in allowed:
            return None

        return LLMResponse(text=suggestion)
