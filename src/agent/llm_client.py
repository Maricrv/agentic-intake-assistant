from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from openai import OpenAI


@dataclass
class LLMResponse:
    text: str


class LLMClient:
    """
    OpenAI-backed LLM client (Responses API).

    Env:
      - OPENAI_API_KEY required
    Config (intent_config["llm"]):
      - enabled: bool
      - model: str (default: gpt-5)
      - reasoning_effort: "low" | "medium" | "high" (default: low)
    """

    def __init__(self, enabled: bool = False, model: str = "gpt-5", reasoning_effort: str = "low"):
        self.enabled = bool(enabled)
        self.model = model or "gpt-5"
        self.reasoning_effort = reasoning_effort or "low"

        self._client: Optional[OpenAI] = None
        if self.enabled:
            # OpenAI SDK reads OPENAI_API_KEY from env
            self._client = OpenAI()

    def _call_text(self, instructions: str, user_input: str) -> str:
        if not self._client:
            return ""

        resp = self._client.responses.create(
            model=self.model,
            reasoning={"effort": self.reasoning_effort},
            instructions=instructions,
            input=user_input,
        )
        # SDK provides output_text convenience
        return (resp.output_text or "").strip()

    def suggest_correction_location(self, value: str) -> Optional[LLMResponse]:
        """
        Returns a *single* corrected location string (e.g., 'Toronto, Canada') or None.
        Must be conservative: return 'NO_SUGGESTION' if not confident.
        """
        if not self.enabled:
            return None

        value = (value or "").strip()
        if not value:
            return None

        instructions = (
            "You are helping clean a user's location field.\n"
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
        confidence = float(data.get("confidence", 0) or 0)

        if confidence < 0.75:
            return None
        if not suggestion or suggestion.upper() == "NO_SUGGESTION":
            return None

        # If it's identical (case-insensitive), no point.
        if suggestion.lower() == value.lower():
            return None

        return LLMResponse(text=suggestion)
