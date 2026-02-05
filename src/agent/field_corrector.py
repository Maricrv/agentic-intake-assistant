from __future__ import annotations

from typing import Optional, Callable
from .llm_client import LLMClient


class FieldCorrector:
    """
    Field corrections with user confirmation (CLI-friendly).
    - Avoids returning None when user rejects correction (keeps original value).
    - Avoids asking when proposed == raw (case-insensitive).
    - ask() is injectable so this can be reused outside CLI later.
    """

    def __init__(self, llm: LLMClient, ask: Optional[Callable[[str], str]] = None):
        self.llm = llm
        self.ask = ask or (lambda prompt: input(prompt))

    def maybe_correct_location_with_confirmation(self, raw_value: str) -> Optional[str]:
        raw_value = (raw_value or "").strip()
        if not raw_value:
            return None

        resp = self.llm.suggest_location_correction(raw_value)

        # If no suggestion, keep what user entered (do not force re-asking later)
        if not resp or not (resp.text or "").strip():
            return raw_value

        proposed = resp.text.strip()

        # If it is effectively the same, keep raw
        if proposed.lower() == raw_value.lower():
            return raw_value

        prompt = f'I think you meant "{proposed}". Use that? (y/n)\n> '

        # Up to 2 attempts to get a clear yes/no
        for _ in range(2):
            ans = (self.ask(prompt) or "").strip().lower()
            if ans in {"y", "yes", "s", "si"}:
                return proposed
            if ans in {"n", "no"}:
                return raw_value

        # If user doesn't answer clearly, keep original
        return raw_value
