from __future__ import annotations

from typing import Optional
from .llm_client import LLMClient


class FieldCorrector:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def _is_yes(self, s: str) -> bool:
        t = (s or "").strip().lower()
        return t in {"y", "yes", "ye", "yep", "yeah", "sure", "ok", "okay"}

    def _is_no(self, s: str) -> bool:
        t = (s or "").strip().lower()
        return t in {"n", "no", "nope"}

    def maybe_correct_location_with_confirmation(self, raw_location: str) -> Optional[str]:
        raw = (raw_location or "").strip()
        if not raw:
            return None

        # If LLM is disabled, do nothing.
        if not getattr(self.llm, "enabled", False):
            return None

        suggestion = self.llm.suggest_location_correction(raw)
        if not suggestion:
            return None

        if suggestion.strip().lower() == raw.strip().lower():
            return None

        print(f'I think you meant "{suggestion}". Use that? (y/n)')
        ans = input("> ").strip()

        if self._is_yes(ans):
            return suggestion
        if self._is_no(ans):
            return None

        # If unclear answer, default to NOT changing
        return None
