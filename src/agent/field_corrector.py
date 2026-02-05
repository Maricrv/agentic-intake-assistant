from __future__ import annotations
from typing import Optional
from .llm_client import LLMClient


class FieldCorrector:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def maybe_correct_location_with_confirmation(self, raw_value: str) -> Optional[str]:
        raw_value = (raw_value or "").strip()
        if not raw_value:
            return None

        resp = self.llm.suggest_location_correction(raw_value)
        if not resp:
            return None

        proposed = (resp.text or "").strip()
        if not proposed:
            return None

        def ask_yn() -> str:
            return input(f'I think you meant "{proposed}". Use that? (y/n)\n> ').strip().lower()

        ans = ask_yn()
        if ans == "":
            ans = ask_yn()

        if ans in {"y", "yes"}:
            return proposed
        return None
