from __future__ import annotations

from typing import Any, Callable, Dict

from .field_extractor import extract_prefill
from .field_corrector import FieldCorrector
from .llm_client import LLMClient
from .normalizers import norm_text, normalize_value, normalize_constraints, is_valid_service_type
from .consistency import keep_existing_on_conflict


class FieldHandlers:
    """
    Encapsulates:
      - prefill extraction + apply
      - apply_field logic (core fields + extra_fields)
    """

    def __init__(
        self,
        intent_config: Dict[str, Any],
        llm: LLMClient,
        corrector: FieldCorrector,
        result: Any,  # IntakeResult
        memory: Dict[str, Any],
        log: Callable[[str], None],
    ):
        self.intent_config = intent_config or {}
        self.llm = llm
        self.corrector = corrector
        self.result = result
        self.memory = memory
        self._log = log

    def extract_prefill_safe(self, text: str) -> Dict[str, Any]:
        """
        Supports old extract_prefill(text) and newer extract_prefill(text, intent_config).
        """
        try:
            data = extract_prefill(text, self.intent_config)  # type: ignore
        except TypeError:
            data = extract_prefill(text)  # type: ignore
        return data or {}

    def apply_prefill(self, field: str, value: str) -> None:
        value = norm_text(value)
        if not value:
            return

        d = self.result.request.details
        self.result.request.sources["prefill"] = True

        if field == "location":
            norm = normalize_value("text", value, self.intent_config)
            if norm != "not_provided":
                d.location = norm
                self._log(f"prefill: location='{norm}'")
            return

        if field == "timeline":
            norm = normalize_value("timeline", value, self.intent_config)
            if norm != "not_provided":
                d.timeline = norm
                self._log(f"prefill: timeline='{norm}'")
            return

        if field == "budget_range":
            norm = normalize_value("budget", value, self.intent_config)
            if norm != "not_provided":
                d.budget_range = norm
                self._log(f"prefill: budget_range='{norm}'")
            return

        if field == "urgency":
            norm = normalize_value("urgency", value, self.intent_config)
            if norm != "not_provided":
                d.urgency = norm
                self._log(f"prefill: urgency='{norm}'")
            return

        if field == "service_type":
            norm = normalize_value("service_type", value, self.intent_config)
            if norm != "not_provided":
                d.service_type = norm
                self._log(f"prefill: service_type='{norm}'")
            return

        # Anything else => extra_fields
        d.extra_fields[field] = normalize_value("text", value, self.intent_config)
        self._log(f"prefill: extra_fields['{field}']")

    def apply_field(self, intent: Dict[str, Any], field: str, raw: str, normalizer_kind: str, allowed: list[str]) -> None:
        """
        Apply user input to IntakeResult details.
        - Core fields go into details.<field>
        - Non-core fields go into details.extra_fields[field]
        - LLM confirmations for service_type and location (when enabled)
        """
        if not norm_text(raw):
            return

        d = self.result.request.details

        # constraints => list
        if field == "constraints":
            val = normalize_constraints(raw, self.intent_config)
            if val:
                d.constraints.append(val)
                self._log(f"user_set: constraints += '{val}'")
            return

        # issue_description
        if field == "issue_description":
            val = normalize_value("text", raw, self.intent_config)
            self.memory["collected"]["issue_description"] = val
            d.issue_description = val
            self._log("user_set: issue_description")
            return

        # location (LLM-assisted)
        if field == "location":
            corrected = self.corrector.maybe_correct_location_with_confirmation(raw)
            raw_to_use = corrected if corrected else raw

            if corrected and corrected != raw:
                if "location_correction" not in self.result.request.sources.get("llm_used", []):
                    self.result.request.sources["llm_used"].append("location_correction")
                self._log(f"llm_suggestion_accepted: location='{corrected}'")

            val = normalize_value("text", raw_to_use, self.intent_config)
            if val == "not_provided":
                return

            res = keep_existing_on_conflict(
                "location",
                d.location,
                val,
                self.result.readiness.inconsistencies,
                self._log,
            )
            if res.applied:
                d.location = val
                self._log(f"user_set: location='{val}'")
            return

        # service_type (allowed per intent + LLM mapping)
        if field == "service_type":
            val = normalize_value("service_type", raw, self.intent_config)

            if allowed:
                allowed_lc = {a.lower(): a for a in allowed}
                if val != "not_provided" and val.lower() not in allowed_lc:
                    resp = self.llm.suggest_service_type_correction(val, allowed)
                    if resp:
                        proposed = (resp.text or "").strip()
                        ans = input(f'I think you meant "{proposed}". Use that? (y/n)\n> ').strip().lower()
                        if ans in {"y", "yes"}:
                            val = proposed
                            if "service_type_correction" not in self.result.request.sources.get("llm_used", []):
                                self.result.request.sources["llm_used"].append("service_type_correction")
                            self._log(f"llm_suggestion_accepted: service_type='{val}'")
                        else:
                            self._log(f"llm_suggestion_rejected: service_type='{proposed}'")

                if val.lower() in allowed_lc:
                    val = allowed_lc[val.lower()]

            if val == "not_provided":
                return

            if not allowed and not is_valid_service_type(val):
                # multi-domain fallback: accept raw text
                val = norm_text(raw)

            res = keep_existing_on_conflict(
                "service_type",
                d.service_type,
                val,
                self.result.readiness.inconsistencies,
                self._log,
            )
            if res.applied:
                self.memory["collected"]["service_type"] = val
                d.service_type = val
                label = str(intent.get("label") or "Service request")
                self.result.request.summary = f"{label}: {val}"
                self._log(f"user_set: service_type='{val}'")
            return

        # If normalizer is urgency/timeline/budget => store appropriately
        if normalizer_kind == "urgency":
            val = normalize_value("urgency", raw, self.intent_config)
            if val == "not_provided":
                return
            if field == "urgency":
                d.urgency = val
                self._log(f"user_set: urgency='{val}'")
            else:
                d.extra_fields[field] = val
                self._log(f"user_set: extra_fields['{field}']='{val}'")
            return

        if normalizer_kind == "timeline":
            val = normalize_value("timeline", raw, self.intent_config)
            if val == "not_provided":
                return
            if field == "timeline":
                d.timeline = val
                self._log(f"user_set: timeline='{val}'")
            else:
                d.extra_fields[field] = val
                self._log(f"user_set: extra_fields['{field}']='{val}'")
            return

        if normalizer_kind == "budget":
            val = normalize_value("budget", raw, self.intent_config)
            if val == "not_provided":
                return
            if field == "budget_range":
                d.budget_range = val
                self._log(f"user_set: budget_range='{val}'")
            else:
                d.extra_fields[field] = val
                self._log(f"user_set: extra_fields['{field}']='{val}'")
            return

        # default => extra_fields
        val = normalize_value(normalizer_kind, raw, self.intent_config)
        self.memory["collected"][field] = val
        d.extra_fields[field] = val
        self._log(f"user_set: extra_fields['{field}']='{val}'")
