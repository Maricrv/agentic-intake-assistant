from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List


@dataclass
class ConsistencyResult:
    applied: bool
    kept_value: str


def keep_existing_on_conflict(
    field: str,
    current_value: str,
    new_value: str,
    inconsistencies: List[str],
    log: Callable[[str], None],
) -> ConsistencyResult:
    """
    Strategy: keep current (first/better) value, ignore the new conflicting one.
    """
    if current_value == "not_provided":
        return ConsistencyResult(applied=True, kept_value=new_value)

    if current_value != new_value:
        inconsistencies.append(f"{field}_conflict: kept '{current_value}', ignored '{new_value}'")
        log(f"inconsistency: {field} '{current_value}' vs '{new_value}'")
        return ConsistencyResult(applied=False, kept_value=current_value)

    return ConsistencyResult(applied=False, kept_value=current_value)
