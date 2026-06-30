"""Progressive enforcement policy for strike escalation (audit TODO #28).

Pure logic, no I/O: given an accumulated strike total, decide the recommended
account-level consequence from a configurable, ordered threshold ladder. The
default ladder is ``warn -> timeout -> ban``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nemoguardian.review.models import EnforcementAction

# Ordered, ascending: each tuple is (min_total_strikes, action).
_DEFAULT_THRESHOLDS: tuple[tuple[float, EnforcementAction], ...] = (
    (1, EnforcementAction.WARN),
    (2, EnforcementAction.TIMEOUT),
    (3, EnforcementAction.BAN),
)


@dataclass
class StrikePolicy:
    """Maps an accumulated strike total to a recommended enforcement action.

    ``thresholds`` is a list of ``(min_total, action)`` pairs. The recommended
    action is the one for the highest threshold whose ``min_total`` is met by the
    user's current total; below the lowest threshold the recommendation is
    :attr:`EnforcementAction.NONE`.
    """

    thresholds: list[tuple[float, EnforcementAction]] = field(
        default_factory=lambda: list(_DEFAULT_THRESHOLDS)
    )

    def __post_init__(self) -> None:
        normalized: list[tuple[float, EnforcementAction]] = []
        for entry in self.thresholds:
            count, action = entry
            normalized.append((float(count), EnforcementAction(action)))
        if not normalized:
            raise ValueError("StrikePolicy requires at least one threshold")
        normalized.sort(key=lambda item: item[0])
        self.thresholds = normalized

    def recommend(self, total: float) -> EnforcementAction:
        """Return the recommended enforcement action for ``total`` strikes."""
        action = EnforcementAction.NONE
        for min_total, candidate in self.thresholds:
            if total >= min_total:
                action = candidate
            else:
                break
        return action

    def next_threshold(self, total: float) -> tuple[float, EnforcementAction] | None:
        """Return the next ``(min_total, action)`` the user has not yet reached."""
        for min_total, candidate in self.thresholds:
            if total < min_total:
                return (min_total, candidate)
        return None

    def to_list(self) -> list[dict[str, object]]:
        return [
            {"min_total": min_total, "action": action.value}
            for min_total, action in self.thresholds
        ]

    @classmethod
    def from_pairs(cls, pairs: list[tuple[float, str | EnforcementAction]]) -> StrikePolicy:
        return cls(thresholds=[(float(c), EnforcementAction(a)) for c, a in pairs])


def default_policy() -> StrikePolicy:
    """Return the default ``warn -> timeout -> ban`` ladder."""
    return StrikePolicy()


__all__ = ["StrikePolicy", "default_policy"]
