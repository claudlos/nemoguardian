"""High-level orchestration for the human-in-the-loop review layer.

:class:`ReviewService` ties the three append-only stores together so the common
moderator workflows are one call each:

* enqueue a flagged case for review (#26)
* resolve a case, optionally adding a strike (#26 / #28)
* file an appeal that links to the original case and, when granted as a false
  positive, restores the case and voids the strikes it earned (#27 / #28)

The service performs no platform I/O — the *actual* restore (un-deleting a
message, lifting a timeout) is adapter-specific. It records and audits the
decision so enforcement stays trustworthy and reversible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nemoguardian.bot.types import ModerationAction, Platform
from nemoguardian.review.models import (
    AppealDecision,
    AppealOutcome,
    CaseStatus,
    ReviewCase,
    StrikeEvent,
)
from nemoguardian.review.store import (
    AppealLog,
    ReviewQueue,
    StrikeLedger,
    _new_id,
)
from nemoguardian.review.strikes import StrikePolicy


@dataclass
class AppealResult:
    """Outcome of an appeal: the decision plus the side effects it triggered."""

    decision: AppealDecision
    case: ReviewCase
    voided_strikes: list[StrikeEvent]


class ReviewService:
    """Coordinates the review queue, appeal log, and strike ledger."""

    def __init__(
        self,
        *,
        queue: ReviewQueue | None = None,
        appeals: AppealLog | None = None,
        strikes: StrikeLedger | None = None,
        policy: StrikePolicy | None = None,
    ) -> None:
        self.queue = queue if queue is not None else ReviewQueue()
        self.appeals = appeals if appeals is not None else AppealLog()
        self.strikes = strikes if strikes is not None else StrikeLedger()
        self.policy = policy

    @classmethod
    def from_dir(cls, directory: str | Path, *, policy: StrikePolicy | None = None) -> ReviewService:
        """Build a service whose three stores live under one directory."""
        base = Path(directory)
        return cls(
            queue=ReviewQueue(base / "review_queue.jsonl"),
            appeals=AppealLog(base / "appeals.jsonl"),
            strikes=StrikeLedger(base / "strikes.jsonl"),
            policy=policy,
        )

    def enqueue(self, **kwargs: Any) -> ReviewCase:
        """Enqueue a flagged case for human review (see :meth:`ReviewQueue.enqueue`)."""
        return self.queue.enqueue(**kwargs)

    def resolve(
        self,
        case_id: str,
        status: CaseStatus | str,
        *,
        resolved_by: str | None = None,
        note: str = "",
        add_strike: bool = False,
        strike_weight: float = 1.0,
    ) -> ReviewCase:
        """Resolve a queued case, optionally recording a strike for the user.

        A strike is only added when ``add_strike`` is set and the resolution is
        ``removed`` (the action was upheld), keeping strike accrual aligned with
        confirmed violations.
        """
        case = self.queue.resolve(case_id, status, resolved_by=resolved_by, note=note)
        if add_strike and case.status is CaseStatus.REMOVED:
            self.strikes.add_strike(
                platform=case.platform,
                workspace_id=case.workspace_id,
                user_id=case.user_id,
                username=case.username,
                weight=strike_weight,
                reason=note or case.reason,
                categories=case.categories,
                case_id=case.case_id,
            )
        return case

    def appeal(
        self,
        case_id: str,
        outcome: AppealOutcome | str,
        *,
        false_positive: bool = False,
        reason: str = "",
        decided_by: str | None = None,
        void_strikes: bool = True,
        original_action: ModerationAction | str | None = None,
    ) -> AppealResult:
        """File an appeal against a prior action and apply the restore flow.

        When ``outcome`` is :attr:`AppealOutcome.RESTORED` the linked review case is
        moved to ``restored`` and, if ``void_strikes`` is set, every strike the case
        earned is voided so a false positive does not push the user up the ladder.
        An ``upheld`` outcome records the denial without side effects.

        Raises ``KeyError`` if the review case does not exist.
        """
        outcome = AppealOutcome(outcome)
        case = self.queue.get_case(case_id)
        if case is None:
            raise KeyError(f"review case not found: {case_id}")
        action = (
            ModerationAction(original_action) if original_action is not None else case.action
        )
        appeal_id = _new_id("appeal")
        restored = outcome is AppealOutcome.RESTORED
        voided: list[StrikeEvent] = []
        if restored and void_strikes:
            voided = self.strikes.void_for_case(
                case.case_id, reason=reason or "appeal: false positive"
            )
        decision = AppealDecision(
            appeal_id=appeal_id,
            case_id=case.case_id,
            platform=case.platform,
            workspace_id=case.workspace_id,
            user_id=case.user_id,
            original_action=action,
            outcome=outcome,
            false_positive=bool(false_positive or restored),
            reason=str(reason),
            decided_by=decided_by,
            source_case_id=case.source_case_id,
            void_strikes=bool(voided),
            details={"voided_strike_ids": [event.strike_id for event in voided]},
        )
        self.appeals.record(decision)
        if restored:
            case = self.queue.resolve(
                case.case_id,
                CaseStatus.RESTORED,
                resolved_by=decided_by,
                note=reason,
                appeal_id=appeal_id,
                details={"appeal_outcome": outcome.value},
            )
        return AppealResult(decision=decision, case=case, voided_strikes=voided)

    def recommend(
        self, platform: Platform | str, workspace_id: str, user_id: str
    ) -> dict[str, Any]:
        """Recommend an enforcement action for a user from current strikes."""
        return self.strikes.recommend(
            platform, workspace_id, user_id, policy=self.policy
        )


__all__ = ["AppealResult", "ReviewService"]
