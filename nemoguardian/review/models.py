"""Dataclasses and enums for the human-in-the-loop review layer.

The review layer reuses the append-only JSONL + dataclass + PII-redaction
pattern from :mod:`nemoguardian.bot.audit`. Records are immutable: a status
change appends a *new* record for the same ``case_id`` and the current state is
the most recently appended record (latest-wins fold). This keeps a tamper-evident
trail of every moderator decision.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from nemoguardian.bot.types import ModerationAction, Platform
from nemoguardian.schemas import VerdictLabel


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


class CaseStatus(str, Enum):
    """Lifecycle of a queued review case (audit TODO #26)."""

    PENDING = "pending"      # awaiting human review
    APPROVED = "approved"    # moderator let the content stand (kept up)
    REMOVED = "removed"      # moderator confirmed the action / took the content down
    RESTORED = "restored"    # a prior removal reversed via the appeal flow (#27)


# Statuses that mean the case no longer needs a moderator's attention.
RESOLVED_STATUSES = frozenset({CaseStatus.APPROVED, CaseStatus.REMOVED, CaseStatus.RESTORED})


class AppealOutcome(str, Enum):
    """Result of an appeal against a prior moderation action (audit TODO #27)."""

    RESTORED = "restored"    # appeal granted -> content should be restored
    UPHELD = "upheld"        # appeal denied -> original action stands


class EnforcementAction(str, Enum):
    """Progressive enforcement ladder for strike escalation (audit TODO #28).

    Intentionally distinct from :class:`~nemoguardian.bot.types.ModerationAction`
    (which is per-message): these are per-user account consequences.
    """

    NONE = "none"
    WARN = "warn"
    TIMEOUT = "timeout"
    BAN = "ban"


@dataclass
class ReviewCase:
    """A single moderation case queued for human review.

    Resolving a case appends a new ``ReviewCase`` with the same ``case_id`` and an
    updated ``status``; :class:`ReviewQueue` folds to the latest record.
    """

    case_id: str
    platform: Platform
    workspace_id: str
    user_id: str
    username: str = ""
    channel_id: str = ""
    message_id: str = ""
    status: CaseStatus = CaseStatus.PENDING
    verdict: VerdictLabel = VerdictLabel.CONTROVERSIAL
    score: float = 0.0
    reason: str = ""
    categories: list[str] = field(default_factory=list)
    action: ModerationAction = ModerationAction.FLAG
    text_excerpt: str = ""
    text_sha256: str = ""
    source_case_id: str | None = None    # links back to the bot audit case
    resolved_by: str | None = None       # moderator id who resolved the case
    resolution_note: str = ""
    appeal_id: str | None = None         # set when an appeal touches this case
    created_at: str = field(default_factory=_utcnow)
    updated_at: str = field(default_factory=_utcnow)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["platform"] = self.platform.value
        data["status"] = self.status.value
        data["verdict"] = self.verdict.value
        data["action"] = self.action.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReviewCase:
        return cls(
            case_id=str(data.get("case_id", "")),
            platform=Platform(data.get("platform", Platform.WEBHOOK.value)),
            workspace_id=str(data.get("workspace_id", "")),
            user_id=str(data.get("user_id", "")),
            username=str(data.get("username", "")),
            channel_id=str(data.get("channel_id", "")),
            message_id=str(data.get("message_id", "")),
            status=CaseStatus(data.get("status", CaseStatus.PENDING.value)),
            verdict=VerdictLabel(data.get("verdict", VerdictLabel.CONTROVERSIAL.value)),
            score=float(data.get("score", 0.0) or 0.0),
            reason=str(data.get("reason", "")),
            categories=[str(c) for c in data.get("categories") or []],
            action=ModerationAction(data.get("action", ModerationAction.FLAG.value)),
            text_excerpt=str(data.get("text_excerpt", "")),
            text_sha256=str(data.get("text_sha256", "")),
            source_case_id=_opt_str(data.get("source_case_id")),
            resolved_by=_opt_str(data.get("resolved_by")),
            resolution_note=str(data.get("resolution_note", "")),
            appeal_id=_opt_str(data.get("appeal_id")),
            created_at=str(data.get("created_at") or _utcnow()),
            updated_at=str(data.get("updated_at") or _utcnow()),
            details=dict(data.get("details") or {}),
        )


@dataclass
class AppealDecision:
    """A recorded appeal/restore decision linked to a review case (TODO #27).

    The *actual* platform restore is adapter-specific; this records and audits the
    DECISION (who decided, why, and whether it was a false positive).
    """

    appeal_id: str
    case_id: str                          # the review case being appealed
    platform: Platform
    workspace_id: str
    user_id: str
    original_action: ModerationAction
    outcome: AppealOutcome
    false_positive: bool = False
    reason: str = ""
    decided_by: str | None = None
    source_case_id: str | None = None     # the original bot audit case, if any
    void_strikes: bool = False            # whether linked strikes were voided
    created_at: str = field(default_factory=_utcnow)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["platform"] = self.platform.value
        data["original_action"] = self.original_action.value
        data["outcome"] = self.outcome.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppealDecision:
        return cls(
            appeal_id=str(data.get("appeal_id", "")),
            case_id=str(data.get("case_id", "")),
            platform=Platform(data.get("platform", Platform.WEBHOOK.value)),
            workspace_id=str(data.get("workspace_id", "")),
            user_id=str(data.get("user_id", "")),
            original_action=ModerationAction(
                data.get("original_action", ModerationAction.FLAG.value)
            ),
            outcome=AppealOutcome(data.get("outcome", AppealOutcome.UPHELD.value)),
            false_positive=bool(data.get("false_positive", False)),
            reason=str(data.get("reason", "")),
            decided_by=_opt_str(data.get("decided_by")),
            source_case_id=_opt_str(data.get("source_case_id")),
            void_strikes=bool(data.get("void_strikes", False)),
            created_at=str(data.get("created_at") or _utcnow()),
            details=dict(data.get("details") or {}),
        )


@dataclass
class StrikeEvent:
    """A single strike-ledger event (audit TODO #28).

    ``kind == "strike"`` adds ``weight`` to a user's running total; ``kind == "void"``
    cancels the strike identified by ``voids`` (used by the appeal/restore flow so a
    false positive does not push a user up the enforcement ladder).
    """

    strike_id: str
    platform: Platform
    workspace_id: str
    user_id: str
    kind: str = "strike"                  # "strike" | "void"
    username: str = ""
    weight: float = 1.0
    reason: str = ""
    categories: list[str] = field(default_factory=list)
    case_id: str | None = None            # review/audit case that earned the strike
    voids: str | None = None              # strike_id this event cancels (kind="void")
    created_at: str = field(default_factory=_utcnow)
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["platform"] = self.platform.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StrikeEvent:
        return cls(
            strike_id=str(data.get("strike_id", "")),
            platform=Platform(data.get("platform", Platform.WEBHOOK.value)),
            workspace_id=str(data.get("workspace_id", "")),
            user_id=str(data.get("user_id", "")),
            kind=str(data.get("kind", "strike")),
            username=str(data.get("username", "")),
            weight=float(data.get("weight", 1.0) or 0.0),
            reason=str(data.get("reason", "")),
            categories=[str(c) for c in data.get("categories") or []],
            case_id=_opt_str(data.get("case_id")),
            voids=_opt_str(data.get("voids")),
            created_at=str(data.get("created_at") or _utcnow()),
            details=dict(data.get("details") or {}),
        )


def _opt_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text or None


__all__ = [
    "RESOLVED_STATUSES",
    "AppealDecision",
    "AppealOutcome",
    "CaseStatus",
    "EnforcementAction",
    "ReviewCase",
    "StrikeEvent",
]
