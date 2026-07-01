"""Append-only JSONL stores for the human-in-the-loop review layer.

Three small file-backed stores, each reusing the append-only JSONL pattern from
:mod:`nemoguardian.bot.audit` (newline-delimited JSON, thread-locked writes,
latest-record-wins folds):

* :class:`ReviewQueue`  — moderator review queue (audit TODO #26)
* :class:`AppealLog`    — appeal / restore decisions (audit TODO #27)
* :class:`StrikeLedger` — strike tracking + progressive enforcement (audit TODO #28)

Records are never mutated in place: a status change appends a new record. This
keeps a tamper-evident trail and lets a self-hosted moderator start without a
database. The stores intentionally share no global state, so tests can point each
at its own ``tmp_path``.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any

from nemoguardian.bot.audit import redacted_excerpt, text_hash
from nemoguardian.bot.types import ModerationAction, Platform
from nemoguardian.review.models import (
    RESOLVED_STATUSES,
    AppealDecision,
    AppealOutcome,
    CaseStatus,
    EnforcementAction,
    ReviewCase,
    StrikeEvent,
)
from nemoguardian.review.strikes import StrikePolicy, default_policy

DEFAULT_REVIEW_PATH = Path(
    os.environ.get("NEMOGUARDIAN_REVIEW_PATH", "/tmp/nemoguardian_review_queue.jsonl")
)
DEFAULT_APPEAL_PATH = Path(
    os.environ.get("NEMOGUARDIAN_APPEAL_PATH", "/tmp/nemoguardian_review_appeals.jsonl")
)
DEFAULT_STRIKE_PATH = Path(
    os.environ.get("NEMOGUARDIAN_STRIKE_PATH", "/tmp/nemoguardian_review_strikes.jsonl")
)


class _JsonlStore:
    """Shared thread-locked append-only JSONL backing store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def _append(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(payload, sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        records: list[dict[str, Any]] = []
        for line in lines:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class ReviewQueue(_JsonlStore):
    """Moderator review queue (audit TODO #26).

    Controversial / flagged cases are enqueued as ``pending``; a moderator later
    resolves each to ``approved`` (content kept), ``removed`` (action confirmed),
    or ``restored`` (a prior removal reversed). Current state is the most recently
    appended record per ``case_id``.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        super().__init__(path if path is not None else DEFAULT_REVIEW_PATH)

    def enqueue(
        self,
        *,
        platform: Platform | str,
        workspace_id: str,
        user_id: str,
        text: str | None = None,
        case_id: str | None = None,
        username: str = "",
        channel_id: str = "",
        message_id: str = "",
        verdict: Any = None,
        score: float = 0.0,
        reason: str = "",
        categories: list[str] | None = None,
        action: ModerationAction | str = ModerationAction.FLAG,
        source_case_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> ReviewCase:
        """Enqueue a case for human review and return the stored ``ReviewCase``.

        If ``text`` is given it is PII-redacted into an excerpt and hashed; the raw
        text is never persisted. Re-enqueuing an already-resolved ``case_id`` is a
        no-op that returns the existing record (idempotent intake).
        """
        from nemoguardian.schemas import VerdictLabel

        case_id = case_id or _new_id("rev")
        existing = self.get_case(case_id)
        if existing is not None:
            return existing
        excerpt = redacted_excerpt(text) if text else ""
        sha = text_hash(text) if text else ""
        case = ReviewCase(
            case_id=case_id,
            platform=Platform(platform),
            workspace_id=str(workspace_id),
            user_id=str(user_id),
            username=str(username),
            channel_id=str(channel_id),
            message_id=str(message_id),
            status=CaseStatus.PENDING,
            verdict=VerdictLabel(verdict) if verdict is not None else VerdictLabel.CONTROVERSIAL,
            score=float(score),
            reason=str(reason),
            categories=[str(c) for c in categories or []],
            action=ModerationAction(action),
            text_excerpt=excerpt,
            text_sha256=sha,
            source_case_id=source_case_id,
            details=dict(details or {}),
        )
        self._append(case.to_dict())
        return case

    def resolve(
        self,
        case_id: str,
        status: CaseStatus | str,
        *,
        resolved_by: str | None = None,
        note: str = "",
        appeal_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> ReviewCase:
        """Resolve a pending case by appending an updated record.

        Raises ``KeyError`` if the case does not exist and ``ValueError`` for an
        invalid target status (``pending`` is not a valid resolution).
        """
        status = CaseStatus(status)
        if status not in RESOLVED_STATUSES:
            raise ValueError(f"cannot resolve to non-terminal status: {status.value}")
        current = self.get_case(case_id)
        if current is None:
            raise KeyError(f"review case not found: {case_id}")
        updated = ReviewCase(
            case_id=current.case_id,
            platform=current.platform,
            workspace_id=current.workspace_id,
            user_id=current.user_id,
            username=current.username,
            channel_id=current.channel_id,
            message_id=current.message_id,
            status=status,
            verdict=current.verdict,
            score=current.score,
            reason=current.reason,
            categories=list(current.categories),
            action=current.action,
            text_excerpt=current.text_excerpt,
            text_sha256=current.text_sha256,
            source_case_id=current.source_case_id,
            resolved_by=resolved_by,
            resolution_note=str(note),
            appeal_id=appeal_id if appeal_id is not None else current.appeal_id,
            created_at=current.created_at,
            updated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            details={**current.details, **(details or {})},
        )
        self._append(updated.to_dict())
        return updated

    def _folded(self) -> dict[str, ReviewCase]:
        """Fold the append-only log into the latest record per ``case_id``."""
        latest: dict[str, ReviewCase] = {}
        for raw in self._read():
            case = ReviewCase.from_dict(raw)
            if case.case_id:
                latest[case.case_id] = case
        return latest

    def get_case(self, case_id: str) -> ReviewCase | None:
        return self._folded().get(case_id)

    def list_cases(
        self,
        *,
        platform: Platform | str | None = None,
        workspace_id: str | None = None,
        user_id: str | None = None,
        channel_id: str | None = None,
        status: CaseStatus | str | None = None,
        category: str | None = None,
        limit: int = 50,
    ) -> list[ReviewCase]:
        """List cases (newest-first) matching the given filters."""
        if limit <= 0:
            return []
        platform_value = Platform(platform).value if platform is not None else None
        status_value = CaseStatus(status).value if status is not None else None
        workspace = str(workspace_id) if workspace_id is not None else None
        user = str(user_id) if user_id is not None else None
        channel = str(channel_id) if channel_id is not None else None
        cases = sorted(
            self._folded().values(), key=lambda c: (c.updated_at, c.case_id), reverse=True
        )
        out: list[ReviewCase] = []
        for case in cases:
            if platform_value is not None and case.platform.value != platform_value:
                continue
            if workspace is not None and case.workspace_id != workspace:
                continue
            if user is not None and case.user_id != user:
                continue
            if channel is not None and case.channel_id != channel:
                continue
            if status_value is not None and case.status.value != status_value:
                continue
            if category is not None and str(category) not in case.categories:
                continue
            out.append(case)
            if len(out) >= limit:
                break
        return out

    def list_pending(
        self,
        *,
        platform: Platform | str | None = None,
        workspace_id: str | None = None,
        user_id: str | None = None,
        channel_id: str | None = None,
        category: str | None = None,
        limit: int = 50,
    ) -> list[ReviewCase]:
        """List cases still awaiting moderator review (status ``pending``)."""
        return self.list_cases(
            platform=platform,
            workspace_id=workspace_id,
            user_id=user_id,
            channel_id=channel_id,
            status=CaseStatus.PENDING,
            category=category,
            limit=limit,
        )

    def counts(
        self,
        *,
        platform: Platform | str | None = None,
        workspace_id: str | None = None,
    ) -> dict[str, int]:
        """Return a status -> count summary for the (optionally filtered) queue."""
        summary = {status.value: 0 for status in CaseStatus}
        for case in self.list_cases(
            platform=platform, workspace_id=workspace_id, limit=1_000_000
        ):
            summary[case.status.value] += 1
        return summary


class AppealLog(_JsonlStore):
    """Appeal / restore decision log (audit TODO #27)."""

    def __init__(self, path: str | Path | None = None) -> None:
        super().__init__(path if path is not None else DEFAULT_APPEAL_PATH)

    def record(self, decision: AppealDecision) -> AppealDecision:
        self._append(decision.to_dict())
        return decision

    def for_case(self, case_id: str) -> list[AppealDecision]:
        """Return all appeal decisions linked to ``case_id`` (newest-first)."""
        decisions = [
            AppealDecision.from_dict(raw)
            for raw in self._read()
            if str(raw.get("case_id")) == str(case_id)
        ]
        decisions.sort(key=lambda d: (d.created_at, d.appeal_id), reverse=True)
        return decisions

    def find(self, appeal_id: str) -> AppealDecision | None:
        for raw in reversed(self._read()):
            if str(raw.get("appeal_id")) == str(appeal_id):
                return AppealDecision.from_dict(raw)
        return None

    def list_appeals(
        self,
        *,
        platform: Platform | str | None = None,
        workspace_id: str | None = None,
        user_id: str | None = None,
        outcome: AppealOutcome | str | None = None,
        limit: int = 50,
    ) -> list[AppealDecision]:
        if limit <= 0:
            return []
        platform_value = Platform(platform).value if platform is not None else None
        outcome_value = AppealOutcome(outcome).value if outcome is not None else None
        workspace = str(workspace_id) if workspace_id is not None else None
        user = str(user_id) if user_id is not None else None
        decisions = [AppealDecision.from_dict(raw) for raw in self._read()]
        decisions.sort(key=lambda d: (d.created_at, d.appeal_id), reverse=True)
        out: list[AppealDecision] = []
        for decision in decisions:
            if platform_value is not None and decision.platform.value != platform_value:
                continue
            if workspace is not None and decision.workspace_id != workspace:
                continue
            if user is not None and decision.user_id != user:
                continue
            if outcome_value is not None and decision.outcome.value != outcome_value:
                continue
            out.append(decision)
            if len(out) >= limit:
                break
        return out


class StrikeLedger(_JsonlStore):
    """Strike tracking + progressive enforcement (audit TODO #28).

    Strikes accumulate per ``(platform, workspace_id, user_id)``. A ``void`` event
    cancels an earlier strike (used by the appeal/restore flow so a false positive
    does not escalate a user). :meth:`recommend` maps the active total to an
    enforcement action via a :class:`StrikePolicy`.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        super().__init__(path if path is not None else DEFAULT_STRIKE_PATH)

    def add_strike(
        self,
        *,
        platform: Platform | str,
        workspace_id: str,
        user_id: str,
        weight: float = 1.0,
        username: str = "",
        reason: str = "",
        categories: list[str] | None = None,
        case_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> StrikeEvent:
        event = StrikeEvent(
            strike_id=_new_id("strike"),
            platform=Platform(platform),
            workspace_id=str(workspace_id),
            user_id=str(user_id),
            kind="strike",
            username=str(username),
            weight=float(weight),
            reason=str(reason),
            categories=[str(c) for c in categories or []],
            case_id=case_id,
            details=dict(details or {}),
        )
        self._append(event.to_dict())
        return event

    def void_strike(
        self, strike_id: str, *, reason: str = "", details: dict[str, Any] | None = None
    ) -> StrikeEvent:
        """Cancel a previously recorded strike. Raises ``KeyError`` if unknown."""
        target = self._find_strike(strike_id)
        if target is None:
            raise KeyError(f"strike not found: {strike_id}")
        event = StrikeEvent(
            strike_id=_new_id("void"),
            platform=target.platform,
            workspace_id=target.workspace_id,
            user_id=target.user_id,
            kind="void",
            username=target.username,
            weight=target.weight,
            reason=str(reason),
            case_id=target.case_id,
            voids=target.strike_id,
            details=dict(details or {}),
        )
        self._append(event.to_dict())
        return event

    def void_for_case(
        self, case_id: str, *, reason: str = "", details: dict[str, Any] | None = None
    ) -> list[StrikeEvent]:
        """Void every active strike linked to ``case_id`` (used by restore flow)."""
        voided: list[StrikeEvent] = []
        for strike in self._active_strikes():
            if strike.case_id is not None and str(strike.case_id) == str(case_id):
                voided.append(self.void_strike(strike.strike_id, reason=reason, details=details))
        return voided

    def _events(self) -> list[StrikeEvent]:
        return [StrikeEvent.from_dict(raw) for raw in self._read()]

    def _find_strike(self, strike_id: str) -> StrikeEvent | None:
        for event in self._events():
            if event.kind == "strike" and event.strike_id == strike_id:
                return event
        return None

    def _active_strikes(
        self,
        *,
        platform: Platform | str | None = None,
        workspace_id: str | None = None,
        user_id: str | None = None,
        since: dt.datetime | None = None,
    ) -> list[StrikeEvent]:
        events = self._events()
        voided_ids = {e.voids for e in events if e.kind == "void" and e.voids}
        platform_value = Platform(platform).value if platform is not None else None
        workspace = str(workspace_id) if workspace_id is not None else None
        user = str(user_id) if user_id is not None else None
        out: list[StrikeEvent] = []
        for event in events:
            if event.kind != "strike":
                continue
            if event.strike_id in voided_ids:
                continue
            if platform_value is not None and event.platform.value != platform_value:
                continue
            if workspace is not None and event.workspace_id != workspace:
                continue
            if user is not None and event.user_id != user:
                continue
            if since is not None and not _at_or_after(event.created_at, since):
                continue
            out.append(event)
        return out

    def active_strikes(
        self,
        platform: Platform | str,
        workspace_id: str,
        user_id: str,
        *,
        since: dt.datetime | None = None,
    ) -> list[StrikeEvent]:
        """Return the user's non-voided strikes (oldest-first)."""
        strikes = self._active_strikes(
            platform=platform, workspace_id=workspace_id, user_id=user_id, since=since
        )
        strikes.sort(key=lambda e: (e.created_at, e.strike_id))
        return strikes

    def total(
        self,
        platform: Platform | str,
        workspace_id: str,
        user_id: str,
        *,
        since: dt.datetime | None = None,
    ) -> float:
        """Return the user's accumulated active strike weight."""
        return sum(
            event.weight
            for event in self._active_strikes(
                platform=platform, workspace_id=workspace_id, user_id=user_id, since=since
            )
        )

    def recommend(
        self,
        platform: Platform | str,
        workspace_id: str,
        user_id: str,
        *,
        policy: StrikePolicy | None = None,
        since: dt.datetime | None = None,
    ) -> dict[str, Any]:
        """Compute the recommended enforcement action for a user.

        Returns a dict with the active ``total``, the recommended ``action`` value,
        and the ``next_threshold`` the user has not yet crossed (or ``None``).
        """
        policy = policy or default_policy()
        total = self.total(platform, workspace_id, user_id, since=since)
        action: EnforcementAction = policy.recommend(total)
        nxt = policy.next_threshold(total)
        return {
            "platform": Platform(platform).value,
            "workspace_id": str(workspace_id),
            "user_id": str(user_id),
            "total": total,
            "action": action.value,
            "next_threshold": (
                {"min_total": nxt[0], "action": nxt[1].value} if nxt is not None else None
            ),
        }

    def leaderboard(
        self,
        platform: Platform | str,
        workspace_id: str,
        *,
        policy: StrikePolicy | None = None,
        limit: int = 10,
        since: dt.datetime | None = None,
    ) -> list[dict[str, Any]]:
        """Return users ranked by active strike weight (worst first)."""
        if limit <= 0:
            return []
        policy = policy or default_policy()
        by_user: dict[str, dict[str, Any]] = {}
        for event in self._active_strikes(
            platform=platform, workspace_id=workspace_id, since=since
        ):
            entry = by_user.setdefault(
                event.user_id,
                {"user_id": event.user_id, "username": event.username, "total": 0.0, "strikes": 0},
            )
            entry["total"] += event.weight
            entry["strikes"] += 1
            if event.username:
                entry["username"] = event.username
        rows = list(by_user.values())
        for entry in rows:
            entry["action"] = policy.recommend(entry["total"]).value
        rows.sort(key=lambda e: (-e["total"], e["user_id"]))
        return rows[:limit]


def _at_or_after(created_at: str, since: dt.datetime) -> bool:
    try:
        parsed = dt.datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return False
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    if since.tzinfo is None:
        since = since.replace(tzinfo=dt.timezone.utc)
    return parsed >= since


__all__ = [
    "DEFAULT_APPEAL_PATH",
    "DEFAULT_REVIEW_PATH",
    "DEFAULT_STRIKE_PATH",
    "AppealLog",
    "ReviewQueue",
    "StrikeLedger",
]
