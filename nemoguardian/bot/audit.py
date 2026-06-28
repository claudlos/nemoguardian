"""Append-only moderation audit records."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import threading
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nemoguardian.bot.types import ModerationAction, Platform
from nemoguardian.schemas import Mode, VerdictLabel

DEFAULT_AUDIT_PATH = Path(
    os.environ.get("NEMOGUARDIAN_BOT_AUDIT_PATH", "/tmp/nemoguardian_bot_audit.jsonl")
)

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)")
_PAYMENT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")


@dataclass
class AuditRecord:
    case_id: str
    platform: Platform
    workspace_id: str
    channel_id: str
    message_id: str
    user_id: str
    username: str
    action: ModerationAction
    verdict: VerdictLabel
    score: float
    mode: Mode
    categories: list[str] = field(default_factory=list)
    matched_policy_rule: str | None = None
    request_id: str | None = None
    latency_ms: float | None = None
    text_sha256: str = ""
    text_excerpt: str = ""
    dry_run: bool = False
    execution_status: str = "planned"
    error: str | None = None
    created_at: str = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["platform"] = self.platform.value
        data["action"] = self.action.value
        data["verdict"] = self.verdict.value
        data["mode"] = self.mode.value
        return data


class AuditLog:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_AUDIT_PATH
        self._lock = threading.Lock()

    def append(self, record: AuditRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record.to_dict(), sort_keys=True)
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        return self._read_records()[-limit:]

    def find_case(self, case_id: str) -> dict[str, Any] | None:
        for record in reversed(self._read_records()):
            if record.get("case_id") == case_id:
                return record
        return None

    def history(
        self,
        platform: Platform | str,
        workspace_id: str,
        *,
        user_id: str | None = None,
        channel_id: str | None = None,
        category: str | None = None,
        limit: int = 10,
        since: dt.datetime | None = None,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        platform_value = Platform(platform).value
        workspace_value = str(workspace_id)
        user_value = str(user_id) if user_id is not None else None
        channel_value = str(channel_id) if channel_id is not None else None
        category_value = str(category) if category is not None else None
        matches = []
        for record in reversed(self._read_records()):
            if record.get("platform") != platform_value:
                continue
            if str(record.get("workspace_id")) != workspace_value:
                continue
            if user_value is not None and str(record.get("user_id")) != user_value:
                continue
            if channel_value is not None and str(record.get("channel_id")) != channel_value:
                continue
            if category_value is not None and not _record_has_category(record, category_value):
                continue
            if since is not None and not _record_at_or_after(record, since):
                continue
            matches.append(record)
            if len(matches) >= limit:
                break
        return matches

    def failures(
        self,
        platform: Platform | str,
        workspace_id: str,
        *,
        limit: int = 10,
        since: dt.datetime | None = None,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        platform_value = Platform(platform).value
        workspace_value = str(workspace_id)
        matches = []
        for record in reversed(self._read_records()):
            if record.get("platform") != platform_value:
                continue
            if str(record.get("workspace_id")) != workspace_value:
                continue
            if since is not None and not _record_at_or_after(record, since):
                continue
            if not _is_failure_record(record):
                continue
            matches.append(record)
            if len(matches) >= limit:
                break
        return matches

    def dry_run_cases(
        self,
        platform: Platform | str,
        workspace_id: str,
        *,
        limit: int = 10,
        since: dt.datetime | None = None,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        platform_value = Platform(platform).value
        workspace_value = str(workspace_id)
        matches = []
        for record in reversed(self._read_records()):
            if record.get("platform") != platform_value:
                continue
            if str(record.get("workspace_id")) != workspace_value:
                continue
            if since is not None and not _record_at_or_after(record, since):
                continue
            if not _is_dry_run_action_record(record):
                continue
            matches.append(record)
            if len(matches) >= limit:
                break
        return matches

    def top_errors(
        self,
        platform: Platform | str,
        workspace_id: str,
        *,
        limit: int = 10,
        case_limit: int = 500,
        since: dt.datetime | None = None,
    ) -> list[dict[str, Any]]:
        if limit <= 0 or case_limit <= 0:
            return []
        records = self.failures(platform, workspace_id, limit=case_limit, since=since)
        by_error: dict[str, dict[str, Any]] = {}
        for record in records:
            for error in _record_errors(record):
                entry = by_error.setdefault(
                    error,
                    {
                        "error": error,
                        "total": 0,
                        "failed": 0,
                        "partial": 0,
                        "actions": Counter(),
                        "channels": Counter(),
                        "latest_case_id": record.get("case_id"),
                        "latest_created_at": record.get("created_at"),
                    },
                )
                entry["total"] += 1
                if record.get("execution_status") == "failed":
                    entry["failed"] += 1
                if record.get("execution_status") == "partial":
                    entry["partial"] += 1
                entry["actions"].update([str(record.get("action") or "unknown")])
                entry["channels"].update([str(record.get("channel_id") or "unknown")])

        rows = []
        for entry in by_error.values():
            rows.append(
                {
                    **entry,
                    "actions": dict(entry["actions"]),
                    "channels": dict(entry["channels"]),
                }
            )
        rows.sort(key=lambda entry: (-entry["total"], -entry["failed"], entry["error"]))
        return rows[:limit]

    def summary(
        self,
        platform: Platform | str,
        workspace_id: str,
        *,
        user_id: str | None = None,
        channel_id: str | None = None,
        category: str | None = None,
        limit: int = 100,
        since: dt.datetime | None = None,
    ) -> dict[str, Any]:
        records = self.history(
            platform,
            workspace_id,
            user_id=user_id,
            channel_id=channel_id,
            category=category,
            limit=limit,
            since=since,
        )
        category_counts: Counter[str] = Counter()
        for record in records:
            category_counts.update(str(category) for category in record.get("categories") or [])
        return {
            "platform": Platform(platform).value,
            "workspace_id": str(workspace_id),
            "user_id": str(user_id) if user_id is not None else None,
            "channel_id": str(channel_id) if channel_id is not None else None,
            "category": str(category) if category is not None else None,
            "limit": max(0, limit),
            "since": since.isoformat() if since is not None else None,
            "total": len(records),
            "verdicts": dict(_count_field(records, "verdict")),
            "actions": dict(_count_field(records, "action")),
            "statuses": dict(_count_field(records, "execution_status")),
            "categories": dict(category_counts),
            "dry_run": sum(1 for record in records if record.get("dry_run")),
            "errors": sum(1 for record in records if _is_failure_record(record)),
            "newest_case_id": records[0].get("case_id") if records else None,
            "oldest_case_id": records[-1].get("case_id") if records else None,
        }

    def top_users(
        self,
        platform: Platform | str,
        workspace_id: str,
        *,
        limit: int = 10,
        case_limit: int = 500,
        since: dt.datetime | None = None,
    ) -> list[dict[str, Any]]:
        if limit <= 0 or case_limit <= 0:
            return []
        records = self.history(platform, workspace_id, limit=case_limit, since=since)
        by_user: dict[str, dict[str, Any]] = {}
        for record in records:
            user_id = str(record.get("user_id") or "unknown")
            entry = by_user.setdefault(
                user_id,
                {
                    "user_id": user_id,
                    "username": str(record.get("username") or "unknown"),
                    "total": 0,
                    "unsafe": 0,
                    "controversial": 0,
                    "actions": Counter(),
                    "categories": Counter(),
                    "latest_case_id": record.get("case_id"),
                    "latest_created_at": record.get("created_at"),
                },
            )
            entry["total"] += 1
            if record.get("verdict") == "unsafe":
                entry["unsafe"] += 1
            if record.get("verdict") == "controversial":
                entry["controversial"] += 1
            entry["actions"].update([str(record.get("action") or "unknown")])
            entry["categories"].update(str(category) for category in record.get("categories") or [])

        rows = []
        for entry in by_user.values():
            rows.append(
                {
                    **entry,
                    "actions": dict(entry["actions"]),
                    "categories": dict(entry["categories"]),
                }
            )
        rows.sort(key=lambda entry: (-entry["total"], -entry["unsafe"], entry["user_id"]))
        return rows[:limit]

    def top_channels(
        self,
        platform: Platform | str,
        workspace_id: str,
        *,
        limit: int = 10,
        case_limit: int = 500,
        since: dt.datetime | None = None,
    ) -> list[dict[str, Any]]:
        if limit <= 0 or case_limit <= 0:
            return []
        records = self.history(platform, workspace_id, limit=case_limit, since=since)
        by_channel: dict[str, dict[str, Any]] = {}
        for record in records:
            channel_id = str(record.get("channel_id") or "unknown")
            entry = by_channel.setdefault(
                channel_id,
                {
                    "channel_id": channel_id,
                    "total": 0,
                    "unsafe": 0,
                    "controversial": 0,
                    "actions": Counter(),
                    "categories": Counter(),
                    "latest_case_id": record.get("case_id"),
                    "latest_created_at": record.get("created_at"),
                },
            )
            entry["total"] += 1
            if record.get("verdict") == "unsafe":
                entry["unsafe"] += 1
            if record.get("verdict") == "controversial":
                entry["controversial"] += 1
            entry["actions"].update([str(record.get("action") or "unknown")])
            entry["categories"].update(str(category) for category in record.get("categories") or [])

        rows = []
        for entry in by_channel.values():
            rows.append(
                {
                    **entry,
                    "actions": dict(entry["actions"]),
                    "categories": dict(entry["categories"]),
                }
            )
        rows.sort(key=lambda entry: (-entry["total"], -entry["unsafe"], entry["channel_id"]))
        return rows[:limit]

    def top_rules(
        self,
        platform: Platform | str,
        workspace_id: str,
        *,
        limit: int = 10,
        case_limit: int = 500,
        since: dt.datetime | None = None,
    ) -> list[dict[str, Any]]:
        if limit <= 0 or case_limit <= 0:
            return []
        records = self.history(platform, workspace_id, limit=case_limit, since=since)
        by_rule: dict[str, dict[str, Any]] = {}
        for record in records:
            rule = str(record.get("matched_policy_rule") or "unmatched")
            entry = by_rule.setdefault(
                rule,
                {
                    "rule": rule,
                    "total": 0,
                    "unsafe": 0,
                    "controversial": 0,
                    "actions": Counter(),
                    "categories": Counter(),
                    "latest_case_id": record.get("case_id"),
                    "latest_created_at": record.get("created_at"),
                },
            )
            entry["total"] += 1
            if record.get("verdict") == "unsafe":
                entry["unsafe"] += 1
            if record.get("verdict") == "controversial":
                entry["controversial"] += 1
            entry["actions"].update([str(record.get("action") or "unknown")])
            entry["categories"].update(str(category) for category in record.get("categories") or [])

        rows = []
        for entry in by_rule.values():
            rows.append(
                {
                    **entry,
                    "actions": dict(entry["actions"]),
                    "categories": dict(entry["categories"]),
                }
            )
        rows.sort(key=lambda entry: (-entry["total"], -entry["unsafe"], entry["rule"]))
        return rows[:limit]

    def top_categories(
        self,
        platform: Platform | str,
        workspace_id: str,
        *,
        limit: int = 10,
        case_limit: int = 500,
        since: dt.datetime | None = None,
    ) -> list[dict[str, Any]]:
        if limit <= 0 or case_limit <= 0:
            return []
        records = self.history(platform, workspace_id, limit=case_limit, since=since)
        by_category: dict[str, dict[str, Any]] = {}
        for record in records:
            categories = record.get("categories") or ["uncategorized"]
            for category_value in categories:
                category = str(category_value or "uncategorized")
                entry = by_category.setdefault(
                    category,
                    {
                        "category": category,
                        "total": 0,
                        "unsafe": 0,
                        "controversial": 0,
                        "actions": Counter(),
                        "rules": Counter(),
                        "latest_case_id": record.get("case_id"),
                        "latest_created_at": record.get("created_at"),
                    },
                )
                entry["total"] += 1
                if record.get("verdict") == "unsafe":
                    entry["unsafe"] += 1
                if record.get("verdict") == "controversial":
                    entry["controversial"] += 1
                entry["actions"].update([str(record.get("action") or "unknown")])
                entry["rules"].update([str(record.get("matched_policy_rule") or "unmatched")])

        rows = []
        for entry in by_category.values():
            rows.append(
                {
                    **entry,
                    "actions": dict(entry["actions"]),
                    "rules": dict(entry["rules"]),
                }
            )
        rows.sort(key=lambda entry: (-entry["total"], -entry["unsafe"], entry["category"]))
        return rows[:limit]

    def _read_records(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        with self._lock:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        records = []
        for line in lines:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return records


def excerpt(text: str, limit: int = 500) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 1] + "..."


def redact_text(text: str) -> str:
    redacted = _EMAIL_RE.sub("[email]", text)
    redacted = _SSN_RE.sub("[ssn]", redacted)
    redacted = _PHONE_RE.sub("[phone]", redacted)
    return _PAYMENT_CARD_RE.sub("[payment-card]", redacted)


def redacted_excerpt(text: str, limit: int = 500) -> str:
    return excerpt(redact_text(text), limit=limit)


def since_hours_ago(hours: float | int | None, *, now: dt.datetime | None = None) -> dt.datetime | None:
    if hours is None:
        return None
    now = _ensure_aware(now or dt.datetime.now(dt.timezone.utc))
    return now - dt.timedelta(hours=float(hours))


def _record_at_or_after(record: dict[str, Any], since: dt.datetime) -> bool:
    created_at = _parse_created_at(record.get("created_at"))
    return created_at is not None and created_at >= _ensure_aware(since)


def _parse_created_at(value: Any) -> dt.datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return _ensure_aware(parsed)


def _ensure_aware(value: dt.datetime) -> dt.datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=dt.timezone.utc)
    return value.astimezone(dt.timezone.utc)


def _count_field(records: list[dict[str, Any]], field_name: str) -> Counter[str]:
    return Counter(str(record.get(field_name) or "unknown") for record in records)


def _is_failure_record(record: dict[str, Any]) -> bool:
    return bool(record.get("error") or record.get("execution_status") in {"failed", "partial"})


def _is_dry_run_action_record(record: dict[str, Any]) -> bool:
    return bool(record.get("dry_run") and str(record.get("action") or "allow") != "allow")


def _record_has_category(record: dict[str, Any], category: str) -> bool:
    return category in {str(value) for value in record.get("categories") or []}


def _record_errors(record: dict[str, Any]) -> list[str]:
    raw_error = record.get("error")
    if isinstance(raw_error, str) and raw_error.strip():
        errors = [error.strip() for error in raw_error.split(";") if error.strip()]
        if errors:
            return errors
    return [str(record.get("execution_status") or "unknown")]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = [
    "DEFAULT_AUDIT_PATH",
    "AuditLog",
    "AuditRecord",
    "excerpt",
    "redact_text",
    "redacted_excerpt",
    "since_hours_ago",
    "text_hash",
]
