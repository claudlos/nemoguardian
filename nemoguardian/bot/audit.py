"""Append-only moderation audit records."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nemoguardian.bot.types import ModerationAction, Platform
from nemoguardian.schemas import Mode, VerdictLabel

DEFAULT_AUDIT_PATH = Path(
    os.environ.get("NEMOGUARDIAN_BOT_AUDIT_PATH", "/tmp/nemoguardian_bot_audit.jsonl")
)


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
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        records = []
        for line in lines[-limit:]:
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


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = ["DEFAULT_AUDIT_PATH", "AuditLog", "AuditRecord", "excerpt", "text_hash"]
