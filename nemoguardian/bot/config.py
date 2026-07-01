"""File-backed platform bot configuration.

The config store is intentionally small and JSON-backed so a self-hosted bot can
start without a database migration. The public surface is stable enough to move
to SQLite/Postgres later.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nemoguardian.bot.types import Platform
from nemoguardian.schemas import Mode

DEFAULT_CONFIG_PATH = Path(
    os.environ.get("NEMOGUARDIAN_BOT_CONFIG_PATH", "/tmp/nemoguardian_bot_config.json")
)


@dataclass
class BotConfig:
    platform: Platform
    workspace_id: str
    enabled: bool = True
    policy_preset: str = "discord"
    policy_text: str = "block PII, scams, harassment, slurs, and threats"
    mode: Mode = Mode.STANDARD
    log_channel_id: str | None = None
    public_warning: bool = True
    react_controversial: bool = True
    delete_unsafe: bool = True
    timeout_unsafe: bool = False
    timeout_seconds: int = 600
    dm_users: bool = False
    dry_run: bool = False
    review_queue: bool = True  # enqueue controversial/flagged cases when a ReviewService is wired in
    ignored_channel_ids: set[str] = field(default_factory=set)
    ignored_role_ids: set[str] = field(default_factory=set)
    exempt_user_ids: set[str] = field(default_factory=set)

    @classmethod
    def default(cls, platform: Platform | str, workspace_id: str) -> BotConfig:
        platform_enum = Platform(platform)
        if platform_enum == Platform.DISCORD:
            return cls(platform=platform_enum, workspace_id=str(workspace_id), policy_preset="discord")
        if platform_enum == Platform.TWITCH:
            return cls(
                platform=platform_enum,
                workspace_id=str(workspace_id),
                policy_preset="twitch",
                mode=Mode.FAST,
                public_warning=False,
            )
        if platform_enum == Platform.SLACK:
            return cls(platform=platform_enum, workspace_id=str(workspace_id), policy_preset="slack")
        if platform_enum == Platform.TELEGRAM:
            return cls(
                platform=platform_enum,
                workspace_id=str(workspace_id),
                policy_preset="telegram",
            )
        return cls(
            platform=platform_enum,
            workspace_id=str(workspace_id),
            policy_preset="generic",
            public_warning=False,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BotConfig:
        raw = dict(data)
        raw["platform"] = Platform(raw["platform"])
        raw["workspace_id"] = str(raw["workspace_id"])
        raw["mode"] = Mode(raw.get("mode", Mode.STANDARD.value))
        for key in ("ignored_channel_ids", "ignored_role_ids", "exempt_user_ids"):
            raw[key] = {str(value) for value in raw.get(key, [])}
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["platform"] = self.platform.value
        data["mode"] = self.mode.value
        for key in ("ignored_channel_ids", "ignored_role_ids", "exempt_user_ids"):
            data[key] = sorted(data[key])
        return data


class ConfigStore:
    """Tiny JSON config store keyed by platform/workspace."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
        self._lock = threading.Lock()

    def get(self, platform: Platform | str, workspace_id: str) -> BotConfig:
        key = self._key(platform, workspace_id)
        with self._lock:
            data = self._read()
            if key not in data:
                return BotConfig.default(platform, workspace_id)
            return BotConfig.from_dict(data[key])

    def save(self, config: BotConfig) -> BotConfig:
        key = self._key(config.platform, config.workspace_id)
        with self._lock:
            data = self._read()
            data[key] = config.to_dict()
            self._write(data)
        return config

    def update(self, platform: Platform | str, workspace_id: str, **changes: Any) -> BotConfig:
        config = self.get(platform, workspace_id)
        for key, value in changes.items():
            if value is not None:
                setattr(config, key, value)
        return self.save(config)

    def _read(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def _write(self, data: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=self.path.name, dir=self.path.parent)
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, indent=2, sort_keys=True)
                handle.write("\n")
            tmp_path.replace(self.path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @staticmethod
    def _key(platform: Platform | str, workspace_id: str) -> str:
        return f"{Platform(platform).value}:{workspace_id}"


__all__ = ["DEFAULT_CONFIG_PATH", "BotConfig", "ConfigStore"]
