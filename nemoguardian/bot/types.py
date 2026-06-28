"""Common bot enums used by platform adapters."""

from __future__ import annotations

from enum import Enum


class Platform(str, Enum):
    DISCORD = "discord"
    TWITCH = "twitch"
    WEBHOOK = "webhook"


class ModerationAction(str, Enum):
    ALLOW = "allow"
    FLAG = "flag"
    DELETE = "delete"
    TIMEOUT = "timeout"


__all__ = ["ModerationAction", "Platform"]
