"""Common bot enums used by platform adapters."""

from __future__ import annotations

from enum import Enum


class Platform(str, Enum):
    DISCORD = "discord"
    TWITCH = "twitch"
    WEBHOOK = "webhook"
    SLACK = "slack"
    TELEGRAM = "telegram"
    MATRIX = "matrix"
    REDDIT = "reddit"


class ModerationAction(str, Enum):
    """Normalized, cross-platform moderation action vocabulary.

    Adapters translate a platform event into one of these values so audit
    records, policy presets and the capability layer all speak the same
    language regardless of which platform produced the event. New values are
    additive: existing ``allow``/``flag``/``delete``/``timeout`` behavior is
    unchanged.
    """

    ALLOW = "allow"                # take no action
    FLAG = "flag"                  # surface for review (reaction / mod-log / queue)
    DELETE = "delete"             # remove the offending message
    TIMEOUT = "timeout"          # temporarily restrict the author (Discord-style)
    MUTE = "mute"                 # silence the author without a fixed timeout window
    BAN = "ban"                   # remove the author from the space
    QUEUE = "queue"               # hold for human moderator review
    NOTIFY_MODS = "notify_mods"   # alert moderators (mod-log / channel ping)
    NOTIFY_USER = "notify_user"   # inform the author (DM / ephemeral notice)


__all__ = ["ModerationAction", "Platform"]
