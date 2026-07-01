"""Platform adapters (Discord, Slack, Telegram, Twitch, generic webhook).

Every adapter conforms to the shared :class:`~nemoguardian.adapters.base.PlatformAdapter`
interface and declares its normalized :class:`~nemoguardian.bot.types.ModerationAction`
capabilities; :func:`~nemoguardian.adapters.base.degrade_action` gracefully degrades
any action a platform cannot perform to ``flag`` with an auditable reason.
"""

from nemoguardian.adapters import base, discord, slack, telegram, twitch, webhook
from nemoguardian.adapters.base import (
    DEGRADE_TARGET,
    NORMALIZED_ACTIONS,
    ActionDecision,
    PlatformAdapter,
    degrade_action,
    supports,
)

__all__ = [
    "DEGRADE_TARGET",
    "NORMALIZED_ACTIONS",
    "ActionDecision",
    "PlatformAdapter",
    "base",
    "degrade_action",
    "discord",
    "slack",
    "supports",
    "telegram",
    "twitch",
    "webhook",
]
