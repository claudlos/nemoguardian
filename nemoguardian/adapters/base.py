"""Shared platform-adapter foundation.

This module defines the small, normalized surface every platform adapter
conforms to, plus capability detection and graceful-degradation helpers. It is
deliberately *additive*: the existing Discord/Twitch/webhook adapters keep their
ergonomic, function-based entry points (``make_handler``, ``make_moderator``,
``moderate_and_forward``) and simply *also* expose a thin adapter class plus a
``capabilities()`` declaration that satisfy this interface. Future Slack /
Telegram adapters plug in by implementing :class:`PlatformAdapter` without
re-touching the core.

Design notes
------------
* :class:`PlatformAdapter` is a ``runtime_checkable`` :class:`~typing.Protocol`
  (structural, not a heavy ABC) so each adapter keeps its platform-shaped
  signatures. The Protocol pins the method *names* and the normalized vocabulary
  they exchange, not exact argument lists.
* :data:`NORMALIZED_ACTIONS` is the single shared action vocabulary
  (:class:`~nemoguardian.bot.types.ModerationAction`).
* :func:`degrade_action` never silently drops enforcement: an action a platform
  cannot perform degrades to :data:`DEGRADE_TARGET` (``flag``) with a clear,
  auditable reason string.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from nemoguardian.bot.types import ModerationAction, Platform

if TYPE_CHECKING:
    from collections.abc import Iterable

#: The normalized, cross-platform action vocabulary. Every adapter speaks this.
NORMALIZED_ACTIONS: frozenset[ModerationAction] = frozenset(ModerationAction)

#: Where an unsupported action degrades to. ``flag`` surfaces the case to
#: moderators without enforcing something the platform cannot do — the safe
#: middle ground between full enforcement and silently dropping it.
DEGRADE_TARGET: ModerationAction = ModerationAction.FLAG


@runtime_checkable
class PlatformAdapter(Protocol):
    """Structural interface implemented by every platform adapter.

    The methods mirror the lifecycle of a moderated event:

    * ``capabilities()`` -> the normalized actions this platform can carry out.
    * ``doctor()``       -> permission / health / readiness snapshot (never raises).
    * ``configure()``    -> read or update the per-workspace ``BotConfig``.
    * ``handle_event()`` -> platform event -> moderation -> planned/applied action.
    * ``apply_action()`` -> apply the planned (capability-degraded) action.
    * ``record_audit()`` -> append an audit record for the handled event.

    Signatures are intentionally loose (``*args`` / ``**kwargs``): each platform
    keeps its own natural event and permission shapes; this Protocol only fixes
    the method *names* and the normalized vocabulary they exchange.
    """

    #: The platform this adapter moderates.
    platform: Platform

    def capabilities(self) -> set[ModerationAction]: ...

    def doctor(self, *args: Any, **kwargs: Any) -> Any: ...

    def configure(self, *args: Any, **kwargs: Any) -> Any: ...

    def handle_event(self, *args: Any, **kwargs: Any) -> Any: ...

    def apply_action(self, *args: Any, **kwargs: Any) -> Any: ...

    def record_audit(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class ActionDecision:
    """Outcome of resolving a desired action against a platform's capabilities."""

    action: ModerationAction          #: what will actually be carried out
    requested: ModerationAction       #: what the policy originally asked for
    degraded: bool                    #: True if ``requested`` != ``action``
    reason: str | None = None         #: audit-ready explanation when degraded


def supports(capabilities: Iterable[ModerationAction], action: ModerationAction | str) -> bool:
    """Return ``True`` if ``action`` is within ``capabilities``."""
    return ModerationAction(action) in set(capabilities)


def degrade_action(
    desired: ModerationAction | str,
    capabilities: Iterable[ModerationAction],
    platform: Platform | str,
) -> ActionDecision:
    """Resolve ``desired`` against a platform's ``capabilities``.

    If the platform supports the action it passes through unchanged. Otherwise
    it degrades to :data:`DEGRADE_TARGET` (``flag``) and records a clear,
    auditable reason (e.g. ``"timeout unsupported on twitch -> degraded to
    flag"``). Enforcement is never silently dropped.
    """
    desired = ModerationAction(desired)
    caps = set(capabilities)
    platform_name = platform.value if isinstance(platform, Platform) else str(platform)
    if desired in caps:
        return ActionDecision(action=desired, requested=desired, degraded=False)
    return ActionDecision(
        action=DEGRADE_TARGET,
        requested=desired,
        degraded=True,
        reason=(
            f"{desired.value} unsupported on {platform_name} "
            f"-> degraded to {DEGRADE_TARGET.value}"
        ),
    )


__all__ = [
    "DEGRADE_TARGET",
    "NORMALIZED_ACTIONS",
    "ActionDecision",
    "PlatformAdapter",
    "degrade_action",
    "supports",
]
