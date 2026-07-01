"""Twitch chat adapter.

Connects to a Twitch channel via twitchio, moderates every chat message with the
cascade. Token via TWITCH_TOKEN env var, channel name as CLI arg.

Run with:
    TWITCH_TOKEN=xxx python -m nemoguardian.adapters.twitch channel_name
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable

from nemoguardian.adapters.base import ActionDecision, degrade_action
from nemoguardian.bot import (
    AuditLog,
    BotConfig,
    ConfigStore,
    ModerationContext,
    ModerationEngine,
    ModerationEvaluation,
    Platform,
)
from nemoguardian.bot.types import ModerationAction
from nemoguardian.cascade import Cascade, CascadeConfig
from nemoguardian.review.service import ReviewService

#: Normalized actions this Twitch chat flow actually carries out today. The live
#: adapter evaluates and flags for review/audit, but it does not yet call Twitch
#: moderation APIs with enough message/channel context to delete, timeout, or ban.
TWITCH_CAPABILITIES: frozenset[ModerationAction] = frozenset(
    {
        ModerationAction.ALLOW,
        ModerationAction.FLAG,
    }
)


def capabilities() -> set[ModerationAction]:
    """Return the normalized actions the Twitch adapter can carry out."""
    return set(TWITCH_CAPABILITIES)


def make_moderator(
    cascade: Cascade | None = None,
    *,
    config_store: ConfigStore | None = None,
    audit_log: AuditLog | None = None,
    review_service: ReviewService | None = None,
    channel_id: str = "twitch",
    emit: Callable[[str], None] = print,
    engine: ModerationEngine | None = None,
):
    if engine is None:
        cascade = cascade or Cascade(CascadeConfig.from_env())
        engine = ModerationEngine(
            Platform.TWITCH,
            cascade=cascade,
            config_store=config_store,
            audit_log=audit_log,
            review_service=review_service,
        )

    async def moderate(text: str, *, user_id: str = "unknown", username: str = "unknown") -> str:
        config = engine.config_for(channel_id)
        context = ModerationContext(
            platform=Platform.TWITCH,
            workspace_id=channel_id,
            channel_id=channel_id,
            message_id=f"twitch-{abs(hash((channel_id, user_id, text))) % 10_000_000}",
            user_id=user_id,
            username=username,
            text=text,
        )
        evaluation = await asyncio.to_thread(engine.evaluate, context, config)
        action = evaluation.plan.action.value
        if evaluation.result is not None:
            decision = degrade_action(evaluation.plan.action, capabilities(), Platform.TWITCH)
            if decision.degraded:
                evaluation.plan.action = decision.action
            action = evaluation.plan.action.value
            engine.record(evaluation, execution_status=action, error=decision.reason)
            emit(
                f"[twitch] {action}: {text[:60]} "
                f"({evaluation.result.verdict.value}, score={evaluation.result.score})"
            )
        else:
            emit(f"[twitch] allow: {text[:60]} (skipped={evaluation.skip_reason})")
        return action

    return moderate


class TwitchAdapter:
    """Thin :class:`~nemoguardian.adapters.base.PlatformAdapter` over Twitch chat.

    Wraps :func:`make_moderator` so behavior is unchanged while exposing the
    normalized interface. Twitch enforcement happens inside ``handle_event``
    (decide + record + emit); :meth:`apply_action` surfaces the
    capability-degraded decision for a planned action.
    """

    platform = Platform.TWITCH

    def __init__(
        self,
        cascade: Cascade | None = None,
        *,
        config_store: ConfigStore | None = None,
        audit_log: AuditLog | None = None,
        review_service: ReviewService | None = None,
        channel_id: str = "twitch",
        emit: Callable[[str], None] = print,
    ) -> None:
        self.channel_id = channel_id
        self.engine = ModerationEngine(
            Platform.TWITCH,
            cascade=cascade,
            config_store=config_store,
            audit_log=audit_log,
            review_service=review_service,
        )
        self._moderate = make_moderator(channel_id=channel_id, emit=emit, engine=self.engine)

    def capabilities(self) -> set[ModerationAction]:
        return capabilities()

    def doctor(self, workspace_id: str | None = None) -> dict[str, object]:
        """Return a lightweight readiness snapshot (never raises)."""
        workspace = str(workspace_id) if workspace_id is not None else self.channel_id
        config = self.engine.config_for(workspace)
        token_configured = bool(os.environ.get("TWITCH_TOKEN"))
        issues: list[str] = []
        if not config.enabled:
            issues.append("moderation is disabled")
        if not token_configured:
            issues.append("TWITCH_TOKEN is not set")
        return {
            "platform": Platform.TWITCH.value,
            "workspace_id": workspace,
            "enabled": config.enabled,
            "token_configured": token_configured,
            "readiness": "ready" if not issues else "needs attention",
            "issues": issues,
        }

    def configure(self, workspace_id: str | None = None, **changes: object) -> BotConfig:
        workspace = str(workspace_id) if workspace_id is not None else self.channel_id
        if changes:
            return self.engine.config_store.update(Platform.TWITCH, workspace, **changes)
        return self.engine.config_for(workspace)

    async def handle_event(
        self,
        text: str,
        *,
        user_id: str = "unknown",
        username: str = "unknown",
    ) -> str:
        return await self._moderate(text, user_id=user_id, username=username)

    def apply_action(self, evaluation: ModerationEvaluation) -> ActionDecision:
        """Resolve the planned action against Twitch capabilities."""
        return degrade_action(evaluation.plan.action, self.capabilities(), Platform.TWITCH)

    def record_audit(
        self,
        evaluation: ModerationEvaluation,
        *,
        execution_status: str,
        error: str | None = None,
    ) -> None:
        self.engine.record(evaluation, execution_status=execution_status, error=error)


def run_bot(channel: str) -> None:
    token = os.environ.get("TWITCH_TOKEN")
    if not token:
        raise RuntimeError("TWITCH_TOKEN env var required")

    from twitchio.ext import commands

    bot = commands.Bot(token=token, prefix="!", initial_channels=[channel])
    moderate = make_moderator(channel_id=channel, review_service=ReviewService())

    @bot.event
    async def event_message(message) -> None:
        if message.echo:
            return
        await moderate(
            message.content,
            user_id=str(getattr(message.author, "id", "unknown")),
            username=str(getattr(message.author, "name", "unknown")),
        )

    bot.run()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m nemoguardian.adapters.twitch <channel>")
        sys.exit(1)
    run_bot(sys.argv[1])


__all__ = [
    "TWITCH_CAPABILITIES",
    "TwitchAdapter",
    "capabilities",
    "make_moderator",
    "run_bot",
]
