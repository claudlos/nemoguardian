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

from nemoguardian.bot import AuditLog, ConfigStore, ModerationContext, ModerationEngine, Platform
from nemoguardian.cascade import Cascade, CascadeConfig


def make_moderator(
    cascade: Cascade | None = None,
    *,
    config_store: ConfigStore | None = None,
    audit_log: AuditLog | None = None,
    channel_id: str = "twitch",
    emit: Callable[[str], None] = print,
):
    cascade = cascade or Cascade(CascadeConfig.from_env())
    engine = ModerationEngine(
        Platform.TWITCH,
        cascade=cascade,
        config_store=config_store,
        audit_log=audit_log,
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
            engine.record(evaluation, execution_status=action)
            emit(
                f"[twitch] {action}: {text[:60]} "
                f"({evaluation.result.verdict.value}, score={evaluation.result.score})"
            )
        else:
            emit(f"[twitch] allow: {text[:60]} (skipped={evaluation.skip_reason})")
        return action

    return moderate


def run_bot(channel: str) -> None:
    token = os.environ.get("TWITCH_TOKEN")
    if not token:
        raise RuntimeError("TWITCH_TOKEN env var required")

    from twitchio.ext import commands

    bot = commands.Bot(token=token, prefix="!", initial_channels=[channel])
    moderate = make_moderator()

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


__all__ = ["make_moderator", "run_bot"]
