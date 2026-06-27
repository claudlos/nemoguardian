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

from nemoguardian.cascade import Cascade, CascadeConfig
from nemoguardian.policy.presets import get_preset
from nemoguardian.schemas import Mode, ModerateRequest, VerdictLabel


def make_moderator():
    cascade = Cascade(CascadeConfig())
    policy = get_preset("twitch")

    async def moderate(text: str) -> None:
        request = ModerateRequest(text=text, mode=Mode.FAST)
        result = await asyncio.to_thread(cascade.moderate, request, policy_engine=policy)
        # In a real Twitch integration: send /delete or /timeout via PRIVMSG.
        action = {
            VerdictLabel.SAFE: "allow",
            VerdictLabel.CONTROVERSIAL: "flag",
            VerdictLabel.UNSAFE: "delete",
        }.get(result.verdict, "allow")
        print(f"[twitch] {action}: {text[:60]} ({result.verdict.value}, score={result.score})")

    return moderate


def run_bot(channel: str) -> None:
    token = os.environ.get("TWITCH_TOKEN")
    if not token:
        raise RuntimeError("TWITCH_TOKEN env var required")

    import twitchio
    from twitchio.ext import commands

    bot = commands.Bot(token=token, prefix="!", initial_channels=[channel])
    moderate = make_moderator()

    @bot.event
    async def event_message(message) -> None:
        if message.echo:
            return
        await moderate(message.content)

    bot.run()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python -m nemoguardian.adapters.twitch <channel>")
        sys.exit(1)
    run_bot(sys.argv[1])


__all__ = ["make_moderator", "run_bot"]
