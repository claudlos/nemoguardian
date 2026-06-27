"""Discord bot adapter.

Uses discord.py. The bot moderates every message against the cascade and the
default Discord preset policy. Add the bot token via DISCORD_BOT_TOKEN env var.

Run with:
    DISCORD_BOT_TOKEN=xxx python -m nemoguardian.adapters.discord
"""

from __future__ import annotations

import asyncio
import os

from nemoguardian.cascade import Cascade, CascadeConfig
from nemoguardian.policy.presets import get_preset
from nemoguardian.schemas import Mode, ModerateRequest, VerdictLabel


def make_handler(cascade: Cascade | None = None):
    """Build an async message handler that runs the cascade."""
    cascade = cascade or Cascade(CascadeConfig())
    policy = get_preset("discord")

    async def on_message(message) -> None:
        if message.author.bot:
            return
        request = ModerateRequest(text=message.content or "", mode=Mode.STANDARD)
        result = await asyncio.to_thread(cascade.moderate, request, policy_engine=policy)
        if result.verdict == VerdictLabel.UNSAFE:
            await message.delete()
            await message.channel.send(
                f"⚠️ {message.author.mention}, that message was blocked by nemoguardian: "
                f"{', '.join(result.categories) or 'policy violation'}"
            )
        elif result.verdict == VerdictLabel.CONTROVERSIAL:
            await message.add_reaction("⚠️")

    return on_message


def run_bot() -> None:
    """Entry point: start a Discord bot that moderates every message."""
    token = os.environ.get("DISCORD_BOT_TOKEN")
    if not token:
        raise RuntimeError("DISCORD_BOT_TOKEN env var required")

    import discord

    intents = discord.Intents.default()
    intents.message_content = True
    client = discord.Client(intents=intents)
    handler = make_handler()

    @client.event
    async def on_ready() -> None:
        print(f"[nemoguardian] bot ready as {client.user}")

    @client.event
    async def on_message(message) -> None:
        await handler(message)

    client.run(token)


if __name__ == "__main__":
    run_bot()


__all__ = ["make_handler", "run_bot"]
