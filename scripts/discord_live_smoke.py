#!/usr/bin/env python3
"""Live Discord smoke test for the moderation bot.

Uses two Discord bot applications:
- DISCORD_BOT_TOKEN: the nemoguardian moderator bot.
- DISCORD_TEST_SENDER_TOKEN: a separate sender bot used only for E2E smoke messages.

The moderator ignores bot-authored messages by default. This script fetches the
sender bot ID and allowlists only that ID through
NEMOGUARDIAN_DISCORD_E2E_BOT_AUTHOR_IDS before starting the moderator bot.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

DISCORD_API = "https://discord.com/api/v10"


class SmokeFailure(RuntimeError):
    def __init__(self, evidence: dict[str, Any]) -> None:
        self.evidence = evidence
        super().__init__(str(evidence.get("error", "live smoke failed")))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guild-id", default=os.environ.get("DISCORD_GUILD_ID"))
    parser.add_argument("--channel-id", default=os.environ.get("DISCORD_TEST_CHANNEL_ID"))
    parser.add_argument("--mod-log-channel-id", default=os.environ.get("DISCORD_MOD_LOG_CHANNEL_ID"))
    parser.add_argument("--bot-token", default=os.environ.get("DISCORD_BOT_TOKEN"))
    parser.add_argument("--sender-token", default=os.environ.get("DISCORD_TEST_SENDER_TOKEN"))
    parser.add_argument(
        "--config-path",
        default=os.environ.get("NEMOGUARDIAN_BOT_CONFIG_PATH", "/tmp/nemoguardian_discord_live_config.json"),
    )
    parser.add_argument(
        "--audit-path",
        default=os.environ.get("NEMOGUARDIAN_BOT_AUDIT_PATH", "/tmp/nemoguardian_discord_live_audit.jsonl"),
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "standard", "deep"],
        default=os.environ.get("NEMOGUARDIAN_DISCORD_SMOKE_MODE", "standard"),
    )
    parser.add_argument(
        "--text",
        default="Hey @everyone, my SSN is 123-45-6789, DM me for cash.",
    )
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Disable dry-run and verify that the smoke message is deleted.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=90.0)
    return parser.parse_args()


def _require(value: str | None, name: str) -> str:
    if not value:
        raise SystemExit(f"{name} is required")
    return value


def _bot_headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bot {token}"}


async def _discord_get(client: httpx.AsyncClient, token: str, path: str) -> httpx.Response:
    response = await client.get(f"{DISCORD_API}{path}", headers=_bot_headers(token))
    response.raise_for_status()
    return response


async def _sender_identity(client: httpx.AsyncClient, sender_token: str) -> dict[str, Any]:
    response = await _discord_get(client, sender_token, "/users/@me")
    return response.json()


async def _send_message(client: httpx.AsyncClient, sender_token: str, channel_id: str, text: str) -> dict[str, Any]:
    response = await client.post(
        f"{DISCORD_API}/channels/{channel_id}/messages",
        headers=_bot_headers(sender_token),
        json={"content": text, "allowed_mentions": {"parse": []}},
    )
    response.raise_for_status()
    return response.json()


async def _message_deleted(client: httpx.AsyncClient, sender_token: str, channel_id: str, message_id: str) -> bool:
    response = await client.get(
        f"{DISCORD_API}/channels/{channel_id}/messages/{message_id}",
        headers=_bot_headers(sender_token),
    )
    if response.status_code == 404:
        return True
    response.raise_for_status()
    return False


async def _wait_for_case(audit_log: Any, case_id: str, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        record = audit_log.find_case(case_id)
        if record is not None:
            return record
        await asyncio.sleep(1)
    raise TimeoutError(f"timed out waiting for audit case {case_id}")


async def _wait_for_bot_ready(ready: asyncio.Event, bot_task: asyncio.Task[None], timeout_seconds: float) -> None:
    ready_task = asyncio.create_task(ready.wait())
    try:
        done, _pending = await asyncio.wait(
            {ready_task, bot_task},
            timeout=timeout_seconds,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if not done:
            raise TimeoutError("timed out waiting for Discord bot readiness")
        if bot_task in done:
            bot_task.result()
            raise RuntimeError("Discord bot exited before becoming ready")
    finally:
        ready_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await ready_task


async def run(args: argparse.Namespace) -> dict[str, Any]:
    guild_id = _require(args.guild_id, "DISCORD_GUILD_ID or --guild-id")
    channel_id = _require(args.channel_id, "DISCORD_TEST_CHANNEL_ID or --channel-id")
    bot_token = _require(args.bot_token, "DISCORD_BOT_TOKEN or --bot-token")
    sender_token = _require(args.sender_token, "DISCORD_TEST_SENDER_TOKEN or --sender-token")
    config_path = Path(args.config_path)
    audit_path = Path(args.audit_path)

    async with httpx.AsyncClient(timeout=20.0) as client:
        sender = await _sender_identity(client, sender_token)

    os.environ["DISCORD_GUILD_ID"] = guild_id
    os.environ["NEMOGUARDIAN_BOT_CONFIG_PATH"] = str(config_path)
    os.environ["NEMOGUARDIAN_BOT_AUDIT_PATH"] = str(audit_path)
    os.environ["NEMOGUARDIAN_DISCORD_E2E_BOT_AUTHOR_IDS"] = str(sender["id"])

    from nemoguardian.adapters import discord as discord_adapter
    from nemoguardian.bot import BotConfig, ConfigStore, Platform
    from nemoguardian.bot.audit import AuditLog
    from nemoguardian.schemas import Mode

    config = BotConfig.default(Platform.DISCORD, guild_id)
    config.enabled = True
    config.mode = Mode(args.mode)
    config.log_channel_id = str(args.mod_log_channel_id or channel_id)
    config.dry_run = not args.enforce
    config.delete_unsafe = True
    config.public_warning = False
    config.dm_users = False
    ConfigStore(config_path).save(config)

    bot = discord_adapter.build_bot()
    ready = asyncio.Event()
    observed_message: dict[str, Any] = {}

    @bot.listen("on_ready")
    async def _smoke_on_ready() -> None:
        ready.set()

    @bot.listen("on_message")
    async def _smoke_on_message(message_obj: Any) -> None:
        author = getattr(message_obj, "author", None)
        channel = getattr(message_obj, "channel", None)
        if str(getattr(author, "id", "")) != str(sender["id"]):
            return
        if str(getattr(channel, "id", "")) != channel_id:
            return
        content = str(getattr(message_obj, "content", "") or "")
        observed_message.clear()
        observed_message.update(
            {
                "message_id": str(getattr(message_obj, "id", "")),
                "content_length": len(content),
                "author_bot": bool(getattr(author, "bot", False)),
                "channel_id": str(getattr(channel, "id", "")),
            }
        )

    bot_task = asyncio.create_task(bot.start(bot_token))
    try:
        await _wait_for_bot_ready(ready, bot_task, args.timeout_seconds)

        async with httpx.AsyncClient(timeout=20.0) as client:
            message = await _send_message(client, sender_token, channel_id, args.text)
            case_id = f"discord-{guild_id}-{message['id']}"
            try:
                record = await _wait_for_case(AuditLog(audit_path), case_id, args.timeout_seconds)
            except TimeoutError as exc:
                gateway_seen = observed_message.get("message_id") == str(message["id"])
                content_length = observed_message.get("content_length")
                if gateway_seen and content_length == 0:
                    hint = "Moderator saw the message event, but Discord delivered empty content. Enable Message Content Intent for the moderator bot."
                elif gateway_seen:
                    hint = "Moderator saw the message content, but no audit record was written. Check handler/model logs."
                else:
                    hint = "Moderator did not observe the sender message. Check bot channel permissions and gateway intents."
                raise SmokeFailure(
                    {
                        "ok": False,
                        "error": str(exc),
                        "hint": hint,
                        "guild_id": guild_id,
                        "channel_id": channel_id,
                        "sender_bot_id": str(sender["id"]),
                        "message_id": str(message["id"]),
                        "case_id": case_id,
                        "gateway_message_seen": gateway_seen,
                        "gateway_observed": observed_message or None,
                        "audit_path": str(audit_path),
                        "config_path": str(config_path),
                    }
                ) from exc
            deleted = await _message_deleted(client, sender_token, channel_id, message["id"])
    finally:
        await bot.close()
        bot_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await bot_task

    evidence = {
        "ok": True,
        "guild_id": guild_id,
        "channel_id": channel_id,
        "sender_bot_id": str(sender["id"]),
        "message_id": str(message["id"]),
        "case_id": case_id,
        "mode": args.mode,
        "dry_run": not args.enforce,
        "execution_status": record.get("execution_status"),
        "action": record.get("action"),
        "verdict": record.get("verdict"),
        "score": record.get("score"),
        "deleted": deleted,
        "audit_path": str(audit_path),
        "config_path": str(config_path),
    }
    if args.enforce and not deleted:
        evidence["ok"] = False
        raise SmokeFailure(evidence)
    return evidence


def main() -> int:
    args = parse_args()
    try:
        evidence = asyncio.run(run(args))
    except SmokeFailure as exc:
        print(json.dumps(exc.evidence, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    except Exception as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
