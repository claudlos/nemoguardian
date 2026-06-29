#!/usr/bin/env python3
"""Run a multi-actor Discord moderation scenario for demos and E2E checks."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.discord_live_smoke import (  # noqa: E402
    SmokeFailure,
    _message_deleted,
    _require,
    _send_message,
    _sender_identity,
    _wait_for_bot_ready,
    _wait_for_case,
)

DEFAULT_SCENARIO: list[dict[str, Any]] = [
    {
        "actor": 0,
        "label": "good-helper",
        "text": "Can someone explain how to enable two-factor authentication?",
        "expect_action": "allow",
        "expect_verdict": "safe",
    },
    {
        "actor": 1,
        "label": "bad-actor",
        "text": "Hey @everyone, my SSN is 123-45-6789, DM me for cash.",
        "expect_action": "delete",
        "expect_verdict": "unsafe",
    },
    {
        "actor": 0,
        "label": "good-helper",
        "text": "Please do not post private info here. Mods can help if needed.",
        "expect_action": "allow",
        "expect_verdict": "safe",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guild-id", default=os.environ.get("DISCORD_GUILD_ID"))
    parser.add_argument("--channel-id", default=os.environ.get("DISCORD_TEST_CHANNEL_ID"))
    parser.add_argument("--mod-log-channel-id", default=os.environ.get("DISCORD_MOD_LOG_CHANNEL_ID"))
    parser.add_argument("--bot-token", default=os.environ.get("DISCORD_BOT_TOKEN"))
    parser.add_argument(
        "--actor-tokens",
        default=(
            os.environ.get("DISCORD_ACTOR_TOKENS")
            or os.environ.get("DISCORD_TEST_SENDER_TOKENS")
            or os.environ.get("DISCORD_TEST_SENDER_TOKEN")
        ),
        help="Comma-separated sender bot tokens. Falls back to DISCORD_TEST_SENDER_TOKEN.",
    )
    parser.add_argument("--scenario-json", help="Optional JSON file containing scenario turns.")
    parser.add_argument(
        "--config-path",
        # Dedicated actor var wins so sourcing discord-live.env (which sets the
        # shared NEMOGUARDIAN_BOT_* paths for the live smoke) doesn't silently
        # redirect the actor scenario's files.
        default=(
            os.environ.get("NEMOGUARDIAN_ACTOR_CONFIG_PATH")
            or os.environ.get("NEMOGUARDIAN_BOT_CONFIG_PATH")
            or "/tmp/nemoguardian_discord_actor_config.json"
        ),
    )
    parser.add_argument(
        "--audit-path",
        default=(
            os.environ.get("NEMOGUARDIAN_ACTOR_AUDIT_PATH")
            or os.environ.get("NEMOGUARDIAN_BOT_AUDIT_PATH")
            or "/tmp/nemoguardian_discord_actor_audit.jsonl"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["fast", "standard", "deep"],
        default=os.environ.get("NEMOGUARDIAN_DISCORD_SCENARIO_MODE", "standard"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--turn-delay-seconds", type=float, default=1.0)
    parser.add_argument("--timeout-unsafe", action="store_true")
    parser.add_argument("--timeout-action-seconds", type=int, default=600)
    parser.add_argument("--public-warning", action="store_true")
    parser.add_argument("--enforce", action="store_true", help="Disable dry-run and apply moderation actions.")
    parser.add_argument(
        "--no-check-expectations",
        action="store_true",
        help="Record evidence only; do not fail on expected action/verdict mismatches.",
    )
    return parser.parse_args()


def parse_actor_tokens(raw: str | None) -> list[str]:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def load_scenario(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return [dict(turn) for turn in DEFAULT_SCENARIO]
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("scenario JSON must be a list of turns")
    turns: list[dict[str, Any]] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"scenario turn {index} must be an object")
        if not str(item.get("text", "")).strip():
            raise ValueError(f"scenario turn {index} must include non-empty text")
        turns.append(dict(item))
    return turns


def _actor_index(turn: dict[str, Any], actor_count: int) -> int:
    index = int(turn.get("actor", 0))
    if index < 0:
        raise ValueError("actor index cannot be negative")
    if actor_count <= 0:
        raise ValueError("at least one actor token is required")
    return index if index < actor_count else index % actor_count


def _expectation_failures(
    turn: dict[str, Any],
    record: dict[str, Any],
    *,
    deleted: bool,
    enforce: bool,
) -> list[str]:
    failures: list[str] = []
    expected_action = turn.get("expect_action")
    if expected_action is not None and record.get("action") != expected_action:
        failures.append(f"expected action {expected_action}, got {record.get('action')}")

    expected_verdict = turn.get("expect_verdict")
    if expected_verdict is not None and record.get("verdict") != expected_verdict:
        failures.append(f"expected verdict {expected_verdict}, got {record.get('verdict')}")

    if enforce and expected_action == "delete" and not deleted:
        failures.append("expected enforced delete, but message is still visible")
    if enforce and expected_action == "allow" and deleted:
        failures.append("expected allowed message to remain visible, but it was deleted")
    return failures


async def run(args: argparse.Namespace) -> dict[str, Any]:
    guild_id = _require(args.guild_id, "DISCORD_GUILD_ID or --guild-id")
    channel_id = _require(args.channel_id, "DISCORD_TEST_CHANNEL_ID or --channel-id")
    bot_token = _require(args.bot_token, "DISCORD_BOT_TOKEN or --bot-token")
    actor_tokens = parse_actor_tokens(args.actor_tokens)
    if not actor_tokens:
        raise SystemExit("DISCORD_ACTOR_TOKENS or DISCORD_TEST_SENDER_TOKEN is required")

    scenario = load_scenario(args.scenario_json)
    config_path = Path(args.config_path)
    audit_path = Path(args.audit_path)

    async with httpx.AsyncClient(timeout=20.0) as client:
        actors = [await _sender_identity(client, token) for token in actor_tokens]

    actor_ids = [str(actor["id"]) for actor in actors]
    os.environ["DISCORD_GUILD_ID"] = guild_id
    os.environ["NEMOGUARDIAN_BOT_CONFIG_PATH"] = str(config_path)
    os.environ["NEMOGUARDIAN_BOT_AUDIT_PATH"] = str(audit_path)
    os.environ["NEMOGUARDIAN_DISCORD_E2E_BOT_AUTHOR_IDS"] = ",".join(actor_ids)

    from nemoguardian.adapters import discord as discord_adapter
    from nemoguardian.bot import BotConfig, ConfigStore, Platform
    from nemoguardian.bot.audit import AuditLog, redacted_excerpt
    from nemoguardian.schemas import Mode

    config = BotConfig.default(Platform.DISCORD, guild_id)
    config.enabled = True
    config.mode = Mode(args.mode)
    config.log_channel_id = str(args.mod_log_channel_id or channel_id)
    config.dry_run = not args.enforce
    config.delete_unsafe = True
    config.timeout_unsafe = bool(args.timeout_unsafe)
    config.timeout_seconds = max(60, min(args.timeout_action_seconds, 2_419_200))
    config.public_warning = bool(args.public_warning)
    config.dm_users = False
    ConfigStore(config_path).save(config)

    bot = discord_adapter.build_bot()
    ready = asyncio.Event()

    @bot.listen("on_ready")
    async def _scenario_on_ready() -> None:
        ready.set()

    bot_task = asyncio.create_task(bot.start(bot_token))
    turns: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    try:
        await _wait_for_bot_ready(ready, bot_task, args.timeout_seconds)
        audit = AuditLog(audit_path)

        async with httpx.AsyncClient(timeout=20.0) as client:
            for turn_number, turn in enumerate(scenario, start=1):
                actor_index = _actor_index(turn, len(actor_tokens))
                actor = actors[actor_index]
                message = await _send_message(
                    client,
                    actor_tokens[actor_index],
                    channel_id,
                    str(turn["text"]),
                )
                case_id = f"discord-{guild_id}-{message['id']}"
                record = await _wait_for_case(audit, case_id, args.timeout_seconds)
                deleted = await _message_deleted(
                    client,
                    actor_tokens[actor_index],
                    channel_id,
                    message["id"],
                )
                turn_failures = []
                if not args.no_check_expectations:
                    turn_failures = _expectation_failures(
                        turn,
                        record,
                        deleted=deleted,
                        enforce=args.enforce,
                    )
                evidence_turn = {
                    "turn": turn_number,
                    "label": str(turn.get("label", f"actor-{actor_index}")),
                    "actor_index": actor_index,
                    "actor_bot_id": str(actor["id"]),
                    "message_id": str(message["id"]),
                    "case_id": case_id,
                    "text_excerpt": redacted_excerpt(str(turn["text"])),
                    "verdict": record.get("verdict"),
                    "score": record.get("score"),
                    "action": record.get("action"),
                    "execution_status": record.get("execution_status"),
                    "deleted": deleted,
                    "failures": turn_failures,
                }
                turns.append(evidence_turn)
                if turn_failures:
                    failures.append({"turn": turn_number, "failures": turn_failures})
                if args.turn_delay_seconds > 0:
                    await asyncio.sleep(args.turn_delay_seconds)
    finally:
        await bot.close()
        bot_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await bot_task

    evidence = {
        "ok": not failures,
        "mode": args.mode,
        "dry_run": not args.enforce,
        "guild_id": guild_id,
        "channel_id": channel_id,
        "actor_count": len(actor_ids),
        "actor_bot_ids": actor_ids,
        "turn_count": len(turns),
        "turns": turns,
        "failures": failures,
        "audit_path": str(audit_path),
        "config_path": str(config_path),
    }
    if failures:
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
