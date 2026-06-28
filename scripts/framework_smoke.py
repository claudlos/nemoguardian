"""End-to-end framework smoke check for host + platform adapters.

This script is intended for the GPU demo host after the API service is running.
It verifies the running service plus credential-safe Discord/Twitch/webhook
adapter behavior. Real Discord/Twitch network connections are optional because
tokens must stay out of the public repo and logs.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PII_EXAMPLE = "Hey @everyone, drop your SSN and I will send $100 to whoever DMs me first"
DEFAULT_POLICY = "block PII and financial scams"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=os.environ.get("NEMOGUARDIAN_DEMO_BASE_URL", "http://localhost:8000"),
        help="Base URL for the running nemoguardian server",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("NEMOGUARDIAN_API_KEY", ""),
        help="API key for authenticated /v1/moderate checks",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail if /health does not report CUDA/GPU runtime",
    )
    parser.add_argument(
        "--require-triage",
        action="store_true",
        help="Fail if /health does not report configured NVIDIA/OpenRouter triage",
    )
    parser.add_argument(
        "--moderate",
        action="store_true",
        help="Call /demo/moderate in standard mode; this may load local 4B weights",
    )
    parser.add_argument(
        "--live-webhook",
        action="store_true",
        help="Exercise webhook adapter against the live /v1/moderate endpoint",
    )
    parser.add_argument(
        "--require-api-key",
        action="store_true",
        help="Fail if no API key is available for live adapter checks",
    )
    parser.add_argument(
        "--require-discord-token",
        action="store_true",
        help="Fail if DISCORD_BOT_TOKEN is not present for a live bot run",
    )
    parser.add_argument(
        "--require-twitch-token",
        action="store_true",
        help="Fail if TWITCH_TOKEN is not present for a live bot run",
    )
    parser.add_argument("--output", type=Path, help="Write evidence JSON to this path")
    args = parser.parse_args()

    evidence: dict[str, Any] = {
        "tool": "framework_smoke",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_url": args.base_url.rstrip("/"),
        "requirements": {
            "require_gpu": args.require_gpu,
            "require_triage": args.require_triage,
            "moderate": args.moderate,
            "live_webhook": args.live_webhook,
            "require_api_key": args.require_api_key,
            "require_discord_token": args.require_discord_token,
            "require_twitch_token": args.require_twitch_token,
        },
        "checks": [],
    }

    _check_host(
        args.base_url,
        evidence,
        require_gpu=args.require_gpu,
        require_triage=args.require_triage,
    )
    _check_token_presence(evidence, name="DISCORD_BOT_TOKEN", required=args.require_discord_token)
    _check_token_presence(evidence, name="TWITCH_TOKEN", required=args.require_twitch_token)

    if args.require_api_key:
        _add(
            evidence,
            "api_key_present",
            bool(args.api_key),
            "API key available" if args.api_key else "missing API key",
        )

    asyncio.run(_check_adapters(args, evidence))

    passed = all(check["ok"] for check in evidence["checks"])
    evidence["passed"] = passed

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")

    for check in evidence["checks"]:
        state = "PASS" if check["ok"] else "FAIL"
        print(f"{state} {check['name']}: {check['detail']}")

    if args.output:
        print(f"evidence: {args.output}")

    return 0 if passed else 1


def _check_host(
    base_url: str,
    evidence: dict[str, Any],
    *,
    require_gpu: bool,
    require_triage: bool,
) -> None:
    status, body = _request(base_url, "/health")
    health = _json(body)
    health_ok = status == 200 and isinstance(health, dict)
    _add(evidence, "health_reachable", health_ok, f"HTTP {status}")
    if not health_ok:
        return

    runtime_device = str(health.get("runtime_device", ""))
    gpu_ok = "cuda" in runtime_device.lower() or bool(health.get("gpu_available"))
    if require_gpu:
        _add(evidence, "runtime_gpu", gpu_ok, f"runtime_device={runtime_device!r}")

    triage_ok = bool(health.get("triage_configured")) and health.get("triage_provider") in {
        "nvidia",
        "openrouter",
    }
    if require_triage:
        _add(evidence, "triage_configured", triage_ok, f"provider={health.get('triage_provider')!r}")


async def _check_adapters(args: argparse.Namespace, evidence: dict[str, Any]) -> None:
    from nemoguardian.adapters import discord, twitch, webhook
    from nemoguardian.schemas import VerdictLabel

    await _check_discord_adapter(discord, VerdictLabel, evidence)
    await _check_twitch_adapter(twitch, VerdictLabel, evidence)
    await _check_webhook_adapter_contract(webhook, evidence)

    if args.moderate:
        _check_demo_moderation(args.base_url, evidence)
    if args.live_webhook:
        await _check_live_webhook(webhook, args.base_url, args.api_key, evidence)


async def _check_discord_adapter(discord: Any, verdict_label: Any, evidence: dict[str, Any]) -> None:
    unsafe_cascade = _FakeCascade(verdict_label.UNSAFE, categories=["pii"])
    unsafe_message = _FakeDiscordMessage("drop your SSN")
    await discord.make_handler(unsafe_cascade)(unsafe_message)
    _add(
        evidence,
        "discord_unsafe_delete",
        unsafe_message.deleted and bool(unsafe_message.channel.messages),
        "unsafe message deleted and warning sent",
        {"cascade_calls": len(unsafe_cascade.calls)},
    )

    controversial_cascade = _FakeCascade(verdict_label.CONTROVERSIAL)
    controversial_message = _FakeDiscordMessage("borderline message")
    await discord.make_handler(controversial_cascade)(controversial_message)
    _add(
        evidence,
        "discord_controversial_reaction",
        controversial_message.reactions == [discord.WARNING_REACTION],
        "controversial message receives warning reaction",
    )

    bot_cascade = _FakeCascade(verdict_label.UNSAFE)
    bot_message = _FakeDiscordMessage("bot echo", bot=True)
    await discord.make_handler(bot_cascade)(bot_message)
    _add(
        evidence,
        "discord_bot_ignored",
        not bot_cascade.calls and not bot_message.deleted,
        "bot-authored message skipped",
    )


async def _check_twitch_adapter(twitch: Any, verdict_label: Any, evidence: dict[str, Any]) -> None:
    cascade = _FakeCascade(verdict_label.UNSAFE)
    emitted: list[str] = []
    action = await twitch.make_moderator(cascade, emit=emitted.append)("drop your SSN")
    _add(
        evidence,
        "twitch_delete_action",
        action == "delete" and bool(cascade.calls) and bool(emitted),
        f"action={action!r}",
        {"mode": cascade.calls[0]["mode"] if cascade.calls else None},
    )


async def _check_webhook_adapter_contract(webhook: Any, evidence: dict[str, Any]) -> None:
    client = _FakeWebhookClient()
    verdict = await webhook.moderate_and_forward(
        "drop your SSN",
        forward_url="http://forward.test/hook",
        policy=DEFAULT_POLICY,
        policy_preset="discord",
        api_key="nmg_framework_smoke",
        client=client,
    )
    first_post = client.posts[0] if client.posts else {}
    second_post = client.posts[1] if len(client.posts) > 1 else {}
    _add(
        evidence,
        "webhook_auth_header",
        first_post.get("headers", {}).get("Authorization") == "Bearer nmg_framework_smoke",
        "Authorization bearer header sent to moderator",
    )
    _add(
        evidence,
        "webhook_forward_payload",
        verdict.get("verdict") == "unsafe" and second_post.get("json", {}).get("verdict") == verdict,
        "moderation verdict forwarded to webhook target",
    )


def _check_demo_moderation(base_url: str, evidence: dict[str, Any]) -> None:
    status, body = _request(
        base_url,
        "/demo/moderate",
        method="POST",
        params={"policy_preset": "discord"},
        json_body={"text": PII_EXAMPLE, "policy": DEFAULT_POLICY, "mode": "standard"},
        timeout=300,
    )
    data = _json(body)
    verdict = data.get("verdict") if isinstance(data, dict) else None
    ok = status == 200 and verdict == "unsafe"
    _add(evidence, "demo_moderate_4b", ok, f"HTTP {status}, verdict={verdict!r}")
    if isinstance(data, dict):
        model_errors = {
            name: verdict.get("error")
            for name, verdict in data.get("model_verdicts", {}).items()
            if isinstance(verdict, dict) and verdict.get("error")
        }
        _add(
            evidence,
            "demo_moderate_model_errors",
            not model_errors,
            "no model errors" if not model_errors else f"errors={model_errors}",
        )


async def _check_live_webhook(
    webhook: Any,
    base_url: str,
    api_key: str,
    evidence: dict[str, Any],
) -> None:
    if not api_key:
        _add(evidence, "live_webhook", False, "missing API key")
        return

    server, payloads = _start_forward_receiver()
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        forward_url = f"http://127.0.0.1:{server.server_port}/hook"
        verdict = await webhook.moderate_and_forward(
            PII_EXAMPLE,
            forward_url=forward_url,
            policy=DEFAULT_POLICY,
            policy_preset="discord",
            mode="standard",
            moderator_url=base_url.rstrip("/"),
            api_key=api_key,
        )
    except Exception as exc:
        _add(evidence, "live_webhook", False, f"{type(exc).__name__}: {exc}")
    else:
        _add(
            evidence,
            "live_webhook",
            verdict.get("verdict") == "unsafe" and bool(payloads),
            f"verdict={verdict.get('verdict')!r}, forwarded={bool(payloads)}",
        )
    finally:
        server.shutdown()
        server.server_close()


def _check_token_presence(evidence: dict[str, Any], *, name: str, required: bool) -> None:
    present = bool(os.environ.get(name, "").strip())
    if required:
        _add(evidence, name.lower() + "_present", present, "present" if present else "missing")
    else:
        _add(
            evidence,
            name.lower() + "_optional",
            True,
            "present" if present else "not set; live bot connection skipped",
        )


def _request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    params: dict[str, str] | None = None,
    json_body: dict[str, Any] | None = None,
    timeout: float = 15,
) -> tuple[int, str]:
    url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    if params:
        url = f"{url}?{urlencode(params)}"

    headers = {"Accept": "application/json"}
    data = None
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except (ConnectionError, OSError, URLError) as exc:
        return 0, str(exc)


def _start_forward_receiver() -> tuple[ThreadingHTTPServer, list[dict[str, Any]]]:
    payloads: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                payloads.append(json.loads(raw.decode("utf-8")))
            except json.JSONDecodeError:
                payloads.append({"raw": raw.decode("utf-8", errors="replace")})
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format: str, *args: Any) -> None:
            return

    return ThreadingHTTPServer(("127.0.0.1", 0), Handler), payloads


class _FakeCascade:
    def __init__(self, verdict: Any, *, categories: list[str] | None = None) -> None:
        self.verdict = verdict
        self.categories = categories or []
        self.calls: list[dict[str, Any]] = []

    def moderate(self, request: Any, *, policy_engine: Any = None) -> Any:
        self.calls.append(
            {
                "text": request.text,
                "mode": request.mode.value,
                "policy_engine": policy_engine is not None,
            }
        )
        return SimpleNamespace(
            verdict=self.verdict,
            categories=self.categories,
            score=0.95,
            mode=request.mode,
            matched_policy_rule="framework-smoke" if self.categories else None,
            request_id="framework-smoke",
            total_latency_ms=1.0,
        )


class _FakeDiscordGuild:
    id = 123
    name = "Framework Smoke Guild"

    def get_channel(self, channel_id: int) -> None:
        return None


class _FakeDiscordAuthor:
    id = 42
    mention = "@tester"

    def __init__(self, *, bot: bool) -> None:
        self.bot = bot
        self.roles: list[Any] = []

    def __str__(self) -> str:
        return "tester"


class _FakeDiscordMessage:
    def __init__(self, content: str, *, bot: bool = False) -> None:
        self.id = 789
        self.content = content
        self.author = _FakeDiscordAuthor(bot=bot)
        self.channel = SimpleNamespace(id=456, messages=[])
        self.guild = _FakeDiscordGuild()
        self.deleted = False
        self.reactions: list[str] = []
        self.jump_url = "https://discord.test/message/789"

        async def send(message: str) -> None:
            self.channel.messages.append(message)

        self.channel.send = send

    async def delete(self) -> None:
        self.deleted = True

    async def add_reaction(self, reaction: str) -> None:
        self.reactions.append(reaction)


class _FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return

    def json(self) -> dict[str, Any]:
        return self.payload


class _FakeWebhookClient:
    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []

    async def post(self, url: str, **kwargs: Any) -> _FakeHTTPResponse:
        self.posts.append({"url": url, **kwargs})
        if url.endswith("/v1/moderate"):
            return _FakeHTTPResponse({"verdict": "unsafe", "score": 0.95})
        return _FakeHTTPResponse({"ok": True})


def _json(body: str) -> Any:
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def _add(
    evidence: dict[str, Any],
    name: str,
    ok: bool,
    detail: str,
    data: Any | None = None,
) -> None:
    item: dict[str, Any] = {"name": name, "ok": ok, "detail": detail}
    if data is not None:
        item["data"] = data
    evidence["checks"].append(item)


if __name__ == "__main__":
    raise SystemExit(main())
