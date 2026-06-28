"""Tests for platform adapter glue."""

from __future__ import annotations

from typing import Any

from nemoguardian.adapters import discord, twitch, webhook
from nemoguardian.bot import AuditLog, BotConfig, ConfigStore, Platform
from nemoguardian.schemas import Mode, ModerateResponse, VerdictLabel


class FakeCascade:
    def __init__(self, verdict: VerdictLabel, *, categories: list[str] | None = None) -> None:
        self.verdict = verdict
        self.categories = categories or []
        self.calls: list[dict[str, Any]] = []

    def moderate(self, request, *, policy_engine=None):
        self.calls.append(
            {
                "text": request.text,
                "mode": request.mode.value,
                "policy": request.policy,
                "policy_engine": policy_engine is not None,
            }
        )
        return ModerateResponse(
            verdict=self.verdict,
            score=0.9,
            reasons=["fake"],
            categories=self.categories,
            matched_policy_rule="fake-rule" if self.categories else None,
            model_verdicts={},
            total_latency_ms=1.0,
            mode=request.mode,
            request_id="req-test",
        )


class FakeChannel:
    def __init__(self, channel_id: int = 456) -> None:
        self.id = channel_id
        self.messages: list[str] = []

    async def send(self, message: str) -> None:
        self.messages.append(message)


class FakeGuild:
    def __init__(self) -> None:
        self.id = 123
        self.name = "Test Guild"
        self.log_channel = FakeChannel(999)

    def get_channel(self, channel_id: int):
        return self.log_channel if channel_id == self.log_channel.id else None


class FakeAuthor:
    id = 42
    bot = False
    mention = "@tester"

    def __init__(self) -> None:
        self.roles: list[Any] = []

    def __str__(self) -> str:
        return "tester"


class FakeDiscordMessage:
    def __init__(self, content: str, *, bot: bool = False) -> None:
        self.id = 789
        self.content = content
        self.author = FakeAuthor()
        self.author.bot = bot
        self.channel = FakeChannel()
        self.guild = FakeGuild()
        self.deleted = False
        self.reactions: list[str] = []
        self.jump_url = "https://discord.test/message/789"

    async def delete(self) -> None:
        self.deleted = True

    async def add_reaction(self, reaction: str) -> None:
        self.reactions.append(reaction)


def _stores(tmp_path):
    return (
        ConfigStore(tmp_path / "bot-config.json"),
        AuditLog(tmp_path / "bot-audit.jsonl"),
    )


async def test_discord_adapter_deletes_unsafe_message(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    cascade = FakeCascade(VerdictLabel.UNSAFE, categories=["PII"])
    message = FakeDiscordMessage("drop your SSN")

    await discord.make_handler(cascade, config_store=config_store, audit_log=audit_log)(message)

    assert message.deleted is True
    assert message.channel.messages
    assert "@tester" in message.channel.messages[0]
    assert "PII" in message.channel.messages[0]
    assert cascade.calls[0]["mode"] == "standard"
    records = audit_log.recent()
    assert records[0]["action"] == "delete"
    assert records[0]["execution_status"] == "delete+public-warning"


async def test_discord_adapter_sends_mod_log_when_configured(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    config = BotConfig.default(Platform.DISCORD, "123")
    config.log_channel_id = "999"
    config_store.save(config)
    message = FakeDiscordMessage("drop your SSN")

    await discord.make_handler(
        FakeCascade(VerdictLabel.UNSAFE, categories=["PII"]),
        config_store=config_store,
        audit_log=audit_log,
    )(message)

    assert message.guild.log_channel.messages
    assert "nemoguardian moderation" in message.guild.log_channel.messages[0]
    assert "verdict: `unsafe`" in message.guild.log_channel.messages[0]


async def test_discord_adapter_reacts_to_controversial_message(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    message = FakeDiscordMessage("borderline")

    await discord.make_handler(
        FakeCascade(VerdictLabel.CONTROVERSIAL),
        config_store=config_store,
        audit_log=audit_log,
    )(message)

    assert message.deleted is False
    assert message.reactions == [discord.WARNING_REACTION]
    assert audit_log.recent()[0]["action"] == "flag"


async def test_discord_adapter_ignores_bot_messages(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    cascade = FakeCascade(VerdictLabel.UNSAFE)
    message = FakeDiscordMessage("bot echo", bot=True)

    await discord.make_handler(cascade, config_store=config_store, audit_log=audit_log)(message)

    assert cascade.calls == []
    assert message.deleted is False


async def test_twitch_adapter_returns_delete_action(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    cascade = FakeCascade(VerdictLabel.UNSAFE)
    emitted: list[str] = []

    action = await twitch.make_moderator(
        cascade,
        config_store=config_store,
        audit_log=audit_log,
        emit=emitted.append,
    )("drop your SSN")

    assert action == "delete"
    assert emitted
    assert cascade.calls[0]["mode"] == "fast"
    assert audit_log.recent()[0]["platform"] == "twitch"


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeHTTPClient:
    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []

    async def post(self, url: str, **kwargs: Any) -> FakeHTTPResponse:
        self.posts.append({"url": url, **kwargs})
        if url.endswith("/v1/moderate"):
            return FakeHTTPResponse({"verdict": "unsafe", "score": 0.9})
        return FakeHTTPResponse({"ok": True})


async def test_webhook_adapter_sends_env_api_key(monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_API_KEY", "nmg_env_key")
    client = FakeHTTPClient()

    verdict = await webhook.moderate_and_forward(
        "drop your SSN",
        forward_url="http://forward.test/hook",
        moderator_url="http://moderator.test",
        policy_preset="discord",
        client=client,
    )

    assert verdict["verdict"] == "unsafe"
    assert client.posts[0]["headers"] == {"Authorization": "Bearer nmg_env_key"}
    assert client.posts[0]["params"] == {"policy_preset": "discord"}
    assert client.posts[1]["json"] == {"text": "drop your SSN", "verdict": verdict}


def test_config_store_round_trips_platform_defaults(tmp_path):
    store = ConfigStore(tmp_path / "config.json")
    discord_config = store.get(Platform.DISCORD, "guild-1")
    twitch_config = store.get(Platform.TWITCH, "channel-1")

    assert discord_config.mode == Mode.STANDARD
    assert twitch_config.mode == Mode.FAST
    assert twitch_config.policy_preset == "twitch"

    discord_config.ignored_channel_ids.add("123")
    store.save(discord_config)

    assert store.get(Platform.DISCORD, "guild-1").ignored_channel_ids == {"123"}
