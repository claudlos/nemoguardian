"""Tests for platform adapter glue."""

from __future__ import annotations

import asyncio
import runpy
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from nemoguardian.adapters import discord, twitch, webhook
from nemoguardian.bot import (
    AuditLog,
    AuditRecord,
    BotConfig,
    ConfigStore,
    ModerationAction,
    ModerationEvaluation,
    ModerationPlan,
    Platform,
    since_hours_ago,
)
from nemoguardian.bot.audit import text_hash
from nemoguardian.schemas import Mode, ModerateResponse, VerdictLabel


class FakeCascade:
    def __init__(
        self,
        verdict: VerdictLabel,
        *,
        categories: list[str] | None = None,
        matched_policy_rule: str | None = None,
    ) -> None:
        self.verdict = verdict
        self.categories = categories or []
        self.matched_policy_rule = matched_policy_rule
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
            matched_policy_rule=self.matched_policy_rule or ("fake-rule" if self.categories else None),
            model_verdicts={},
            total_latency_ms=1.0,
            mode=request.mode,
            request_id="req-test",
        )


class FakeChannel:
    def __init__(self, channel_id: int = 456) -> None:
        self.id = channel_id
        self.messages: list[str] = []
        self.fail_send = False

    async def send(self, message: str) -> None:
        if self.fail_send:
            raise RuntimeError("send failed")
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
        self.dms: list[str] = []
        self.timeout_calls: list[dict[str, Any]] = []
        self.fail_dm = False
        self.fail_timeout = False

    def __str__(self) -> str:
        return "tester"

    async def send(self, message: str) -> None:
        if self.fail_dm:
            raise RuntimeError("dm failed")
        self.dms.append(message)

    async def timeout(self, until, *, reason: str) -> None:
        if self.fail_timeout:
            raise RuntimeError("timeout failed")
        self.timeout_calls.append({"until": until, "reason": reason})


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


def _evaluation(
    message: FakeDiscordMessage,
    *,
    config: BotConfig | None = None,
    verdict: VerdictLabel = VerdictLabel.UNSAFE,
    action: ModerationAction = ModerationAction.DELETE,
    categories: list[str] | None = None,
    result: ModerateResponse | None | bool = True,
    **plan_changes: Any,
) -> ModerationEvaluation:
    config = config or BotConfig.default(Platform.DISCORD, str(message.guild.id))
    context = discord._context_from_message(message)
    response = None
    if result is not None and result is not False:
        response = ModerateResponse(
            verdict=verdict,
            score=0.9,
            reasons=["fake"],
            categories=categories or ["PII"],
            matched_policy_rule="fake-rule",
            model_verdicts={},
            total_latency_ms=123.0,
            mode=config.mode,
            request_id="req-test",
        )
    plan = ModerationPlan(action=action, reason=", ".join(categories or ["PII"]))
    for key, value in plan_changes.items():
        setattr(plan, key, value)
    return ModerationEvaluation(context=context, config=config, result=response, plan=plan)


class FakeResponse:
    def __init__(self) -> None:
        self.messages: list[dict[str, Any]] = []

    async def send_message(self, text: str, *, ephemeral: bool) -> None:
        self.messages.append({"text": text, "ephemeral": ephemeral})


class FakeInteraction:
    def __init__(self, *, manage_guild: bool = True) -> None:
        self.guild_id = 123
        self.channel_id = 456
        self.id = 999
        self.user = SimpleNamespace(
            id=42,
            roles=[],
            guild_permissions=SimpleNamespace(manage_guild=manage_guild),
            __str__=lambda: "tester",
        )
        self.app_permissions = SimpleNamespace(
            view_channel=True,
            read_message_history=True,
            send_messages=True,
            manage_messages=True,
            embed_links=True,
            moderate_members=True,
        )
        self.response = FakeResponse()


def _install_fake_discord(monkeypatch: pytest.MonkeyPatch):
    discord_module = ModuleType("discord")
    app_commands_module = ModuleType("discord.app_commands")
    ext_module = ModuleType("discord.ext")
    commands_module = ModuleType("discord.ext.commands")

    class FakeIntents:
        def __init__(self) -> None:
            self.guilds = False
            self.message_content = False

        @classmethod
        def default(cls) -> FakeIntents:
            return cls()

    class FakeObject:
        def __init__(self, *, id: int) -> None:
            self.id = id

    class FakePermissions:
        def __init__(self, **permissions: bool) -> None:
            self.__dict__.update(permissions)

    class FakeChoice:
        def __init__(self, *, name: str, value: str) -> None:
            self.name = name
            self.value = value

        @classmethod
        def __class_getitem__(cls, _item):
            return cls

    class FakeGroup:
        def __init__(self, *, name: str, description: str, default_permissions=None) -> None:
            self.name = name
            self.description = description
            self.default_permissions = default_permissions
            self.commands: dict[str, Any] = {}

        def command(self, *, name: str, description: str):
            def decorator(func):
                self.commands[name] = func
                return func

            return decorator

    class FakeTree:
        def __init__(self) -> None:
            self.group: FakeGroup | None = None
            self.copy_targets: list[Any] = []
            self.sync_targets: list[Any] = []

        def add_command(self, group: FakeGroup) -> None:
            self.group = group

        def copy_global_to(self, *, guild) -> None:
            self.copy_targets.append(guild)

        async def sync(self, *, guild=None) -> None:
            self.sync_targets.append(guild)

    class FakeBot:
        def __init__(self, *, command_prefix: str, intents: FakeIntents) -> None:
            self.command_prefix = command_prefix
            self.intents = intents
            self.tree = FakeTree()
            self.events: dict[str, Any] = {}
            self.user = "fake-bot"
            self.ran_token: str | None = None

        def event(self, func):
            self.events[func.__name__] = func
            return func

        def run(self, token: str) -> None:
            self.ran_token = token

    def passthrough_decorator(**_kwargs):
        def decorator(func):
            return func

        return decorator

    def choices(**_kwargs):
        def decorator(func):
            return func

        return decorator

    discord_module.Intents = FakeIntents
    discord_module.Object = FakeObject
    discord_module.Permissions = FakePermissions
    discord_module.TextChannel = SimpleNamespace
    discord_module.Role = SimpleNamespace
    discord_module.Member = SimpleNamespace
    discord_module.app_commands = app_commands_module
    app_commands_module.Group = FakeGroup
    app_commands_module.Choice = FakeChoice
    app_commands_module.default_permissions = passthrough_decorator
    app_commands_module.choices = choices
    commands_module.Bot = FakeBot
    ext_module.commands = commands_module

    monkeypatch.setitem(sys.modules, "discord", discord_module)
    monkeypatch.setitem(sys.modules, "discord.app_commands", app_commands_module)
    monkeypatch.setitem(sys.modules, "discord.ext", ext_module)
    monkeypatch.setitem(sys.modules, "discord.ext.commands", commands_module)
    return FakeBot


def _install_fake_twitch(monkeypatch: pytest.MonkeyPatch, events_to_run: list[Any]):
    twitchio_module = ModuleType("twitchio")
    ext_module = ModuleType("twitchio.ext")
    commands_module = ModuleType("twitchio.ext.commands")
    bots: list[Any] = []

    class FakeTwitchBot:
        def __init__(self, *, token: str, prefix: str, initial_channels: list[str]) -> None:
            self.token = token
            self.prefix = prefix
            self.initial_channels = initial_channels
            self.events: dict[str, Any] = {}
            self.run_called = False
            bots.append(self)

        def event(self, func):
            self.events[func.__name__] = func
            return func

        def run(self) -> None:
            self.run_called = True
            for message in events_to_run:
                asyncio.run(self.events["event_message"](message))

    commands_module.Bot = FakeTwitchBot
    ext_module.commands = commands_module
    twitchio_module.ext = ext_module
    monkeypatch.setitem(sys.modules, "twitchio", twitchio_module)
    monkeypatch.setitem(sys.modules, "twitchio.ext", ext_module)
    monkeypatch.setitem(sys.modules, "twitchio.ext.commands", commands_module)
    return bots


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


async def test_discord_mod_log_and_audit_redact_sensitive_evidence(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    config = BotConfig.default(Platform.DISCORD, "123")
    config.log_channel_id = "999"
    config_store.save(config)
    message = FakeDiscordMessage(
        "email jane@example.com SSN 123-45-6789 phone 555-123-4567 card 4111 1111 1111 1111"
    )

    await discord.make_handler(
        FakeCascade(VerdictLabel.UNSAFE, categories=["PII"]),
        config_store=config_store,
        audit_log=audit_log,
    )(message)

    mod_log = message.guild.log_channel.messages[0]
    record = audit_log.recent()[0]
    for value in ("jane@example.com", "123-45-6789", "555-123-4567", "4111 1111 1111 1111"):
        assert value not in mod_log
        assert value not in record["text_excerpt"]
    for marker in ("[email]", "[ssn]", "[phone]", "[payment-card]"):
        assert marker in mod_log
        assert marker in record["text_excerpt"]
    assert record["text_sha256"] == text_hash(message.content)
    assert record["details"]["text_redacted"] is True


async def test_discord_adapter_ignores_messages_without_guild(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    cascade = FakeCascade(VerdictLabel.UNSAFE)
    message = FakeDiscordMessage("dm content")
    message.guild = None

    await discord.make_handler(cascade, config_store=config_store, audit_log=audit_log)(message)

    assert cascade.calls == []
    assert message.deleted is False
    assert audit_log.recent() == []


async def test_discord_apply_actions_allows_without_side_effects():
    message = FakeDiscordMessage("normal chat")
    config = BotConfig.default(Platform.DISCORD, "123")
    config.log_channel_id = "999"
    evaluation = _evaluation(
        message,
        config=config,
        verdict=VerdictLabel.SAFE,
        action=ModerationAction.ALLOW,
        categories=[],
    )

    status, error = await discord.apply_discord_actions(message, evaluation)

    assert (status, error) == ("allowed", None)
    assert message.deleted is False
    assert message.guild.log_channel.messages == []


async def test_discord_apply_actions_dry_run_logs_without_enforcement():
    message = FakeDiscordMessage("drop your SSN")
    config = BotConfig.default(Platform.DISCORD, "123")
    config.log_channel_id = "999"
    config.dry_run = True
    evaluation = _evaluation(
        message,
        config=config,
        delete_message=True,
        public_warning=True,
    )

    status, error = await discord.apply_discord_actions(message, evaluation)

    assert (status, error) == ("dry-run", None)
    assert message.deleted is False
    assert message.channel.messages == []
    assert "applied: `dry-run`" in message.guild.log_channel.messages[0]


async def test_discord_apply_actions_timeout_and_dm_success():
    message = FakeDiscordMessage("serious violation")
    config = BotConfig.default(Platform.DISCORD, "123")
    config.timeout_seconds = 120
    config.dm_users = True
    evaluation = _evaluation(
        message,
        config=config,
        action=ModerationAction.TIMEOUT,
        timeout_user=True,
        notify_user=True,
    )

    status, error = await discord.apply_discord_actions(message, evaluation)

    assert (status, error) == ("timeout+dm", None)
    assert message.author.timeout_calls[0]["reason"] == "PII"
    assert message.author.dms == ["Your message in Test Guild was moderated: PII"]


async def test_discord_apply_actions_collects_partial_errors():
    message = FakeDiscordMessage("borderline")
    message.author.fail_dm = True
    evaluation = _evaluation(
        message,
        action=ModerationAction.DELETE,
        delete_message=True,
        add_reaction=True,
        notify_user=True,
        public_warning=False,
    )

    async def fail_delete() -> None:
        raise RuntimeError("delete failed")

    message.delete = fail_delete
    status, error = await discord.apply_discord_actions(message, evaluation)

    assert status == "partial"
    assert error == "delete:RuntimeError;dm:RuntimeError"
    assert message.reactions == [discord.WARNING_REACTION]


async def test_discord_apply_actions_reports_failed_when_every_action_errors():
    message = FakeDiscordMessage("unsafe")
    message.channel.fail_send = True

    async def fail_reaction(_reaction: str) -> None:
        raise RuntimeError("reaction failed")

    async def fail_delete() -> None:
        raise RuntimeError("delete failed")

    message.add_reaction = fail_reaction
    message.delete = fail_delete
    message.author.fail_timeout = True
    message.author.fail_dm = True
    evaluation = _evaluation(
        message,
        action=ModerationAction.TIMEOUT,
        add_reaction=True,
        delete_message=True,
        timeout_user=True,
        public_warning=True,
        notify_user=True,
    )

    status, error = await discord.apply_discord_actions(message, evaluation)

    assert status == "failed"
    assert error == (
        "reaction:RuntimeError;delete:RuntimeError;timeout:RuntimeError;"
        "public-warning:RuntimeError;dm:RuntimeError"
    )


async def test_discord_apply_actions_reports_unsupported_timeout():
    message = FakeDiscordMessage("unsafe")
    message.author = SimpleNamespace(id=42, bot=False, mention="@tester", roles=[])
    evaluation = _evaluation(
        message,
        action=ModerationAction.TIMEOUT,
        timeout_user=True,
        categories=["harassment"],
    )

    status, error = await discord.apply_discord_actions(message, evaluation)

    assert (status, error) == ("failed", "timeout:unsupported-author")


async def test_discord_audit_log_supports_case_lookup_and_history(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    message = FakeDiscordMessage("drop your SSN")

    await discord.make_handler(
        FakeCascade(VerdictLabel.UNSAFE, categories=["PII"]),
        config_store=config_store,
        audit_log=audit_log,
    )(message)

    record = audit_log.recent()[0]
    assert audit_log.find_case(record["case_id"])["message_id"] == "789"
    assert audit_log.find_case("missing-case") is None

    history = audit_log.history(Platform.DISCORD, "123", user_id="42", limit=5)
    assert history[0]["case_id"] == record["case_id"]
    assert audit_log.history(Platform.DISCORD, "missing") == []


async def test_discord_audit_summary_counts_recent_cases(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    unsafe_message = FakeDiscordMessage("drop your SSN")
    controversial_message = FakeDiscordMessage("borderline")
    controversial_message.id = 790
    controversial_message.channel = FakeChannel(888)

    await discord.make_handler(
        FakeCascade(VerdictLabel.UNSAFE, categories=["PII"], matched_policy_rule="block-pii"),
        config_store=config_store,
        audit_log=audit_log,
    )(unsafe_message)
    await discord.make_handler(
        FakeCascade(
            VerdictLabel.CONTROVERSIAL,
            categories=["harassment"],
            matched_policy_rule="watch-harassment",
        ),
        config_store=config_store,
        audit_log=audit_log,
    )(controversial_message)

    summary = audit_log.summary(Platform.DISCORD, "123", limit=10)
    text = discord._stats_text(summary)

    assert summary["total"] == 2
    assert summary["verdicts"] == {"controversial": 1, "unsafe": 1}
    assert summary["actions"] == {"delete": 1, "flag": 1}
    assert summary["categories"] == {"PII": 1, "harassment": 1}
    assert "total cases: `2`" in text
    assert "unsafe:1" in text
    assert "flag:1" in text
    assert "(last 24h)" in discord._stats_text(summary, since_hours=24)
    channel_history = audit_log.history(Platform.DISCORD, "123", channel_id="888", limit=5)
    channel_summary = audit_log.summary(Platform.DISCORD, "123", channel_id="888", limit=10)
    channel_text = discord._stats_text(channel_summary)
    assert [record["case_id"] for record in channel_history] == [audit_log.recent()[-1]["case_id"]]
    assert channel_summary["channel_id"] == "888"
    assert channel_summary["total"] == 1
    assert channel_summary["verdicts"] == {"controversial": 1}
    assert "channel <#888>" in channel_text
    category_history = audit_log.history(Platform.DISCORD, "123", category="harassment", limit=5)
    category_summary = audit_log.summary(Platform.DISCORD, "123", category="harassment", limit=10)
    category_text = discord._stats_text(category_summary)
    assert [record["case_id"] for record in category_history] == [audit_log.recent()[-1]["case_id"]]
    assert category_summary["category"] == "harassment"
    assert category_summary["total"] == 1
    assert category_summary["actions"] == {"flag": 1}
    assert "category `harassment`" in category_text
    rule_history = audit_log.history(Platform.DISCORD, "123", rule="watch-harassment", limit=5)
    rule_summary = audit_log.summary(Platform.DISCORD, "123", rule="watch-harassment", limit=10)
    rule_text = discord._stats_text(rule_summary)
    assert [record["case_id"] for record in rule_history] == [audit_log.recent()[-1]["case_id"]]
    assert rule_summary["rule"] == "watch-harassment"
    assert rule_summary["total"] == 1
    assert rule_summary["actions"] == {"flag": 1}
    assert "rule `watch-harassment`" in rule_text
    enforcement_history = audit_log.history(
        Platform.DISCORD,
        "123",
        action=ModerationAction.DELETE,
        verdict=VerdictLabel.UNSAFE,
        status="delete+public-warning",
        limit=5,
    )
    enforcement_summary = audit_log.summary(
        Platform.DISCORD,
        "123",
        action=ModerationAction.DELETE,
        verdict=VerdictLabel.UNSAFE,
        status="delete+public-warning",
        limit=10,
    )
    enforcement_text = discord._stats_text(enforcement_summary)
    assert [record["case_id"] for record in enforcement_history] == [audit_log.recent()[0]["case_id"]]
    assert enforcement_summary["action"] == "delete"
    assert enforcement_summary["verdict"] == "unsafe"
    assert enforcement_summary["status"] == "delete+public-warning"
    assert enforcement_summary["total"] == 1
    assert enforcement_summary["categories"] == {"PII": 1}
    assert "action `delete`" in enforcement_text
    assert "verdict `unsafe`" in enforcement_text
    assert "status `delete+public-warning`" in enforcement_text
    live_summary = audit_log.summary(Platform.DISCORD, "123", dry_run=False, limit=10)
    live_text = discord._stats_text(live_summary)
    assert live_summary["dry_run_filter"] is False
    assert live_summary["total"] == 2
    assert live_summary["dry_run"] == 0
    assert "dry run `False`" in live_text
    assert discord._stats_text(audit_log.summary(Platform.DISCORD, "missing")) == (
        "**nemoguardian stats**\nNo moderation cases found."
    )
    assert discord._stats_text(audit_log.summary(Platform.DISCORD, "missing"), since_hours=1) == (
        "**nemoguardian stats** (last 1h)\nNo moderation cases found."
    )


async def test_discord_audit_failures_filters_partial_and_failed_records(tmp_path):
    _, audit_log = _stores(tmp_path)
    audit_log.append(
        AuditRecord(
            case_id="discord-123-old-failure",
            platform=Platform.DISCORD,
            workspace_id="123",
            channel_id="456",
            message_id="1",
            user_id="42",
            username="tester",
            action=ModerationAction.DELETE,
            verdict=VerdictLabel.UNSAFE,
            score=0.91,
            mode=Mode.STANDARD,
            execution_status="failed",
            error="delete:Forbidden",
            created_at="2000-01-01T00:00:00+00:00",
        )
    )
    audit_log.append(
        AuditRecord(
            case_id="discord-123-partial",
            platform=Platform.DISCORD,
            workspace_id="123",
            channel_id="789",
            message_id="2",
            user_id="77",
            username="repeat",
            action=ModerationAction.TIMEOUT,
            verdict=VerdictLabel.UNSAFE,
            score=0.88,
            mode=Mode.STANDARD,
            execution_status="partial",
            error="timeout:Forbidden",
        )
    )

    records = audit_log.failures(Platform.DISCORD, "123", limit=5)
    text = discord._failures_text(records)
    windowed = audit_log.failures(Platform.DISCORD, "123", limit=5, since=since_hours_ago(1))

    assert [record["case_id"] for record in records] == [
        "discord-123-partial",
        "discord-123-old-failure",
    ]
    assert [record["case_id"] for record in windowed] == ["discord-123-partial"]
    assert "status `partial`" in text
    assert "error `timeout:Forbidden`" in text
    assert "(last 2h)" in discord._failures_text(records, since_hours=2)
    assert discord._failures_text([]) == (
        "**nemoguardian failures**\nNo failed moderation actions found."
    )
    assert discord._failures_text([], since_hours=2) == (
        "**nemoguardian failures** (last 2h)\nNo failed moderation actions found."
    )


async def test_discord_audit_dry_run_cases_filters_planned_actions(tmp_path):
    _, audit_log = _stores(tmp_path)
    audit_log.append(
        AuditRecord(
            case_id="discord-123-old-dry-run",
            platform=Platform.DISCORD,
            workspace_id="123",
            channel_id="456",
            message_id="1",
            user_id="42",
            username="tester",
            action=ModerationAction.DELETE,
            verdict=VerdictLabel.UNSAFE,
            score=0.91,
            mode=Mode.STANDARD,
            categories=["PII"],
            dry_run=True,
            execution_status="dry-run",
            created_at="2000-01-01T00:00:00+00:00",
        )
    )
    audit_log.append(
        AuditRecord(
            case_id="discord-123-current-dry-run",
            platform=Platform.DISCORD,
            workspace_id="123",
            channel_id="789",
            message_id="2",
            user_id="77",
            username="repeat",
            action=ModerationAction.TIMEOUT,
            verdict=VerdictLabel.UNSAFE,
            score=0.88,
            mode=Mode.STANDARD,
            categories=["harassment"],
            dry_run=True,
            execution_status="dry-run",
        )
    )
    audit_log.append(
        AuditRecord(
            case_id="discord-123-allowed-dry-run",
            platform=Platform.DISCORD,
            workspace_id="123",
            channel_id="789",
            message_id="3",
            user_id="77",
            username="repeat",
            action=ModerationAction.ALLOW,
            verdict=VerdictLabel.SAFE,
            score=0.01,
            mode=Mode.STANDARD,
            dry_run=True,
            execution_status="allowed",
        )
    )

    records = audit_log.dry_run_cases(Platform.DISCORD, "123", limit=5)
    text = discord._dry_run_cases_text(records)
    windowed = audit_log.dry_run_cases(Platform.DISCORD, "123", limit=5, since=since_hours_ago(1))
    dry_run_history = audit_log.history(Platform.DISCORD, "123", dry_run=True, limit=5)
    dry_run_summary = audit_log.summary(Platform.DISCORD, "123", dry_run=True, limit=10)
    dry_run_text = discord._stats_text(dry_run_summary)

    assert [record["case_id"] for record in records] == [
        "discord-123-current-dry-run",
        "discord-123-old-dry-run",
    ]
    assert [record["case_id"] for record in windowed] == ["discord-123-current-dry-run"]
    assert [record["case_id"] for record in dry_run_history] == [
        "discord-123-allowed-dry-run",
        "discord-123-current-dry-run",
        "discord-123-old-dry-run",
    ]
    assert dry_run_summary["dry_run_filter"] is True
    assert dry_run_summary["total"] == 3
    assert dry_run_summary["dry_run"] == 3
    assert "dry run `True`" in dry_run_text
    assert "timeout/unsafe" in text
    assert "categories `harassment`" in text
    assert "(last 2h)" in discord._dry_run_cases_text(records, since_hours=2)
    assert discord._dry_run_cases_text([]) == (
        "**nemoguardian dry-run cases**\nNo dry-run action cases found."
    )
    assert discord._dry_run_cases_text([], since_hours=2) == (
        "**nemoguardian dry-run cases** (last 2h)\nNo dry-run action cases found."
    )


async def test_discord_audit_top_errors_groups_recurring_execution_errors(tmp_path):
    _, audit_log = _stores(tmp_path)
    audit_log.append(
        AuditRecord(
            case_id="discord-123-delete-failure",
            platform=Platform.DISCORD,
            workspace_id="123",
            channel_id="456",
            message_id="1",
            user_id="42",
            username="tester",
            action=ModerationAction.DELETE,
            verdict=VerdictLabel.UNSAFE,
            score=0.91,
            mode=Mode.STANDARD,
            execution_status="failed",
            error="delete:Forbidden",
        )
    )
    audit_log.append(
        AuditRecord(
            case_id="discord-123-partial",
            platform=Platform.DISCORD,
            workspace_id="123",
            channel_id="789",
            message_id="2",
            user_id="77",
            username="repeat",
            action=ModerationAction.TIMEOUT,
            verdict=VerdictLabel.UNSAFE,
            score=0.88,
            mode=Mode.STANDARD,
            execution_status="partial",
            error="delete:Forbidden;timeout:Forbidden",
        )
    )

    rows = audit_log.top_errors(Platform.DISCORD, "123", limit=5, case_limit=10)
    text = discord._errors_text(rows, case_limit=10)

    assert rows[0]["error"] == "delete:Forbidden"
    assert rows[0]["total"] == 2
    assert rows[0]["failed"] == 1
    assert rows[0]["partial"] == 1
    assert rows[0]["actions"] == {"delete": 1, "timeout": 1}
    assert rows[1]["error"] == "timeout:Forbidden"
    assert "error `delete:Forbidden`" in text
    assert "channels `456:1, 789:1`" in text
    assert "(last 2h)" in discord._errors_text(rows, case_limit=10, since_hours=2)
    assert discord._errors_text([], case_limit=10) == (
        "**nemoguardian errors**\nNo moderation errors found."
    )
    assert discord._errors_text([], case_limit=10, since_hours=2) == (
        "**nemoguardian errors** (last 2h)\nNo moderation errors found."
    )


async def test_discord_audit_slow_cases_orders_by_latency(tmp_path):
    _, audit_log = _stores(tmp_path)
    audit_log.append(
        AuditRecord(
            case_id="discord-123-old-slow",
            platform=Platform.DISCORD,
            workspace_id="123",
            channel_id="456",
            message_id="1",
            user_id="42",
            username="tester",
            action=ModerationAction.DELETE,
            verdict=VerdictLabel.UNSAFE,
            score=0.91,
            mode=Mode.STANDARD,
            latency_ms=1500.0,
            execution_status="delete",
            created_at="2000-01-01T00:00:00+00:00",
        )
    )
    audit_log.append(
        AuditRecord(
            case_id="discord-123-current-slow",
            platform=Platform.DISCORD,
            workspace_id="123",
            channel_id="789",
            message_id="2",
            user_id="77",
            username="repeat",
            action=ModerationAction.FLAG,
            verdict=VerdictLabel.CONTROVERSIAL,
            score=0.72,
            mode=Mode.STANDARD,
            latency_ms=900.0,
            execution_status="reaction",
        )
    )
    audit_log.append(
        AuditRecord(
            case_id="discord-123-current-fast",
            platform=Platform.DISCORD,
            workspace_id="123",
            channel_id="789",
            message_id="3",
            user_id="77",
            username="repeat",
            action=ModerationAction.ALLOW,
            verdict=VerdictLabel.SAFE,
            score=0.01,
            mode=Mode.STANDARD,
            latency_ms=120.0,
            execution_status="allowed",
        )
    )
    audit_log.append(
        AuditRecord(
            case_id="discord-123-missing-latency",
            platform=Platform.DISCORD,
            workspace_id="123",
            channel_id="789",
            message_id="4",
            user_id="77",
            username="repeat",
            action=ModerationAction.ALLOW,
            verdict=VerdictLabel.SAFE,
            score=0.01,
            mode=Mode.STANDARD,
            execution_status="allowed",
        )
    )

    rows = audit_log.slow_cases(Platform.DISCORD, "123", limit=3, case_limit=10)
    windowed = audit_log.slow_cases(Platform.DISCORD, "123", limit=3, case_limit=10, since=since_hours_ago(1))
    text = discord._slow_cases_text(rows, case_limit=10)

    assert [record["case_id"] for record in rows] == [
        "discord-123-old-slow",
        "discord-123-current-slow",
        "discord-123-current-fast",
    ]
    assert [record["case_id"] for record in windowed] == [
        "discord-123-current-slow",
        "discord-123-current-fast",
    ]
    assert "latency `1.50s`" in text
    assert "latency `900ms`" in text
    assert "(last 2h)" in discord._slow_cases_text(rows, case_limit=10, since_hours=2)
    assert discord._slow_cases_text([], case_limit=10) == (
        "**nemoguardian slow cases**\nNo latency data found."
    )
    assert discord._slow_cases_text([], case_limit=10, since_hours=2) == (
        "**nemoguardian slow cases** (last 2h)\nNo latency data found."
    )


async def test_discord_audit_high_scores_orders_by_score(tmp_path):
    _, audit_log = _stores(tmp_path)
    audit_log.append(
        AuditRecord(
            case_id="discord-123-old-high-score",
            platform=Platform.DISCORD,
            workspace_id="123",
            channel_id="456",
            message_id="1",
            user_id="42",
            username="tester",
            action=ModerationAction.DELETE,
            verdict=VerdictLabel.UNSAFE,
            score=0.98,
            mode=Mode.STANDARD,
            categories=["PII"],
            execution_status="delete",
            created_at="2000-01-01T00:00:00+00:00",
        )
    )
    audit_log.append(
        AuditRecord(
            case_id="discord-123-current-high-score",
            platform=Platform.DISCORD,
            workspace_id="123",
            channel_id="789",
            message_id="2",
            user_id="77",
            username="repeat",
            action=ModerationAction.FLAG,
            verdict=VerdictLabel.CONTROVERSIAL,
            score=0.88,
            mode=Mode.STANDARD,
            categories=["harassment"],
            execution_status="reaction",
        )
    )
    audit_log.append(
        AuditRecord(
            case_id="discord-123-current-low-score",
            platform=Platform.DISCORD,
            workspace_id="123",
            channel_id="789",
            message_id="3",
            user_id="77",
            username="repeat",
            action=ModerationAction.ALLOW,
            verdict=VerdictLabel.SAFE,
            score=0.03,
            mode=Mode.STANDARD,
            execution_status="allowed",
        )
    )

    rows = audit_log.high_score_cases(Platform.DISCORD, "123", limit=3, case_limit=10)
    windowed = audit_log.high_score_cases(
        Platform.DISCORD,
        "123",
        limit=3,
        case_limit=10,
        since=since_hours_ago(1),
    )

    assert [record["case_id"] for record in rows] == [
        "discord-123-old-high-score",
        "discord-123-current-high-score",
        "discord-123-current-low-score",
    ]
    assert [record["case_id"] for record in windowed] == [
        "discord-123-current-high-score",
        "discord-123-current-low-score",
    ]
    assert rows[0]["score"] == 0.98
    assert rows[0]["categories"] == ["PII"]


async def test_discord_build_bot_registers_slash_commands():
    pytest.importorskip("discord")

    bot = discord.build_bot()
    try:
        group = bot.tree.get_commands()[0]
        command_names = {command.name for command in group.commands}
        assert {
            "setup",
            "status",
            "doctor",
            "mode",
            "policy",
            "log_channel",
            "dry_run",
            "enabled",
            "actions",
            "timeout",
            "ignore_channel",
            "ignore_role",
            "exempt_user",
            "case",
            "history",
            "stats",
            "failures",
            "dry_run_cases",
            "errors",
            "slow_cases",
            "offenders",
            "channels",
            "rules",
            "categories",
            "test",
        }.issubset(command_names)
    finally:
        await bot.close()


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


async def test_discord_audit_top_users_orders_repeat_offenders(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    first = FakeDiscordMessage("drop your SSN")
    second = FakeDiscordMessage("another unsafe message")
    second.id = 790
    third = FakeDiscordMessage("borderline")
    third.id = 791
    third.author.id = 77

    for message in (first, second):
        await discord.make_handler(
            FakeCascade(VerdictLabel.UNSAFE, categories=["PII"]),
            config_store=config_store,
            audit_log=audit_log,
        )(message)
    await discord.make_handler(
        FakeCascade(VerdictLabel.CONTROVERSIAL, categories=["harassment"]),
        config_store=config_store,
        audit_log=audit_log,
    )(third)

    rows = audit_log.top_users(Platform.DISCORD, "123", limit=5, case_limit=10)
    text = discord._offenders_text(rows, case_limit=10)

    assert rows[0]["user_id"] == "42"
    assert rows[0]["total"] == 2
    assert rows[0]["unsafe"] == 2
    assert rows[1]["user_id"] == "77"
    assert "cases `2`" in text
    assert "delete:2" in text
    assert "(last 1.5h)" in discord._offenders_text(rows, case_limit=10, since_hours=1.5)
    assert discord._offenders_text([], case_limit=10) == (
        "**nemoguardian offenders**\nNo moderated users found."
    )
    assert discord._offenders_text([], case_limit=10, since_hours=2) == (
        "**nemoguardian offenders** (last 2h)\nNo moderated users found."
    )


async def test_discord_audit_top_channels_orders_hotspots(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    first = FakeDiscordMessage("drop your SSN")
    second = FakeDiscordMessage("another unsafe message")
    second.id = 790
    third = FakeDiscordMessage("borderline")
    third.id = 791
    third.channel = FakeChannel(888)

    for message in (first, second):
        await discord.make_handler(
            FakeCascade(VerdictLabel.UNSAFE, categories=["PII"]),
            config_store=config_store,
            audit_log=audit_log,
        )(message)
    await discord.make_handler(
        FakeCascade(VerdictLabel.CONTROVERSIAL, categories=["harassment"]),
        config_store=config_store,
        audit_log=audit_log,
    )(third)

    rows = audit_log.top_channels(Platform.DISCORD, "123", limit=5, case_limit=10)
    text = discord._channels_text(rows, case_limit=10)

    assert rows[0]["channel_id"] == "456"
    assert rows[0]["total"] == 2
    assert rows[0]["unsafe"] == 2
    assert rows[1]["channel_id"] == "888"
    assert "channel <#456>" in text
    assert "delete:2" in text
    assert "(last 2h)" in discord._channels_text(rows, case_limit=10, since_hours=2)
    assert discord._channels_text([], case_limit=10) == (
        "**nemoguardian channels**\nNo moderated channels found."
    )
    assert discord._channels_text([], case_limit=10, since_hours=2) == (
        "**nemoguardian channels** (last 2h)\nNo moderated channels found."
    )


async def test_discord_audit_top_rules_orders_policy_hotspots(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    first = FakeDiscordMessage("drop your SSN")
    second = FakeDiscordMessage("another unsafe message")
    second.id = 790
    third = FakeDiscordMessage("borderline")
    third.id = 791

    for message in (first, second):
        await discord.make_handler(
            FakeCascade(
                VerdictLabel.UNSAFE,
                categories=["PII"],
                matched_policy_rule="block-pii",
            ),
            config_store=config_store,
            audit_log=audit_log,
        )(message)
    await discord.make_handler(
        FakeCascade(
            VerdictLabel.CONTROVERSIAL,
            categories=["harassment"],
            matched_policy_rule="watch-harassment",
        ),
        config_store=config_store,
        audit_log=audit_log,
    )(third)

    rows = audit_log.top_rules(Platform.DISCORD, "123", limit=5, case_limit=10)
    text = discord._rules_text(rows, case_limit=10)

    assert rows[0]["rule"] == "block-pii"
    assert rows[0]["total"] == 2
    assert rows[0]["unsafe"] == 2
    assert rows[0]["categories"] == {"PII": 2}
    assert rows[1]["rule"] == "watch-harassment"
    assert "rule `block-pii`" in text
    assert "categories `PII:2`" in text
    assert "(last 2h)" in discord._rules_text(rows, case_limit=10, since_hours=2)
    assert discord._rules_text([], case_limit=10) == (
        "**nemoguardian rules**\nNo policy rules found."
    )
    assert discord._rules_text([], case_limit=10, since_hours=2) == (
        "**nemoguardian rules** (last 2h)\nNo policy rules found."
    )


async def test_discord_audit_top_categories_orders_content_hotspots(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    first = FakeDiscordMessage("drop your SSN")
    second = FakeDiscordMessage("another unsafe message")
    second.id = 790
    third = FakeDiscordMessage("borderline")
    third.id = 791

    for message in (first, second):
        await discord.make_handler(
            FakeCascade(
                VerdictLabel.UNSAFE,
                categories=["PII"],
                matched_policy_rule="block-pii",
            ),
            config_store=config_store,
            audit_log=audit_log,
        )(message)
    await discord.make_handler(
        FakeCascade(
            VerdictLabel.CONTROVERSIAL,
            categories=["harassment"],
            matched_policy_rule="watch-harassment",
        ),
        config_store=config_store,
        audit_log=audit_log,
    )(third)

    rows = audit_log.top_categories(Platform.DISCORD, "123", limit=5, case_limit=10)
    text = discord._categories_text(rows, case_limit=10)

    assert rows[0]["category"] == "PII"
    assert rows[0]["total"] == 2
    assert rows[0]["unsafe"] == 2
    assert rows[0]["actions"] == {"delete": 2}
    assert rows[0]["rules"] == {"block-pii": 2}
    assert rows[1]["category"] == "harassment"
    assert "category `PII`" in text
    assert "rules `block-pii:2`" in text
    assert "(last 2h)" in discord._categories_text(rows, case_limit=10, since_hours=2)
    assert discord._categories_text([], case_limit=10) == (
        "**nemoguardian categories**\nNo moderation categories found."
    )
    assert discord._categories_text([], case_limit=10, since_hours=2) == (
        "**nemoguardian categories** (last 2h)\nNo moderation categories found."
    )


async def test_discord_adapter_ignores_bot_messages(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    cascade = FakeCascade(VerdictLabel.UNSAFE)
    message = FakeDiscordMessage("bot echo", bot=True)

    await discord.make_handler(cascade, config_store=config_store, audit_log=audit_log)(message)

    assert cascade.calls == []
    assert message.deleted is False


async def test_discord_adapter_can_moderate_allowlisted_e2e_bot_author(monkeypatch, tmp_path):
    config_store, audit_log = _stores(tmp_path)
    monkeypatch.setenv("NEMOGUARDIAN_DISCORD_E2E_BOT_AUTHOR_IDS", "42, 100")
    cascade = FakeCascade(VerdictLabel.UNSAFE, categories=["PII"])
    message = FakeDiscordMessage("bot-authored smoke text", bot=True)

    await discord.make_handler(cascade, config_store=config_store, audit_log=audit_log)(message)

    assert message.deleted is True
    assert cascade.calls[0]["text"] == "bot-authored smoke text"
    assert audit_log.recent()[0]["execution_status"] == "delete+public-warning"


async def test_discord_adapter_skips_configured_exclusions(tmp_path):
    scenarios = [
        ("ignored_channel_ids", "456", lambda message: None),
        ("ignored_role_ids", "111", lambda message: message.author.roles.append(SimpleNamespace(id=111))),
        ("exempt_user_ids", "42", lambda message: None),
    ]

    for index, (field, value, prepare_message) in enumerate(scenarios):
        config_store = ConfigStore(tmp_path / f"bot-config-{index}.json")
        audit_log = AuditLog(tmp_path / f"bot-audit-{index}.jsonl")
        config = BotConfig.default(Platform.DISCORD, "123")
        getattr(config, field).add(value)
        config_store.save(config)
        cascade = FakeCascade(VerdictLabel.UNSAFE)
        message = FakeDiscordMessage("drop your SSN")
        prepare_message(message)

        await discord.make_handler(cascade, config_store=config_store, audit_log=audit_log)(message)

        assert cascade.calls == []
        assert message.deleted is False
        assert audit_log.recent() == []


async def test_discord_adapter_honors_action_toggles(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    config = BotConfig.default(Platform.DISCORD, "123")
    config.delete_unsafe = False
    config.public_warning = False
    config_store.save(config)
    message = FakeDiscordMessage("drop your SSN")

    await discord.make_handler(
        FakeCascade(VerdictLabel.UNSAFE, categories=["PII"]),
        config_store=config_store,
        audit_log=audit_log,
    )(message)

    assert message.deleted is False
    assert message.channel.messages == []
    assert audit_log.recent()[0]["execution_status"] == "planned"

    config_store, audit_log = _stores(tmp_path / "controversial")
    config = BotConfig.default(Platform.DISCORD, "123")
    config.react_controversial = False
    config_store.save(config)
    message = FakeDiscordMessage("borderline")

    await discord.make_handler(
        FakeCascade(VerdictLabel.CONTROVERSIAL),
        config_store=config_store,
        audit_log=audit_log,
    )(message)

    assert message.reactions == []
    assert audit_log.recent()[0]["execution_status"] == "planned"


async def test_discord_mod_log_handles_missing_and_client_fallback_channels():
    message = FakeDiscordMessage("drop your SSN")
    config = BotConfig.default(Platform.DISCORD, "123")
    config.log_channel_id = "777"
    evaluation = _evaluation(message, config=config)

    await discord._send_mod_log(message, evaluation, applied=["delete"], errors=[])
    assert message.guild.log_channel.messages == []

    fallback_channel = FakeChannel(777)
    client = SimpleNamespace(get_channel=lambda channel_id: fallback_channel if channel_id == 777 else None)
    message.guild = SimpleNamespace(id=123, name="Test Guild")
    message._state = SimpleNamespace(_get_client=lambda: client)

    await discord._send_mod_log(message, evaluation, applied=["delete"], errors=[])

    assert "applied: `delete`" in fallback_channel.messages[0]
    skipped = _evaluation(message, config=config, result=None)
    assert discord._mod_log_text(message, skipped, applied=[], errors=[]) == (
        "nemoguardian skipped a message."
    )


async def test_discord_permission_gate_and_format_helpers():
    allowed = FakeInteraction(manage_guild=True)
    denied = FakeInteraction(manage_guild=False)
    values = {"456"}
    skipped = ModerationEvaluation(
        context=discord._context_from_message(FakeDiscordMessage("")),
        config=BotConfig.default(Platform.DISCORD, "123"),
        result=None,
        plan=ModerationPlan(action=ModerationAction.ALLOW, reason="disabled"),
        skipped=True,
        skip_reason="disabled",
    )

    assert await discord._require_manage_guild(allowed) is True
    assert await discord._require_manage_guild(denied) is False
    assert "Manage Server permission" in denied.response.messages[0]["text"]
    discord._toggle_id(values, "456", enabled=False)
    assert values == set()
    assert discord._test_text(skipped) == "Skipped: `disabled`"
    assert discord._format_counts({}) == "none"
    assert discord._safe_since_hours(None) is None
    assert discord._safe_since_hours(-5) == 0.0
    assert discord._safe_since_hours(24 * 400) == 24 * 365
    assert discord._format_score("bad") == "unknown"
    assert discord._format_latency_ms(None) == "unknown"
    assert discord._format_latency_ms(1500) == "1.50s"


def test_discord_run_bot_requires_token(monkeypatch):
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="DISCORD_BOT_TOKEN env var required"):
        discord.run_bot()


def test_discord_run_bot_starts_built_bot(monkeypatch):
    class FakeRunnableBot:
        def __init__(self) -> None:
            self.tokens: list[str] = []

        def run(self, token: str) -> None:
            self.tokens.append(token)

    bot = FakeRunnableBot()
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "discord-test-token")
    monkeypatch.setattr(discord, "build_bot", lambda: bot)

    discord.run_bot()

    assert bot.tokens == ["discord-test-token"]


async def test_discord_build_bot_registers_and_runs_admin_commands(monkeypatch, tmp_path):
    _install_fake_discord(monkeypatch)
    config_store = ConfigStore(tmp_path / "slash-config.json")
    audit_log = AuditLog(tmp_path / "slash-audit.jsonl")
    handled_messages: list[Any] = []
    recorded_tests: list[tuple[ModerationEvaluation, str]] = []

    async def fake_handler(message) -> None:
        handled_messages.append(message)

    class FakeModerationEngine:
        def __init__(self, *args, **kwargs) -> None:
            self.audit_log = kwargs.get("audit_log")

        def evaluate(self, context, config):
            result = ModerateResponse(
                verdict=VerdictLabel.SAFE,
                score=0.1,
                reasons=["ok"],
                categories=[],
                matched_policy_rule=None,
                model_verdicts={},
                total_latency_ms=12.0,
                mode=config.mode,
                request_id="slash-req",
            )
            return ModerationEvaluation(
                context=context,
                config=config,
                result=result,
                plan=ModerationPlan(action=ModerationAction.ALLOW, reason="allowed"),
            )

        def record(self, evaluation, *, execution_status: str, error: str | None = None) -> None:
            recorded_tests.append((evaluation, execution_status))

    monkeypatch.setattr(discord, "ConfigStore", lambda: config_store)
    monkeypatch.setattr(discord, "AuditLog", lambda: audit_log)
    monkeypatch.setattr(discord, "make_handler", lambda *args, **kwargs: fake_handler)
    monkeypatch.setattr(discord, "ModerationEngine", FakeModerationEngine)
    monkeypatch.setenv("DISCORD_GUILD_ID", "123")

    bot = discord.build_bot()
    await bot.setup_hook()
    monkeypatch.delenv("DISCORD_GUILD_ID", raising=False)
    await bot.setup_hook()
    await bot.events["on_ready"]()
    await bot.events["on_message"](FakeDiscordMessage("hello"))
    group = bot.tree.group
    assert group is not None

    interaction = FakeInteraction()
    await group.commands["setup"](interaction, log_channel=SimpleNamespace(id=999))
    await group.commands["status"](interaction)
    await group.commands["doctor"](interaction)
    await group.commands["mode"](interaction, mode=SimpleNamespace(value="fast"))
    await group.commands["policy"](interaction, text="block spam and scams")
    await group.commands["log_channel"](interaction, channel=SimpleNamespace(id=777))
    await group.commands["dry_run"](interaction, enabled=True)
    await group.commands["enabled"](interaction, enabled=False)
    await group.commands["actions"](
        interaction,
        delete_unsafe=False,
        public_warning=False,
        react_controversial=False,
        dm_users=True,
    )
    await group.commands["timeout"](interaction, enabled=True, seconds=30)
    await group.commands["ignore_channel"](interaction, channel=SimpleNamespace(id=456), ignored=True)
    await group.commands["ignore_role"](interaction, role=SimpleNamespace(id=111), ignored=True)
    await group.commands["exempt_user"](interaction, user=SimpleNamespace(id=42), exempt=True)
    audit_log.append(
        AuditRecord(
            case_id="foreign-case",
            platform=Platform.TWITCH,
            workspace_id="other-workspace",
            channel_id="channel",
            message_id="message",
            user_id="viewer",
            username="viewer",
            action=ModerationAction.DELETE,
            verdict=VerdictLabel.UNSAFE,
            score=0.9,
            mode=Mode.FAST,
        )
    )
    await group.commands["case"](interaction, case_id="foreign-case")
    await group.commands["case"](interaction, case_id=" missing ")
    await group.commands["history"](interaction, limit=99, since_hours=-1)
    await group.commands["stats"](interaction, limit=999, since_hours=24 * 400)
    await group.commands["failures"](interaction, limit=99, since_hours=2)
    await group.commands["dry_run_cases"](interaction, limit=99, since_hours=2)
    await group.commands["errors"](interaction, limit=99, case_limit=9999, since_hours=2)
    await group.commands["slow_cases"](interaction, limit=99, case_limit=9999, since_hours=2)
    await group.commands["offenders"](interaction, limit=99, case_limit=9999, since_hours=2)
    await group.commands["channels"](interaction, limit=99, case_limit=9999, since_hours=2)
    await group.commands["rules"](interaction, limit=99, case_limit=9999, since_hours=2)
    await group.commands["categories"](interaction, limit=99, case_limit=9999, since_hours=2)
    await group.commands["test"](interaction, text="hello")

    denied = FakeInteraction(manage_guild=False)
    denied_calls = [
        ("setup", {"log_channel": SimpleNamespace(id=999)}),
        ("status", {}),
        ("doctor", {}),
        ("mode", {"mode": SimpleNamespace(value="deep")}),
        ("policy", {"text": "blocked"}),
        ("log_channel", {"channel": SimpleNamespace(id=444)}),
        ("dry_run", {"enabled": False}),
        ("enabled", {"enabled": True}),
        (
            "actions",
            {
                "delete_unsafe": True,
                "public_warning": True,
                "react_controversial": True,
                "dm_users": False,
            },
        ),
        ("timeout", {"enabled": False, "seconds": 120}),
        ("ignore_channel", {"channel": SimpleNamespace(id=321), "ignored": False}),
        ("ignore_role", {"role": SimpleNamespace(id=222), "ignored": False}),
        ("exempt_user", {"user": SimpleNamespace(id=43), "exempt": False}),
        ("case", {"case_id": "discord-123-789"}),
        ("history", {"limit": 1, "since_hours": 1}),
        ("stats", {"limit": 1, "since_hours": 1}),
        ("failures", {"limit": 1, "since_hours": 1}),
        ("dry_run_cases", {"limit": 1, "since_hours": 1}),
        ("errors", {"limit": 1, "case_limit": 1, "since_hours": 1}),
        ("slow_cases", {"limit": 1, "case_limit": 1, "since_hours": 1}),
        ("offenders", {"limit": 1, "case_limit": 1, "since_hours": 1}),
        ("channels", {"limit": 1, "case_limit": 1, "since_hours": 1}),
        ("rules", {"limit": 1, "case_limit": 1, "since_hours": 1}),
        ("categories", {"limit": 1, "case_limit": 1, "since_hours": 1}),
        ("test", {"text": "denied"}),
    ]
    for command_name, kwargs in denied_calls:
        await group.commands[command_name](denied, **kwargs)

    config = config_store.get(Platform.DISCORD, "123")
    assert bot.intents.guilds is True
    assert bot.intents.message_content is True
    assert bot.tree.copy_targets[0].id == 123
    assert bot.tree.sync_targets[-1] is None
    assert handled_messages[0].content == "hello"
    assert config.enabled is False
    assert config.mode == Mode.FAST
    assert config.policy_text == "block spam and scams"
    assert config.log_channel_id == "777"
    assert config.dry_run is True
    assert config.delete_unsafe is False
    assert config.public_warning is False
    assert config.react_controversial is False
    assert config.dm_users is True
    assert config.timeout_unsafe is True
    assert config.timeout_seconds == 60
    assert config.ignored_channel_ids == {"456"}
    assert config.ignored_role_ids == {"111"}
    assert config.exempt_user_ids == {"42"}
    assert recorded_tests[0][1] == "slash-test"
    assert any("nemoguardian test result" in message["text"] for message in interaction.response.messages)
    assert len(denied.response.messages) == len(denied_calls)
    assert all("Manage Server permission" in message["text"] for message in denied.response.messages)


def test_discord_module_main_starts_fake_bot(monkeypatch):
    _install_fake_discord(monkeypatch)
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "discord-test-token")

    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        namespace = runpy.run_module("nemoguardian.adapters.discord", run_name="__main__")

    assert namespace["__name__"] == "__main__"


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


async def test_twitch_adapter_emits_skipped_message_without_audit(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    config = BotConfig.default(Platform.TWITCH, "channel-1")
    config.enabled = False
    config_store.save(config)
    cascade = FakeCascade(VerdictLabel.UNSAFE)
    emitted: list[str] = []

    action = await twitch.make_moderator(
        cascade,
        config_store=config_store,
        audit_log=audit_log,
        channel_id="channel-1",
        emit=emitted.append,
    )("drop your SSN", user_id="viewer-1", username="viewer")

    assert action == "allow"
    assert cascade.calls == []
    assert audit_log.recent() == []
    assert emitted == ["[twitch] allow: drop your SSN (skipped=disabled)"]


def test_twitch_run_bot_requires_token(monkeypatch):
    monkeypatch.delenv("TWITCH_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="TWITCH_TOKEN env var required"):
        twitch.run_bot("channel")


def test_twitch_run_bot_registers_event_and_ignores_echo(monkeypatch):
    messages = [
        SimpleNamespace(echo=True, content="bot echo", author=SimpleNamespace(id="bot", name="bot")),
        SimpleNamespace(echo=False, content="drop your SSN", author=SimpleNamespace(id="7", name="viewer")),
    ]
    bots = _install_fake_twitch(monkeypatch, messages)
    calls: list[dict[str, str]] = []

    async def fake_moderate(text: str, *, user_id: str, username: str) -> str:
        calls.append({"text": text, "user_id": user_id, "username": username})
        return "delete"

    monkeypatch.setenv("TWITCH_TOKEN", "twitch-test-token")
    monkeypatch.setattr(twitch, "make_moderator", lambda: fake_moderate)

    twitch.run_bot("nemoguardian")

    assert bots[0].token == "twitch-test-token"
    assert bots[0].prefix == "!"
    assert bots[0].initial_channels == ["nemoguardian"]
    assert bots[0].run_called is True
    assert calls == [{"text": "drop your SSN", "user_id": "7", "username": "viewer"}]


def test_twitch_module_main_starts_fake_bot(monkeypatch):
    bots = _install_fake_twitch(monkeypatch, [])
    monkeypatch.setenv("TWITCH_TOKEN", "twitch-test-token")
    monkeypatch.setattr(sys, "argv", ["python -m nemoguardian.adapters.twitch", "nemoguardian"])

    with pytest.warns(RuntimeWarning, match="found in sys.modules"):
        runpy.run_module("nemoguardian.adapters.twitch", run_name="__main__")

    assert bots[0].token == "twitch-test-token"
    assert bots[0].initial_channels == ["nemoguardian"]
    assert bots[0].run_called is True


def test_twitch_module_main_requires_channel(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["python -m nemoguardian.adapters.twitch"])

    with pytest.warns(RuntimeWarning, match="found in sys.modules"), pytest.raises(SystemExit) as exc:
        runpy.run_module("nemoguardian.adapters.twitch", run_name="__main__")

    assert exc.value.code == 1
    assert "usage: python -m nemoguardian.adapters.twitch <channel>" in capsys.readouterr().out


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


async def test_webhook_adapter_uses_default_http_clients_and_explicit_api_key(monkeypatch):
    clients: list[Any] = []

    class FakeContextHTTPClient:
        def __init__(self, *, timeout: float) -> None:
            self.timeout = timeout
            self.posts: list[dict[str, Any]] = []
            clients.append(self)

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> bool:
            return False

        async def post(self, url: str, **kwargs: Any) -> FakeHTTPResponse:
            self.posts.append({"url": url, **kwargs})
            if url.endswith("/v1/moderate"):
                return FakeHTTPResponse({"verdict": "safe", "score": 0.1})
            return FakeHTTPResponse({"ok": True})

    monkeypatch.setenv("NEMOGUARDIAN_API_KEY", "env-key")
    monkeypatch.setattr(webhook.httpx, "AsyncClient", FakeContextHTTPClient)

    verdict = await webhook.moderate_and_forward(
        "hello",
        forward_url="http://forward.test/hook",
        moderator_url="http://moderator.test",
        policy="block spam",
        mode="fast",
        api_key=" explicit-key ",
    )

    assert verdict == {"verdict": "safe", "score": 0.1}
    assert [client.timeout for client in clients] == [30.0, 10.0]
    assert clients[0].posts == [
        {
            "url": "http://moderator.test/v1/moderate",
            "json": {"text": "hello", "policy": "block spam", "mode": "fast"},
            "params": None,
            "headers": {"Authorization": "Bearer explicit-key"},
        }
    ]
    assert clients[1].posts == [
        {
            "url": "http://forward.test/hook",
            "json": {"text": "hello", "verdict": verdict},
        }
    ]
    assert webhook._auth_headers("   ") == {}


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


def test_discord_doctor_text_reports_readiness_gaps():
    config = BotConfig.default(Platform.DISCORD, "123")
    config.enabled = False
    config.timeout_unsafe = True
    permissions = SimpleNamespace(
        view_channel=True,
        read_message_history=True,
        send_messages=False,
        manage_messages=False,
        embed_links=False,
        moderate_members=False,
    )

    text = discord._doctor_text(config, permissions, message_content_enabled=False)

    assert "needs attention" in text
    assert "moderation is disabled" in text
    assert "mod-log channel is not set" in text
    assert "Send Messages" in text
    assert "Manage Messages" in text
    assert "Moderate Members" in text


def test_discord_status_text_includes_exclusion_controls():
    config = BotConfig.default(Platform.DISCORD, "123")
    discord._apply_action_options(
        config,
        delete_unsafe=False,
        public_warning=False,
        react_controversial=False,
        dm_users=True,
    )
    config.ignored_channel_ids.add("456")
    config.ignored_role_ids.add("111")
    config.exempt_user_ids.add("42")

    text = discord._status_text(config)

    assert "delete unsafe: `False`" in text
    assert "public warning: `False`" in text
    assert "react controversial: `False`" in text
    assert "dm users: `True`" in text
    assert "ignored channels: `456`" in text
    assert "ignored roles: `111`" in text
    assert "exempt users: `42`" in text


def test_discord_case_and_history_text_helpers():
    record = {
        "case_id": "discord-123-789",
        "username": "tester",
        "user_id": "42",
        "channel_id": "456",
        "message_id": "789",
        "verdict": "unsafe",
        "score": 0.9,
        "mode": "standard",
        "action": "delete",
        "execution_status": "delete+public-warning",
        "dry_run": False,
        "categories": ["PII"],
        "matched_policy_rule": "fake-rule",
        "request_id": "req-test",
        "created_at": "2026-06-27T00:00:00+00:00",
        "error": "delete failed",
        "text_excerpt": "drop your SSN",
        "details": {"permalink": "https://discord.test/message/789"},
    }

    assert discord._case_text(None) == "Case not found."
    case_text = discord._case_text(record)
    history_text = discord._history_text([record])

    assert "discord-123-789" in case_text
    assert "delete+public-warning" in case_text
    assert "delete failed" in case_text
    assert "https://discord.test/message/789" in case_text
    assert "discord-123-789" in history_text
    assert discord._history_text([]) == "No moderation history found."
