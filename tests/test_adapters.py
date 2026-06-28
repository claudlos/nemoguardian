"""Tests for platform adapter glue."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from nemoguardian.adapters import discord, twitch, webhook
from nemoguardian.bot import (
    AuditLog,
    AuditRecord,
    BotConfig,
    ConfigStore,
    ModerationAction,
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

    assert [record["case_id"] for record in records] == [
        "discord-123-current-dry-run",
        "discord-123-old-dry-run",
    ]
    assert [record["case_id"] for record in windowed] == ["discord-123-current-dry-run"]
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


def test_discord_doctor_text_reports_readiness_gaps():
    config = BotConfig.default(Platform.DISCORD, "123")
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
        "text_excerpt": "drop your SSN",
        "details": {"permalink": "https://discord.test/message/789"},
    }

    assert discord._case_text(None) == "Case not found."
    case_text = discord._case_text(record)
    history_text = discord._history_text([record])

    assert "discord-123-789" in case_text
    assert "delete+public-warning" in case_text
    assert "https://discord.test/message/789" in case_text
    assert "discord-123-789" in history_text
    assert discord._history_text([]) == "No moderation history found."
