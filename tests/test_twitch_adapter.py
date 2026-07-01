"""Tests for the real-enforcement Twitch adapter (#21, #22).

No live secret, network, GPU, or Twitch SDK: a fake cascade produces a
deterministic verdict and a fake chat client records dispatched actions, so the
full action flow (plan -> escalate -> degrade -> dispatch -> audit) plus chat
commands are exercised offline.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from nemoguardian.adapters import twitch
from nemoguardian.bot import AuditLog, ConfigStore
from nemoguardian.bot.types import ModerationAction, Platform
from nemoguardian.schemas import Mode, ModerateResponse, VerdictLabel


class FakeCascade:
    def __init__(self, verdict: VerdictLabel, *, categories: list[str] | None = None) -> None:
        self.verdict = verdict
        self.categories = categories or []
        self.calls: list[dict[str, Any]] = []

    def moderate(self, request, *, policy_engine=None):
        self.calls.append({"text": request.text, "mode": request.mode.value})
        return ModerateResponse(
            verdict=self.verdict,
            score=0.9,
            reasons=["fake"],
            categories=self.categories,
            matched_policy_rule="fake-rule" if self.categories else None,
            model_verdicts={},
            mode=request.mode,
            total_latency_ms=1.0,
        )


class FakeTwitchClient:
    """Records the moderation calls the adapter dispatches."""

    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.timeouts: list[tuple[str, int, str]] = []
        self.bans: list[tuple[str, str]] = []
        self.messages: list[tuple[str, str]] = []

    async def delete_message(self, message: twitch.TwitchMessage) -> None:
        self.deleted.append(message.message_id)

    async def timeout_user(self, message: twitch.TwitchMessage, *, seconds: int, reason: str) -> None:
        self.timeouts.append((message.user_id, seconds, reason))

    async def ban_user(self, message: twitch.TwitchMessage, *, reason: str) -> None:
        self.bans.append((message.user_id, reason))

    async def send_message(self, channel: str, text: str) -> None:
        self.messages.append((channel, text))


def _stores(tmp_path):
    return (
        ConfigStore(tmp_path / "config.json"),
        AuditLog(tmp_path / "audit.jsonl"),
    )


# --------------------------------------------------------------------------- #
# Capabilities + degradation
# --------------------------------------------------------------------------- #
def test_capabilities_are_a_real_moderator():
    assert twitch.capabilities() == {
        ModerationAction.ALLOW,
        ModerationAction.FLAG,
        ModerationAction.DELETE,
        ModerationAction.TIMEOUT,
        ModerationAction.BAN,
    }


def test_twitch_decision_degrades_unsupported():
    passthrough = twitch.twitch_decision(ModerationAction.BAN)
    assert passthrough.action is ModerationAction.BAN
    assert passthrough.degraded is False

    degraded = twitch.twitch_decision(ModerationAction.MUTE)
    assert degraded.action is ModerationAction.FLAG
    assert degraded.degraded is True
    assert degraded.reason == "mute unsupported on twitch -> degraded to flag"


# --------------------------------------------------------------------------- #
# Real action dispatch (#21)
# --------------------------------------------------------------------------- #
async def test_unsafe_message_is_deleted_via_client(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    client = FakeTwitchClient()
    moderate = twitch.make_moderator(
        FakeCascade(VerdictLabel.UNSAFE, categories=["Scam"]),
        config_store=config_store,
        audit_log=audit_log,
        channel_id="chan",
        client=client,
        emit=lambda _msg: None,
    )

    action = await moderate("drop your SSN", user_id="viewer-1", username="viewer", message_id="m1")

    assert action == "delete"
    assert client.deleted == ["m1"]
    assert client.timeouts == [] and client.bans == []
    record = audit_log.recent()[0]
    assert record["action"] == "delete"
    assert record["execution_status"] == "delete"
    assert record["error"] is None


async def test_timeout_when_config_prefers_timeout(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    config_store.update(Platform.TWITCH, "chan", timeout_unsafe=True, timeout_seconds=45)
    client = FakeTwitchClient()
    moderate = twitch.make_moderator(
        FakeCascade(VerdictLabel.UNSAFE),
        config_store=config_store,
        audit_log=audit_log,
        channel_id="chan",
        client=client,
        emit=lambda _msg: None,
    )

    action = await moderate("bad", user_id="viewer-2", username="viewer", message_id="m2")

    assert action == "timeout"
    assert client.timeouts == [("viewer-2", 45, "policy violation")]
    assert client.deleted == []


async def test_repeat_offender_escalates_delete_timeout_ban(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    client = FakeTwitchClient()
    moderate = twitch.make_moderator(
        FakeCascade(VerdictLabel.UNSAFE),
        config_store=config_store,
        audit_log=audit_log,
        channel_id="chan",
        client=client,
        emit=lambda _msg: None,
    )

    first = await moderate("bad1", user_id="troll", username="troll", message_id="a")
    second = await moderate("bad2", user_id="troll", username="troll", message_id="b")
    third = await moderate("bad3", user_id="troll", username="troll", message_id="c")

    assert (first, second, third) == ("delete", "timeout", "ban")
    assert client.deleted == ["a"]
    assert [t[0] for t in client.timeouts] == ["troll"]
    assert client.bans == [("troll", "policy violation")]


def test_escalate_action_pure_function():
    assert twitch.escalate_action(ModerationAction.DELETE, 1) is ModerationAction.DELETE
    assert twitch.escalate_action(ModerationAction.DELETE, 2) is ModerationAction.TIMEOUT
    assert twitch.escalate_action(ModerationAction.DELETE, 3) is ModerationAction.BAN
    # A configured timeout stays a timeout on the first offense.
    assert twitch.escalate_action(ModerationAction.TIMEOUT, 1) is ModerationAction.TIMEOUT
    # Non-escalating actions pass through untouched.
    assert twitch.escalate_action(ModerationAction.FLAG, 9) is ModerationAction.FLAG


async def test_dry_run_dispatches_nothing(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    config_store.update(Platform.TWITCH, "chan", dry_run=True)
    client = FakeTwitchClient()
    moderate = twitch.make_moderator(
        FakeCascade(VerdictLabel.UNSAFE),
        config_store=config_store,
        audit_log=audit_log,
        channel_id="chan",
        client=client,
        emit=lambda _msg: None,
    )

    action = await moderate("bad", user_id="viewer", username="viewer", message_id="m")

    assert action == "delete"  # planned action recorded honestly...
    assert client.deleted == [] and client.timeouts == [] and client.bans == []  # ...but not dispatched
    record = audit_log.recent()[0]
    assert record["execution_status"] == "dry-run"
    assert record["dry_run"] is True


async def test_exempt_user_is_skipped(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    config = config_store.get(Platform.TWITCH, "chan")
    config.exempt_user_ids = {"vip"}
    config_store.save(config)
    client = FakeTwitchClient()
    emitted: list[str] = []
    moderate = twitch.make_moderator(
        FakeCascade(VerdictLabel.UNSAFE),
        config_store=config_store,
        audit_log=audit_log,
        channel_id="chan",
        client=client,
        emit=emitted.append,
    )

    action = await moderate("bad", user_id="vip", username="vip", message_id="m")

    assert action == "allow"
    assert client.deleted == [] and audit_log.recent() == []
    assert emitted == ["[twitch] allow: bad (skipped=exempt-user)"]


async def test_no_client_degrades_enforcement_to_flag(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    moderate = twitch.make_moderator(
        FakeCascade(VerdictLabel.UNSAFE),
        config_store=config_store,
        audit_log=audit_log,
        channel_id="chan",
        emit=lambda _msg: None,
    )

    action = await moderate("bad", user_id="viewer", username="viewer", message_id="m")

    assert action == "flag"
    record = audit_log.recent()[0]
    assert record["action"] == "flag"
    assert record["error"] == "delete unsupported on twitch -> degraded to flag"


async def test_controversial_flags_without_dispatch(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    client = FakeTwitchClient()
    moderate = twitch.make_moderator(
        FakeCascade(VerdictLabel.CONTROVERSIAL, categories=["Spam"]),
        config_store=config_store,
        audit_log=audit_log,
        channel_id="chan",
        client=client,
        emit=lambda _msg: None,
    )

    action = await moderate("maybe bad", user_id="viewer", username="viewer", message_id="m")

    assert action == "flag"
    assert client.deleted == [] and client.timeouts == [] and client.bans == []
    assert audit_log.recent()[0]["action"] == "flag"


async def test_public_warning_sends_chat_message(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    config_store.update(Platform.TWITCH, "chan", public_warning=True)
    client = FakeTwitchClient()
    moderate = twitch.make_moderator(
        FakeCascade(VerdictLabel.UNSAFE),
        config_store=config_store,
        audit_log=audit_log,
        channel_id="chan",
        client=client,
        emit=lambda _msg: None,
    )

    await moderate("bad", user_id="viewer", username="viewer", message_id="m")

    assert client.deleted == ["m"]
    assert client.messages and client.messages[0][0] == "chan"
    assert "nemoguardian" in client.messages[0][1]


async def test_client_failure_is_never_raised(tmp_path):
    config_store, audit_log = _stores(tmp_path)

    class BrokenClient:
        async def delete_message(self, message: twitch.TwitchMessage) -> None:
            raise RuntimeError("twitch down")

    moderate = twitch.make_moderator(
        FakeCascade(VerdictLabel.UNSAFE),
        config_store=config_store,
        audit_log=audit_log,
        channel_id="chan",
        client=BrokenClient(),
        emit=lambda _msg: None,
    )

    action = await moderate("bad", user_id="viewer", username="viewer", message_id="m")

    assert action == "delete"
    record = audit_log.recent()[0]
    assert record["execution_status"] == "failed"
    assert "delete:RuntimeError" in record["error"]


async def test_audit_record_is_redacted(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    client = FakeTwitchClient()
    moderate = twitch.make_moderator(
        FakeCascade(VerdictLabel.UNSAFE),
        config_store=config_store,
        audit_log=audit_log,
        channel_id="chan",
        client=client,
        emit=lambda _msg: None,
    )

    secret = "my SSN is 123-45-6789 and card 4111111111111111"
    await moderate(secret, user_id="viewer", username="viewer", message_id="m")

    record = audit_log.recent()[0]
    assert "123-45-6789" not in str(record.get("text_excerpt"))
    assert "4111111111111111" not in str(record.get("text_excerpt"))
    assert record["text_sha256"]
    assert record["details"]["text_redacted"] is True


# --------------------------------------------------------------------------- #
# Chat commands + auth (#22)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("text", "name", "args"),
    [
        ("!nemo status", "status", ""),
        ("!nemo dryrun on", "dryrun", "on"),
        ("!ng policy block all links here", "policy", "block all links here"),
        ("!guardian mode fast", "mode", "fast"),
        ("!NEMO STATUS", "status", ""),
        ("!nemo", "status", ""),
    ],
)
def test_parse_command(text, name, args):
    command = twitch.parse_command(text)
    assert command is not None
    assert command.name == name
    assert command.args == args


def test_parse_command_ignores_plain_chat():
    assert twitch.parse_command("hello chat, nice stream") is None
    assert twitch.parse_command("!lurk") is None
    assert twitch.parse_command("") is None


@pytest.mark.parametrize(
    "badges",
    [
        SimpleNamespace(is_broadcaster=True, is_mod=False),
        SimpleNamespace(is_mod=True),
        {"moderator": "1"},
        {"broadcaster": "1"},
        {"moderator", "subscriber"},
        "broadcaster",
    ],
)
def test_is_authorized_accepts_broadcaster_and_mods(badges):
    assert twitch.is_authorized(badges) is True


@pytest.mark.parametrize(
    "badges",
    [None, SimpleNamespace(is_mod=False), {"subscriber": "1"}, {"vip"}, "viewer"],
)
def test_is_authorized_rejects_everyone_else(badges):
    assert twitch.is_authorized(badges) is False


def test_handle_command_requires_authorization(tmp_path):
    config_store, _ = _stores(tmp_path)
    response = twitch.handle_command(
        twitch.TwitchCommand("dryrun", "on"),
        config_store=config_store,
        channel_id="chan",
        badges=SimpleNamespace(is_mod=False),
    )
    assert "only the broadcaster and moderators" in response
    # Config untouched by an unauthorized command.
    assert config_store.get(Platform.TWITCH, "chan").dry_run is False


def test_handle_command_mutates_config_for_mods(tmp_path):
    config_store, _ = _stores(tmp_path)
    mod = SimpleNamespace(is_mod=True)

    dry = twitch.handle_command(
        twitch.TwitchCommand("dryrun", "on"), config_store=config_store, channel_id="chan", badges=mod
    )
    assert "dry-run on" in dry
    assert config_store.get(Platform.TWITCH, "chan").dry_run is True

    policy = twitch.handle_command(
        twitch.TwitchCommand("policy", "no crypto shilling"),
        config_store=config_store,
        channel_id="chan",
        badges=mod,
    )
    assert "no crypto shilling" in policy
    assert config_store.get(Platform.TWITCH, "chan").policy_text == "no crypto shilling"

    mode = twitch.handle_command(
        twitch.TwitchCommand("mode", "deep"), config_store=config_store, channel_id="chan", badges=mod
    )
    assert "deep" in mode
    assert config_store.get(Platform.TWITCH, "chan").mode == Mode.DEEP


def test_handle_command_bad_args_and_unknown(tmp_path):
    config_store, _ = _stores(tmp_path)
    mod = SimpleNamespace(is_broadcaster=True)
    assert "usage" in twitch.handle_command(
        twitch.TwitchCommand("dryrun", "maybe"), config_store=config_store, channel_id="chan", badges=mod
    )
    assert "usage" in twitch.handle_command(
        twitch.TwitchCommand("mode", "sideways"), config_store=config_store, channel_id="chan", badges=mod
    )
    assert "unknown command" in twitch.handle_command(
        twitch.TwitchCommand("frobnicate"), config_store=config_store, channel_id="chan", badges=mod
    )
    # status reflects current config for an authorized user.
    assert "nemoguardian status" in twitch.handle_command(
        twitch.TwitchCommand("status"), config_store=config_store, channel_id="chan", badges=mod
    )


# --------------------------------------------------------------------------- #
# Adapter surface
# --------------------------------------------------------------------------- #
async def test_adapter_handle_event_enforces_with_client(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    client = FakeTwitchClient()
    adapter = twitch.TwitchAdapter(
        FakeCascade(VerdictLabel.UNSAFE),
        config_store=config_store,
        audit_log=audit_log,
        channel_id="chan",
        client=client,
        emit=lambda _msg: None,
    )

    action = await adapter.handle_event("bad", user_id="v", username="v", message_id="m")
    assert action == "delete"
    assert client.deleted == ["m"]


def test_adapter_handle_command_and_doctor(tmp_path):
    config_store, audit_log = _stores(tmp_path)
    client = FakeTwitchClient()
    adapter = twitch.TwitchAdapter(
        FakeCascade(VerdictLabel.SAFE),
        config_store=config_store,
        audit_log=audit_log,
        channel_id="chan",
        client=client,
    )

    assert adapter.handle_command("hello") is None
    response = adapter.handle_command("!nemo dryrun on", badges=SimpleNamespace(is_broadcaster=True))
    assert "dry-run on" in response
    assert adapter.configure().dry_run is True

    report = adapter.doctor()
    assert report["platform"] == "twitch"
    assert report["client_configured"] is True
    assert set(report["capabilities"]) == {"allow", "flag", "delete", "timeout", "ban"}
