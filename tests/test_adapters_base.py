"""Tests for the shared platform-adapter foundation.

Covers: the PlatformAdapter interface conformance of the existing adapters,
capability detection + graceful degradation, the new ModerationAction values,
the Platform enum additions, and the slack/telegram presets.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from nemoguardian.adapters import discord, twitch, webhook
from nemoguardian.adapters.base import (
    DEGRADE_TARGET,
    NORMALIZED_ACTIONS,
    ActionDecision,
    PlatformAdapter,
    degrade_action,
    supports,
)
from nemoguardian.bot import (
    BotConfig,
    ConfigStore,
    ModerationContext,
    ModerationEvaluation,
    ModerationPlan,
)
from nemoguardian.bot.types import ModerationAction, Platform
from nemoguardian.policy.presets import PRESETS, get_preset
from nemoguardian.schemas import VerdictLabel


class StubCascade:
    """Records calls so we can assert an adapter did/didn't moderate."""

    def __init__(self) -> None:
        self.calls: list[Any] = []

    def moderate(self, request, *, policy_engine=None):  # pragma: no cover - guarded
        self.calls.append(request)
        raise AssertionError("cascade.moderate should not run in these tests")


def _discord_adapter(tmp_path) -> discord.DiscordAdapter:
    return discord.DiscordAdapter(
        StubCascade(),
        config_store=ConfigStore(tmp_path / "config.json"),
    )


def _twitch_adapter(tmp_path) -> twitch.TwitchAdapter:
    return twitch.TwitchAdapter(
        StubCascade(),
        config_store=ConfigStore(tmp_path / "config.json"),
        channel_id="chan-1",
    )


# --------------------------------------------------------------------------- #
# #18 Normalized action vocabulary + #3 Platform enum additions
# --------------------------------------------------------------------------- #
def test_moderation_action_vocabulary_is_complete():
    values = {action.value for action in ModerationAction}
    assert values == {
        "allow",
        "flag",
        "delete",
        "timeout",
        "mute",
        "ban",
        "queue",
        "notify_mods",
        "notify_user",
    }
    # Existing values keep their identity/behavior.
    assert ModerationAction.ALLOW.value == "allow"
    assert ModerationAction.FLAG.value == "flag"
    assert ModerationAction.DELETE.value == "delete"
    assert ModerationAction.TIMEOUT.value == "timeout"


def test_platform_enum_has_slack_and_telegram():
    assert Platform("slack") is Platform.SLACK
    assert Platform("telegram") is Platform.TELEGRAM
    # Existing platforms still resolve.
    assert {Platform.DISCORD, Platform.TWITCH, Platform.WEBHOOK} <= set(Platform)


def test_normalized_actions_covers_every_action():
    assert frozenset(ModerationAction) == NORMALIZED_ACTIONS


# --------------------------------------------------------------------------- #
# #17 PlatformAdapter interface conformance
# --------------------------------------------------------------------------- #
def test_existing_adapters_conform_to_platform_adapter(tmp_path):
    adapters = [
        _discord_adapter(tmp_path),
        _twitch_adapter(tmp_path),
        webhook.WebhookAdapter(forward_url="http://forward.test/hook"),
    ]
    for adapter in adapters:
        assert isinstance(adapter, PlatformAdapter)
        assert isinstance(adapter.platform, Platform)
        for method in ("capabilities", "doctor", "configure", "handle_event",
                       "apply_action", "record_audit"):
            assert callable(getattr(adapter, method))
        caps = adapter.capabilities()
        assert caps
        assert caps <= NORMALIZED_ACTIONS


def test_adapter_capabilities_match_module_helpers(tmp_path):
    assert _discord_adapter(tmp_path).capabilities() == discord.capabilities()
    assert _twitch_adapter(tmp_path).capabilities() == twitch.capabilities()
    assert webhook.WebhookAdapter().capabilities() == webhook.capabilities()


def test_capability_sets_reflect_platform_enforcement():
    assert ModerationAction.TIMEOUT in discord.capabilities()
    assert ModerationAction.NOTIFY_USER in discord.capabilities()
    assert ModerationAction.BAN in twitch.capabilities()
    # A webhook only forwards a verdict; it cannot delete/timeout/ban.
    assert ModerationAction.DELETE not in webhook.capabilities()
    assert ModerationAction.TIMEOUT not in webhook.capabilities()
    # Every platform can at least allow + flag (the degrade target).
    for caps in (discord.capabilities(), twitch.capabilities(), webhook.capabilities()):
        assert {ModerationAction.ALLOW, ModerationAction.FLAG} <= caps


# --------------------------------------------------------------------------- #
# #19 Capability detection + graceful degradation
# --------------------------------------------------------------------------- #
def test_supports_helper():
    assert supports(discord.capabilities(), ModerationAction.TIMEOUT) is True
    assert supports(discord.capabilities(), "ban") is False


def test_degrade_action_passthrough_when_supported():
    decision = degrade_action(ModerationAction.TIMEOUT, discord.capabilities(), Platform.DISCORD)
    assert isinstance(decision, ActionDecision)
    assert decision.action is ModerationAction.TIMEOUT
    assert decision.requested is ModerationAction.TIMEOUT
    assert decision.degraded is False
    assert decision.reason is None


def test_degrade_action_degrades_unsupported_to_flag_with_reason():
    decision = degrade_action(ModerationAction.TIMEOUT, webhook.capabilities(), Platform.WEBHOOK)
    assert decision.action is DEGRADE_TARGET is ModerationAction.FLAG
    assert decision.requested is ModerationAction.TIMEOUT
    assert decision.degraded is True
    assert decision.reason == "timeout unsupported on webhook -> degraded to flag"


def test_degrade_action_accepts_string_inputs_and_string_platform():
    decision = degrade_action("ban", discord.capabilities(), "discord")
    assert decision.action is ModerationAction.FLAG
    assert decision.requested is ModerationAction.BAN
    assert decision.reason == "ban unsupported on discord -> degraded to flag"


def test_adapter_apply_action_degrades_via_capabilities(tmp_path):
    # Webhook can never enforce: any enforcement action degrades to flag.
    decision = webhook.WebhookAdapter().apply_action(ModerationAction.BAN)
    assert decision.action is ModerationAction.FLAG
    assert decision.degraded is True

    # Twitch supports timeout (passthrough) but not mute (degrades).
    twitch_adapter = _twitch_adapter(tmp_path)
    plan = ModerationPlan(action=ModerationAction.TIMEOUT)
    evaluation = ModerationEvaluation(
        context=_context(),
        config=BotConfig.default(Platform.TWITCH, "chan-1"),
        result=None,
        plan=plan,
    )
    assert twitch_adapter.apply_action(evaluation).degraded is False
    plan.action = ModerationAction.MUTE
    degraded = twitch_adapter.apply_action(evaluation)
    assert degraded.action is ModerationAction.FLAG
    assert degraded.reason == "mute unsupported on twitch -> degraded to flag"


def _context() -> ModerationContext:
    return ModerationContext(
        platform=Platform.TWITCH,
        workspace_id="chan-1",
        channel_id="chan-1",
        message_id="m1",
        user_id="u1",
        username="viewer",
        text="hello",
    )


# --------------------------------------------------------------------------- #
# Adapter conveniences delegate without changing behavior
# --------------------------------------------------------------------------- #
def test_discord_adapter_configure_and_doctor(tmp_path):
    adapter = _discord_adapter(tmp_path)
    updated = adapter.configure("123", enabled=False)
    assert updated.enabled is False
    assert adapter.configure("123").enabled is False  # read-back persists
    report = adapter.doctor("123", message_content_enabled=True)
    assert "readiness" in report


async def test_discord_adapter_handle_event_delegates_to_handler(tmp_path):
    adapter = _discord_adapter(tmp_path)
    message = SimpleNamespace(author=SimpleNamespace(bot=False, id=1), guild=None)
    # guild is None -> the shared handler skips before moderating; no raise.
    await adapter.handle_event(message)
    assert adapter.engine.cascade.calls == []


async def test_discord_adapter_apply_action_delegates(tmp_path):
    adapter = _discord_adapter(tmp_path)
    evaluation = ModerationEvaluation(
        context=_context(),
        config=BotConfig.default(Platform.DISCORD, "123"),
        result=None,
        plan=ModerationPlan(action=ModerationAction.ALLOW),
    )
    status, error = await adapter.apply_action(SimpleNamespace(), evaluation)
    assert (status, error) == ("allowed", None)


def test_twitch_and_webhook_doctor_never_raise(tmp_path):
    twitch_report = _twitch_adapter(tmp_path).doctor()
    assert twitch_report["platform"] == "twitch"
    assert "readiness" in twitch_report

    webhook_report = webhook.WebhookAdapter().doctor()
    assert webhook_report["readiness"] == "needs attention"
    assert webhook.WebhookAdapter(forward_url="http://x").doctor()["readiness"] == "ready"


async def test_webhook_handle_event_requires_a_forward_target():
    with pytest.raises(ValueError, match="forward_url is required"):
        await webhook.WebhookAdapter().handle_event("hi")


# --------------------------------------------------------------------------- #
# #5 Slack / Telegram presets load alongside discord/twitch/generic
# --------------------------------------------------------------------------- #
def test_slack_and_telegram_presets_registered():
    assert {"slack", "telegram"} <= set(PRESETS)
    assert get_preset("slack").name == "slack-default"
    assert get_preset("telegram").name == "telegram-default"


@pytest.mark.parametrize("preset_name", ["slack", "telegram"])
def test_new_presets_block_pii(preset_name):
    policy = get_preset(preset_name)
    decision = policy.evaluate(verdict=VerdictLabel.SAFE, score=0.1, categories=["PII"])
    assert decision.final_label == VerdictLabel.UNSAFE
    assert decision.matched_rule == "force-block-pii"


@pytest.mark.parametrize(
    ("platform", "expected_preset"),
    [(Platform.SLACK, "slack"), (Platform.TELEGRAM, "telegram")],
)
def test_config_defaults_use_new_presets(platform, expected_preset, tmp_path):
    config = BotConfig.default(platform, "workspace-1")
    assert config.policy_preset == expected_preset
    # Round-trips through the JSON-backed store.
    store = ConfigStore(tmp_path / "config.json")
    store.save(config)
    assert store.get(platform, "workspace-1").policy_preset == expected_preset
