"""Tests for the platform-neutral moderation engine."""

from __future__ import annotations

from nemoguardian.bot import (
    AuditLog,
    BotConfig,
    ModerationAction,
    ModerationContext,
    ModerationEngine,
    Platform,
)
from nemoguardian.bot.engine import plan_for_result
from nemoguardian.schemas import Mode, ModerateResponse, VerdictLabel


def _context(*, text: str = "hello") -> ModerationContext:
    return ModerationContext(
        platform=Platform.DISCORD,
        workspace_id="guild-1",
        channel_id="channel-1",
        message_id="message-1",
        user_id="user-1",
        username="tester",
        text=text,
    )


def test_engine_skips_empty_messages_and_does_not_record_skips(tmp_path):
    audit_log = AuditLog(tmp_path / "audit.jsonl")
    engine = ModerationEngine(
        Platform.DISCORD,
        cascade=object(),
        audit_log=audit_log,
    )
    evaluation = engine.evaluate(_context(text="   "), BotConfig.default(Platform.DISCORD, "guild-1"))

    assert evaluation.skipped is True
    assert evaluation.skip_reason == "empty-message"

    engine.record(evaluation, execution_status="skipped")

    assert audit_log.recent() == []


def test_plan_for_safe_result_allows_message():
    result = ModerateResponse(
        verdict=VerdictLabel.SAFE,
        score=0.1,
        reasons=["ok"],
        categories=[],
        matched_policy_rule=None,
        model_verdicts={},
        total_latency_ms=1.0,
        mode=Mode.STANDARD,
        request_id="req-safe",
    )

    plan = plan_for_result(result, BotConfig.default(Platform.DISCORD, "guild-1"))

    assert plan.action == ModerationAction.ALLOW
    assert plan.reason == "allowed"
