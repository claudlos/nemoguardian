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
from nemoguardian.bot.engine import REVIEW_DIR_ENV, plan_for_result
from nemoguardian.review.service import ReviewService
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


class _FakeCascade:
    """Cascade stub returning a fixed verdict (default: controversial → review)."""

    def __init__(self, verdict: VerdictLabel = VerdictLabel.CONTROVERSIAL) -> None:
        self._verdict = verdict

    def moderate(self, request, policy_engine=None) -> ModerateResponse:
        return ModerateResponse(
            verdict=self._verdict,
            score=0.55,
            reasons=["borderline"],
            categories=["harassment"],
            matched_policy_rule=None,
            model_verdicts={},
            total_latency_ms=1.0,
            mode=request.mode,
            request_id="req-fake",
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


def test_engine_enqueues_review_via_env_dir_when_enabled(monkeypatch, tmp_path):
    review_dir = tmp_path / "review"
    monkeypatch.setenv(REVIEW_DIR_ENV, str(review_dir))
    engine = ModerationEngine(
        Platform.DISCORD,
        cascade=_FakeCascade(),
        audit_log=AuditLog(tmp_path / "audit.jsonl"),
    )
    config = BotConfig.default(Platform.DISCORD, "guild-1")
    assert config.review_queue is True  # opt-in flag is on by default

    evaluation = engine.evaluate(_context(text="borderline take"), config)
    engine.record(evaluation, execution_status="executed")

    # A default ReviewService was lazily built from the env dir and a pending
    # case was enqueued — no adapter had to inject a ReviewService.
    pending = ReviewService.from_dir(review_dir).queue.list_pending()
    assert len(pending) == 1
    assert pending[0].source_case_id == pending[0].case_id
    assert pending[0].action == ModerationAction.FLAG


def test_engine_does_not_enqueue_when_review_queue_disabled(monkeypatch, tmp_path):
    review_dir = tmp_path / "review"
    monkeypatch.setenv(REVIEW_DIR_ENV, str(review_dir))
    engine = ModerationEngine(
        Platform.DISCORD,
        cascade=_FakeCascade(),
        audit_log=AuditLog(tmp_path / "audit.jsonl"),
    )
    config = BotConfig.default(Platform.DISCORD, "guild-1")
    config.review_queue = False

    evaluation = engine.evaluate(_context(text="borderline take"), config)
    engine.record(evaluation, execution_status="executed")

    # Opt-out: nothing is written, and the store file is never even created.
    assert not (review_dir / "review_queue.jsonl").exists()


def test_engine_no_enqueue_without_env_dir(monkeypatch, tmp_path):
    monkeypatch.delenv(REVIEW_DIR_ENV, raising=False)
    engine = ModerationEngine(
        Platform.DISCORD,
        cascade=_FakeCascade(),
        audit_log=AuditLog(tmp_path / "audit.jsonl"),
    )
    config = BotConfig.default(Platform.DISCORD, "guild-1")

    evaluation = engine.evaluate(_context(text="borderline take"), config)
    # Fail-safe: no configured review dir → enqueue is a silent no-op.
    engine.record(evaluation, execution_status="executed")
    assert engine.enqueue_review(evaluation) is None
