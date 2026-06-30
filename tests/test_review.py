"""Tests for the human-in-the-loop review layer (audit TODOs #26, #27, #28)."""

from __future__ import annotations

import datetime as dt

import pytest

from nemoguardian.bot.types import ModerationAction, Platform
from nemoguardian.review import (
    AppealOutcome,
    AppealResult,
    CaseStatus,
    EnforcementAction,
    ReviewService,
    StrikePolicy,
    default_policy,
)
from nemoguardian.review.store import AppealLog, ReviewQueue, StrikeLedger
from nemoguardian.schemas import VerdictLabel


# --------------------------------------------------------------------------- #
# ReviewQueue (#26)
# --------------------------------------------------------------------------- #
def _queue(tmp_path) -> ReviewQueue:
    return ReviewQueue(tmp_path / "review_queue.jsonl")


def test_enqueue_creates_pending_case(tmp_path):
    queue = _queue(tmp_path)
    case = queue.enqueue(
        platform=Platform.DISCORD,
        workspace_id="123",
        user_id="42",
        text="ping me at admin@example.com please",
        reason="possible PII",
        categories=["PII"],
        score=0.6,
    )
    assert case.status is CaseStatus.PENDING
    assert case.case_id
    # Raw text is never persisted; the excerpt is redacted.
    assert "admin@example.com" not in case.text_excerpt
    assert "[email]" in case.text_excerpt
    assert case.text_sha256
    assert queue.get_case(case.case_id).status is CaseStatus.PENDING


def test_enqueue_is_idempotent_on_case_id(tmp_path):
    queue = _queue(tmp_path)
    first = queue.enqueue(
        platform=Platform.DISCORD, workspace_id="w", user_id="u", case_id="fixed-1"
    )
    queue.resolve("fixed-1", CaseStatus.REMOVED)
    # Re-enqueuing a known case must not resurrect it as pending.
    second = queue.enqueue(
        platform=Platform.DISCORD, workspace_id="w", user_id="u", case_id="fixed-1"
    )
    assert first.case_id == second.case_id
    assert second.status is CaseStatus.REMOVED
    assert queue.counts()["pending"] == 0


def test_list_pending_filters(tmp_path):
    queue = _queue(tmp_path)
    queue.enqueue(platform=Platform.DISCORD, workspace_id="w1", user_id="a", categories=["spam"])
    queue.enqueue(platform=Platform.DISCORD, workspace_id="w1", user_id="b", categories=["PII"])
    queue.enqueue(platform=Platform.TWITCH, workspace_id="w2", user_id="a")

    assert len(queue.list_pending()) == 3
    assert len(queue.list_pending(platform=Platform.DISCORD)) == 2
    assert len(queue.list_pending(workspace_id="w1", user_id="a")) == 1
    assert len(queue.list_pending(category="PII")) == 1
    assert queue.list_pending(limit=0) == []


def test_resolve_transitions_and_appends(tmp_path):
    queue = _queue(tmp_path)
    case = queue.enqueue(
        platform=Platform.DISCORD, workspace_id="w", user_id="u", case_id="c1"
    )
    resolved = queue.resolve(
        "c1", CaseStatus.APPROVED, resolved_by="mod-1", note="false alarm"
    )
    assert resolved.status is CaseStatus.APPROVED
    assert resolved.resolved_by == "mod-1"
    assert resolved.created_at == case.created_at  # preserved across the fold
    # Latest-wins fold: only the resolved state remains current.
    assert queue.get_case("c1").status is CaseStatus.APPROVED
    assert queue.counts()["approved"] == 1
    assert queue.counts()["pending"] == 0
    # Append-only: both records are on disk.
    assert len(queue._read()) == 2


def test_resolve_unknown_case_raises(tmp_path):
    with pytest.raises(KeyError):
        _queue(tmp_path).resolve("nope", CaseStatus.REMOVED)


def test_resolve_rejects_pending_status(tmp_path):
    queue = _queue(tmp_path)
    queue.enqueue(platform=Platform.DISCORD, workspace_id="w", user_id="u", case_id="c1")
    with pytest.raises(ValueError):
        queue.resolve("c1", CaseStatus.PENDING)


def test_counts_summarizes_statuses(tmp_path):
    queue = _queue(tmp_path)
    for idx in range(3):
        queue.enqueue(
            platform=Platform.DISCORD, workspace_id="w", user_id="u", case_id=f"c{idx}"
        )
    queue.resolve("c0", CaseStatus.REMOVED)
    queue.resolve("c1", CaseStatus.APPROVED)
    counts = queue.counts()
    assert counts == {"pending": 1, "approved": 1, "removed": 1, "restored": 0}


# --------------------------------------------------------------------------- #
# StrikeLedger + StrikePolicy (#28)
# --------------------------------------------------------------------------- #
def _ledger(tmp_path) -> StrikeLedger:
    return StrikeLedger(tmp_path / "strikes.jsonl")


def test_strike_policy_escalation_ladder():
    policy = default_policy()
    assert policy.recommend(0) is EnforcementAction.NONE
    assert policy.recommend(1) is EnforcementAction.WARN
    assert policy.recommend(2) is EnforcementAction.TIMEOUT
    assert policy.recommend(3) is EnforcementAction.BAN
    assert policy.recommend(99) is EnforcementAction.BAN


def test_strike_policy_next_threshold():
    policy = default_policy()
    assert policy.next_threshold(0) == (1, EnforcementAction.WARN)
    assert policy.next_threshold(1) == (2, EnforcementAction.TIMEOUT)
    assert policy.next_threshold(5) is None


def test_strike_policy_custom_and_sorted():
    policy = StrikePolicy.from_pairs([(5, "ban"), (2, "timeout")])
    # Thresholds are normalized to ascending order.
    assert [a for _, a in policy.thresholds] == [
        EnforcementAction.TIMEOUT,
        EnforcementAction.BAN,
    ]
    assert policy.recommend(1) is EnforcementAction.NONE
    assert policy.recommend(2) is EnforcementAction.TIMEOUT
    assert policy.recommend(5) is EnforcementAction.BAN


def test_strike_policy_requires_thresholds():
    with pytest.raises(ValueError):
        StrikePolicy(thresholds=[])


def test_strike_accumulation_and_recommend(tmp_path):
    ledger = _ledger(tmp_path)
    assert ledger.total(Platform.DISCORD, "w", "u") == 0
    ledger.add_strike(platform=Platform.DISCORD, workspace_id="w", user_id="u", case_id="c1")
    ledger.add_strike(platform=Platform.DISCORD, workspace_id="w", user_id="u", case_id="c2")
    assert ledger.total(Platform.DISCORD, "w", "u") == 2
    rec = ledger.recommend(Platform.DISCORD, "w", "u")
    assert rec["action"] == EnforcementAction.TIMEOUT.value
    assert rec["total"] == 2
    assert rec["next_threshold"] == {"min_total": 3, "action": "ban"}


def test_strike_weight_and_scoping(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.add_strike(platform=Platform.DISCORD, workspace_id="w", user_id="u", weight=3)
    # Different user / workspace / platform are isolated.
    ledger.add_strike(platform=Platform.DISCORD, workspace_id="w", user_id="other")
    ledger.add_strike(platform=Platform.TWITCH, workspace_id="w", user_id="u")
    assert ledger.total(Platform.DISCORD, "w", "u") == 3
    assert ledger.recommend(Platform.DISCORD, "w", "u")["action"] == EnforcementAction.BAN.value
    assert ledger.total(Platform.DISCORD, "w", "other") == 1


def test_void_strike_decrements_total(tmp_path):
    ledger = _ledger(tmp_path)
    s1 = ledger.add_strike(platform=Platform.DISCORD, workspace_id="w", user_id="u")
    ledger.add_strike(platform=Platform.DISCORD, workspace_id="w", user_id="u")
    assert ledger.total(Platform.DISCORD, "w", "u") == 2
    ledger.void_strike(s1.strike_id, reason="appeal")
    assert ledger.total(Platform.DISCORD, "w", "u") == 1
    assert len(ledger.active_strikes(Platform.DISCORD, "w", "u")) == 1


def test_void_unknown_strike_raises(tmp_path):
    with pytest.raises(KeyError):
        _ledger(tmp_path).void_strike("missing")


def test_void_for_case(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.add_strike(platform=Platform.DISCORD, workspace_id="w", user_id="u", case_id="c1")
    ledger.add_strike(platform=Platform.DISCORD, workspace_id="w", user_id="u", case_id="c1")
    ledger.add_strike(platform=Platform.DISCORD, workspace_id="w", user_id="u", case_id="c2")
    voided = ledger.void_for_case("c1")
    assert len(voided) == 2
    assert ledger.total(Platform.DISCORD, "w", "u") == 1


def test_strike_since_filter(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.add_strike(platform=Platform.DISCORD, workspace_id="w", user_id="u")
    future = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)
    assert ledger.total(Platform.DISCORD, "w", "u", since=future) == 0


def test_leaderboard_ranks_users(tmp_path):
    ledger = _ledger(tmp_path)
    ledger.add_strike(platform=Platform.DISCORD, workspace_id="w", user_id="light")
    for _ in range(3):
        ledger.add_strike(
            platform=Platform.DISCORD, workspace_id="w", user_id="heavy", username="Heavy"
        )
    board = ledger.leaderboard(Platform.DISCORD, "w")
    assert board[0]["user_id"] == "heavy"
    assert board[0]["total"] == 3
    assert board[0]["action"] == EnforcementAction.BAN.value
    assert board[0]["username"] == "Heavy"
    assert ledger.leaderboard(Platform.DISCORD, "w", limit=0) == []


# --------------------------------------------------------------------------- #
# AppealLog + ReviewService restore flow (#27)
# --------------------------------------------------------------------------- #
def _service(tmp_path) -> ReviewService:
    return ReviewService.from_dir(tmp_path)


def test_service_resolve_removed_adds_strike(tmp_path):
    service = _service(tmp_path)
    case = service.enqueue(
        platform=Platform.DISCORD,
        workspace_id="w",
        user_id="u",
        case_id="c1",
        action=ModerationAction.DELETE,
        reason="slur",
    )
    assert case.status is CaseStatus.PENDING
    service.resolve("c1", CaseStatus.REMOVED, resolved_by="mod", add_strike=True)
    assert service.strikes.total(Platform.DISCORD, "w", "u") == 1


def test_service_resolve_approved_does_not_strike(tmp_path):
    service = _service(tmp_path)
    service.enqueue(platform=Platform.DISCORD, workspace_id="w", user_id="u", case_id="c1")
    # add_strike is ignored unless the resolution is REMOVED.
    service.resolve("c1", CaseStatus.APPROVED, add_strike=True)
    assert service.strikes.total(Platform.DISCORD, "w", "u") == 0


def test_appeal_restore_flow_voids_strikes_and_links(tmp_path):
    service = _service(tmp_path)
    service.enqueue(
        platform=Platform.DISCORD,
        workspace_id="w",
        user_id="u",
        case_id="c1",
        action=ModerationAction.DELETE,
    )
    service.resolve("c1", CaseStatus.REMOVED, add_strike=True)
    assert service.strikes.total(Platform.DISCORD, "w", "u") == 1

    result = service.appeal(
        "c1", AppealOutcome.RESTORED, reason="false positive", decided_by="mod-2"
    )
    assert isinstance(result, AppealResult)
    # The restore reverses the strike so a false positive doesn't escalate.
    assert service.strikes.total(Platform.DISCORD, "w", "u") == 0
    assert len(result.voided_strikes) == 1
    # The case is restored and linked to the appeal.
    assert result.case.status is CaseStatus.RESTORED
    assert result.case.appeal_id == result.decision.appeal_id
    assert result.decision.case_id == "c1"
    assert result.decision.false_positive is True
    assert result.decision.original_action is ModerationAction.DELETE
    # Appeal is linked back from the appeal log.
    linked = service.appeals.for_case("c1")
    assert len(linked) == 1
    assert linked[0].appeal_id == result.decision.appeal_id


def test_appeal_upheld_has_no_side_effects(tmp_path):
    service = _service(tmp_path)
    service.enqueue(platform=Platform.DISCORD, workspace_id="w", user_id="u", case_id="c1")
    service.resolve("c1", CaseStatus.REMOVED, add_strike=True)
    result = service.appeal("c1", AppealOutcome.UPHELD, reason="confirmed violation")
    # Denied appeal: strike and removal stand.
    assert service.strikes.total(Platform.DISCORD, "w", "u") == 1
    assert result.case.status is CaseStatus.REMOVED
    assert result.voided_strikes == []
    assert result.decision.false_positive is False


def test_appeal_keep_strikes_flag(tmp_path):
    service = _service(tmp_path)
    service.enqueue(platform=Platform.DISCORD, workspace_id="w", user_id="u", case_id="c1")
    service.resolve("c1", CaseStatus.REMOVED, add_strike=True)
    service.appeal("c1", AppealOutcome.RESTORED, void_strikes=False)
    # Restore recorded but strikes retained by explicit choice.
    assert service.strikes.total(Platform.DISCORD, "w", "u") == 1
    assert service.queue.get_case("c1").status is CaseStatus.RESTORED


def test_appeal_unknown_case_raises(tmp_path):
    with pytest.raises(KeyError):
        _service(tmp_path).appeal("missing", AppealOutcome.RESTORED)


def test_appeal_log_list_and_find(tmp_path):
    service = _service(tmp_path)
    service.enqueue(platform=Platform.DISCORD, workspace_id="w", user_id="u", case_id="c1")
    service.resolve("c1", CaseStatus.REMOVED)
    result = service.appeal("c1", AppealOutcome.RESTORED)
    log: AppealLog = service.appeals
    assert log.find(result.decision.appeal_id) is not None
    assert log.find("missing") is None
    listed = log.list_appeals(platform=Platform.DISCORD, workspace_id="w")
    assert len(listed) == 1
    assert log.list_appeals(outcome=AppealOutcome.UPHELD) == []
    assert log.list_appeals(limit=0) == []


def test_service_recommend_uses_policy(tmp_path):
    service = ReviewService.from_dir(
        tmp_path, policy=StrikePolicy.from_pairs([(1, "ban")])
    )
    service.strikes.add_strike(platform=Platform.DISCORD, workspace_id="w", user_id="u")
    rec = service.recommend(Platform.DISCORD, "w", "u")
    assert rec["action"] == EnforcementAction.BAN.value


# --------------------------------------------------------------------------- #
# round-trip serialization
# --------------------------------------------------------------------------- #
def test_review_case_dict_round_trip(tmp_path):
    queue = _queue(tmp_path)
    case = queue.enqueue(
        platform=Platform.TWITCH,
        workspace_id="w",
        user_id="u",
        verdict=VerdictLabel.UNSAFE,
        categories=["spam", "scam"],
        action=ModerationAction.TIMEOUT,
    )
    from nemoguardian.review.models import ReviewCase

    restored = ReviewCase.from_dict(case.to_dict())
    assert restored == case
    assert restored.verdict is VerdictLabel.UNSAFE
    assert restored.action is ModerationAction.TIMEOUT
