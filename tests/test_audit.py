"""Tests for moderation audit log helpers."""

from __future__ import annotations

import datetime as dt

from nemoguardian.bot import AuditLog, AuditRecord, ModerationAction, Platform
from nemoguardian.bot.audit import (
    _ensure_aware,
    _parse_created_at,
    _record_at_or_after,
    _record_errors,
    _record_latency_ms,
    _record_score,
    excerpt,
    redact_text,
    redacted_excerpt,
    since_hours_ago,
)
from nemoguardian.schemas import Mode, VerdictLabel


def _record(
    case_id: str,
    *,
    platform: Platform = Platform.DISCORD,
    workspace_id: str = "workspace",
    channel_id: str = "channel",
    user_id: str = "user",
    username: str = "tester",
    action: ModerationAction = ModerationAction.DELETE,
    verdict: VerdictLabel = VerdictLabel.UNSAFE,
    score: float = 0.9,
    categories: list[str] | None = None,
    matched_policy_rule: str | None = "block-pii",
    dry_run: bool = False,
    execution_status: str = "delete",
    error: str | None = None,
    latency_ms: float | None = 100.0,
    created_at: str = "2026-06-28T12:00:00+00:00",
) -> AuditRecord:
    return AuditRecord(
        case_id=case_id,
        platform=platform,
        workspace_id=workspace_id,
        channel_id=channel_id,
        message_id=f"message-{case_id}",
        user_id=user_id,
        username=username,
        action=action,
        verdict=verdict,
        score=score,
        mode=Mode.STANDARD,
        categories=categories or ["PII"],
        matched_policy_rule=matched_policy_rule,
        dry_run=dry_run,
        execution_status=execution_status,
        error=error,
        latency_ms=latency_ms,
        created_at=created_at,
    )


def test_audit_recent_empty_limit_missing_file_and_bad_json(tmp_path):
    missing = AuditLog(tmp_path / "missing.jsonl")
    assert missing.recent() == []
    assert missing.recent(0) == []

    path = tmp_path / "audit.jsonl"
    path.write_text("{not-json}\n", encoding="utf-8")
    audit = AuditLog(path)
    audit.append(_record("case-1"))

    assert [record["case_id"] for record in audit.recent()] == ["case-1"]
    assert audit.find_case("missing") is None


def test_audit_history_filters_skip_non_matching_records_and_stop_at_limit(tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl")
    audit.append(_record("match", dry_run=True))

    mismatches = [
        _record("old", dry_run=True, created_at="2026-06-27T00:00:00+00:00"),
        _record("dry-run", dry_run=False),
        _record("status", dry_run=True, execution_status="reaction"),
        _record("verdict", dry_run=True, verdict=VerdictLabel.CONTROVERSIAL),
        _record("action", dry_run=True, action=ModerationAction.FLAG),
        _record("rule", dry_run=True, matched_policy_rule="watch-harassment"),
        _record("category", dry_run=True, categories=["harassment"]),
        _record("channel", dry_run=True, channel_id="other-channel"),
        _record("user", dry_run=True, user_id="other-user"),
        _record("workspace", dry_run=True, workspace_id="other-workspace"),
        _record("platform", platform=Platform.TWITCH, dry_run=True),
    ]
    for record in mismatches:
        audit.append(record)

    assert audit.history(Platform.DISCORD, "workspace", limit=0) == []
    rows = audit.history(
        Platform.DISCORD,
        "workspace",
        user_id="user",
        channel_id="channel",
        category="PII",
        rule="block-pii",
        action=ModerationAction.DELETE,
        verdict=VerdictLabel.UNSAFE,
        status="delete",
        dry_run=True,
        since=dt.datetime(2026, 6, 28, tzinfo=dt.timezone.utc),
        limit=1,
    )

    assert [record["case_id"] for record in rows] == ["match"]


def test_failure_and_dry_run_filters_skip_non_matching_records_and_stop_at_limit(tmp_path):
    failure_log = AuditLog(tmp_path / "failures.jsonl")
    failure_log.append(_record("failure", execution_status="failed"))
    failure_log.append(_record("wrong-workspace-failure", workspace_id="other", execution_status="failed"))
    failure_log.append(_record("wrong-platform-failure", platform=Platform.TWITCH, execution_status="failed"))

    assert failure_log.failures(Platform.DISCORD, "workspace", limit=0) == []
    assert [record["case_id"] for record in failure_log.failures(Platform.DISCORD, "workspace", limit=1)] == [
        "failure"
    ]

    dry_run_log = AuditLog(tmp_path / "dry-run.jsonl")
    dry_run_log.append(_record("planned-action", dry_run=True, action=ModerationAction.TIMEOUT))
    dry_run_log.append(
        _record(
            "wrong-workspace-dry-run",
            workspace_id="other",
            dry_run=True,
            action=ModerationAction.TIMEOUT,
        )
    )
    dry_run_log.append(
        _record(
            "wrong-platform-dry-run",
            platform=Platform.TWITCH,
            dry_run=True,
            action=ModerationAction.TIMEOUT,
        )
    )

    assert dry_run_log.dry_run_cases(Platform.DISCORD, "workspace", limit=0) == []
    assert [record["case_id"] for record in dry_run_log.dry_run_cases(Platform.DISCORD, "workspace", limit=1)] == [
        "planned-action"
    ]


def test_audit_rollup_limit_guards(tmp_path):
    audit = AuditLog(tmp_path / "audit.jsonl")

    assert audit.top_errors(Platform.DISCORD, "workspace", limit=0) == []
    assert audit.top_errors(Platform.DISCORD, "workspace", case_limit=0) == []
    assert audit.slow_cases(Platform.DISCORD, "workspace", limit=0) == []
    assert audit.slow_cases(Platform.DISCORD, "workspace", case_limit=0) == []
    assert audit.high_score_cases(Platform.DISCORD, "workspace", limit=0) == []
    assert audit.high_score_cases(Platform.DISCORD, "workspace", case_limit=0) == []
    assert audit.top_users(Platform.DISCORD, "workspace", limit=0) == []
    assert audit.top_users(Platform.DISCORD, "workspace", case_limit=0) == []
    assert audit.top_channels(Platform.DISCORD, "workspace", limit=0) == []
    assert audit.top_channels(Platform.DISCORD, "workspace", case_limit=0) == []
    assert audit.top_rules(Platform.DISCORD, "workspace", limit=0) == []
    assert audit.top_rules(Platform.DISCORD, "workspace", case_limit=0) == []
    assert audit.top_categories(Platform.DISCORD, "workspace", limit=0) == []
    assert audit.top_categories(Platform.DISCORD, "workspace", case_limit=0) == []


def test_text_redaction_excerpt_and_time_helpers():
    raw = (
        "  email admin@example.com ssn 123-45-6789 phone (312) 555-0101 "
        "card 4242 4242 4242 4242  "
    )

    assert excerpt("  hello\nworld  ") == "hello world"
    assert excerpt("abcdefghij", limit=5) == "abcd..."
    assert redact_text(raw) == "  email [email] ssn [ssn] phone [phone] card [payment-card]  "
    assert redacted_excerpt(raw, limit=35) == "email [email] ssn [ssn] phone [pho..."
    assert since_hours_ago(None) is None

    naive_now = dt.datetime(2026, 6, 28, 12, 0)
    since = since_hours_ago(2, now=naive_now)
    assert since == dt.datetime(2026, 6, 28, 10, 0, tzinfo=dt.timezone.utc)
    assert _ensure_aware(dt.datetime(2026, 6, 28, 7, 0, tzinfo=dt.timezone(dt.timedelta(hours=-5)))) == naive_now.replace(
        tzinfo=dt.timezone.utc
    )

    assert _parse_created_at(123) is None
    assert _parse_created_at("") is None
    assert _parse_created_at("not-a-date") is None
    assert _parse_created_at("2026-06-28T12:00:00Z") == naive_now.replace(tzinfo=dt.timezone.utc)
    assert _record_at_or_after({"created_at": "not-a-date"}, naive_now) is False


def test_metric_helpers_reject_malformed_values_and_fallback_errors(tmp_path):
    assert _record_latency_ms({"latency_ms": "bad"}) is None
    assert _record_latency_ms({"latency_ms": None}) is None
    assert _record_latency_ms({"latency_ms": float("nan")}) is None
    assert _record_latency_ms({"latency_ms": -1}) is None
    assert _record_latency_ms({"latency_ms": "42.5"}) == 42.5

    assert _record_score({"score": "bad"}) is None
    assert _record_score({"score": None}) is None
    assert _record_score({"score": float("inf")}) is None
    assert _record_score({"score": "0.85"}) == 0.85

    audit = AuditLog(tmp_path / "scores.jsonl")
    audit.append(_record("valid-score", score=0.6))
    audit.append(_record("bad-score", score=float("nan")))
    assert [record["case_id"] for record in audit.high_score_cases(Platform.DISCORD, "workspace")] == [
        "valid-score"
    ]

    assert _record_errors({"error": " delete:Forbidden ; timeout:Forbidden "}) == [
        "delete:Forbidden",
        "timeout:Forbidden",
    ]
    assert _record_errors({"error": " ; ", "execution_status": "partial"}) == ["partial"]
    assert _record_errors({}) == ["unknown"]
