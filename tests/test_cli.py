"""CLI tests."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from nemoguardian.bot import AuditLog, AuditRecord, ModerationAction, Platform
from nemoguardian.cli import app
from nemoguardian.schemas import Mode, VerdictLabel

runner = CliRunner()


def _seed_audit(tmp_path):
    path = tmp_path / "bot-audit.jsonl"
    audit = AuditLog(path)
    audit.append(
        AuditRecord(
            case_id="discord-123-1",
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
            matched_policy_rule="block-pii",
            latency_ms=400.0,
            execution_status="delete+public-warning",
            created_at="2000-01-01T00:00:00+00:00",
        )
    )
    audit.append(
        AuditRecord(
            case_id="discord-123-2",
            platform=Platform.DISCORD,
            workspace_id="123",
            channel_id="789",
            message_id="2",
            user_id="42",
            username="tester",
            action=ModerationAction.FLAG,
            verdict=VerdictLabel.CONTROVERSIAL,
            score=0.67,
            mode=Mode.STANDARD,
            categories=["harassment"],
            matched_policy_rule="watch-harassment",
            latency_ms=900.0,
            execution_status="reaction",
        )
    )
    return path


def _run(*args: str):
    return runner.invoke(app, list(args))


def test_bot_audit_cli_stats_history_and_offenders(tmp_path):
    path = _seed_audit(tmp_path)

    stats = _run(
        "bot-audit",
        "stats",
        "--path",
        str(path),
        "--workspace-id",
        "123",
        "--platform",
        "discord",
    )
    assert stats.exit_code == 0
    stats_body = json.loads(stats.stdout)
    assert stats_body["total"] == 2
    assert stats_body["verdicts"] == {"controversial": 1, "unsafe": 1}

    history = _run(
        "bot-audit",
        "history",
        "--path",
        str(path),
        "--workspace-id",
        "123",
        "--limit",
        "1",
    )
    assert history.exit_code == 0
    history_body = json.loads(history.stdout)
    assert history_body[0]["case_id"] == "discord-123-2"

    offenders = _run(
        "bot-audit",
        "offenders",
        "--path",
        str(path),
        "--workspace-id",
        "123",
    )
    assert offenders.exit_code == 0
    offenders_body = json.loads(offenders.stdout)
    assert offenders_body[0]["user_id"] == "42"
    assert offenders_body[0]["total"] == 2

    channels = _run(
        "bot-audit",
        "channels",
        "--path",
        str(path),
        "--workspace-id",
        "123",
    )
    assert channels.exit_code == 0
    channels_body = json.loads(channels.stdout)
    assert channels_body[0]["channel_id"] == "456"
    assert channels_body[0]["total"] == 1

    channel_history = _run(
        "bot-audit",
        "history",
        "--path",
        str(path),
        "--workspace-id",
        "123",
        "--channel-id",
        "789",
    )
    assert channel_history.exit_code == 0
    assert [record["case_id"] for record in json.loads(channel_history.stdout)] == ["discord-123-2"]

    channel_stats = _run(
        "bot-audit",
        "stats",
        "--path",
        str(path),
        "--workspace-id",
        "123",
        "--channel-id",
        "789",
    )
    assert channel_stats.exit_code == 0
    channel_stats_body = json.loads(channel_stats.stdout)
    assert channel_stats_body["channel_id"] == "789"
    assert channel_stats_body["total"] == 1
    assert channel_stats_body["verdicts"] == {"controversial": 1}

    category_history = _run(
        "bot-audit",
        "history",
        "--path",
        str(path),
        "--workspace-id",
        "123",
        "--category",
        "harassment",
    )
    assert category_history.exit_code == 0
    assert [record["case_id"] for record in json.loads(category_history.stdout)] == ["discord-123-2"]

    category_stats = _run(
        "bot-audit",
        "stats",
        "--path",
        str(path),
        "--workspace-id",
        "123",
        "--category",
        "harassment",
    )
    assert category_stats.exit_code == 0
    category_stats_body = json.loads(category_stats.stdout)
    assert category_stats_body["category"] == "harassment"
    assert category_stats_body["total"] == 1
    assert category_stats_body["actions"] == {"flag": 1}

    rule_history = _run(
        "bot-audit",
        "history",
        "--path",
        str(path),
        "--workspace-id",
        "123",
        "--rule",
        "watch-harassment",
    )
    assert rule_history.exit_code == 0
    assert [record["case_id"] for record in json.loads(rule_history.stdout)] == ["discord-123-2"]

    rule_stats = _run(
        "bot-audit",
        "stats",
        "--path",
        str(path),
        "--workspace-id",
        "123",
        "--rule",
        "watch-harassment",
    )
    assert rule_stats.exit_code == 0
    rule_stats_body = json.loads(rule_stats.stdout)
    assert rule_stats_body["rule"] == "watch-harassment"
    assert rule_stats_body["total"] == 1
    assert rule_stats_body["actions"] == {"flag": 1}

    enforcement_history = _run(
        "bot-audit",
        "history",
        "--path",
        str(path),
        "--workspace-id",
        "123",
        "--action",
        "delete",
        "--verdict",
        "unsafe",
        "--status",
        "delete+public-warning",
    )
    assert enforcement_history.exit_code == 0
    assert [record["case_id"] for record in json.loads(enforcement_history.stdout)] == ["discord-123-1"]

    enforcement_stats = _run(
        "bot-audit",
        "stats",
        "--path",
        str(path),
        "--workspace-id",
        "123",
        "--action",
        "delete",
        "--verdict",
        "unsafe",
        "--status",
        "delete+public-warning",
    )
    assert enforcement_stats.exit_code == 0
    enforcement_stats_body = json.loads(enforcement_stats.stdout)
    assert enforcement_stats_body["action"] == "delete"
    assert enforcement_stats_body["verdict"] == "unsafe"
    assert enforcement_stats_body["status"] == "delete+public-warning"
    assert enforcement_stats_body["total"] == 1
    assert enforcement_stats_body["categories"] == {"PII": 1}

    rules = _run(
        "bot-audit",
        "rules",
        "--path",
        str(path),
        "--workspace-id",
        "123",
    )
    assert rules.exit_code == 0
    rules_body = json.loads(rules.stdout)
    assert rules_body[0]["rule"] == "block-pii"
    assert rules_body[0]["total"] == 1

    categories = _run(
        "bot-audit",
        "categories",
        "--path",
        str(path),
        "--workspace-id",
        "123",
    )
    assert categories.exit_code == 0
    categories_body = json.loads(categories.stdout)
    assert categories_body[0]["category"] == "PII"
    assert categories_body[0]["total"] == 1

    slow_cases = _run(
        "bot-audit",
        "slow-cases",
        "--path",
        str(path),
        "--workspace-id",
        "123",
    )
    assert slow_cases.exit_code == 0
    assert [record["case_id"] for record in json.loads(slow_cases.stdout)] == [
        "discord-123-2",
        "discord-123-1",
    ]

    high_scores = _run(
        "bot-audit",
        "high-scores",
        "--path",
        str(path),
        "--workspace-id",
        "123",
    )
    assert high_scores.exit_code == 0
    assert [record["case_id"] for record in json.loads(high_scores.stdout)] == [
        "discord-123-1",
        "discord-123-2",
    ]

    windowed_stats = _run(
        "bot-audit",
        "stats",
        "--path",
        str(path),
        "--workspace-id",
        "123",
        "--since-hours",
        "1",
    )
    assert windowed_stats.exit_code == 0
    windowed_stats_body = json.loads(windowed_stats.stdout)
    assert windowed_stats_body["total"] == 1
    assert windowed_stats_body["verdicts"] == {"controversial": 1}
    assert windowed_stats_body["since"] is not None

    windowed_history = _run(
        "bot-audit",
        "history",
        "--path",
        str(path),
        "--workspace-id",
        "123",
        "--since-hours",
        "1",
    )
    assert windowed_history.exit_code == 0
    assert [record["case_id"] for record in json.loads(windowed_history.stdout)] == ["discord-123-2"]

    windowed_offenders = _run(
        "bot-audit",
        "offenders",
        "--path",
        str(path),
        "--workspace-id",
        "123",
        "--since-hours",
        "1",
    )
    assert windowed_offenders.exit_code == 0
    windowed_offenders_body = json.loads(windowed_offenders.stdout)
    assert windowed_offenders_body[0]["total"] == 1

    windowed_channels = _run(
        "bot-audit",
        "channels",
        "--path",
        str(path),
        "--workspace-id",
        "123",
        "--since-hours",
        "1",
    )
    assert windowed_channels.exit_code == 0
    windowed_channels_body = json.loads(windowed_channels.stdout)
    assert windowed_channels_body[0]["total"] == 1

    windowed_rules = _run(
        "bot-audit",
        "rules",
        "--path",
        str(path),
        "--workspace-id",
        "123",
        "--since-hours",
        "1",
    )
    assert windowed_rules.exit_code == 0
    windowed_rules_body = json.loads(windowed_rules.stdout)
    assert windowed_rules_body[0]["rule"] == "watch-harassment"
    assert windowed_rules_body[0]["total"] == 1

    windowed_categories = _run(
        "bot-audit",
        "categories",
        "--path",
        str(path),
        "--workspace-id",
        "123",
        "--since-hours",
        "1",
    )
    assert windowed_categories.exit_code == 0
    windowed_categories_body = json.loads(windowed_categories.stdout)
    assert windowed_categories_body[0]["category"] == "harassment"
    assert windowed_categories_body[0]["total"] == 1

    windowed_slow_cases = _run(
        "bot-audit",
        "slow-cases",
        "--path",
        str(path),
        "--workspace-id",
        "123",
        "--since-hours",
        "1",
    )
    assert windowed_slow_cases.exit_code == 0
    assert [record["case_id"] for record in json.loads(windowed_slow_cases.stdout)] == [
        "discord-123-2"
    ]

    windowed_high_scores = _run(
        "bot-audit",
        "high-scores",
        "--path",
        str(path),
        "--workspace-id",
        "123",
        "--since-hours",
        "1",
    )
    assert windowed_high_scores.exit_code == 0
    assert [record["case_id"] for record in json.loads(windowed_high_scores.stdout)] == [
        "discord-123-2"
    ]

    audit = AuditLog(path)
    audit.append(
        AuditRecord(
            case_id="discord-123-old-dry-run",
            platform=Platform.DISCORD,
            workspace_id="123",
            channel_id="456",
            message_id="5",
            user_id="42",
            username="tester",
            action=ModerationAction.DELETE,
            verdict=VerdictLabel.UNSAFE,
            score=0.9,
            mode=Mode.STANDARD,
            categories=["PII"],
            dry_run=True,
            execution_status="dry-run",
            created_at="2000-01-01T00:00:00+00:00",
        )
    )
    audit.append(
        AuditRecord(
            case_id="discord-123-current-dry-run",
            platform=Platform.DISCORD,
            workspace_id="123",
            channel_id="789",
            message_id="6",
            user_id="77",
            username="repeat",
            action=ModerationAction.FLAG,
            verdict=VerdictLabel.CONTROVERSIAL,
            score=0.7,
            mode=Mode.STANDARD,
            categories=["harassment"],
            dry_run=True,
            execution_status="dry-run",
        )
    )
    audit.append(
        AuditRecord(
            case_id="discord-123-allowed-dry-run",
            platform=Platform.DISCORD,
            workspace_id="123",
            channel_id="789",
            message_id="7",
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

    dry_runs = _run(
        "bot-audit",
        "dry-runs",
        "--path",
        str(path),
        "--workspace-id",
        "123",
    )
    assert dry_runs.exit_code == 0
    dry_runs_body = json.loads(dry_runs.stdout)
    assert [record["case_id"] for record in dry_runs_body] == [
        "discord-123-current-dry-run",
        "discord-123-old-dry-run",
    ]

    windowed_dry_runs = _run(
        "bot-audit",
        "dry-runs",
        "--path",
        str(path),
        "--workspace-id",
        "123",
        "--since-hours",
        "1",
    )
    assert windowed_dry_runs.exit_code == 0
    windowed_dry_runs_body = json.loads(windowed_dry_runs.stdout)
    assert [record["case_id"] for record in windowed_dry_runs_body] == [
        "discord-123-current-dry-run"
    ]

    dry_run_history = _run(
        "bot-audit",
        "history",
        "--path",
        str(path),
        "--workspace-id",
        "123",
        "--dry-run",
    )
    assert dry_run_history.exit_code == 0
    assert [record["case_id"] for record in json.loads(dry_run_history.stdout)] == [
        "discord-123-allowed-dry-run",
        "discord-123-current-dry-run",
        "discord-123-old-dry-run",
    ]

    dry_run_stats = _run(
        "bot-audit",
        "stats",
        "--path",
        str(path),
        "--workspace-id",
        "123",
        "--dry-run",
    )
    assert dry_run_stats.exit_code == 0
    dry_run_stats_body = json.loads(dry_run_stats.stdout)
    assert dry_run_stats_body["dry_run_filter"] is True
    assert dry_run_stats_body["total"] == 3
    assert dry_run_stats_body["dry_run"] == 3

    live_history = _run(
        "bot-audit",
        "history",
        "--path",
        str(path),
        "--workspace-id",
        "123",
        "--live",
    )
    assert live_history.exit_code == 0
    assert [record["case_id"] for record in json.loads(live_history.stdout)] == [
        "discord-123-2",
        "discord-123-1",
    ]

    live_stats = _run(
        "bot-audit",
        "stats",
        "--path",
        str(path),
        "--workspace-id",
        "123",
        "--live",
    )
    assert live_stats.exit_code == 0
    live_stats_body = json.loads(live_stats.stdout)
    assert live_stats_body["dry_run_filter"] is False
    assert live_stats_body["total"] == 2
    assert live_stats_body["dry_run"] == 0

    audit.append(
        AuditRecord(
            case_id="discord-123-old-failure",
            platform=Platform.DISCORD,
            workspace_id="123",
            channel_id="789",
            message_id="3",
            user_id="77",
            username="repeat",
            action=ModerationAction.DELETE,
            verdict=VerdictLabel.UNSAFE,
            score=0.9,
            mode=Mode.STANDARD,
            execution_status="failed",
            error="delete:Forbidden",
            created_at="2000-01-01T00:00:00+00:00",
        )
    )
    audit.append(
        AuditRecord(
            case_id="discord-123-current-failure",
            platform=Platform.DISCORD,
            workspace_id="123",
            channel_id="789",
            message_id="4",
            user_id="77",
            username="repeat",
            action=ModerationAction.TIMEOUT,
            verdict=VerdictLabel.UNSAFE,
            score=0.9,
            mode=Mode.STANDARD,
            execution_status="partial",
            error="timeout:Forbidden",
        )
    )

    failures = _run(
        "bot-audit",
        "failures",
        "--path",
        str(path),
        "--workspace-id",
        "123",
    )
    assert failures.exit_code == 0
    failures_body = json.loads(failures.stdout)
    assert [record["case_id"] for record in failures_body] == [
        "discord-123-current-failure",
        "discord-123-old-failure",
    ]

    errors = _run(
        "bot-audit",
        "errors",
        "--path",
        str(path),
        "--workspace-id",
        "123",
    )
    assert errors.exit_code == 0
    errors_body = json.loads(errors.stdout)
    assert errors_body[0]["error"] == "delete:Forbidden"
    assert errors_body[0]["total"] == 1

    windowed_failures = _run(
        "bot-audit",
        "failures",
        "--path",
        str(path),
        "--workspace-id",
        "123",
        "--since-hours",
        "1",
    )
    assert windowed_failures.exit_code == 0
    windowed_failures_body = json.loads(windowed_failures.stdout)
    assert [record["case_id"] for record in windowed_failures_body] == [
        "discord-123-current-failure"
    ]

    windowed_errors = _run(
        "bot-audit",
        "errors",
        "--path",
        str(path),
        "--workspace-id",
        "123",
        "--since-hours",
        "1",
    )
    assert windowed_errors.exit_code == 0
    windowed_errors_body = json.loads(windowed_errors.stdout)
    assert windowed_errors_body[0]["error"] == "timeout:Forbidden"
    assert windowed_errors_body[0]["total"] == 1


def test_bot_audit_cli_case_lookup(tmp_path):
    path = _seed_audit(tmp_path)

    found = _run("bot-audit", "case", "discord-123-1", "--path", str(path))
    assert found.exit_code == 0
    assert json.loads(found.stdout)["message_id"] == "1"

    missing = _run("bot-audit", "case", "missing", "--path", str(path))
    assert missing.exit_code != 0
    assert "case not found: missing" in missing.output
