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
            execution_status="delete+public-warning",
            created_at="2000-01-01T00:00:00+00:00",
        )
    )
    audit.append(
        AuditRecord(
            case_id="discord-123-2",
            platform=Platform.DISCORD,
            workspace_id="123",
            channel_id="456",
            message_id="2",
            user_id="42",
            username="tester",
            action=ModerationAction.FLAG,
            verdict=VerdictLabel.CONTROVERSIAL,
            score=0.67,
            mode=Mode.STANDARD,
            categories=["harassment"],
            matched_policy_rule="watch-harassment",
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
    assert channels_body[0]["total"] == 2

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


def test_bot_audit_cli_case_lookup(tmp_path):
    path = _seed_audit(tmp_path)

    found = _run("bot-audit", "case", "discord-123-1", "--path", str(path))
    assert found.exit_code == 0
    assert json.loads(found.stdout)["message_id"] == "1"

    missing = _run("bot-audit", "case", "missing", "--path", str(path))
    assert missing.exit_code != 0
    assert "case not found: missing" in missing.output
