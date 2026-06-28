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
            execution_status="delete+public-warning",
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


def test_bot_audit_cli_case_lookup(tmp_path):
    path = _seed_audit(tmp_path)

    found = _run("bot-audit", "case", "discord-123-1", "--path", str(path))
    assert found.exit_code == 0
    assert json.loads(found.stdout)["message_id"] == "1"

    missing = _run("bot-audit", "case", "missing", "--path", str(path))
    assert missing.exit_code != 0
    assert "case not found: missing" in missing.output
