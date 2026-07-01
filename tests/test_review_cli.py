"""CLI tests for the review sub-app."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from nemoguardian.bot.types import Platform
from nemoguardian.cli import app
from nemoguardian.review.store import StrikeLedger

runner = CliRunner()


def _paths(tmp_path):
    return {
        "review": tmp_path / "review.jsonl",
        "appeal": tmp_path / "appeals.jsonl",
        "strike": tmp_path / "strikes.jsonl",
    }


def test_cli_enqueue_pending_resolve(tmp_path):
    paths = _paths(tmp_path)
    result = runner.invoke(
        app,
        [
            "review", "enqueue",
            "--workspace-id", "w", "--user-id", "u",
            "--text", "reach me at a@b.com", "--reason", "pii",
            "--category", "PII", "--score", "0.7",
            "--path", str(paths["review"]),
        ],
    )
    assert result.exit_code == 0, result.output
    case = json.loads(result.output)
    assert case["status"] == "pending"
    assert "a@b.com" not in case["text_excerpt"]
    case_id = case["case_id"]

    pending = runner.invoke(
        app, ["review", "pending", "--workspace-id", "w", "--path", str(paths["review"])]
    )
    assert pending.exit_code == 0
    assert len(json.loads(pending.output)) == 1

    resolved = runner.invoke(
        app,
        [
            "review", "resolve", case_id, "--status", "removed", "--strike",
            "--review-path", str(paths["review"]), "--strike-path", str(paths["strike"]),
        ],
    )
    assert resolved.exit_code == 0
    assert json.loads(resolved.output)["status"] == "removed"

    # The strike landed -> the user is recommended for a warning.
    strikes = runner.invoke(
        app,
        ["review", "strikes", "--workspace-id", "w", "--user-id", "u", "--path", str(paths["strike"])],
    )
    assert strikes.exit_code == 0
    assert json.loads(strikes.output)["action"] == "warn"


def test_cli_appeal_restore(tmp_path):
    paths = _paths(tmp_path)
    enq = runner.invoke(
        app,
        ["review", "enqueue", "--workspace-id", "w", "--user-id", "u", "--path", str(paths["review"])],
    )
    case_id = json.loads(enq.output)["case_id"]
    runner.invoke(
        app,
        [
            "review", "resolve", case_id, "--status", "removed", "--strike",
            "--review-path", str(paths["review"]), "--strike-path", str(paths["strike"]),
        ],
    )
    appeal = runner.invoke(
        app,
        [
            "review", "appeal", case_id, "--outcome", "restored", "--reason", "fp",
            "--review-path", str(paths["review"]),
            "--appeal-path", str(paths["appeal"]),
            "--strike-path", str(paths["strike"]),
        ],
    )
    assert appeal.exit_code == 0, appeal.output
    payload = json.loads(appeal.output)
    assert payload["case"]["status"] == "restored"
    assert len(payload["voided_strike_ids"]) == 1

    # Strike was voided by the restore.
    strikes = runner.invoke(
        app,
        ["review", "strikes", "--workspace-id", "w", "--user-id", "u", "--path", str(paths["strike"])],
    )
    assert json.loads(strikes.output)["total"] == 0

    appeals = runner.invoke(
        app, ["review", "appeals", "--case-id", case_id, "--path", str(paths["appeal"])]
    )
    assert len(json.loads(appeals.output)) == 1


def test_cli_case_success(tmp_path):
    paths = _paths(tmp_path)
    enq = runner.invoke(
        app,
        ["review", "enqueue", "--workspace-id", "w", "--user-id", "u", "--path", str(paths["review"])],
    )
    case_id = json.loads(enq.output)["case_id"]

    result = runner.invoke(app, ["review", "case", case_id, "--path", str(paths["review"])])

    assert result.exit_code == 0
    assert json.loads(result.output)["case_id"] == case_id


def test_cli_case_not_found(tmp_path):
    paths = _paths(tmp_path)
    result = runner.invoke(
        app, ["review", "case", "missing", "--path", str(paths["review"])]
    )
    assert result.exit_code != 0


def test_cli_resolve_missing_case_reports_error(tmp_path):
    paths = _paths(tmp_path)
    result = runner.invoke(
        app,
        [
            "review", "resolve", "missing", "--status", "removed",
            "--review-path", str(paths["review"]),
            "--strike-path", str(paths["strike"]),
        ],
    )

    assert result.exit_code != 0
    assert "review case not found" in result.output


def test_cli_appeal_missing_case_reports_error(tmp_path):
    paths = _paths(tmp_path)
    result = runner.invoke(
        app,
        [
            "review", "appeal", "missing", "--outcome", "upheld",
            "--review-path", str(paths["review"]),
            "--appeal-path", str(paths["appeal"]),
            "--strike-path", str(paths["strike"]),
        ],
    )

    assert result.exit_code != 0
    assert "review case not found" in result.output


def test_cli_counts(tmp_path):
    paths = _paths(tmp_path)
    runner.invoke(
        app,
        ["review", "enqueue", "--workspace-id", "w", "--user-id", "u", "--path", str(paths["review"])],
    )
    result = runner.invoke(
        app, ["review", "counts", "--workspace-id", "w", "--path", str(paths["review"])]
    )
    assert result.exit_code == 0
    assert json.loads(result.output)["pending"] == 1


def test_cli_appeals_workspace_listing(tmp_path):
    paths = _paths(tmp_path)
    enq = runner.invoke(
        app,
        ["review", "enqueue", "--workspace-id", "w", "--user-id", "u", "--path", str(paths["review"])],
    )
    case_id = json.loads(enq.output)["case_id"]
    runner.invoke(
        app,
        [
            "review", "resolve", case_id, "--status", "removed",
            "--review-path", str(paths["review"]),
            "--strike-path", str(paths["strike"]),
        ],
    )
    runner.invoke(
        app,
        [
            "review", "appeal", case_id, "--outcome", "upheld",
            "--review-path", str(paths["review"]),
            "--appeal-path", str(paths["appeal"]),
            "--strike-path", str(paths["strike"]),
        ],
    )

    result = runner.invoke(
        app,
        [
            "review", "appeals", "--workspace-id", "w",
            "--platform", "discord", "--path", str(paths["appeal"]),
        ],
    )

    assert result.exit_code == 0
    assert len(json.loads(result.output)) == 1


def test_cli_strikes_leaderboard(tmp_path):
    paths = _paths(tmp_path)
    ledger = StrikeLedger(paths["strike"])
    ledger.add_strike(platform=Platform.DISCORD, workspace_id="w", user_id="light")
    ledger.add_strike(platform=Platform.DISCORD, workspace_id="w", user_id="heavy")
    ledger.add_strike(platform=Platform.DISCORD, workspace_id="w", user_id="heavy")

    result = runner.invoke(
        app,
        ["review", "strikes", "--workspace-id", "w", "--path", str(paths["strike"])],
    )

    assert result.exit_code == 0
    rows = json.loads(result.output)
    assert rows[0]["user_id"] == "heavy"
    assert rows[0]["total"] == 2
