"""CLI tests for the review sub-app."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from nemoguardian.cli import app

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


def test_cli_case_not_found(tmp_path):
    paths = _paths(tmp_path)
    result = runner.invoke(
        app, ["review", "case", "missing", "--path", str(paths["review"])]
    )
    assert result.exit_code != 0


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
