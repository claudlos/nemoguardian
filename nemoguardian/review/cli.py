"""Typer sub-app for the human-in-the-loop review layer.

Registered in :mod:`nemoguardian.cli` with a single ``app.add_typer`` line so it
stays merge-conflict-friendly. Commands let a moderator inspect the queue,
resolve cases, file appeals, and read the strike ladder from the shell.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from nemoguardian.bot.types import Platform
from nemoguardian.review.models import AppealOutcome, CaseStatus
from nemoguardian.review.service import ReviewService
from nemoguardian.review.store import AppealLog, ReviewQueue, StrikeLedger

review_app = typer.Typer(help="Moderator review queue, appeals, and strike tracking.")


def _echo_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


@review_app.command("enqueue")
def review_enqueue(
    workspace_id: str = typer.Option(..., "--workspace-id", help="Platform workspace/guild ID."),
    user_id: str = typer.Option(..., "--user-id", help="Offending user ID."),
    platform: Platform = typer.Option(Platform.DISCORD, "--platform", help="Bot platform."),
    text: str | None = typer.Option(None, "--text", help="Message text (PII-redacted before store)."),
    reason: str = typer.Option("", "--reason", help="Why the case was flagged."),
    category: list[str] = typer.Option(None, "--category", help="Moderation category (repeatable)."),
    score: float = typer.Option(0.0, "--score", min=0.0, max=1.0, help="Cascade score."),
    path: Path | None = typer.Option(None, "--path", help="Review queue JSONL path."),
) -> None:
    """Enqueue a flagged case for human review."""
    case = ReviewQueue(path).enqueue(
        platform=platform,
        workspace_id=workspace_id,
        user_id=user_id,
        text=text,
        reason=reason,
        categories=list(category or []),
        score=score,
    )
    _echo_json(case.to_dict())


@review_app.command("pending")
def review_pending(
    workspace_id: str | None = typer.Option(None, "--workspace-id", help="Workspace filter."),
    platform: Platform | None = typer.Option(None, "--platform", help="Platform filter."),
    user_id: str | None = typer.Option(None, "--user-id", help="User filter."),
    category: str | None = typer.Option(None, "--category", help="Category filter."),
    limit: int = typer.Option(50, "--limit", min=1, max=1_000, help="Maximum cases to print."),
    path: Path | None = typer.Option(None, "--path", help="Review queue JSONL path."),
) -> None:
    """List cases awaiting moderator review."""
    cases = ReviewQueue(path).list_pending(
        platform=platform,
        workspace_id=workspace_id,
        user_id=user_id,
        category=category,
        limit=limit,
    )
    _echo_json([case.to_dict() for case in cases])


@review_app.command("case")
def review_case(
    case_id: str = typer.Argument(..., help="Review case ID."),
    path: Path | None = typer.Option(None, "--path", help="Review queue JSONL path."),
) -> None:
    """Print one review case as JSON."""
    case = ReviewQueue(path).get_case(case_id)
    if case is None:
        raise typer.BadParameter(f"review case not found: {case_id}")
    _echo_json(case.to_dict())


@review_app.command("counts")
def review_counts(
    workspace_id: str | None = typer.Option(None, "--workspace-id", help="Workspace filter."),
    platform: Platform | None = typer.Option(None, "--platform", help="Platform filter."),
    path: Path | None = typer.Option(None, "--path", help="Review queue JSONL path."),
) -> None:
    """Print queue status counts as JSON."""
    _echo_json(ReviewQueue(path).counts(platform=platform, workspace_id=workspace_id))


@review_app.command("resolve")
def review_resolve(
    case_id: str = typer.Argument(..., help="Review case ID."),
    status: CaseStatus = typer.Option(..., "--status", help="approved / removed / restored."),
    resolved_by: str | None = typer.Option(None, "--by", help="Moderator ID."),
    note: str = typer.Option("", "--note", help="Resolution note."),
    add_strike: bool = typer.Option(False, "--strike", help="Add a strike (only when removed)."),
    strike_weight: float = typer.Option(1.0, "--weight", min=0.0, help="Strike weight."),
    review_path: Path | None = typer.Option(None, "--review-path", help="Review queue JSONL path."),
    strike_path: Path | None = typer.Option(None, "--strike-path", help="Strike ledger JSONL path."),
) -> None:
    """Resolve a queued case, optionally recording a strike."""
    service = ReviewService(queue=ReviewQueue(review_path), strikes=StrikeLedger(strike_path))
    try:
        case = service.resolve(
            case_id,
            status,
            resolved_by=resolved_by,
            note=note,
            add_strike=add_strike,
            strike_weight=strike_weight,
        )
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _echo_json(case.to_dict())


@review_app.command("appeal")
def review_appeal(
    case_id: str = typer.Argument(..., help="Review case ID to appeal."),
    outcome: AppealOutcome = typer.Option(..., "--outcome", help="restored / upheld."),
    reason: str = typer.Option("", "--reason", help="Appeal reason / decision rationale."),
    decided_by: str | None = typer.Option(None, "--by", help="Moderator ID."),
    keep_strikes: bool = typer.Option(False, "--keep-strikes", help="Do not void linked strikes."),
    review_path: Path | None = typer.Option(None, "--review-path", help="Review queue JSONL path."),
    appeal_path: Path | None = typer.Option(None, "--appeal-path", help="Appeal log JSONL path."),
    strike_path: Path | None = typer.Option(None, "--strike-path", help="Strike ledger JSONL path."),
) -> None:
    """File an appeal and apply the restore flow."""
    service = ReviewService(
        queue=ReviewQueue(review_path),
        appeals=AppealLog(appeal_path),
        strikes=StrikeLedger(strike_path),
    )
    try:
        result = service.appeal(
            case_id,
            outcome,
            reason=reason,
            decided_by=decided_by,
            void_strikes=not keep_strikes,
        )
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc
    _echo_json(
        {
            "decision": result.decision.to_dict(),
            "case": result.case.to_dict(),
            "voided_strike_ids": [event.strike_id for event in result.voided_strikes],
        }
    )


@review_app.command("appeals")
def review_appeals(
    case_id: str | None = typer.Option(None, "--case-id", help="Filter to one review case."),
    workspace_id: str | None = typer.Option(None, "--workspace-id", help="Workspace filter."),
    platform: Platform | None = typer.Option(None, "--platform", help="Platform filter."),
    limit: int = typer.Option(50, "--limit", min=1, max=1_000, help="Maximum decisions to print."),
    path: Path | None = typer.Option(None, "--path", help="Appeal log JSONL path."),
) -> None:
    """List appeal/restore decisions as JSON."""
    log = AppealLog(path)
    if case_id is not None:
        decisions = log.for_case(case_id)
    else:
        decisions = log.list_appeals(platform=platform, workspace_id=workspace_id, limit=limit)
    _echo_json([decision.to_dict() for decision in decisions])


@review_app.command("strikes")
def review_strikes(
    workspace_id: str = typer.Option(..., "--workspace-id", help="Workspace ID."),
    platform: Platform = typer.Option(Platform.DISCORD, "--platform", help="Bot platform."),
    user_id: str | None = typer.Option(None, "--user-id", help="Show one user's recommendation."),
    limit: int = typer.Option(10, "--limit", min=1, max=100, help="Leaderboard size."),
    path: Path | None = typer.Option(None, "--path", help="Strike ledger JSONL path."),
) -> None:
    """Show the strike recommendation for a user, or the workspace leaderboard."""
    ledger = StrikeLedger(path)
    if user_id is not None:
        _echo_json(ledger.recommend(platform, workspace_id, user_id))
    else:
        _echo_json(ledger.leaderboard(platform, workspace_id, limit=limit))


__all__ = ["review_app"]
