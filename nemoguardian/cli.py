"""nemoguardian CLI.

Use:
    nemoguardian serve [--port 8000] [--host 127.0.0.1]
    nemoguardian demo [--text "..."]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer

from nemoguardian.bot import AuditLog, Platform, since_hours_ago
from nemoguardian.cascade import Cascade, CascadeConfig
from nemoguardian.policy.presets import get_preset
from nemoguardian.schemas import Mode, ModerateRequest

app = typer.Typer(help="Multi-model LLM moderation cascade.")
audit_app = typer.Typer(help="Inspect moderation bot audit logs.")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    workers: int = typer.Option(1, help="Uvicorn workers"),
    log_level: str = typer.Option("info"),
) -> None:
    """Run the FastAPI server."""
    import uvicorn

    uvicorn.run(
        "nemoguardian.server:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        reload=False,
    )


@app.command()
def demo(
    text: str = typer.Option(
        "Drop your SSN in chat and I'll send you $100!",
        help="Text to moderate.",
    ),
    policy: str = typer.Option(
        "block PII and financial scams",
        help="Custom safety policy.",
    ),
    mode: Mode = typer.Option(Mode.STANDARD, help="Cascade mode"),
    preset: str = typer.Option("discord", help="Policy preset"),
) -> None:
    """Run the cascade on a single text and print the verdict."""
    cascade = Cascade(CascadeConfig.from_env())
    request = ModerateRequest(text=text, policy=policy, mode=mode)
    policy_engine = get_preset(preset)
    result = cascade.moderate(request, policy_engine=policy_engine)
    typer.echo(json.dumps(result.model_dump(), indent=2, default=str))


@app.command("discord-bot")
def discord_bot() -> None:
    """Run the Discord moderation bot."""
    from nemoguardian.adapters.discord import run_bot

    run_bot()


@audit_app.command("case")
def audit_case(
    case_id: str = typer.Argument(..., help="Case ID to look up."),
    path: Path | None = typer.Option(None, "--path", help="Audit JSONL path."),
) -> None:
    """Print one audit case as JSON."""
    record = AuditLog(path).find_case(case_id)
    if record is None:
        raise typer.BadParameter(f"case not found: {case_id}")
    _echo_json(record)


@audit_app.command("history")
def audit_history(
    workspace_id: str = typer.Option(..., "--workspace-id", help="Platform workspace/guild/channel ID."),
    platform: Platform = typer.Option(Platform.DISCORD, "--platform", help="Bot platform."),
    user_id: str | None = typer.Option(None, "--user-id", help="Optional user ID filter."),
    channel_id: str | None = typer.Option(None, "--channel-id", help="Optional channel ID filter."),
    category: str | None = typer.Option(None, "--category", help="Optional moderation category filter."),
    rule: str | None = typer.Option(None, "--rule", help="Optional matched policy rule filter."),
    action: str | None = typer.Option(None, "--action", help="Optional moderation action filter."),
    verdict: str | None = typer.Option(None, "--verdict", help="Optional moderation verdict filter."),
    status: str | None = typer.Option(None, "--status", help="Optional execution status filter."),
    dry_run: bool | None = typer.Option(None, "--dry-run/--live", help="Optional dry-run or live-case filter."),
    limit: int = typer.Option(10, "--limit", min=1, max=100, help="Maximum records to print."),
    since_hours: float | None = typer.Option(None, "--since-hours", min=0.0, help="Only include newer cases."),
    path: Path | None = typer.Option(None, "--path", help="Audit JSONL path."),
) -> None:
    """Print recent audit cases as JSON."""
    records = AuditLog(path).history(
        platform,
        workspace_id,
        user_id=user_id,
        channel_id=channel_id,
        category=category,
        rule=rule,
        action=action,
        verdict=verdict,
        status=status,
        dry_run=dry_run,
        limit=limit,
        since=since_hours_ago(since_hours),
    )
    _echo_json(records)


@audit_app.command("stats")
def audit_stats(
    workspace_id: str = typer.Option(..., "--workspace-id", help="Platform workspace/guild/channel ID."),
    platform: Platform = typer.Option(Platform.DISCORD, "--platform", help="Bot platform."),
    user_id: str | None = typer.Option(None, "--user-id", help="Optional user ID filter."),
    channel_id: str | None = typer.Option(None, "--channel-id", help="Optional channel ID filter."),
    category: str | None = typer.Option(None, "--category", help="Optional moderation category filter."),
    rule: str | None = typer.Option(None, "--rule", help="Optional matched policy rule filter."),
    action: str | None = typer.Option(None, "--action", help="Optional moderation action filter."),
    verdict: str | None = typer.Option(None, "--verdict", help="Optional moderation verdict filter."),
    status: str | None = typer.Option(None, "--status", help="Optional execution status filter."),
    dry_run: bool | None = typer.Option(None, "--dry-run/--live", help="Optional dry-run or live-case filter."),
    limit: int = typer.Option(100, "--limit", min=1, max=1_000, help="Recent cases to summarize."),
    since_hours: float | None = typer.Option(None, "--since-hours", min=0.0, help="Only include newer cases."),
    path: Path | None = typer.Option(None, "--path", help="Audit JSONL path."),
) -> None:
    """Print audit case counts as JSON."""
    summary = AuditLog(path).summary(
        platform,
        workspace_id,
        user_id=user_id,
        channel_id=channel_id,
        category=category,
        rule=rule,
        action=action,
        verdict=verdict,
        status=status,
        dry_run=dry_run,
        limit=limit,
        since=since_hours_ago(since_hours),
    )
    _echo_json(summary)


@audit_app.command("failures")
def audit_failures(
    workspace_id: str = typer.Option(..., "--workspace-id", help="Platform workspace/guild/channel ID."),
    platform: Platform = typer.Option(Platform.DISCORD, "--platform", help="Bot platform."),
    limit: int = typer.Option(10, "--limit", min=1, max=100, help="Maximum failed records to print."),
    since_hours: float | None = typer.Option(None, "--since-hours", min=0.0, help="Only include newer cases."),
    path: Path | None = typer.Option(None, "--path", help="Audit JSONL path."),
) -> None:
    """Print recent failed or partial moderation actions as JSON."""
    records = AuditLog(path).failures(
        platform,
        workspace_id,
        limit=limit,
        since=since_hours_ago(since_hours),
    )
    _echo_json(records)


@audit_app.command("dry-runs")
def audit_dry_runs(
    workspace_id: str = typer.Option(..., "--workspace-id", help="Platform workspace/guild/channel ID."),
    platform: Platform = typer.Option(Platform.DISCORD, "--platform", help="Bot platform."),
    limit: int = typer.Option(10, "--limit", min=1, max=100, help="Maximum dry-run action cases to print."),
    since_hours: float | None = typer.Option(None, "--since-hours", min=0.0, help="Only include newer cases."),
    path: Path | None = typer.Option(None, "--path", help="Audit JSONL path."),
) -> None:
    """Print dry-run cases that would take moderation action as JSON."""
    records = AuditLog(path).dry_run_cases(
        platform,
        workspace_id,
        limit=limit,
        since=since_hours_ago(since_hours),
    )
    _echo_json(records)


@audit_app.command("errors")
def audit_errors(
    workspace_id: str = typer.Option(..., "--workspace-id", help="Platform workspace/guild/channel ID."),
    platform: Platform = typer.Option(Platform.DISCORD, "--platform", help="Bot platform."),
    limit: int = typer.Option(10, "--limit", min=1, max=50, help="Maximum error types to print."),
    case_limit: int = typer.Option(500, "--case-limit", min=1, max=5_000, help="Recent failed cases to inspect."),
    since_hours: float | None = typer.Option(None, "--since-hours", min=0.0, help="Only include newer cases."),
    path: Path | None = typer.Option(None, "--path", help="Audit JSONL path."),
) -> None:
    """Print recurring moderation execution errors as JSON."""
    rows = AuditLog(path).top_errors(
        platform,
        workspace_id,
        limit=limit,
        case_limit=case_limit,
        since=since_hours_ago(since_hours),
    )
    _echo_json(rows)


@audit_app.command("slow-cases")
def audit_slow_cases(
    workspace_id: str = typer.Option(..., "--workspace-id", help="Platform workspace/guild/channel ID."),
    platform: Platform = typer.Option(Platform.DISCORD, "--platform", help="Bot platform."),
    limit: int = typer.Option(10, "--limit", min=1, max=100, help="Maximum slow cases to print."),
    case_limit: int = typer.Option(500, "--case-limit", min=1, max=5_000, help="Recent cases to inspect."),
    since_hours: float | None = typer.Option(None, "--since-hours", min=0.0, help="Only include newer cases."),
    path: Path | None = typer.Option(None, "--path", help="Audit JSONL path."),
) -> None:
    """Print recent cases with the highest recorded latency as JSON."""
    records = AuditLog(path).slow_cases(
        platform,
        workspace_id,
        limit=limit,
        case_limit=case_limit,
        since=since_hours_ago(since_hours),
    )
    _echo_json(records)


@audit_app.command("offenders")
def audit_offenders(
    workspace_id: str = typer.Option(..., "--workspace-id", help="Platform workspace/guild/channel ID."),
    platform: Platform = typer.Option(Platform.DISCORD, "--platform", help="Bot platform."),
    limit: int = typer.Option(10, "--limit", min=1, max=50, help="Maximum users to print."),
    case_limit: int = typer.Option(500, "--case-limit", min=1, max=5_000, help="Recent cases to inspect."),
    since_hours: float | None = typer.Option(None, "--since-hours", min=0.0, help="Only include newer cases."),
    path: Path | None = typer.Option(None, "--path", help="Audit JSONL path."),
) -> None:
    """Print users with the most recent moderation cases as JSON."""
    rows = AuditLog(path).top_users(
        platform,
        workspace_id,
        limit=limit,
        case_limit=case_limit,
        since=since_hours_ago(since_hours),
    )
    _echo_json(rows)


@audit_app.command("channels")
def audit_channels(
    workspace_id: str = typer.Option(..., "--workspace-id", help="Platform workspace/guild/channel ID."),
    platform: Platform = typer.Option(Platform.DISCORD, "--platform", help="Bot platform."),
    limit: int = typer.Option(10, "--limit", min=1, max=50, help="Maximum channels to print."),
    case_limit: int = typer.Option(500, "--case-limit", min=1, max=5_000, help="Recent cases to inspect."),
    since_hours: float | None = typer.Option(None, "--since-hours", min=0.0, help="Only include newer cases."),
    path: Path | None = typer.Option(None, "--path", help="Audit JSONL path."),
) -> None:
    """Print channels with the most recent moderation cases as JSON."""
    rows = AuditLog(path).top_channels(
        platform,
        workspace_id,
        limit=limit,
        case_limit=case_limit,
        since=since_hours_ago(since_hours),
    )
    _echo_json(rows)


@audit_app.command("rules")
def audit_rules(
    workspace_id: str = typer.Option(..., "--workspace-id", help="Platform workspace/guild/channel ID."),
    platform: Platform = typer.Option(Platform.DISCORD, "--platform", help="Bot platform."),
    limit: int = typer.Option(10, "--limit", min=1, max=50, help="Maximum policy rules to print."),
    case_limit: int = typer.Option(500, "--case-limit", min=1, max=5_000, help="Recent cases to inspect."),
    since_hours: float | None = typer.Option(None, "--since-hours", min=0.0, help="Only include newer cases."),
    path: Path | None = typer.Option(None, "--path", help="Audit JSONL path."),
) -> None:
    """Print policy rules with the most recent moderation cases as JSON."""
    rows = AuditLog(path).top_rules(
        platform,
        workspace_id,
        limit=limit,
        case_limit=case_limit,
        since=since_hours_ago(since_hours),
    )
    _echo_json(rows)


@audit_app.command("categories")
def audit_categories(
    workspace_id: str = typer.Option(..., "--workspace-id", help="Platform workspace/guild/channel ID."),
    platform: Platform = typer.Option(Platform.DISCORD, "--platform", help="Bot platform."),
    limit: int = typer.Option(10, "--limit", min=1, max=50, help="Maximum categories to print."),
    case_limit: int = typer.Option(500, "--case-limit", min=1, max=5_000, help="Recent cases to inspect."),
    since_hours: float | None = typer.Option(None, "--since-hours", min=0.0, help="Only include newer cases."),
    path: Path | None = typer.Option(None, "--path", help="Audit JSONL path."),
) -> None:
    """Print moderation categories with the most recent cases as JSON."""
    rows = AuditLog(path).top_categories(
        platform,
        workspace_id,
        limit=limit,
        case_limit=case_limit,
        since=since_hours_ago(since_hours),
    )
    _echo_json(rows)


def _echo_json(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


app.add_typer(audit_app, name="bot-audit")


if __name__ == "__main__":
    app()
