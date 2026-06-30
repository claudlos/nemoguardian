"""GPU operator CLI — spend caps, status, health, teardown, watchdog.

Registered onto the root Typer app via a single ``app.add_typer`` line in
``cli.py`` (keeps merge conflicts to one line). All commands operate through
:mod:`nemoguardian.providers.ops`, which enforces the spend caps and never
auto-spends without an explicit ``--confirm``.

    nemoguardian gpu-ops provision --gpu "RTX 3090" --price-cents 7 --vram 24 --hours 6
    nemoguardian gpu-ops provision ... --confirm        # actually launch
    nemoguardian gpu-ops status   vastai-abc123
    nemoguardian gpu-ops health   https://abc123.nemoguardian.dev
    nemoguardian gpu-ops teardown vastai-abc123 --reason done
    nemoguardian gpu-ops watchdog vastai-abc123 --max-hours 6
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import typer

from nemoguardian.providers import ops
from nemoguardian.providers.base import Offer, ProviderName
from nemoguardian.providers.registry import get_provider

ops_app = typer.Typer(help="GPU operator guardrails: caps, teardown, watchdog, health.")


def _echo(payload: Any) -> None:
    typer.echo(json.dumps(payload, indent=2, sort_keys=True, default=str))


def _event_log(path: str | None) -> ops.OpsEventLog | None:
    return ops.OpsEventLog(path) if path else None


@ops_app.command()
def provision(
    gpu: str = typer.Option(..., "--gpu", help="GPU model, e.g. 'RTX 3090'."),
    price_cents: int = typer.Option(..., "--price-cents", help="Offer price in cents/hour."),
    vram: int = typer.Option(24, "--vram", help="GPU VRAM in GB."),
    hours: float = typer.Option(6.0, "--hours", help="Reservation length in hours."),
    region: str = typer.Option("Global", "--region", help="Offer region."),
    offer_id: str = typer.Option("", "--offer-id", help="Provider-specific offer id."),
    provider: ProviderName = typer.Option(ProviderName.VAST_AI, "--provider", help="GPU provider."),
    confirm: bool = typer.Option(
        False, "--confirm", help="ACTUALLY provision (spend). Default is a dry-run plan."
    ),
    event_path: str | None = typer.Option(None, "--event-log", help="Append ops events to this JSONL path."),
) -> None:
    """Plan (default) or provision (``--confirm``) an offer, enforcing spend caps.

    Without ``--confirm`` this never touches the provider — it returns a PLANNED
    dry-run. Caps that reject the offer always win (no spend, exit code 2).
    """
    config = ops.GpuOpsConfig.from_env()
    offer = Offer(
        provider=provider,
        gpu_model=gpu,
        vram_gb=vram,
        price_per_hour_usd=price_cents / 100.0,
        region=region,
        offer_id=offer_id,
    )
    result = asyncio.run(
        ops.provision_guarded(
            get_provider(provider),
            offer,
            config=config,
            reserve_hours=hours,
            confirm=confirm,
            event_log=_event_log(event_path),
        )
    )
    _echo(result.to_dict())
    # Non-zero exit when nothing was provisioned and a cap rejected it, so
    # shell callers / CI can gate on overspend.
    if result.status is ops.ProvisionStatus.REJECTED:
        raise typer.Exit(code=2)
    if result.status is ops.ProvisionStatus.FAILED:
        raise typer.Exit(code=1)


@ops_app.command()
def status(
    instance_id: str = typer.Argument(..., help="Provider instance id."),
    provider: ProviderName = typer.Option(ProviderName.VAST_AI, "--provider", help="GPU provider."),
    max_attempts: int | None = typer.Option(None, "--max-attempts", help="Poll attempt ceiling."),
    interval: float | None = typer.Option(None, "--interval", help="Seconds between polls."),
) -> None:
    """Poll an instance until it is LIVE (or terminal / attempts exhausted)."""
    config = ops.GpuOpsConfig.from_env()
    result = asyncio.run(
        ops.poll_until_running(
            get_provider(provider),
            instance_id,
            config=config,
            max_attempts=max_attempts,
            interval=interval,
        )
    )
    _echo(
        {
            "instance_id": result.instance_id,
            "state": result.state.value,
            "uptime_seconds": result.uptime_seconds,
            "last_health_check": result.last_health_check,
            "error_message": result.error_message,
        }
    )


@ops_app.command()
def health(
    endpoint_url: str = typer.Argument(..., help="Instance base URL, e.g. https://host.nemoguardian.dev"),
    path: str | None = typer.Option(None, "--path", help="Health path (default /health)."),
    timeout: float | None = typer.Option(None, "--timeout", help="Request timeout seconds."),
) -> None:
    """Probe an instance's health endpoint. Exit non-zero when unhealthy."""
    config = ops.GpuOpsConfig.from_env()
    result = asyncio.run(
        ops.health_check(endpoint_url, config=config, path=path, timeout=timeout)
    )
    _echo(result.to_dict())
    if not result.ok:
        raise typer.Exit(code=1)


@ops_app.command()
def teardown(
    instance_id: str = typer.Argument(..., help="Provider instance id to destroy."),
    provider: ProviderName = typer.Option(ProviderName.VAST_AI, "--provider", help="GPU provider."),
    reason: str = typer.Option("manual", "--reason", help="Why the instance is being torn down."),
    event_path: str | None = typer.Option(None, "--event-log", help="Append ops events to this JSONL path."),
) -> None:
    """Destroy a rented instance (idempotent)."""
    result = asyncio.run(
        ops.teardown(
            get_provider(provider),
            instance_id,
            reason=reason,
            event_log=_event_log(event_path),
        )
    )
    _echo(result.to_dict())
    if not result.ok:
        raise typer.Exit(code=1)


@ops_app.command()
def watchdog(
    instance_id: str = typer.Argument(..., help="Provider instance id to watch."),
    provider: ProviderName = typer.Option(ProviderName.VAST_AI, "--provider", help="GPU provider."),
    max_hours: float | None = typer.Option(None, "--max-hours", help="Override max reserve hours."),
    max_checks: int = typer.Option(100_000, "--max-checks", help="Watchdog iteration ceiling."),
    event_path: str | None = typer.Option(None, "--event-log", help="Append ops events to this JSONL path."),
) -> None:
    """Watch an instance and auto-tear-down once it exceeds the reserve window.

    Blocks until a guardrail trips (time cap / terminal state). Intended to run
    as a sidecar next to a rented instance so it can never run forever.
    """
    config = ops.GpuOpsConfig.from_env()
    if max_hours is not None:
        config = ops.replace_max_reserve_hours(config, max_hours)
    result = asyncio.run(
        ops.watchdog(
            get_provider(provider),
            instance_id,
            config=config,
            max_checks=max_checks,
            event_log=_event_log(event_path),
        )
    )
    _echo(result.to_dict())


__all__ = ["ops_app"]
