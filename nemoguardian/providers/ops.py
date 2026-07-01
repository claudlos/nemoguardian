"""GPU operator guardrails — spend caps, teardown, watchdog, health.

Rented GPUs are a real spend hazard: an instance left running (or provisioned
above budget) silently bleeds money. This module is the safety layer between
the rest of nemoguardian and a live provider API:

* **Spend caps** (audit TODO #45): refuse to provision an offer above a price
  cap or a reservation longer than a max-hours cap, with a clear reason. The
  default posture is **no auto-spend** — a live provision requires an explicit
  ``confirm=True`` (or the ``NEMOGUARDIAN_OPS_CONFIRM`` env gate). Without it we
  return a *planned* dry-run result and touch no money.
* **Teardown + watchdog** (#46): an idempotent teardown plus a watchdog loop
  that auto-destroys an instance once it exceeds ``max_reserve_hours``, goes
  idle, or signals completion ("destroy on completion").
* **Status polling + health** (#47): poll a provider until an instance is LIVE
  and probe its ``/health`` endpoint. All HTTP/clients are injectable so unit
  tests need no live calls.

Design rules mirrored from the adapters: these helpers **fail safe**. Polling,
health checks, and teardown never raise on transport failure — they surface the
problem in the returned result, and a failed health/poll is treated as
*not healthy / not running* rather than as success.

Persistence reuses the append-only JSONL pattern from ``bot/audit.py``: a
:class:`OpsEvent` dataclass is serialised one-per-line via :class:`OpsEventLog`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
import json
import os
import threading
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol

import httpx

from nemoguardian.providers.base import (
    Instance,
    InstanceState,
    InstanceStatus,
    Offer,
    ProvisionError,
)

# --------------------------------------------------------------------------- #
# Config — the spend caps                                                      #
# --------------------------------------------------------------------------- #


def _env_flag(name: str) -> bool:
    """Truthy-string env gate ("1"/"true"/"yes"/"on", case-insensitive)."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class GpuOpsConfig:
    """Operator guardrail configuration.

    Prices are kept in **integer cents/hour** so a cap comparison is exact and
    free of float drift. Hours are floats (fractional reservations are fine).
    """

    max_hourly_price_cents: int = 50          # refuse offers above $0.50/hr
    max_reserve_hours: float = 12.0           # refuse / tear down past this
    max_idle_seconds: float | None = 1800.0   # tear down after 30m idle (None = off)
    poll_interval_seconds: float = 5.0        # status poll / watchdog cadence
    poll_max_attempts: int = 60               # status poll attempt ceiling
    health_path: str = "/health"
    health_timeout_seconds: float = 10.0
    require_confirm: bool = True              # no auto-spend unless explicitly confirmed

    @classmethod
    def from_env(cls) -> GpuOpsConfig:
        """Build config from ``NEMOGUARDIAN_OPS_*`` env vars (all optional)."""

        def _f(name: str, default: float) -> float:
            raw = os.environ.get(name)
            return float(raw) if raw not in (None, "") else default

        def _i(name: str, default: int) -> int:
            raw = os.environ.get(name)
            return int(raw) if raw not in (None, "") else default

        max_idle_raw = os.environ.get("NEMOGUARDIAN_OPS_MAX_IDLE_SECONDS")
        max_idle: float | None
        if max_idle_raw in (None, ""):
            max_idle = 1800.0
        elif max_idle_raw.strip().lower() in {"none", "off", "0"}:
            max_idle = None
        else:
            max_idle = float(max_idle_raw)

        return cls(
            max_hourly_price_cents=_i("NEMOGUARDIAN_OPS_MAX_HOURLY_PRICE_CENTS", 50),
            max_reserve_hours=_f("NEMOGUARDIAN_OPS_MAX_RESERVE_HOURS", 12.0),
            max_idle_seconds=max_idle,
            poll_interval_seconds=_f("NEMOGUARDIAN_OPS_POLL_INTERVAL_SECONDS", 5.0),
            poll_max_attempts=_i("NEMOGUARDIAN_OPS_POLL_MAX_ATTEMPTS", 60),
            health_path=os.environ.get("NEMOGUARDIAN_OPS_HEALTH_PATH", "/health"),
            health_timeout_seconds=_f("NEMOGUARDIAN_OPS_HEALTH_TIMEOUT_SECONDS", 10.0),
            # require_confirm stays True unless the env gate is set.
            require_confirm=not _env_flag("NEMOGUARDIAN_OPS_CONFIRM"),
        )


def replace_max_reserve_hours(config: GpuOpsConfig, max_reserve_hours: float) -> GpuOpsConfig:
    """Return a copy of ``config`` with a different reserve-hours cap (frozen-safe)."""
    return dataclasses.replace(config, max_reserve_hours=max_reserve_hours)


# --------------------------------------------------------------------------- #
# Result types                                                                 #
# --------------------------------------------------------------------------- #


class ProvisionStatus(str, Enum):
    PLANNED = "planned"          # dry-run; caps OK but not confirmed → no spend
    PROVISIONED = "provisioned"  # live instance created
    REJECTED = "rejected"        # a cap refused the request → no spend
    FAILED = "failed"            # provider call failed → no live instance


@dataclass
class CapCheck:
    """Outcome of applying the spend caps to a candidate offer + reservation."""

    ok: bool
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "reasons": list(self.reasons)}


@dataclass
class ProvisionResult:
    """Outcome of a guarded provision attempt."""

    status: ProvisionStatus
    reason: str
    offer: Offer
    reserve_hours: float
    hourly_price_cents: int
    estimated_cost_usd: float
    cap_check: CapCheck
    instance: Instance | None = None

    @property
    def did_spend(self) -> bool:
        return self.status is ProvisionStatus.PROVISIONED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "reason": self.reason,
            "offer": self.offer.to_dict(),
            "reserve_hours": self.reserve_hours,
            "hourly_price_cents": self.hourly_price_cents,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
            "cap_check": self.cap_check.to_dict(),
            "instance_id": self.instance.instance_id if self.instance else None,
            "did_spend": self.did_spend,
        }


@dataclass
class TeardownResult:
    instance_id: str
    reason: str
    ok: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class HealthResult:
    url: str
    ok: bool
    status_code: int | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WatchdogResult:
    """Why the watchdog stopped watching."""

    instance_id: str
    reason: str          # max_reserve_hours | idle | completed | terminated | stopped
    torn_down: bool
    checks: int
    elapsed_hours: float
    teardown: TeardownResult | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["teardown"] = self.teardown.to_dict() if self.teardown else None
        return data


# --------------------------------------------------------------------------- #
# Append-only event log (mirrors bot/audit.py)                                 #
# --------------------------------------------------------------------------- #

DEFAULT_OPS_EVENT_PATH = Path(
    os.environ.get("NEMOGUARDIAN_OPS_EVENT_PATH", "/tmp/nemoguardian_ops_events.jsonl")
)


@dataclass
class OpsEvent:
    """One lifecycle event in an instance's spend story."""

    event: str                          # provision_planned | provisioned | rejected | ...
    instance_id: str | None = None
    offer_id: str | None = None
    reason: str = ""
    hourly_price_cents: int | None = None
    estimated_cost_usd: float | None = None
    created_at: str = field(
        default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat()
    )
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OpsEventLog:
    """Append-only JSONL log of GPU ops events (thread-safe, fail-soft).

    Same shape as :class:`nemoguardian.bot.audit.AuditLog` — one JSON object
    per line, never rewrites history. Writes are best-effort: a logging failure
    must never break a teardown.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_OPS_EVENT_PATH
        self._lock = threading.Lock()

    def append(self, event: OpsEvent) -> None:
        line = json.dumps(event.to_dict(), default=str)
        with self._lock:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(line + "\n")
            except OSError:
                # Logging is best-effort; never let it sink an operation.
                return

    def read_all(self) -> list[dict[str, Any]]:
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                return [json.loads(ln) for ln in fh if ln.strip()]
        except FileNotFoundError:
            return []


def _emit(log: OpsEventLog | None, event: OpsEvent) -> None:
    if log is not None:
        log.append(event)


# --------------------------------------------------------------------------- #
# Minimal provider/http protocols (kept narrow for easy mocking)              #
# --------------------------------------------------------------------------- #


class _Provisioner(Protocol):
    async def provision(
        self,
        offer: Offer,
        *,
        ssh_public_key: str | None = ...,
        image: str = ...,
        env: dict[str, str] | None = ...,
    ) -> Instance: ...

    async def status(self, instance_id: str) -> InstanceStatus: ...

    async def destroy(self, instance_id: str) -> None: ...


# Sleep + clock are injected so tests advance time without waiting.
SleepFn = Callable[[float], Awaitable[None]]
ClockFn = Callable[[], float]


def _default_http_client_factory(*, timeout: float) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout)


# --------------------------------------------------------------------------- #
# #45 — spend caps                                                             #
# --------------------------------------------------------------------------- #


def offer_price_cents(offer: Offer) -> int:
    """Hourly price of an offer in integer cents (rounded to the nearest cent)."""
    return round(offer.price_per_hour_usd * 100)


def check_caps(offer: Offer, reserve_hours: float, config: GpuOpsConfig) -> CapCheck:
    """Apply the spend caps. Returns a :class:`CapCheck` with clear reasons.

    Never raises — pure decision function used both at plan and provision time.
    """
    reasons: list[str] = []
    price_cents = offer_price_cents(offer)
    if price_cents > config.max_hourly_price_cents:
        reasons.append(
            f"offer price {price_cents}c/hr exceeds cap "
            f"{config.max_hourly_price_cents}c/hr"
        )
    if reserve_hours <= 0:
        reasons.append("reserve_hours must be > 0")
    if reserve_hours > config.max_reserve_hours:
        reasons.append(
            f"reservation {reserve_hours}h exceeds max {config.max_reserve_hours}h"
        )
    return CapCheck(ok=not reasons, reasons=reasons)


async def provision_guarded(
    provider: _Provisioner,
    offer: Offer,
    *,
    config: GpuOpsConfig,
    reserve_hours: float,
    confirm: bool = False,
    ssh_public_key: str | None = None,
    image: str = "nemoguardian/self-hosted:latest",
    env: dict[str, str] | None = None,
    event_log: OpsEventLog | None = None,
) -> ProvisionResult:
    """Provision an offer *only if* it passes the spend caps **and** is confirmed.

    Order of operations (fail-safe, no-auto-spend):

    1. Apply caps. If any cap fails → ``REJECTED`` and **no** provider call.
    2. If neither ``confirm`` nor ``require_confirm=False`` → ``PLANNED``
       (a dry-run plan); still **no** provider call, no money spent.
    3. Otherwise call ``provider.provision``. A provider failure degrades to
       ``FAILED`` (never a half-charged surprise) instead of bubbling up.
    """
    price_cents = offer_price_cents(offer)
    estimated = offer.price_per_hour_usd * reserve_hours
    cap = check_caps(offer, reserve_hours, config)

    def _result(status: ProvisionStatus, reason: str, instance: Instance | None = None) -> ProvisionResult:
        return ProvisionResult(
            status=status,
            reason=reason,
            offer=offer,
            reserve_hours=reserve_hours,
            hourly_price_cents=price_cents,
            estimated_cost_usd=estimated,
            cap_check=cap,
            instance=instance,
        )

    if not cap.ok:
        reason = "; ".join(cap.reasons)
        _emit(event_log, OpsEvent(
            event="rejected", offer_id=offer.offer_id or None, reason=reason,
            hourly_price_cents=price_cents, estimated_cost_usd=estimated,
        ))
        return _result(ProvisionStatus.REJECTED, reason)

    confirmed = confirm or not config.require_confirm
    if not confirmed:
        reason = "dry-run: caps OK; pass confirm=True (or set NEMOGUARDIAN_OPS_CONFIRM) to provision"
        _emit(event_log, OpsEvent(
            event="provision_planned", offer_id=offer.offer_id or None, reason=reason,
            hourly_price_cents=price_cents, estimated_cost_usd=estimated,
        ))
        return _result(ProvisionStatus.PLANNED, reason)

    try:
        instance = await provider.provision(
            offer, ssh_public_key=ssh_public_key, image=image, env=env,
        )
    except ProvisionError as exc:
        _emit(event_log, OpsEvent(
            event="failed", offer_id=offer.offer_id or None, reason=str(exc),
            hourly_price_cents=price_cents, estimated_cost_usd=estimated,
        ))
        return _result(ProvisionStatus.FAILED, f"provider error: {exc}")

    _emit(event_log, OpsEvent(
        event="provisioned", instance_id=instance.instance_id,
        offer_id=offer.offer_id or None, reason="confirmed provision",
        hourly_price_cents=price_cents, estimated_cost_usd=estimated,
    ))
    return _result(ProvisionStatus.PROVISIONED, "confirmed provision", instance)


# --------------------------------------------------------------------------- #
# #46 — teardown + watchdog                                                    #
# --------------------------------------------------------------------------- #


async def teardown(
    provider: _Provisioner,
    instance_id: str,
    *,
    reason: str = "manual",
    event_log: OpsEventLog | None = None,
) -> TeardownResult:
    """Destroy a rented instance. Idempotent + fail-safe.

    ``provider.destroy`` is contractually idempotent and swallows transport
    errors, but we still guard against an unexpected raise so a teardown
    request can always be issued safely.
    """
    try:
        await provider.destroy(instance_id)
        result = TeardownResult(instance_id=instance_id, reason=reason, ok=True)
    except Exception as exc:  # provider should not raise, but never let it block
        result = TeardownResult(
            instance_id=instance_id, reason=reason, ok=False, error=str(exc)
        )
    _emit(event_log, OpsEvent(
        event="teardown", instance_id=instance_id, reason=reason,
        details={"ok": result.ok, "error": result.error},
    ))
    return result


_TERMINAL_STATES = {InstanceState.FAILED, InstanceState.DESTROYED}


async def watchdog(
    provider: _Provisioner,
    instance_id: str,
    *,
    config: GpuOpsConfig,
    started_at: float | None = None,
    now_fn: ClockFn = time.monotonic,
    sleep: SleepFn = asyncio.sleep,
    idle_seconds: Callable[[], float] | None = None,
    is_complete: Callable[[], bool] | None = None,
    max_checks: int = 100_000,
    event_log: OpsEventLog | None = None,
) -> WatchdogResult:
    """Watch an instance and auto-tear-down when a guardrail trips.

    Tears down (and stops) on the first of:

    * **max_reserve_hours** — elapsed wall time since ``started_at`` exceeds the
      cap. This is the core "GPUs must never run forever" guarantee.
    * **idle** — ``idle_seconds()`` exceeds ``config.max_idle_seconds``.
    * **completed** — ``is_complete()`` returns True ("destroy on completion").

    Stops *without* tearing down when the provider already reports a terminal
    state (FAILED / DESTROYED).

    ``now_fn`` and ``sleep`` are injected so tests drive the clock deterministically
    and never actually wait.
    """
    start = started_at if started_at is not None else now_fn()
    checks = 0

    def _elapsed_hours() -> float:
        return max(0.0, (now_fn() - start) / 3600.0)

    while checks < max_checks:
        checks += 1
        elapsed_h = _elapsed_hours()

        # 1. Hard time cap — the anti-runaway guarantee.
        if elapsed_h >= config.max_reserve_hours:
            td = await teardown(
                provider, instance_id, reason="max_reserve_hours", event_log=event_log
            )
            return WatchdogResult(instance_id, "max_reserve_hours", td.ok, checks, elapsed_h, td)

        # 2. Completion — destroy on completion.
        if is_complete is not None and is_complete():
            td = await teardown(
                provider, instance_id, reason="completed", event_log=event_log
            )
            return WatchdogResult(instance_id, "completed", td.ok, checks, elapsed_h, td)

        # 3. Idle cap.
        if (
            config.max_idle_seconds is not None
            and idle_seconds is not None
            and idle_seconds() >= config.max_idle_seconds
        ):
            td = await teardown(
                provider, instance_id, reason="idle", event_log=event_log
            )
            return WatchdogResult(instance_id, "idle", td.ok, checks, elapsed_h, td)

        # 4. Already gone? Nothing to tear down. Transport/status API failures are
        # not terminal proof; keep polling so a transient outage never leaves the
        # watchdog stopped while the GPU keeps running.
        try:
            status = await provider.status(instance_id)
        except Exception as exc:
            _emit(event_log, OpsEvent(
                event="watchdog_status_error", instance_id=instance_id,
                reason=type(exc).__name__, details={"error": str(exc)},
            ))
        else:
            if status.state in _TERMINAL_STATES:
                _emit(event_log, OpsEvent(
                    event="watchdog_terminated", instance_id=instance_id,
                    reason=f"provider state {status.state.value}",
                ))
                return WatchdogResult(instance_id, "terminated", False, checks, elapsed_h, None)

        await sleep(config.poll_interval_seconds)

    # Safety valve: exhausted the check budget without tripping a cap.
    return WatchdogResult(instance_id, "stopped", False, checks, _elapsed_hours(), None)


# --------------------------------------------------------------------------- #
# #47 — status polling + health                                                #
# --------------------------------------------------------------------------- #


async def _safe_status(provider: _Provisioner, instance_id: str) -> InstanceStatus:
    """provider.status, but a transport raise degrades to a FAILED status."""
    try:
        return await provider.status(instance_id)
    except Exception as exc:
        return InstanceStatus(
            instance_id=instance_id,
            state=InstanceState.FAILED,
            error_message=str(exc),
        )


async def poll_until_running(
    provider: _Provisioner,
    instance_id: str,
    *,
    config: GpuOpsConfig | None = None,
    max_attempts: int | None = None,
    interval: float | None = None,
    sleep: SleepFn = asyncio.sleep,
) -> InstanceStatus:
    """Poll ``provider.status`` until the instance is LIVE (or terminal/exhausted).

    Returns the last :class:`InstanceStatus`. Stops early on LIVE or on a
    terminal state (FAILED/DESTROYED). ``sleep`` is injected so tests resolve
    instantly. Never raises — a transport failure becomes a FAILED status and
    ends the poll (fail-safe: we do not report a dead instance as running).
    """
    cfg = config or GpuOpsConfig()
    attempts = max_attempts if max_attempts is not None else cfg.poll_max_attempts
    wait = interval if interval is not None else cfg.poll_interval_seconds

    status = InstanceStatus(instance_id=instance_id, state=InstanceState.PENDING)
    for attempt in range(max(1, attempts)):
        status = await _safe_status(provider, instance_id)
        if status.state is InstanceState.LIVE or status.state in _TERMINAL_STATES:
            return status
        if attempt < attempts - 1:
            await sleep(wait)
    return status


async def health_check(
    endpoint_url: str,
    *,
    config: GpuOpsConfig | None = None,
    path: str | None = None,
    timeout: float | None = None,
    client_factory: Callable[..., Any] = _default_http_client_factory,
) -> HealthResult:
    """GET ``<endpoint_url><health_path>`` and report whether it is healthy.

    ``client_factory(timeout=...)`` must return an async-context-manager HTTP
    client exposing ``await .get(url)`` (httpx.AsyncClient by default). Tests
    pass a fake client so no network call happens. A non-2xx or any transport
    error → ``ok=False`` (fail-safe: unknown health is treated as unhealthy).
    """
    cfg = config or GpuOpsConfig()
    health_path = path if path is not None else cfg.health_path
    to = timeout if timeout is not None else cfg.health_timeout_seconds
    url = endpoint_url.rstrip("/") + "/" + health_path.lstrip("/")

    try:
        async with client_factory(timeout=to) as client:
            resp = await client.get(url)
            status_code = int(getattr(resp, "status_code", 0) or 0)
            resp.raise_for_status()
        return HealthResult(url=url, ok=True, status_code=status_code or 200)
    except Exception as exc:
        return HealthResult(url=url, ok=False, status_code=None, error=str(exc))


__all__ = [
    "DEFAULT_OPS_EVENT_PATH",
    "CapCheck",
    "GpuOpsConfig",
    "HealthResult",
    "OpsEvent",
    "OpsEventLog",
    "ProvisionResult",
    "ProvisionStatus",
    "TeardownResult",
    "WatchdogResult",
    "check_caps",
    "health_check",
    "offer_price_cents",
    "poll_until_running",
    "provision_guarded",
    "replace_max_reserve_hours",
    "teardown",
    "watchdog",
]
