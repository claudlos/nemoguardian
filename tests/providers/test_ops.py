"""Tests for GPU operator guardrails: spend caps, teardown, watchdog, health.

No live calls — the provider client, HTTP client, clock, and sleep are all
injected/faked.
"""

from __future__ import annotations

from typing import Any

import pytest

from nemoguardian.providers import ops
from nemoguardian.providers.base import (
    Instance,
    InstanceState,
    InstanceStatus,
    Offer,
    ProviderName,
    ProvisionError,
)

# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #


def _offer(price_usd: float, *, offer_id: str = "of-1", vram: int = 24) -> Offer:
    return Offer(
        provider=ProviderName.VAST_AI,
        gpu_model="RTX 3090",
        vram_gb=vram,
        price_per_hour_usd=price_usd,
        region="Global",
        offer_id=offer_id,
    )


class FakeProvider:
    """Records provision/destroy and serves scripted status states."""

    def __init__(
        self,
        *,
        states: list[InstanceState] | None = None,
        provision_error: Exception | None = None,
        status_error: Exception | None = None,
        destroy_error: Exception | None = None,
    ) -> None:
        self._states = states or [InstanceState.LIVE]
        self._idx = 0
        self.provision_error = provision_error
        self.status_error = status_error
        self.destroy_error = destroy_error
        self.provision_calls: list[dict[str, Any]] = []
        self.destroy_calls: list[str] = []
        self.status_calls: list[str] = []

    async def provision(
        self,
        offer: Offer,
        *,
        ssh_public_key: str | None = None,
        image: str = "img",
        env: dict[str, str] | None = None,
    ) -> Instance:
        self.provision_calls.append({"offer": offer, "image": image, "env": env})
        if self.provision_error:
            raise self.provision_error
        return Instance(
            provider=ProviderName.VAST_AI,
            instance_id="inst-123",
            gpu_model=offer.gpu_model,
            vram_gb=offer.vram_gb,
            region=offer.region,
            state=InstanceState.PROVISIONING,
            hourly_price_usd=offer.price_per_hour_usd,
        )

    async def status(self, instance_id: str) -> InstanceStatus:
        self.status_calls.append(instance_id)
        if self.status_error:
            raise self.status_error
        state = self._states[min(self._idx, len(self._states) - 1)]
        self._idx += 1
        return InstanceStatus(instance_id=instance_id, state=state, uptime_seconds=42)

    async def destroy(self, instance_id: str) -> None:
        self.destroy_calls.append(instance_id)
        if self.destroy_error:
            raise self.destroy_error


class FakeClock:
    """Returns scripted monotonic timestamps (seconds)."""

    def __init__(self, values: list[float]) -> None:
        self.values = values
        self.i = 0

    def __call__(self) -> float:
        v = self.values[min(self.i, len(self.values) - 1)]
        self.i += 1
        return v


def _no_sleep_factory() -> tuple[list[float], Any]:
    waits: list[float] = []

    async def _sleep(seconds: float) -> None:
        waits.append(seconds)

    return waits, _sleep


class _FakeHTTPResponse:
    def __init__(self, status_code: int = 200, *, error: Exception | None = None) -> None:
        self.status_code = status_code
        self._error = error

    def raise_for_status(self) -> None:
        if self._error:
            raise self._error


class _FakeHTTPClient:
    def __init__(self, response: _FakeHTTPResponse | None = None, *, get_error: Exception | None = None):
        self.response = response or _FakeHTTPResponse()
        self.get_error = get_error
        self.calls: list[str] = []
        self.timeout: float | None = None

    async def __aenter__(self) -> _FakeHTTPClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def get(self, url: str) -> _FakeHTTPResponse:
        self.calls.append(url)
        if self.get_error:
            raise self.get_error
        return self.response


# --------------------------------------------------------------------------- #
# #45 — spend caps                                                             #
# --------------------------------------------------------------------------- #


def test_offer_price_cents_ceils():
    # Exact whole-cent prices are unaffected (float noise is quantised away).
    assert ops.offer_price_cents(_offer(0.07)) == 7
    assert ops.offer_price_cents(_offer(0.0)) == 0
    # Anything with a sub-cent remainder rounds *up* so a cap can't be squeaked
    # past: 20.5c -> 21c, not the old round-half 20c.
    assert ops.offer_price_cents(_offer(0.205)) == 21


def test_offer_just_over_cap_is_rejected():
    # $0.5049/hr is strictly above a 50c/hr cap. Round-half admitted it (50c);
    # ceil rejects it (51c) so nothing above the cap slips through.
    cfg = ops.GpuOpsConfig(max_hourly_price_cents=50)
    assert ops.offer_price_cents(_offer(0.5049)) == 51
    check = ops.check_caps(_offer(0.5049), reserve_hours=3, config=cfg)
    assert check.ok is False
    assert any("exceeds cap" in r for r in check.reasons)


async def test_provision_guarded_auto_confirms_zero_cost_offer():
    # A $0/hr offer (e.g. on_prem) has no spend to gate, so it provisions even
    # when confirm is False and the config still requires confirmation.
    cfg = ops.GpuOpsConfig(require_confirm=True)
    provider = FakeProvider()
    result = await ops.provision_guarded(
        provider, _offer(0.0), config=cfg, reserve_hours=3, confirm=False
    )
    assert result.status is ops.ProvisionStatus.PROVISIONED
    assert result.instance is not None
    assert len(provider.provision_calls) == 1


def test_check_caps_accepts_within_limits():
    cfg = ops.GpuOpsConfig(max_hourly_price_cents=50, max_reserve_hours=12)
    check = ops.check_caps(_offer(0.21), reserve_hours=6, config=cfg)
    assert check.ok is True
    assert check.reasons == []


def test_check_caps_rejects_price_over_cap():
    cfg = ops.GpuOpsConfig(max_hourly_price_cents=20, max_reserve_hours=12)
    check = ops.check_caps(_offer(0.40), reserve_hours=6, config=cfg)
    assert check.ok is False
    assert any("exceeds cap" in r for r in check.reasons)


def test_check_caps_rejects_too_many_hours_and_nonpositive():
    cfg = ops.GpuOpsConfig(max_hourly_price_cents=50, max_reserve_hours=8)
    over = ops.check_caps(_offer(0.10), reserve_hours=24, config=cfg)
    assert over.ok is False
    assert any("exceeds max" in r for r in over.reasons)

    zero = ops.check_caps(_offer(0.10), reserve_hours=0, config=cfg)
    assert zero.ok is False
    assert any("> 0" in r for r in zero.reasons)


async def test_provision_guarded_rejects_over_price_cap_without_calling_provider():
    cfg = ops.GpuOpsConfig(max_hourly_price_cents=20, require_confirm=False)
    provider = FakeProvider()
    result = await ops.provision_guarded(
        provider, _offer(0.40), config=cfg, reserve_hours=6, confirm=True
    )
    assert result.status is ops.ProvisionStatus.REJECTED
    assert result.did_spend is False
    assert provider.provision_calls == []  # never spent


async def test_provision_guarded_dry_run_when_not_confirmed():
    cfg = ops.GpuOpsConfig(max_hourly_price_cents=50, require_confirm=True)
    provider = FakeProvider()
    result = await ops.provision_guarded(
        provider, _offer(0.21), config=cfg, reserve_hours=6, confirm=False
    )
    assert result.status is ops.ProvisionStatus.PLANNED
    assert result.instance is None
    assert result.estimated_cost_usd == pytest.approx(0.21 * 6)
    assert provider.provision_calls == []  # no auto-spend


async def test_provision_guarded_provisions_when_confirmed():
    cfg = ops.GpuOpsConfig(max_hourly_price_cents=50, require_confirm=True)
    provider = FakeProvider()
    result = await ops.provision_guarded(
        provider, _offer(0.21), config=cfg, reserve_hours=6, confirm=True
    )
    assert result.status is ops.ProvisionStatus.PROVISIONED
    assert result.did_spend is True
    assert result.instance is not None
    assert result.instance.instance_id == "inst-123"
    assert len(provider.provision_calls) == 1


async def test_provision_guarded_env_gate_disables_confirm_requirement():
    cfg = ops.GpuOpsConfig(require_confirm=False)  # mirrors NEMOGUARDIAN_OPS_CONFIRM gate
    provider = FakeProvider()
    result = await ops.provision_guarded(
        provider, _offer(0.21), config=cfg, reserve_hours=6, confirm=False
    )
    assert result.status is ops.ProvisionStatus.PROVISIONED
    assert len(provider.provision_calls) == 1


async def test_provision_guarded_provider_error_degrades_to_failed():
    cfg = ops.GpuOpsConfig(require_confirm=False)
    provider = FakeProvider(provision_error=ProvisionError("denied"))
    result = await ops.provision_guarded(
        provider, _offer(0.21), config=cfg, reserve_hours=6, confirm=True
    )
    assert result.status is ops.ProvisionStatus.FAILED
    assert "denied" in result.reason
    assert result.instance is None


# --------------------------------------------------------------------------- #
# #46 — teardown + watchdog                                                    #
# --------------------------------------------------------------------------- #


async def test_teardown_calls_destroy():
    provider = FakeProvider()
    result = await ops.teardown(provider, "inst-9", reason="done")
    assert result.ok is True
    assert provider.destroy_calls == ["inst-9"]


async def test_teardown_is_failsafe_when_destroy_raises():
    class Boom(FakeProvider):
        async def destroy(self, instance_id: str) -> None:
            raise RuntimeError("kaboom")

    provider = Boom()
    result = await ops.teardown(provider, "inst-9")
    assert result.ok is False
    assert "kaboom" in (result.error or "")


async def test_watchdog_tears_down_after_max_reserve_hours():
    # elapsed crosses 1.0h on the second iteration → teardown.
    cfg = ops.GpuOpsConfig(max_reserve_hours=1.0, max_idle_seconds=None, poll_interval_seconds=5)
    provider = FakeProvider(states=[InstanceState.LIVE, InstanceState.LIVE])
    clock = FakeClock([100.0, 4000.0])  # iter1 (~0.03h) → sleep, iter2 (>1h) → teardown
    waits, sleep = _no_sleep_factory()

    result = await ops.watchdog(
        provider, "inst-7", config=cfg, started_at=0.0,
        now_fn=clock, sleep=sleep,
    )
    assert result.reason == "max_reserve_hours"
    assert result.torn_down is True
    assert provider.destroy_calls == ["inst-7"]
    assert len(waits) == 1  # slept once before the cap tripped


async def test_watchdog_reports_failed_teardown_without_claiming_success():
    cfg = ops.GpuOpsConfig(max_reserve_hours=1.0, max_idle_seconds=None, poll_interval_seconds=5)
    provider = FakeProvider(states=[InstanceState.LIVE], destroy_error=RuntimeError("destroy down"))
    _waits, sleep = _no_sleep_factory()

    result = await ops.watchdog(
        provider, "inst-7", config=cfg, started_at=0.0,
        now_fn=FakeClock([4000.0]), sleep=sleep,
    )

    assert result.reason == "max_reserve_hours"
    assert result.torn_down is False
    assert result.teardown is not None
    assert result.teardown.ok is False
    assert provider.destroy_calls == ["inst-7"]


async def test_watchdog_tears_down_on_idle():
    cfg = ops.GpuOpsConfig(max_reserve_hours=100, max_idle_seconds=60, poll_interval_seconds=5)
    provider = FakeProvider(states=[InstanceState.LIVE])
    _waits, sleep = _no_sleep_factory()
    result = await ops.watchdog(
        provider, "inst-7", config=cfg, started_at=0.0,
        now_fn=FakeClock([0.0, 1.0]), sleep=sleep,
        idle_seconds=lambda: 120.0,
    )
    assert result.reason == "idle"
    assert provider.destroy_calls == ["inst-7"]


async def test_watchdog_tears_down_on_completion():
    cfg = ops.GpuOpsConfig(max_reserve_hours=100, max_idle_seconds=None)
    provider = FakeProvider(states=[InstanceState.LIVE])
    _waits, sleep = _no_sleep_factory()
    result = await ops.watchdog(
        provider, "inst-7", config=cfg, started_at=0.0,
        now_fn=FakeClock([0.0, 1.0]), sleep=sleep,
        is_complete=lambda: True,
    )
    assert result.reason == "completed"
    assert provider.destroy_calls == ["inst-7"]


async def test_watchdog_stops_without_teardown_when_already_terminal():
    cfg = ops.GpuOpsConfig(max_reserve_hours=100, max_idle_seconds=None)
    provider = FakeProvider(states=[InstanceState.DESTROYED])
    _waits, sleep = _no_sleep_factory()
    result = await ops.watchdog(
        provider, "inst-7", config=cfg, started_at=0.0,
        now_fn=FakeClock([0.0, 1.0]), sleep=sleep,
    )
    assert result.reason == "terminated"
    assert result.torn_down is False
    assert provider.destroy_calls == []


async def test_watchdog_keeps_polling_after_transient_status_error():
    cfg = ops.GpuOpsConfig(max_reserve_hours=100, max_idle_seconds=None, poll_interval_seconds=5)

    class FlakyStatusProvider(FakeProvider):
        async def status(self, instance_id: str) -> InstanceStatus:
            self.status_calls.append(instance_id)
            if len(self.status_calls) == 1:
                raise RuntimeError("provider temporarily unavailable")
            return InstanceStatus(instance_id=instance_id, state=InstanceState.DESTROYED)

    provider = FlakyStatusProvider()
    waits, sleep = _no_sleep_factory()
    result = await ops.watchdog(
        provider, "inst-7", config=cfg, started_at=0.0,
        now_fn=FakeClock([0.0, 1.0, 2.0]), sleep=sleep,
    )

    assert result.reason == "terminated"
    assert result.torn_down is False
    assert provider.destroy_calls == []
    assert len(provider.status_calls) == 2
    assert waits == [5]


# --------------------------------------------------------------------------- #
# #47 — status polling + health                                                #
# --------------------------------------------------------------------------- #


async def test_poll_until_running_resolves_on_live():
    provider = FakeProvider(
        states=[InstanceState.PROVISIONING, InstanceState.PROVISIONING, InstanceState.LIVE]
    )
    waits, sleep = _no_sleep_factory()
    status = await ops.poll_until_running(
        provider, "inst-1", max_attempts=10, interval=5, sleep=sleep
    )
    assert status.state is InstanceState.LIVE
    assert len(provider.status_calls) == 3
    assert waits == [5, 5]  # slept between the first two non-live polls


async def test_poll_until_running_stops_on_terminal():
    provider = FakeProvider(states=[InstanceState.FAILED])
    waits, sleep = _no_sleep_factory()
    status = await ops.poll_until_running(provider, "inst-1", max_attempts=10, sleep=sleep)
    assert status.state is InstanceState.FAILED
    assert len(provider.status_calls) == 1
    assert waits == []


async def test_poll_until_running_exhausts_attempts():
    provider = FakeProvider(states=[InstanceState.PROVISIONING])
    waits, sleep = _no_sleep_factory()
    status = await ops.poll_until_running(
        provider, "inst-1", max_attempts=3, interval=2, sleep=sleep
    )
    assert status.state is InstanceState.PROVISIONING
    assert len(provider.status_calls) == 3
    assert waits == [2, 2]


async def test_poll_until_running_failsafe_on_status_raise():
    provider = FakeProvider(status_error=RuntimeError("boom"))
    _waits, sleep = _no_sleep_factory()
    status = await ops.poll_until_running(provider, "inst-1", max_attempts=5, sleep=sleep)
    assert status.state is InstanceState.FAILED
    assert status.error_message == "boom"


async def test_health_check_ok():
    client = _FakeHTTPClient(_FakeHTTPResponse(200))

    def factory(*, timeout: float) -> _FakeHTTPClient:
        client.timeout = timeout
        return client

    result = await ops.health_check(
        "https://host.nemoguardian.dev/", path="/health", timeout=3.0, client_factory=factory
    )
    assert result.ok is True
    assert result.status_code == 200
    assert client.calls == ["https://host.nemoguardian.dev/health"]
    assert client.timeout == 3.0


async def test_health_check_unhealthy_on_error():
    client = _FakeHTTPClient(get_error=RuntimeError("connection refused"))

    def factory(*, timeout: float) -> _FakeHTTPClient:
        return client

    result = await ops.health_check(
        "https://host.nemoguardian.dev", client_factory=factory
    )
    assert result.ok is False
    assert result.status_code is None
    assert "connection refused" in (result.error or "")


async def test_health_check_unhealthy_on_non_2xx():
    client = _FakeHTTPClient(_FakeHTTPResponse(503, error=RuntimeError("503 Service Unavailable")))

    def factory(*, timeout: float) -> _FakeHTTPClient:
        return client

    result = await ops.health_check("https://host.nemoguardian.dev", client_factory=factory)
    assert result.ok is False


# --------------------------------------------------------------------------- #
# Config + event log                                                           #
# --------------------------------------------------------------------------- #


def test_config_from_env(monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_OPS_MAX_HOURLY_PRICE_CENTS", "25")
    monkeypatch.setenv("NEMOGUARDIAN_OPS_MAX_RESERVE_HOURS", "4.5")
    monkeypatch.setenv("NEMOGUARDIAN_OPS_MAX_IDLE_SECONDS", "off")
    monkeypatch.setenv("NEMOGUARDIAN_OPS_CONFIRM", "1")
    cfg = ops.GpuOpsConfig.from_env()
    assert cfg.max_hourly_price_cents == 25
    assert cfg.max_reserve_hours == 4.5
    assert cfg.max_idle_seconds is None
    assert cfg.require_confirm is False  # env gate flips off the confirm requirement


def test_config_from_env_defaults(monkeypatch):
    for var in [
        "NEMOGUARDIAN_OPS_MAX_HOURLY_PRICE_CENTS",
        "NEMOGUARDIAN_OPS_MAX_RESERVE_HOURS",
        "NEMOGUARDIAN_OPS_MAX_IDLE_SECONDS",
        "NEMOGUARDIAN_OPS_CONFIRM",
    ]:
        monkeypatch.delenv(var, raising=False)
    cfg = ops.GpuOpsConfig.from_env()
    assert cfg.require_confirm is True  # safe default: no auto-spend
    assert cfg.max_idle_seconds == 1800.0


def test_replace_max_reserve_hours():
    cfg = ops.GpuOpsConfig(max_reserve_hours=12)
    cfg2 = ops.replace_max_reserve_hours(cfg, 3.0)
    assert cfg2.max_reserve_hours == 3.0
    assert cfg.max_reserve_hours == 12  # original untouched (frozen)


def test_event_log_appends_jsonl(tmp_path):
    path = tmp_path / "ops.jsonl"
    log = ops.OpsEventLog(path)
    log.append(ops.OpsEvent(event="provisioned", instance_id="i1", reason="ok"))
    log.append(ops.OpsEvent(event="teardown", instance_id="i1", reason="done"))
    rows = log.read_all()
    assert [r["event"] for r in rows] == ["provisioned", "teardown"]
    assert rows[0]["instance_id"] == "i1"


async def test_provision_guarded_writes_events(tmp_path):
    log = ops.OpsEventLog(tmp_path / "ev.jsonl")
    cfg = ops.GpuOpsConfig(require_confirm=True)
    provider = FakeProvider()
    await ops.provision_guarded(
        provider, _offer(0.21), config=cfg, reserve_hours=6, confirm=False, event_log=log
    )
    rows = log.read_all()
    assert rows[-1]["event"] == "provision_planned"


async def test_teardown_writes_event(tmp_path):
    log = ops.OpsEventLog(tmp_path / "ev.jsonl")
    provider = FakeProvider()
    await ops.teardown(provider, "i9", reason="cap", event_log=log)
    rows = log.read_all()
    assert rows[-1]["event"] == "teardown"
    assert rows[-1]["reason"] == "cap"
