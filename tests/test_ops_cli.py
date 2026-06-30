"""CLI tests for the `gpu-ops` sub-app (no live calls — provider is faked)."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

import nemoguardian.ops_cli as ops_cli
from nemoguardian.cli import app
from nemoguardian.providers.base import Instance, InstanceState, InstanceStatus, ProviderName

runner = CliRunner()


class _CliFakeProvider:
    def __init__(self, *, status_state: InstanceState = InstanceState.LIVE) -> None:
        self.status_state = status_state
        self.destroy_calls: list[str] = []
        self.provision_calls: int = 0

    async def provision(self, offer, *, ssh_public_key=None, image="img", env=None) -> Instance:
        self.provision_calls += 1
        return Instance(
            provider=ProviderName.VAST_AI,
            instance_id="inst-cli-1",
            gpu_model=offer.gpu_model,
            vram_gb=offer.vram_gb,
            region=offer.region,
            state=InstanceState.PROVISIONING,
        )

    async def status(self, instance_id: str) -> InstanceStatus:
        return InstanceStatus(instance_id=instance_id, state=self.status_state, uptime_seconds=10)

    async def destroy(self, instance_id: str) -> None:
        self.destroy_calls.append(instance_id)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in [
        "NEMOGUARDIAN_OPS_CONFIRM",
        "NEMOGUARDIAN_OPS_MAX_HOURLY_PRICE_CENTS",
        "NEMOGUARDIAN_OPS_MAX_RESERVE_HOURS",
        "NEMOGUARDIAN_OPS_MAX_IDLE_SECONDS",
    ]:
        monkeypatch.delenv(var, raising=False)


def _patch_provider(monkeypatch, provider) -> None:
    monkeypatch.setattr(ops_cli, "get_provider", lambda name: provider)


def test_provision_dry_run_by_default(monkeypatch):
    provider = _CliFakeProvider()
    _patch_provider(monkeypatch, provider)
    result = runner.invoke(
        app, ["gpu-ops", "provision", "--gpu", "RTX 3090", "--price-cents", "7", "--hours", "6"]
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "planned"
    assert payload["did_spend"] is False
    assert provider.provision_calls == 0


def test_provision_confirm_spends(monkeypatch):
    provider = _CliFakeProvider()
    _patch_provider(monkeypatch, provider)
    result = runner.invoke(
        app,
        ["gpu-ops", "provision", "--gpu", "RTX 3090", "--price-cents", "7", "--hours", "6", "--confirm"],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["status"] == "provisioned"
    assert payload["instance_id"] == "inst-cli-1"
    assert provider.provision_calls == 1


def test_provision_over_cap_rejected_exit_2(monkeypatch):
    provider = _CliFakeProvider()
    _patch_provider(monkeypatch, provider)
    result = runner.invoke(
        app,
        ["gpu-ops", "provision", "--gpu", "H100", "--price-cents", "999", "--hours", "6", "--confirm"],
    )
    assert result.exit_code == 2
    payload = json.loads(result.output)
    assert payload["status"] == "rejected"
    assert provider.provision_calls == 0


def test_status_command(monkeypatch):
    provider = _CliFakeProvider(status_state=InstanceState.LIVE)
    _patch_provider(monkeypatch, provider)
    result = runner.invoke(app, ["gpu-ops", "status", "inst-9", "--max-attempts", "1"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["state"] == "live"
    assert payload["instance_id"] == "inst-9"


def test_teardown_command(monkeypatch):
    provider = _CliFakeProvider()
    _patch_provider(monkeypatch, provider)
    result = runner.invoke(app, ["gpu-ops", "teardown", "inst-9", "--reason", "done"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True
    assert provider.destroy_calls == ["inst-9"]


def test_health_command_ok(monkeypatch):
    async def _fake_health(endpoint_url, *, config=None, path=None, timeout=None, **kw):
        return ops_cli.ops.HealthResult(url=endpoint_url + "/health", ok=True, status_code=200)

    monkeypatch.setattr(ops_cli.ops, "health_check", _fake_health)
    result = runner.invoke(app, ["gpu-ops", "health", "https://host.nemoguardian.dev"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["ok"] is True


def test_health_command_unhealthy_exit_1(monkeypatch):
    async def _fake_health(endpoint_url, *, config=None, path=None, timeout=None, **kw):
        return ops_cli.ops.HealthResult(url=endpoint_url + "/health", ok=False, error="refused")

    monkeypatch.setattr(ops_cli.ops, "health_check", _fake_health)
    result = runner.invoke(app, ["gpu-ops", "health", "https://host.nemoguardian.dev"])
    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload["ok"] is False


def test_watchdog_command_wires_args(monkeypatch):
    provider = _CliFakeProvider()
    _patch_provider(monkeypatch, provider)
    captured = {}

    async def _fake_watchdog(prov, instance_id, *, config, max_checks, event_log):
        captured["instance_id"] = instance_id
        captured["max_reserve_hours"] = config.max_reserve_hours
        return ops_cli.ops.WatchdogResult(
            instance_id=instance_id, reason="max_reserve_hours", torn_down=True,
            checks=1, elapsed_hours=6.0,
        )

    monkeypatch.setattr(ops_cli.ops, "watchdog", _fake_watchdog)
    result = runner.invoke(app, ["gpu-ops", "watchdog", "inst-9", "--max-hours", "6"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.output)
    assert payload["reason"] == "max_reserve_hours"
    assert captured["instance_id"] == "inst-9"
    assert captured["max_reserve_hours"] == 6.0
