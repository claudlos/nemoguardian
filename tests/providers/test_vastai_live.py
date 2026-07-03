"""VastAIProvider live REST path — httpx.MockTransport, no network, no key leaves the box."""

from __future__ import annotations

import json
from types import SimpleNamespace

import httpx
import pytest

from nemoguardian.providers import vastai as vastai_mod
from nemoguardian.providers.base import InstanceState, ProvisionError
from nemoguardian.providers.vastai import VastAIProvider


def _mock_httpx(monkeypatch, handler) -> list[httpx.Request]:
    """Route the module's httpx.AsyncClient through a MockTransport."""
    requests: list[httpx.Request] = []

    def recording_handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return handler(request)

    def client_factory(**kwargs):
        kwargs.pop("timeout", None)
        return httpx.AsyncClient(transport=httpx.MockTransport(recording_handler))

    monkeypatch.setattr(
        vastai_mod, "httpx", SimpleNamespace(AsyncClient=client_factory)
    )
    return requests


def _live_provider() -> VastAIProvider:
    return VastAIProvider(api_key="vast-test-key")


# --- list_offers -------------------------------------------------------------

async def test_live_list_offers_parses_and_sorts(monkeypatch):
    raw_offers = [
        {"id": 222, "gpu_name": "RTX_4090", "gpu_ram": 24576, "dph_total": 0.30,
         "geolocation": "US-East", "inet_down": 8000},
        {"id": 111, "gpu_name": "RTX_3090", "gpu_ram": 24576, "dph_total": 0.11,
         "geolocation": "EU", "inet_down": 1000},
    ]
    requests = _mock_httpx(
        monkeypatch, lambda req: httpx.Response(200, json={"offers": raw_offers})
    )

    offers = await _live_provider().list_offers()

    assert [o.offer_id for o in offers] == ["111", "222"]  # sorted by price
    o = offers[0]
    assert o.gpu_model == "RTX 3090"  # underscores become spaces
    assert o.vram_gb == 24  # MB → GB
    assert o.price_per_hour_usd == 0.11
    assert o.region == "EU"
    assert o.bandwidth_gbps == 1.0  # Mbps → Gbps
    assert requests[0].headers["Authorization"] == "Bearer vast-test-key"


async def test_live_search_payload_filters(monkeypatch):
    requests = _mock_httpx(monkeypatch, lambda req: httpx.Response(200, json={"offers": []}))
    await _live_provider().list_offers(gpu_model="rtx 3090", max_price_usd=0.25)
    payload = json.loads(requests[0].content)
    assert payload["gpu_name"] == ["RTX_3090"]
    assert payload["dph_max"] == 0.25
    assert payload["rentable"] == {"eq": True}


async def test_live_list_offers_network_failure_falls_back_to_cache(monkeypatch):
    def exploding(request):
        raise httpx.ConnectError("no route to host")

    _mock_httpx(monkeypatch, exploding)
    offers = await _live_provider().list_offers()
    assert offers  # cached catalog served instead
    assert all(o.offer_id.startswith("cached-") for o in offers)


# --- provision ---------------------------------------------------------------

def _cached_offer():
    return vastai_mod._CACHED_OFFERS[0]  # RTX 3090 @ $0.07


async def test_live_provision_success(monkeypatch):
    requests = _mock_httpx(
        monkeypatch,
        lambda req: httpx.Response(
            200, json={"new_contract": 987654, "ssh_host": "ssh4.vast.ai", "ssh_port": 2222}
        ),
    )
    instance = await _live_provider().provision(
        _cached_offer(), ssh_public_key="ssh-ed25519 AAAA test", env={"K": "v"}
    )
    assert instance.instance_id == "987654"
    assert instance.state == InstanceState.PROVISIONING
    assert "ssh -p 2222 root@ssh4.vast.ai" in instance.ssh_command
    payload = json.loads(requests[0].content)
    assert payload["ssh_key"] == "ssh-ed25519 AAAA test"
    assert payload["env"] == {"K": "v"}
    assert payload["price"] == pytest.approx(0.07 * 1.1)  # bid over asking


async def test_live_provision_http_error_raises_provision_error(monkeypatch):
    _mock_httpx(monkeypatch, lambda req: httpx.Response(402, json={"error": "no credit"}))
    with pytest.raises(ProvisionError, match="create_inst failed"):
        await _live_provider().provision(_cached_offer())


# --- status / destroy ----------------------------------------------------------

@pytest.mark.parametrize(
    ("actual_status", "expected"),
    [
        ("running", InstanceState.LIVE),
        ("created", InstanceState.PROVISIONING),
        ("loading", InstanceState.PROVISIONING),
        ("exited", InstanceState.DESTROYED),
        ("error", InstanceState.FAILED),
        ("weird-new-state", InstanceState.PROVISIONING),  # unknown → provisioning
    ],
)
async def test_live_status_state_mapping(monkeypatch, actual_status, expected):
    _mock_httpx(
        monkeypatch,
        lambda req: httpx.Response(
            200, json={"actual_status": actual_status, "duration": 0.5, "status_msg": None}
        ),
    )
    status = await _live_provider().status("987654")
    assert status.state == expected
    assert status.uptime_seconds == 1800  # 0.5h → seconds


async def test_live_status_error_returns_failed(monkeypatch):
    def exploding(request):
        raise httpx.ReadTimeout("timeout")

    _mock_httpx(monkeypatch, exploding)
    status = await _live_provider().status("987654")
    assert status.state == InstanceState.FAILED
    assert "timeout" in (status.error_message or "")


async def test_live_destroy_puts_destroyed_state(monkeypatch):
    requests = _mock_httpx(monkeypatch, lambda req: httpx.Response(200, json={}))
    await _live_provider().destroy("987654")
    assert requests[0].method == "PUT"
    assert json.loads(requests[0].content) == {"state": "destroyed"}


async def test_destroy_swallows_errors(monkeypatch):
    def exploding(request):
        raise httpx.ConnectError("gone")

    _mock_httpx(monkeypatch, exploding)
    await _live_provider().destroy("987654")  # must not raise


# --- demo mode (no key) ----------------------------------------------------------

async def test_demo_provision_synthesizes_instance(monkeypatch):
    monkeypatch.delenv("VASTAI_API_KEY", raising=False)
    provider = VastAIProvider()
    assert not provider.reachable
    instance = await provider.provision(_cached_offer())
    assert instance.metadata["demo_mode"] is True
    assert instance.instance_id.startswith("vastai-")
