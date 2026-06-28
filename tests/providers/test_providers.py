"""Tests for the GPU provider registry + Vast.ai + stubs."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import nemoguardian.providers.vastai as vastai_module
from nemoguardian.providers import (
    ProviderName,
    default_registry,
    get_provider,
    list_providers,
    offers_fitting_cascade,
)
from nemoguardian.providers.base import (
    CASCADE_VRAM_COMFORT_GB,
    InstanceState,
    Offer,
    ProvisionError,
    estimate_cascade_vram_gb,
)
from nemoguardian.providers.onprem import OnPremProvider
from nemoguardian.providers.stubs import (
    DigitalOceanProvider,
    GPUhubProvider,
    HetznerProvider,
    LambdaProvider,
    LyceumProvider,
    RunPodProvider,
    SaladProvider,
)
from nemoguardian.providers.vastai import VastAIProvider


class _FakeResponse:
    def __init__(self, payload: dict[str, Any] | None = None, *, error: Exception | None = None):
        self.payload = payload or {}
        self.error = error

    def raise_for_status(self) -> None:
        if self.error:
            raise self.error

    def json(self) -> dict[str, Any]:
        return self.payload


class _FakeVastAIClient:
    def __init__(
        self,
        *,
        post_response: _FakeResponse | None = None,
        get_response: _FakeResponse | None = None,
        put_response: _FakeResponse | None = None,
        post_error: Exception | None = None,
        get_error: Exception | None = None,
        put_error: Exception | None = None,
    ) -> None:
        self.post_response = post_response or _FakeResponse()
        self.get_response = get_response or _FakeResponse()
        self.put_response = put_response or _FakeResponse()
        self.post_error = post_error
        self.get_error = get_error
        self.put_error = put_error
        self.calls: list[dict[str, Any]] = []
        self.timeout: float | None = None

    async def __aenter__(self) -> _FakeVastAIClient:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    async def post(self, url: str, *, json: dict[str, Any], headers: dict[str, str]):
        self.calls.append({"method": "POST", "url": url, "json": json, "headers": headers})
        if self.post_error:
            raise self.post_error
        return self.post_response

    async def get(self, url: str, *, headers: dict[str, str]):
        self.calls.append({"method": "GET", "url": url, "headers": headers})
        if self.get_error:
            raise self.get_error
        return self.get_response

    async def put(self, url: str, *, json: dict[str, Any], headers: dict[str, str]):
        self.calls.append({"method": "PUT", "url": url, "json": json, "headers": headers})
        if self.put_error:
            raise self.put_error
        return self.put_response


def _patch_vastai_http(monkeypatch: pytest.MonkeyPatch, fake: _FakeVastAIClient) -> None:
    def client_factory(*, timeout: float) -> _FakeVastAIClient:
        fake.timeout = timeout
        return fake

    monkeypatch.setattr(vastai_module.httpx, "AsyncClient", client_factory)


# ---------- Catalog / helpers ----------


def test_catalog_covers_all_providers():
    catalog = list_providers()
    assert len(catalog) == len(ProviderName)
    assert {c.name for c in catalog} == set(ProviderName)


def test_catalog_entries_serialize():
    entry = list_providers()[0]
    data = entry.to_dict()

    assert data["name"] == entry.name.value
    assert data["display_name"] == entry.display_name
    assert data["supports_provisioning"] == entry.supports_provisioning


def test_cascade_vram_default():
    assert estimate_cascade_vram_gb() == CASCADE_VRAM_COMFORT_GB
    assert estimate_cascade_vram_gb(quantized=True) < CASCADE_VRAM_COMFORT_GB


def test_offers_fitting_cascade_filters_small_gpus():
    offers = [
        Offer(ProviderName.VAST_AI, "RTX 3060", 12, 0.03, "Global"),
        Offer(ProviderName.VAST_AI, "RTX 3090", 24, 0.07, "Global"),
        Offer(ProviderName.HETZNER, "RTX PRO 6000", 96, 2.10, "EU"),
    ]
    fits = offers_fitting_cascade(offers)
    assert len(fits) == 2
    assert {o.gpu_model for o in fits} == {"RTX 3090", "RTX PRO 6000"}


# ---------- Registry ----------


def test_default_registry_has_all_providers():
    reg = default_registry()
    assert len(reg.all()) == len(ProviderName)
    for pname in ProviderName:
        assert pname in reg.providers


def test_get_provider_returns_correct_impl():
    assert isinstance(get_provider(ProviderName.VAST_AI), VastAIProvider)
    assert isinstance(get_provider(ProviderName.ON_PREM), OnPremProvider)
    assert isinstance(get_provider(ProviderName.HETZNER), HetznerProvider)
    assert isinstance(get_provider(ProviderName.RUNPOD), RunPodProvider)


# ---------- Vast.ai ----------


def test_vastai_cached_offers_without_api_key(monkeypatch):
    monkeypatch.delenv("VASTAI_API_KEY", raising=False)
    p = VastAIProvider(api_key="")
    assert p.reachable is False
    offers = asyncio.run(p.list_offers())
    assert len(offers) >= 5
    assert offers[0].gpu_model == "RTX 3090"
    assert offers[0].price_per_hour_usd == 0.07


def test_vastai_filter_by_gpu_model(monkeypatch):
    monkeypatch.delenv("VASTAI_API_KEY", raising=False)
    p = VastAIProvider(api_key="")
    a100 = asyncio.run(p.list_offers(gpu_model="A100 40GB"))
    assert all(o.gpu_model == "A100 40GB" for o in a100)
    assert len(a100) == 1


def test_vastai_filter_by_max_price(monkeypatch):
    monkeypatch.delenv("VASTAI_API_KEY", raising=False)
    p = VastAIProvider(api_key="")
    cheap = asyncio.run(p.list_offers(max_price_usd=0.10))
    assert all(o.price_per_hour_usd <= 0.10 for o in cheap)


@pytest.mark.asyncio
async def test_vastai_live_offers_parse_sorted_and_payload(monkeypatch):
    fake = _FakeVastAIClient(
        post_response=_FakeResponse({
            "offers": [
                {
                    "gpu_name": "RTX_4090",
                    "gpu_ram": 24576,
                    "dph_total": 0.20,
                    "geolocation": "US",
                    "id": 22,
                    "inet_down": 5000,
                },
                {
                    "gpu_name": "RTX_3090",
                    "gpu_ram": 24576,
                    "dph_total": 0.08,
                    "geolocation": "Canada",
                    "id": 11,
                    "inet_down": 2500,
                },
            ],
        }),
    )
    _patch_vastai_http(monkeypatch, fake)

    provider = VastAIProvider(api_key="test-key", timeout=9.5)
    offers = await provider.list_offers(gpu_model="RTX 3090", max_price_usd=0.12)

    assert provider.reachable is True
    assert fake.timeout == 9.5
    assert fake.calls[0]["url"].endswith("/asks/")
    assert fake.calls[0]["headers"] == {"Authorization": "Bearer test-key"}
    assert fake.calls[0]["json"]["gpu_name"] == ["RTX_3090"]
    assert fake.calls[0]["json"]["dph_max"] == 0.12
    assert [offer.gpu_model for offer in offers] == ["RTX 3090", "RTX 4090"]
    assert offers[0].vram_gb == 24
    assert offers[0].offer_id == "11"
    assert offers[0].bandwidth_gbps == 2.5


@pytest.mark.asyncio
async def test_vastai_live_offer_failure_falls_back_to_cached(monkeypatch):
    fake = _FakeVastAIClient(post_error=RuntimeError("network unavailable"))
    _patch_vastai_http(monkeypatch, fake)

    provider = VastAIProvider(api_key="test-key")
    offers = await provider.list_offers(gpu_model="RTX 3090", max_price_usd=0.10)

    assert [offer.gpu_model for offer in offers] == ["RTX 3090"]
    assert offers[0].price_per_hour_usd == 0.07


@pytest.mark.asyncio
async def test_vastai_demo_status_and_destroy_are_local(monkeypatch):
    monkeypatch.delenv("VASTAI_API_KEY", raising=False)
    provider = VastAIProvider(api_key="")

    status = await provider.status("demo-1")
    await provider.destroy("demo-1")

    assert status.state == InstanceState.LIVE
    assert status.uptime_seconds == 600
    assert status.last_health_check == "2026-06-24T16:00:00Z"


@pytest.mark.asyncio
async def test_vastai_live_provision_posts_payload_and_returns_instance(monkeypatch):
    fake = _FakeVastAIClient(
        post_response=_FakeResponse({
            "new_contract": 12345,
            "ssh_port": 2222,
            "ssh_host": "gpu.example.net",
        }),
    )
    _patch_vastai_http(monkeypatch, fake)
    offer = Offer(ProviderName.VAST_AI, "RTX 3090", 24, 0.08, "Canada", offer_id="11")

    provider = VastAIProvider(api_key="test-key")
    instance = await provider.provision(
        offer,
        ssh_public_key="ssh-ed25519 AAAA test",
        image="nemoguardian:test",
        env={"NEMOGUARDIAN_MODE": "test"},
    )

    payload = fake.calls[0]["json"]
    assert fake.calls[0]["url"].endswith("/create_inst/")
    assert payload["image"] == "nemoguardian:test"
    assert payload["gpu_name"] == "RTX 3090"
    assert payload["price"] == pytest.approx(0.088)
    assert payload["ssh_key"] == "ssh-ed25519 AAAA test"
    assert payload["env"] == {"NEMOGUARDIAN_MODE": "test"}
    assert instance.instance_id == "12345"
    assert instance.state == InstanceState.PROVISIONING
    assert instance.ssh_command == "ssh -p 2222 root@gpu.example.net"
    assert instance.metadata == {"vastai_contract": "12345"}


@pytest.mark.asyncio
async def test_vastai_live_provision_raises_provider_error(monkeypatch):
    fake = _FakeVastAIClient(post_response=_FakeResponse(error=RuntimeError("denied")))
    _patch_vastai_http(monkeypatch, fake)
    offer = Offer(ProviderName.VAST_AI, "RTX 3090", 24, 0.08, "Canada")

    provider = VastAIProvider(api_key="test-key")
    with pytest.raises(ProvisionError, match=r"vast\.ai create_inst failed: denied"):
        await provider.provision(offer)


@pytest.mark.asyncio
async def test_vastai_live_status_maps_running_instance(monkeypatch):
    fake = _FakeVastAIClient(
        get_response=_FakeResponse({
            "actual_status": "running",
            "duration": 1.25,
            "last_status": "2026-06-24T16:00:00Z",
            "status_msg": "healthy",
        }),
    )
    _patch_vastai_http(monkeypatch, fake)

    provider = VastAIProvider(api_key="test-key")
    status = await provider.status("12345")

    assert fake.calls[0]["url"].endswith("/instances/12345/")
    assert status.instance_id == "12345"
    assert status.state == InstanceState.LIVE
    assert status.uptime_seconds == 4500
    assert status.last_health_check == "2026-06-24T16:00:00Z"
    assert status.error_message == "healthy"


@pytest.mark.asyncio
async def test_vastai_live_status_failure_returns_failed_status(monkeypatch):
    fake = _FakeVastAIClient(get_error=RuntimeError("timeout"))
    _patch_vastai_http(monkeypatch, fake)

    provider = VastAIProvider(api_key="test-key")
    status = await provider.status("12345")

    assert status.state == InstanceState.FAILED
    assert status.error_message == "timeout"


@pytest.mark.asyncio
async def test_vastai_live_destroy_posts_destroyed_state(monkeypatch):
    fake = _FakeVastAIClient()
    _patch_vastai_http(monkeypatch, fake)

    provider = VastAIProvider(api_key="test-key")
    await provider.destroy("12345")

    assert fake.calls[0] == {
        "method": "PUT",
        "url": "https://vast.ai/api/v0/instances/12345/",
        "json": {"state": "destroyed"},
        "headers": {"Authorization": "Bearer test-key"},
    }


@pytest.mark.asyncio
async def test_vastai_live_destroy_swallows_errors(monkeypatch):
    fake = _FakeVastAIClient(put_error=RuntimeError("already gone"))
    _patch_vastai_http(monkeypatch, fake)

    provider = VastAIProvider(api_key="test-key")
    await provider.destroy("12345")

    assert fake.calls[0]["method"] == "PUT"


# ---------- Stubs ----------


def test_stub_providers_return_static_offers():
    for cls in [
        HetznerProvider, DigitalOceanProvider, LambdaProvider,
        RunPodProvider, SaladProvider, GPUhubProvider, LyceumProvider,
    ]:
        p = cls()
        offers = asyncio.run(p.list_offers())
        assert len(offers) >= 1, f"{cls.__name__} returned no offers"
        for o in offers:
            assert o.provider == p.name


def test_stub_provisioning_raises():
    p = HetznerProvider()
    offer = asyncio.run(p.list_offers())[0]
    with pytest.raises(ProvisionError, match="not yet implemented"):
        asyncio.run(p.provision(offer))


def test_stub_status_returns_documented():
    p = HetznerProvider()
    status = asyncio.run(p.status("hetzner-fake-1"))
    assert status.state == InstanceState.DESTROYED


def test_stub_provider_filters_and_destroy():
    p = RunPodProvider()
    offers = asyncio.run(p.list_offers(gpu_model="L4", max_price_usd=0.15))

    assert [offer.gpu_model for offer in offers] == ["L4"]
    asyncio.run(p.destroy("runpod-fake-1"))


# ---------- On-prem ----------


def test_onprem_renders_docker_compose():
    p = OnPremProvider()
    offer = asyncio.run(p.list_offers())[0]
    instance = asyncio.run(p.provision(
        offer,
        ssh_public_key="ssh-ed25519 AAAA test",
        env={"NEMOGUARDIAN_API_KEY": "nmg_test"},
    ))
    assert instance.state == InstanceState.LIVE
    compose = instance.metadata["docker_compose"]
    assert "docker compose" in compose
    assert "nvidia" in compose.lower()
    # Env values are rendered into the docker-compose text.
    assert "nmg_test" in compose
    assert "NEMOGUARDIAN_API_KEY" in compose


def test_onprem_negative_price_status_destroy_and_placeholder():
    p = OnPremProvider()

    assert asyncio.run(p.list_offers(max_price_usd=-0.01)) == []
    offer = asyncio.run(p.list_offers())[0]
    instance = asyncio.run(p.provision(offer))
    status = asyncio.run(p.status(instance.instance_id))
    asyncio.run(p.destroy(instance.instance_id))

    assert "NEMOGUARDIAN_API_KEY: <paste from billing/welcome>" in instance.metadata["docker_compose"]
    assert instance.metadata["ssh_public_key"] == ""
    assert status.state == InstanceState.LIVE
    assert status.error_message == "on-prem status reported via /health endpoint ping"


# ---------- Cross-provider catalog ----------


def test_all_offers_have_required_fields():
    reg = default_registry()
    for provider in reg.all():
        offers = asyncio.run(provider.list_offers())
        for o in offers:
            assert o.gpu_model
            assert o.vram_gb >= 0
            assert o.price_per_hour_usd >= 0
            d = o.to_dict()
            assert d["monthly_price_usd"] >= 0
            assert d["provider"] == provider.name.value


# ---------- Cross-provider routing ----------


@pytest.mark.asyncio
async def test_provision_cheapest_fit_picks_rtx_3090(monkeypatch):
    monkeypatch.delenv("VASTAI_API_KEY", raising=False)
    import nemoguardian.providers.registry as reg_module
    from nemoguardian.providers.registry import provision_cheapest_fit
    reg_module._REGISTRY = None
    offer, instance = await provision_cheapest_fit()
    assert offer.price_per_hour_usd <= 0.07
    assert instance.provider == offer.provider


@pytest.mark.asyncio
async def test_provision_cheapest_fit_reports_no_fit_after_provider_errors(monkeypatch):
    import nemoguardian.providers.registry as reg_module
    from nemoguardian.providers.base import Registry
    from nemoguardian.providers.registry import provision_cheapest_fit

    class EmptyProvider:
        name = ProviderName.VAST_AI

        async def list_offers(self, *, max_price_usd=None, gpu_model=None):
            return []

    class BrokenProvider:
        name = ProviderName.SALAD

        async def list_offers(self, *, max_price_usd=None, gpu_model=None):
            raise RuntimeError("provider down")

    registry = Registry({
        ProviderName.VAST_AI: EmptyProvider(),
        ProviderName.SALAD: BrokenProvider(),
    })
    monkeypatch.setattr(reg_module, "_REGISTRY", registry)

    with pytest.raises(ProvisionError, match="no offers fit the cascade"):
        await provision_cheapest_fit(preferred_providers=[ProviderName.VAST_AI, ProviderName.SALAD])
