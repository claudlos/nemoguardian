"""Tests for the GPU provider registry + Vast.ai + stubs."""

from __future__ import annotations

import asyncio

import pytest

from nemoguardian.providers import (
    ProviderName,
    Registry,
    default_registry,
    get_provider,
    list_providers,
    offers_fitting_cascade,
)
from nemoguardian.providers.base import (
    CASCADE_VRAM_COMFORT_GB,
    CATALOG,
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


# ---------- Catalog / helpers ----------


def test_catalog_covers_all_providers():
    catalog = list_providers()
    assert len(catalog) == len(ProviderName)
    assert {c.name for c in catalog} == set(ProviderName)


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
    from nemoguardian.providers.registry import provision_cheapest_fit

    import nemoguardian.providers.registry as reg_module
    reg_module._REGISTRY = None
    offer, instance = await provision_cheapest_fit()
    assert offer.price_per_hour_usd <= 0.07
    assert instance.provider == offer.provider
