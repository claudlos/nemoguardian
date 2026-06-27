"""Provider registry — singleton wiring + helpers used by the API."""

from __future__ import annotations

from nemoguardian.providers.base import (
    CATALOG,
    GPUProvider,
    Offer,
    ProviderCatalogEntry,
    ProviderName,
    Registry,
    offers_fitting_cascade,
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


def default_registry() -> Registry:
    """Build the default provider registry. Cheap to construct."""
    reg = Registry()
    reg.register(VastAIProvider())         # live (if VASTAI_API_KEY set)
    reg.register(HetznerProvider())
    reg.register(DigitalOceanProvider())
    reg.register(LambdaProvider())
    reg.register(RunPodProvider())
    reg.register(SaladProvider())
    reg.register(GPUhubProvider())
    reg.register(LyceumProvider())
    reg.register(OnPremProvider())
    return reg


_REGISTRY: Registry | None = None


def _get_registry() -> Registry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = default_registry()
    return _REGISTRY


def list_providers() -> list[ProviderCatalogEntry]:
    """All known providers with their catalog metadata."""
    return [CATALOG[name] for name in ProviderName]


def get_provider(name: ProviderName) -> GPUProvider:
    return _get_registry().get(name)


async def provision_cheapest_fit(
    *,
    preferred_providers: list[ProviderName] | None = None,
    ssh_public_key: str | None = None,
    image: str = "nemoguardian/self-hosted:latest",
    env: dict[str, str] | None = None,
    max_price_usd: float | None = None,
) -> tuple[Offer, object]:
    """Find the cheapest viable offer across providers, then provision it.

    Returns (offer, instance). Raises ProvisionError if nothing fits.
    """
    providers = preferred_providers or [
        ProviderName.VAST_AI,
        ProviderName.SALAD,
        ProviderName.RUNPOD,
        ProviderName.GPUHUB,
        ProviderName.HETZNER,
        ProviderName.LAMBDA,
        ProviderName.DIGITAL_OCEAN,
        ProviderName.LYCEUM,
    ]
    reg = _get_registry()

    # Collect ALL viable offers across providers, pick cheapest.
    all_offers: list[Offer] = []
    for pname in providers:
        try:
            offers = await reg.get(pname).list_offers(max_price_usd=max_price_usd)
            all_offers.extend(offers)
        except Exception:
            continue
    fits = offers_fitting_cascade(all_offers)
    if not fits:
        from nemoguardian.providers.base import ProvisionError
        raise ProvisionError(
            f"no offers fit the cascade (need >=24GB VRAM)"
            + (f" under ${max_price_usd}/hr" if max_price_usd else "")
        )
    cheapest = sorted(fits, key=lambda o: o.price_per_hour_usd)[0]
    instance = await reg.get(cheapest.provider).provision(
        cheapest, ssh_public_key=ssh_public_key, image=image, env=env,
    )
    return cheapest, instance


__all__ = [
    "default_registry",
    "list_providers",
    "get_provider",
    "provision_cheapest_fit",
]
