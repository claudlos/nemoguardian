"""Stub providers for cloud GPU services we haven't implemented yet.

Each stub returns a hard-coded offer catalog that matches the real provider's
public pricing, so the cost-comparison UI is honest even before we wire up the
actual API. `provision()` raises `ProvisionError("not yet wired up")` — the
caller should catch that and fall back to Vast.ai (or whichever provider has a
real implementation).
"""

from __future__ import annotations

from nemoguardian.providers.base import (
    Offer,
    ProviderName,
    StubProvider,
)


def _o(*, provider: ProviderName, model: str, vram: int, price: float, region: str, note: str = "") -> Offer:
    return Offer(
        provider=provider,
        gpu_model=model,
        vram_gb=vram,
        price_per_hour_usd=price,
        region=region,
        offer_id=f"{provider.value}-{model.replace(' ', '').lower()}-{int(price * 100):03d}",
        reliability_score=0.85,
        notes=note,
    )


class HetznerProvider(StubProvider):
    name = ProviderName.HETZNER

    def _static_offers(self) -> list[Offer]:
        return [
            _o(provider=ProviderName.HETZNER, model="RTX 4000 SFF", vram=20, price=0.41,
               region="EU (DE/FI)", note="GEX44 dedicated. Cascade is tight at FP16 — INT8 only."),
            _o(provider=ProviderName.HETZNER, model="RTX PRO 6000 96GB", vram=96, price=2.10,
               region="EU (DE/FI)", note="GEX131 dedicated. GDPR-compliant."),
        ]


class DigitalOceanProvider(StubProvider):
    name = ProviderName.DIGITAL_OCEAN

    def _static_offers(self) -> list[Offer]:
        return [
            _o(provider=ProviderName.DIGITAL_OCEAN, model="RTX 4000", vram=20, price=0.76,
               region="US", note="Cascade tight — single model fits."),
            _o(provider=ProviderName.DIGITAL_OCEAN, model="RTX 6000 Ada", vram=48, price=1.57, region="US"),
            _o(provider=ProviderName.DIGITAL_OCEAN, model="L40S", vram=48, price=1.57, region="US"),
            _o(provider=ProviderName.DIGITAL_OCEAN, model="H100", vram=80, price=3.39, region="US"),
            _o(provider=ProviderName.DIGITAL_OCEAN, model="MI300X", vram=192, price=1.99, region="US"),
        ]


class LambdaProvider(StubProvider):
    name = ProviderName.LAMBDA

    def _static_offers(self) -> list[Offer]:
        return [
            _o(provider=ProviderName.LAMBDA, model="A10G", vram=24, price=0.69, region="US",
               note="Polished UI; on-demand + reserved."),
            _o(provider=ProviderName.LAMBDA, model="A100", vram=40, price=1.29, region="US"),
            _o(provider=ProviderName.LAMBDA, model="H100", vram=80, price=2.49, region="US"),
            _o(provider=ProviderName.LAMBDA, model="GH200", vram=96, price=2.99, region="US"),
        ]


class RunPodProvider(StubProvider):
    name = ProviderName.RUNPOD

    def _static_offers(self) -> list[Offer]:
        return [
            _o(provider=ProviderName.RUNPOD, model="RTX 3090", vram=24, price=0.12, region="Global"),
            _o(provider=ProviderName.RUNPOD, model="L4", vram=24, price=0.14, region="Global"),
            _o(provider=ProviderName.RUNPOD, model="RTX 4090", vram=24, price=0.34, region="Global"),
            _o(provider=ProviderName.RUNPOD, model="L40S", vram=48, price=0.50, region="Global"),
            _o(provider=ProviderName.RUNPOD, model="A100", vram=40, price=0.79, region="Global"),
        ]


class SaladProvider(StubProvider):
    name = ProviderName.SALAD

    def _static_offers(self) -> list[Offer]:
        return [
            _o(provider=ProviderName.SALAD, model="RTX 3090", vram=24, price=0.07, region="Global",
               note="Cheapest consumer GPU pool."),
            _o(provider=ProviderName.SALAD, model="RTX 4090", vram=24, price=0.18, region="Global"),
            _o(provider=ProviderName.SALAD, model="RTX 5090", vram=32, price=0.40, region="Global"),
            _o(provider=ProviderName.SALAD, model="A100", vram=40, price=1.50, region="Global"),
        ]


class GPUhubProvider(StubProvider):
    name = ProviderName.GPUHUB

    def _static_offers(self) -> list[Offer]:
        return [
            _o(provider=ProviderName.GPUHUB, model="RTX Pro 6000", vram=96, price=0.25, region="Global",
               note="Zero egress fees; ideal for serving high-traffic APIs."),
            _o(provider=ProviderName.GPUHUB, model="A100", vram=40, price=0.50, region="Global"),
        ]


class LyceumProvider(StubProvider):
    name = ProviderName.LYCEUM

    def _static_offers(self) -> list[Offer]:
        return [
            _o(provider=ProviderName.LYCEUM, model="H200", vram=141, price=2.20, region="EU-sovereign",
               note="Scale to 8,000 GPUs. EU compliance."),
            _o(provider=ProviderName.LYCEUM, model="B200", vram=192, price=3.40, region="EU-sovereign"),
            _o(provider=ProviderName.LYCEUM, model="B300", vram=288, price=3.80, region="EU-sovereign"),
        ]


__all__ = [
    "HetznerProvider",
    "DigitalOceanProvider",
    "LambdaProvider",
    "RunPodProvider",
    "SaladProvider",
    "GPUhubProvider",
    "LyceumProvider",
]
