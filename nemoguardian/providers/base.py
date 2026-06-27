"""GPU provider base protocol + data classes.

A `GPUProvider` exposes a uniform interface to enumerate offers, provision
an instance, check its status, and destroy it. Providers that don't yet have
a real implementation can subclass `StubProvider` to return catalog metadata
without making API calls.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol, runtime_checkable

# --- Provider IDs (canonical names) --------------------------------------

class ProviderName(str, Enum):
    VAST_AI = "vastai"
    HETZNER = "hetzner"
    DIGITAL_OCEAN = "digitalocean"
    LAMBDA = "lambda"
    RUNPOD = "runpod"
    SALAD = "salad"
    GPUHUB = "gpuhub"
    LYCEUM = "lyceum"
    ON_PREM = "on_prem"


# --- Catalog (static per-provider) ----------------------------------------

@dataclass(frozen=True)
class ProviderCatalogEntry:
    """Static metadata about a provider. Used by the cost-comparison UI."""

    name: ProviderName
    display_name: str
    region: str                        # "EU" | "US" | "Global" | "DE" | etc.
    supports_provisioning: bool       # False for on-prem / stub providers
    smallest_gpu_vram_gb: int
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "display_name": self.display_name,
            "region": self.region,
            "supports_provisioning": self.supports_provisioning,
            "smallest_gpu_vram_gb": self.smallest_gpu_vram_gb,
            "notes": self.notes,
        }


CATALOG: dict[ProviderName, ProviderCatalogEntry] = {
    ProviderName.VAST_AI: ProviderCatalogEntry(
        ProviderName.VAST_AI, "Vast.ai", "Global",
        True, 16,
        "Marketplace; 2,071 offers; RTX 3090 from $0.07/hr. You already have a key.",
    ),
    ProviderName.HETZNER: ProviderCatalogEntry(
        ProviderName.HETZNER, "Hetzner", "EU (Germany/Finland)",
        True, 20,
        "GEX44 RTX 4000 SFF (20GB, €0.375/hr) or GEX131 RTX Pro 6000 (96GB, €1.92/hr). GDPR-compliant.",
    ),
    ProviderName.DIGITAL_OCEAN: ProviderCatalogEntry(
        ProviderName.DIGITAL_OCEAN, "DigitalOcean", "US/EU",
        True, 20,
        "GPU Droplets; RTX 4000 from $0.76/hr; H100 from $3.39/hr. Simple per-second billing.",
    ),
    ProviderName.LAMBDA: ProviderCatalogEntry(
        ProviderName.LAMBDA, "Lambda Labs", "US",
        True, 24,
        "A10G / A100 / H100; reserved & spot pricing. Polished UI.",
    ),
    ProviderName.RUNPOD: ProviderCatalogEntry(
        ProviderName.RUNPOD, "RunPod", "Global",
        True, 16,
        "Serverless + dedicated; L4 from $0.14/hr; RTX 3090 from $0.12/hr.",
    ),
    ProviderName.SALAD: ProviderCatalogEntry(
        ProviderName.SALAD, "Salad", "Global",
        True, 12,
        "Cheapest consumer GPU prices; RTX 3090 from $0.07/hr.",
    ),
    ProviderName.GPUHUB: ProviderCatalogEntry(
        ProviderName.GPUHUB, "GPUhub", "Global",
        True, 24,
        "Zero egress fees; good for serving high-traffic APIs.",
    ),
    ProviderName.LYCEUM: ProviderCatalogEntry(
        ProviderName.LYCEUM, "Lyceum", "EU-sovereign",
        True, 80,
        "H200 from $2.20/hr; B200 from $3.40/hr; scale to 8,000 GPUs.",
    ),
    ProviderName.ON_PREM: ProviderCatalogEntry(
        ProviderName.ON_PREM, "On-prem (Docker)", "Your datacenter",
        False, 0,
        "Customer runs docker compose on their own hardware. We just ship the image.",
    ),
}


# --- Offer / Instance / Status --------------------------------------------

@dataclass(frozen=True)
class Offer:
    """A single rentable compute offer."""

    provider: ProviderName
    gpu_model: str
    vram_gb: int
    price_per_hour_usd: float
    region: str
    offer_id: str = ""                # provider-specific id (empty for stubs)
    instance_type: str = ""           # provider-specific machine type
    bandwidth_gbps: float | None = None
    ephemeral_storage_gb: int | None = None
    reliability_score: float = 1.0    # 0.0-1.0 (heuristic)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider.value,
            "gpu_model": self.gpu_model,
            "vram_gb": self.vram_gb,
            "price_per_hour_usd": self.price_per_hour_usd,
            "region": self.region,
            "offer_id": self.offer_id,
            "instance_type": self.instance_type,
            "bandwidth_gbps": self.bandwidth_gbps,
            "ephemeral_storage_gb": self.ephemeral_storage_gb,
            "reliability_score": self.reliability_score,
            "notes": self.notes,
            "monthly_price_usd": round(self.price_per_hour_usd * 720, 2),
        }


class InstanceState(str, Enum):
    PENDING = "pending"
    PROVISIONING = "provisioning"
    LIVE = "live"
    FAILED = "failed"
    DESTROYED = "destroyed"


@dataclass
class Instance:
    """A running instance on a provider."""

    provider: ProviderName
    instance_id: str
    gpu_model: str
    vram_gb: int
    region: str
    state: InstanceState = InstanceState.PENDING
    ssh_command: str | None = None
    endpoint_url: str | None = None
    hourly_price_usd: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InstanceStatus:
    instance_id: str
    state: InstanceState
    uptime_seconds: int = 0
    last_health_check: str | None = None
    error_message: str | None = None


# --- Cascade fit helpers --------------------------------------------------

# Cascade at FP16: Qwen3Guard-4B (8GB) + Nemotron-CSR-4B (8GB) + Qwen3Guard-Stream-0.6B (1.5GB)
# Plus ~6GB for vLLM overhead + KV cache. Round to 24GB minimum (FP16) / 17.5GB absolute floor.
CASCADE_VRAM_FLOOR_GB = 17.5
CASCADE_VRAM_COMFORT_GB = 24.0
CASCADE_VRAM_HEADROOM_GB = 32.0     # ideal — leaves room for hot prompts


def estimate_cascade_vram_gb(*, quantized: bool = False) -> float:
    """Estimated VRAM for the full cascade at the chosen precision."""
    if quantized:
        return 12.0  # INT8 → roughly half
    return CASCADE_VRAM_COMFORT_GB


def offers_fitting_cascade(offers: list[Offer]) -> list[Offer]:
    """Filter offers that can host the cascade at FP16 with comfort headroom."""
    return [o for o in offers if o.vram_gb >= CASCADE_VRAM_COMFORT_GB]


# --- Errors ---------------------------------------------------------------

class ProvisionError(RuntimeError):
    """Raised when a provider can't fulfill a provision request."""


# --- Provider protocol ----------------------------------------------------

@runtime_checkable
class GPUProvider(Protocol):
    name: ProviderName

    async def list_offers(self, *, gpu_model: str | None = None, max_price_usd: float | None = None) -> list[Offer]:
        """List viable offers from this provider, optionally filtered."""

    async def provision(
        self,
        offer: Offer,
        *,
        ssh_public_key: str | None = None,
        image: str = "nemoguardian/self-hosted:latest",
        env: dict[str, str] | None = None,
    ) -> Instance:
        """Launch an instance from a selected offer."""

    async def status(self, instance_id: str) -> InstanceStatus:
        """Check live state of a previously-provisioned instance."""

    async def destroy(self, instance_id: str) -> None:
        """Tear down the instance. Idempotent — safe to call twice."""


# --- Stub base ------------------------------------------------------------

class StubProvider(abc.ABC):
    """Base for providers that don't yet have a live API implementation.

    Subclasses must define `name` + `_static_offers()` + `_instance_for()`.
    list_offers, provision, status, and destroy are all wired for free.
    """

    name: ProviderName

    def __init__(self, *, reachable: bool = False) -> None:
        self.reachable = reachable  # "reachable" = real API, not stubbed

    @abc.abstractmethod
    def _static_offers(self) -> list[Offer]:
        """Hard-coded offers for this provider. Updated when a real client ships."""

    async def list_offers(
        self, *, gpu_model: str | None = None, max_price_usd: float | None = None
    ) -> list[Offer]:
        offers = self._static_offers()
        if gpu_model:
            offers = [o for o in offers if o.gpu_model.lower() == gpu_model.lower()]
        if max_price_usd is not None:
            offers = [o for o in offers if o.price_per_hour_usd <= max_price_usd]
        return sorted(offers, key=lambda o: o.price_per_hour_usd)

    async def provision(
        self,
        offer: Offer,
        *,
        ssh_public_key: str | None = None,
        image: str = "nemoguardian/self-hosted:latest",
        env: dict[str, str] | None = None,
    ) -> Instance:
        raise ProvisionError(
            f"{self.name.value} provisioning not yet implemented in the demo; "
            f"see provider catalog for static offer info"
        )

    async def status(self, instance_id: str) -> InstanceStatus:
        return InstanceStatus(
            instance_id=instance_id,
            state=InstanceState.DESTROYED,
            error_message="stub provider — no live status",
        )

    async def destroy(self, instance_id: str) -> None:
        return None


# --- Registry -------------------------------------------------------------

@dataclass
class Registry:
    providers: dict[ProviderName, GPUProvider] = field(default_factory=dict)

    def register(self, provider: GPUProvider) -> None:
        self.providers[provider.name] = provider

    def all(self) -> list[GPUProvider]:
        return list(self.providers.values())

    def get(self, name: ProviderName) -> GPUProvider:
        return self.providers[name]


__all__ = [
    "CASCADE_VRAM_COMFORT_GB",
    "CASCADE_VRAM_FLOOR_GB",
    "CASCADE_VRAM_HEADROOM_GB",
    "CATALOG",
    "GPUProvider",
    "Instance",
    "InstanceState",
    "InstanceStatus",
    "Offer",
    "ProviderCatalogEntry",
    "ProviderName",
    "ProvisionError",
    "Registry",
    "StubProvider",
    "estimate_cascade_vram_gb",
    "offers_fitting_cascade",
]
