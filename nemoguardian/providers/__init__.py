"""GPU provider registry.

Each provider implements the GPUProvider protocol. The registry exposes a
catalog of all known providers, lets the API/UI enumerate offers, and picks
the cheapest viable offer for a given model set.

Dev focus: Vast.ai (full implementation). All others ship as stubs that
return the provider's catalog metadata + a clear "coming soon" status so the
UI can show them in the cost comparison table.

The hero line for the demo: "the same nemoguardian image runs on RTX 3090 at
$0.07/hr, A100 at $0.14/hr, or H100 at $0.19/hr — pick your poison."
"""

from nemoguardian.providers.base import (
    GPUProvider,
    Instance,
    InstanceStatus,
    Offer,
    ProviderName,
    ProvisionError,
    Registry,
    estimate_cascade_vram_gb,
    offers_fitting_cascade,
)
from nemoguardian.providers.registry import (
    default_registry,
    list_providers,
    get_provider,
    provision_cheapest_fit,
)

__all__ = [
    "GPUProvider",
    "Instance",
    "InstanceStatus",
    "Offer",
    "ProviderName",
    "ProvisionError",
    "Registry",
    "estimate_cascade_vram_gb",
    "offers_fitting_cascade",
    "default_registry",
    "list_providers",
    "get_provider",
    "provision_cheapest_fit",
]
