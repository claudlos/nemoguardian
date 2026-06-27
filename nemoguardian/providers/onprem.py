"""On-prem provider — generates a docker-compose snippet for the customer.

This isn't really a "provider" in the cloud sense, but it fits the same
interface so the provisioning UI can offer it as one of the choices.
"Provisioning" means: render the docker-compose, SSH key, and env file.
"""

from __future__ import annotations

import secrets

from nemoguardian.providers.base import (
    Instance,
    InstanceState,
    InstanceStatus,
    Offer,
    ProviderName,
)

_OFFER = Offer(
    provider=ProviderName.ON_PREM,
    gpu_model="customer-hardware",
    vram_gb=0,
    price_per_hour_usd=0.0,
    region="Your datacenter",
    offer_id="onprem-render",
    notes="Customer hosts nemoguardian. We ship a Docker image + docker-compose.",
)


class OnPremProvider:
    name = ProviderName.ON_PREM

    async def list_offers(
        self, *, gpu_model: str | None = None, max_price_usd: float | None = None
    ) -> list[Offer]:
        if max_price_usd is not None and max_price_usd < 0:
            return []
        return [_OFFER]

    async def provision(
        self,
        offer: Offer,
        *,
        ssh_public_key: str | None = None,
        image: str = "nemoguardian/self-hosted:latest",
        env: dict[str, str] | None = None,
    ) -> Instance:
        instance_id = f"onprem-{secrets.token_hex(4)}"
        docker_compose = _render_compose(image=image, env=env or {})
        ssh_command = f"# On-prem: see {instance_id}.md for setup steps\n"
        return Instance(
            provider=self.name,
            instance_id=instance_id,
            gpu_model=offer.gpu_model,
            vram_gb=0,
            region=offer.region,
            state=InstanceState.LIVE,
            ssh_command=ssh_command,
            hourly_price_usd=0.0,
            metadata={
                "docker_compose": docker_compose,
                "ssh_public_key": ssh_public_key or "",
                "instructions_url": f"/billing/jobs/{instance_id}/onprem-setup.md",
            },
        )

    async def status(self, instance_id: str) -> InstanceStatus:
        # On-prem status is the customer's responsibility.
        return InstanceStatus(
            instance_id=instance_id,
            state=InstanceState.LIVE,
            uptime_seconds=0,
            error_message="on-prem status reported via /health endpoint ping",
        )

    async def destroy(self, instance_id: str) -> None:
        # We can't tear down a customer's own hardware. Just record.
        return None


def _render_compose(*, image: str, env: dict[str, str]) -> str:
    """Render the docker-compose.yml for the customer's host."""
    env_lines = "\n".join(f"      {k}: {v}" for k, v in env.items())
    if not env_lines:
        env_lines = "      # NEMOGUARDIAN_API_KEY: <paste from billing/welcome>"
    return (
        f"# nemoguardian self-hosted — bring your own GPU\n"
        f"# Recommended: NVIDIA driver 535+, Docker 24+, NVIDIA Container Toolkit\n\n"
        f"version: '3.9'\n\n"
        f"services:\n"
        f"  nemoguardian:\n"
        f"    image: {image}\n"
        f"    runtime: nvidia\n"
        f"    ports:\n"
        f"      - \"8000:8000\"\n"
        f"    environment:\n"
        f"{env_lines}\n"
        f"    deploy:\n"
        f"      resources:\n"
        f"        reservations:\n"
        f"          devices:\n"
        f"            - driver: nvidia\n"
        f"              count: 1\n"
        f"              capabilities: [gpu]\n"
        f"    restart: unless-stopped\n\n"
        f"# docker compose up -d\n"
        f"# curl http://localhost:8000/health\n"
    )


__all__ = ["OnPremProvider"]
