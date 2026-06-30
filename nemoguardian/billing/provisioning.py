"""Self-hosted tier provisioning.

When a customer signs up for the self-hosted plan we provision a remote
instance running nemoguardian. Today this is a stub that simulates the
provider API; tomorrow it would call Vast.ai / DigitalOcean / Lambda Labs.

Flow:
1. Customer completes Stripe Checkout for self-hosted.
2. Webhook handler calls `provision_instance()` with provider + ssh key.
3. We create a ProvisioningJob (DB row).
4. We POST to the provider's API (simulated) and update the row.
5. Customer can poll /billing/jobs/<id> to see the status + endpoint.
"""

from __future__ import annotations

import asyncio
import os
import secrets
from typing import TYPE_CHECKING, Literal

from nemoguardian.billing import db
from nemoguardian.billing.plans import Tier
from nemoguardian.providers.ops import (
    GpuOpsConfig,
    ProvisionResult,
    ProvisionStatus,
    provision_guarded,
)

if TYPE_CHECKING:
    from nemoguardian.providers.base import Offer
    from nemoguardian.providers.ops import OpsEventLog, _Provisioner

Provider = Literal["vastai", "digitalocean", "lambda", "on_prem"]


SUPPORTED_PROVIDERS: list[Provider] = ["vastai", "digitalocean", "lambda", "on_prem"]
_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()


async def provision_instance(
    *,
    customer_id: int,
    provider: Provider = "vastai",
    ssh_public_key: str | None = None,
) -> db.ProvisioningJob:
    """Create + run a provisioning job. Returns the job record.

    The provider call is simulated. In production this would dispatch to the
    right cloud API. For `on_prem` we just generate a docker-compose snippet.
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"unsupported provider {provider!r}")
    job = db.create_provisioning_job(customer_id=customer_id, tier=Tier.SELF_HOSTED, provider=provider)
    task = asyncio.create_task(_run_job(job.id, provider, ssh_public_key))
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)
    return job


async def _run_job(job_id: int, provider: Provider, ssh_public_key: str | None) -> None:
    """Background task: simulate the provider API and update the job row."""
    db.update_provisioning_job(job_id, status="provisioning")
    try:
        await asyncio.sleep(0.5)  # simulate API latency
        if provider == "on_prem":
            endpoint_url = None
            ssh_command = _render_onprem_snippet(ssh_public_key)
            instance_id = f"onprem-{job_id}"
        else:
            instance_id = f"{provider}-{secrets.token_hex(6)}"
            endpoint_url = f"https://{instance_id}.nemoguardian.dev"
            ssh_command = (
                f"ssh -i ~/.ssh/nemoguardian_{instance_id} "
                f"nemoguardian@{instance_id}.nemoguardian.dev"
            )

        # Demo safeguard: ensure the API key secret stays on the user side.
        env_key = os.environ.get(f"NEMOGUARDIAN_{provider.upper()}_API_KEY")
        if env_key:
            # Real call would happen here. For the demo we still mark live.
            pass

        db.update_provisioning_job(
            job_id,
            status="live",
            instance_id=instance_id,
            endpoint_url=endpoint_url,
            ssh_command=ssh_command,
        )
    except Exception as exc:
        db.update_provisioning_job(job_id, status="failed", error_message=str(exc))


# --------------------------------------------------------------------------- #
# Guarded provisioning (audit #45 hookup)                                      #
# --------------------------------------------------------------------------- #
#
# The legacy ``provision_instance`` path above *simulates* a provider and is kept
# for the demo. The guarded path below routes a real provision through
# :func:`nemoguardian.providers.ops.provision_guarded` so spend caps
# (max-price / max-hours) and the no-auto-spend confirm gate are enforced
# **before any live spend**. Over-cap → ``rejected`` (no provider call); caps OK
# but unconfirmed → ``planned`` (a dry-run, still no spend); confirmed + in-cap →
# the provider is called and the job goes ``live``. Billing-ledger logic in
# ``billing/db.py`` is untouched — this only gates the provisioning call.

# Provision results that did not create a live instance map onto these job
# statuses (the ``provisioning_jobs.status`` column is free-form TEXT).
_GUARDED_JOB_STATUS: dict[ProvisionStatus, str] = {
    ProvisionStatus.PROVISIONED: "live",
    ProvisionStatus.PLANNED: "planned",
    ProvisionStatus.REJECTED: "rejected",
    ProvisionStatus.FAILED: "failed",
}


async def provision_instance_guarded(
    *,
    customer_id: int,
    provider: Provider,
    offer: Offer,
    provider_client: _Provisioner,
    reserve_hours: float,
    confirm: bool = False,
    ops_config: GpuOpsConfig | None = None,
    ssh_public_key: str | None = None,
    image: str = "nemoguardian/self-hosted:latest",
    env: dict[str, str] | None = None,
    event_log: OpsEventLog | None = None,
) -> db.ProvisioningJob:
    """Create + run a *guarded* provisioning job and return the job record.

    Unlike :func:`provision_instance`, the provider call is gated by
    :func:`provision_guarded`: the spend caps and the explicit-confirm guard run
    before any money is spent. The job is run inline (the guard decision is cheap
    and the returned job already reflects the outcome).
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(f"unsupported provider {provider!r}")
    job = db.create_provisioning_job(
        customer_id=customer_id, tier=Tier.SELF_HOSTED, provider=provider
    )
    await _run_guarded_job(
        job.id,
        offer,
        provider_client=provider_client,
        reserve_hours=reserve_hours,
        confirm=confirm,
        ops_config=ops_config,
        ssh_public_key=ssh_public_key,
        image=image,
        env=env,
        event_log=event_log,
    )
    return db.get_provisioning_job(job.id)


async def _run_guarded_job(
    job_id: int,
    offer: Offer,
    *,
    provider_client: _Provisioner,
    reserve_hours: float,
    confirm: bool = False,
    ops_config: GpuOpsConfig | None = None,
    ssh_public_key: str | None = None,
    image: str = "nemoguardian/self-hosted:latest",
    env: dict[str, str] | None = None,
    event_log: OpsEventLog | None = None,
) -> ProvisionResult:
    """Run the guarded provision for ``job_id`` and reflect the result on the row.

    Returns the :class:`ProvisionResult` so callers (and tests) can assert on the
    cap decision directly. The provider is only ever called when the caps pass
    *and* the request is confirmed — otherwise no spend happens.
    """
    config = ops_config or GpuOpsConfig.from_env()
    db.update_provisioning_job(job_id, status="provisioning")
    result = await provision_guarded(
        provider_client,
        offer,
        config=config,
        reserve_hours=reserve_hours,
        confirm=confirm,
        ssh_public_key=ssh_public_key,
        image=image,
        env=env,
        event_log=event_log,
    )
    status = _GUARDED_JOB_STATUS.get(result.status, "failed")
    if result.status is ProvisionStatus.PROVISIONED and result.instance is not None:
        inst = result.instance
        db.update_provisioning_job(
            job_id,
            status=status,
            instance_id=inst.instance_id,
            endpoint_url=inst.endpoint_url,
            ssh_command=inst.ssh_command,
        )
    else:
        # planned / rejected / failed → no live instance; record the reason.
        db.update_provisioning_job(job_id, status=status, error_message=result.reason)
    return result


def _render_onprem_snippet(ssh_public_key: str | None) -> str:
    """A docker-compose snippet the user runs locally."""
    key_block = ssh_public_key or "<paste your SSH public key>"
    return (
        "# 1. Drop this in a fresh directory on your machine\n"
        "mkdir nemoguardian-self-hosted && cd nemoguardian-self-hosted\n\n"
        "# 2. Save your SSH public key\n"
        f"echo '{key_block}' > authorized_keys\n\n"
        "# 3. Pull and run\n"
        "docker compose up -d\n\n"
        "# 4. Verify\n"
        "curl http://localhost:8000/health\n"
    )


__all__ = [
    "SUPPORTED_PROVIDERS",
    "Provider",
    "provision_instance",
    "provision_instance_guarded",
]
