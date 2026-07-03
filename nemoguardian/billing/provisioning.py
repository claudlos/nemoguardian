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
import re
import secrets
from typing import Literal

from fastapi import HTTPException

from nemoguardian.billing import db
from nemoguardian.billing.plans import Tier

Provider = Literal["vastai", "digitalocean", "lambda", "on_prem"]


SUPPORTED_PROVIDERS: list[Provider] = ["vastai", "digitalocean", "lambda", "on_prem"]
_BACKGROUND_TASKS: set[asyncio.Task[None]] = set()

# Acceptable SSH public-key types and a base64 body. Reject anything with shell
# metacharacters, quotes, or newlines so the value can never break out of the
# snippet we hand back to the customer.
_SSH_KEY_RE = re.compile(
    r"^(ssh-(rsa|ed25519)|ecdsa-sha2-nistp(256|384|521)|sk-(ssh-ed25519|ecdsa-sha2-nistp256))"
    r" [A-Za-z0-9+/=]+( .*)?$"
)


def validate_ssh_public_key(key: str | None) -> str | None:
    """Validate a caller-supplied SSH public key or raise 400.

    Returns the key unchanged if valid, or None if the caller omitted it.
    Rejects keys with shell metacharacters, quotes, or newlines so the value is
    always safe to interpolate into the copy-paste snippet we return.
    """
    if key is None:
        return None
    key = key.strip()
    if not key:
        return None
    if "\n" in key or "\r" in key or "'" in key or '"' in key or "`" in key or "$" in key:
        raise HTTPException(
            status_code=400,
            detail="ssh_public_key must be a single line without quotes or shell metacharacters",
        )
    if not _SSH_KEY_RE.match(key):
        raise HTTPException(
            status_code=400,
            detail=(
                "ssh_public_key must be a single OpenSSH public key "
                "(e.g. 'ssh-ed25519 AAAA... comment')"
            ),
        )
    return key


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


def _render_onprem_snippet(ssh_public_key: str | None) -> str:
    """A docker-compose snippet the user runs locally.

    The SSH key is validated upstream (validate_ssh_public_key) and rendered via
    a quoted heredoc so that even a malformed value cannot break the command the
    user is told to paste.
    """
    key_block = ssh_public_key or "<paste your SSH public key>"
    return (
        "# 1. Drop this in a fresh directory on your machine\n"
        "mkdir nemoguardian-self-hosted && cd nemoguardian-self-hosted\n\n"
        "# 2. Save your SSH public key\n"
        "cat > authorized_keys <<'NEMOGUARDIAN_EOF'\n"
        f"{key_block}\n"
        "NEMOGUARDIAN_EOF\n\n"
        "# 3. Pull and run\n"
        "docker compose up -d\n\n"
        "# 4. Verify\n"
        "curl http://localhost:8000/health\n"
    )


__all__ = ["SUPPORTED_PROVIDERS", "Provider", "provision_instance"]
