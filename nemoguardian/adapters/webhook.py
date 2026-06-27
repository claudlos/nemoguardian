"""Generic webhook adapter.

POST any text to /moderate (the nemoguardian server) and forward to a webhook.
Used for ad-hoc integration with anything that speaks HTTP.
"""

from __future__ import annotations

from typing import Any

import httpx


async def moderate_and_forward(
    text: str,
    *,
    forward_url: str,
    policy: str | None = None,
    mode: str = "standard",
    moderator_url: str = "http://localhost:8000",
) -> dict[str, Any]:
    """Call the nemoguardian server, then POST the result to `forward_url`."""
    payload = {"text": text, "policy": policy, "mode": mode}
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(f"{moderator_url}/v1/moderate", json=payload)
        resp.raise_for_status()
        verdict = resp.json()
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(forward_url, json={"text": text, "verdict": verdict})
    return verdict


__all__ = ["moderate_and_forward"]
