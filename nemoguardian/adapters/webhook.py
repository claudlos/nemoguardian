"""Generic webhook adapter.

POST any text to /moderate (the nemoguardian server) and forward to a webhook.
Used for ad-hoc integration with anything that speaks HTTP.
"""

from __future__ import annotations

import os
from typing import Any

import httpx


async def moderate_and_forward(
    text: str,
    *,
    forward_url: str,
    policy: str | None = None,
    mode: str = "standard",
    moderator_url: str = "http://localhost:8000",
    policy_preset: str | None = None,
    api_key: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Call the nemoguardian server, then POST the result to `forward_url`.

    `api_key` defaults to `NEMOGUARDIAN_API_KEY` so the adapter works with the
    authenticated production endpoint as well as local self-hosted demos.
    """
    payload = {"text": text, "policy": policy, "mode": mode}
    params = {"policy_preset": policy_preset} if policy_preset else None
    headers = _auth_headers(api_key)

    if client is not None:
        return await _moderate_and_forward_with_client(
            client,
            text=text,
            forward_url=forward_url,
            moderator_url=moderator_url,
            payload=payload,
            params=params,
            headers=headers,
        )

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{moderator_url}/v1/moderate",
            json=payload,
            params=params,
            headers=headers,
        )
        resp.raise_for_status()
        verdict = resp.json()
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(forward_url, json={"text": text, "verdict": verdict})
    return verdict


async def _moderate_and_forward_with_client(
    client: httpx.AsyncClient,
    *,
    text: str,
    forward_url: str,
    moderator_url: str,
    payload: dict[str, Any],
    params: dict[str, str] | None,
    headers: dict[str, str],
) -> dict[str, Any]:
    resp = await client.post(
        f"{moderator_url}/v1/moderate",
        json=payload,
        params=params,
        headers=headers,
    )
    resp.raise_for_status()
    verdict = resp.json()
    await client.post(forward_url, json={"text": text, "verdict": verdict})
    return verdict


def _auth_headers(api_key: str | None) -> dict[str, str]:
    resolved = api_key if api_key is not None else os.environ.get("NEMOGUARDIAN_API_KEY", "")
    resolved = resolved.strip()
    return {"Authorization": f"Bearer {resolved}"} if resolved else {}


__all__ = ["moderate_and_forward"]
