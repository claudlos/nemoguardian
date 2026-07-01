"""Generic webhook adapter.

POST any text to /moderate (the nemoguardian server) and forward to a webhook.
Used for ad-hoc integration with anything that speaks HTTP.

Forwarding modes (``forward_text``), safest first:

* ``"verdict_only"`` (DEFAULT) — forward the moderation verdict (which already
  carries the score, categories and reasons) plus a SHA-256 hash of the original
  text, but NEVER the text itself. Nothing sensitive leaves the box.
* ``"redacted"`` — forward the text with PII (emails, SSNs, phone numbers,
  payment cards) redacted via the same helper the audit log uses.
* ``"full"`` — forward the original, unredacted text. Off by default; must be
  explicitly requested because it can leak PII / scam payloads downstream.

The mode is resolved from the explicit ``forward_text`` argument and falls back
to the ``NEMOGUARDIAN_WEBHOOK_FORWARD_TEXT`` environment variable, then to the
safe default. An unrecognised value degrades to ``"verdict_only"`` (fail-safe).
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from nemoguardian.adapters.base import ActionDecision, degrade_action
from nemoguardian.bot.audit import redact_text, text_hash
from nemoguardian.bot.types import ModerationAction, Platform

DEFAULT_FORWARD_MODE = "verdict_only"
FORWARD_MODES = ("verdict_only", "redacted", "full")

#: A webhook can only *forward* a verdict downstream — it cannot enforce on a
#: remote platform. So it supports ``allow``/``flag`` and ``notify_mods`` (the
#: forward itself), and any enforcement action degrades to ``flag``.
WEBHOOK_CAPABILITIES: frozenset[ModerationAction] = frozenset(
    {
        ModerationAction.ALLOW,
        ModerationAction.FLAG,
        ModerationAction.NOTIFY_MODS,
    }
)


def capabilities() -> set[ModerationAction]:
    """Return the normalized actions the webhook adapter can carry out."""
    return set(WEBHOOK_CAPABILITIES)


def resolve_forward_mode(forward_text: str | None) -> str:
    """Resolve the forwarding mode, degrading to the safe default.

    Resolution order: explicit argument → ``NEMOGUARDIAN_WEBHOOK_FORWARD_TEXT``
    env var → ``"verdict_only"``. Any unrecognised value is treated as the safe
    default so a typo never leaks raw text.
    """
    candidate = forward_text if forward_text is not None else os.environ.get(
        "NEMOGUARDIAN_WEBHOOK_FORWARD_TEXT"
    )
    if candidate is None:
        return DEFAULT_FORWARD_MODE
    normalized = candidate.strip().lower()
    return normalized if normalized in FORWARD_MODES else DEFAULT_FORWARD_MODE


def build_forward_payload(text: str, verdict: dict[str, Any], forward_text: str | None) -> dict[str, Any]:
    """Build the payload POSTed to the forward target for the given mode.

    The verdict dict already carries score/categories/reasons, so every mode
    forwards it alongside ``forward_text`` (the mode used) and ``text_sha256``
    (a stable fingerprint that lets a downstream consumer correlate or dedupe
    without ever seeing the content). Only ``redacted``/``full`` add a ``text``
    field.
    """
    mode = resolve_forward_mode(forward_text)
    payload: dict[str, Any] = {
        "verdict": verdict,
        "forward_text": mode,
        "text_sha256": text_hash(text),
    }
    if mode == "redacted":
        payload["text"] = redact_text(text)
    elif mode == "full":
        payload["text"] = text
    return payload


async def moderate_and_forward(
    text: str,
    *,
    forward_url: str,
    policy: str | None = None,
    mode: str = "standard",
    moderator_url: str = "http://localhost:8000",
    policy_preset: str | None = None,
    api_key: str | None = None,
    forward_text: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Call the nemoguardian server, then POST the result to `forward_url`.

    `api_key` defaults to `NEMOGUARDIAN_API_KEY` so the adapter works with the
    authenticated production endpoint as well as local self-hosted demos.

    `forward_text` selects what is sent to `forward_url` (see module docstring).
    It defaults to the safest mode (``verdict_only``): the original text never
    leaves the box unless ``"redacted"`` or ``"full"`` is explicitly requested.
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
            forward_text=forward_text,
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
    forward_payload = build_forward_payload(text, verdict, forward_text)
    async with httpx.AsyncClient(timeout=10.0) as client:
        await client.post(forward_url, json=forward_payload)
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
    forward_text: str | None,
) -> dict[str, Any]:
    resp = await client.post(
        f"{moderator_url}/v1/moderate",
        json=payload,
        params=params,
        headers=headers,
    )
    resp.raise_for_status()
    verdict = resp.json()
    forward_payload = build_forward_payload(text, verdict, forward_text)
    await client.post(forward_url, json=forward_payload)
    return verdict


def _auth_headers(api_key: str | None) -> dict[str, str]:
    resolved = api_key if api_key is not None else os.environ.get("NEMOGUARDIAN_API_KEY", "")
    resolved = resolved.strip()
    return {"Authorization": f"Bearer {resolved}"} if resolved else {}


class WebhookAdapter:
    """Thin :class:`~nemoguardian.adapters.base.PlatformAdapter` over the
    verdict-forwarding webhook flow.

    A webhook cannot enforce on a remote platform — it only forwards the verdict
    downstream — so :meth:`apply_action` degrades every enforcement action to
    ``flag`` with an auditable reason, and :meth:`record_audit` is a no-op (the
    forwarded verdict *is* the downstream record).
    """

    platform = Platform.WEBHOOK

    def __init__(
        self,
        *,
        forward_url: str | None = None,
        moderator_url: str = "http://localhost:8000",
        forward_text: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.forward_url = forward_url
        self.moderator_url = moderator_url
        self.forward_text = forward_text
        self.api_key = api_key

    def capabilities(self) -> set[ModerationAction]:
        return capabilities()

    def doctor(self) -> dict[str, object]:
        """Return a readiness snapshot for the forward target (never raises)."""
        ready = bool(self.forward_url)
        return {
            "platform": Platform.WEBHOOK.value,
            "forward_url_configured": ready,
            "forward_mode": resolve_forward_mode(self.forward_text),
            "readiness": "ready" if ready else "needs attention",
            "issues": [] if ready else ["forward_url is not set"],
        }

    def configure(
        self,
        *,
        forward_url: str | None = None,
        moderator_url: str | None = None,
        forward_text: str | None = None,
        api_key: str | None = None,
    ) -> WebhookAdapter:
        if forward_url is not None:
            self.forward_url = forward_url
        if moderator_url is not None:
            self.moderator_url = moderator_url
        if forward_text is not None:
            self.forward_text = forward_text
        if api_key is not None:
            self.api_key = api_key
        return self

    async def handle_event(
        self,
        text: str,
        *,
        forward_url: str | None = None,
        policy: str | None = None,
        mode: str = "standard",
        policy_preset: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> dict[str, Any]:
        """Moderate ``text`` and forward the verdict to the configured target."""
        target = forward_url or self.forward_url
        if not target:
            raise ValueError("forward_url is required (pass it or set it via configure())")
        return await moderate_and_forward(
            text,
            forward_url=target,
            policy=policy,
            mode=mode,
            moderator_url=self.moderator_url,
            policy_preset=policy_preset,
            api_key=self.api_key,
            forward_text=self.forward_text,
            client=client,
        )

    def apply_action(self, action: ModerationAction | str) -> ActionDecision:
        """Resolve ``action`` against webhook capabilities (enforcement -> flag)."""
        return degrade_action(action, self.capabilities(), Platform.WEBHOOK)

    def record_audit(self, *args: Any, **kwargs: Any) -> None:
        """No-op: the forwarded verdict is the webhook's downstream record."""
        return None


__all__ = [
    "DEFAULT_FORWARD_MODE",
    "FORWARD_MODES",
    "WEBHOOK_CAPABILITIES",
    "WebhookAdapter",
    "build_forward_payload",
    "capabilities",
    "moderate_and_forward",
    "resolve_forward_mode",
]
