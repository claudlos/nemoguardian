"""API-key authentication dependency for FastAPI.

Wraps the moderation endpoint: an API key in the `Authorization: Bearer nmg_xxx`
header resolves to a Customer, which the handler uses to enforce tier limits.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from nemoguardian.billing import db
from nemoguardian.billing.plans import Plan, Tier, get_plan

# Persist last_used_at at most once per key per this interval — it's a write
# behind a global SQLite lock on every request otherwise.
_TOUCH_INTERVAL = timedelta(seconds=60)

_PLACEHOLDER_ENV_KEYS = {
    "",
    "nmg_change_me",
    "nmg_default_change_me",
    "nmg_paste_your_key_here",
    "nmg_replace_with_demo_key",
}


@dataclass
class AuthContext:
    """Resolved caller identity + tier metadata."""

    customer: db.Customer
    plan: Plan
    raw_key: str
    key_id: int | None = None  # None for the env-bootstrap key
    scope: str = "full"  # 'full' | 'instance' (moderation-only)


async def require_api_key(
    authorization: Annotated[str | None, Header()] = None,
) -> AuthContext:
    """FastAPI dependency. Raises 401 if no key, 403 if revoked."""
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing Authorization header (expected: Bearer <api-key>)",
        )
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid Authorization header (expected: Bearer <api-key>)",
        )
    raw_key = parts[1].strip()
    if not raw_key.startswith("nmg_"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key must start with 'nmg_'",
        )
    # Sync SQLite lookups run off the event loop.
    found = await asyncio.to_thread(db.lookup_api_key, raw_key)
    if found is not None:
        customer, key = found
        if _touch_due(key.last_used_at):
            await asyncio.to_thread(db.touch_api_key, key.id)
        return AuthContext(
            customer=customer,
            plan=get_plan(customer.tier_enum),
            raw_key=raw_key,
            key_id=key.id,
            scope=key.scope,
        )

    customer = await asyncio.to_thread(_lookup_env_bootstrap_customer, raw_key)
    if customer is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or revoked API key",
        )
    return AuthContext(customer=customer, plan=get_plan(customer.tier_enum), raw_key=raw_key)


def _touch_due(last_used_at: str | None) -> bool:
    if not last_used_at:
        return True
    try:
        last = datetime.fromisoformat(last_used_at)
    except ValueError:
        return True
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - last >= _TOUCH_INTERVAL


def require_full_scope(
    auth: Annotated[AuthContext, Depends(require_api_key)],
) -> AuthContext:
    """Dependency for billing/key/provisioning routes (NMG-009).

    Instance-scoped keys (minted for provisioned VMs) can moderate but must not
    manage keys, billing, or spin up further instances.
    """
    if auth.scope != "full":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="this API key is instance-scoped (moderation only)",
        )
    return auth


def enforce_feature(auth: AuthContext, feature: str) -> None:
    """Raise 402 if the caller's plan doesn't include the feature."""
    if feature not in auth.plan.features:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=(
                f"feature {feature!r} not available on tier {auth.plan.tier.value!r}; "
                f"upgrade at /billing/checkout?plan={_upgrade_target(feature).value}"
            ),
        )


def _upgrade_target(feature: str) -> Tier:
    """Cheapest tier that includes the feature."""
    from nemoguardian.billing.plans import PLANS, Tier

    # Order matters — start from cheapest.
    for tier in (Tier.FREE, Tier.PRO, Tier.SCALE, Tier.SELF_HOSTED):
        if feature in PLANS[tier].features:
            return tier
    return Tier.SCALE  # fallback


def _lookup_env_bootstrap_customer(raw_key: str) -> db.Customer | None:
    """Resolve the self-hosted env API key without requiring a pre-seeded DB row."""
    env_key = os.environ.get("NEMOGUARDIAN_API_KEY", "").strip()
    if env_key in _PLACEHOLDER_ENV_KEYS or raw_key != env_key:
        return None

    tier = _env_tier()
    email = os.environ.get("NEMOGUARDIAN_SELF_HOSTED_EMAIL", "self-hosted@nemoguardian.local")
    customer = db.upsert_customer(email=email)
    if customer.tier_enum != tier:
        db.set_customer_tier(customer.id, tier)
        customer = db.get_customer(customer.id)
    return customer


def _env_tier() -> Tier:
    raw = os.environ.get("NEMOGUARDIAN_TIER", Tier.SELF_HOSTED.value)
    try:
        return Tier(raw)
    except ValueError:
        return Tier.SELF_HOSTED


__all__ = ["AuthContext", "enforce_feature", "require_api_key", "require_full_scope"]
