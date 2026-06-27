"""Subscription plans / pricing tiers.

Four tiers, ranging from free hobbyist to self-hosted enterprise.

Each plan declares:
- monthly allowance (free calls included)
- per-call overage rate
- features enabled (which cascade models they unlock)
- max API keys per account
- monthly price (USD cents)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Tier(str, Enum):
    FREE = "free"
    PRO = "pro"
    SCALE = "scale"
    SELF_HOSTED = "self_hosted"


@dataclass(frozen=True)
class Plan:
    tier: Tier
    display_name: str
    monthly_price_cents: int
    monthly_allowance: int          # free calls per month
    overage_per_call_cents: float   # what we charge past the allowance, per call
    features: frozenset[str]
    max_api_keys: int
    cascade_mode_default: str       # "fast" | "standard" | "deep"
    description: str


# Stable IDs — match what we'd create as Stripe Products/Prices in the dashboard.
STRIPE_PRODUCT_IDS = {
    Tier.FREE: "prod_nemoguardian_free",
    Tier.PRO: "prod_nemoguardian_pro",
    Tier.SCALE: "prod_nemoguardian_scale",
    Tier.SELF_HOSTED: "prod_nemoguardian_self_hosted",
}

# In a real deploy these come from `stripe prices create`. Hard-coded for the demo.
STRIPE_PRICE_IDS = {
    Tier.PRO: "price_nemoguardian_pro_monthly",
    Tier.SCALE: "price_nemoguardian_scale_monthly",
    Tier.SELF_HOSTED: "price_nemoguardian_self_hosted_setup",
}

FEATURE_FAST = "cascade.fast"
FEATURE_STANDARD = "cascade.standard"
FEATURE_DEEP = "cascade.deep"
FEATURE_STREAM = "cascade.stream"
FEATURE_CUSTOM_POLICY = "policy.custom"
FEATURE_PRIORITY = "priority.latency"
FEATURE_SELF_HOSTED = "deploy.self_hosted"


PLANS: dict[Tier, Plan] = {
    Tier.FREE: Plan(
        tier=Tier.FREE,
        display_name="Free",
        monthly_price_cents=0,
        monthly_allowance=1_000,
        overage_per_call_cents=0,  # hard cap, no overage
        features=frozenset({FEATURE_FAST}),
        max_api_keys=1,
        cascade_mode_default="fast",
        description="1,000 calls/mo, Qwen3Guard-Stream 0.6B only. Good for a small Discord server.",
    ),
    Tier.PRO: Plan(
        tier=Tier.PRO,
        display_name="Pro",
        monthly_price_cents=1_900,           # $19/mo
        monthly_allowance=50_000,
        overage_per_call_cents=0.10,         # $0.001 / call
        features=frozenset({FEATURE_FAST, FEATURE_STANDARD, FEATURE_STREAM, FEATURE_CUSTOM_POLICY}),
        max_api_keys=5,
        cascade_mode_default="standard",
        description="50,000 calls/mo, full standard cascade + custom policy. For active Discord/Twitch channels.",
    ),
    Tier.SCALE: Plan(
        tier=Tier.SCALE,
        display_name="Scale",
        monthly_price_cents=9_900,           # $99/mo
        monthly_allowance=500_000,
        overage_per_call_cents=0.08,         # volume discount
        features=frozenset({FEATURE_FAST, FEATURE_STANDARD, FEATURE_DEEP, FEATURE_STREAM,
                             FEATURE_CUSTOM_POLICY, FEATURE_PRIORITY}),
        max_api_keys=20,
        cascade_mode_default="deep",
        description="500,000 calls/mo + DEEP mode (Nemotron 3 Ultra triage) + priority latency. For brands / enterprise.",
    ),
    Tier.SELF_HOSTED: Plan(
        tier=Tier.SELF_HOSTED,
        display_name="Self-hosted",
        monthly_price_cents=0,               # setup fee + $0/mo license in this model
        monthly_allowance=10_000_000,        # effectively unlimited; for billing safety
        overage_per_call_cents=0,
        features=frozenset({FEATURE_FAST, FEATURE_STANDARD, FEATURE_DEEP, FEATURE_STREAM,
                             FEATURE_CUSTOM_POLICY, FEATURE_PRIORITY, FEATURE_SELF_HOSTED}),
        max_api_keys=100,
        cascade_mode_default="deep",
        description="We deploy nemoguardian on your infra (cloud or on-prem). $499 one-time setup.",
    ),
}


def get_plan(tier: Tier) -> Plan:
    return PLANS[tier]


def has_feature(plan: Plan, feature: str) -> bool:
    return feature in plan.features


__all__ = [
    "FEATURE_CUSTOM_POLICY",
    "FEATURE_DEEP",
    "FEATURE_FAST",
    "FEATURE_PRIORITY",
    "FEATURE_SELF_HOSTED",
    "FEATURE_STANDARD",
    "FEATURE_STREAM",
    "PLANS",
    "STRIPE_PRICE_IDS",
    "STRIPE_PRODUCT_IDS",
    "Plan",
    "Tier",
    "get_plan",
    "has_feature",
]
