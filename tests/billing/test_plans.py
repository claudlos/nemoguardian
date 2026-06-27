"""Plan/feature tests."""

from __future__ import annotations

import pytest

from nemoguardian.billing.plans import (
    FEATURE_CUSTOM_POLICY,
    FEATURE_DEEP,
    FEATURE_FAST,
    FEATURE_PRIORITY,
    FEATURE_SELF_HOSTED,
    FEATURE_STANDARD,
    FEATURE_STREAM,
    PLANS,
    Tier,
    get_plan,
    has_feature,
)


def test_free_plan_basics():
    p = get_plan(Tier.FREE)
    assert p.monthly_price_cents == 0
    assert p.monthly_allowance == 1_000
    assert p.max_api_keys == 1
    assert FEATURE_FAST in p.features
    assert FEATURE_STANDARD not in p.features
    assert FEATURE_DEEP not in p.features
    assert FEATURE_SELF_HOSTED not in p.features


def test_pro_unlocks_standard_and_stream():
    p = get_plan(Tier.PRO)
    assert p.monthly_price_cents == 1_900
    assert FEATURE_FAST in p.features
    assert FEATURE_STANDARD in p.features
    assert FEATURE_STREAM in p.features
    assert FEATURE_CUSTOM_POLICY in p.features
    assert FEATURE_DEEP not in p.features


def test_scale_unlocks_deep_and_priority():
    p = get_plan(Tier.SCALE)
    assert p.monthly_price_cents == 9_900
    assert FEATURE_DEEP in p.features
    assert FEATURE_PRIORITY in p.features
    assert FEATURE_CUSTOM_POLICY in p.features


def test_self_hosted_unlocks_all_features():
    p = get_plan(Tier.SELF_HOSTED)
    assert FEATURE_SELF_HOSTED in p.features
    assert FEATURE_DEEP in p.features
    assert FEATURE_PRIORITY in p.features


def test_has_feature_helper():
    assert has_feature(get_plan(Tier.PRO), FEATURE_FAST)
    assert not has_feature(get_plan(Tier.FREE), FEATURE_DEEP)


def test_overage_only_for_paid_tiers():
    for tier in (Tier.PRO, Tier.SCALE):
        assert get_plan(tier).overage_per_call_cents > 0
    assert get_plan(Tier.FREE).overage_per_call_cents == 0
    assert get_plan(Tier.SELF_HOSTED).overage_per_call_cents == 0


def test_all_tiers_have_unique_stripe_ids():
    assert len({p.tier for p in PLANS.values()}) == 4
