"""Aggregator unit tests — fast, no model loading."""

from __future__ import annotations

from nemoguardian.aggregator import AggregatorConfig, aggregate
from nemoguardian.schemas import ModelVerdict, VerdictLabel


def _mv(model_id: str, verdict: str, score: float, **kw) -> ModelVerdict:
    return ModelVerdict(
        model_id=model_id,
        verdict=VerdictLabel(verdict),
        score=score,
        latency_ms=10.0,
        **kw,
    )


def test_aggregate_all_safe():
    out = aggregate(
        {
            "qwen3_guard_gen": _mv("q", "safe", 0.1),
            "nemotron_csr": _mv("c", "safe", 0.05),
        }
    )
    assert out.verdict == VerdictLabel.SAFE
    assert out.score < 0.30


def test_aggregate_all_unsafe():
    out = aggregate(
        {
            "qwen3_guard_gen": _mv("q", "unsafe", 0.95),
            "nemotron_csr": _mv("c", "unsafe", 0.90),
        }
    )
    assert out.verdict == VerdictLabel.UNSAFE
    assert out.score >= 0.70


def test_aggregate_controversial_split():
    out = aggregate(
        {
            "qwen3_guard_gen": _mv("q", "controversial", 0.5),
            "nemotron_csr": _mv("c", "safe", 0.1),
        }
    )
    # weighted: 0.4*0.5 + 0.4*0.1 = 0.24; below 0.30 → safe
    assert out.verdict == VerdictLabel.SAFE


def test_aggregate_fail_closed_on_single_hard_unsafe():
    out = aggregate(
        {
            "qwen3_guard_gen": _mv("q", "safe", 0.05),
            "nemotron_csr": _mv("c", "unsafe", 0.85),  # override threshold
        }
    )
    assert out.verdict == VerdictLabel.UNSAFE


def test_aggregate_skips_errored_models():
    out = aggregate(
        {
            "qwen3_guard_gen": _mv("q", "safe", 0.05, error="boom"),
            "nemotron_csr": _mv("c", "unsafe", 0.85),
        }
    )
    # qwen was skipped; csr alone still triggers override
    assert out.verdict == VerdictLabel.UNSAFE
    assert any("Qwen3Guard failed" in r for r in out.reasons)


def test_aggregate_categories_dedup():
    out = aggregate(
        {
            "qwen3_guard_gen": _mv("q", "unsafe", 0.9, categories=["PII", "Violent"]),
            "nemotron_csr": _mv("c", "unsafe", 0.9, categories=["PII", "Violent"]),
        }
    )
    assert out.categories.count("PII") == 1
    assert out.categories.count("Violent") == 1


def test_aggregate_with_triage_deep_mode():
    out = aggregate(
        {
            "qwen3_guard_gen": _mv("q", "safe", 0.1),
            "nemotron_csr": _mv("c", "unsafe", 0.9),
            "triage": _mv("t", "unsafe", 0.85),
        }
    )
    # Weighted: 0.4*0.1 + 0.4*0.9 + 0.2*0.85 = 0.53 → controversial
    # Plus override from csr@0.9 → unsafe
    assert out.verdict == VerdictLabel.UNSAFE
