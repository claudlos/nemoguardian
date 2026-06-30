"""Metrics extensions: safe precision, FP/FN, latency + estimated cost."""

from __future__ import annotations

import pytest

from nemoguardian.eval.metrics import (
    DEFAULT_GPU_USD_PER_HR,
    LatencyCost,
    Metrics,
    cost_by_mode,
    summarize_latency_cost,
)


def test_safe_precision_and_aliases():
    # 8 unsafe caught, 2 missed; 18 safe cleared, 2 safe flagged.
    m = Metrics(tp=8, fp=2, tn=18, fn=2)
    assert m.unsafe_recall == m.recall == pytest.approx(0.8)
    assert m.false_positives == 2
    assert m.false_negatives == 2
    # safe precision (NPV) = tn / (tn + fn) = 18 / 20
    assert m.safe_precision == pytest.approx(0.9)
    d = m.as_dict()
    assert d["safe_precision"] == pytest.approx(0.9)
    assert d["false_negatives"] == 2
    assert d["unsafe_recall"] == pytest.approx(0.8)


def test_safe_precision_zero_division_is_safe():
    assert Metrics(tp=0, fp=0, tn=0, fn=0).safe_precision == 0.0


def test_summarize_latency_cost_math():
    # 4 cases, total 4000 ms = 4 s. Cost = 4/3600 h * rate.
    lc = summarize_latency_cost([1000, 1000, 1000, 1000], gpu_usd_per_hr=3.6)
    assert isinstance(lc, LatencyCost)
    assert lc.n == 4
    assert lc.mean_ms == pytest.approx(1000.0)
    assert lc.total_ms == pytest.approx(4000.0)
    # 4000 ms / 3_600_000 * 3.6 = 0.004
    assert lc.est_cost_usd == pytest.approx(0.004)
    d = lc.as_dict()
    assert d["est_cost_per_1k_usd"] == pytest.approx(0.004 / 4 * 1000)
    assert "ESTIMATE" in d["cost_note"]


def test_summarize_latency_cost_percentiles_and_filtering():
    # Zeros/negatives are dropped (cases that were never timed).
    lc = summarize_latency_cost([0, -5, 10, 20, 30, 40, 50])
    assert lc.n == 5
    assert lc.max_ms == 50.0
    assert lc.p50_ms == 30.0  # nearest-rank median of [10,20,30,40,50]
    assert lc.p95_ms == 50.0
    assert lc.gpu_usd_per_hr == DEFAULT_GPU_USD_PER_HR


def test_summarize_latency_cost_empty():
    lc = summarize_latency_cost([])
    assert lc.n == 0
    assert lc.mean_ms == 0.0
    assert lc.est_cost_usd == 0.0
    assert lc.as_dict()["est_cost_per_1k_usd"] == 0.0


def test_cost_by_mode_breakdown():
    by_mode = cost_by_mode(
        {"fast": [10, 10], "deep": [100, 100, 100]}, gpu_usd_per_hr=3.6
    )
    assert set(by_mode) == {"fast", "deep"}
    assert by_mode["fast"].n == 2
    assert by_mode["deep"].mean_ms == pytest.approx(100.0)
    # deep is pricier per request than fast.
    assert by_mode["deep"].est_cost_usd > by_mode["fast"].est_cost_usd
