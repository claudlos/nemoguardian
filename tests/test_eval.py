"""Eval harness tests — pure metric/runner logic, no model loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nemoguardian.eval.dataset import EvalCase, load_benchmark
from nemoguardian.eval.metrics import Metrics, score_predictions
from nemoguardian.eval.runner import PredictResult, mean_latency_ms, run_eval

_REPO_BENCHMARK = Path(__file__).resolve().parent.parent / "data" / "benchmark.jsonl"


def test_metrics_basic_confusion():
    m = Metrics(tp=8, fp=2, tn=18, fn=2)
    assert m.n == 30
    assert m.precision == pytest.approx(0.8)
    assert m.recall == pytest.approx(0.8)
    assert m.fpr == pytest.approx(0.1)
    assert m.f1 == pytest.approx(0.8)
    assert m.accuracy == pytest.approx(26 / 30)


def test_metrics_zero_division_is_safe():
    m = Metrics(tp=0, fp=0, tn=0, fn=0)
    assert m.precision == 0.0
    assert m.recall == 0.0
    assert m.fpr == 0.0
    assert m.f1 == 0.0


def test_score_predictions_overall_and_per_category():
    rows = [
        (True, True, "pii"),     # TP
        (True, False, "scam"),   # FN  (missed attack)
        (False, True, "benign"), # FP  (benign flagged)
        (False, False, "benign"),# TN
    ]
    report = score_predictions(rows)
    assert report.overall.tp == 1
    assert report.overall.fn == 1
    assert report.overall.fp == 1
    assert report.overall.tn == 1
    # benign category: 2 safe, FPR = 1/2
    benign = report.by_category["benign"]
    assert benign.n_safe == 2
    assert benign.metrics.fpr == pytest.approx(0.5)
    assert benign.as_dict()["recall"] is None  # no unsafe rows → recall N/A
    # scam category: 1 unsafe, recall = 0
    scam = report.by_category["scam"]
    assert scam.metrics.recall == 0.0
    assert scam.as_dict()["fpr"] is None


def test_run_eval_flag_on_unsafe_only():
    cases = [
        EvalCase("a", "x", "unsafe", "scam"),
        EvalCase("b", "y", "safe", "benign"),
        EvalCase("c", "z", "unsafe", "pii"),
    ]
    verdicts = {"a": "unsafe", "b": "controversial", "c": "controversial"}

    def predict(case: EvalCase) -> PredictResult:
        return PredictResult(verdict=verdicts[case.id], latency_ms=5.0)

    # Only "unsafe" flags: a flagged (TP), c not flagged (FN), b not flagged (TN).
    report, preds = run_eval(cases, predict, flag_on={"unsafe"})
    assert report.overall.tp == 1
    assert report.overall.fn == 1
    assert report.overall.tn == 1
    assert report.overall.fp == 0
    assert mean_latency_ms(preds) == pytest.approx(5.0)


def test_run_eval_flag_on_includes_controversial():
    cases = [EvalCase("c", "z", "unsafe", "pii")]

    def predict(_case: EvalCase) -> PredictResult:
        return PredictResult(verdict="controversial")

    report, _ = run_eval(cases, predict, flag_on={"unsafe", "controversial"})
    assert report.overall.tp == 1  # controversial now counts as a catch


def test_load_repo_benchmark_is_valid_and_balanced():
    cases = load_benchmark(_REPO_BENCHMARK)
    assert len(cases) >= 40
    n_unsafe = sum(1 for c in cases if c.is_unsafe)
    n_safe = len(cases) - n_unsafe
    # Needs a real benign set to measure FPR, including hard negatives.
    assert n_safe >= 15
    assert n_unsafe >= 15
    assert any(c.category == "benign_hard_negative" for c in cases)


def test_load_benchmark_rejects_bad_label(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"id": "1", "text": "x", "label": "maybe", "category": "pii"}))
    with pytest.raises(ValueError, match="label must be one of"):
        load_benchmark(p)


def test_load_benchmark_rejects_duplicate_ids(tmp_path):
    p = tmp_path / "dup.jsonl"
    row = {"id": "1", "text": "x", "label": "safe", "category": "benign"}
    p.write_text(json.dumps(row) + "\n" + json.dumps(row))
    with pytest.raises(ValueError, match="duplicate case id"):
        load_benchmark(p)
