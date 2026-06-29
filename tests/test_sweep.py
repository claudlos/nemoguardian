"""Tests for the offline aggregator-threshold sweep (no GPU, no models)."""

from __future__ import annotations

import json

from nemoguardian.aggregator import AggregatorConfig
from scripts.sweep_thresholds import _evaluate, _fidelity_check, _verdicts_from_record

_FLAG_UNSAFE = frozenset({"unsafe"})


def _rec(case_id, category, gold, models, aggregate="safe"):
    return {"id": case_id, "category": category, "gold": gold, "aggregate": aggregate, "models": models}


def test_verdicts_from_record_roundtrip():
    rec = _rec("a", "pii", "unsafe", {"qwen3_guard_gen": {"verdict": "unsafe", "score": 1.0, "error": None}})
    v = _verdicts_from_record(rec)
    assert v["qwen3_guard_gen"].verdict.value == "unsafe"
    assert v["qwen3_guard_gen"].score == 1.0


def test_lowering_override_recovers_subthreshold_unsafe():
    # qwen votes unsafe at 0.4 — below the default override (0.5), so default misses it.
    records = [
        _rec("u1", "illegal", "unsafe", {"qwen3_guard_gen": {"verdict": "unsafe", "score": 0.4, "error": None},
                                          "nemotron_csr": {"verdict": "safe", "score": 0.1, "error": None}}),
    ]
    default = _evaluate(records, AggregatorConfig(), _FLAG_UNSAFE)
    assert default.overall.recall == 0.0  # missed at override 0.5
    lowered = _evaluate(records, AggregatorConfig(override_on_unsafe=0.4), _FLAG_UNSAFE)
    assert lowered.overall.recall == 1.0  # recovered at override 0.4


def test_aggressive_config_can_raise_fpr():
    # A benign case where a model emitted a borderline unsafe score.
    records = [
        _rec("b1", "benign", "safe", {"qwen3_guard_gen": {"verdict": "unsafe", "score": 0.45, "error": None},
                                       "nemotron_csr": {"verdict": "safe", "score": 0.0, "error": None}}),
    ]
    assert _evaluate(records, AggregatorConfig(), _FLAG_UNSAFE).overall.fpr == 0.0
    aggressive = _evaluate(records, AggregatorConfig(override_on_unsafe=0.4), _FLAG_UNSAFE)
    assert aggressive.overall.fpr == 1.0  # the tradeoff: recovering recall can cost FPR


def test_fidelity_check_passes_when_aggregate_matches_default():
    # captured aggregate computed with the default config -> zero mismatches
    records = [
        _rec("x", "pii", "unsafe",
             {"nemotron_csr": {"verdict": "unsafe", "score": 0.9, "error": None}}, aggregate="unsafe"),
        _rec("y", "benign", "safe",
             {"qwen3_guard_gen": {"verdict": "safe", "score": 0.05, "error": None}}, aggregate="safe"),
    ]
    assert _fidelity_check(records) == 0


def test_sweep_reads_jsonl_dump(tmp_path):
    dump = tmp_path / "d.jsonl"
    recs = [
        _rec("u", "scam", "unsafe", {"nemotron_csr": {"verdict": "unsafe", "score": 0.9, "error": None}}, "unsafe"),
        _rec("b", "benign", "safe", {"qwen3_guard_gen": {"verdict": "safe", "score": 0.0, "error": None}}, "safe"),
    ]
    dump.write_text("\n".join(json.dumps(r) for r in recs))
    loaded = [json.loads(line) for line in dump.read_text().splitlines()]
    rep = _evaluate(loaded, AggregatorConfig(), _FLAG_UNSAFE)
    assert rep.overall.tp == 1 and rep.overall.tn == 1 and rep.overall.fpr == 0.0
