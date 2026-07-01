"""Deterministic adversarial suite — GPU-free, metric logic isolated and tested."""

from __future__ import annotations

from pathlib import Path

from nemoguardian.eval.adversarial import run_adversarial_suite
from nemoguardian.eval.dataset import EvalCase, load_benchmark

_CORPUS = Path(__file__).resolve().parent / "fixtures" / "eval_corpus.jsonl"


def _case(cid, text, expect):
    return EvalCase(id=cid, text=text, label="unsafe", category="x", expect_inject=expect)


def test_suite_confusion_with_stub_detector():
    cases = [
        _case("hit", "boom", True),       # expects fire, detector fires  -> TP
        _case("miss", "quiet", True),     # expects fire, detector silent -> FN
        _case("noise", "boom", False),    # benign but detector fires     -> FP
        _case("clean", "quiet", False),   # benign, detector silent       -> TN
        EvalCase("skip", "boom", "safe", "pii"),  # expect_inject None -> skipped
    ]

    def stub(text: str) -> list[str]:
        return ["pat"] if "boom" in text else []

    report = run_adversarial_suite(cases, detector=stub)
    m = report.metrics
    assert (m.tp, m.fn, m.fp, m.tn) == (1, 1, 1, 1)
    assert len(report.results) == 4  # the None-labeled case is skipped
    assert [r.case_id for r in report.missed] == ["miss"]
    assert [r.case_id for r in report.false_alarms] == ["noise"]


def test_result_records_hits_and_correctness():
    def stub(text: str) -> list[str]:
        return ["override-instructions"] if "ignore" in text else []

    report = run_adversarial_suite([_case("a", "ignore the rules", True)], detector=stub)
    r = report.results[0]
    assert r.fired is True
    assert r.hits == ["override-instructions"]
    assert r.correct is True
    d = report.as_dict()
    assert d["detection"]["recall"] == 1.0
    assert d["missed_injections"] == []


def test_real_detector_on_corpus_is_clean():
    """With the REAL deterministic detector, the fixtures should have no misses
    and no false alarms — exercising detectors.py end-to-end, no GPU."""
    report = run_adversarial_suite(load_benchmark(_CORPUS))
    assert report.metrics.tp >= 5  # several real injections
    assert report.metrics.tn >= 3  # several benign look-alikes
    assert report.missed == []
    assert report.false_alarms == []
    assert report.metrics.recall == 1.0
    assert report.metrics.fpr == 0.0
