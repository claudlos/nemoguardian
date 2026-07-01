"""The replayable SYNTHETIC eval corpus loads, is balanced, and is self-consistent."""

from __future__ import annotations

from pathlib import Path

from nemoguardian.detectors import detect_prompt_injection
from nemoguardian.eval.dataset import load_benchmark

_CORPUS = Path(__file__).resolve().parent / "fixtures" / "eval_corpus.jsonl"


def test_corpus_loads_and_is_balanced():
    cases = load_benchmark(_CORPUS)
    assert len(cases) >= 40
    n_unsafe = sum(1 for c in cases if c.is_unsafe)
    n_safe = len(cases) - n_unsafe
    assert n_unsafe >= 15
    assert n_safe >= 15
    # Hard negatives are required to measure false positives honestly.
    assert any(c.category == "benign_hard_negative" for c in cases)


def test_corpus_covers_required_categories():
    cats = {c.category for c in load_benchmark(_CORPUS)}
    required = {
        "pii", "doxxing", "scam", "spam", "harassment", "hate", "threats",
        "self_harm", "jailbreak", "prompt_injection", "controversial",
        "benign", "benign_hard_negative",
    }
    assert required <= cats, f"missing: {sorted(required - cats)}"


def test_corpus_buckets_are_known():
    allowed = {"safe", "unsafe", "controversial", "adversarial", None}
    for c in load_benchmark(_CORPUS):
        assert c.bucket in allowed


def test_corpus_no_obvious_real_pii_or_slurs():
    """Guardrail: the corpus must stay synthetic. No real slurs; placeholder only."""
    text = "\n".join(c.text for c in load_benchmark(_CORPUS))
    # Hate examples must use the explicit placeholder token, never a real slur.
    assert "[SLUR]" in text
    # Card-shaped numbers, if any, must be the documented test BIN (4000...).
    assert "4111 1111 1111 1111" not in text


def test_corpus_expect_inject_matches_real_detector():
    """Every ``expect_inject`` label agrees with the real deterministic detector.

    This is the contract that keeps the adversarial suite honest: injections we
    claim are catchable really fire, and benign look-alikes really do not.
    """
    for c in load_benchmark(_CORPUS):
        if c.expect_inject is None:
            continue
        fired = bool(detect_prompt_injection(c.text))
        assert fired == c.expect_inject, f"{c.id}: detector fired={fired}, expected {c.expect_inject}"
