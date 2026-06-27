"""Schema validation tests."""

from __future__ import annotations

from nemoguardian.schemas import (
    Mode,
    ModelVerdict,
    ModerateRequest,
    VerdictLabel,
)


def test_moderate_request_minimal():
    req = ModerateRequest(text="hello world")
    assert req.mode == Mode.STANDARD
    assert req.policy is None


def test_moderate_request_rejects_empty_text():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ModerateRequest(text="")


def test_model_verdict_score_validation():
    """Score outside [0, 1] must raise."""
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ModelVerdict(model_id="x", verdict=VerdictLabel.SAFE, score=1.5, latency_ms=0)
    with pytest.raises(ValidationError):
        ModelVerdict(model_id="x", verdict=VerdictLabel.SAFE, score=-0.1, latency_ms=0)
    # Boundary values are accepted.
    assert ModelVerdict(model_id="x", verdict=VerdictLabel.SAFE, score=0.0, latency_ms=0).score == 0.0
    assert ModelVerdict(model_id="x", verdict=VerdictLabel.SAFE, score=1.0, latency_ms=0).score == 1.0


def test_moderate_request_modes():
    for m in [Mode.FAST, Mode.STANDARD, Mode.DEEP]:
        req = ModerateRequest(text="x", mode=m)
        assert req.mode == m
