"""ModerationModel failure contract — errors become inert verdicts, never raise."""

from __future__ import annotations

from nemoguardian.aggregator import aggregate
from nemoguardian.models.base import ModerationModel
from nemoguardian.schemas import VerdictLabel


class GoodModel(ModerationModel):
    model_id = "good"

    def _load(self) -> None:
        pass

    def _moderate_impl(self, text, *, policy=None, response=None):
        return {"verdict": "unsafe", "score": 0.9, "categories": ["Violent"], "reasoning": "bad"}


class ExplodingLoad(ModerationModel):
    model_id = "boom-load"

    def _load(self) -> None:
        raise RuntimeError("CUDA out of memory")

    def _moderate_impl(self, text, *, policy=None, response=None):
        raise AssertionError("unreachable")


class ExplodingInfer(ModerationModel):
    model_id = "boom-infer"

    def _load(self) -> None:
        pass

    def _moderate_impl(self, text, *, policy=None, response=None):
        raise ValueError("unparseable model output: 'garbage'")


def test_successful_model_returns_verdict_with_latency():
    out = GoodModel().moderate("text")
    assert out.verdict == VerdictLabel.UNSAFE
    assert out.score == 0.9
    assert out.error is None
    assert out.latency_ms >= 0


def test_load_failure_never_raises():
    out = ExplodingLoad().moderate("text")
    assert out.error is not None
    assert "CUDA out of memory" in out.error


def test_inference_failure_never_raises():
    out = ExplodingInfer().moderate("text")
    assert out.error is not None
    assert "unparseable" in out.error


def test_errored_verdict_is_excluded_not_a_safe_vote():
    """The safety-critical contract: a crashed model must not pass content."""
    errored = ExplodingInfer().moderate("text")
    good = GoodModel().moderate("text")

    # Errored + unsafe survivor → survivor decides (unsafe).
    agg = aggregate({"qwen3_guard_gen": errored, "nemotron_csr": good})
    assert agg.verdict == VerdictLabel.UNSAFE

    # Everything errored → fail-closed default kicks in.
    all_err = aggregate(
        {
            "qwen3_guard_gen": ExplodingInfer().moderate("t"),
            "nemotron_csr": ExplodingLoad().moderate("t"),
        }
    )
    assert all_err.verdict == VerdictLabel.UNSAFE


def test_lazy_load_called_once():
    class CountingModel(GoodModel):
        loads = 0

        def _load(self) -> None:
            type(self).loads += 1

    m = CountingModel()
    m.moderate("a")
    m.moderate("b")
    assert CountingModel.loads == 1
