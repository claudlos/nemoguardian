"""Cascade orchestration tests."""

from __future__ import annotations

from nemoguardian.cascade import Cascade, CascadeConfig
from nemoguardian.schemas import Mode, ModerateRequest, VerdictLabel


class ExplodingModel:
    def moderate(self, *args, **kwargs):
        raise AssertionError("model should not run")


class StaticStream:
    _loaded = True

    def stream_classify(self, text: str):
        yield "safe", 0.96, 0
        yield "unsafe", 0.91, 1


def test_fast_mode_uses_stream_summary_only():
    cascade = Cascade(CascadeConfig(enable_triage=False))
    cascade._qwen_stream = StaticStream()
    cascade._qwen_gen = ExplodingModel()
    cascade._csr = ExplodingModel()

    result = cascade.moderate(ModerateRequest(text="bad", mode=Mode.FAST))

    assert result.verdict == VerdictLabel.UNSAFE
    assert set(result.model_verdicts) == {"qwen3_guard_stream"}
    assert result.model_verdicts["qwen3_guard_stream"].score == 0.91
