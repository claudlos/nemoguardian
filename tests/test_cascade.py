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


from nemoguardian.schemas import ModelVerdict  # noqa: E402


class StaticModel:
    """Fake ModerationModel returning a canned verdict."""

    def __init__(self, model_id: str, label: VerdictLabel, score: float) -> None:
        self.model_id = model_id
        self.label = label
        self.score = score
        self.calls: list[dict] = []

    def moderate(self, text, *, policy=None, response=None):
        self.calls.append({"text": text, "policy": policy})
        return ModelVerdict(
            model_id=self.model_id, verdict=self.label, score=self.score, latency_ms=1.0
        )


class StaticTriage:
    def __init__(self, label: VerdictLabel = VerdictLabel.CONTROVERSIAL, score: float = 0.5):
        self.label = label
        self.score = score
        self.calls: list[dict] = []

    def adjudicate(self, *, text, policy, qwen_verdict, csr_verdict):
        self.calls.append(
            {"text": text, "qwen": qwen_verdict, "csr": csr_verdict, "policy": policy}
        )
        return ModelVerdict(
            model_id="triage", verdict=self.label, score=self.score, latency_ms=1.0
        )


def test_standard_mode_runs_gen_and_csr_not_triage():
    cascade = Cascade(CascadeConfig(enable_triage=True))
    qwen = StaticModel("q", VerdictLabel.SAFE, 0.0)
    csr = StaticModel("c", VerdictLabel.SAFE, 0.0)
    triage = StaticTriage()
    cascade._qwen_gen = qwen
    cascade._csr = csr
    cascade._triage = triage
    cascade._qwen_stream = ExplodingModel()

    result = cascade.moderate(
        ModerateRequest(text="hello", mode=Mode.STANDARD, policy="no memes")
    )

    assert set(result.model_verdicts) == {"qwen3_guard_gen", "nemotron_csr"}
    assert result.verdict == VerdictLabel.SAFE
    assert qwen.calls[0]["policy"] == "no memes"
    assert csr.calls[0]["policy"] == "no memes"
    assert triage.calls == []  # triage is deep-mode only


def test_deep_mode_adjudicates_with_triage():
    cascade = Cascade(CascadeConfig(enable_triage=True))
    qwen = StaticModel("q", VerdictLabel.SAFE, 0.1)
    csr = StaticModel("c", VerdictLabel.UNSAFE, 0.9)
    triage = StaticTriage(VerdictLabel.UNSAFE, 0.8)
    cascade._qwen_gen = qwen
    cascade._csr = csr
    cascade._triage = triage

    result = cascade.moderate(ModerateRequest(text="edge case", mode=Mode.DEEP))

    assert set(result.model_verdicts) == {"qwen3_guard_gen", "nemotron_csr", "triage"}
    # Triage saw both classifier verdicts.
    assert triage.calls[0]["qwen"].model_id == "q"
    assert triage.calls[0]["csr"].model_id == "c"
    assert result.verdict == VerdictLabel.UNSAFE


def test_deep_mode_skips_triage_when_disabled():
    cascade = Cascade(CascadeConfig(enable_triage=False))
    cascade._qwen_gen = StaticModel("q", VerdictLabel.SAFE, 0.0)
    cascade._csr = StaticModel("c", VerdictLabel.SAFE, 0.0)

    result = cascade.moderate(ModerateRequest(text="x", mode=Mode.DEEP))

    assert "triage" not in result.model_verdicts


def test_stream_document_verdict_maps_risk_bands():
    cascade = Cascade(CascadeConfig(enable_triage=False))

    class ControversialStream:
        _loaded = True

        def stream_classify(self, text):
            yield "safe", 0.99, 0
            yield "controversial", 0.85, 1

    cascade._qwen_stream = ControversialStream()
    verdict = cascade._stream_document_verdict("text")
    assert verdict.verdict == VerdictLabel.CONTROVERSIAL
    # Controversial risk is clamped into (0.3, 0.69].
    assert 0.3 <= verdict.score <= 0.69

    class AllSafeStream:
        _loaded = True

        def stream_classify(self, text):
            yield "safe", 0.99, 0
            yield "safe", 0.97, 1

    cascade._qwen_stream = AllSafeStream()
    verdict = cascade._stream_document_verdict("text")
    assert verdict.verdict == VerdictLabel.SAFE
    assert verdict.score <= 0.29
