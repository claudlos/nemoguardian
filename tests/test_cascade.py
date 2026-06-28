"""Cascade orchestration tests."""

from __future__ import annotations

import nemoguardian.cascade as cascade_module
from nemoguardian.cascade import Cascade, CascadeConfig
from nemoguardian.policy.nemoclaw import PolicyDecision
from nemoguardian.schemas import Mode, ModelVerdict, ModerateRequest, VerdictLabel


class ExplodingModel:
    def moderate(self, *args, **kwargs):
        raise AssertionError("model should not run")


class StaticStream:
    _loaded = True

    def stream_classify(self, text: str):
        yield "safe", 0.96, 0
        yield "unsafe", 0.91, 1


class StaticModel:
    is_loaded = True

    def __init__(self, verdict: VerdictLabel, *, score: float, categories: list[str] | None = None):
        self.verdict = verdict
        self.score = score
        self.categories = categories or []
        self.calls: list[dict] = []

    def moderate(self, text: str, *, policy: str | None = None, response: str | None = None):
        self.calls.append({"text": text, "policy": policy, "response": response})
        return ModelVerdict(
            model_id=f"static-{self.verdict.value}",
            verdict=self.verdict,
            score=self.score,
            categories=self.categories,
            reasoning="static",
            latency_ms=1.0,
        )


class StaticTriage:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def adjudicate(self, *, text: str, policy: str | None, qwen_verdict, csr_verdict):
        self.calls.append({
            "text": text,
            "policy": policy,
            "qwen_verdict": qwen_verdict.verdict,
            "csr_verdict": csr_verdict.verdict,
        })
        return ModelVerdict(
            model_id="triage",
            verdict=VerdictLabel.CONTROVERSIAL,
            score=0.55,
            categories=["triage reason"],
            reasoning="triage reason",
            latency_ms=2.0,
        )


class OverridePolicy:
    def evaluate(self, *, verdict, score, categories, policy_text=None):
        return PolicyDecision(
            matched_rule="force-test-block",
            final_label=VerdictLabel.UNSAFE,
            final_score=0.99,
        )


class PassivePolicy:
    def evaluate(self, *, verdict, score, categories, policy_text=None):
        return PolicyDecision(matched_rule="observed-only")


def test_fast_mode_uses_stream_summary_only():
    cascade = Cascade(CascadeConfig(enable_triage=False))
    cascade._qwen_stream = StaticStream()
    cascade._qwen_gen = ExplodingModel()
    cascade._csr = ExplodingModel()

    result = cascade.moderate(ModerateRequest(text="bad", mode=Mode.FAST))

    assert result.verdict == VerdictLabel.UNSAFE
    assert set(result.model_verdicts) == {"qwen3_guard_stream"}
    assert result.model_verdicts["qwen3_guard_stream"].score == 0.91


def test_lazy_properties_build_models_once_and_report_loaded(monkeypatch):
    calls: dict[str, list[dict]] = {"qwen": [], "stream": [], "csr": [], "triage": []}

    class FakeQwen:
        is_loaded = True

        def __init__(self, *, model_name: str, load_in_4bit: bool) -> None:
            calls["qwen"].append({"model_name": model_name, "load_in_4bit": load_in_4bit})

    class FakeStream:
        _loaded = True

        def __init__(self, *, model_name: str) -> None:
            calls["stream"].append({"model_name": model_name})

    class FakeCSR:
        is_loaded = False

        def __init__(self, *, model_name: str, reasoning: bool, load_in_4bit: bool) -> None:
            calls["csr"].append({
                "model_name": model_name,
                "reasoning": reasoning,
                "load_in_4bit": load_in_4bit,
            })

    class FakeTriage:
        def __init__(self, *, base_url: str | None, model_name: str) -> None:
            calls["triage"].append({"base_url": base_url, "model_name": model_name})

    monkeypatch.setattr(cascade_module, "Qwen3GuardGen", FakeQwen)
    monkeypatch.setattr(cascade_module, "Qwen3GuardStream", FakeStream)
    monkeypatch.setattr(cascade_module, "NemotronCSR", FakeCSR)
    monkeypatch.setattr(cascade_module, "NemotronTriage", FakeTriage)

    cascade = Cascade(CascadeConfig(
        qwen_gen_model="qwen-test",
        qwen_stream_model="stream-test",
        csr_model="csr-test",
        triage_model="triage-test",
        triage_base_url="https://triage.test/v1",
        qwen_gen_4bit=False,
        csr_4bit=False,
        reasoning=False,
        enable_triage=True,
    ))

    assert cascade.qwen_gen is cascade.qwen_gen
    assert cascade.qwen_stream is cascade.qwen_stream
    assert cascade.csr is cascade.csr
    assert cascade.triage is cascade.triage
    assert calls == {
        "qwen": [{"model_name": "qwen-test", "load_in_4bit": False}],
        "stream": [{"model_name": "stream-test"}],
        "csr": [{"model_name": "csr-test", "reasoning": False, "load_in_4bit": False}],
        "triage": [{"base_url": "https://triage.test/v1", "model_name": "triage-test"}],
    }
    assert cascade.loaded_models() == {
        "qwen3_guard_gen": True,
        "qwen3_guard_stream": True,
        "nemotron_csr": False,
    }
    assert cascade.model_config_summary()["triage_base_url"] == "https://triage.test/v1"


def test_cascade_config_from_env_and_bool_defaults(monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_QWEN_MODEL", "qwen-env")
    monkeypatch.setenv("NEMOGUARDIAN_QWEN_STREAM_MODEL", "stream-env")
    monkeypatch.setenv("NEMOGUARDIAN_CSR_MODEL", "csr-env")
    monkeypatch.setenv("NEMOGUARDIAN_TRIAGE_MODEL", "triage-env")
    monkeypatch.setenv("NVIDIA_BASE_URL", "https://nim-env.test/v1")
    monkeypatch.setenv("NEMOGUARDIAN_QUANTIZE", "maybe")
    monkeypatch.setenv("NEMOGUARDIAN_QWEN_4BIT", "off")
    monkeypatch.setenv("NEMOGUARDIAN_CSR_4BIT", "on")
    monkeypatch.setenv("NEMOGUARDIAN_REASONING", "no")
    monkeypatch.setenv("NEMOGUARDIAN_ENABLE_TRIAGE", "false")

    config = CascadeConfig.from_env()

    assert config.qwen_gen_model == "qwen-env"
    assert config.qwen_stream_model == "stream-env"
    assert config.csr_model == "csr-env"
    assert config.triage_model == "triage-env"
    assert config.triage_base_url == "https://nim-env.test/v1"
    assert config.qwen_gen_4bit is False
    assert config.csr_4bit is True
    assert config.reasoning is False
    assert config.enable_triage is False
    assert cascade_module._env_bool("DOES_NOT_EXIST", True) is True
    monkeypatch.setenv("NEMOGUARDIAN_UNKNOWN_BOOL", "not-a-bool")
    assert cascade_module._env_bool("NEMOGUARDIAN_UNKNOWN_BOOL", False) is False


def test_triage_disabled_property_and_status(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cascade = Cascade(CascadeConfig(enable_triage=False))

    assert cascade.triage is None
    assert cascade.triage_status() == {
        "configured": False,
        "provider": None,
        "model": cascade.config.triage_model,
        "base_url": "https://integrate.api.nvidia.com/v1",
    }


def test_triage_status_prefers_nvidia_and_default_openrouter_url(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvidia-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test")
    cascade = Cascade(CascadeConfig(enable_triage=True))

    assert cascade.triage_status()["provider"] == "nvidia"

    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    assert cascade.triage_status()["provider"] == "openrouter"
    summary = Cascade(CascadeConfig(triage_base_url=None)).model_config_summary()
    assert summary["triage_base_url"] == "https://openrouter.ai/api/v1"


def test_standard_mode_runs_qwen_and_csr_then_policy_override():
    cascade = Cascade(CascadeConfig(enable_triage=False))
    cascade._qwen_gen = StaticModel(VerdictLabel.SAFE, score=0.05)
    cascade._csr = StaticModel(VerdictLabel.CONTROVERSIAL, score=0.45, categories=["PII"])

    result = cascade.moderate(
        ModerateRequest(text="check this", policy="block PII", mode=Mode.STANDARD),
        policy_engine=OverridePolicy(),
    )

    assert result.verdict == VerdictLabel.UNSAFE
    assert result.score == 0.99
    assert result.matched_policy_rule == "force-test-block"
    assert set(result.model_verdicts) == {"qwen3_guard_gen", "nemotron_csr"}
    assert cascade._qwen_gen.calls == [{"text": "check this", "policy": "block PII", "response": None}]
    assert cascade._csr.calls == [{"text": "check this", "policy": "block PII", "response": None}]
    assert result.request_id
    assert result.timestamp


def test_policy_match_without_override_preserves_aggregate_verdict():
    cascade = Cascade(CascadeConfig(enable_triage=False))
    cascade._qwen_gen = StaticModel(VerdictLabel.SAFE, score=0.01)
    cascade._csr = StaticModel(VerdictLabel.SAFE, score=0.02)

    result = cascade.moderate(
        ModerateRequest(text="safe", mode=Mode.STANDARD),
        policy_engine=PassivePolicy(),
    )

    assert result.verdict == VerdictLabel.SAFE
    assert result.matched_policy_rule == "observed-only"


def test_standard_mode_respects_model_toggles():
    cascade = Cascade(CascadeConfig(enable_triage=False))
    cascade._qwen_gen = ExplodingModel()
    cascade._csr = ExplodingModel()

    result = cascade.moderate(
        ModerateRequest(
            text="skip all",
            mode=Mode.STANDARD,
            use_qwen_gen=False,
            use_nemotron_csr=False,
        )
    )

    assert result.verdict == VerdictLabel.SAFE
    assert result.model_verdicts == {}


def test_deep_mode_runs_triage_when_models_available():
    cascade = Cascade(CascadeConfig(enable_triage=True))
    cascade._qwen_gen = StaticModel(VerdictLabel.SAFE, score=0.05)
    cascade._csr = StaticModel(VerdictLabel.UNSAFE, score=0.95, categories=["PII"])
    cascade._triage = StaticTriage()

    result = cascade.moderate(
        ModerateRequest(text="disagreement", policy="block PII", mode=Mode.DEEP)
    )

    assert set(result.model_verdicts) == {"qwen3_guard_gen", "nemotron_csr", "triage"}
    assert result.model_verdicts["triage"].verdict == VerdictLabel.CONTROVERSIAL
    assert cascade._triage.calls == [{
        "text": "disagreement",
        "policy": "block PII",
        "qwen_verdict": VerdictLabel.SAFE,
        "csr_verdict": VerdictLabel.UNSAFE,
    }]


def test_stream_document_verdict_scores_safe_and_controversial_tokens():
    class MixedStream:
        _loaded = True

        def stream_classify(self, text: str):
            yield "safe", 0.80, 0          # risk 0.20
            yield "controversial", 0.95, 1 # capped to 0.69
            yield "safe", 0.99, 2          # risk 0.01

    cascade = Cascade(CascadeConfig(qwen_stream_model="stream-test"))
    cascade._qwen_stream = MixedStream()

    verdict = cascade._stream_document_verdict("abc")

    assert verdict.verdict == VerdictLabel.CONTROVERSIAL
    assert verdict.score == 0.69
    assert verdict.model_id == "stream-test"
    assert verdict.reasoning == "Highest streaming token risk at token index 1."


def test_stream_document_verdict_handles_empty_stream():
    class EmptyStream:
        _loaded = True

        def stream_classify(self, text: str):
            return
            yield

    cascade = Cascade(CascadeConfig(qwen_stream_model="stream-test"))
    cascade._qwen_stream = EmptyStream()

    verdict = cascade._stream_document_verdict("abc")

    assert verdict.verdict == VerdictLabel.SAFE
    assert verdict.score == 0.0
    assert verdict.reasoning == "Highest streaming token risk at token index 0."


def test_stream_token_verdicts_delegates_to_stream_model():
    cascade = Cascade(CascadeConfig())
    cascade._qwen_stream = StaticStream()

    assert list(cascade.stream_token_verdicts("bad")) == [
        ("safe", 0.96, 0),
        ("unsafe", 0.91, 1),
    ]
