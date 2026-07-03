"""NemotronTriage.adjudicate — deep-mode adjudication, fully mocked (no network)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nemoguardian.models.nemotron_triage import NemotronTriage
from nemoguardian.schemas import ModelVerdict, VerdictLabel


def _verdict(label: VerdictLabel, score: float, reasoning: str = "r") -> ModelVerdict:
    return ModelVerdict(
        model_id="m", verdict=label, score=score, latency_ms=1.0, reasoning=reasoning
    )


class FakeClient:
    """Mimics openai.OpenAI far enough for adjudicate()."""

    def __init__(self, content: str | None = None, exc: Exception | None = None) -> None:
        self.calls: list[dict] = []
        self._content = content
        self._exc = exc
        completions = SimpleNamespace(create=self._create)
        self.chat = SimpleNamespace(completions=completions)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        message = SimpleNamespace(content=self._content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _triage(client: FakeClient | None) -> NemotronTriage:
    t = NemotronTriage(api_key="test-key" if client is not None else None)
    t._client = client
    return t


def _adjudicate(t: NemotronTriage) -> ModelVerdict:
    return t.adjudicate(
        text="borderline text",
        policy="no PII",
        qwen_verdict=_verdict(VerdictLabel.SAFE, 0.1),
        csr_verdict=_verdict(VerdictLabel.UNSAFE, 0.9),
    )


def test_disabled_without_api_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    t = NemotronTriage()
    out = _adjudicate(t)
    assert out.verdict == VerdictLabel.SAFE
    assert out.reasoning == "triage disabled (no API key)"


def test_clean_json_response():
    client = FakeClient('{"verdict": "unsafe", "score": 0.85, "reasons": ["policy hit"]}')
    out = _adjudicate(_triage(client))
    assert out.verdict == VerdictLabel.UNSAFE
    assert out.score == 0.85
    assert out.categories == ["policy hit"]
    assert out.error is None


def test_code_fenced_json_response():
    client = FakeClient('```json\n{"verdict": "controversial", "score": 0.5, "reasons": []}\n```')
    out = _adjudicate(_triage(client))
    assert out.verdict == VerdictLabel.CONTROVERSIAL
    assert out.score == 0.5


def test_json_embedded_in_prose_uses_regex_fallback():
    client = FakeClient(
        'Sure! Here is my assessment: {"verdict": "unsafe", "score": 0.9, "reasons": ["x"]}'
        " Hope that helps."
    )
    out = _adjudicate(_triage(client))
    assert out.verdict == VerdictLabel.UNSAFE


def test_unparseable_output_defaults_safe_verdict():
    client = FakeClient("I refuse to answer in JSON.")
    out = _adjudicate(_triage(client))
    assert out.verdict == VerdictLabel.SAFE
    assert out.score == 0.0


def test_unknown_verdict_string_maps_to_safe():
    client = FakeClient('{"verdict": "banana", "score": 0.9, "reasons": []}')
    out = _adjudicate(_triage(client))
    assert out.verdict == VerdictLabel.SAFE


def test_api_error_returns_error_verdict():
    client = FakeClient(exc=RuntimeError("upstream 503"))
    out = _adjudicate(_triage(client))
    assert out.error is not None
    assert "upstream 503" in out.error
    # Error verdicts are excluded from aggregation, so the label is inert.
    assert out.verdict == VerdictLabel.SAFE


def test_timeout_passed_to_api_call(monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_TRIAGE_TIMEOUT_S", "7")
    client = FakeClient('{"verdict": "safe", "score": 0.0, "reasons": []}')
    _adjudicate(_triage(client))
    assert client.calls[0]["timeout"] == 7.0


def test_prompt_includes_both_classifier_verdicts():
    client = FakeClient('{"verdict": "safe", "score": 0.0, "reasons": []}')
    _adjudicate(_triage(client))
    prompt = client.calls[0]["messages"][0]["content"]
    assert "verdict=safe" in prompt
    assert "verdict=unsafe" in prompt
    assert "borderline text" in prompt
    assert "no PII" in prompt


def test_score_out_of_json_is_coerced_float():
    client = FakeClient('{"verdict": "unsafe", "score": "0.75", "reasons": []}')
    out = _adjudicate(_triage(client))
    assert out.score == pytest.approx(0.75)
