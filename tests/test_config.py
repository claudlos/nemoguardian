"""Runtime configuration tests."""

from __future__ import annotations

from nemoguardian.cascade import Cascade, CascadeConfig


def test_cascade_config_from_env(monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_QWEN_MODEL", "Qwen/test-gen")
    monkeypatch.setenv("NEMOGUARDIAN_QWEN_STREAM_MODEL", "Qwen/test-stream")
    monkeypatch.setenv("NEMOGUARDIAN_CSR_MODEL", "nvidia/test-csr")
    monkeypatch.setenv("NEMOGUARDIAN_TRIAGE_MODEL", "nvidia/test-ultra")
    monkeypatch.setenv("NEMOGUARDIAN_TRIAGE_BASE_URL", "https://triage.example/v1")
    monkeypatch.setenv("NEMOGUARDIAN_QUANTIZE", "0")
    monkeypatch.setenv("NEMOGUARDIAN_QWEN_4BIT", "1")
    monkeypatch.setenv("NEMOGUARDIAN_REASONING", "false")

    config = CascadeConfig.from_env()

    assert config.qwen_gen_model == "Qwen/test-gen"
    assert config.qwen_stream_model == "Qwen/test-stream"
    assert config.csr_model == "nvidia/test-csr"
    assert config.triage_model == "nvidia/test-ultra"
    assert config.triage_base_url == "https://triage.example/v1"
    assert config.qwen_gen_4bit is True
    assert config.csr_4bit is False
    assert config.reasoning is False


def test_triage_status_prefers_openrouter_when_only_openrouter_key_is_set(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)
    monkeypatch.delenv("NEMOGUARDIAN_TRIAGE_BASE_URL", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    cascade = Cascade(CascadeConfig.from_env())
    status = cascade.triage_status()

    assert status["configured"] is True
    assert status["provider"] == "openrouter"
    assert status["base_url"] == "https://openrouter.ai/api/v1"
