"""Tests for hackathon submission helper scripts."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from scripts import (
    demo_host_check,
    discord_actor_scenario,
    final_submission_check,
    framework_smoke,
    pre_submit_local,
    real_model_smoke,
    triage_api_smoke,
)


def test_demo_host_check_writes_passing_light_evidence(monkeypatch, tmp_path):
    def fake_request(base_url, path, **kwargs):
        if path == "/health":
            return 200, json.dumps({
                "status": "ok",
                "runtime_device": "cpu",
                "gpu_available": False,
                "triage_configured": False,
                "triage_provider": None,
                "model_config": {
                    "qwen_gen_model": "Qwen/Qwen3Guard-Gen-4B",
                    "qwen_stream_model": "Qwen/Qwen3Guard-Stream-0.6B",
                    "nemotron_csr_model": "nvidia/Nemotron-Content-Safety-Reasoning-4B",
                    "nemotron_triage_model": "nvidia/nemotron-3-ultra-550b-a55b",
                },
            })
        if path == "/providers/offers":
            return 200, json.dumps({
                "offers": [
                    {
                        "provider": "vastai",
                        "gpu_model": "RTX 3090",
                        "price_per_hour_usd": 0.07,
                        "vram_gb": 24,
                    }
                ]
            })
        if path == "/demo":
            return 200, "<html>Moderation Console</html>"
        if path == "/v1/moderate/stream":
            assert kwargs["method"] == "POST"
            return 401, json.dumps({"detail": "missing Authorization header"})
        raise AssertionError(f"unexpected request: {path}")

    output = tmp_path / "evidence.json"
    monkeypatch.setattr(demo_host_check, "_request", fake_request)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo_host_check.py",
            "--base-url",
            "http://demo.test",
            "--output",
            str(output),
        ],
    )

    assert demo_host_check.main() == 0
    evidence = json.loads(output.read_text())
    assert evidence["passed"] is True
    assert evidence["tool"] == "demo_host_check"
    assert evidence["generated_at"]
    assert evidence["repo"]["short_commit"]
    assert evidence["requirements"]["require_gpu"] is False
    assert {check["name"] for check in evidence["checks"]} >= {
        "health_reachable",
        "model_config",
        "commodity_gpu_price",
        "demo_page",
        "stream_requires_auth",
    }


def test_demo_host_check_gpu_and_triage_requirements_fail_on_cpu(monkeypatch, tmp_path):
    def fake_request(base_url, path, **kwargs):
        if path == "/health":
            return 200, json.dumps({
                "status": "ok",
                "runtime_device": "cpu",
                "gpu_available": False,
                "triage_configured": False,
                "triage_provider": None,
                "model_config": {
                    "qwen_gen_model": "Qwen/Qwen3Guard-Gen-4B",
                    "qwen_stream_model": "Qwen/Qwen3Guard-Stream-0.6B",
                    "nemotron_csr_model": "nvidia/Nemotron-Content-Safety-Reasoning-4B",
                    "nemotron_triage_model": "nvidia/nemotron-3-ultra-550b-a55b",
                },
            })
        if path == "/providers/offers":
            return 200, json.dumps({
                "offers": [{"gpu_model": "RTX 3090", "price_per_hour_usd": 0.07}]
            })
        if path == "/demo":
            return 200, "Moderation Console"
        if path == "/v1/moderate/stream":
            return 401, "{}"
        raise AssertionError(f"unexpected request: {path}")

    output = tmp_path / "evidence.json"
    monkeypatch.setattr(demo_host_check, "_request", fake_request)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo_host_check.py",
            "--require-gpu",
            "--require-triage",
            "--output",
            str(output),
        ],
    )

    assert demo_host_check.main() == 1
    evidence = json.loads(output.read_text())
    assert evidence["requirements"]["require_gpu"] is True
    assert evidence["requirements"]["require_triage"] is True
    failed = {check["name"] for check in evidence["checks"] if not check["ok"]}
    assert failed == {"runtime_gpu", "triage_configured"}


def test_demo_host_check_deep_moderation_requires_triage_model(monkeypatch, tmp_path):
    def fake_request(base_url, path, **kwargs):
        if path == "/health":
            return 200, json.dumps({
                "status": "ok",
                "runtime_device": "cuda: RTX 3090",
                "gpu_available": True,
                "triage_configured": True,
                "triage_provider": "nvidia",
                "model_config": {
                    "qwen_gen_model": "Qwen/Qwen3Guard-Gen-4B",
                    "qwen_stream_model": "Qwen/Qwen3Guard-Stream-0.6B",
                    "nemotron_csr_model": "nvidia/Nemotron-Content-Safety-Reasoning-4B",
                    "nemotron_triage_model": "nvidia/nemotron-3-ultra-550b-a55b",
                },
            })
        if path == "/providers/offers":
            return 200, json.dumps({
                "offers": [{"gpu_model": "RTX 3090", "price_per_hour_usd": 0.07}]
            })
        if path == "/demo":
            return 200, "Moderation Console"
        if path == "/v1/moderate/stream":
            return 401, "{}"
        if path == "/demo/moderate":
            assert kwargs["json_body"]["mode"] == "deep"
            return 200, json.dumps({
                "verdict": "unsafe",
                "score": 0.9,
                "mode": "deep",
                "matched_policy_rule": "force-block-pii",
                "total_latency_ms": 100.0,
                "model_verdicts": {
                    "qwen3_guard_gen": {"error": None},
                    "nemotron_csr": {"error": None},
                },
            })
        raise AssertionError(f"unexpected request: {path}")

    output = tmp_path / "evidence.json"
    monkeypatch.setattr(demo_host_check, "_request", fake_request)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "demo_host_check.py",
            "--require-gpu",
            "--require-triage",
            "--deep",
            "--output",
            str(output),
        ],
    )

    assert demo_host_check.main() == 1
    evidence = json.loads(output.read_text())
    assert evidence["requirements"]["deep"] is True
    failed = {check["name"] for check in evidence["checks"] if not check["ok"]}
    assert failed == {"deep_triage_result"}


def test_demo_host_check_repo_metadata_reports_clean_status(monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd[1:] == ["status", "--porcelain"]:
            return SimpleNamespace(stdout="")
        if cmd[1:] == ["branch", "--show-current"]:
            return SimpleNamespace(stdout="main\n")
        if cmd[1:] == ["rev-parse", "HEAD"]:
            return SimpleNamespace(stdout="abc123\n")
        if cmd[1:] == ["rev-parse", "--short", "HEAD"]:
            return SimpleNamespace(stdout="abc\n")
        raise AssertionError(f"unexpected git command: {cmd}")

    monkeypatch.setattr(demo_host_check.subprocess, "run", fake_run)

    assert demo_host_check._repo_metadata() == {
        "branch": "main",
        "commit": "abc123",
        "short_commit": "abc",
        "dirty": False,
    }


def test_framework_smoke_writes_adapter_evidence(monkeypatch, tmp_path):
    def fake_request(base_url, path, **kwargs):
        if path == "/health":
            return 200, json.dumps({
                "status": "ok",
                "runtime_device": "cuda: RTX 3090",
                "gpu_available": True,
                "triage_configured": True,
                "triage_provider": "openrouter",
            })
        raise AssertionError(f"unexpected request: {path}")

    output = tmp_path / "framework-evidence.json"
    monkeypatch.setattr(framework_smoke, "_request", fake_request)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "framework_smoke.py",
            "--base-url",
            "http://demo.test",
            "--require-gpu",
            "--require-triage",
            "--output",
            str(output),
        ],
    )

    assert framework_smoke.main() == 0
    evidence = json.loads(output.read_text())
    assert evidence["tool"] == "framework_smoke"
    assert evidence["passed"] is True
    assert {check["name"] for check in evidence["checks"]} >= {
        "runtime_gpu",
        "triage_configured",
        "discord_unsafe_delete",
        "discord_controversial_reaction",
        "discord_bot_ignored",
        "twitch_delete_action",
        "webhook_auth_header",
        "webhook_forward_payload",
    }


def test_pre_submit_local_runs_expected_gate(monkeypatch, tmp_path):
    commands: list[list[str]] = []

    def runner(cmd):
        commands.append(list(cmd))
        return 0

    monkeypatch.setattr(
        pre_submit_local,
        "_repo_metadata",
        lambda: {"branch": "main", "commit": "abc", "short_commit": "abc", "dirty": False},
    )
    summary = pre_submit_local.run_local_checks(
        image="test/image:latest",
        container_name="nmg-test",
        port=0,
        wait_seconds=5,
        output_dir=tmp_path,
        command_runner=runner,
        port_picker=lambda: 8123,
    )

    assert summary["passed"] is True
    assert summary["base_url"] == "http://localhost:8123"
    assert summary["artifacts"]["light_evidence"] == str(
        tmp_path.resolve() / "docker-demo-evidence-light.json"
    )
    assert [check["name"] for check in summary["checks"]] == [
        "python_verify",
        "docker_build",
        "image_license_files",
        "container_start",
        "container_light_demo_check",
        "container_stop",
    ]
    assert commands[0] == ["make", "verify"]
    assert commands[1] == ["make", "docker-build", "IMAGE=test/image:latest"]
    assert commands[2][:5] == ["docker", "run", "--rm", "--entrypoint", "sh"]
    assert commands[3][:6] == ["docker", "run", "--rm", "-d", "--name", "nmg-test"]
    assert commands[4][1:] == [
        "scripts/demo_host_check.py",
        "--base-url",
        "http://localhost:8123",
        "--wait-seconds",
        "5",
        "--output",
        str(tmp_path.resolve() / "docker-demo-evidence-light.json"),
    ]
    assert commands[5] == ["docker", "stop", "nmg-test"]


def test_pre_submit_local_skips_docker_dependents_after_build_failure(monkeypatch, tmp_path):
    commands: list[list[str]] = []

    def runner(cmd):
        commands.append(list(cmd))
        if list(cmd)[:2] == ["make", "docker-build"]:
            return 2
        return 0

    monkeypatch.setattr(pre_submit_local, "_repo_metadata", lambda: {})
    summary = pre_submit_local.run_local_checks(
        image="test/image:latest",
        container_name="nmg-test",
        port=8123,
        wait_seconds=5,
        output_dir=tmp_path,
        command_runner=runner,
    )

    assert summary["passed"] is False
    assert [check["name"] for check in summary["checks"]] == [
        "python_verify",
        "docker_build",
        "image_license_files",
        "container_light_demo_check",
    ]
    assert all(command[0] != "docker" for command in commands)


def test_pre_submit_local_stops_container_after_light_check_failure(monkeypatch, tmp_path):
    commands: list[list[str]] = []

    def runner(cmd):
        command = list(cmd)
        commands.append(command)
        if "scripts/demo_host_check.py" in command:
            return 7
        return 0

    monkeypatch.setattr(pre_submit_local, "_repo_metadata", lambda: {})
    summary = pre_submit_local.run_local_checks(
        image="test/image:latest",
        container_name="nmg-test",
        port=8123,
        wait_seconds=5,
        output_dir=tmp_path,
        command_runner=runner,
    )

    assert summary["passed"] is False
    assert summary["checks"][-2]["name"] == "container_light_demo_check"
    assert summary["checks"][-2]["ok"] is False
    assert summary["checks"][-1]["name"] == "container_stop"
    assert commands[-1] == ["docker", "stop", "nmg-test"]


def test_final_submission_check_accepts_complete_artifacts(monkeypatch, tmp_path):
    video_url = "https://videos.example/nemoguardian-demo"
    local_summary = _write_json(
        tmp_path / "local-summary.json",
        _local_summary(commit="abc123", passed=True),
    )
    gpu_evidence = _write_json(
        tmp_path / "demo-evidence.json",
        _gpu_evidence(commit="abc123", passed=True),
    )
    form = tmp_path / "SUBMISSION_FORM.md"
    form.write_text(
        f"# Form\n\nhttps://github.com/claudlos/nemoguardian\n\nDemo Video\n\n{video_url}\n"
    )
    monkeypatch.setattr(final_submission_check, "ROOT", tmp_path)
    for relative in [
        "LICENSE",
        "NOTICE",
        "README.md",
        "SUBMISSION.md",
        "docs/THIRD_PARTY_MODELS.md",
        "docs/MODEL_CATALOG.md",
        "docs/JUDGE_GUIDE.md",
        "docs/VASTAI_DEMO_RUNBOOK.md",
        "docs/VIDEO_SCRIPT.md",
    ]:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("ok\n")

    result = final_submission_check.run_final_checks(
        gpu_evidence_path=gpu_evidence,
        local_summary_path=local_summary,
        submission_form_path=form,
        video_url=video_url,
        expected_commit="abc123",
    )

    assert result["passed"] is True
    assert {check["name"] for check in result["checks"]} == {
        "required_files",
        "video_url",
        "submission_form",
        "local_pre_submit_summary",
        "gpu_evidence",
    }


def test_final_submission_check_rejects_lightweight_gpu_evidence(monkeypatch, tmp_path):
    video_url = "https://videos.example/nemoguardian-demo"
    local_summary = _write_json(tmp_path / "local-summary.json", _local_summary())
    gpu_evidence = _write_json(
        tmp_path / "demo-evidence.json",
        _gpu_evidence(
            requirements={
                "require_gpu": False,
                "require_triage": False,
                "moderate": False,
                "deep": False,
            }
        ),
    )
    form = tmp_path / "SUBMISSION_FORM.md"
    form.write_text(f"https://github.com/claudlos/nemoguardian\n{video_url}\n")
    monkeypatch.setattr(final_submission_check, "ROOT", tmp_path)

    result = final_submission_check.run_final_checks(
        gpu_evidence_path=gpu_evidence,
        local_summary_path=local_summary,
        submission_form_path=form,
        video_url=video_url,
        expected_commit="abc123",
    )

    assert result["passed"] is False
    gpu_check = next(check for check in result["checks"] if check["name"] == "gpu_evidence")
    assert gpu_check["ok"] is False
    assert "'require_gpu': False" in gpu_check["detail"]


def test_final_submission_check_rejects_placeholder_video(monkeypatch, tmp_path):
    local_summary = _write_json(tmp_path / "local-summary.json", _local_summary())
    gpu_evidence = _write_json(tmp_path / "demo-evidence.json", _gpu_evidence())
    form = tmp_path / "SUBMISSION_FORM.md"
    form.write_text("https://github.com/claudlos/nemoguardian\nTBD\n")
    monkeypatch.setattr(final_submission_check, "ROOT", tmp_path)

    result = final_submission_check.run_final_checks(
        gpu_evidence_path=gpu_evidence,
        local_summary_path=local_summary,
        submission_form_path=form,
        video_url="TBD",
        expected_commit="abc123",
    )

    assert result["passed"] is False
    failures = {check["name"] for check in result["checks"] if not check["ok"]}
    assert {"video_url", "submission_form"} <= failures


def test_real_model_smoke_preflight_rejects_missing_cuda(monkeypatch):
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert real_model_smoke._preflight(deep=False, min_vram_gb=20) == 11


def test_real_model_smoke_preflight_rejects_small_gpu(monkeypatch):
    props = SimpleNamespace(name="Tiny GPU", total_memory=4 * 1024**3)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda index: props,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert real_model_smoke._preflight(deep=False, min_vram_gb=20) == 12


def test_real_model_smoke_preflight_requires_key_for_deep(monkeypatch):
    props = SimpleNamespace(name="RTX 3090", total_memory=24 * 1024**3)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda index: props,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert real_model_smoke._preflight(deep=True, min_vram_gb=20) == 13


def test_real_model_smoke_preflight_accepts_deep_gpu_with_key(monkeypatch):
    props = SimpleNamespace(name="RTX 3090", total_memory=24 * 1024**3)
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_properties=lambda index: props,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("NVIDIA_API_KEY", "set-but-not-real")

    assert real_model_smoke._preflight(deep=True, min_vram_gb=20) == 0


def test_real_model_smoke_model_profile_overrides(monkeypatch):
    monkeypatch.delenv("NEMOGUARDIAN_QWEN_MODEL", raising=False)
    args = SimpleNamespace(
        qwen_model="Qwen/Qwen3Guard-Gen-8B",
        qwen_stream_model="Qwen/Qwen3Guard-Stream-0.6B",
        csr_model="nvidia/Nemotron-Content-Safety-Reasoning-4B",
        triage_model="nvidia/nemotron-3-ultra-550b-a55b:free",
        triage_base_url="https://openrouter.ai/api/v1",
        quantize="on",
        qwen_4bit="off",
        csr_4bit="env",
    )

    config = real_model_smoke._config_from_args(args)

    assert config.qwen_gen_model == "Qwen/Qwen3Guard-Gen-8B"
    assert config.qwen_stream_model == "Qwen/Qwen3Guard-Stream-0.6B"
    assert config.csr_model == "nvidia/Nemotron-Content-Safety-Reasoning-4B"
    assert config.triage_model == "nvidia/nemotron-3-ultra-550b-a55b:free"
    assert config.triage_base_url == "https://openrouter.ai/api/v1"
    assert config.qwen_gen_4bit is False
    assert config.csr_4bit is True


def test_triage_api_smoke_requires_provider_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    evidence = triage_api_smoke.run(
        SimpleNamespace(
            text="text",
            policy="block PII",
            model="triage-test",
            base_url=None,
            expect_verdict=None,
        )
    )

    assert evidence == {
        "ok": False,
        "error": "NVIDIA_API_KEY or OPENROUTER_API_KEY is required",
        "provider": None,
    }


def test_triage_api_smoke_calls_openrouter(monkeypatch):
    calls = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"verdict":"unsafe","score":0.92,"reasons":["PII"]}'
                        )
                    )
                ]
            )

    class FakeOpenAI:
        def __init__(self, *, api_key, base_url, timeout=None):
            self.api_key = api_key
            self.base_url = base_url
            self.timeout = timeout
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-key")

    evidence = triage_api_smoke.run(
        SimpleNamespace(
            text="my SSN is 123-45-6789",
            policy="block PII",
            model="triage-test",
            base_url=None,
            expect_verdict="unsafe",
        )
    )

    assert evidence["ok"] is True
    assert evidence["provider"] == "openrouter"
    assert evidence["model"] == "triage-test"
    assert evidence["verdict"] == "unsafe"
    assert evidence["score"] == 0.92
    assert evidence["error"] is None
    assert calls[0]["model"] == "triage-test"


def test_discord_actor_scenario_helpers(tmp_path):
    assert discord_actor_scenario.parse_actor_tokens(" a , b ,, ") == ["a", "b"]
    assert discord_actor_scenario.parse_actor_tokens(None) == []

    default = discord_actor_scenario.load_scenario(None)
    assert default[0]["label"] == "good-helper"
    assert default[1]["expect_action"] == "delete"

    custom_path = tmp_path / "scenario.json"
    custom_path.write_text(json.dumps([{"actor": 4, "text": "hello"}]))
    custom = discord_actor_scenario.load_scenario(str(custom_path))
    assert custom == [{"actor": 4, "text": "hello"}]
    assert discord_actor_scenario._actor_index(custom[0], 2) == 0

    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{}")
    with pytest.raises(ValueError, match="scenario JSON must be a list"):
        discord_actor_scenario.load_scenario(str(bad_path))

    failures = discord_actor_scenario._expectation_failures(
        {"expect_action": "allow", "expect_verdict": "safe"},
        {"action": "delete", "verdict": "unsafe"},
        deleted=True,
        enforce=True,
    )
    assert failures == [
        "expected action allow, got delete",
        "expected verdict safe, got unsafe",
        "expected allowed message to remain visible, but it was deleted",
    ]


def _write_json(path, data):
    path.write_text(json.dumps(data))
    return path


def _local_summary(*, commit="abc123", passed=True):
    return {
        "tool": "pre_submit_local",
        "passed": passed,
        "repo": {"short_commit": commit},
        "checks": [
            {"name": "python_verify", "ok": True},
            {"name": "docker_build", "ok": True},
            {"name": "image_license_files", "ok": True},
            {"name": "container_light_demo_check", "ok": True},
        ],
    }


def _gpu_evidence(*, commit="abc123", passed=True, requirements=None):
    return {
        "tool": "demo_host_check",
        "passed": passed,
        "repo": {"short_commit": commit, "dirty": False},
        "requirements": requirements or {
            "require_gpu": True,
            "require_triage": True,
            "moderate": True,
            "deep": True,
        },
        "checks": [
            {"name": name, "ok": True}
            for name in sorted(final_submission_check.REQUIRED_GPU_CHECKS)
        ],
    }
