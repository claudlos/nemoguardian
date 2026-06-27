"""Tests for hackathon submission helper scripts."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

from scripts import demo_host_check, pre_submit_local, real_model_smoke


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
                    "nemotron_triage_model": "nvidia/nemotron-3-ultra-220b-a12b",
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
                    "nemotron_triage_model": "nvidia/nemotron-3-ultra-220b-a12b",
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
                    "nemotron_triage_model": "nvidia/nemotron-3-ultra-220b-a12b",
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
