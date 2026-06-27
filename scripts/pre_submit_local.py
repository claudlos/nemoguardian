"""Run the local, non-GPU pre-submit gate.

This automates the checks that can be proven before renting the final GPU host:

- Python lint/tests
- Docker image build
- license/model-doc presence inside the image
- container boot + lightweight demo host evidence

It intentionally does not run real-model inference. Use the GPU runbook for the
final `--require-gpu --require-triage --moderate --deep` evidence file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shlex
import socket
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = "nemoguardian/self-hosted:latest"
DEFAULT_OUTPUT_DIR = ROOT / "build"
LICENSE_ASSERTION = (
    "test -s /opt/nemoguardian/LICENSE && "
    "test -s /opt/nemoguardian/NOTICE && "
    "test -s /opt/nemoguardian/docs/THIRD_PARTY_MODELS.md && "
    "test -s /opt/nemoguardian/docs/MODEL_CATALOG.md"
)

CommandRunner = Callable[[Sequence[str]], int]
PortPicker = Callable[[], int]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image tag to build and smoke")
    parser.add_argument(
        "--container-name",
        default=None,
        help="Temporary container name; defaults to a process-scoped name",
    )
    parser.add_argument("--port", type=int, default=0, help="Host port; 0 chooses a free port")
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=60.0,
        help="Seconds to wait for the container health endpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for local evidence artifacts",
    )
    args = parser.parse_args()

    container_name = args.container_name or f"nemoguardian-pre-submit-{os.getpid()}"
    summary = run_local_checks(
        image=args.image,
        container_name=container_name,
        port=args.port,
        wait_seconds=args.wait_seconds,
        output_dir=args.output_dir,
    )
    summary_path = Path(summary["artifacts"]["summary"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    for check in summary["checks"]:
        state = "PASS" if check["ok"] else "FAIL"
        print(f"{state} {check['name']}: {check['detail']}")
    print(f"summary: {summary_path}")
    print(f"light evidence: {summary['artifacts']['light_evidence']}")
    return 0 if summary["passed"] else 1


def run_local_checks(
    *,
    image: str = DEFAULT_IMAGE,
    container_name: str,
    port: int,
    wait_seconds: float,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    command_runner: CommandRunner | None = None,
    port_picker: PortPicker | None = None,
) -> dict[str, Any]:
    runner = command_runner or _run_command
    pick_port = port_picker or _pick_available_port
    selected_port = port if port > 0 else pick_port()
    base_url = f"http://localhost:{selected_port}"
    output_dir = output_dir.resolve()
    evidence_path = output_dir / "docker-demo-evidence-light.json"
    summary_path = output_dir / "local-pre-submit-summary.json"
    checks: list[dict[str, Any]] = []
    container_started = False

    def record_command(name: str, cmd: Sequence[str]) -> bool:
        code = runner(cmd)
        ok = code == 0
        checks.append({
            "name": name,
            "ok": ok,
            "detail": f"exit_code={code}",
            "command": list(cmd),
        })
        return ok

    def skip(name: str, reason: str) -> None:
        checks.append({"name": name, "ok": False, "detail": f"skipped: {reason}"})

    verify_ok = record_command("python_verify", ["make", "verify"])
    if not verify_ok:
        skip("docker_build", "python verification failed")
        return _summary(
            checks=checks,
            image=image,
            base_url=base_url,
            evidence_path=evidence_path,
            summary_path=summary_path,
        )

    build_ok = record_command("docker_build", ["make", "docker-build", f"IMAGE={image}"])
    if not build_ok:
        skip("image_license_files", "docker build failed")
        skip("container_light_demo_check", "docker build failed")
        return _summary(
            checks=checks,
            image=image,
            base_url=base_url,
            evidence_path=evidence_path,
            summary_path=summary_path,
        )

    record_command(
        "image_license_files",
        ["docker", "run", "--rm", "--entrypoint", "sh", image, "-c", LICENSE_ASSERTION],
    )

    try:
        container_started = record_command(
            "container_start",
            [
                "docker",
                "run",
                "--rm",
                "-d",
                "--name",
                container_name,
                "-p",
                f"{selected_port}:8000",
                "-e",
                "NEMOGUARDIAN_API_KEY=nmg_local_pre_submit_key",
                "-e",
                "NEMOGUARDIAN_ENABLE_DEMO_ENDPOINT=1",
                image,
            ],
        )
        if container_started:
            record_command(
                "container_light_demo_check",
                [
                    sys.executable,
                    "scripts/demo_host_check.py",
                    "--base-url",
                    base_url,
                    "--wait-seconds",
                    str(wait_seconds),
                    "--output",
                    str(evidence_path),
                ],
            )
        else:
            skip("container_light_demo_check", "container did not start")
    finally:
        if container_started:
            record_command("container_stop", ["docker", "stop", container_name])

    return _summary(
        checks=checks,
        image=image,
        base_url=base_url,
        evidence_path=evidence_path,
        summary_path=summary_path,
    )


def _summary(
    *,
    checks: list[dict[str, Any]],
    image: str,
    base_url: str,
    evidence_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    return {
        "tool": "pre_submit_local",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "image": image,
        "base_url": base_url,
        "repo": _repo_metadata(),
        "artifacts": {
            "light_evidence": str(evidence_path),
            "summary": str(summary_path),
        },
        "checks": checks,
        "passed": all(check["ok"] for check in checks),
    }


def _run_command(cmd: Sequence[str]) -> int:
    print(f"$ {shlex.join(cmd)}", flush=True)
    completed = subprocess.run(list(cmd), cwd=ROOT, check=False)
    return completed.returncode


def _pick_available_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _repo_metadata() -> dict[str, str | bool | None]:
    status = _git_output("status", "--porcelain")
    return {
        "branch": _git_output("branch", "--show-current"),
        "commit": _git_output("rev-parse", "HEAD"),
        "short_commit": _git_output("rev-parse", "--short", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


if __name__ == "__main__":
    raise SystemExit(main())
