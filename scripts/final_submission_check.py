"""Validate final hackathon submission artifacts.

Run this after the GPU host has produced `demo-evidence.json` and after the
demo video has a hosted URL. The script checks the final artifacts against the
same requirements in `docs/PRE_SUBMIT_CHECKLIST.md`.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_GPU_CHECKS = {
    "health_reachable",
    "health_status",
    "model_config",
    "runtime_gpu",
    "triage_configured",
    "provider_offers_http",
    "provider_offers_present",
    "commodity_gpu_price",
    "demo_page",
    "stream_requires_auth",
    "demo_moderate_http",
    "demo_moderate_verdict",
    "demo_moderate_model_errors",
    "deep_triage_result",
}
VIDEO_URL_RE = re.compile(r"^https?://\S+$")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-evidence", type=Path, default=ROOT / "demo-evidence.json")
    parser.add_argument(
        "--local-summary",
        type=Path,
        default=ROOT / "build" / "local-pre-submit-summary.json",
    )
    parser.add_argument(
        "--submission-form",
        type=Path,
        default=ROOT / "docs" / "SUBMISSION_FORM.md",
    )
    parser.add_argument("--video-url", required=True, help="Hosted final demo video URL")
    parser.add_argument(
        "--expected-commit",
        default=None,
        help="Expected short commit in evidence; defaults to current HEAD short SHA",
    )
    args = parser.parse_args()

    result = run_final_checks(
        gpu_evidence_path=args.gpu_evidence,
        local_summary_path=args.local_summary,
        submission_form_path=args.submission_form,
        video_url=args.video_url,
        expected_commit=args.expected_commit,
    )
    for check in result["checks"]:
        state = "PASS" if check["ok"] else "FAIL"
        print(f"{state} {check['name']}: {check['detail']}")
    return 0 if result["passed"] else 1


def run_final_checks(
    *,
    gpu_evidence_path: Path,
    local_summary_path: Path,
    submission_form_path: Path,
    video_url: str,
    expected_commit: str | None = None,
    repo_short_commit: str | None = None,
) -> dict[str, Any]:
    expected = expected_commit or repo_short_commit or _git_output("rev-parse", "--short", "HEAD")
    checks: list[dict[str, Any]] = []

    _check_required_files(checks)
    _check_video_url(checks, video_url)
    _check_submission_form(checks, submission_form_path, video_url)
    _check_local_summary(checks, local_summary_path, expected)
    _check_gpu_evidence(checks, gpu_evidence_path, expected)

    return {
        "tool": "final_submission_check",
        "expected_commit": expected,
        "checks": checks,
        "passed": all(check["ok"] for check in checks),
    }


def _check_required_files(checks: list[dict[str, Any]]) -> None:
    required = [
        ROOT / "LICENSE",
        ROOT / "NOTICE",
        ROOT / "README.md",
        ROOT / "SUBMISSION.md",
        ROOT / "docs" / "THIRD_PARTY_MODELS.md",
        ROOT / "docs" / "MODEL_CATALOG.md",
        ROOT / "docs" / "JUDGE_GUIDE.md",
        ROOT / "docs" / "VASTAI_DEMO_RUNBOOK.md",
        ROOT / "docs" / "VIDEO_SCRIPT.md",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file() or path.stat().st_size == 0]
    _add(
        checks,
        "required_files",
        not missing,
        "all required files present" if not missing else f"missing/empty: {', '.join(missing)}",
    )


def _check_video_url(checks: list[dict[str, Any]], video_url: str) -> None:
    clean = video_url.strip()
    ok = bool(VIDEO_URL_RE.match(clean)) and "TBD" not in clean.upper()
    _add(checks, "video_url", ok, f"url={clean!r}" if clean else "missing video URL")


def _check_submission_form(checks: list[dict[str, Any]], form_path: Path, video_url: str) -> None:
    text = _read_text(form_path)
    if text is None:
        _add(checks, "submission_form", False, f"missing form: {form_path}")
        return
    has_repo = "https://github.com/claudlos/nemoguardian" in text
    has_no_tbd = "TBD" not in text
    mentions_video = video_url.strip() in text if video_url.strip() else False
    _add(
        checks,
        "submission_form",
        has_repo and has_no_tbd and mentions_video,
        _form_detail(has_repo=has_repo, has_no_tbd=has_no_tbd, mentions_video=mentions_video),
    )


def _check_local_summary(
    checks: list[dict[str, Any]],
    summary_path: Path,
    expected_commit: str | None,
) -> None:
    data = _read_json(summary_path)
    if not isinstance(data, dict):
        _add(checks, "local_pre_submit_summary", False, f"missing/invalid JSON: {summary_path}")
        return
    passed = data.get("passed") is True
    tool_ok = data.get("tool") == "pre_submit_local"
    commit_ok = _short_commit_matches(data, expected_commit)
    required = {"python_verify", "docker_build", "image_license_files", "container_light_demo_check"}
    check_names = {item.get("name") for item in data.get("checks", []) if isinstance(item, dict) and item.get("ok")}
    checks_ok = required <= check_names
    ok = passed and tool_ok and commit_ok and checks_ok
    detail = (
        f"passed={passed}, tool={data.get('tool')!r}, commit_ok={commit_ok}, "
        f"required_checks_ok={checks_ok}"
    )
    _add(checks, "local_pre_submit_summary", ok, detail)


def _check_gpu_evidence(
    checks: list[dict[str, Any]],
    evidence_path: Path,
    expected_commit: str | None,
) -> None:
    data = _read_json(evidence_path)
    if not isinstance(data, dict):
        _add(checks, "gpu_evidence", False, f"missing/invalid JSON: {evidence_path}")
        return
    requirements = data.get("requirements", {})
    requirement_flags = {
        "require_gpu": requirements.get("require_gpu") is True,
        "require_triage": requirements.get("require_triage") is True,
        "moderate": requirements.get("moderate") is True,
        "deep": requirements.get("deep") is True,
    }
    check_status = {
        item.get("name"): item.get("ok")
        for item in data.get("checks", [])
        if isinstance(item, dict)
    }
    missing_or_failed = sorted(
        name for name in REQUIRED_GPU_CHECKS if check_status.get(name) is not True
    )
    passed = data.get("passed") is True
    tool_ok = data.get("tool") == "demo_host_check"
    commit_ok = _short_commit_matches(data, expected_commit)
    dirty_ok = data.get("repo", {}).get("dirty") is False
    flags_ok = all(requirement_flags.values())
    ok = passed and tool_ok and commit_ok and dirty_ok and flags_ok and not missing_or_failed
    detail = (
        f"passed={passed}, tool={data.get('tool')!r}, commit_ok={commit_ok}, "
        f"dirty_ok={dirty_ok}, flags={requirement_flags}, "
        f"missing_or_failed={missing_or_failed}"
    )
    _add(checks, "gpu_evidence", ok, detail)


def _short_commit_matches(data: dict[str, Any], expected_commit: str | None) -> bool:
    if not expected_commit:
        return False
    actual = str(data.get("repo", {}).get("short_commit", ""))
    return actual == expected_commit or actual.startswith(expected_commit) or expected_commit.startswith(actual)


def _form_detail(*, has_repo: bool, has_no_tbd: bool, mentions_video: bool) -> str:
    missing = []
    if not has_repo:
        missing.append("repo URL")
    if not has_no_tbd:
        missing.append("TBD placeholders remain")
    if not mentions_video:
        missing.append("video URL not pasted into form")
    return "ready" if not missing else "; ".join(missing)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text()
    except Exception:
        return None


def _add(checks: list[dict[str, Any]], name: str, ok: bool, detail: str) -> None:
    checks.append({"name": name, "ok": ok, "detail": detail})


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
