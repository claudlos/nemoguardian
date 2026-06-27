"""Check a running nemoguardian demo host and write submission evidence.

This script is safe to run before loading model weights. By default it checks
health, the demo page, provider offers, and production auth boundaries. Add
`--moderate` when the GPU host is ready for real model inference.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urljoin
from urllib.request import Request, urlopen

PII_EXAMPLE = "Hey @everyone, drop your SSN and I will send $100 to whoever DMs me first"
DEFAULT_POLICY = "block PII and financial scams"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default=os.environ.get("NEMOGUARDIAN_DEMO_BASE_URL", "http://localhost:8000"),
        help="Base URL for the running nemoguardian server",
    )
    parser.add_argument(
        "--wait-seconds",
        type=float,
        default=0.0,
        help="Poll /health for this many seconds before failing",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Require /health to report CUDA/GPU runtime",
    )
    parser.add_argument(
        "--require-triage",
        action="store_true",
        help="Require /health to report configured NVIDIA/OpenRouter triage",
    )
    parser.add_argument(
        "--max-cheapest-usd",
        type=float,
        default=0.10,
        help="Maximum acceptable cheapest fitting GPU offer in USD/hr",
    )
    parser.add_argument(
        "--moderate",
        action="store_true",
        help="Call /demo/moderate; this loads model weights",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Use deep mode for --moderate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write evidence JSON to this path",
    )
    args = parser.parse_args()

    evidence: dict[str, Any] = {
        "tool": "demo_host_check",
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "base_url": args.base_url.rstrip("/"),
        "requirements": {
            "require_gpu": args.require_gpu,
            "require_triage": args.require_triage,
            "max_cheapest_usd": args.max_cheapest_usd,
            "moderate": args.moderate,
            "deep": args.deep,
        },
        "repo": _repo_metadata(),
        "checks": [],
    }

    health = _wait_for_health(args.base_url, args.wait_seconds, evidence)
    if health is not None:
        _check_health(
            health,
            evidence,
            require_gpu=args.require_gpu,
            require_triage=args.require_triage,
        )

    _check_providers(args.base_url, evidence, max_cheapest_usd=args.max_cheapest_usd)
    _check_demo_page(args.base_url, evidence)
    _check_stream_auth_boundary(args.base_url, evidence)

    if args.moderate or args.deep:
        _check_demo_moderation(args.base_url, evidence, deep=args.deep)

    passed = all(check["ok"] for check in evidence["checks"])
    evidence["passed"] = passed

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")

    for check in evidence["checks"]:
        state = "PASS" if check["ok"] else "FAIL"
        print(f"{state} {check['name']}: {check['detail']}")

    if args.output:
        print(f"evidence: {args.output}")

    return 0 if passed else 1


def _wait_for_health(
    base_url: str,
    wait_seconds: float,
    evidence: dict[str, Any],
) -> dict[str, Any] | None:
    deadline = time.monotonic() + wait_seconds
    last_error = ""
    while True:
        status, body = _request(base_url, "/health")
        if status == 200:
            data = _json(body)
            if isinstance(data, dict):
                _add(evidence, "health_reachable", True, "/health returned JSON", data)
                return data
            last_error = "health response was not JSON"
        else:
            last_error = f"/health returned HTTP {status}"

        if time.monotonic() >= deadline:
            _add(evidence, "health_reachable", False, last_error)
            return None
        time.sleep(1)


def _check_health(
    health: dict[str, Any],
    evidence: dict[str, Any],
    *,
    require_gpu: bool,
    require_triage: bool,
) -> None:
    status = health.get("status")
    _add(evidence, "health_status", status == "ok", f"status={status!r}")

    model_config = health.get("model_config") or {}
    expected_models = {
        "qwen_gen_model": "Qwen3Guard-Gen",
        "qwen_stream_model": "Qwen3Guard-Stream",
        "nemotron_csr_model": "Nemotron-Content-Safety-Reasoning",
        "nemotron_triage_model": "nemotron-3-ultra",
    }
    missing = [
        key for key, needle in expected_models.items()
        if needle not in str(model_config.get(key, ""))
    ]
    _add(
        evidence,
        "model_config",
        not missing,
        "expected model IDs present" if not missing else f"missing/changed: {', '.join(missing)}",
        model_config,
    )

    runtime_device = str(health.get("runtime_device", ""))
    gpu_ok = "cuda" in runtime_device.lower() or bool(health.get("gpu_available"))
    if require_gpu:
        _add(evidence, "runtime_gpu", gpu_ok, f"runtime_device={runtime_device!r}")
    else:
        _add(evidence, "runtime_reported", bool(runtime_device), f"runtime_device={runtime_device!r}")

    triage_ok = bool(health.get("triage_configured")) and health.get("triage_provider") in {
        "nvidia",
        "openrouter",
    }
    if require_triage:
        _add(evidence, "triage_configured", triage_ok, f"provider={health.get('triage_provider')!r}")
    else:
        _add(
            evidence,
            "triage_reported",
            "triage_configured" in health,
            f"configured={health.get('triage_configured')!r}",
        )


def _check_providers(
    base_url: str,
    evidence: dict[str, Any],
    *,
    max_cheapest_usd: float,
) -> None:
    status, body = _request(base_url, "/providers/offers", params={"only_fits": "true"})
    data = _json(body)
    offers = data.get("offers", []) if isinstance(data, dict) else []
    _add(evidence, "provider_offers_http", status == 200, f"HTTP {status}")
    _add(evidence, "provider_offers_present", bool(offers), f"offers={len(offers)}")
    if offers:
        cheapest = min(offers, key=lambda offer: float(offer.get("price_per_hour_usd", 9999)))
        price = float(cheapest.get("price_per_hour_usd", 9999))
        _add(
            evidence,
            "commodity_gpu_price",
            price <= max_cheapest_usd,
            f"cheapest={cheapest.get('gpu_model')} at ${price:.2f}/hr",
            cheapest,
        )


def _check_demo_page(base_url: str, evidence: dict[str, Any]) -> None:
    status, body = _request(base_url, "/demo")
    ok = status == 200 and "Moderation Console" in body
    _add(evidence, "demo_page", ok, f"HTTP {status}, console={'Moderation Console' in body}")


def _check_stream_auth_boundary(base_url: str, evidence: dict[str, Any]) -> None:
    status, _body = _request(
        base_url,
        "/v1/moderate/stream",
        method="POST",
        json_body={"text": "hello"},
    )
    _add(evidence, "stream_requires_auth", status == 401, f"HTTP {status}")


def _check_demo_moderation(base_url: str, evidence: dict[str, Any], *, deep: bool) -> None:
    mode = "deep" if deep else "standard"
    status, body = _request(
        base_url,
        "/demo/moderate",
        method="POST",
        params={"policy_preset": "discord"},
        json_body={
            "text": PII_EXAMPLE,
            "policy": DEFAULT_POLICY,
            "mode": mode,
        },
        timeout=300,
    )
    data = _json(body)
    ok_http = status == 200 and isinstance(data, dict)
    _add(evidence, "demo_moderate_http", ok_http, f"HTTP {status}")
    if not ok_http:
        _add(evidence, "demo_moderate_verdict", False, "no JSON verdict")
        return

    verdict_ok = data.get("verdict") == "unsafe"
    _add(
        evidence,
        "demo_moderate_verdict",
        verdict_ok,
        f"verdict={data.get('verdict')!r}, mode={data.get('mode')!r}",
        {
            "verdict": data.get("verdict"),
            "score": data.get("score"),
            "mode": data.get("mode"),
            "matched_policy_rule": data.get("matched_policy_rule"),
            "total_latency_ms": data.get("total_latency_ms"),
        },
    )

    model_verdicts = data.get("model_verdicts", {})
    model_errors = {
        name: verdict.get("error")
        for name, verdict in model_verdicts.items()
        if isinstance(verdict, dict) and verdict.get("error")
    }
    _add(
        evidence,
        "demo_moderate_model_errors",
        not model_errors,
        "no model errors" if not model_errors else f"errors={model_errors}",
        {"models": sorted(model_verdicts), "errors": model_errors},
    )

    if deep:
        _add(
            evidence,
            "deep_triage_result",
            "triage" in model_verdicts,
            f"models={', '.join(sorted(model_verdicts))}",
        )


def _request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    params: dict[str, str] | None = None,
    json_body: dict[str, Any] | None = None,
    timeout: float = 15,
) -> tuple[int, str]:
    url = urljoin(base_url.rstrip("/") + "/", path.lstrip("/"))
    if params:
        url = f"{url}?{urlencode(params)}"

    headers = {"Accept": "application/json"}
    data = None
    if json_body is not None:
        data = json.dumps(json_body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = Request(url, data=data, headers=headers, method=method)
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except (ConnectionError, OSError, URLError) as exc:
        return 0, str(exc)


def _repo_metadata() -> dict[str, str | bool | None]:
    status = _git_output("status", "--porcelain")
    return {
        "branch": _git_output("branch", "--show-current"),
        "commit": _git_output("rev-parse", "HEAD"),
        "short_commit": _git_output("rev-parse", "--short", "HEAD"),
        "dirty": None if status is None else bool(status),
    }


def _git_output(*args: str) -> str | None:
    root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def _json(body: str) -> Any:
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def _add(
    evidence: dict[str, Any],
    name: str,
    ok: bool,
    detail: str,
    data: Any | None = None,
) -> None:
    item: dict[str, Any] = {"name": name, "ok": ok, "detail": detail}
    if data is not None:
        item["data"] = data
    evidence["checks"].append(item)


if __name__ == "__main__":
    raise SystemExit(main())
