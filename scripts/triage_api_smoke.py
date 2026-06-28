#!/usr/bin/env python3
"""API-only smoke test for deep-mode Nemotron triage.

This does not load local guard models. It only verifies that the configured
OpenAI-compatible triage provider accepts a request and returns parseable
moderation adjudication JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--text",
        default="A user posted: my SSN is 123-45-6789 and I will pay anyone who DMs me.",
    )
    parser.add_argument("--policy", default="block PII, scams, harassment, slurs, and threats")
    parser.add_argument("--model", default=os.environ.get("NEMOGUARDIAN_TRIAGE_MODEL"))
    parser.add_argument("--base-url", default=os.environ.get("NEMOGUARDIAN_TRIAGE_BASE_URL"))
    parser.add_argument(
        "--expect-verdict",
        choices=["safe", "controversial", "unsafe"],
        default=None,
        help="Optional expected triage verdict.",
    )
    return parser.parse_args()


def _provider() -> str | None:
    if os.environ.get("NVIDIA_API_KEY"):
        return "nvidia"
    if os.environ.get("OPENROUTER_API_KEY"):
        return "openrouter"
    return None


def _synthetic_verdicts():
    from nemoguardian.schemas import ModelVerdict, VerdictLabel

    qwen_verdict = ModelVerdict(
        model_id="synthetic-qwen",
        verdict=VerdictLabel.CONTROVERSIAL,
        score=0.55,
        categories=[],
        reasoning="Fast guard flagged the message as borderline but did not emit categories.",
        latency_ms=4.0,
    )
    csr_verdict = ModelVerdict(
        model_id="synthetic-csr",
        verdict=VerdictLabel.UNSAFE,
        score=0.95,
        categories=["PII", "scam"],
        reasoning="The text includes a fake SSN pattern and a cash-DM solicitation.",
        latency_ms=800.0,
    )
    return qwen_verdict, csr_verdict


def run(args: argparse.Namespace) -> dict[str, Any]:
    from nemoguardian.bot.audit import redacted_excerpt, text_hash
    from nemoguardian.models.nemotron_triage import NemotronTriage

    provider = _provider()
    if provider is None:
        return {
            "ok": False,
            "error": "NVIDIA_API_KEY or OPENROUTER_API_KEY is required",
            "provider": None,
        }

    qwen_verdict, csr_verdict = _synthetic_verdicts()
    triage = NemotronTriage(base_url=args.base_url, model_name=args.model)
    verdict = triage.adjudicate(
        args.text,
        args.policy,
        qwen_verdict,
        csr_verdict,
    )

    evidence: dict[str, Any] = {
        "ok": verdict.error is None,
        "provider": provider,
        "base_url": triage.base_url,
        "model": triage.model_name,
        "text_sha256": text_hash(args.text),
        "text_excerpt": redacted_excerpt(args.text),
        "verdict": verdict.verdict.value,
        "score": verdict.score,
        "reasoning": verdict.reasoning,
        "latency_ms": verdict.latency_ms,
        "error": verdict.error,
    }
    if args.expect_verdict is not None and verdict.verdict.value != args.expect_verdict:
        evidence["ok"] = False
        evidence["expectation_error"] = (
            f"expected {args.expect_verdict}, got {verdict.verdict.value}"
        )
    return evidence


def main() -> int:
    evidence = run(parse_args())
    print(json.dumps(evidence, indent=2, sort_keys=True))
    if evidence["ok"]:
        return 0
    if evidence.get("provider") is None:
        return 13
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
