"""Console demo — type into stdin, see the cascade verdict.

Run without GPU by setting NEMOGUARDIAN_DEMO_MOCK=1 to use fake verdicts.

Usage:
    NEMOGUARDIAN_DEMO_MOCK=1 python examples/console_demo.py
    # Or with real models (downloads ~10GB first time):
    python examples/console_demo.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nemoguardian.aggregator import aggregate
from nemoguardian.cascade import Cascade, CascadeConfig
from nemoguardian.policy.presets import get_preset
from nemoguardian.schemas import (
    Mode,
    ModelVerdict,
    ModerateRequest,
    VerdictLabel,
)

CACHE_PATH = Path("/tmp/nemoguardian_demo_cache.jsonl")


def mock_cascade(text: str, policy: str | None, mode: Mode) -> ModelVerdict:
    """Cheap deterministic mock for offline demo runs."""
    text_l = text.lower()
    if "ssn" in text_l or "social security" in text_l or "@everyone" in text_l:
        return ModelVerdict(
            model_id="mock",
            verdict=VerdictLabel.UNSAFE,
            score=0.95,
            categories=["PII"],
            reasoning=f"PII pattern detected in: {text[:60]}",
            latency_ms=15.0,
        )
    if "buy" in text_l and "stock" in text_l:
        return ModelVerdict(
            model_id="mock",
            verdict=VerdictLabel.UNSAFE,
            score=0.85,
            categories=[],
            reasoning="Financial-advice pattern (custom policy)",
            latency_ms=22.0,
        )
    if "violence" in text_l or "kill" in text_l:
        return ModelVerdict(
            model_id="mock",
            verdict=VerdictLabel.CONTROVERSIAL,
            score=0.55,
            categories=["Violent"],
            reasoning="Mild violent language; flagged for review",
            latency_ms=18.0,
        )
    return ModelVerdict(
        model_id="mock",
        verdict=VerdictLabel.SAFE,
        score=0.05,
        categories=[],
        reasoning="No policy violation detected",
        latency_ms=12.0,
    )


def run() -> None:
    policy_engine = get_preset("discord")
    cascade: Cascade | None = None
    if "NEMOGUARDIAN_DEMO_MOCK" not in os.environ:
        try:
            cascade = Cascade(CascadeConfig.from_env())
            print("[nemoguardian] loading models (first run downloads ~10GB)...")
            # Force eager load so the user sees status, not a hang on first text.
            cascade.qwen_gen.ensure_loaded()
            cascade.csr.ensure_loaded()
            print("[nemoguardian] models loaded.\n")
        except Exception as exc:
            print(f"[nemoguardian] could not load real models: {exc}")
            print("[nemoguardian] falling back to mock verdicts.")
            cascade = None

    print("=" * 60)
    print(" nemoguardian — interactive demo")
    print(" Type a message, press Enter to see the verdict.")
    print(" Commands: :policy <text>  |  :preset <name>  |  :quit")
    print("=" * 60 + "\n")

    active_policy: str | None = "block PII and financial scams"
    active_preset = "discord"

    while True:
        try:
            user_text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return

        if not user_text:
            continue
        if user_text in {":quit", ":q", ":exit"}:
            print("bye.")
            return
        if user_text.startswith(":policy"):
            active_policy = user_text.split(" ", 1)[1] if " " in user_text else None
            print(f"  (policy: {active_policy!r})\n")
            continue
        if user_text.startswith(":preset"):
            active_preset = user_text.split(" ", 1)[1].strip()
            try:
                policy_engine = get_preset(active_preset)
            except KeyError as exc:
                print(f"  ! {exc}")
                continue
            print(f"  (preset: {active_preset!r})\n")
            continue

        start = time.perf_counter()
        if cascade is not None:
            request = ModerateRequest(text=user_text, policy=active_policy, mode=Mode.STANDARD)
            result = cascade.moderate(request, policy_engine=policy_engine)
            verdict = result.verdict
            score = result.score
            categories = result.categories
            reasons = result.reasons
            models = result.model_verdicts
        else:
            qwen = mock_cascade(user_text, active_policy, Mode.STANDARD)
            csr = mock_cascade(user_text, active_policy, Mode.STANDARD)
            agg = aggregate({"qwen3_guard_gen": qwen, "nemotron_csr": csr})
            decision = policy_engine.evaluate(
                verdict=agg.verdict,
                score=agg.score,
                categories=agg.categories,
                policy_text=active_policy,
            )
            verdict = decision.final_label or agg.verdict
            score = agg.score
            categories = agg.categories
            reasons = agg.reasons
            models = {"qwen3_guard_gen": qwen, "nemotron_csr": csr}
        elapsed = (time.perf_counter() - start) * 1000.0

        # Color the verdict
        color = {"safe": "\033[32m", "controversial": "\033[33m", "unsafe": "\033[31m"}.get(verdict.value, "")
        reset = "\033[0m"

        print(f"  verdict: {color}{verdict.value.upper()}{reset}  score={score:.2f}  ({elapsed:.0f}ms)")
        if categories:
            print(f"  categories: {', '.join(categories)}")
        for r in reasons[:3]:
            print(f"    · {r}")
        if isinstance(models, dict):
            for k, v in models.items():
                latency = getattr(v, "latency_ms", 0.0)
                print(f"    [{k}] {v.verdict.value} ({v.score:.2f}) {latency:.0f}ms")
        print()


if __name__ == "__main__":
    run()
