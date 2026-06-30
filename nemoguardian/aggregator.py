"""Multi-model aggregator.

Combines per-model verdicts into a single final verdict + reasons list + score.

Rules (default weights):
- Qwen3Guard-Gen: 0.40
- Qwen3Guard-Stream: 0.40 (fast mode document summary)
- Nemotron-CSR:   0.40
- Nemotron triage: 0.20 (only in DEEP mode, only when present)

Aggregation: weighted mean of scores → threshold → label.
- score < 0.30 → safe
- 0.30 ≤ score < 0.70 → controversial
- score ≥ 0.70 → unsafe

If ANY contributing model returned verdict=unsafe with score ≥ 0.50, we override
the weighted mean and label unsafe (fail-closed on hard violations).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nemoguardian.schemas import ModelVerdict, VerdictLabel


@dataclass
class AggregatorConfig:
    qwen_weight: float = 0.40
    csr_weight: float = 0.40
    triage_weight: float = 0.20
    injection_weight: float = 0.50  # deterministic prompt-injection heuristic
    safe_threshold: float = 0.30
    unsafe_threshold: float = 0.70
    override_on_unsafe: float = 0.50  # any model over this → unsafe regardless


@dataclass
class AggregatedVerdict:
    verdict: VerdictLabel
    score: float
    reasons: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    matched_policy_rule: str | None = None


def aggregate(
    verdicts: dict[str, ModelVerdict],
    *,
    config: AggregatorConfig | None = None,
) -> AggregatedVerdict:
    """Combine per-model verdicts.

    ``verdicts`` keys must be one of:
    - "qwen3_guard_gen" → Qwen3Guard-Gen (full doc)
    - "qwen3_guard_stream" → Qwen3Guard-Stream (fast mode)
    - "nemotron_csr"    → Nemotron Content Safety Reasoning
    - "triage"          → Nemotron 3 Ultra triage (deep mode)
    """
    cfg = config or AggregatorConfig()

    weighted_score = 0.0
    weighted_total = 0.0
    categories: list[str] = []
    reasons: list[str] = []

    mapping: dict[str, tuple[float, str]] = {
        "qwen3_guard_gen": (cfg.qwen_weight, "Qwen3Guard"),
        "qwen3_guard_stream": (cfg.qwen_weight, "Qwen3Guard-Stream"),
        "nemotron_csr": (cfg.csr_weight, "Nemotron-CSR"),
        "triage": (cfg.triage_weight, "Triage"),
        "prompt_injection": (cfg.injection_weight, "PromptInjection"),
    }

    # Hard override: any model with verdict=unsafe AND score ≥ override_on_unsafe
    override_to_unsafe = False
    for key, verdict in verdicts.items():
        if verdict.error:
            # Don't let a failed model push us to unsafe — but surface in reasons.
            reasons.append(f"{mapping.get(key, (0, key))[1]} failed: {verdict.error}")
            continue
        if verdict.verdict == VerdictLabel.UNSAFE and verdict.score >= cfg.override_on_unsafe:
            override_to_unsafe = True

    for key, (weight, label) in mapping.items():
        verdict = verdicts.get(key)
        if verdict is None or verdict.error:
            continue
        weighted_score += weight * verdict.score
        weighted_total += weight
        categories.extend(verdict.categories)
        if verdict.reasoning:
            # Truncate to keep reasons readable.
            snippet = verdict.reasoning.split("\n")[0][:160].strip()
            if snippet:
                reasons.append(f"[{label}] {snippet}")

    if verdicts and weighted_total == 0:
        # Models WERE run but every vote was errored/dropped (e.g. reasoning
        # models that truncated before emitting a label). Fail safe-by-design:
        # escalate to CONTROVERSIAL rather than silently returning "safe" (which
        # is what an empty weighted mean does). An empty ``verdicts`` dict is a
        # deliberate caller opt-out (all model toggles off) and is left as-is.
        reasons.append(
            "[NemoGuardian] No model produced a usable verdict — escalating to controversial."
        )
        return AggregatedVerdict(
            verdict=VerdictLabel.CONTROVERSIAL,
            score=0.5,
            reasons=reasons[:6],
            categories=[],
        )

    if weighted_total > 0:
        weighted_score = weighted_score / weighted_total

    # Determine verdict (strict boundaries so a perfectly mid score stays CONTROVERSIAL)
    if override_to_unsafe:
        verdict = VerdictLabel.UNSAFE
        weighted_score = max(weighted_score, 0.7)
    elif weighted_score > cfg.unsafe_threshold:
        verdict = VerdictLabel.UNSAFE
    elif weighted_score > cfg.safe_threshold:
        verdict = VerdictLabel.CONTROVERSIAL
    else:
        verdict = VerdictLabel.SAFE

    # Dedupe categories while preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for c in categories:
        if c and c not in seen:
            seen.add(c)
            deduped.append(c)

    return AggregatedVerdict(
        verdict=verdict,
        score=round(weighted_score, 4),
        reasons=reasons[:6],
        categories=deduped,
    )


__all__ = ["AggregatedVerdict", "AggregatorConfig", "aggregate"]
