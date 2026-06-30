"""Run a moderate callable over a benchmark and score it.

Decoupled from the cascade: the caller passes a ``predict`` callable mapping an
:class:`EvalCase` to a :class:`PredictResult`. ``scripts/eval_benchmark.py``
wires the real GPU cascade; tests pass a cheap stub.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from nemoguardian.eval.dataset import EvalCase
from nemoguardian.eval.metrics import EvalReport, score_predictions

# Default: only a hard ``unsafe`` verdict counts as a block/flag. Pass a wider
# set (e.g. {"unsafe", "controversial"}) to measure escalation behaviour.
DEFAULT_FLAG_ON = frozenset({"unsafe"})


@dataclass
class PredictResult:
    verdict: str  # "safe" | "controversial" | "unsafe"
    latency_ms: float = 0.0
    error: str | None = None


@dataclass
class Prediction:
    case_id: str
    category: str
    gold_unsafe: bool
    verdict: str
    flagged: bool
    latency_ms: float
    error: str | None = None


def run_eval(
    cases: Iterable[EvalCase],
    predict: Callable[[EvalCase], PredictResult],
    *,
    flag_on: Iterable[str] = DEFAULT_FLAG_ON,
) -> tuple[EvalReport, list[Prediction]]:
    """Run ``predict`` over ``cases`` and return ``(report, predictions)``."""
    flag_set = {v.lower() for v in flag_on}
    predictions: list[Prediction] = []
    for case in cases:
        result = predict(case)
        verdict = result.verdict.lower()
        predictions.append(
            Prediction(
                case_id=case.id,
                category=case.category,
                gold_unsafe=case.is_unsafe,
                verdict=verdict,
                flagged=verdict in flag_set,
                latency_ms=result.latency_ms,
                error=result.error,
            )
        )
    report = score_predictions(
        [(p.gold_unsafe, p.flagged, p.category) for p in predictions]
    )
    return report, predictions


def mean_latency_ms(predictions: list[Prediction]) -> float:
    vals = [p.latency_ms for p in predictions if p.latency_ms]
    return sum(vals) / len(vals) if vals else 0.0


__all__ = ["DEFAULT_FLAG_ON", "PredictResult", "Prediction", "mean_latency_ms", "run_eval"]
