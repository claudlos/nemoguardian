"""Pure metric computation for the eval harness (no model/torch deps).

Moderation is scored as a binary detection task: does the system FLAG a piece of
content (gold ``unsafe``) or let it through (gold ``safe``)? The caller decides
which verdicts count as "flagged" (e.g. just ``unsafe``, or ``unsafe`` +
``controversial``) and passes booleans here.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, field


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


@dataclass(frozen=True)
class Metrics:
    """Binary detection metrics. Positive class = unsafe/flagged."""

    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    @property
    def precision(self) -> float:
        return _safe_div(self.tp, self.tp + self.fp)

    @property
    def recall(self) -> float:
        """Detection rate on unsafe content (a.k.a. sensitivity / TPR)."""
        return _safe_div(self.tp, self.tp + self.fn)

    @property
    def fpr(self) -> float:
        """False-positive rate: benign content wrongly flagged."""
        return _safe_div(self.fp, self.fp + self.tn)

    @property
    def unsafe_recall(self) -> float:
        """Fraction of gold-unsafe content the system flags. Alias of ``recall``."""
        return self.recall

    @property
    def safe_precision(self) -> float:
        """Of everything the system let through, the fraction truly safe (NPV).

        The headline fail-safe metric: low safe precision == unsafe content
        slipping through as "safe".
        """
        return _safe_div(self.tn, self.tn + self.fn)

    @property
    def false_positives(self) -> int:
        """Benign content wrongly flagged (operator annoyance / over-blocking)."""
        return self.fp

    @property
    def false_negatives(self) -> int:
        """Unsafe content wrongly let through (the dangerous failure mode)."""
        return self.fn

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return _safe_div(2 * p * r, p + r)

    @property
    def accuracy(self) -> float:
        return _safe_div(self.tp + self.tn, self.n)

    def as_dict(self) -> dict[str, float | int]:
        return {
            "n": self.n,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
            "false_positives": self.fp,
            "false_negatives": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "unsafe_recall": round(self.unsafe_recall, 4),
            "safe_precision": round(self.safe_precision, 4),
            "fpr": round(self.fpr, 4),
            "f1": round(self.f1, 4),
            "accuracy": round(self.accuracy, 4),
        }


@dataclass(frozen=True)
class CategoryReport:
    category: str
    n: int
    n_unsafe: int
    n_safe: int
    metrics: Metrics

    def as_dict(self) -> dict:
        # For all-unsafe categories recall is the headline; for all-benign
        # categories FPR is. Surface both; callers pick.
        return {
            "category": self.category,
            "n": self.n,
            "n_unsafe": self.n_unsafe,
            "n_safe": self.n_safe,
            **self.metrics.as_dict(),
            # These come AFTER the spread so they override the raw metric values
            # with N/A where the category has no examples of that class.
            "recall": round(self.metrics.recall, 4) if self.n_unsafe else None,
            "fpr": round(self.metrics.fpr, 4) if self.n_safe else None,
        }


@dataclass(frozen=True)
class EvalReport:
    overall: Metrics
    by_category: dict[str, CategoryReport] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "overall": self.overall.as_dict(),
            "by_category": {k: v.as_dict() for k, v in sorted(self.by_category.items())},
        }


def score_predictions(rows: list[tuple[bool, bool, str]]) -> EvalReport:
    """Score ``(gold_unsafe, flagged, category)`` triples into an EvalReport."""
    overall = _confusion(((g, p) for g, p, _ in rows))
    by_category: dict[str, CategoryReport] = {}
    categories = sorted({c for _, _, c in rows})
    for cat in categories:
        cat_rows = [(g, p) for g, p, c in rows if c == cat]
        metrics = _confusion(iter(cat_rows))
        n_unsafe = sum(1 for g, _ in cat_rows if g)
        by_category[cat] = CategoryReport(
            category=cat,
            n=len(cat_rows),
            n_unsafe=n_unsafe,
            n_safe=len(cat_rows) - n_unsafe,
            metrics=metrics,
        )
    return EvalReport(overall=overall, by_category=by_category)


def _confusion(pairs) -> Metrics:
    tp = fp = tn = fn = 0
    for gold_unsafe, flagged in pairs:
        if gold_unsafe and flagged:
            tp += 1
        elif gold_unsafe and not flagged:
            fn += 1
        elif not gold_unsafe and flagged:
            fp += 1
        else:
            tn += 1
    return Metrics(tp=tp, fp=fp, tn=tn, fn=fn)


# --- Latency + (estimated) cost ------------------------------------------------
# An EXAMPLE community single-GPU spot rate, used only to turn measured wall-clock
# latency into a rough relative dollar figure. It is NOT a billed/quoted price and
# NOT a benchmark of any provider — override it for your own hardware.
DEFAULT_GPU_USD_PER_HR = 0.40


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Nearest-rank percentile of an already-sorted, non-empty list."""
    if not sorted_vals:
        return 0.0
    k = max(0, min(len(sorted_vals) - 1, math.ceil(pct / 100.0 * len(sorted_vals)) - 1))
    return sorted_vals[k]


@dataclass(frozen=True)
class LatencyCost:
    """Latency summary plus a ROUGH ESTIMATED cost for one mode/slice.

    ``est_cost_usd`` = total wall-clock time x ``gpu_usd_per_hr``. It is an
    ESTIMATE for relative comparison between modes, deliberately labeled as such;
    it is not a billed amount. Cases with no measured latency are ignored.
    """

    n: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float
    total_ms: float
    gpu_usd_per_hr: float
    est_cost_usd: float

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "n": self.n,
            "mean_ms": round(self.mean_ms, 1),
            "p50_ms": round(self.p50_ms, 1),
            "p95_ms": round(self.p95_ms, 1),
            "max_ms": round(self.max_ms, 1),
            "total_ms": round(self.total_ms, 1),
            "gpu_usd_per_hr": self.gpu_usd_per_hr,
            "est_cost_usd": round(self.est_cost_usd, 6),
            "est_cost_per_1k_usd": round(_safe_div(self.est_cost_usd, self.n) * 1000, 4),
            "cost_note": "ESTIMATE: latency x example GPU $/hr, not a billed price",
        }


def summarize_latency_cost(
    latencies_ms: Iterable[float],
    *,
    gpu_usd_per_hr: float = DEFAULT_GPU_USD_PER_HR,
) -> LatencyCost:
    """Summarize a set of per-case latencies into latency stats + estimated cost."""
    vals = sorted(float(v) for v in latencies_ms if v and v > 0)
    n = len(vals)
    total = sum(vals)
    mean = _safe_div(total, n)
    est_cost = total / 3_600_000.0 * gpu_usd_per_hr
    return LatencyCost(
        n=n,
        mean_ms=mean,
        p50_ms=_percentile(vals, 50),
        p95_ms=_percentile(vals, 95),
        max_ms=vals[-1] if vals else 0.0,
        total_ms=total,
        gpu_usd_per_hr=gpu_usd_per_hr,
        est_cost_usd=est_cost,
    )


def cost_by_mode(
    latencies_by_mode: dict[str, Iterable[float]],
    *,
    gpu_usd_per_hr: float = DEFAULT_GPU_USD_PER_HR,
) -> dict[str, LatencyCost]:
    """Per-mode (fast/standard/deep) latency + estimated-cost breakdown."""
    return {
        mode: summarize_latency_cost(vals, gpu_usd_per_hr=gpu_usd_per_hr)
        for mode, vals in latencies_by_mode.items()
    }


__all__ = [
    "DEFAULT_GPU_USD_PER_HR",
    "CategoryReport",
    "EvalReport",
    "LatencyCost",
    "Metrics",
    "cost_by_mode",
    "score_predictions",
    "summarize_latency_cost",
]
