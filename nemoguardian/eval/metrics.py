"""Pure metric computation for the eval harness (no model/torch deps).

Moderation is scored as a binary detection task: does the system FLAG a piece of
content (gold ``unsafe``) or let it through (gold ``safe``)? The caller decides
which verdicts count as "flagged" (e.g. just ``unsafe``, or ``unsafe`` +
``controversial``) and passes booleans here.
"""

from __future__ import annotations

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
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
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


__all__ = ["CategoryReport", "EvalReport", "Metrics", "score_predictions"]
