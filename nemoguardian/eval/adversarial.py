"""Adversarial prompt-injection / jailbreak eval over the DETERMINISTIC detector.

This suite exercises :func:`nemoguardian.detectors.detect_prompt_injection` on
the adversarial fixtures — it needs **no GPU and no model**, so it runs in plain
unit tests and in CI. It answers two judge-facing questions:

1. **Recall** — of the labeled injection attempts, how many does the cheap
   deterministic layer catch before the LLM guards even run?
2. **False positives** — of the benign look-alikes ("ignore the typo", "no
   limits on PTO"), how many does it wrongly flag? (Should be zero.)

Only cases that *declare* ``expect_inject`` participate: ``True`` rows are real
injections (positive class), ``False`` rows are benign look-alikes. Cases that
leave ``expect_inject`` unset are content-harm items the detector is not
responsible for, and are skipped here.

The model-dependent half of moderation (the LLM content guards) lives in the
cascade and is scored separately in ``scripts/eval_benchmark.py`` on a GPU host;
this module deliberately isolates the part whose metric logic is GPU-free.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from nemoguardian.detectors import detect_prompt_injection
from nemoguardian.eval.dataset import EvalCase
from nemoguardian.eval.metrics import Metrics

# Default detector: the real deterministic injection scanner. Injectable so the
# metric logic can be unit-tested against a stub without importing model code.
Detector = Callable[[str], list[str]]


@dataclass(frozen=True)
class AdversarialResult:
    case_id: str
    category: str
    expect_inject: bool
    fired: bool
    hits: list[str] = field(default_factory=list)

    @property
    def correct(self) -> bool:
        return self.fired == self.expect_inject

    def as_dict(self) -> dict:
        return {
            "id": self.case_id,
            "category": self.category,
            "expect_inject": self.expect_inject,
            "fired": self.fired,
            "hits": list(self.hits),
            "correct": self.correct,
        }


@dataclass(frozen=True)
class AdversarialReport:
    """Confusion + per-case detail for the deterministic adversarial suite.

    Positive class = "an injection the detector is expected to catch". So:
    - ``tp`` caught injections, ``fn`` MISSED injections (the recall failures),
    - ``fp`` benign look-alikes wrongly flagged, ``tn`` benign correctly cleared.
    """

    metrics: Metrics
    results: list[AdversarialResult]

    @property
    def missed(self) -> list[AdversarialResult]:
        return [r for r in self.results if r.expect_inject and not r.fired]

    @property
    def false_alarms(self) -> list[AdversarialResult]:
        return [r for r in self.results if not r.expect_inject and r.fired]

    def as_dict(self) -> dict:
        return {
            "n": len(self.results),
            "detection": self.metrics.as_dict(),
            "missed_injections": [r.case_id for r in self.missed],
            "false_alarms": [r.case_id for r in self.false_alarms],
            "results": [r.as_dict() for r in self.results],
        }


def run_adversarial_suite(
    cases: Iterable[EvalCase],
    *,
    detector: Detector = detect_prompt_injection,
) -> AdversarialReport:
    """Run the deterministic detector over adversarial-labeled cases.

    Cases whose ``expect_inject`` is ``None`` are skipped (not the detector's
    job). Returns a confusion report plus per-case results.
    """
    results: list[AdversarialResult] = []
    tp = fp = tn = fn = 0
    for case in cases:
        if case.expect_inject is None:
            continue
        hits = detector(case.text)
        fired = bool(hits)
        results.append(
            AdversarialResult(
                case_id=case.id,
                category=case.category,
                expect_inject=case.expect_inject,
                fired=fired,
                hits=list(hits),
            )
        )
        if case.expect_inject and fired:
            tp += 1
        elif case.expect_inject and not fired:
            fn += 1
        elif not case.expect_inject and fired:
            fp += 1
        else:
            tn += 1
    return AdversarialReport(metrics=Metrics(tp=tp, fp=fp, tn=tn, fn=fn), results=results)


__all__ = [
    "AdversarialReport",
    "AdversarialResult",
    "Detector",
    "run_adversarial_suite",
]
