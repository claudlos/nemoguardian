#!/usr/bin/env python
"""Evaluate the moderation cascade against a labeled benchmark.

Reports per-category and overall precision / recall / FPR / F1 for the aggregate
cascade verdict AND for each individual model, plus mean latency. This is the
quantitative gate the project was missing — run it to tune thresholds/weights
and to catch fail-open regressions.

Usage (on a GPU host, after `make`-style setup):

    python scripts/eval_benchmark.py --mode standard
    python scripts/eval_benchmark.py --mode deep --flag-on unsafe,controversial --json

Runs the real models, so use a GPU host. The metric logic itself is unit-tested
in tests/test_eval.py without a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nemoguardian.eval.dataset import EvalCase, load_benchmark
from nemoguardian.eval.metrics import score_predictions
from nemoguardian.eval.runner import (
    DEFAULT_FLAG_ON,
    PredictResult,
    mean_latency_ms,
    run_eval,
)

_DEFAULT_BENCHMARK = Path(__file__).resolve().parent.parent / "data" / "benchmark.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--benchmark", default=str(_DEFAULT_BENCHMARK))
    p.add_argument("--mode", choices=["fast", "standard", "deep"], default="standard")
    p.add_argument(
        "--flag-on",
        default="unsafe",
        help="Comma-separated verdicts that count as a block (e.g. 'unsafe,controversial').",
    )
    p.add_argument("--limit", type=int, default=0, help="Evaluate only the first N cases.")
    p.add_argument("--json", action="store_true", help="Emit a JSON report instead of a table.")
    p.add_argument(
        "--dump-predictions",
        metavar="PATH",
        help="Write per-case per-model verdicts+scores to JSONL for offline "
        "aggregator-threshold sweeps (see scripts/sweep_thresholds.py).",
    )
    return p.parse_args(argv)


def build_cascade_predict(mode: str):
    """Return ``(predict, per_model_collector, dump_records)`` backed by the cascade."""
    from nemoguardian.cascade import Cascade, CascadeConfig
    from nemoguardian.schemas import Mode, ModerateRequest

    cascade = Cascade(CascadeConfig.from_env())
    per_model_verdicts: dict[str, list[tuple[bool, str, str]]] = {}
    dump_records: list[dict] = []

    def predict(case: EvalCase) -> PredictResult:
        start = time.perf_counter()
        try:
            resp = cascade.moderate(
                ModerateRequest(text=case.text, policy=case.policy, mode=Mode(mode))
            )
        except Exception as exc:  # never let one bad case abort the sweep
            dump_records.append(
                {"id": case.id, "category": case.category, "gold": case.label,
                 "aggregate": "controversial", "error": str(exc), "models": {}}
            )
            return PredictResult(verdict="controversial", latency_ms=0.0, error=str(exc))
        latency = (time.perf_counter() - start) * 1000.0
        models = {
            key: {"verdict": mv.verdict.value, "score": mv.score, "error": mv.error}
            for key, mv in resp.model_verdicts.items()
        }
        dump_records.append(
            {"id": case.id, "category": case.category, "gold": case.label,
             "aggregate": resp.verdict.value, "models": models}
        )
        for key, mv in resp.model_verdicts.items():
            if mv.error:
                continue
            per_model_verdicts.setdefault(key, []).append(
                (case.is_unsafe, mv.verdict.value, case.category)
            )
        return PredictResult(verdict=resp.verdict.value, latency_ms=latency)

    return predict, per_model_verdicts, dump_records


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    flag_on = {v.strip().lower() for v in args.flag_on.split(",") if v.strip()} or DEFAULT_FLAG_ON
    cases = load_benchmark(args.benchmark)
    if args.limit:
        cases = cases[: args.limit]

    predict, per_model_verdicts, dump_records = build_cascade_predict(args.mode)
    report, predictions = run_eval(cases, predict, flag_on=flag_on)

    if args.dump_predictions:
        with open(args.dump_predictions, "w", encoding="utf-8") as fh:
            for rec in dump_records:
                fh.write(json.dumps(rec) + "\n")

    # Per-model reports (scored only where the model produced a usable verdict).
    per_model_reports = {}
    for key, rows in per_model_verdicts.items():
        scored = [(g, v in flag_on, c) for g, v, c in rows]
        per_model_reports[key] = score_predictions(scored)

    errors = [p for p in predictions if p.error]
    out = {
        "benchmark": args.benchmark,
        "mode": args.mode,
        "flag_on": sorted(flag_on),
        "n_cases": len(cases),
        "mean_latency_ms": round(mean_latency_ms(predictions), 1),
        "errors": len(errors),
        "cascade": report.as_dict(),
        "per_model": {k: v.as_dict() for k, v in per_model_reports.items()},
    }

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        _print_table(out, report, per_model_reports)
    return 0


def _print_table(out: dict, report, per_model_reports) -> None:
    o = report.overall
    print(f"\n=== NemoGuardian eval — mode={out['mode']} flag_on={out['flag_on']} ===")
    print(f"cases={out['n_cases']}  mean_latency={out['mean_latency_ms']}ms  errors={out['errors']}")
    print(
        f"\nCASCADE  precision={o.precision:.3f}  recall={o.recall:.3f}  "
        f"FPR={o.fpr:.3f}  F1={o.f1:.3f}  acc={o.accuracy:.3f}"
    )
    print(f"  {'category':<24}{'n':>4}{'recall':>9}{'fpr':>8}")
    for cat, rep in sorted(report.by_category.items()):
        d = rep.as_dict()
        recall = f"{d['recall']:.3f}" if d["recall"] is not None else "  -  "
        fpr = f"{d['fpr']:.3f}" if d["fpr"] is not None else "  -  "
        print(f"  {cat:<24}{d['n']:>4}{recall:>9}{fpr:>8}")
    for key, rep in sorted(per_model_reports.items()):
        m = rep.overall
        print(
            f"\n{key:<16} precision={m.precision:.3f}  recall={m.recall:.3f}  "
            f"FPR={m.fpr:.3f}  F1={m.f1:.3f}  (n={m.n})"
        )


if __name__ == "__main__":
    raise SystemExit(main())
