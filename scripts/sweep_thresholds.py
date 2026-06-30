#!/usr/bin/env python
"""Sweep aggregator thresholds over a captured predictions dump (GPU-free).

Decouples expensive model inference from cheap aggregator tuning: capture once
with ``eval_benchmark.py --dump-predictions dump.jsonl`` on a GPU host, then run
this offline as many times as you like to explore the precision/recall/FPR
frontier.

    python scripts/sweep_thresholds.py dump.jsonl

The dump's per-model scores are re-fed through the real ``aggregate()``, so the
swept numbers are exact, not approximated. A fidelity check confirms the default
config reproduces the verdict the cascade actually emitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nemoguardian.aggregator import AggregatorConfig, aggregate
from nemoguardian.eval.metrics import score_predictions
from nemoguardian.schemas import ModelVerdict, VerdictLabel

_FLAG_SETS = {
    "unsafe": frozenset({"unsafe"}),
    "unsafe+controversial": frozenset({"unsafe", "controversial"}),
}


def _verdicts_from_record(rec: dict) -> dict[str, ModelVerdict]:
    out: dict[str, ModelVerdict] = {}
    for key, mv in rec.get("models", {}).items():
        out[key] = ModelVerdict(
            model_id=key,
            verdict=VerdictLabel(mv["verdict"]),
            score=float(mv["score"]),
            latency_ms=0.0,
            error=mv.get("error"),
        )
    return out


def _evaluate(records: list[dict], cfg: AggregatorConfig, flag_on: frozenset[str]):
    rows = []
    for rec in records:
        agg = aggregate(_verdicts_from_record(rec), config=cfg)
        flagged = agg.verdict.value in flag_on
        rows.append((rec["gold"] == "unsafe", flagged, rec["category"]))
    return score_predictions(rows)


def _fidelity_check(records: list[dict]) -> int:
    """Re-aggregating with defaults must match the captured cascade verdict."""
    default = AggregatorConfig()
    mismatches = 0
    for rec in records:
        if "error" in rec and not rec.get("models"):
            continue  # case errored out entirely; aggregate of {} differs by design
        agg = aggregate(_verdicts_from_record(rec), config=default)
        if agg.verdict.value != rec.get("aggregate"):
            mismatches += 1
    return mismatches


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("dump", help="predictions JSONL from eval_benchmark.py --dump-predictions")
    p.add_argument(
        "--overrides",
        default="0.3,0.4,0.5,0.6,0.7",
        help="comma list of override_on_unsafe values to sweep",
    )
    args = p.parse_args(argv)

    records = [json.loads(line) for line in Path(args.dump).read_text().splitlines() if line.strip()]
    overrides = [float(x) for x in args.overrides.split(",")]

    mism = _fidelity_check(records)
    print(f"fidelity: default config reproduces captured verdict on "
          f"{len(records) - mism}/{len(records)} cases"
          + (f"  (!! {mism} mismatches)" if mism else "  ✓"))

    base = AggregatorConfig()
    print(f"\nbaseline thresholds: safe<{base.safe_threshold} | unsafe>{base.unsafe_threshold} | "
          f"override_on_unsafe>={base.override_on_unsafe}\n")

    # Frontier: override_on_unsafe x flag_on
    print(f"{'flag_on':<22}{'override':>9}{'recall':>9}{'FPR':>8}{'prec':>8}{'F1':>8}")
    best = None
    for flag_name, flag_on in _FLAG_SETS.items():
        for ov in overrides:
            cfg = replace(base, override_on_unsafe=ov)
            rep = _evaluate(records, cfg, flag_on)
            m = rep.overall
            print(f"{flag_name:<22}{ov:>9.2f}{m.recall:>9.3f}{m.fpr:>8.3f}{m.precision:>8.3f}{m.f1:>8.3f}")
            # "best" = highest recall while keeping FPR == 0
            if m.fpr == 0.0 and (best is None or m.recall > best[0]):
                best = (m.recall, flag_name, ov, rep)

    if best:
        recall, flag_name, ov, rep = best
        print(f"\nBest zero-FPR config: flag_on={flag_name}, override_on_unsafe={ov:.2f} "
              f"-> recall {recall:.3f}")
        print("  per-category recall:")
        for cat, r in sorted(rep.by_category.items()):
            d = r.as_dict()
            if d["recall"] is not None:
                print(f"    {cat:<22} {d['recall']:.3f}  (n={d['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
