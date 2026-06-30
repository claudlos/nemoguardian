#!/usr/bin/env python
"""Generate a judge-/operator-facing eval report: eval-report.json + .html.

This runs WITHOUT a GPU or network. It:
  1. loads the replayable SYNTHETIC corpus (tests/fixtures/eval_corpus.jsonl),
  2. runs the deterministic prompt-injection detector over the adversarial cases
     and scores injection recall + benign false-alarms,
  3. measures the deterministic layer's own latency and turns it into an
     ESTIMATED cost (clearly labeled, not a billed price),
  4. optionally folds in full-model cascade metrics from a prior GPU run
     (``scripts/eval_benchmark.py --json``), and
  5. writes a self-contained JSON + HTML report.

Usage:
    python scripts/eval_report.py
    python scripts/eval_report.py --out-dir build --cascade-json cascade.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nemoguardian.detectors import detect_prompt_injection
from nemoguardian.eval.adversarial import run_adversarial_suite
from nemoguardian.eval.dataset import load_benchmark
from nemoguardian.eval.metrics import summarize_latency_cost
from nemoguardian.eval.report import build_report, write_html, write_json

_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CORPUS = _ROOT / "tests" / "fixtures" / "eval_corpus.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", default=str(_DEFAULT_CORPUS))
    p.add_argument("--out-dir", default=".", help="Directory for eval-report.{json,html}.")
    p.add_argument(
        "--cascade-json",
        help="Optional JSON from scripts/eval_benchmark.py --json (a GPU run). "
        "Its 'cascade' block and any 'latencies_by_mode' are folded into the report.",
    )
    p.add_argument("--gpu-usd-per-hr", type=float, default=None)
    return p.parse_args(argv)


def _measure_detector_cost(cases, *, gpu_usd_per_hr: float | None) -> dict:
    """Time the deterministic detector per case → a real, GPU-free cost estimate."""
    latencies: list[float] = []
    for case in cases:
        start = time.perf_counter()
        detect_prompt_injection(case.text)
        latencies.append((time.perf_counter() - start) * 1000.0)
    kw = {} if gpu_usd_per_hr is None else {"gpu_usd_per_hr": gpu_usd_per_hr}
    return {"deterministic": summarize_latency_cost(latencies, **kw).as_dict()}


def _load_cascade(path: str, *, gpu_usd_per_hr: float | None) -> tuple[dict | None, dict]:
    """Pull cascade metrics + optional per-mode latencies from an eval_benchmark JSON."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    cascade = data.get("cascade")
    cost: dict = {}
    latencies_by_mode = data.get("latencies_by_mode") or {}
    kw = {} if gpu_usd_per_hr is None else {"gpu_usd_per_hr": gpu_usd_per_hr}
    for mode, lats in latencies_by_mode.items():
        cost[mode] = summarize_latency_cost(lats, **kw).as_dict()
    return cascade, cost


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = load_benchmark(args.corpus)
    adversarial = run_adversarial_suite(cases)
    cost_by_mode = _measure_detector_cost(cases, gpu_usd_per_hr=args.gpu_usd_per_hr)

    cascade = None
    if args.cascade_json:
        cascade, gpu_cost = _load_cascade(args.cascade_json, gpu_usd_per_hr=args.gpu_usd_per_hr)
        cost_by_mode.update(gpu_cost)

    report = build_report(
        cases, adversarial, cascade=cascade, cost_by_mode=cost_by_mode
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = write_json(report, out_dir / "eval-report.json")
    html_path = write_html(report, out_dir / "eval-report.html")

    det = adversarial.as_dict()["detection"]
    print(
        f"corpus={report['corpus']['n']} cases  "
        f"adversarial: recall={det['recall']} fpr={det['fpr']} "
        f"missed={adversarial.as_dict()['missed_injections']} "
        f"false_alarms={adversarial.as_dict()['false_alarms']}"
    )
    print(f"wrote {json_path}")
    print(f"wrote {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
