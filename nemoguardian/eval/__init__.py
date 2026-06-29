"""Evaluation harness for measuring moderation accuracy.

Provides:
- ``dataset``: load a labeled benchmark (JSONL) of moderation cases.
- ``metrics``: pure precision/recall/FPR/F1 computation, per-category and overall.
- ``runner``: run any moderate callable over the benchmark and score it.

The metric and runner logic is dependency-light and GPU-free so it can be unit
tested without loading models. ``scripts/eval_benchmark.py`` wires the real
cascade.
"""

from nemoguardian.eval.dataset import EvalCase, load_benchmark
from nemoguardian.eval.metrics import CategoryReport, EvalReport, Metrics, score_predictions
from nemoguardian.eval.runner import Prediction, run_eval

__all__ = [
    "CategoryReport",
    "EvalCase",
    "EvalReport",
    "Metrics",
    "Prediction",
    "load_benchmark",
    "run_eval",
    "score_predictions",
]
