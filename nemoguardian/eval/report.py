"""Build a judge-/operator-facing eval report (JSON + self-contained HTML).

Pure data assembly + string templating — NO external dependencies, no GPU, no
network. ``scripts/eval_report.py`` wires the real inputs; the rendering is unit
tested directly.

The report has three sections:
- **corpus**         : inventory of the replayable fixture (counts by label /
  bucket / category) so a reviewer can see exactly what was measured.
- **adversarial**    : deterministic prompt-injection detection results (recall
  on injections + false alarms on benign look-alikes), GPU-free.
- **cascade** / **cost** : OPTIONAL full-model quality metrics + per-mode latency
  and ESTIMATED cost, supplied only when a GPU run's outputs are passed in.
"""

from __future__ import annotations

import datetime as dt
import html
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from nemoguardian.eval.adversarial import AdversarialReport
from nemoguardian.eval.dataset import EvalCase

_SCHEMA_VERSION = 1


def corpus_summary(cases: Iterable[EvalCase]) -> dict:
    """Counts of the corpus by label, bucket, and fine category."""
    cases = list(cases)
    by_label: Counter[str] = Counter(c.label for c in cases)
    by_bucket: Counter[str] = Counter(c.bucket or "unspecified" for c in cases)
    by_category: Counter[str] = Counter(c.category for c in cases)
    return {
        "n": len(cases),
        "n_unsafe": sum(1 for c in cases if c.is_unsafe),
        "n_safe": sum(1 for c in cases if not c.is_unsafe),
        "by_label": dict(sorted(by_label.items())),
        "by_bucket": dict(sorted(by_bucket.items())),
        "by_category": dict(sorted(by_category.items())),
        "synthetic": True,
        "note": "All cases are hand-written SYNTHETIC data: no real PII, "
        "no real card/account numbers, no real slurs, no copyrighted text.",
    }


def build_report(
    cases: Iterable[EvalCase],
    adversarial: AdversarialReport,
    *,
    cascade: dict | None = None,
    cost_by_mode: dict | None = None,
    generated_at: str | None = None,
) -> dict:
    """Assemble the full report dict. ``cascade``/``cost_by_mode`` are optional."""
    return {
        "schema_version": _SCHEMA_VERSION,
        "generated_at": generated_at
        or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "corpus": corpus_summary(cases),
        "adversarial": adversarial.as_dict(),
        # Present only when a GPU run supplied them; kept explicit so the HTML can
        # honestly say "not run" instead of fabricating numbers.
        "cascade": cascade,
        "cost_by_mode": cost_by_mode,
    }


def write_json(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


# --- HTML rendering (self-contained, no external assets) -----------------------

_CSS = """
:root { color-scheme: light dark; }
body { font: 15px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 60rem;
       padding: 0 1rem; }
h1 { margin-bottom: .25rem; } h2 { margin-top: 2rem; border-bottom: 2px solid #8884;
       padding-bottom: .25rem; }
table { border-collapse: collapse; width: 100%; margin: .75rem 0; }
th, td { border: 1px solid #8884; padding: .35rem .6rem; text-align: left; }
th { background: #8881; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pill { display: inline-block; padding: .1rem .5rem; border-radius: 1rem;
        font-size: .8rem; background: #8882; }
.good { color: #1a7f37; font-weight: 600; } .bad { color: #b3261e; font-weight: 600; }
.note { background: #f5a6231a; border-left: 4px solid #f5a623; padding: .6rem .9rem;
        margin: 1rem 0; border-radius: 4px; }
.muted { color: #8889; } code { background: #8881; padding: 0 .25rem; border-radius: 3px; }
"""


def _esc(v: object) -> str:
    return html.escape(str(v))


def _kv_table(d: dict, *, key_hdr: str = "key", val_hdr: str = "value") -> str:
    rows = "".join(
        f"<tr><td>{_esc(k)}</td><td class='num'>{_esc(v)}</td></tr>"
        for k, v in d.items()
    )
    return f"<table><tr><th>{_esc(key_hdr)}</th><th>{_esc(val_hdr)}</th></tr>{rows}</table>"


def _detection_table(detection: dict) -> str:
    keys = [
        ("n", "cases"), ("tp", "caught injections (TP)"),
        ("fn", "MISSED injections (FN)"), ("fp", "benign flagged (FP)"),
        ("tn", "benign cleared (TN)"), ("recall", "injection recall"),
        ("fpr", "false-positive rate"),
    ]
    rows = "".join(
        f"<tr><td>{_esc(label)}</td><td class='num'>{_esc(detection.get(k))}</td></tr>"
        for k, label in keys
    )
    return f"<table><tr><th>metric</th><th>value</th></tr>{rows}</table>"


def _adversarial_detail(results: list[dict]) -> str:
    head = "<tr><th>id</th><th>category</th><th>expect</th><th>fired</th><th>hits</th></tr>"
    body = []
    for r in results:
        ok = r["correct"]
        cls = "good" if ok else "bad"
        fired = "yes" if r["fired"] else "no"
        body.append(
            f"<tr><td><code>{_esc(r['id'])}</code></td><td>{_esc(r['category'])}</td>"
            f"<td>{_esc(r['expect_inject'])}</td>"
            f"<td class='{cls}'>{_esc(fired)}</td>"
            f"<td class='muted'>{_esc(', '.join(r['hits']) or '—')}</td></tr>"
        )
    return f"<table>{head}{''.join(body)}</table>"


def _cascade_section(cascade: dict) -> str:
    overall = cascade.get("overall", {})
    head = (
        "<tr><th>metric</th><th>value</th></tr>"
    )
    keys = [
        ("n", "cases"), ("unsafe_recall", "unsafe recall"),
        ("safe_precision", "safe precision (NPV)"), ("precision", "precision"),
        ("false_negatives", "false negatives"), ("false_positives", "false positives"),
        ("fpr", "false-positive rate"), ("f1", "F1"), ("accuracy", "accuracy"),
    ]
    rows = "".join(
        f"<tr><td>{_esc(label)}</td><td class='num'>{_esc(overall.get(k))}</td></tr>"
        for k, label in keys
    )
    cat = cascade.get("by_category", {})
    cat_head = "<tr><th>category</th><th>n</th><th>recall</th><th>fpr</th></tr>"
    cat_rows = "".join(
        f"<tr><td>{_esc(c)}</td><td class='num'>{_esc(d.get('n'))}</td>"
        f"<td class='num'>{_esc(d.get('recall'))}</td>"
        f"<td class='num'>{_esc(d.get('fpr'))}</td></tr>"
        for c, d in cat.items()
    )
    return (
        "<h2>Cascade quality (full models)</h2>"
        f"<table>{head}{rows}</table>"
        f"<table>{cat_head}{cat_rows}</table>"
    )


def _cost_section(cost_by_mode: dict) -> str:
    head = (
        "<tr><th>mode</th><th>n</th><th>mean ms</th><th>p50</th><th>p95</th>"
        "<th>est $ /1k</th></tr>"
    )
    rows = "".join(
        f"<tr><td>{_esc(mode)}</td><td class='num'>{_esc(d.get('n'))}</td>"
        f"<td class='num'>{_esc(d.get('mean_ms'))}</td>"
        f"<td class='num'>{_esc(d.get('p50_ms'))}</td>"
        f"<td class='num'>{_esc(d.get('p95_ms'))}</td>"
        f"<td class='num'>{_esc(d.get('est_cost_per_1k_usd'))}</td></tr>"
        for mode, d in cost_by_mode.items()
    )
    return (
        "<h2>Latency &amp; estimated cost by mode</h2>"
        "<p class='muted'>Cost is an ESTIMATE: measured latency &times; an example "
        "GPU $/hr rate, for relative comparison only — not a billed price.</p>"
        f"<table>{head}{rows}</table>"
    )


def render_html(report: dict) -> str:
    """Render the report dict to a single self-contained HTML document."""
    corpus = report.get("corpus", {})
    adversarial = report.get("adversarial", {})
    detection = adversarial.get("detection", {})
    cascade = report.get("cascade")
    cost = report.get("cost_by_mode")

    missed = adversarial.get("missed_injections", [])
    false_alarms = adversarial.get("false_alarms", [])
    adv_verdict = (
        "<span class='good'>clean</span>"
        if not missed and not false_alarms
        else "<span class='bad'>see misses/false-alarms below</span>"
    )

    parts: list[str] = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>NemoGuardian eval report</title>",
        f"<style>{_CSS}</style></head><body>",
        "<h1>NemoGuardian eval report</h1>",
        f"<p class='muted'>generated {_esc(report.get('generated_at'))} &middot; "
        f"schema v{_esc(report.get('schema_version'))}</p>",
        "<div class='note'><strong>Synthetic data.</strong> "
        f"{_esc(corpus.get('note', ''))}</div>",
        "<h2>Corpus inventory</h2>",
        f"<p>{_esc(corpus.get('n'))} cases &mdash; "
        f"<span class='pill'>{_esc(corpus.get('n_unsafe'))} unsafe</span> "
        f"<span class='pill'>{_esc(corpus.get('n_safe'))} safe</span></p>",
        "<h3>By bucket</h3>", _kv_table(corpus.get("by_bucket", {}), key_hdr="bucket", val_hdr="n"),
        "<h3>By category</h3>", _kv_table(corpus.get("by_category", {}), key_hdr="category", val_hdr="n"),
        "<h2>Adversarial detector (deterministic, GPU-free)</h2>",
        f"<p>Status: {adv_verdict}</p>",
        _detection_table(detection),
        "<h3>Per-case</h3>",
        _adversarial_detail(adversarial.get("results", [])),
    ]

    if cascade:
        parts.append(_cascade_section(cascade))
    else:
        parts.append(
            "<h2>Cascade quality (full models)</h2>"
            "<p class='muted'>Not run in this report (needs a GPU host). "
            "Run <code>scripts/eval_benchmark.py</code> and pass its JSON to "
            "include full-model metrics here.</p>"
        )

    if cost:
        parts.append(_cost_section(cost))

    parts.append("</body></html>")
    return "".join(parts)


def write_html(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(render_html(report), encoding="utf-8")
    return path


__all__ = [
    "build_report",
    "corpus_summary",
    "render_html",
    "write_html",
    "write_json",
]
