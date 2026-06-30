"""Benchmark dataset loading for the eval harness.

A benchmark is a JSONL file; one moderation case per line:

    {"id": "pii-01", "text": "...", "label": "unsafe", "category": "pii",
     "policy": "block PII and scams"}

- ``label``    : gold binary truth — "unsafe" or "safe".
- ``category`` : grouping for per-category metrics (e.g. pii, scam, toxicity,
                 jailbreak, violence, benign, benign_hard_negative).
- ``policy``   : optional custom policy passed to the cascade for this case.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

_GOLD_LABELS = {"safe", "unsafe"}


@dataclass(frozen=True)
class EvalCase:
    id: str
    text: str
    label: str  # "safe" | "unsafe"  (gold binary truth)
    category: str
    policy: str | None = None
    # Coarse triage bucket (descriptive only; scoring still uses ``label``):
    # "safe" | "unsafe" | "controversial" | "adversarial". Optional.
    bucket: str | None = None
    # When set, asserts whether the DETERMINISTIC prompt-injection detector
    # should fire on this case: ``True`` for real injections, ``False`` for
    # benign look-alikes. ``None`` means the deterministic layer is not the
    # responsible party (a content guard's job) and the adversarial suite skips it.
    expect_inject: bool | None = None

    @property
    def is_unsafe(self) -> bool:
        return self.label == "unsafe"


def load_benchmark(path: str | Path) -> list[EvalCase]:
    """Load and validate a JSONL benchmark file."""
    path = Path(path)
    cases: list[EvalCase] = []
    seen_ids: set[str] = set()
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
        case = _parse_row(row, path, lineno)
        if case.id in seen_ids:
            raise ValueError(f"{path}:{lineno}: duplicate case id {case.id!r}")
        seen_ids.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError(f"{path}: no cases found")
    return cases


def _parse_row(row: dict, path: Path, lineno: int) -> EvalCase:
    for field in ("id", "text", "label", "category"):
        if field not in row:
            raise ValueError(f"{path}:{lineno}: missing required field {field!r}")
    label = str(row["label"]).lower()
    if label not in _GOLD_LABELS:
        raise ValueError(
            f"{path}:{lineno}: label must be one of {sorted(_GOLD_LABELS)}, got {row['label']!r}"
        )
    expect_inject = row.get("expect_inject")
    if expect_inject is not None and not isinstance(expect_inject, bool):
        raise ValueError(
            f"{path}:{lineno}: expect_inject must be a bool, got {expect_inject!r}"
        )
    return EvalCase(
        id=str(row["id"]),
        text=str(row["text"]),
        label=label,
        category=str(row["category"]),
        policy=row.get("policy"),
        bucket=str(row["bucket"]) if row.get("bucket") is not None else None,
        expect_inject=expect_inject,
    )


__all__ = ["EvalCase", "load_benchmark"]
