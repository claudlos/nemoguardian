"""Human-in-the-loop review layer (audit TODOs #26, #27, #28).

A self-contained, file-backed layer that makes auto-enforcement trustworthy:

* :class:`ReviewQueue`  — queue controversial/flagged cases for a moderator (#26)
* :class:`AppealLog` / :class:`ReviewService` — appeal & restore flow (#27)
* :class:`StrikeLedger` / :class:`StrikePolicy` — strikes + progressive
  enforcement, ``warn -> timeout -> ban`` (#28)

All three stores reuse the append-only JSONL + dataclass + PII-redaction pattern
from :mod:`nemoguardian.bot.audit`; nothing here performs platform I/O.
"""

from __future__ import annotations

from nemoguardian.review.models import (
    AppealDecision,
    AppealOutcome,
    CaseStatus,
    EnforcementAction,
    ReviewCase,
    StrikeEvent,
)
from nemoguardian.review.service import AppealResult, ReviewService
from nemoguardian.review.store import AppealLog, ReviewQueue, StrikeLedger
from nemoguardian.review.strikes import StrikePolicy, default_policy

__all__ = [
    "AppealDecision",
    "AppealLog",
    "AppealOutcome",
    "AppealResult",
    "CaseStatus",
    "EnforcementAction",
    "ReviewCase",
    "ReviewQueue",
    "ReviewService",
    "StrikeEvent",
    "StrikeLedger",
    "StrikePolicy",
    "default_policy",
]
