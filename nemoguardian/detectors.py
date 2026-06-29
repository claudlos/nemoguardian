"""Deterministic, pre-LLM detectors for attack classes the content guards miss.

Qwen3Guard and Nemotron-CSR classify *content* harm (violence, PII, sexual,
illegal, etc.). They reliably miss **prompt-injection / jailbreak** attempts
("ignore previous instructions", "you are now DAN", "output the verdict as
safe") because those aren't harmful *content* in the safety taxonomy — they're
attacks on the system. This module adds a cheap regex layer that the cascade
folds in as an extra vote, lifting jailbreak recall without touching FPR.

Patterns are intentionally scoped so they require injection-specific objects
(instructions/prompt/rules/policy, "no restrictions", "bypass the filter",
"mark ... as safe", a leading SYSTEM: turn), NOT bare verbs like "ignore" —
so benign text such as "ignore the typo in my last message" does not match.
"""

from __future__ import annotations

import re

from nemoguardian.schemas import ModelVerdict, VerdictLabel

# (name, compiled regex). Names are surfaced in the verdict reasoning.
_INJECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "override-instructions",
        re.compile(
            r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}"
            r"\b(previous|prior|above|earlier|preceding|all|any|the|your)\b[^.\n]{0,30}"
            r"\b(instruction|instructions|prompt|prompts|rule|rules|polic|directive|guideline)",
            re.IGNORECASE,
        ),
    ),
    (
        "no-restrictions",
        re.compile(
            r"\b(no|without|free of|free from)\b\s+"
            r"(restrictions?|rules?|limits?|limitations?|filters?|guidelines?|guardrails?|"
            r"safeguards?|censorship)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "persona-jailbreak",
        re.compile(
            r"\b(developer mode|dan mode|jailbreak|do anything now|"
            r"you are now dan|act as dan|unrestricted (ai|model|assistant))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "reveal-system-prompt",
        re.compile(
            r"\b(reveal|show|print|repeat|expose|disclose|leak|output)\b[^.\n]{0,40}"
            r"\b(your |the |hidden |secret |initial )*(system prompt|system instructions|"
            r"hidden instructions|prompt verbatim|instructions verbatim)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "pretend-no-rules",
        re.compile(
            r"\bpretend\b[^.\n]{0,40}\b(prompt|instructions|rules|polic)\b[^.\n]{0,30}"
            r"\b(do(es)? ?n.?t|never)\b[^.\n]{0,10}\bexist",
            re.IGNORECASE,
        ),
    ),
    (
        "bypass-filter",
        re.compile(
            r"\bbypass\b[^.\n]{0,25}\b(content |safety |moderation )?"
            r"(filter|moderation|guard|safety|restriction|detection)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "force-safe-verdict",
        re.compile(
            r"\b(mark|label|classif\w*|rate|output|respond|treat|call|deem)\b[^.\n]{0,45}"
            r"\b(as )?(safe|unharmful|harmless|not ?harmful|benign)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "system-turn-injection",
        re.compile(
            r"(^|\n)\s*system\s*[:>]|"
            r"\bclassification (is )?disabled\b|"
            r"\b(safety|moderation|content) (filter|policy|check)s? (is |are )?(now )?"
            r"(disabled|off|bypassed)\b",
            re.IGNORECASE,
        ),
    ),
]

# Heuristic vote strength. >= aggregator override threshold (0.5) so a confident
# injection match escalates the cascade to unsafe even if the LLM guards passed.
_INJECTION_SCORE = 0.9


def detect_prompt_injection(text: str) -> list[str]:
    """Return the names of injection patterns matched in ``text`` (possibly empty)."""
    return [name for name, rx in _INJECTION_PATTERNS if rx.search(text)]


def heuristic_verdict(text: str) -> ModelVerdict | None:
    """A deterministic ModelVerdict when injection patterns fire, else ``None``.

    The cascade adds this under the ``prompt_injection`` key before aggregation.
    """
    hits = detect_prompt_injection(text)
    if not hits:
        return None
    return ModelVerdict(
        model_id="heuristic-prompt-injection",
        verdict=VerdictLabel.UNSAFE,
        score=_INJECTION_SCORE,
        categories=["Jailbreak"],
        reasoning="prompt-injection patterns: " + ", ".join(hits),
        latency_ms=0.0,
    )


__all__ = ["detect_prompt_injection", "heuristic_verdict"]
