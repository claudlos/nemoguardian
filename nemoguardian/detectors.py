"""Deterministic, pre-LLM detectors for attack classes the content guards miss.

Qwen3Guard and Nemotron-CSR classify *content* harm (violence, PII, sexual,
illegal, etc.). They reliably miss **prompt-injection / jailbreak** attempts
("ignore previous instructions", "you are now DAN", "output the verdict as
safe") because those aren't harmful *content* in the safety taxonomy — they're
attacks on the system. This module adds a cheap regex layer that the cascade
folds in as an extra vote, lifting jailbreak recall without touching FPR.

Two matching layers run against the input:
1. **Surface** patterns on a lightly-normalized copy (NFKC + zero-width/control
   stripped, lowercased). Catches plain text plus unicode/fullwidth/zero-width
   obfuscation with no FPR cost (same words).
2. **Collapsed** patterns on an alphanumeric-only, de-leetspeaked copy. Catches
   character-spacing ("i g n o r e"), punctuation-splitting ("i.g.n.o.r.e") and
   leetspeak ("1gn0re pr3vi0us"). Patterns still require an injection-specific
   object (instructions/prompt/rules/policy, "no restrictions", "bypass the
   filter", "mark ... as safe", a leading SYSTEM: turn), so benign text such as
   "ignore the typo" or "no limits on PTO" does not match.
"""

from __future__ import annotations

import re
import unicodedata

from nemoguardian.schemas import ModelVerdict, VerdictLabel

# --- Layer 1: surface patterns (run on the NFKC/lowercased copy) ---------------
_SURFACE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "override-instructions",
        re.compile(
            r"\b(ignore|disregard|forget|override|bypass)\b[^.\n]{0,40}"
            r"\b(previous|prior|above|earlier|preceding|all|any|the|your)\b[^.\n]{0,30}"
            r"\b(instruction|instructions|prompt|prompts|rule|rules|polic|directive|guideline)",
        ),
    ),
    (
        # "limits"/"limitations" deliberately excluded — benign ("no limits on PTO").
        "no-restrictions",
        re.compile(
            r"\b(no|without|free of|free from)\s+"
            r"(restrictions?|rules?|filters?|guidelines?|guardrails?|safeguards?|censorship)\b",
        ),
    ),
    (
        "persona-jailbreak",
        re.compile(
            r"\b(developer mode|dan mode|jailbreak|do anything now|"
            r"you are now dan|act as dan|unrestricted (ai|model|assistant))\b",
        ),
    ),
    (
        "reveal-system-prompt",
        re.compile(
            r"\b(reveal|show|print|repeat|expose|disclose|leak|output)\b[^.\n]{0,40}"
            r"\b(your |the |hidden |secret |initial )*(system prompt|system instructions|"
            r"hidden instructions|prompt verbatim|instructions verbatim)\b",
        ),
    ),
    (
        "pretend-no-rules",
        re.compile(
            r"\bpretend\b[^.\n]{0,40}\b(prompt|instructions|rules|polic)\b[^.\n]{0,30}"
            r"\b(do(es)? ?n.?t|never)\b[^.\n]{0,10}\bexist",
        ),
    ),
    (
        "bypass-filter",
        re.compile(
            r"\bbypass\b[^.\n]{0,25}\b(content |safety |moderation )?"
            r"(filter|moderation|guard|safety|restriction|detection)\b",
        ),
    ),
    (
        "force-safe-verdict",
        re.compile(
            r"\b(mark|label|classif\w*|rate|output|respond|treat|call|deem)\b[^.\n]{0,45}"
            r"\b(as )?(safe|unharmful|harmless|not ?harmful|benign)\b",
        ),
    ),
    (
        "system-turn-injection",
        re.compile(
            r"(^|\n)\s*system\s*[:>]|"
            r"\bclassification (is )?disabled\b|"
            r"\b(safety|moderation|content) (filter|policy|check)s? (is |are )?(now )?"
            r"(disabled|off|bypassed)\b",
        ),
    ),
]

# --- Layer 2: collapsed patterns (run on alnum-only, de-leetspeaked copy) -------
# No word boundaries; keywords joined by bounded alnum gaps. Each still requires
# an injection-specific object so benign collapsed text won't match.
_COLLAPSED_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "override-instructions",
        re.compile(
            r"(ignore|disregard|forget|override)\w{0,12}"
            r"(previous|prior|above|earlier|all|any|your)\w{0,12}"
            r"(instruction|prompt|rule|polic|directive|guideline)"
        ),
    ),
    ("no-restrictions", re.compile(r"no(restriction|rule|guardrail|safeguard|censorship)s?\b")),
    ("persona-jailbreak", re.compile(r"(youarenowdan|actasdan|danmode|developermode|jailbreakmode|doanythingnow)")),
    ("bypass-filter", re.compile(r"bypass\w{0,12}(filter|moderation|guard|safety|restriction|detection)")),
    (
        "reveal-system-prompt",
        re.compile(r"(reveal|show|print|repeat|expose|disclose|leak)\w{0,12}(systemprompt|systeminstruction|hiddeninstruction|promptverbatim|instructionsverbatim)"),
    ),
    (
        "force-safe-verdict",
        re.compile(r"(mark|label|classif|output|treat|rate)\w{0,18}(as)?(safe|unharmful|harmless|benign)"),
    ),
]

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})

# Heuristic vote strength. >= aggregator override threshold (0.5) so a confident
# injection match escalates the cascade to unsafe even if the LLM guards passed.
_INJECTION_SCORE = 0.9


def _surface_norm(text: str) -> str:
    """NFKC, strip zero-width/control (keep \\n\\t), lowercase."""
    t = unicodedata.normalize("NFKC", text)
    t = "".join(
        ch for ch in t if ch in "\n\t" or unicodedata.category(ch) not in ("Cf", "Cc")
    )
    return t.lower()


def _collapsed_norm(surface: str) -> str:
    """De-leetspeak then keep only [a-z0-9] (defeats spacing/punctuation splits)."""
    return re.sub(r"[^a-z0-9]+", "", surface.translate(_LEET))


def detect_prompt_injection(text: str) -> list[str]:
    """Return the names of injection patterns matched in ``text`` (deduped)."""
    surface = _surface_norm(text)
    collapsed = _collapsed_norm(surface)
    hits: list[str] = []
    for name, rx in _SURFACE_PATTERNS:
        if rx.search(surface):
            hits.append(name)
    for name, rx in _COLLAPSED_PATTERNS:
        if name not in hits and rx.search(collapsed):
            hits.append(name)
    return hits


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
