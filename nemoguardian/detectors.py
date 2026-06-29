"""Deterministic, pre-LLM detectors for attack classes the content guards miss.

Qwen3Guard and Nemotron-CSR classify *content* harm (violence, PII, sexual,
illegal, etc.). They reliably miss **prompt-injection / jailbreak** attempts
("ignore previous instructions", "you are now DAN", "output the verdict as
safe") because those aren't harmful *content* in the safety taxonomy — they're
attacks on the system. This module adds a cheap deterministic layer that the
cascade folds in as an extra vote, lifting jailbreak recall without touching FPR.

Defense layers (all run on every input):
1. **Surface** — NFKC + zero-width/control strip + homoglyph fold + lowercase.
   Catches plain text and unicode/fullwidth/zero-width/homoglyph obfuscation.
2. **Collapsed** — alphanumeric-only, de-leetspeaked copy of the surface.
   Catches character-spacing ("i g n o r e"), punctuation-splitting and leetspeak.
3. **Decoded** — base64 segments and a rot13 pass are decoded and re-normalized,
   then scanned with the same patterns. Catches encoded payloads.
4. **Multilingual** — injection patterns for several Latin-script languages
   (es/fr/de/pt/it) run on the surface.

Patterns require an injection-specific object (instructions/prompt/rules/policy,
"no restrictions", "bypass the filter", "mark ... as safe", a leading SYSTEM:
turn), so benign text — "ignore the typo", "no limits on PTO", or the German
"ignoriere meinen Tippfehler" (ignore my typo) — does not match.
"""

from __future__ import annotations

import base64
import codecs
import contextlib
import re
import unicodedata

from nemoguardian.schemas import ModelVerdict, VerdictLabel

# --- Homoglyph fold: common Cyrillic/Greek look-alikes -> ASCII -----------------
_HOMOGLYPHS = str.maketrans(
    {
        # Cyrillic lowercase
        "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
        "і": "i", "ј": "j", "ѕ": "s", "к": "k", "м": "m", "н": "h", "т": "t",
        "в": "b", "г": "r", "ԁ": "d", "ո": "n", "հ": "h", "ѵ": "v", "ｇ": "g",
        # Cyrillic uppercase
        "А": "a", "Е": "e", "О": "o", "Р": "p", "С": "c", "У": "y", "Х": "x",
        "І": "i", "Ѕ": "s", "К": "k", "М": "m", "Н": "h", "Т": "t", "В": "b",
        # Greek lowercase
        "ο": "o", "ε": "e", "α": "a", "ρ": "p", "ν": "v", "ι": "i", "τ": "t",
        "υ": "u", "κ": "k", "χ": "x", "γ": "y", "ϲ": "c", "ѡ": "w",
    }
)

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})

# --- Layer 1: surface patterns (English) ---------------------------------------
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

# --- Multilingual injection (es/fr/de/pt/it), run on the surface ---------------
# verb (ignore/forget/disregard) within range of an instructions/rules noun.
_MULTILINGUAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "override-instructions-ml",
        re.compile(
            r"\b(ignor\w*|olvid\w*|oubli\w*|dimentica\w*|desconsider\w*|missachte\w*)\b"
            r"[^.\n]{0,45}"
            r"\b(instruc\w*|istruzion\w*|anweisung\w*|règles?|regeln?|regla\w*|regra\w*|"
            r"regola\w*|direktiv\w*|consign\w*)",
        ),
    ),
]

# --- Layer 2: collapsed patterns (alnum-only, de-leetspeaked) -------------------
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

_B64_RE = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")

# Heuristic vote strength. >= aggregator override threshold (0.5) so a confident
# injection match escalates the cascade to unsafe even if the LLM guards passed.
_INJECTION_SCORE = 0.9


def _surface_norm(text: str) -> str:
    """NFKC, strip zero-width/control (keep \\n\\t), fold homoglyphs, lowercase."""
    t = unicodedata.normalize("NFKC", text)
    t = "".join(
        ch for ch in t if ch in "\n\t" or unicodedata.category(ch) not in ("Cf", "Cc")
    )
    return t.translate(_HOMOGLYPHS).lower()


def _collapsed_norm(surface: str) -> str:
    """De-leetspeak then keep only [a-z0-9] (defeats spacing/punctuation splits)."""
    return re.sub(r"[^a-z0-9]+", "", surface.translate(_LEET))


def _decoded_variants(text: str, surface: str) -> list[str]:
    """Decoded payloads to additionally scan: base64 segments + a rot13 pass."""
    variants: list[str] = []
    # rot13 (letters only); reverses a rot13-encoded payload, gibberish otherwise.
    with contextlib.suppress(Exception):
        variants.append(_surface_norm(codecs.decode(surface, "rot13")))
    # base64 — decode from the RAW text (case-sensitive) and keep text-like results.
    for tok in _B64_RE.findall(text):
        try:
            raw = base64.b64decode(tok + "=" * (-len(tok) % 4), validate=False)
        except Exception:
            continue
        decoded = raw.decode("utf-8", "ignore")
        if sum(ch.isalpha() for ch in decoded) >= 6:
            variants.append(_surface_norm(decoded))
    return variants


def detect_prompt_injection(text: str) -> list[str]:
    """Return the names of injection patterns matched in ``text`` (deduped)."""
    surface = _surface_norm(text)
    scan = [surface, *_decoded_variants(text, surface)]
    hits: list[str] = []
    for s in scan:
        collapsed = _collapsed_norm(s)
        for name, rx in (*_SURFACE_PATTERNS, *_MULTILINGUAL_PATTERNS):
            if name not in hits and rx.search(s):
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
