"""Torch-free unit tests for model output parsing and revision pinning.

These import only the parse helpers and regexes — no weights, no torch.
"""

from __future__ import annotations

import pytest

from nemoguardian.models.base import resolve_revision
from nemoguardian.models.nemotron_csr import (
    _PROMPT_HARM_RE,
    _RESPONSE_HARM_RE,
    DEFAULT_CSR_REVISION,
    NemotronCSR,
)
from nemoguardian.models.qwen3_guard import (
    DEFAULT_GEN_REVISION,
    Qwen3GuardGen,
    _parse_gen_output,
)

# --- Qwen3Guard-Gen output parsing ------------------------------------------

def test_parse_gen_output_well_formed():
    out = _parse_gen_output(
        "Safety: Unsafe\nCategories: Violent, PII\nRefusal: No"
    )
    assert out["label"] == "Unsafe"
    assert "Violent" in out["categories"]
    assert "PII" in out["categories"]
    assert out["refusal"] == "No"


def test_parse_gen_output_safe_with_none_category():
    out = _parse_gen_output("Safety: Safe\nCategories: None\nRefusal: No")
    assert out["label"] == "Safe"
    assert out["categories"] == []


def test_parse_gen_output_controversial():
    out = _parse_gen_output("Safety: Controversial\nCategories: Politically Sensitive Topics")
    assert out["label"] == "Controversial"


@pytest.mark.parametrize(
    "drifted",
    [
        "",
        "I think this content is fine.",
        "safety: unsafe",  # wrong casing
        "SAFETY LEVEL: Unsafe",  # wrong field name
    ],
)
def test_parse_gen_output_raises_on_drift(drifted):
    """Format drift must NOT silently vote safe — it raises, which becomes an
    error verdict that the aggregator excludes."""
    with pytest.raises(ValueError, match="unparseable"):
        _parse_gen_output(drifted)


# --- Nemotron-CSR prompt building + harm regexes -----------------------------

def _csr(reasoning: bool = True) -> NemotronCSR:
    return NemotronCSR(reasoning=reasoning)


def test_csr_build_prompt_default_template_substitutes():
    p = _csr()._build_prompt("how do I pick a lock", None, None)
    assert "how do I pick a lock" in p
    assert "(no response provided)" in p
    assert "<USER_PROMPT>" not in p
    assert p.rstrip().endswith("/think")


def test_csr_build_prompt_reasoning_off_strips_think():
    p = _csr(reasoning=False)._build_prompt("hi", None, "hello")
    assert "/think" not in p
    assert "hello" in p


def test_csr_build_prompt_custom_policy():
    p = _csr()._build_prompt("text", "no memes", "resp")
    assert "SAFETY POLICY: no memes" in p
    assert "reason step by step" in p


def test_csr_harm_regexes_match_expected_format():
    content = "Reasoning...\nPrompt harm: harmful\nResponse Harm: unharmful"
    assert _PROMPT_HARM_RE.search(content).group(1) == "harmful"
    assert _RESPONSE_HARM_RE.search(content).group(1) == "unharmful"


def test_csr_harm_regexes_reject_drift():
    # Casing drift ("Prompt Harm:") must not match — _moderate_impl raises on
    # zero matches instead of defaulting safe.
    assert _PROMPT_HARM_RE.search("Prompt Harm: harmful") is None
    assert _RESPONSE_HARM_RE.search("response harm: harmful") is None


# --- Revision pinning ---------------------------------------------------------

def test_resolve_revision_default_model_gets_pin(monkeypatch):
    monkeypatch.delenv("NEMOGUARDIAN_QWEN_REVISION", raising=False)
    rev = resolve_revision(
        Qwen3GuardGen.DEFAULT_MODEL,
        Qwen3GuardGen.DEFAULT_MODEL,
        "NEMOGUARDIAN_QWEN_REVISION",
        DEFAULT_GEN_REVISION,
    )
    assert rev == DEFAULT_GEN_REVISION
    assert len(rev) == 40  # full commit SHA, not a floating ref


def test_resolve_revision_env_wins(monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_CSR_REVISION", "deadbeef")
    rev = resolve_revision(
        NemotronCSR.DEFAULT_MODEL,
        NemotronCSR.DEFAULT_MODEL,
        "NEMOGUARDIAN_CSR_REVISION",
        DEFAULT_CSR_REVISION,
    )
    assert rev == "deadbeef"


def test_resolve_revision_custom_model_floats(monkeypatch):
    monkeypatch.delenv("NEMOGUARDIAN_QWEN_REVISION", raising=False)
    rev = resolve_revision(
        "someone/custom-guard",
        Qwen3GuardGen.DEFAULT_MODEL,
        "NEMOGUARDIAN_QWEN_REVISION",
        DEFAULT_GEN_REVISION,
    )
    assert rev is None  # the pinned SHA belongs to the default checkpoint only
