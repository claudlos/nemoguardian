"""Policy tests — fast, no model loading."""

from __future__ import annotations

from nemoguardian.cascade import Cascade, CascadeConfig
from nemoguardian.policy.nemoclaw import NemoclawPolicy
from nemoguardian.policy.presets import get_preset
from nemoguardian.schemas import ModelVerdict, ModerateRequest, VerdictLabel


def test_discord_preset_blocks_pii():
    policy = get_preset("discord")
    decision = policy.evaluate(
        verdict=VerdictLabel.SAFE,  # model says safe
        score=0.1,
        categories=["PII"],  # but PII detected
    )
    assert decision.final_label == VerdictLabel.UNSAFE
    assert decision.matched_rule == "force-block-pii"


def test_twitch_preset_softens_violent():
    policy = get_preset("twitch")
    decision = policy.evaluate(
        verdict=VerdictLabel.UNSAFE,
        score=0.4,
        categories=["Violent"],
    )
    assert decision.final_label == VerdictLabel.CONTROVERSIAL
    assert decision.matched_rule == "soften-violent-tw"


def test_generic_preset_no_rules():
    policy = get_preset("generic")
    decision = policy.evaluate(verdict=VerdictLabel.SAFE, score=0.5, categories=[])
    assert decision.matched_rule is None
    assert decision.final_label is None


def test_policy_from_yaml(tmp_path):
    yaml_text = """
name: custom-test
rules:
  - id: my-rule
    when:
      score_above: 0.5
    then:
      final_label: unsafe
"""
    path = tmp_path / "policy.yaml"
    path.write_text(yaml_text)
    policy = NemoclawPolicy.from_yaml(path)
    decision = policy.evaluate(verdict=VerdictLabel.SAFE, score=0.6, categories=[])
    assert decision.matched_rule == "my-rule"
    assert decision.final_label == VerdictLabel.UNSAFE


def test_policy_first_match_wins():
    policy = NemoclawPolicy.from_dict(
        {
            "name": "ordered",
            "rules": [
                {"id": "first", "when": {"categories_include": ["PII"]}, "then": {"final_label": "unsafe"}},
                {"id": "second", "when": {"categories_include": ["PII"]}, "then": {"final_label": "controversial"}},
            ],
        }
    )
    decision = policy.evaluate(verdict=VerdictLabel.SAFE, score=0.1, categories=["PII"])
    assert decision.matched_rule == "first"


def test_policy_with_policy_text():
    policy = NemoclawPolicy.from_dict(
        {
            "name": "financial",
            "rules": [
                {
                    "id": "no-financial",
                    "when": {"policy_text_contains": "no financial", "model_verdict": "controversial"},
                    "then": {"final_label": "unsafe"},
                },
            ],
        }
    )
    decision = policy.evaluate(
        verdict=VerdictLabel.CONTROVERSIAL,
        score=0.5,
        categories=[],
        policy_text="Please no financial advice in this chat.",
    )
    assert decision.matched_rule == "no-financial"


def test_policy_text_condition_requires_policy_text():
    policy = NemoclawPolicy.from_dict(
        {
            "name": "financial",
            "rules": [
                {
                    "id": "no-financial",
                    "when": {"policy_text_contains": "no financial", "model_verdict": "controversial"},
                    "then": {"final_label": "unsafe"},
                },
            ],
        }
    )
    decision = policy.evaluate(
        verdict=VerdictLabel.CONTROVERSIAL,
        score=0.5,
        categories=[],
    )
    assert decision.matched_rule is None


def test_cascade_passes_policy_text_to_policy_gate():
    class StaticModel:
        def __init__(self, verdict: VerdictLabel, score: float) -> None:
            self.verdict = verdict
            self.score = score

        def moderate(self, text: str, *, policy: str | None = None):
            return ModelVerdict(
                model_id="static",
                verdict=self.verdict,
                score=self.score,
                latency_ms=1.0,
            )

    cascade = Cascade(CascadeConfig(enable_triage=False))
    cascade._qwen_gen = StaticModel(VerdictLabel.SAFE, 0.2)
    cascade._csr = StaticModel(VerdictLabel.CONTROVERSIAL, 0.6)
    policy = NemoclawPolicy.from_dict(
        {
            "name": "financial",
            "rules": [
                {
                    "id": "no-financial",
                    "when": {"policy_text_contains": "no financial", "model_verdict": "controversial"},
                    "then": {"final_label": "unsafe", "final_score": 0.85},
                },
            ],
        }
    )

    result = cascade.moderate(
        ModerateRequest(text="Should I buy this stock?", policy="no financial advice"),
        policy_engine=policy,
    )

    assert result.verdict == VerdictLabel.UNSAFE
    assert result.score == 0.85
    assert result.matched_policy_rule == "no-financial"
