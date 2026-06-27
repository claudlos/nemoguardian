"""NemoClaw-style policy gate.

A YAML policy file declares per-platform rules. The gate runs AFTER the cascade
and may:
- Override the verdict (e.g., force-block on PII regardless of model score)
- Tag a verdict with the matched rule for audit
- Apply platform-specific thresholds

Example policy file:

```yaml
name: discord-default
description: Default Discord server policy
rules:
  - id: force-block-pii
    when:
      categories_include: ["PII"]
    then:
      final_label: unsafe
      final_score: 1.0
  - id: soften-financial-advice
    when:
      policy_text_contains: "no financial advice"
      model_verdict: controversial
    then:
      final_label: unsafe
      final_score: 0.85
  - id: allow-low-confidence-on-mature-themes
    when:
      categories_include: ["Sexual Content or Sexual Acts"]
      score_below: 0.6
    then:
      final_label: controversial
```
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from nemoguardian.schemas import VerdictLabel


@dataclass
class PolicyRule:
    id: str
    when: dict[str, Any]
    then: dict[str, Any]


@dataclass
class PolicyDecision:
    matched_rule: str | None
    final_label: VerdictLabel | None = None
    final_score: float | None = None


@dataclass
class NemoclawPolicy:
    name: str
    description: str = ""
    rules: list[PolicyRule] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> NemoclawPolicy:
        with Path(path).open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NemoclawPolicy:
        rules: list[PolicyRule] = []
        for raw in data.get("rules", []) or []:
            rules.append(
                PolicyRule(
                    id=str(raw.get("id") or f"rule-{len(rules)}"),
                    when=dict(raw.get("when") or {}),
                    then=dict(raw.get("then") or {}),
                )
            )
        return cls(
            name=str(data.get("name", "unnamed")),
            description=str(data.get("description", "")),
            rules=rules,
        )

    def evaluate(
        self,
        *,
        verdict: VerdictLabel,
        score: float,
        categories: list[str],
        policy_text: str | None = None,
    ) -> PolicyDecision:
        """Apply each rule in order; first match wins."""
        for rule in self.rules:
            if self._matches(rule, verdict=verdict, score=score, categories=categories, policy_text=policy_text):
                then = rule.then
                label = then.get("final_label")
                final_label = VerdictLabel(label) if isinstance(label, str) else None
                final_score = then.get("final_score")
                if final_score is not None:
                    final_score = float(final_score)
                return PolicyDecision(
                    matched_rule=rule.id,
                    final_label=final_label,
                    final_score=final_score,
                )
        return PolicyDecision(matched_rule=None)

    @staticmethod
    def _matches(
        rule: PolicyRule,
        *,
        verdict: VerdictLabel,
        score: float,
        categories: list[str],
        policy_text: str | None,
    ) -> bool:
        cond = rule.when
        if "categories_include" in cond:
            needed = set(cond["categories_include"])
            if not needed.intersection(categories):
                return False
        if "model_verdict" in cond and str(cond["model_verdict"]).lower() != verdict.value:
            return False
        if "score_above" in cond and score <= float(cond["score_above"]):
            return False
        if "score_below" in cond and score >= float(cond["score_below"]):
            return False
        if "policy_text_contains" in cond:
            if not policy_text:
                return False
            needle = str(cond["policy_text_contains"]).lower()
            if needle not in policy_text.lower():
                return False
        return True


__all__ = ["NemoclawPolicy", "PolicyDecision", "PolicyRule"]
