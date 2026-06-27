"""API-backed Nemotron 3 Ultra triage call.

Used in `deep` mode to:
- Summarize the disagreement between Qwen3Guard and Nemotron-CSR
- Explain WHY the cascade is uncertain
- Suggest a final verdict with reasoning for the audit log

This is the only place a general-purpose Nemotron 3 Ultra call is used. The
550B triage model is not loaded on the demo GPU; the RTX 3090 runs the 4B guard
models locally and calls NVIDIA/OpenRouter for deep-mode adjudication.
"""

from __future__ import annotations

from nemoguardian.schemas import ModelVerdict, VerdictLabel

TRIAGE_PROMPT = """You are a moderation adjudicator. Two specialist classifiers just
disagreed about the following text. Read their verdicts, the text, and any reasoning
they provided, then issue a final verdict.

Text under review:
\"\"\"{text}\"\"\"

Classifier A (Qwen3Guard): verdict={a_label}, categories={a_categories}, reasoning={a_reasoning}
Classifier B (Nemotron-CSR): verdict={b_label}, reasoning={b_reasoning}

Custom policy in effect: {policy}

Output your final verdict and a one-sentence explanation in this exact JSON shape:
{{"verdict": "safe" | "controversial" | "unsafe", "score": 0.0..1.0, "reasons": ["..."]}}
"""


class NemotronTriage:
    """Triage caller — wraps any general-purpose chat model.

    Defaults to Nemotron 3 Ultra via OpenAI-compatible API. For the hackathon
    demo we use the NVIDIA NIM free endpoint (`NVIDIA_API_KEY`).
    """

    DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "nvidia/nemotron-3-ultra-550b-a55b"

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model_name: str | None = None,
    ) -> None:
        import os

        self.api_key = api_key or os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = (
            base_url
            or os.environ.get("NEMOGUARDIAN_TRIAGE_BASE_URL")
            or os.environ.get("NVIDIA_BASE_URL")
            or self._default_base_url()
        )
        self.model_name = model_name or os.environ.get("NEMOGUARDIAN_TRIAGE_MODEL", self.DEFAULT_MODEL)
        if not self.api_key:
            # The triage step is optional — server can run without it.
            self._client = None
        else:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def _default_base_url() -> str:
        import os

        if os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("NVIDIA_API_KEY"):
            return NemotronTriage.OPENROUTER_BASE_URL
        return NemotronTriage.DEFAULT_BASE_URL

    def adjudicate(
        self,
        text: str,
        policy: str | None,
        qwen_verdict: ModelVerdict,
        csr_verdict: ModelVerdict,
    ) -> ModelVerdict:
        """Issue a final verdict explaining any disagreement."""
        import json
        import re
        import time

        if self._client is None:
            return ModelVerdict(
                model_id=self.model_name,
                verdict=VerdictLabel.SAFE,
                score=0.0,
                latency_ms=0.0,
                reasoning="triage disabled (no API key)",
            )

        prompt = TRIAGE_PROMPT.format(
            text=text,
            a_label=qwen_verdict.verdict.value,
            a_categories=qwen_verdict.categories,
            a_reasoning=(qwen_verdict.reasoning or "")[:400],
            b_label=csr_verdict.verdict.value,
            b_reasoning=(csr_verdict.reasoning or "")[:400],
            policy=policy or "(no custom policy)",
        )
        start = time.perf_counter()
        try:
            resp = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=256,
            )
            content = resp.choices[0].message.content or ""
            latency = (time.perf_counter() - start) * 1000.0

            # Parse the JSON output, tolerate code fences.
            content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
            try:
                data = json.loads(content)
            except json.JSONDecodeError:
                # Last-ditch: extract the first {...} block.
                m = re.search(r"\{.*\}", content, re.DOTALL)
                data = json.loads(m.group(0)) if m else {"verdict": "safe", "score": 0.0, "reasons": []}

            verdict_str = str(data.get("verdict", "safe")).lower()
            verdict = VerdictLabel(verdict_str) if verdict_str in {"safe", "controversial", "unsafe"} else VerdictLabel.SAFE
            score = float(data.get("score", 0.0))
            reasons = list(data.get("reasons", []))
            return ModelVerdict(
                model_id=self.model_name,
                verdict=verdict,
                score=score,
                categories=reasons,
                reasoning="; ".join(reasons) or content.strip()[:512],
                latency_ms=latency,
            )
        except Exception as exc:
            return ModelVerdict(
                model_id=self.model_name,
                verdict=VerdictLabel.SAFE,
                score=0.0,
                latency_ms=(time.perf_counter() - start) * 1000.0,
                reasoning=f"triage failed: {exc}",
                error=str(exc),
            )


__all__ = ["NemotronTriage"]
