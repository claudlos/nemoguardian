"""Cascade orchestration.

Given a `ModerateRequest`, decides which models to run, in what order, and
returns a populated `ModerateResponse`. Models are loaded lazily and cached
singleton-style inside the orchestrator.
"""

from __future__ import annotations

import datetime as dt
import time
import uuid
from dataclasses import dataclass, field

from nemoguardian.aggregator import AggregatedVerdict, aggregate
from nemoguardian.models.nemotron_csr import NemotronCSR
from nemoguardian.models.nemotron_triage import NemotronTriage
from nemoguardian.models.qwen3_guard import Qwen3GuardGen, Qwen3GuardStream
from nemoguardian.policy.nemoclaw import NemoclawPolicy, PolicyDecision
from nemoguardian.schemas import (
    Mode,
    ModelVerdict,
    ModerateRequest,
    ModerateResponse,
)


@dataclass
class CascadeConfig:
    qwen_gen_4bit: bool = True
    csr_4bit: bool = True
    reasoning: bool = True  # Nemotron-CSR reasoning-on mode
    enable_triage: bool = True


@dataclass
class Cascade:
    config: CascadeConfig = field(default_factory=CascadeConfig)
    _qwen_gen: Qwen3GuardGen | None = field(default=None, init=False, repr=False)
    _qwen_stream: Qwen3GuardStream | None = field(default=None, init=False, repr=False)
    _csr: NemotronCSR | None = field(default=None, init=False, repr=False)
    _triage: NemotronTriage | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        # Lazy: only construct model objects on first call, not on import.
        pass

    @property
    def qwen_gen(self) -> Qwen3GuardGen:
        if self._qwen_gen is None:
            self._qwen_gen = Qwen3GuardGen(load_in_4bit=self.config.qwen_gen_4bit)
        return self._qwen_gen

    @property
    def qwen_stream(self) -> Qwen3GuardStream:
        if self._qwen_stream is None:
            self._qwen_stream = Qwen3GuardStream()
        return self._qwen_stream

    @property
    def csr(self) -> NemotronCSR:
        if self._csr is None:
            self._csr = NemotronCSR(reasoning=self.config.reasoning, load_in_4bit=self.config.csr_4bit)
        return self._csr

    @property
    def triage(self) -> NemotronTriage:
        if self._triage is None and self.config.enable_triage:
            self._triage = NemotronTriage()
        return self._triage  # may be None

    def loaded_models(self) -> dict[str, bool]:
        return {
            "qwen3_guard_gen": bool(self._qwen_gen and self._qwen_gen.is_loaded),
            "qwen3_guard_stream": bool(self._qwen_stream and self._qwen_stream._loaded),
            "nemotron_csr": bool(self._csr and self._csr.is_loaded),
        }

    def moderate(
        self,
        request: ModerateRequest,
        *,
        policy_engine: NemoclawPolicy | None = None,
    ) -> ModerateResponse:
        """Run the full cascade for one request."""
        start = time.perf_counter()
        request_id = uuid.uuid4().hex[:12]

        model_verdicts: dict[str, ModelVerdict] = {}

        if request.use_qwen_gen:
            model_verdicts["qwen3_guard_gen"] = self.qwen_gen.moderate(request.text, policy=request.policy)

        if request.use_nemotron_csr:
            model_verdicts["nemotron_csr"] = self.csr.moderate(request.text, policy=request.policy)

        # Triage only in DEEP mode, only when both Qwen and CSR ran.
        if (
            request.mode == Mode.DEEP
            and self.triage is not None
            and "qwen3_guard_gen" in model_verdicts
            and "nemotron_csr" in model_verdicts
        ):
            model_verdicts["triage"] = self.triage.adjudicate(
                text=request.text,
                policy=request.policy,
                qwen_verdict=model_verdicts["qwen3_guard_gen"],
                csr_verdict=model_verdicts["nemotron_csr"],
            )

        aggregated: AggregatedVerdict = aggregate(model_verdicts)

        # NemoClaw policy gate
        matched_rule: str | None = None
        if policy_engine is not None:
            decision: PolicyDecision = policy_engine.evaluate(
                verdict=aggregated.verdict,
                score=aggregated.score,
                categories=aggregated.categories,
                policy_text=request.policy,
            )
            matched_rule = decision.matched_rule
            if decision.final_label is not None and decision.final_label != aggregated.verdict:
                # NemoClaw overrode the model verdict (e.g., force-block on PII).
                aggregated.verdict = decision.final_label
                if decision.final_score is not None:
                    aggregated.score = decision.final_score

        total_ms = (time.perf_counter() - start) * 1000.0

        return ModerateResponse(
            verdict=aggregated.verdict,
            score=aggregated.score,
            reasons=aggregated.reasons,
            categories=aggregated.categories,
            matched_policy_rule=matched_rule,
            model_verdicts=model_verdicts,
            total_latency_ms=round(total_ms, 2),
            mode=request.mode,
            request_id=request_id,
            timestamp=dt.datetime.now(dt.timezone.utc).isoformat(),
        )

    def stream_token_verdicts(self, text: str):
        """Per-token verdicts from Qwen3Guard-Stream."""
        yield from self.qwen_stream.stream_classify(text)


__all__ = ["Cascade", "CascadeConfig"]
