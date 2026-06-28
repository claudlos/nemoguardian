"""Cascade orchestration.

Given a `ModerateRequest`, decides which models to run, in what order, and
returns a populated `ModerateResponse`. Models are loaded lazily and cached
singleton-style inside the orchestrator.
"""

from __future__ import annotations

import datetime as dt
import os
import re
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
    VerdictLabel,
)

_EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_PHONE_RE = re.compile(r"(?<!\w)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\w)")
_PAYMENT_CARD_RE = re.compile(r"\b(?:\d[ -]*?){13,19}\b")


@dataclass
class CascadeConfig:
    qwen_gen_model: str = Qwen3GuardGen.DEFAULT_MODEL
    qwen_stream_model: str = Qwen3GuardStream.DEFAULT_MODEL
    csr_model: str = NemotronCSR.DEFAULT_MODEL
    triage_model: str = NemotronTriage.DEFAULT_MODEL
    triage_base_url: str | None = None
    qwen_gen_4bit: bool = True
    csr_4bit: bool = True
    reasoning: bool = True  # Nemotron-CSR reasoning-on mode
    enable_triage: bool = True

    @classmethod
    def from_env(cls) -> CascadeConfig:
        quantize = _env_bool("NEMOGUARDIAN_QUANTIZE", True)
        return cls(
            qwen_gen_model=os.environ.get("NEMOGUARDIAN_QWEN_MODEL", Qwen3GuardGen.DEFAULT_MODEL),
            qwen_stream_model=os.environ.get(
                "NEMOGUARDIAN_QWEN_STREAM_MODEL", Qwen3GuardStream.DEFAULT_MODEL
            ),
            csr_model=os.environ.get("NEMOGUARDIAN_CSR_MODEL", NemotronCSR.DEFAULT_MODEL),
            triage_model=os.environ.get("NEMOGUARDIAN_TRIAGE_MODEL", NemotronTriage.DEFAULT_MODEL),
            triage_base_url=os.environ.get("NEMOGUARDIAN_TRIAGE_BASE_URL")
            or os.environ.get("NVIDIA_BASE_URL"),
            qwen_gen_4bit=_env_bool("NEMOGUARDIAN_QWEN_4BIT", quantize),
            csr_4bit=_env_bool("NEMOGUARDIAN_CSR_4BIT", quantize),
            reasoning=_env_bool("NEMOGUARDIAN_REASONING", True),
            enable_triage=_env_bool("NEMOGUARDIAN_ENABLE_TRIAGE", True),
        )


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
            self._qwen_gen = Qwen3GuardGen(
                model_name=self.config.qwen_gen_model,
                load_in_4bit=self.config.qwen_gen_4bit,
            )
        return self._qwen_gen

    @property
    def qwen_stream(self) -> Qwen3GuardStream:
        if self._qwen_stream is None:
            self._qwen_stream = Qwen3GuardStream(model_name=self.config.qwen_stream_model)
        return self._qwen_stream

    @property
    def csr(self) -> NemotronCSR:
        if self._csr is None:
            self._csr = NemotronCSR(
                model_name=self.config.csr_model,
                reasoning=self.config.reasoning,
                load_in_4bit=self.config.csr_4bit,
            )
        return self._csr

    @property
    def triage(self) -> NemotronTriage:
        if self._triage is None and self.config.enable_triage:
            self._triage = NemotronTriage(
                base_url=self.config.triage_base_url,
                model_name=self.config.triage_model,
            )
        return self._triage  # may be None

    def loaded_models(self) -> dict[str, bool]:
        return {
            "qwen3_guard_gen": bool(self._qwen_gen and self._qwen_gen.is_loaded),
            "qwen3_guard_stream": bool(self._qwen_stream and self._qwen_stream._loaded),
            "nemotron_csr": bool(self._csr and self._csr.is_loaded),
        }

    def model_config_summary(self) -> dict[str, str | bool | None]:
        return {
            "qwen_gen_model": self.config.qwen_gen_model,
            "qwen_stream_model": self.config.qwen_stream_model,
            "nemotron_csr_model": self.config.csr_model,
            "nemotron_triage_model": self.config.triage_model,
            "triage_base_url": self.config.triage_base_url or _default_triage_base_url(),
            "qwen_gen_4bit": self.config.qwen_gen_4bit,
            "csr_4bit": self.config.csr_4bit,
            "reasoning": self.config.reasoning,
            "enable_triage": self.config.enable_triage,
        }

    def triage_status(self) -> dict[str, str | bool | None]:
        configured = bool(os.environ.get("NVIDIA_API_KEY") or os.environ.get("OPENROUTER_API_KEY"))
        provider = None
        if os.environ.get("NVIDIA_API_KEY"):
            provider = "nvidia"
        elif os.environ.get("OPENROUTER_API_KEY"):
            provider = "openrouter"
        return {
            "configured": configured and self.config.enable_triage,
            "provider": provider,
            "model": self.config.triage_model,
            "base_url": self.config.triage_base_url or _default_triage_base_url(),
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

        if request.mode == Mode.FAST:
            if request.use_qwen_stream:
                model_verdicts["qwen3_guard_stream"] = self._stream_document_verdict(request.text)
        elif request.use_qwen_gen:
            model_verdicts["qwen3_guard_gen"] = self.qwen_gen.moderate(request.text, policy=request.policy)

        if request.mode != Mode.FAST and request.use_nemotron_csr:
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
        for category in _text_policy_categories(request.text):
            if category not in aggregated.categories:
                aggregated.categories.append(category)
                aggregated.reasons.append(f"[NemoClaw] Detected {category} pattern in message text.")

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

    def _stream_document_verdict(self, text: str) -> ModelVerdict:
        start = time.perf_counter()
        best_label = VerdictLabel.SAFE
        best_score = 0.0
        best_idx = 0
        for label, confidence, idx in self.qwen_stream.stream_classify(text):
            verdict = VerdictLabel(label)
            if verdict == VerdictLabel.UNSAFE:
                risk_score = max(0.7, confidence)
            elif verdict == VerdictLabel.CONTROVERSIAL:
                risk_score = max(0.3, min(confidence, 0.69))
            else:
                risk_score = max(0.0, min(1.0 - confidence, 0.29))
            if risk_score > best_score:
                best_label = verdict
                best_score = risk_score
                best_idx = idx
        return ModelVerdict(
            model_id=self.config.qwen_stream_model,
            verdict=best_label,
            score=round(best_score, 4),
            categories=[],
            reasoning=f"Highest streaming token risk at token index {best_idx}.",
            latency_ms=round((time.perf_counter() - start) * 1000.0, 2),
        )


def _text_policy_categories(text: str) -> list[str]:
    if _SSN_RE.search(text) or _EMAIL_RE.search(text) or _PHONE_RE.search(text) or _PAYMENT_CARD_RE.search(text):
        return ["PII"]
    return []


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _default_triage_base_url() -> str:
    if os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("NVIDIA_API_KEY"):
        return "https://openrouter.ai/api/v1"
    return NemotronTriage.DEFAULT_BASE_URL


__all__ = ["Cascade", "CascadeConfig"]
