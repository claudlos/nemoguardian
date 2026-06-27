"""Pydantic schemas for the nemoguardian API."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class Mode(str, Enum):
    """How heavy a cascade to run."""

    FAST = "fast"            # Qwen3Guard-Stream only (token-level, ~1ms/token)
    STANDARD = "standard"    # + Qwen3Guard-Gen + Nemotron-CSR (reasoning off)
    DEEP = "deep"            # + reasoning-on + Nemotron 3 Ultra triage explanation


class VerdictLabel(str, Enum):
    SAFE = "safe"
    CONTROVERSIAL = "controversial"
    UNSAFE = "unsafe"


class ModelVerdict(BaseModel):
    """One model's contribution to the final decision."""

    model_id: str = Field(..., description="HuggingFace model id")
    verdict: VerdictLabel
    score: float = Field(..., ge=0.0, le=1.0, description="0=safe, 1=unsafe")
    categories: list[str] = Field(default_factory=list)
    reasoning: str | None = Field(default=None, description="Verbatim reasoning trace if any")
    latency_ms: float = Field(..., ge=0.0)
    error: str | None = Field(default=None, description="Populated if the model failed")


class ModerateRequest(BaseModel):
    """Request body for POST /v1/moderate."""

    text: str = Field(..., min_length=1, max_length=100_000)
    policy: str | None = Field(default=None, description="Custom safety policy text (passed to Nemotron-CSR).")
    mode: Mode = Field(default=Mode.STANDARD)
    # Optional per-model toggles. Defaults are filled by the cascade.
    use_qwen_stream: bool = True
    use_qwen_gen: bool = True
    use_nemotron_csr: bool = True
    use_nemotron_triage: bool = False


class ModerateResponse(BaseModel):
    """Response body from POST /v1/moderate."""

    verdict: VerdictLabel
    score: float = Field(..., ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)
    matched_policy_rule: str | None = None
    model_verdicts: dict[str, ModelVerdict] = Field(default_factory=dict)
    total_latency_ms: float
    mode: Mode
    request_id: str | None = None
    timestamp: str | None = None
    usage_info: dict | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"] = "ok"
    models_loaded: dict[str, bool] = Field(default_factory=dict)
    gpu_available: bool = False
    gpu_name: str | None = None


class StreamChunk(BaseModel):
    """One chunk emitted by the streaming endpoint."""

    token_index: int
    partial_text: str
    verdict_so_far: VerdictLabel
    score_so_far: float
    is_terminal: bool = Field(default=False, description="True if this chunk ends the stream")
    latency_ms: float


__all__ = [
    "Mode",
    "VerdictLabel",
    "ModelVerdict",
    "ModerateRequest",
    "ModerateResponse",
    "HealthResponse",
    "StreamChunk",
]
