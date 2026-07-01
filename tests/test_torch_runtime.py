"""Tests for dtype/kwarg selection — especially the Gemma bf16 fix."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from nemoguardian.models import torch_runtime
from nemoguardian.models.torch_runtime import (
    attn_impl_kwargs,
    bnb_compute_dtype,
    dtype_kwarg_name,
    dtype_kwargs,
    runtime_torch_dtype,
)


class _Torch:
    """Minimal torch stand-in. dtypes are sentinel strings."""

    float16 = "float16"
    bfloat16 = "bfloat16"
    float32 = "float32"

    def __init__(self, *, cuda: bool, bf16: bool | None) -> None:
        if cuda:
            ns = SimpleNamespace(is_available=lambda: True)
            if bf16 is not None:
                ns.is_bf16_supported = lambda: bf16
            self.cuda = ns
        else:
            self.cuda = SimpleNamespace(is_available=lambda: False)


class _BrokenCudaTorch(_Torch):
    def __init__(self) -> None:
        self.cuda = SimpleNamespace(is_available=lambda: (_ for _ in ()).throw(RuntimeError("cuda probe failed")))


def test_runtime_dtype_prefers_bf16_on_capable_cuda():
    # Ampere+ GPU with bf16 support → bf16 (required for Gemma/Nemotron-CSR).
    assert runtime_torch_dtype(_Torch(cuda=True, bf16=True)) == "bfloat16"


def test_runtime_dtype_falls_back_to_fp16_without_bf16():
    assert runtime_torch_dtype(_Torch(cuda=True, bf16=False)) == "float16"


def test_runtime_dtype_cpu_is_float32():
    assert runtime_torch_dtype(_Torch(cuda=False, bf16=None)) == "float32"


def test_bnb_compute_dtype_bf16_on_capable_cuda():
    assert bnb_compute_dtype(_Torch(cuda=True, bf16=True)) == "bfloat16"


def test_bnb_compute_dtype_fp16_otherwise():
    # 4-bit only runs on GPU; without bf16 support (or no CUDA) → fp16, never fp32.
    assert bnb_compute_dtype(_Torch(cuda=True, bf16=False)) == "float16"
    assert bnb_compute_dtype(_Torch(cuda=False, bf16=None)) == "float16"


def test_runtime_dtype_assumes_bf16_when_cuda_probe_missing():
    torch = _Torch(cuda=True, bf16=None)

    assert runtime_torch_dtype(torch) == "bfloat16"


def test_runtime_dtype_env_override(monkeypatch):
    monkeypatch.setenv("NEMOGUARDIAN_TORCH_DTYPE", "auto")
    assert runtime_torch_dtype(_Torch(cuda=False, bf16=None)) == "auto"

    monkeypatch.setenv("NEMOGUARDIAN_TORCH_DTYPE", "fp16")
    assert runtime_torch_dtype(_Torch(cuda=False, bf16=None)) == "float16"

    monkeypatch.setenv("NEMOGUARDIAN_TORCH_DTYPE", "wat")
    with pytest.raises(ValueError, match="unsupported NEMOGUARDIAN_TORCH_DTYPE"):
        runtime_torch_dtype(_Torch(cuda=False, bf16=None))


def test_cuda_probe_failures_fall_back_safely():
    torch = _BrokenCudaTorch()

    assert torch_runtime._cuda_bf16_supported(torch) is False
    assert runtime_torch_dtype(torch) == "float32"
    assert bnb_compute_dtype(torch) == "float16"


def test_dtype_kwarg_name_tracks_transformers_version(monkeypatch):
    transformers = ModuleType("transformers")

    transformers.__version__ = "4.56.0"
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    assert dtype_kwarg_name() == "dtype"

    transformers.__version__ = "4.55.9"
    assert dtype_kwarg_name() == "torch_dtype"

    transformers.__version__ = "not-a-version"
    assert dtype_kwarg_name() == "torch_dtype"


def test_dtype_kwargs_uses_detected_kwarg(monkeypatch):
    transformers = ModuleType("transformers")
    transformers.__version__ = "4.56.1"
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    assert dtype_kwargs(_Torch(cuda=False, bf16=None)) == {"dtype": "float32"}


def test_attn_impl_kwargs(monkeypatch):
    monkeypatch.delenv("NEMOGUARDIAN_ATTN_IMPL", raising=False)
    assert attn_impl_kwargs() == {}

    monkeypatch.setenv("NEMOGUARDIAN_ATTN_IMPL", " sdpa ")
    assert attn_impl_kwargs() == {"attn_implementation": "sdpa"}
