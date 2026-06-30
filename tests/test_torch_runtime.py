"""Tests for dtype/kwarg selection — especially the Gemma bf16 fix."""

from __future__ import annotations

from types import SimpleNamespace

from nemoguardian.models.torch_runtime import bnb_compute_dtype, runtime_torch_dtype


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
