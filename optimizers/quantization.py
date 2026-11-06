"""INT8 dynamic quantization and FP16 mixed precision for GPT-2 family models.

**INT8 (dynamic quantization).** ``torch.ao.quantization.quantize_dynamic``
stores the weights of every ``nn.Linear`` as INT8, once, ahead of time. At run
time each layer quantizes its activations on the fly (per batch, from their
observed range), runs an INT8 matrix multiply and dequantizes the result.
Weights shrink 4x and matmuls use integer kernels, so it suits CPU inference of
memory-bound layers. It needs no calibration data.

Hugging Face GPT-2 implements its projections as ``transformers.Conv1D``
(a Linear with transposed weights), which ``quantize_dynamic`` doesn't
recognise. ``conv1d_to_linear`` rewrites them as ``nn.Linear`` first, so every
attention and MLP projection is quantized.

Two choices here are about accuracy:

* **Per-channel weight scales** (one scale per output row) instead of one scale
  per tensor. On GPT-2 this gives lower perplexity at the same speed.
* **The LM head stays FP32 by default.** It is tied to the token embedding and
  its outputs are the logits themselves, so INT8 error there shows up directly
  in the next-token distribution. Quantizing it roughly triples perplexity on
  GPT-2, though it makes a decode step faster (the head is about a third of
  GPT-2 small's FLOPs). ``quantize_lm_head=True`` turns it on.

**FP16 (autocast).** ``torch.autocast(device_type, dtype=torch.float16)`` runs
matmuls and other eligible ops in half precision and keeps precision-sensitive
ops (softmax, layer norm, reductions) in FP32. The FP32 weights stay as they
are, and ops cast on the fly. On CUDA this uses tensor cores. On CPU, FP16
matmuls are usually *slower* than FP32 unless the CPU has native FP16 support.
The benchmark reports whatever it measures.
"""

from __future__ import annotations

import contextlib
import copy
from collections.abc import Iterator

import torch
from torch import nn
from transformers.pytorch_utils import Conv1D


def select_quantized_engine() -> str:
    """Use the best INT8 backend PyTorch has for this CPU.

    The ``x86`` and ``fbgemm`` backends use AVX2/AVX-512 (VNNI) kernels;
    ``qnnpack`` is the ARM backend.
    """
    engines = torch.backends.quantized.supported_engines
    for engine in ("x86", "fbgemm", "qnnpack"):
        if engine in engines:
            torch.backends.quantized.engine = engine
            return engine
    raise RuntimeError(f"no INT8 quantized engine available (supported: {engines})")


def conv1d_to_linear(module: nn.Module) -> int:
    """Replace every ``transformers.Conv1D`` in ``module`` with an equivalent ``nn.Linear``.

    Conv1D computes ``x @ W + b`` with ``W`` of shape (in, out); nn.Linear
    computes ``x @ W.T + b``. Returns the number of layers replaced.
    """
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, Conv1D):
            in_features, out_features = child.weight.shape
            linear = nn.Linear(in_features, out_features, bias=True)
            with torch.no_grad():
                linear.weight.copy_(child.weight.t())
                linear.bias.copy_(child.bias)
            linear.to(child.weight.device)
            setattr(module, name, linear)
            replaced += 1
        else:
            replaced += conv1d_to_linear(child)
    return replaced


def quantize_int8_dynamic(
    model: nn.Module,
    inplace: bool = False,
    per_channel: bool = True,
    quantize_lm_head: bool = False,
) -> nn.Module:
    """Return an INT8 dynamically quantized copy of ``model`` (CPU only)."""
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("PyTorch dynamic quantization runs on CPU; move the model to CPU first")
    select_quantized_engine()
    target = model if inplace else copy.deepcopy(model)
    conv1d_to_linear(target)
    qconfig = (
        torch.ao.quantization.per_channel_dynamic_qconfig
        if per_channel
        else torch.ao.quantization.default_dynamic_qconfig
    )
    spec: dict[type[nn.Module] | str, object] = {nn.Linear: qconfig}
    if not quantize_lm_head:
        spec["lm_head"] = None
    quantized: nn.Module = torch.ao.quantization.quantize_dynamic(
        target, spec, dtype=torch.qint8, inplace=True
    )
    return quantized.eval()


def count_quantized_linears(model: nn.Module) -> int:
    from torch.ao.nn.quantized.dynamic import Linear as DynamicQuantizedLinear

    return sum(isinstance(m, DynamicQuantizedLinear) for m in model.modules())


@contextlib.contextmanager
def fp16_autocast(device: torch.device | str) -> Iterator[None]:
    """Context manager that runs eligible ops in FP16 via ``torch.autocast``."""
    device_type = torch.device(device).type
    with torch.autocast(device_type=device_type, dtype=torch.float16):
        yield
