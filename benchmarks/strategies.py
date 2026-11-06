"""The optimization strategies the benchmark compares, behind one interface.

Each strategy builds a ``Runner`` that can:

* greedy-generate N new tokens from a batch of prompts
* score a token sequence (mean next-token negative log-likelihood), which the
  benchmark turns into perplexity so speed can be weighed against accuracy

| Strategy                 | Precision | KV cache                 | Engine                       |
|--------------------------|-----------|--------------------------|------------------------------|
| `fp32`                   | FP32      | HF DynamicCache (concat) | PyTorch eager (baseline)     |
| `fp32-no-cache`          | FP32      | none (recompute prefix)  | PyTorch eager                |
| `fp32-static-kv`         | FP32      | pre-allocated, reused    | PyTorch eager                |
| `fp16-autocast`          | FP16      | HF DynamicCache          | PyTorch eager + autocast     |
| `int8-dynamic`           | INT8 W/A  | HF DynamicCache          | PyTorch dynamic quantization |
| `int8-dynamic-static-kv` | INT8 W/A  | pre-allocated, reused    | PyTorch dynamic quantization |
| `onnxruntime`            | FP32      | graph inputs/outputs     | ONNX Runtime (CUDA or CPU)   |
| `onnxruntime-int8`       | INT8 W/A  | graph inputs/outputs     | ONNX Runtime INT8 (CPU)      |
| `tensorrt`               | FP16      | graph inputs/outputs     | ONNX Runtime TensorRT EP     |
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ParamSpec, TypeVar

import numpy as np
import torch
import torch.nn.functional as F

from optimizers.kv_cache import StaticKVGPT2
from optimizers.models import LoadedModel, model_size_mb
from optimizers.quantization import count_quantized_linears, fp16_autocast, quantize_int8_dynamic

P = ParamSpec("P")
R = TypeVar("R")

STRATEGIES: tuple[str, ...] = (
    "fp32",
    "fp32-no-cache",
    "fp32-static-kv",
    "fp16-autocast",
    "int8-dynamic",
    "int8-dynamic-static-kv",
    "onnxruntime",
    "onnxruntime-int8",
    "tensorrt",
)
BASELINE = "fp32"


class StrategyUnavailable(RuntimeError):
    """The strategy can't run here, e.g. TensorRT without an NVIDIA GPU."""


@dataclass
class Runner:
    strategy: str
    device: str
    generate: Callable[[torch.Tensor, int], torch.Tensor]  # (B, T) -> (B, T + N)
    nll: Callable[[torch.Tensor], float]  # (1, T) -> mean NLL of tokens 1..T-1
    weights_mb: float
    engine: str
    traces_allocations: bool = True  # PyTorch allocations visible to torch.profiler
    info: dict[str, object] = field(default_factory=dict)


def _hf_generate(model: Any, use_cache: bool = True) -> Callable[[torch.Tensor, int], torch.Tensor]:
    def generate(ids: torch.Tensor, n: int) -> torch.Tensor:
        with torch.inference_mode():
            out: torch.Tensor = model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=n,
                min_new_tokens=n,
                do_sample=False,
                use_cache=use_cache,
            )
        return out

    return generate


def _torch_nll(model: Any) -> Callable[[torch.Tensor], float]:
    def nll(ids: torch.Tensor) -> float:
        with torch.inference_mode():
            logits = model(ids).logits.float()
        return float(F.cross_entropy(logits[0, :-1], ids[0, 1:]))

    return nll


def _with_autocast(fn: Callable[P, R], device: torch.device) -> Callable[P, R]:
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
        with fp16_autocast(device):
            return fn(*args, **kwargs)

    return wrapped


def build_runner(
    strategy: str,
    lm: LoadedModel,
    device: torch.device,
    artifacts: Path,
    threads: int | None = None,
) -> Runner:
    """Build ``strategy`` on top of the loaded FP32 model.

    INT8 strategies quantize ``lm.model`` in place to save memory, so build
    exactly one runner per process (the benchmark runs each in a subprocess).
    """
    model = lm.model
    dev = device.type
    if strategy == "fp32":
        return Runner(
            strategy, dev, _hf_generate(model), _torch_nll(model), model_size_mb(model), "pytorch"
        )
    if strategy == "fp32-no-cache":
        return Runner(
            strategy,
            dev,
            _hf_generate(model, use_cache=False),
            _torch_nll(model),
            model_size_mb(model),
            "pytorch",
        )
    if strategy == "fp32-static-kv":
        dec = StaticKVGPT2(model)
        return Runner(
            strategy,
            dev,
            dec.generate,
            _torch_nll(model),
            model_size_mb(model),
            "pytorch",
            info={"kv_cache": "preallocated"},
        )
    if strategy == "fp16-autocast":
        gen = _with_autocast(_hf_generate(model), device)
        nll = _with_autocast(_torch_nll(model), device)
        return Runner(
            strategy,
            dev,
            gen,
            nll,
            model_size_mb(model),
            "pytorch+autocast",
            info={"autocast_dtype": "float16"},
        )
    if strategy in ("int8-dynamic", "int8-dynamic-static-kv"):
        if dev != "cpu":
            raise StrategyUnavailable("PyTorch dynamic INT8 quantization runs on CPU only")
        q = quantize_int8_dynamic(model, inplace=True)
        info: dict[str, object] = {
            "engine": torch.backends.quantized.engine,
            "quantized_linears": count_quantized_linears(q),
        }
        gen = StaticKVGPT2(q).generate if strategy.endswith("static-kv") else _hf_generate(q)
        return Runner(
            strategy, dev, gen, _torch_nll(q), model_size_mb(q), "pytorch-int8", info=info
        )
    if strategy in ("onnxruntime", "onnxruntime-int8", "tensorrt"):
        return _ort_runner(strategy, lm, device, artifacts, threads)
    raise ValueError(f"unknown strategy {strategy!r}; choose from {', '.join(STRATEGIES)}")


def _ort_runner(
    strategy: str, lm: LoadedModel, device: torch.device, artifacts: Path, threads: int | None
) -> Runner:
    import onnxruntime as ort

    from optimizers.onnx_export import export_gpt2_step
    from optimizers.tensorrt import OrtGenerator, StepShapes, build_session, quantize_onnx_int8

    available = ort.get_available_providers()
    if strategy == "tensorrt":
        if "TensorrtExecutionProvider" not in available:
            raise StrategyUnavailable(
                "TensorrtExecutionProvider not available (needs onnxruntime-gpu, TensorRT "
                "and an NVIDIA GPU)"
            )
        prefer = "tensorrt"
    elif strategy == "onnxruntime-int8":
        prefer = "cpu"
    else:
        prefer = "cuda" if device.type == "cuda" and "CUDAExecutionProvider" in available else "cpu"

    cfg = lm.model.config
    shapes = StepShapes(cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head)
    base = artifacts / lm.name.replace("/", "_")
    fp32_path = base / "model.onnx"
    if not fp32_path.exists():
        export_gpt2_step(lm.model.cpu(), fp32_path)
    path = fp32_path
    if strategy == "onnxruntime-int8":
        path = base / "int8" / "model.int8.onnx"
        if not path.exists():
            quantize_onnx_int8(fp32_path, path)
    weights_mb = sum(f.stat().st_size for f in path.parent.iterdir() if f.is_file()) / 2**20
    # The PyTorch model is only needed for export; free it before benchmarking.
    lm.model = None
    session = build_session(path, shapes, prefer=prefer, threads=threads)
    gen = OrtGenerator(session, shapes)

    def generate(ids: torch.Tensor, n: int) -> torch.Tensor:
        return torch.from_numpy(gen.generate(ids.cpu().numpy(), n))

    def nll(ids: torch.Tensor) -> float:
        tokens = ids.cpu().numpy()
        k = v = np.zeros((shapes.num_layers, 1, shapes.num_heads, 0, shapes.head_dim), gen.kv_dtype)
        total = 0.0
        for i in range(tokens.shape[1] - 1):
            logits, k, v = gen.step(tokens[:, i : i + 1], k, v)
            row = logits[0].astype(np.float64)
            top = row.max()
            total += math.log(np.exp(row - top).sum()) + top - row[tokens[0, i + 1]]
        return total / (tokens.shape[1] - 1)

    engine = {"tensorrt": "onnxruntime+tensorrt"}.get(strategy, "onnxruntime")
    return Runner(
        strategy,
        device.type,
        generate,
        nll,
        weights_mb,
        engine,
        traces_allocations=False,
        info={"provider": gen.provider, "onnx": str(path)},
    )
