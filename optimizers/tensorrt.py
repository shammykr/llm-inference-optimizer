"""Compile and run the exported ONNX graph with TensorRT, CUDA or plain CPU.

``build_session`` picks the fastest ONNX Runtime execution provider available:

1. **TensorRT** (``TensorrtExecutionProvider``, from ``onnxruntime-gpu`` with
   TensorRT libraries). On the first run TensorRT compiles the graph into an
   engine: it fuses layers, picks kernels for the GPU and enables FP16. The
   engine is cached on disk so later runs skip compilation. Because the KV
   length changes every step, TensorRT gets an optimisation profile with
   min/opt/max shapes.
2. **CUDA** (``CUDAExecutionProvider``) when TensorRT isn't installed.
3. **CPU** (``CPUExecutionProvider``): the fallback when there is no GPU.
   ONNX Runtime still applies its graph optimisations (constant folding, op
   fusion).

``quantize_onnx_int8`` uses ONNX Runtime's own dynamic INT8 quantization
(MatMul weights to INT8). On ARM and x86 CPUs it runs ONNX Runtime's MLAS
integer kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import onnxruntime as ort

from optimizers.onnx_export import INPUT_NAMES

PROVIDER_PRIORITY = ("TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider")


@dataclass(frozen=True)
class StepShapes:
    """Model dimensions needed to build KV inputs and TensorRT profiles."""

    num_layers: int
    num_heads: int
    head_dim: int


def available_providers() -> list[str]:
    available = set(ort.get_available_providers())
    return [p for p in PROVIDER_PRIORITY if p in available]


def tensorrt_profile(
    shapes: StepShapes, max_batch: int, max_prompt: int, max_seq: int
) -> dict[str, str]:
    """min/opt/max shapes for the dynamic inputs (batch, new tokens, past length)."""
    lh, d = f"{shapes.num_layers}", f"{shapes.num_heads}x{{p}}x{shapes.head_dim}"

    def spec(b: int, t: int, p: int) -> str:
        kv = f"{lh}x{b}x" + d.format(p=p)
        return f"input_ids:{b}x{t},past_key:{kv},past_value:{kv}"

    return {
        "trt_profile_min_shapes": spec(1, 1, 0),
        "trt_profile_opt_shapes": spec(1, 1, max_seq // 2),
        "trt_profile_max_shapes": spec(max_batch, max_prompt, max_seq),
    }


def build_session(
    onnx_path: Path,
    shapes: StepShapes,
    prefer: str = "auto",
    threads: int | None = None,
    engine_cache: Path | None = None,
    fp16: bool = True,
    max_batch: int = 8,
    max_prompt: int = 512,
    max_seq: int = 1024,
) -> ort.InferenceSession:
    """Create an ONNX Runtime session with the preferred (or best available) provider.

    ``prefer`` is ``auto``, ``tensorrt``, ``cuda`` or ``cpu``.
    """
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if threads:
        opts.intra_op_num_threads = threads
    wanted = {
        "auto": available_providers(),
        "tensorrt": ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
        "cpu": ["CPUExecutionProvider"],
    }[prefer]
    missing = [p for p in wanted[:1] if p not in ort.get_available_providers()]
    if missing:
        raise RuntimeError(
            f"{missing[0]} is not available in this onnxruntime build "
            f"(available: {ort.get_available_providers()})"
        )
    providers: list[str | tuple[str, dict[str, object]]] = []
    for name in wanted:
        if name == "TensorrtExecutionProvider":
            cache = engine_cache or onnx_path.parent / "trt_engines"
            cache.mkdir(parents=True, exist_ok=True)
            options: dict[str, object] = {
                "trt_fp16_enable": fp16,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": str(cache),
                "trt_timing_cache_enable": True,
                **tensorrt_profile(shapes, max_batch, max_prompt, max_seq),
            }
            providers.append((name, options))
        else:
            providers.append(name)
    return ort.InferenceSession(str(onnx_path), opts, providers=providers)


def quantize_onnx_int8(fp32_path: Path, int8_path: Path) -> Path:
    """ONNX Runtime dynamic INT8 quantization of the MatMul weights."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    int8_path.parent.mkdir(parents=True, exist_ok=True)
    large = fp32_path.with_name(fp32_path.name + ".data").exists()
    # The exporter's intermediate shape annotations (value_info) can conflict
    # with ONNX shape inference after optimisation. Dropping them lets the
    # quantizer infer shapes from scratch.
    model = onnx.load(str(fp32_path))
    del model.graph.value_info[:]
    # GPT-2 ties the token embedding to the LM head, so the exported Gather and
    # the LM-head Gemm share one initializer. Give the Gather its own copy, so
    # the quantizer's edits to the head can't change the embedding lookup.
    head = next(n for n in model.graph.node if "logits" in n.output)
    shared = {name for name in head.input} & {i.name for i in model.graph.initializer}
    for init in list(model.graph.initializer):
        if init.name in shared:
            copy_ = onnx.TensorProto()
            copy_.CopyFrom(init)
            copy_.name = init.name + ".embedding"
            model.graph.initializer.append(copy_)
            for node in model.graph.node:
                if node.op_type == "Gather" and init.name in node.input:
                    node.input[0] = copy_.name
    clean = int8_path.with_name("model.clean.onnx")
    onnx.save(
        model,
        str(clean),
        save_as_external_data=large,
        location=clean.name + ".data" if large else None,
    )
    del model
    quantize_dynamic(
        str(clean),
        str(int8_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
        op_types_to_quantize=["MatMul"],
        nodes_to_exclude=[head.name],  # keep the LM head FP32, as in the PyTorch path
        use_external_data_format=large,
    )
    for leftover in (clean, clean.with_name(clean.name + ".data")):
        leftover.unlink(missing_ok=True)
    return int8_path


class OrtGenerator:
    """Greedy decoding loop over the exported step graph."""

    def __init__(self, session: ort.InferenceSession, shapes: StepShapes) -> None:
        self.session = session
        self.shapes = shapes
        self.provider: str = session.get_providers()[0]
        kv_input = next(i for i in session.get_inputs() if i.name == "past_key")
        self.kv_dtype = np.float16 if "float16" in kv_input.type else np.float32

    def step(
        self, ids: npt.NDArray[np.int64], past_key: npt.NDArray, past_value: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        feed = dict(zip(INPUT_NAMES, (ids, past_key, past_value), strict=True))
        logits, key, value = self.session.run(None, feed)
        return logits, key, value

    def generate(self, input_ids: npt.NDArray[np.int64], max_new_tokens: int) -> npt.NDArray:
        b, t = input_ids.shape
        s = self.shapes
        past = np.zeros((s.num_layers, b, s.num_heads, 0, s.head_dim), dtype=self.kv_dtype)
        out = np.empty((b, t + max_new_tokens), dtype=np.int64)
        out[:, :t] = input_ids
        logits, key, value = self.step(input_ids.astype(np.int64), past, past)
        for i in range(max_new_tokens):
            next_token = logits.argmax(axis=-1).astype(np.int64)
            out[:, t + i] = next_token
            if i + 1 < max_new_tokens:
                logits, key, value = self.step(next_token[:, None], key, value)
        return out
