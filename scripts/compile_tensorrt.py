"""Export a GPT-2 model to ONNX and compile it with TensorRT, or fall back to ONNX Runtime on CPU.

Steps:

1. Export the model as a single KV-cached decoding step
   (``optimizers/onnx_export.py``) to ``artifacts/<model>/model.onnx``.
2. Optionally (``--int8``) write an ONNX Runtime INT8 copy, ``model.int8.onnx``.
3. Build an ONNX Runtime session with the best provider available: TensorRT,
   then CUDA, then CPU. TensorRT compiles and caches its engine on this first
   run.
4. Check the session against PyTorch: prompt logits and a short greedy
   continuation.
5. Time the session against PyTorch eager.

Usage::

    python -m scripts.compile_tensorrt --model gpt2 --int8
    python -m scripts.compile_tensorrt --model gpt2 --provider tensorrt   # needs an NVIDIA GPU
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from optimizers.models import load_model
from optimizers.onnx_export import GPT2StepModule, export_gpt2_step
from optimizers.tensorrt import (
    OrtGenerator,
    StepShapes,
    available_providers,
    build_session,
    quantize_onnx_int8,
)

PROMPT = "The key to fast language model inference is"


def shapes_of(model: Any) -> StepShapes:
    cfg = model.config
    return StepShapes(cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head)


def onnx_paths(artifacts: Path, model: str) -> tuple[Path, Path]:
    base = artifacts / model.replace("/", "_")
    return base / "model.onnx", base / "int8" / "model.int8.onnx"


def ensure_exported(model: torch.nn.Module, path: Path, force: bool = False) -> Path:
    if force or not path.exists():
        start = time.perf_counter()
        export_gpt2_step(model, path)
        print(f"exported {path} in {time.perf_counter() - start:.1f}s")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--provider", choices=["auto", "tensorrt", "cuda", "cpu"], default="auto")
    parser.add_argument("--int8", action="store_true", help="also build an ONNX Runtime INT8 model")
    parser.add_argument("--force", action="store_true", help="re-export even if the file exists")
    parser.add_argument("--new-tokens", type=int, default=32)
    args = parser.parse_args()

    print(f"onnxruntime providers available: {available_providers()}")
    lm = load_model(args.model, "cpu")
    shapes = shapes_of(lm.model)
    fp32_path, int8_path = onnx_paths(args.artifacts, args.model)
    ensure_exported(lm.model, fp32_path, args.force)

    targets = [("onnx-fp32", fp32_path, args.provider)]
    if args.int8:
        if args.force or not int8_path.exists():
            quantize_onnx_int8(fp32_path, int8_path)
            print(f"quantized {int8_path}")
        targets.append(("onnx-int8", int8_path, "cpu"))

    ids = lm.tokenizer(PROMPT, return_tensors="pt").input_ids
    step = GPT2StepModule(lm.model).eval()
    with torch.inference_mode():
        ref_logits = lm.model(ids).logits[:, -1].numpy()
        ref_tokens = lm.model.generate(ids, max_new_tokens=args.new_tokens, do_sample=False)
        lm.model.generate(ids, max_new_tokens=4, do_sample=False)
        start = time.perf_counter()
        lm.model.generate(
            ids, max_new_tokens=args.new_tokens, min_new_tokens=args.new_tokens, do_sample=False
        )
        torch_ms = (time.perf_counter() - start) / args.new_tokens * 1000
    report: dict[str, object] = {"model": args.model, "pytorch_eager_ms_per_token": torch_ms}

    for name, path, provider in targets:
        start = time.perf_counter()
        session = build_session(path, shapes, prefer=provider)
        build_s = time.perf_counter() - start
        gen = OrtGenerator(session, shapes)
        empty = step.empty_past(1).numpy().astype(gen.kv_dtype)
        logits, _, _ = gen.step(ids.numpy(), empty, empty)
        max_diff = float(np.max(np.abs(logits - ref_logits)))
        tokens = gen.generate(ids.numpy(), args.new_tokens)
        agree = float(np.mean(tokens[0] == ref_tokens[0].numpy()))
        gen.generate(ids.numpy(), 4)
        start = time.perf_counter()
        gen.generate(ids.numpy(), args.new_tokens)
        ms = (time.perf_counter() - start) / args.new_tokens * 1000
        print(
            f"{name}: provider={gen.provider} build {build_s:.1f}s | "
            f"max |logit diff| {max_diff:.2e} | tokens matching PyTorch {agree:.0%} | "
            f"{ms:.2f} ms/token (PyTorch eager {torch_ms:.2f})"
        )
        report[name] = {
            "provider": gen.provider,
            "session_build_s": round(build_s, 2),
            "max_abs_logit_diff": max_diff,
            "token_agreement": agree,
            "ms_per_token": round(ms, 3),
        }
        if name == "onnx-fp32" and max_diff > 1e-2 and gen.provider == "CPUExecutionProvider":
            raise SystemExit("ONNX FP32 output does not match PyTorch")

    out = fp32_path.parent / "compile_report.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
