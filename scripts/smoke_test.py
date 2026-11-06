"""CI smoke test: load GPT-2, run one inference per optimization path, check output shapes.

The paths checked:

* **FP32** forward: logits of shape (batch, seq, vocab)
* **static KV-cache** generation: shape (batch, seq + new tokens), and the
  same tokens as Hugging Face ``generate``
* **INT8 dynamic quantization:** logits of the same shape, quantized layers
  present, and a smaller model
* **ONNX Runtime** (TensorRT, CUDA or CPU, whichever is available): one
  exported decoding step, checking logit and KV shapes

Usage::

    python -m scripts.smoke_test --model gpt2
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from optimizers.kv_cache import StaticKVGPT2
from optimizers.models import load_model, model_size_mb
from optimizers.onnx_export import GPT2StepModule, export_gpt2_step
from optimizers.quantization import count_quantized_linears, quantize_int8_dynamic
from optimizers.tensorrt import OrtGenerator, StepShapes, build_session


def check(name: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}{': ' + detail if detail else ''}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--skip-onnx", action="store_true")
    args = parser.parse_args()

    lm = load_model(args.model, "cpu")
    cfg = lm.model.config
    vocab = cfg.vocab_size
    prompts = ["Inference optimization makes models", "The quick brown fox jumps"]
    ids = lm.tokenizer(prompts, return_tensors="pt").input_ids[:, :5]
    batch, seq = ids.shape
    results = []
    print(
        f"model {args.model}: {lm.num_parameters / 1e6:.0f}M parameters, input {tuple(ids.shape)}"
    )

    with torch.inference_mode():
        logits = lm.model(ids).logits
    results.append(
        check(
            "fp32 logits shape",
            tuple(logits.shape) == (batch, seq, vocab),
            str(tuple(logits.shape)),
        )
    )

    new = 8
    static = StaticKVGPT2(lm.model).generate(ids, new)
    with torch.inference_mode():
        hf = lm.model.generate(
            ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=new,
            min_new_tokens=new,
            do_sample=False,
        )
    results.append(
        check(
            "static KV generate shape",
            tuple(static.shape) == (batch, seq + new),
            str(tuple(static.shape)),
        )
    )
    results.append(check("static KV tokens match HF generate", torch.equal(static, hf)))

    if not args.skip_onnx:
        with tempfile.TemporaryDirectory() as tmp:
            path = export_gpt2_step(lm.model, Path(tmp) / "model.onnx")
            shapes = StepShapes(cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head)
            gen = OrtGenerator(build_session(path, shapes), shapes)
            past = GPT2StepModule(lm.model).empty_past(batch).numpy()
            step_logits, key, _ = gen.step(ids.numpy(), past, past)
            expected_kv = (cfg.n_layer, batch, cfg.n_head, seq, shapes.head_dim)
            results.append(
                check(
                    f"onnxruntime ({gen.provider}) logits shape",
                    step_logits.shape == (batch, vocab),
                    str(step_logits.shape),
                )
            )
            results.append(
                check("onnxruntime present KV shape", key.shape == expected_kv, str(key.shape))
            )
            diff = float(np.abs(step_logits - logits[:, -1].numpy()).max())
            results.append(check("onnxruntime logits match PyTorch", diff < 1e-2, f"{diff:.1e}"))

    fp32_mb = model_size_mb(lm.model)
    q = quantize_int8_dynamic(lm.model, inplace=True)
    with torch.inference_mode():
        q_logits = q(ids).logits
    results.append(
        check(
            "int8 logits shape",
            tuple(q_logits.shape) == (batch, seq, vocab),
            str(tuple(q_logits.shape)),
        )
    )
    n_q = count_quantized_linears(q)
    results.append(check("int8 layers quantized", n_q == 4 * cfg.n_layer, f"{n_q} Linear layers"))
    q_mb = model_size_mb(q)
    results.append(
        check("int8 model smaller", q_mb < fp32_mb, f"{fp32_mb:.0f} MB -> {q_mb:.0f} MB")
    )
    out = StaticKVGPT2(q).generate(ids, new)
    results.append(check("int8 + static KV generate shape", tuple(out.shape) == (batch, seq + new)))
    print("sample:", lm.tokenizer.decode(out[0]))

    if not all(results):
        sys.exit(1)
    print("smoke test passed")


if __name__ == "__main__":
    main()
