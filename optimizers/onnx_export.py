"""Export GPT-2 to ONNX as a single decoding step with explicit KV-cache inputs.

Exporting ``model.forward`` as-is would hide the KV cache inside Hugging Face
``Cache`` objects. ``GPT2StepModule`` instead turns one decoding step into a
plain function of tensors:

    (input_ids[B, T], past_key[L, B, H, P, D], past_value[L, B, H, P, D])
        -> (logits[B, V], present_key[L, B, H, P+T, D], present_value[L, B, H, P+T, D])

The same graph runs the prompt (P = 0, T = prompt length) and each generated
token (T = 1). Batch, T and P are dynamic. ``logits`` covers only the last
position, which is all that greedy decoding needs, and it saves projecting
every prompt position onto the 50K-token vocabulary.

Projections are written as ``x @ W + b``, so they export as ONNX ``MatMul`` +
``Add``, the form ONNX Runtime's INT8 dynamic quantizer recognises. The LM head
exports as a ``Gemm``, which that quantizer skips, so in the ONNX INT8 model it
stays FP32, matching the PyTorch INT8 path.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers.pytorch_utils import Conv1D

OPSET = 18
INPUT_NAMES = ("input_ids", "past_key", "past_value")
OUTPUT_NAMES = ("logits", "present_key", "present_value")


def _project(layer: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if isinstance(layer, Conv1D):
        return torch.matmul(x, layer.weight) + layer.bias
    out: torch.Tensor = layer(x)
    return out


class GPT2StepModule(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        cfg = model.config  # type: ignore[union-attr]
        self.transformer: nn.Module = model.transformer  # type: ignore[assignment]
        self.lm_head: nn.Module = model.lm_head  # type: ignore[assignment]
        self.num_layers: int = cfg.n_layer
        self.num_heads: int = cfg.n_head
        self.embed_dim: int = cfg.n_embd
        self.head_dim = self.embed_dim // self.num_heads

    def forward(
        self, input_ids: torch.Tensor, past_key: torch.Tensor, past_value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tr = self.transformer
        b, t = input_ids.shape
        p = past_key.shape[3]
        positions = torch.arange(t, device=input_ids.device) + p
        hidden = tr.wte(input_ids) + tr.wpe(positions)  # type: ignore[operator]
        # Query i (absolute position p + i) may attend to keys 0 .. p + i.
        key_positions = torch.arange(p + t, device=input_ids.device)
        mask = key_positions[None, :] <= positions[:, None]  # (T, P + T)
        keys_out, values_out = [], []
        for layer, block in enumerate(tr.h):  # type: ignore[arg-type]
            qkv = _project(block.attn.c_attn, block.ln_1(hidden))
            q, k, v = qkv.split(self.embed_dim, dim=2)
            q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
            k = torch.cat([past_key[layer], k], dim=2)
            v = torch.cat([past_value[layer], v], dim=2)
            keys_out.append(k)
            values_out.append(v)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
            out = out.transpose(1, 2).reshape(b, t, self.embed_dim)
            hidden = hidden + _project(block.attn.c_proj, out)
            mlp = block.mlp
            hidden = hidden + _project(mlp.c_proj, mlp.act(_project(mlp.c_fc, block.ln_2(hidden))))
        last = tr.ln_f(hidden[:, -1, :])  # type: ignore[operator]
        logits: torch.Tensor = self.lm_head(last)
        return logits, torch.stack(keys_out), torch.stack(values_out)

    def empty_past(self, batch: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.zeros((self.num_layers, batch, self.num_heads, 0, self.head_dim), dtype=dtype)


def export_gpt2_step(model: nn.Module, path: Path, external_data: bool | None = None) -> Path:
    """Export ``model`` (FP32, CPU) as a ``GPT2StepModule`` ONNX graph.

    Graphs over 2 GB (GPT-2 XL) keep their weights in an external data file
    next to the ``.onnx`` file, as the protobuf format requires.
    """
    step = GPT2StepModule(model).eval()
    path.parent.mkdir(parents=True, exist_ok=True)
    example_ids = torch.randint(0, 1000, (1, 4))
    example_past = torch.zeros((step.num_layers, 1, step.num_heads, 3, step.head_dim))
    batch = torch.export.Dim("batch", min=1, max=64)
    seq = torch.export.Dim("seq", min=1, max=1024)
    past = torch.export.Dim("past", min=1, max=1024)
    if external_data is None:
        external_data = sum(p.numel() for p in model.parameters()) * 4 > 1.8 * 2**30
    with torch.no_grad():
        torch.onnx.export(
            step,
            (example_ids, example_past, example_past.clone()),
            str(path),
            input_names=list(INPUT_NAMES),
            output_names=list(OUTPUT_NAMES),
            dynamic_shapes={
                "input_ids": {0: batch, 1: seq},
                "past_key": {1: batch, 3: past},
                "past_value": {1: batch, 3: past},
            },
            opset_version=OPSET,
            dynamo=True,
            external_data=external_data,
            optimize=True,
            verbose=False,
        )
    return path
