"""Pre-allocated KV cache and a GPT-2 decoding loop that uses it.

Without a cache, each decoding step re-runs attention over the whole prefix to
rebuild the keys and values of every earlier token: O(n^2) work for n tokens.

A KV cache keeps those keys and values. Hugging Face's default
(``DynamicCache``) grows them with ``torch.cat`` on every step, which allocates
a new, larger tensor per layer per step and copies the old one into it.

``KVCacheManager`` instead allocates **one** buffer per K and V, shaped
``(layers, batch, heads, max_seq_len, head_dim)``:

* each step writes the new token's keys and values into the next slot in place
* attention reads a view of the filled prefix, with no copy
* ``reset()`` rewinds the write position so the next generation reuses the
  same memory; it reallocates only for a larger batch or a longer sequence

``StaticKVGPT2`` is a small greedy decoder that runs the Hugging Face GPT-2
modules (embeddings, ``c_attn``, ``c_proj``, MLP, layer norms, LM head) with
this cache and ``F.scaled_dot_product_attention``. Because it calls the
modules rather than copying their weights, it works on an FP32, FP16-autocast
or INT8-quantized model unchanged.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class CacheStats:
    allocations: int = 0  # times the K/V buffers were (re)allocated
    reuses: int = 0  # generations that reused existing buffers
    bytes_allocated: int = 0
    tokens_written: int = 0
    history: list[str] = field(default_factory=list)


class KVCacheManager:
    """Fixed-size K/V buffers, written in place and reused across generations."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        max_batch: int = 1,
        max_seq_len: int = 1024,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = torch.device(device)
        self.stats = CacheStats()
        self.keys = torch.empty(0)
        self.values = torch.empty(0)
        self.max_batch = 0
        self.max_seq_len = 0
        self.batch = 0
        self.length = 0
        self._allocate(max_batch, max_seq_len)

    def _allocate(self, batch: int, seq_len: int) -> None:
        shape = (self.num_layers, batch, self.num_heads, seq_len, self.head_dim)
        self.keys = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.values = torch.zeros(shape, dtype=self.dtype, device=self.device)
        self.max_batch, self.max_seq_len = batch, seq_len
        self.stats.allocations += 1
        self.stats.bytes_allocated += 2 * self.keys.numel() * self.keys.element_size()
        self.stats.history.append(f"allocate batch={batch} seq_len={seq_len}")

    @property
    def nbytes(self) -> int:
        return 2 * self.keys.numel() * self.keys.element_size()

    def reset(self, batch: int, seq_len: int, dtype: torch.dtype | None = None) -> None:
        """Prepare for a new generation of ``batch`` sequences up to ``seq_len`` tokens.

        The buffers are reused when they are large enough and of the right dtype.
        """
        if dtype is not None and dtype != self.dtype:
            self.dtype = dtype
            self._allocate(max(batch, self.max_batch), max(seq_len, self.max_seq_len))
        elif batch > self.max_batch or seq_len > self.max_seq_len:
            self._allocate(max(batch, self.max_batch), max(seq_len, self.max_seq_len))
        else:
            self.stats.reuses += 1
        self.batch = batch
        self.length = 0

    def update(
        self, layer: int, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write ``key``/``value`` (B, H, T, D) at the current position.

        Returns views of the keys and values filled so far for this layer,
        without copying them. Call ``advance(T)`` once every layer has been
        updated.
        """
        t = key.shape[2]
        end = self.length + t
        if end > self.max_seq_len:
            raise ValueError(f"KV cache full: {end} > {self.max_seq_len} tokens")
        self.keys[layer, : self.batch, :, self.length : end].copy_(key)
        self.values[layer, : self.batch, :, self.length : end].copy_(value)
        return (
            self.keys[layer, : self.batch, :, :end],
            self.values[layer, : self.batch, :, :end],
        )

    def advance(self, tokens: int) -> None:
        self.length += tokens
        self.stats.tokens_written += tokens * self.batch


class StaticKVGPT2:
    """Greedy GPT-2 decoder over a pre-allocated ``KVCacheManager``."""

    def __init__(
        self,
        model: nn.Module,
        max_seq_len: int | None = None,
        autocast: Callable[[], contextlib.AbstractContextManager[None]] | None = None,
    ) -> None:
        self.model = model
        cfg = model.config  # type: ignore[union-attr]
        self.transformer: nn.Module = model.transformer  # type: ignore[assignment]
        self.lm_head: nn.Module = model.lm_head  # type: ignore[assignment]
        self.num_heads: int = cfg.n_head
        self.embed_dim: int = cfg.n_embd
        self.head_dim = self.embed_dim // self.num_heads
        self.max_positions: int = cfg.n_positions
        self.autocast = autocast or contextlib.nullcontext
        device = self.transformer.wte.weight.device  # type: ignore[union-attr]
        self.cache = KVCacheManager(
            num_layers=cfg.n_layer,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            max_batch=1,
            max_seq_len=max_seq_len or 128,
            device=device,
        )

    def _forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run ``input_ids`` (B, T) through the model, append K/V, return last logits (B, V)."""
        tr = self.transformer
        cache = self.cache
        b, t = input_ids.shape
        start = cache.length
        positions = torch.arange(start, start + t, device=input_ids.device)
        hidden = tr.wte(input_ids) + tr.wpe(positions)  # type: ignore[operator]
        # The prompt (T > 1, empty cache) needs a causal mask. A single new
        # token attends to the whole prefix, so it needs none.
        causal = t > 1 and start == 0
        if t > 1 and start > 0:
            raise ValueError("multi-token steps are only supported for the prompt")
        for layer, block in enumerate(tr.h):  # type: ignore[arg-type]
            attn = block.attn
            qkv = attn.c_attn(block.ln_1(hidden))
            q, k, v = qkv.split(self.embed_dim, dim=2)
            q = q.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
            keys, values = cache.update(layer, k.to(cache.dtype), v.to(cache.dtype))
            out = F.scaled_dot_product_attention(
                q, keys.to(q.dtype), values.to(q.dtype), is_causal=causal
            )
            out = out.transpose(1, 2).reshape(b, t, self.embed_dim)
            hidden = hidden + attn.c_proj(out)
            hidden = hidden + block.mlp(block.ln_2(hidden))
        cache.advance(t)
        last = tr.ln_f(hidden[:, -1:, :])  # type: ignore[operator]
        logits: torch.Tensor = self.lm_head(last)[:, -1, :]
        return logits

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Greedy-decode ``max_new_tokens`` tokens. Returns (B, T + max_new_tokens)."""
        b, t = input_ids.shape
        total = t + max_new_tokens
        if total > self.max_positions:
            raise ValueError(f"{total} tokens exceeds the model's {self.max_positions} positions")
        with self.autocast():
            probe_dtype = self.transformer.wte(input_ids[:, :1]).dtype  # type: ignore[operator]
            qkv_dtype = torch.float16 if torch.is_autocast_enabled() else probe_dtype
            self.cache.reset(b, total, dtype=qkv_dtype)
            out = torch.empty((b, total), dtype=input_ids.dtype, device=input_ids.device)
            out[:, :t] = input_ids
            logits = self._forward(input_ids)
            for i in range(max_new_tokens):
                next_token = logits.argmax(dim=-1)
                out[:, t + i] = next_token
                if i + 1 < max_new_tokens:
                    logits = self._forward(next_token[:, None])
        return out
