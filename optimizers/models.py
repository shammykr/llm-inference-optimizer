"""Loading GPT-2 family models from the Hugging Face Hub."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Short names used by the benchmark. Parameter counts are from the model cards.
MODEL_ALIASES: dict[str, str] = {
    "gpt2": "openai-community/gpt2",  # 124M
    "gpt2-medium": "openai-community/gpt2-medium",  # 355M
    "gpt2-large": "openai-community/gpt2-large",  # 774M
    "gpt2-xl": "openai-community/gpt2-xl",  # 1.56B
}


@dataclass
class LoadedModel:
    name: str
    model: Any  # transformers PreTrainedModel (GPT2LMHeadModel)
    tokenizer: Any
    num_parameters: int


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def quiet_hf() -> None:
    """Silence download progress bars and noisy warnings in benchmark output."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


def load_model(name: str, device: torch.device | str = "cpu") -> LoadedModel:
    """Load a GPT-2 family model in FP32 and eval mode."""
    quiet_hf()
    repo = MODEL_ALIASES.get(name, name)
    tokenizer = AutoTokenizer.from_pretrained(repo)
    model: Any = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32)
    model = model.to(device).eval()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.eos_token_id
    num_params = sum(p.numel() for p in model.parameters())
    return LoadedModel(name, model, tokenizer, num_params)


def model_size_mb(model: torch.nn.Module) -> float:
    """Size of the weights, counting packed INT8 weights of quantized layers.

    Uses ``state_dict`` because quantized dynamic Linear layers store packed
    weights there instead of in ``parameters()``. Tied weights (GPT-2's
    embedding and LM head) are counted once.
    """
    seen: set[int] = set()
    total = 0
    for value in model.state_dict().values():
        tensors: list[torch.Tensor] = []
        if isinstance(value, torch.Tensor):
            tensors = [value]
        elif isinstance(value, tuple):  # packed params: (weight, bias)
            tensors = [v for v in value if isinstance(v, torch.Tensor)]
        for t in tensors:
            if t.is_quantized:
                nbytes = t.int_repr().numel() * t.element_size()
                key = id(t)
            else:
                nbytes = t.numel() * t.element_size()
                key = t.untyped_storage().data_ptr()
            if key in seen:
                continue
            seen.add(key)
            total += nbytes
    return total / 2**20
