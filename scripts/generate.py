"""Generate text with any optimization strategy and print the timing.

Usage::

    python -m scripts.generate --strategy int8-dynamic-static-kv --prompt "Inference is"
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from benchmarks.strategies import STRATEGIES, build_runner
from optimizers.models import load_model, resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--strategy", choices=STRATEGIES, default="fp32")
    parser.add_argument("--prompt", default="The fastest way to run a language model is")
    parser.add_argument("--new-tokens", type=int, default=40)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()

    device = resolve_device(args.device)
    lm = load_model(args.model, device)
    tokenizer = lm.tokenizer
    runner = build_runner(args.strategy, lm, device, args.artifacts)
    ids = tokenizer(args.prompt, return_tensors="pt").input_ids.to(device)
    runner.generate(ids, 4)
    start = time.perf_counter()
    out = runner.generate(ids, args.new_tokens)
    elapsed = time.perf_counter() - start
    print(tokenizer.decode(out[0]))
    print(
        f"\n[{args.strategy} on {runner.info.get('provider', device.type)}] "
        f"{args.new_tokens} tokens in {elapsed:.2f}s "
        f"= {elapsed / args.new_tokens * 1000:.2f} ms/token"
    )


if __name__ == "__main__":
    main()
