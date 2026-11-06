"""Benchmark token generation for every optimization strategy across model sizes.

Each (model, strategy) pair runs in its **own subprocess**, so its memory
numbers aren't inflated by models or sessions left over from earlier runs, and
one failing strategy can't take down the rest.

For each pair the worker:

1. loads the FP32 model and applies the strategy (``benchmarks/strategies.py``)
2. warms up
3. times greedy generation of ``--new-tokens`` tokens from a fixed
   ``--prompt-tokens`` prompt, ``--repeats`` times, inside ``profiling.Profiler``
4. times a 1-token generation separately as the prefill (time to first token)
5. traces PyTorch allocations made during one generation, for PyTorch
   strategies
6. computes perplexity on a fixed paragraph, as an accuracy check

Reported per pair:

* ``ms_per_token``: end-to-end generation time divided by new tokens. This is
  the serving latency per token, prefill included.
* ``tokens_per_s``: batch x new tokens divided by generation time
* ``speedup_vs_fp32``: FP32 baseline ``ms_per_token`` over this strategy's
* memory: CUDA peak allocated/reserved, or peak process RSS on CPU
* weights size, perplexity, MB allocated during generation, ``aten::cat`` calls

Usage::

    python -m benchmarks.run_benchmarks --models gpt2 gpt2-xl
    python -m benchmarks.run_benchmarks --models gpt2 --strategies fp32 int8-dynamic --repeats 5
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import gc
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

from benchmarks.report import render
from benchmarks.strategies import BASELINE, STRATEGIES, StrategyUnavailable, build_runner
from benchmarks.text import EVAL_TEXT
from optimizers.models import load_model, resolve_device
from profiling.profiler import Profiler, allocation_profile, device_summary

RESULT_MARKER = "@@RESULT@@"


def release_freed_memory() -> None:
    """Return memory freed during setup (e.g. FP32 weights replaced by INT8) to the OS.

    Otherwise the allocator keeps it, and peak RSS would include weights the
    strategy no longer uses.
    """
    gc.collect()
    if sys.platform.startswith("linux"):
        with contextlib.suppress(OSError, AttributeError):
            ctypes.CDLL("libc.so.6").malloc_trim(0)


def prompt_ids(tokenizer: Any, prompt_tokens: int, batch: int) -> torch.Tensor:
    ids = tokenizer(EVAL_TEXT, return_tensors="pt").input_ids[:, :prompt_tokens]
    return ids.repeat(batch, 1)


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    if args.threads:
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    row: dict[str, Any] = {"model": args.model, "strategy": args.strategy}
    t0 = time.perf_counter()
    lm = load_model(args.model, device)
    row["num_parameters"] = lm.num_parameters
    tokenizer = lm.tokenizer
    try:
        runner = build_runner(args.strategy, lm, device, Path(args.artifacts), args.threads)
    except StrategyUnavailable as exc:
        return {**row, "status": "skipped", "reason": str(exc)}
    row["setup_s"] = round(time.perf_counter() - t0, 1)
    release_freed_memory()
    ids = prompt_ids(tokenizer, args.prompt_tokens, args.batch_size).to(device)
    eval_ids = tokenizer(EVAL_TEXT, return_tensors="pt").input_ids[:, : args.eval_tokens].to(device)
    n = args.new_tokens

    runner.generate(ids, min(8, n))  # warm-up: kernels, allocator, ORT arenas
    prefill = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        runner.generate(ids, 1)
        prefill.append(time.perf_counter() - start)

    times: list[float] = []
    with Profiler(device) as prof:
        for _ in range(args.repeats):
            start = time.perf_counter()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            out = runner.generate(ids, n)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append(time.perf_counter() - start)
    gen_s = statistics.median(times)

    row.update(
        {
            "status": "ok",
            "engine": runner.engine,
            "info": runner.info,
            "ms_per_token": round(gen_s / n * 1000, 3),
            "tokens_per_s": round(args.batch_size * n / gen_s, 2),
            "prefill_ms": round(statistics.median(prefill) * 1000, 2),
            "generation_s_runs": [round(t, 4) for t in times],
            "weights_mb": round(runner.weights_mb, 1),
            "generated_token_ids": out[0, ids.shape[1] :].tolist(),
            **{k: v for k, v in prof.result.to_dict().items() if k not in ("device", "wall_s")},
        }
    )
    if runner.traces_allocations and not args.skip_alloc_trace:
        alloc = allocation_profile(lambda: runner.generate(ids, n), device)
        row.update({"allocated_mb": alloc.allocated_mb, "cat_calls": alloc.cat_calls})
    if not args.skip_perplexity:
        row["perplexity"] = round(math.exp(runner.nll(eval_ids)), 3)
    return row


def run_pair(model: str, strategy: str, args: argparse.Namespace) -> dict[str, Any]:
    cmd = [
        sys.executable, "-m", "benchmarks.run_benchmarks", "--worker",
        "--model", model, "--strategy", strategy,
        "--device", args.device, "--artifacts", str(args.artifacts),
        "--batch-size", str(args.batch_size), "--prompt-tokens", str(args.prompt_tokens),
        "--new-tokens", str(args.new_tokens), "--repeats", str(args.repeats),
        "--eval-tokens", str(args.eval_tokens),
    ]  # fmt: skip
    if args.threads:
        cmd += ["--threads", str(args.threads)]
    if args.skip_perplexity:
        cmd.append("--skip-perplexity")
    if args.skip_alloc_trace:
        cmd.append("--skip-alloc-trace")
    env = {**os.environ, "PYTHONWARNINGS": "ignore", "TOKENIZERS_PARALLELISM": "false"}
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=args.timeout, env=env, check=False
        )
    except subprocess.TimeoutExpired:
        return {"model": model, "strategy": strategy, "status": "timeout",
                "reason": f">{args.timeout}s"}  # fmt: skip
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_MARKER):
            result: dict[str, Any] = json.loads(line[len(RESULT_MARKER) :])
            return result
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-3:]
    return {"model": model, "strategy": strategy, "status": "error", "reason": " | ".join(tail)}


def add_comparisons(rows: list[dict[str, Any]]) -> None:
    base = next((r for r in rows if r["strategy"] == BASELINE and r.get("status") == "ok"), None)
    if base is None:
        return
    base_tokens = base["generated_token_ids"]
    for r in rows:
        if r.get("status") != "ok":
            continue
        r["speedup_vs_fp32"] = round(base["ms_per_token"] / r["ms_per_token"], 3)
        same = sum(a == b for a, b in zip(r["generated_token_ids"], base_tokens, strict=False))
        r["token_agreement_vs_fp32"] = round(same / max(len(base_tokens), 1), 3)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--models", nargs="+", default=["gpt2", "gpt2-xl"])
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES))
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--threads", type=int, default=None, help="CPU threads (torch and ORT)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--new-tokens", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--eval-tokens", type=int, default=200, help="tokens scored for perplexity")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--timeout", type=float, default=3600.0, help="seconds per pair")
    parser.add_argument("--skip-perplexity", action="store_true")
    parser.add_argument("--skip-alloc-trace", action="store_true")
    parser.add_argument("--skip", nargs="*", default=[], help="model:strategy pairs to skip")
    parser.add_argument("--label", default="")
    parser.add_argument("--out", type=Path, default=Path("results/benchmark.json"))
    # Internal: run a single pair in this process.
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--model", help=argparse.SUPPRESS)
    parser.add_argument("--strategy", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        print(RESULT_MARKER + json.dumps(run_worker(args)), flush=True)
        return

    if args.threads:
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    env = device_summary(device)
    env["machine"] = f"{platform.system()} {platform.machine()} ({platform.processor() or '?'})"
    env["python"] = platform.python_version()
    report: dict[str, Any] = {
        "label": args.label,
        "environment": env,
        "workload": {
            "batch_size": args.batch_size,
            "prompt_tokens": args.prompt_tokens,
            "new_tokens": args.new_tokens,
            "repeats": args.repeats,
            "eval_tokens": args.eval_tokens,
        },
        "results": {},
    }
    for model in args.models:
        rows: list[dict[str, Any]] = []
        for strategy in args.strategies:
            if f"{model}:{strategy}" in args.skip:
                rows.append({"model": model, "strategy": strategy, "status": "skipped",
                             "reason": "skipped by --skip"})  # fmt: skip
                continue
            start = time.perf_counter()
            row = run_pair(model, strategy, args)
            rows.append(row)
            took = time.perf_counter() - start
            if row.get("status") == "ok":
                print(
                    f"{model:>10} {strategy:<24} {row['ms_per_token']:9.2f} ms/token "
                    f"{row['tokens_per_s']:9.1f} tok/s  ppl {row.get('perplexity', '-')}  "
                    f"({took:.0f}s)",
                    flush=True,
                )
            else:
                print(f"{model:>10} {strategy:<24} {row['status']}: {row.get('reason')}",
                      flush=True)  # fmt: skip
        add_comparisons(rows)
        report["results"][model] = rows
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")  # save after every model

    markdown = render(report)
    args.out.with_suffix(".md").write_text(markdown + "\n")
    print()
    print(markdown)
    print(f"wrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
