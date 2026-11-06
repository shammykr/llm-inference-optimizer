"""Render benchmark JSON results as Markdown tables.

Usage::

    python -m benchmarks.report results/benchmark-cpu.json [more.json ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.strategies import BASELINE


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.{digits}f}"
    return str(value)


def render(report: dict[str, Any]) -> str:
    env = report["environment"]
    wl = report["workload"]
    lines = [
        f"**{report.get('label') or env.get('machine', '')}**: device `{env['device']}`, "
        f"torch {env['torch']}, {env['threads']} threads. Batch {wl['batch_size']}, "
        f"{wl['prompt_tokens']}-token prompt, {wl['new_tokens']} new tokens (greedy), "
        f"median of {wl['repeats']} runs.",
        "",
    ]
    for model, rows in report["results"].items():
        params = next((r.get("num_parameters") for r in rows if r.get("num_parameters")), None)
        size = f" ({params / 1e6:,.0f}M parameters)" if params else ""
        lines += [
            f"**{model}**{size}",
            "",
            "| Strategy | ms/token | tokens/s | Speedup vs FP32 | Prefill (ms) | Peak memory (MB) "
            "| Weights (MB) | Perplexity | Allocated during generation (MB) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            if r.get("status") != "ok":
                lines.append(
                    f"| `{r['strategy']}` | {r.get('status')}: {r.get('reason', '')} "
                    "| | | | | | | |"
                )
                continue
            mem = r.get("cuda_peak_allocated_mb") or r.get("peak_rss_mb")
            speed = r.get("speedup_vs_fp32")
            speed_s = f"**{speed:.2f}x**" if r["strategy"] != BASELINE and speed else "1.00x"
            lines.append(
                f"| `{r['strategy']}` | {_fmt(r['ms_per_token'])} | {_fmt(r['tokens_per_s'], 1)} "
                f"| {speed_s} | {_fmt(r['prefill_ms'], 1)} | {_fmt(mem, 0)} "
                f"| {_fmt(r['weights_mb'], 0)} | {_fmt(r.get('perplexity'))} "
                f"| {_fmt(r.get('allocated_mb'), 1)} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", type=Path, nargs="+")
    args = parser.parse_args()
    for f in args.files:
        print(render(json.loads(f.read_text())))


if __name__ == "__main__":
    main()
