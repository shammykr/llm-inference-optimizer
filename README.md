# LLM Inference Optimizer

A toolkit that applies common inference optimizations to transformer language
models and measures what each one does to latency, throughput, memory and
accuracy. It works on Hugging Face GPT-2 models of two sizes: 124M parameters
(`gpt2`) and 1.56B (`gpt2-xl`).

The optimizations:
- **INT8 dynamic quantization**, with PyTorch `torch.ao.quantization`
- **FP16 mixed precision**, with `torch.autocast`
- **a pre-allocated KV cache** that is reused across decoding steps
- **ONNX export**, compiled with **TensorRT** through ONNX Runtime's TensorRT
  execution provider. Without a GPU it falls back to plain ONNX Runtime on CPU.
- **ONNX Runtime INT8**

A profiling module records GPU memory and utilization through `torch.cuda`,
with a CPU fallback when no GPU is present. A benchmark runs every strategy
on every model size.

**Headline results.** These are for GPT-2 124M on CPU, generating 64 tokens one
at a time, against an FP32 baseline. Full tables are in
[Benchmark results](#benchmark-results).

| | Apple M5 (ARM) | x86-64 (GitHub Actions, Xeon 8370C) |
|---|---:|---:|
| PyTorch INT8 dynamic quantization | **1.27x** | **1.62x** |
| PyTorch INT8 + pre-allocated KV cache | **1.45x** | **1.89x** |
| ONNX Runtime INT8 | **2.76x** | **1.99x** |
| PyTorch INT8 + KV cache on GPT-2 XL (1.56B) | not run (out of RAM) | **2.76x** |

## Architecture

```
                         ┌──────────────────────────────────────────────┐
  Hugging Face Hub ────► │ optimizers/models.py   load GPT-2 (FP32)      │
  (gpt2, gpt2-xl)        └───────────────┬──────────────────────────────┘
                                         │
        ┌───────────────────┬────────────┼──────────────────┬───────────────────────┐
        ▼                   ▼            ▼                  ▼                       ▼
 quantization.py      quantization.py   kv_cache.py     onnx_export.py  ───►  tensorrt.py
 INT8 dynamic         FP16 autocast     KVCacheManager  one decoding step     ONNX Runtime session:
 (Conv1D → Linear,    (torch.autocast)  + StaticKVGPT2  with explicit KV      TensorRT EP → CUDA EP → CPU EP
  per-channel)                          decoder         inputs/outputs        + ONNX Runtime INT8
        │                   │            │                  │                       │
        └───────────────────┴────────────┴─────────┬────────┴───────────────────────┘
                                                   ▼
                          benchmarks/strategies.py: one Runner per strategy
                          (generate(ids, n), nll(ids) for perplexity)
                                                   │
                                                   ▼
            benchmarks/run_benchmarks.py: one subprocess per (model, strategy)
                 └─ profiling/profiler.py: CUDA peak allocated/reserved, GPU utilization,
                                           or CPU time / peak RSS; allocation tracing
                                                   │
                                                   ▼
                         results/*.json + Markdown tables (benchmarks/report.py)
```

```
optimizers/
  models.py          load GPT-2 family models; weight-size accounting (including packed INT8)
  quantization.py    INT8 dynamic quantization, FP16 autocast
  kv_cache.py        KVCacheManager (pre-allocated, reused K/V buffers) + StaticKVGPT2 decoder
  onnx_export.py     export one KV-cached decoding step to ONNX
  tensorrt.py        ONNX Runtime sessions (TensorRT / CUDA / CPU), ONNX INT8, ORT decoder
profiling/
  profiler.py        Profiler context manager, allocation tracing, device summary
benchmarks/
  strategies.py      the 9 strategies behind one Runner interface
  run_benchmarks.py  benchmark driver (subprocess per pair), JSON + Markdown output
  report.py          JSON → Markdown tables
  text.py            fixed prompt and perplexity text
scripts/
  compile_tensorrt.py  export → compile (TensorRT or CPU fallback) → parity check → timing
  smoke_test.py        CI smoke test: load model, one inference per path, check shapes
  generate.py          generate text with any strategy
results/             benchmark results (JSON + Markdown)
```

## Optimization strategies

### INT8 dynamic quantization (`optimizers/quantization.py`)

**What it does.** `torch.ao.quantization.quantize_dynamic` turns each
`nn.Linear` weight matrix into INT8 once, ahead of time. During inference,
each layer:
1. quantizes its input activations to INT8, using their observed range
2. runs an integer matrix multiply
3. converts the result back to floating point

**Changes for GPT-2:**
- **Conv1D layers.** Hugging Face GPT-2 builds its projections from
  `transformers.Conv1D` layers, which `quantize_dynamic` skips.
  `conv1d_to_linear` rewrites all 48 attention and MLP projections as
  `nn.Linear` first, so they get quantized.
- **Per-channel weight scales.** Each output row gets its own scale. This is
  more accurate than one scale for the whole tensor.
- **The LM head stays FP32.** It shares weights with the token embedding, and
  its outputs are the logits. Quantizing it roughly tripled perplexity in
  testing, so it's opt-in only: `quantize_lm_head=True`.

**When to use it:** CPU inference, especially for batch-1 token generation,
where each step is limited by how fast the weights can be read from memory.
- It needs no calibration data and no retraining.
- **Benefit:** weights take about a quarter of the space for the quantized
  layers.
- **Cost:** some accuracy, and the size of that cost depends on the model and
  the hardware (see the results).
- It runs on CPU only. The kernels are `fbgemm`/`x86` on Intel/AMD and
  `qnnpack` on ARM.

### FP16 mixed precision (`optimizers/quantization.py`)

**What it does.** Inside `torch.autocast(device_type, dtype=torch.float16)`,
matrix multiplies and other eligible ops run in half precision. Softmax, layer
norm and reductions stay in FP32. The weights stay FP32 and are cast on the
fly.

**When to use it:** on NVIDIA GPUs, where FP16 runs on tensor cores and halves
activation memory traffic. On a CPU without native FP16 matrix multiply,
autocast is much **slower**. The CPU benchmarks below show this.

### Pre-allocated KV cache (`optimizers/kv_cache.py`)

**What it does.** Attention needs the keys and values of every earlier token.
There are three ways to get them:

| Approach | What happens at each step | Cost |
|---|---|---|
| **No cache** (`fp32-no-cache`) | Recompute the keys and values of the whole prefix | O(n²) work for n tokens |
| **Hugging Face default** (`DynamicCache`) | Keep the keys and values, but grow them with `torch.cat` | A new, larger tensor and a full copy for every layer, every step |
| **`KVCacheManager`** (this project) | Write the new token's keys and values into the next slot of a buffer allocated once | No reallocation during generation |

How `KVCacheManager` works:
- **One allocation.** It allocates one K buffer and one V buffer, shaped
  `(layers, batch, heads, max_seq_len, head_dim)`.
- **In-place writes.** Each step writes the new token's keys and values into
  the next slot.
- **No copies on read.** Attention reads a view of the filled prefix.
- **Reuse.** `reset()` lets the next generation reuse the same memory.
  It reallocates only when a request needs a bigger batch or a longer
  sequence.

`StaticKVGPT2` is a greedy decoder that runs the Hugging Face GPT-2 modules
with this cache and `F.scaled_dot_product_attention`. It produces exactly the
same tokens as `model.generate` (checked by the smoke test).

**When to use it:** for autoregressive generation, always use some KV cache.
Pre-allocating it pays off when:
- requests are long or many generations run back to back
- allocator overhead matters, e.g. on GPU, where repeated allocations fragment
  the caching allocator
- you need predictable peak memory

### ONNX export and TensorRT compilation (`optimizers/onnx_export.py`, `optimizers/tensorrt.py`)

**Export.** `GPT2StepModule` exports one decoding step as a pure tensor
function, so the KV cache is visible to the graph compiler:

```
(input_ids, past_key, past_value) → (logits, present_key, present_value)
```

Batch size, step length and past length are all dynamic. Projections export as
`MatMul` + `Add`, which is the form ONNX Runtime's INT8 quantizer recognises.

**Session.** `build_session` picks the best execution provider available:

1. **TensorRT.** It compiles the graph into an engine for the GPU, fusing
   layers, choosing kernels, and enabling FP16 and an optimisation profile
   for the dynamic KV shapes. The engine is cached on disk.
2. **CUDA**, when TensorRT isn't installed.
3. **CPU.** This is the fallback with no GPU. ONNX Runtime still applies its
   graph optimizations.

**INT8.** `quantize_onnx_int8` applies ONNX Runtime's own dynamic INT8
quantization, which runs on its MLAS integer kernels:
- **Untied embedding.** Before quantizing, it gives the token embedding its
  own copy of the weight it shares with the LM head.
- **FP32 head.** It leaves the LM head in FP32.

**When to use it:**
- **TensorRT:** fixed models served on NVIDIA GPUs, where compile time is
  repaid over many requests.
- **ONNX Runtime on CPU:** framework-free CPU deployment.

```sh
python -m scripts.compile_tensorrt --model gpt2 --int8              # CPU fallback here
python -m scripts.compile_tensorrt --model gpt2 --provider tensorrt # on an NVIDIA GPU
```

The script checks the compiled session against PyTorch:
- **Logits:** max absolute difference of 1e-4 on CPU FP32.
- **Tokens:** a short greedy continuation must match.

It also times the session against PyTorch eager.

## Profiling (`profiling/profiler.py`)

`Profiler(device)` is a context manager. What it records depends on the
device:

| On CUDA | On CPU (fallback) |
|---|---|
| CUDA-event wall time | Wall time |
| `torch.cuda.max_memory_allocated` and `max_memory_reserved` | Process CPU utilization (CPU time over wall time, so 200% means two busy cores) |
| The allocator's allocation count | Peak RSS, sampled every 5 ms with `psutil` |
| Device memory in use, from `mem_get_info`. This also counts ONNX Runtime and TensorRT allocations, which PyTorch can't see. | |
| `torch.cuda.utilization()`, sampled every 10 ms (needs `pynvml`) | |

`allocation_profile(fn, device)` runs `fn` under `torch.profiler` with
`profile_memory=True`. It reports the MB PyTorch allocated and the number of
`aten::cat` calls, which is how the benchmark makes KV-cache allocation
overhead visible.

## Benchmark results

**Method.** `python -m benchmarks.run_benchmarks`:
- Each (model, strategy) pair runs in its own subprocess, so memory is
  measured cleanly.
- **Workload:** batch 1, a 32-token prompt, 64 new tokens with greedy
  decoding, median of 3 runs after a warm-up.
- **`ms/token`:** end-to-end generation time divided by new tokens, prefill
  included.
- **Speedup:** the FP32 baseline's `ms/token` divided by the strategy's.
- **Perplexity:** measured on a fixed 200-token paragraph (`benchmarks/text.py`).
- **Peak memory:** process RSS during generation. It can include memory the
  allocator kept after setup; for example, the INT8 strategies first load the
  FP32 model. The weights column is the cleaner measure of model size.

### Apple M5, macOS 26 (ARM CPU, 4 PyTorch threads, `qnnpack` INT8 kernels)

`results/benchmark-apple-m5-cpu.json`

**gpt2** (124M parameters)

| Strategy | ms/token | tokens/s | Speedup vs FP32 | Prefill (ms) | Peak memory (MB) | Weights (MB) | Perplexity | Allocated during generation (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fp32` (baseline) | 8.72 | 114.7 | 1.00x | 19.7 | 772 | 475 | 58.86 | 571.5 |
| `fp32-no-cache` | 29.66 | 33.7 | 0.29x | 18.0 | 850 | 475 | 58.86 | 6,800.5 |
| `fp32-static-kv` | 7.56 | 132.2 | **1.15x** | 17.7 | 856 | 475 | 58.86 | 169.1 |
| `fp16-autocast` | 118.41 | 8.4 | 0.07x | 2,223.4 | 926 | 475 | 58.90 | 15,431.0 |
| `int8-dynamic` | 6.88 | 145.4 | **1.27x** | 21.4 | 1,341 | 232 | 63.50 | 577.6 |
| `int8-dynamic-static-kv` | 6.02 | 166.1 | **1.45x** | 20.1 | 1,332 | 232 | 63.50 | 175.2 |
| `onnxruntime` | 8.22 | 121.7 | 1.06x | 8.8 | 1,163 | 475 | 58.86 | - |
| `onnxruntime-int8` | 3.16 | 316.9 | **2.76x** | 5.7 | 781 | 270 | 89.38 | - |
| `tensorrt` | not run: no NVIDIA GPU | | | | | | | |

**gpt2-xl was not run on this machine.** With the other apps running, the 6 GB
model didn't fit in the free RAM of this 16 GB Mac, and the runs swapped
heavily. Use the x86 results below for the large model.

### x86-64: GitHub Actions `ubuntu-latest` (Intel Xeon Platinum 8370C, 2 threads, `x86`/fbgemm INT8 kernels)

`results/benchmark-x86-github-actions.json`. Produced by
`.github/workflows/benchmark.yml` (run it from the Actions tab).

**gpt2** (124M parameters)

| Strategy | ms/token | tokens/s | Speedup vs FP32 | Prefill (ms) | Peak memory (MB) | Weights (MB) | Perplexity | Allocated during generation (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fp32` (baseline) | 23.18 | 43.1 | 1.00x | 49.7 | 996 | 475 | 58.86 | 570.5 |
| `fp32-no-cache` | 70.88 | 14.1 | 0.33x | 50.0 | 980 | 475 | 58.86 | 6,776.2 |
| `fp32-static-kv` | 20.40 | 49.0 | **1.14x** | 48.9 | 975 | 475 | 58.86 | 168.0 |
| `fp16-autocast` | 395.47 | 2.5 | 0.06x | 3,355.3 | 1,063 | 475 | 58.93 | 15,429.7 |
| `int8-dynamic` | 14.31 | 69.9 | **1.62x** | 25.5 | 1,075 | 232 | 83.33 | 600.6 |
| `int8-dynamic-static-kv` | 12.27 | 81.5 | **1.89x** | 22.7 | 1,074 | 232 | 83.33 | 198.1 |
| `onnxruntime` | 20.09 | 49.8 | 1.15x | 37.5 | 1,730 | 475 | 58.86 | - |
| `onnxruntime-int8` | 11.67 | 85.7 | **1.99x** | 23.2 | 813 | 270 | 86.95 | - |
| `tensorrt` | not run: no NVIDIA GPU | | | | | | | |

**gpt2-xl** (1,558M parameters)

| Strategy | ms/token | tokens/s | Speedup vs FP32 | Prefill (ms) | Peak memory (MB) | Weights (MB) | Perplexity | Allocated during generation (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fp32` (baseline) | 238.99 | 4.2 | 1.00x | 728.4 | 6,613 | 5,942 | 39.10 | 3,797.4 |
| `fp32-no-cache` | 1,051.13 | 0.9 | 0.23x | 721.2 | 6,531 | 5,942 | 39.10 | 55,115.7 |
| `fp32-static-kv` | 235.23 | 4.2 | **1.02x** | 737.8 | 6,585 | 5,942 | 39.10 | 1,299.3 |
| `int8-dynamic` | 92.12 | 10.8 | **2.59x** | 214.0 | 8,185 | 1,723 | 39.80 | 4,047.9 |
| `int8-dynamic-static-kv` | 86.57 | 11.6 | **2.76x** | 209.8 | 8,090 | 1,723 | 39.80 | 1,549.8 |
| `onnxruntime` | 228.01 | 4.4 | 1.05x | 511.1 | 13,159 | 5,944 | 39.10 | - |
| `fp16-autocast`, `onnxruntime-int8` | skipped on CI: too slow, or too much RAM for a 16 GB runner | | | | | | | |
| `tensorrt` | not run: no NVIDIA GPU | | | | | | | |

### What the numbers show

- **INT8 helps most when generation is memory-bound.** On x86, PyTorch INT8
  gives 1.62x on the 124M model and 2.59x on the 1.56B model. With the
  pre-allocated KV cache added, that rises to 1.89x and 2.76x. The larger the
  model, the more each step's time goes on reading weights. The 124M model
  also keeps its 154 MB FP32 LM head and embedding, which caps its gain.
- **The accuracy cost of INT8 depends on the model and the hardware.**
  Perplexity changes:

  | Model and hardware | FP32 | INT8 | Change |
  |---|---:|---:|---:|
  | GPT-2 XL, x86 | 39.10 | 39.80 | +1.8% |
  | GPT-2 124M, ARM | 58.86 | 63.50 | +8% |
  | GPT-2 124M, x86 | 58.86 | 83.33 | +42% |

  The x86 and ARM INT8 kernels quantize activations differently. Validate
  quality, not only speed, on the hardware you'll deploy on.
- **ONNX Runtime INT8** was the fastest option for the small model: 2.76x on
  the M5 and 1.99x on x86. Its perplexity cost was larger, +48–52%.
- **The KV cache matters most.** Removing it makes generation 3–4.5x slower,
  getting worse as sequences grow. Pre-allocating the cache:
  - removes every `aten::cat` call (1,664 for GPT-2 small and 6,272 for GPT-2
    XL over 64 tokens)
  - cuts the memory PyTorch allocates during generation by about 3x (571 MB to
    169 MB, and 3.8 GB to 1.3 GB)
  - makes generation 2–15% faster. On CPU the allocator is cheap. On GPU, the
    allocation savings are the larger benefit.
- **FP16 autocast on CPU is 14–17x slower** than FP32, because these CPUs have
  no fast FP16 matrix multiply. FP16 belongs on the GPU.
- **ONNX Runtime FP32** is 5–15% faster than PyTorch eager, through graph
  optimizations alone.

### Not measured

- **CUDA, TensorRT and the CUDA profiling path have not been run.** The code
  paths exist, but neither machine has an NVIDIA GPU:
  - the TensorRT execution provider with engine cache and dynamic-shape
    profiles
  - the CUDA provider
  - CUDA memory and utilization profiling
  - FP16 autocast on GPU

  On a CUDA machine, `docker build --target gpu` and
  `python -m benchmarks.run_benchmarks` fill in those rows automatically.
  The benchmark reports `tensorrt` as skipped when its provider is missing.

## Setup

Requirements: Python 3.10+. A CUDA GPU is optional.

```sh
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch   # or the CUDA wheel for your GPU
pip install -r requirements-dev.txt

python -m scripts.smoke_test                                     # load GPT-2, check every path
python -m scripts.generate --strategy int8-dynamic-static-kv     # generate text with a strategy
python -m scripts.compile_tensorrt --model gpt2 --int8           # ONNX export + compile + parity
python -m benchmarks.run_benchmarks --models gpt2 gpt2-xl        # full benchmark (gpt2-xl needs ~16 GB RAM)
python -m benchmarks.run_benchmarks --models gpt2 --strategies fp32 int8-dynamic --repeats 5
```

Models download from the Hugging Face Hub on first use: about 0.5 GB for
`gpt2` and 6.4 GB for `gpt2-xl`. For GPU TensorRT support, install
`onnxruntime-gpu` and `tensorrt-cu12` in place of `onnxruntime`, as the
Docker `gpu` target does.

## Docker

```sh
# CPU image (any machine). GPT-2 small is baked in, so the smoke test runs offline.
docker build --target cpu -t llm-inference-optimizer:cpu .
docker run --rm llm-inference-optimizer:cpu scripts.smoke_test
docker run --rm -v "$PWD/results:/app/results" llm-inference-optimizer:cpu \
    benchmarks.run_benchmarks --models gpt2

# GPU image: CUDA 12.8, PyTorch, onnxruntime-gpu and TensorRT (needs the NVIDIA Container Toolkit)
docker build --target gpu -t llm-inference-optimizer:gpu .
docker run --rm --gpus all llm-inference-optimizer:gpu scripts.compile_tensorrt --provider tensorrt
docker run --rm --gpus all -v "$PWD/results:/app/results" llm-inference-optimizer:gpu \
    benchmarks.run_benchmarks --models gpt2 gpt2-xl
```

The entrypoint is `python -m`, so the command picks the module to run. CI
builds the CPU image and runs the smoke test inside it. The GPU image hasn't
been built or run, because no GPU was available.

## CI

`.github/workflows/ci.yml` runs on every push and pull request:

- **lint:** `ruff check`, `ruff format --check` and `mypy`, with typed
  function definitions required.
- **smoke:**
  - Loads GPT-2 and runs one inference per path (FP32, static KV cache, INT8,
    ONNX Runtime).
  - Checks output shapes, parity with Hugging Face, and the INT8 layer count.
  - Runs a quick FP32 vs INT8 benchmark.
- **docker:** builds the CPU image and runs the smoke test inside it.

`.github/workflows/benchmark.yml` (manual trigger) runs the full benchmark on
an x86 runner and uploads the results.

## Tech stack

| Area | Technology |
|---|---|
| Language | Python 3.10+ |
| Models | Hugging Face Transformers, GPT-2 (124M) and GPT-2 XL (1.56B) |
| Framework | PyTorch 2.x: `torch.ao.quantization` (dynamic INT8), `torch.autocast` (FP16), SDPA, `torch.profiler` |
| Compilation | ONNX (opset 18, dynamo exporter), ONNX Runtime (CPU, CUDA and TensorRT execution providers, INT8 quantization), TensorRT |
| Profiling | `torch.cuda` memory stats and utilization, `psutil` |
| Packaging | Docker (CPU target and CUDA/TensorRT target) |
| CI | GitHub Actions, ruff, mypy |

## License

MIT
