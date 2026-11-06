**Apple M5 (10-core CPU, 16 GB), macOS 26**: device `cpu`, torch 2.15.0, 4 threads. Batch 1, 32-token prompt, 64 new tokens (greedy), median of 3 runs.

**gpt2** (124M parameters)

| Strategy | ms/token | tokens/s | Speedup vs FP32 | Prefill (ms) | Peak memory (MB) | Weights (MB) | Perplexity | Allocated during generation (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fp32` | 8.72 | 114.7 | 1.00x | 19.7 | 772 | 475 | 58.86 | 571.5 |
| `fp32-no-cache` | 29.66 | 33.7 | **0.29x** | 18.0 | 850 | 475 | 58.86 | 6,800.5 |
| `fp32-static-kv` | 7.56 | 132.2 | **1.15x** | 17.7 | 856 | 475 | 58.86 | 169.1 |
| `fp16-autocast` | 118.41 | 8.4 | **0.07x** | 2,223.4 | 926 | 475 | 58.90 | 15,431.0 |
| `int8-dynamic` | 6.88 | 145.4 | **1.27x** | 21.4 | 1,341 | 232 | 63.50 | 577.6 |
| `int8-dynamic-static-kv` | 6.02 | 166.1 | **1.45x** | 20.1 | 1,332 | 232 | 63.50 | 175.2 |
| `onnxruntime` | 8.22 | 121.7 | **1.06x** | 8.8 | 1,163 | 475 | 58.86 | - |
| `onnxruntime-int8` | 3.16 | 316.9 | **2.76x** | 5.7 | 781 | 270 | 89.38 | - |
| `tensorrt` | skipped: TensorrtExecutionProvider not available (needs onnxruntime-gpu, TensorRT and an NVIDIA GPU) | | | | | | | |

**gpt2-xl**

| Strategy | ms/token | tokens/s | Speedup vs FP32 | Prefill (ms) | Peak memory (MB) | Weights (MB) | Perplexity | Allocated during generation (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `all` | not run: the 6 GB model did not fit in free RAM on this 16 GB Mac alongside other apps (swapping); see the x86 results | | | | | | | |

