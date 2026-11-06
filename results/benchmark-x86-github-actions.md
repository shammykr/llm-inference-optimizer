**GitHub Actions ubuntu-latest (x86-64)**: device `cpu`, torch 2.15.0+cpu, 2 threads. Batch 1, 32-token prompt, 64 new tokens (greedy), median of 3 runs.

**gpt2** (124M parameters)

| Strategy | ms/token | tokens/s | Speedup vs FP32 | Prefill (ms) | Peak memory (MB) | Weights (MB) | Perplexity | Allocated during generation (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fp32` | 23.18 | 43.1 | 1.00x | 49.7 | 996 | 475 | 58.86 | 570.5 |
| `fp32-no-cache` | 70.88 | 14.1 | **0.33x** | 50.0 | 980 | 475 | 58.86 | 6,776.2 |
| `fp32-static-kv` | 20.40 | 49.0 | **1.14x** | 48.9 | 975 | 475 | 58.86 | 168.0 |
| `fp16-autocast` | 395.47 | 2.5 | **0.06x** | 3,355.3 | 1,063 | 475 | 58.93 | 15,429.7 |
| `int8-dynamic` | 14.31 | 69.9 | **1.62x** | 25.5 | 1,075 | 232 | 83.33 | 600.6 |
| `int8-dynamic-static-kv` | 12.27 | 81.5 | **1.89x** | 22.7 | 1,074 | 232 | 83.33 | 198.1 |
| `onnxruntime` | 20.09 | 49.8 | **1.15x** | 37.5 | 1,730 | 475 | 58.86 | - |
| `onnxruntime-int8` | 11.67 | 85.7 | **1.99x** | 23.2 | 813 | 270 | 86.95 | - |
| `tensorrt` | skipped: TensorrtExecutionProvider not available (needs onnxruntime-gpu, TensorRT and an NVIDIA GPU) | | | | | | | |

**gpt2-xl** (1,558M parameters)

| Strategy | ms/token | tokens/s | Speedup vs FP32 | Prefill (ms) | Peak memory (MB) | Weights (MB) | Perplexity | Allocated during generation (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fp32` | 238.99 | 4.2 | 1.00x | 728.4 | 6,613 | 5,942 | 39.10 | 3,797.4 |
| `fp32-no-cache` | 1,051.13 | 0.9 | **0.23x** | 721.2 | 6,531 | 5,942 | 39.10 | 55,115.7 |
| `fp32-static-kv` | 235.23 | 4.2 | **1.02x** | 737.8 | 6,585 | 5,942 | 39.10 | 1,299.3 |
| `fp16-autocast` | skipped: skipped by --skip | | | | | | | |
| `int8-dynamic` | 92.12 | 10.8 | **2.59x** | 214.0 | 8,185 | 1,723 | 39.80 | 4,047.9 |
| `int8-dynamic-static-kv` | 86.57 | 11.6 | **2.76x** | 209.8 | 8,090 | 1,723 | 39.80 | 1,549.8 |
| `onnxruntime` | 228.01 | 4.4 | **1.05x** | 511.1 | 13,159 | 5,944 | 39.10 | - |
| `onnxruntime-int8` | skipped: skipped by --skip | | | | | | | |
| `tensorrt` | skipped: TensorrtExecutionProvider not available (needs onnxruntime-gpu, TensorRT and an NVIDIA GPU) | | | | | | | |

