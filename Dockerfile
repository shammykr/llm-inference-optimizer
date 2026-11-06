# syntax=docker/dockerfile:1
#
# Two targets:
#   docker build --target cpu -t llm-inference-optimizer:cpu .   (default; any machine)
#   docker build --target gpu -t llm-inference-optimizer:gpu .   (NVIDIA GPU: CUDA + TensorRT)
#
# The entrypoint is `python -m`, so the command picks the module to run:
#   docker run --rm llm-inference-optimizer:cpu scripts.smoke_test
#   docker run --rm -v $PWD/results:/app/results llm-inference-optimizer:cpu \
#       benchmarks.run_benchmarks --models gpt2
#   docker run --rm --gpus all llm-inference-optimizer:gpu benchmarks.run_benchmarks --models gpt2

# ---------- GPU: CUDA runtime + PyTorch (CUDA) + ONNX Runtime GPU + TensorRT ----------
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04 AS gpu
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 HF_HOME=/app/.hf-cache VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-venv \
 && rm -rf /var/lib/apt/lists/* && python3 -m venv /opt/venv
WORKDIR /app
COPY requirements.txt ./
# PyPI's Linux torch wheels bundle CUDA. onnxruntime-gpu provides the CUDA and
# TensorRT execution providers; tensorrt-cu12 provides the TensorRT libraries.
RUN grep -v '^onnxruntime' requirements.txt > /tmp/req.txt \
 && pip install torch -r /tmp/req.txt onnxruntime-gpu tensorrt-cu12 pynvml \
 && TRT_LIBS=$(python -c "import tensorrt_libs, os; print(os.path.dirname(tensorrt_libs.__file__))") \
 && echo "$TRT_LIBS" > /etc/ld.so.conf.d/tensorrt.conf && ldconfig
COPY optimizers ./optimizers
COPY profiling ./profiling
COPY benchmarks ./benchmarks
COPY scripts ./scripts
ENTRYPOINT ["python", "-m"]
CMD ["benchmarks.run_benchmarks", "--models", "gpt2"]

# ---------- CPU: PyTorch (CPU) + ONNX Runtime (CPU). Last stage = default target ----------
FROM python:3.12-slim AS cpu
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 HF_HOME=/app/.hf-cache
WORKDIR /app
COPY requirements.txt ./
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements.txt
COPY optimizers ./optimizers
COPY profiling ./profiling
COPY benchmarks ./benchmarks
COPY scripts ./scripts
# Bake GPT-2 (124M) into the image so the smoke test runs offline.
RUN python -c "from optimizers.models import load_model; load_model('gpt2')"
RUN useradd --create-home --uid 1000 app && chown -R app /app
USER app
ENTRYPOINT ["python", "-m"]
CMD ["benchmarks.run_benchmarks", "--models", "gpt2"]
