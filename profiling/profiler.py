"""GPU and CPU profiling for inference runs.

``Profiler`` wraps a block of inference code and records:

* **CUDA:**
  * wall time, timed with CUDA events and synchronised
  * ``torch.cuda.max_memory_allocated`` (peak tensor memory) and
    ``max_memory_reserved`` (peak memory held by the caching allocator)
  * the allocator's allocation count, from ``torch.cuda.memory_stats``
  * device memory in use, from ``torch.cuda.mem_get_info``. This includes
    ONNX Runtime and TensorRT allocations, which PyTorch's allocator never
    sees.
  * compute utilization: ``torch.cuda.utilization()`` sampled every 10 ms in a
    background thread (needs ``pynvml``)
* **CPU fallback** (no GPU):
  * wall time
  * process CPU utilization: CPU time over wall time, so 400% means four busy
    cores
  * peak resident memory: RSS sampled every 5 ms with ``psutil``, reported as
    the peak and as the increase over the RSS at entry

``allocation_profile`` uses ``torch.profiler`` with ``profile_memory=True`` to
total the bytes PyTorch allocates while a function runs, and to count the
``aten::cat`` calls. This is how the benchmark shows what a dynamic,
concatenating KV cache costs next to a pre-allocated one.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import psutil
import torch
from torch.profiler import ProfilerActivity, profile

MB = 2**20


@dataclass
class ProfileResult:
    device: str
    wall_s: float = 0.0
    cpu_util_pct: float | None = None
    peak_rss_mb: float | None = None
    rss_increase_mb: float | None = None
    cuda_peak_allocated_mb: float | None = None
    cuda_peak_reserved_mb: float | None = None
    cuda_allocations: int | None = None
    cuda_device_used_mb: float | None = None
    gpu_util_mean_pct: float | None = None
    gpu_util_max_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class _Sampler:
    """Runs ``sample()`` every ``interval_s`` seconds in a daemon thread."""

    def __init__(self, sample: Callable[[], float], interval_s: float) -> None:
        self._sample = sample
        self._interval = interval_s
        self.values: list[float] = []
        self.failed = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.values.append(self._sample())
            except Exception:  # e.g. pynvml missing: utilization unavailable
                self.failed = True
                return
            time.sleep(self._interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()


class Profiler:
    """Context manager that profiles the enclosed block on ``device``."""

    def __init__(self, device: torch.device | str) -> None:
        self.device = torch.device(device)
        self.result = ProfileResult(device=self.device.type)
        self._proc = psutil.Process()

    @property
    def cuda(self) -> bool:
        return self.device.type == "cuda"

    def __enter__(self) -> Profiler:
        self._rss_start = self._proc.memory_info().rss
        self._rss = _Sampler(lambda: float(self._proc.memory_info().rss), 0.005)
        self._rss.start()
        if self.cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            self._allocs_start = torch.cuda.memory_stats(self.device).get(
                "allocation.all.allocated", 0
            )
            self._gpu = _Sampler(lambda: float(torch.cuda.utilization(self.device)), 0.01)
            self._gpu.start()
            self._start_event = torch.cuda.Event(enable_timing=True)
            self._end_event = torch.cuda.Event(enable_timing=True)
            self._start_event.record()
        self._cpu_start = self._proc.cpu_times()
        self._wall_start = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        r = self.result
        if self.cuda:
            self._end_event.record()
            torch.cuda.synchronize(self.device)
        r.wall_s = time.perf_counter() - self._wall_start
        cpu_end = self._proc.cpu_times()
        cpu_s = (cpu_end.user - self._cpu_start.user) + (cpu_end.system - self._cpu_start.system)
        self._rss.stop()
        peak = max([self._rss_start, *self._rss.values])
        r.peak_rss_mb = round(peak / MB, 1)
        r.rss_increase_mb = round((peak - self._rss_start) / MB, 1)
        if self.cuda:
            self._gpu.stop()
            r.wall_s = self._start_event.elapsed_time(self._end_event) / 1000.0
            r.cuda_peak_allocated_mb = round(torch.cuda.max_memory_allocated(self.device) / MB, 1)
            r.cuda_peak_reserved_mb = round(torch.cuda.max_memory_reserved(self.device) / MB, 1)
            allocs_end = torch.cuda.memory_stats(self.device).get("allocation.all.allocated", 0)
            r.cuda_allocations = int(allocs_end - self._allocs_start)
            free, total = torch.cuda.mem_get_info(self.device)
            r.cuda_device_used_mb = round((total - free) / MB, 1)
            if self._gpu.values and not self._gpu.failed:
                r.gpu_util_mean_pct = round(sum(self._gpu.values) / len(self._gpu.values), 1)
                r.gpu_util_max_pct = max(self._gpu.values)
        else:
            r.cpu_util_pct = round(100.0 * cpu_s / r.wall_s, 1) if r.wall_s > 0 else None


@dataclass
class AllocationProfile:
    allocated_mb: float  # total bytes PyTorch allocated during the call
    allocating_ops: int  # op invocations that allocated memory
    cat_calls: int  # aten::cat invocations (dynamic KV caches concatenate)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def allocation_profile(fn: Callable[[], object], device: torch.device | str) -> AllocationProfile:
    """Profile the memory allocations PyTorch makes while ``fn`` runs."""
    dev = torch.device(device)
    activities = [ProfilerActivity.CPU]
    if dev.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, profile_memory=True) as prof:
        fn()
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
    total = 0
    allocating = 0
    cats = 0
    for e in prof.key_averages():
        mem = e.self_device_memory_usage if dev.type == "cuda" else e.self_cpu_memory_usage
        if mem > 0:
            total += mem
            allocating += e.count
        if e.key == "aten::cat":
            cats += e.count
    return AllocationProfile(round(total / MB, 2), allocating, cats)


def device_summary(device: torch.device | str) -> dict[str, Any]:
    dev = torch.device(device)
    info: dict[str, Any] = {
        "device": dev.type,
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "cpu_count": psutil.cpu_count(logical=True),
        "memory_gb": round(psutil.virtual_memory().total / 2**30, 1),
    }
    if dev.type == "cuda":
        props = torch.cuda.get_device_properties(dev)
        info.update({"gpu": props.name, "gpu_memory_gb": round(props.total_memory / 2**30, 1)})
    return info
