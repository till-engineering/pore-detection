"""GPU detection and data transfer. PyTorch is optional: without it everything runs on the CPU.

torch is imported lazily (it takes seconds), only when a configured method has a GPU path.
"""

from functools import cache

import numpy as np


@cache
def _torch():
    try:
        import torch
    except ImportError:
        return None
    return torch


def gpu_available() -> bool:
    torch = _torch()
    return torch is not None and torch.cuda.is_available()


def select_device(setting: str, gpu_wanted: bool) -> str:
    """runtime.device (auto | cuda | cpu) -> "cuda" or "cpu".

    gpu_wanted: some configured method has a GPU path. If not, "auto" stays on the CPU
    without importing torch.
    """
    if setting == "cpu" or (setting == "auto" and not gpu_wanted):
        return "cpu"
    if setting == "cuda" and not gpu_available():
        raise RuntimeError("runtime.device is 'cuda' but no CUDA GPU is available "
                           "(PyTorch with CUDA installed?)")
    return "cuda" if gpu_available() else "cpu"


def gpu_name() -> str | None:
    return _torch().cuda.get_device_name() if gpu_available() else None


def free_gpu_memory_bytes() -> int:
    """Currently free GPU memory (driver view, < 1 ms)."""
    free, _total = _torch().cuda.mem_get_info()
    return free


def is_out_of_memory(error: BaseException) -> bool:
    torch = _torch()
    return torch is not None and isinstance(error, torch.cuda.OutOfMemoryError)


def release() -> None:
    """Give memory cached by torch back to the driver, so the next measurement sees it as free."""
    _torch().cuda.empty_cache()


def to_gpu(array: np.ndarray):
    """NumPy array -> torch tensor on the GPU."""
    if not array.flags.writeable:  # read-only views (ctx) -> torch needs its own buffer
        array = array.copy()
    return _torch().from_numpy(np.ascontiguousarray(array)).to("cuda", non_blocking=True)


def to_numpy(tensor) -> np.ndarray:
    """torch tensor (any device) -> NumPy array."""
    return tensor.detach().cpu().numpy()
