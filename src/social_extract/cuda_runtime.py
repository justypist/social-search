from __future__ import annotations

import ctypes
import importlib
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def preload_nvidia_cuda_libraries() -> list[str]:
    """Load CUDA libraries shipped by NVIDIA Python wheels into this process."""
    loaded: list[str] = []
    for library_path in _candidate_libraries():
        try:
            ctypes.CDLL(str(library_path), mode=ctypes.RTLD_GLOBAL)
        except OSError:
            continue
        loaded.append(str(library_path))
    return loaded


def _candidate_libraries() -> list[Path]:
    directories = [_module_directory(name) for name in _NVIDIA_LIB_MODULES]
    libraries: list[Path] = []
    for directory in directories:
        if directory is None:
            continue
        for pattern in _LIBRARY_PATTERNS:
            libraries.extend(sorted(directory.glob(pattern)))
    return libraries


def _module_directory(module_name: str) -> Path | None:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    if module.__file__ is not None:
        return Path(module.__file__).parent
    module_paths = getattr(module, "__path__", None)
    if not module_paths:
        return None
    return Path(next(iter(module_paths)))


_NVIDIA_LIB_MODULES = (
    "nvidia.cuda_nvrtc.lib",
    "nvidia.cublas.lib",
    "nvidia.cudnn.lib",
)

_LIBRARY_PATTERNS = (
    "libnvrtc.so*",
    "libcublas.so*",
    "libcublasLt.so*",
    "libcudnn*.so*",
)
