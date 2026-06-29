# deteksi backend eksekusi
# cpu = numpy + numba njit
# gpu = cupy + numba cuda

import os
import numpy as np

cp = None                  # modul cupy
GPU_PRESENT = False        # flag status gpu
NUMBA_CUDA = False         # flag numba cuda
_DEVICE_NAME = "CPU"

# bisa dipaksa lewat env FISKOM_BACKEND isi cpu atau gpu
_FORCE = os.environ.get("FISKOM_BACKEND", "").lower()


def _enable_numba_cuda():
    # setup env cuda
    # bypass deteksi otomatis numba
    try:
        import importlib.util
        import glob
        import shutil
        nvcc = importlib.util.find_spec("nvidia.cuda_nvcc")
        if not (nvcc and nvcc.submodule_search_locations):
            return
        base = list(nvcc.submodule_search_locations)[0]
        binp = os.path.join(base, "bin")
        if not os.path.isdir(os.path.join(base, "nvvm")) or not os.path.isdir(binp):
            return
        if not glob.glob(os.path.join(binp, "cudart64_*.dll")):
            rt = importlib.util.find_spec("nvidia.cuda_runtime")
            if rt and rt.submodule_search_locations:
                rtbin = os.path.join(list(rt.submodule_search_locations)[0], "bin")
                for f in glob.glob(os.path.join(rtbin, "cudart64_*.dll")):
                    try:
                        shutil.copy2(f, binp)
                    except Exception:
                        pass
        os.environ.setdefault("CUDA_HOME", base)
        os.environ.setdefault("CUDA_PATH", base)
    except Exception:
        pass


# init deteksi gpu (cupy)
try:
    import cupy as _cp
    if _cp.cuda.runtime.getDeviceCount() > 0:
        _t = _cp.arange(8, dtype=_cp.float32)      # test load gpu
        float((_t * 2).sum())
        _cp.cuda.Stream.null.synchronize()
        cp = _cp
        GPU_PRESENT = True
        dev = _cp.cuda.runtime.getDevice()
        _DEVICE_NAME = _cp.cuda.runtime.getDeviceProperties(dev)["name"].decode()
except Exception:
    cp = None
    GPU_PRESENT = False

# init numba cuda
if GPU_PRESENT:
    _enable_numba_cuda()
    try:
        from numba import cuda as _nbcuda
        if _nbcuda.is_available():
            @_nbcuda.jit
            def _smoke(a):
                i = _nbcuda.grid(1)
                if i < a.size:
                    a[i] += 1.0
            _d = _nbcuda.to_device(np.zeros(8, dtype=np.float32))
            _smoke[1, 8](_d)
            _nbcuda.synchronize()
            if float(_d.copy_to_host()[0]) == 1.0:
                NUMBA_CUDA = True
    except Exception:
        NUMBA_CUDA = False


def gpu_usable(): return GPU_PRESENT


def default_mode():
    if _FORCE == "cpu":  return "cpu"
    if _FORCE == "gpu" and GPU_PRESENT: return "gpu"
    return "gpu" if GPU_PRESENT else "cpu"


def array_module(mode):
    # alokasi memori
    return cp if (mode == "gpu" and GPU_PRESENT) else np


def sync():
    # sync device
    if cp: cp.cuda.Stream.null.synchronize()


def to_cpu(arr):
    # konversi d2h
    if cp is not None and isinstance(arr, cp.ndarray): return cp.asnumpy(arr)
    return np.asarray(arr)


def device_name(): return _DEVICE_NAME


def backend_label(mode):
    if mode == "gpu" and GPU_PRESENT:
        extra = "CuPy + Numba CUDA" if NUMBA_CUDA else "CuPy"
        return f"GPU: {_DEVICE_NAME} ({extra})"
    return f"CPU: Numba JIT ({os.cpu_count() or 1} cores)"
