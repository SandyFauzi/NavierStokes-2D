"""Benchmark throughput & FPS solver; bandingkan CPU vs GPU (bila ada GPU).

steps/s = langkah/detik | MLUPS = juta update grid/detik | FPS@N = FPS animasi.
"""

import time
from src import backend
from src.config import SimulationConfig
from src.solver import NavierStokesSolver

NX, NY = 200, 80
N_STEPS = 300
PLOT_EVERY = 25      # asumsi visualisasi tiap 25 langkah (default GUI)


def bench(method, mode, n_steps=N_STEPS, nx=NX, ny=NY):
    cfg = SimulationConfig(nx=nx, ny=ny, Re=100.0, obstacle_type="cylinder",
                           method=method, n_steps=n_steps + 10)
    solver = NavierStokesSolver(cfg, mode)
    for _ in range(5):           # warm-up (JIT Numba / init kernel CuPy)
        solver.step()
    backend.sync()
    t0 = time.perf_counter()
    for _ in range(n_steps):
        solver.step()
    backend.sync()
    dt = time.perf_counter() - t0
    return n_steps / dt if dt > 0 else 0.0


def _row(tag, sps):
    print(f"  {tag:14s} {sps:9.1f} steps/s   {NX*NY*sps/1e6:7.2f} MLUPS   "
          f"{sps/PLOT_EVERY:7.1f} FPS@{PLOT_EVERY}")


def main():
    modes = ["cpu"] + (["gpu"] if backend.GPU_PRESENT else [])
    print("=" * 66)
    print("  BENCHMARK — Solver Navier-Stokes 2D")
    print("=" * 66)
    print(f"  Grid : {NX} x {NY} = {NX*NY:,} sel   |   {N_STEPS} langkah")
    print("-" * 66)

    res = {}
    for mode in modes:
        print(f"  {backend.backend_label(mode)}")
        for method in ("fvm", "fdm"):
            res[(mode, method)] = bench(method, mode)
            _row(f"  {method.upper()}", res[(mode, method)])
        print("-" * 66)

    if "gpu" in modes:
        print("  Speedup GPU / CPU:")
        for method in ("fvm", "fdm"):
            c, g = res[("cpu", method)], res[("gpu", method)]
            if c > 0:
                print(f"    {method.upper()}: {g/c:.2f}x")
    else:
        print("  [Tidak ada GPU NVIDIA — hanya CPU yang diuji]")
    print("=" * 66)


if __name__ == "__main__":
    main()
