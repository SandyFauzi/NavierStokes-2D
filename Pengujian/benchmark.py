import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""Benchmark throughput CPU vs GPU di beberapa ukuran grid.

Menunjukkan TITIK-SILANG: di grid kecil CPU menang (set kerja muat di L3 cache),
di grid besar GPU menang (overhead launch teramortisasi, CPU tumpah ke RAM).

steps/s = langkah/detik | MLUPS = juta update grid/detik | FPS@N = FPS animasi.
"""

import time
from src import backend
from src.config import SimulationConfig
from src.solver import NavierStokesSolver

# grid kecil -> realtime (CPU menang) ... grid besar -> GPU menang
GRIDS = [(200, 80), (400, 160), (600, 240), (800, 320), (1000, 400), (1400, 560)]
PLOT_EVERY = 25


def bench(method, mode, nx, ny, n_steps):
    cfg = SimulationConfig(nx=nx, ny=ny, Re=150.0, obstacle_type="cylinder",
                           method=method, n_steps=n_steps + 20)
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


def main():
    modes = ["cpu"] + (["gpu"] if backend.GPU_PRESENT else [])
    print("=" * 74)
    print("  BENCHMARK — Solver Navier-Stokes 2D (CPU vs GPU, sweep grid)")
    print("=" * 74)
    for mode in modes:
        print(f"  {backend.backend_label(mode)}")
    print("-" * 74)
    print(f"  {'grid':>11} {'sel':>9} {'metode':>6} "
          + "".join(f"{m.upper()+' sps':>11}" for m in modes)
          + (f"{'GPU/CPU':>9}" if "gpu" in modes else ""))
    print("-" * 74)

    crossover = None
    for nx, ny in GRIDS:
        cells = nx * ny
        n_steps = max(60, int(6_000_000 / cells))   # kerja ~tetap di tiap grid
        for method in ("fvm", "fdm"):
            sps = {m: bench(method, m, nx, ny, n_steps) for m in modes}
            line = f"  {nx:>5}x{ny:<5} {cells:>9,} {method.upper():>6} "
            line += "".join(f"{sps[m]:>11.1f}" for m in modes)
            if "gpu" in modes and sps["cpu"] > 0:
                r = sps["gpu"] / sps["cpu"]
                line += f"{r:>8.2f}x"
                if method == "fvm" and r >= 1.0 and crossover is None:
                    crossover = (nx, ny, r)
            print(line)
        print("-" * 74)

    if "gpu" in modes:
        if crossover:
            nx, ny, r = crossover
            print(f"  >> GPU mulai MENGUNGGULI CPU di {nx}x{ny} ({r:.2f}x). "
                  f"Pakai grid >= ini untuk demo GPU>CPU.")
        else:
            print("  >> CPU masih menang di semua grid yang diuji "
                  "(coba grid lebih besar).")
        print("  Catatan: FPS animasi GUI tetap 60 (render decoupled), apa pun steps/s.")
    else:
        print("  [Tidak ada GPU NVIDIA — hanya CPU yang diuji]")
    print("=" * 74)


if __name__ == "__main__":
    main()
