import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# uji stabilitas numerik solver
# metrik: courant (cfl), difusi (von neumann), peclet cell

import numpy as np
from src.config import SimulationConfig
from src.solver import NavierStokesSolver
from src import backend


def stability_numbers(cfg: SimulationConfig):
    dt = cfg.compute_dt()
    dx, dy = cfg.dx, cfg.dy
    h = min(dx, dy)
    inv = 1.0 / dx**2 + 1.0 / dy**2
    return {
        "dt": dt,
        "Courant":   cfg.U_inf * dt / h,
        "Difusi":    cfg.nu * dt * inv,
        "Difusi_T":  cfg.alpha * dt * inv,
        "Peclet_sel": cfg.U_inf * dx / cfg.nu,
    }


def run_stability(cfg: SimulationConfig, n_steps: int = 3000):
    solver = NavierStokesSolver(cfg, "cpu")
    max_div = 0.0
    finite = True
    for k in range(n_steps):
        solver.step()
        if (k + 1) % 250 == 0:
            solver.update_diagnostics()
            max_div = max(max_div, solver.div_err)
            u = backend.to_cpu(solver.d.u)
            if not np.all(np.isfinite(u)):
                finite = False
                break
    return max_div, finite, solver.step_count


def main():
    print("=" * 64)
    print("  UJI STABILITAS NUMERIK : Navier-Stokes 2D")
    print("=" * 64)
    cases = [
        ("Re=150 (default GUI)", SimulationConfig(nx=300, ny=120, Re=150.0)),
        ("Re=1000 (lebih ganas)", SimulationConfig(nx=300, ny=120, Re=1000.0)),
    ]
    for name, cfg in cases:
        s = stability_numbers(cfg)
        print(f"\n  [{name}]  dt = {s['dt']:.5f}")
        print(f"    Courant (CFL adveksi) = {s['Courant']:.4f}   "
              f"[{'OK' if s['Courant'] < 1 else 'TAK STABIL'}, batas < 1]")
        print(f"    Difusi (von Neumann)  = {s['Difusi']:.4f}   "
              f"[{'OK' if s['Difusi'] < 0.5 else 'TAK STABIL'}, batas < 0.5]")
        print(f"    Difusi termal         = {s['Difusi_T']:.4f}   "
              f"[{'OK' if s['Difusi_T'] < 0.5 else 'TAK STABIL'}, batas < 0.5]")
        print(f"    Peclet sel (cell Re)  = {s['Peclet_sel']:.2f}   "
              f"(skema FVM blended menahan oscillation pada Pe tinggi)")
        max_div, finite, nstep = run_stability(cfg)
        status = "STABIL" if finite and max_div < 1e2 else "DIVERGEN"
        print(f"    -> {nstep} langkah: max|div u| = {max_div:.2e}, "
              f"finite = {finite}  ==>  {status}")
    print("\n" + "=" * 64)


if __name__ == "__main__":
    main()
