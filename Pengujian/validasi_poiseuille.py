import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""Validasi solver vs solusi analitik Poiseuille (profil kecepatan parabolik)."""

import os
import numpy as np
import matplotlib.pyplot as plt
from src.config import SimulationConfig
from src.solver import NavierStokesSolver
from src import backend


def run_poiseuille_validation():
    # saluran kosong, dinding no-slip, upwind murni (obstacle ditaruh jauh)
    cfg = SimulationConfig(
        Lx=10.0, Ly=2.0, nx=100, ny=40, Re=20.0, U_inf=1.0,
        obstacle_type="cylinder", obs_cx=-999.0, obs_cy=-999.0,
        obs_D=1.0,                 # nu = U*D/Re; D=1 -> nu=0.05 (cukup kental utk parabolik)
        method="fvm", n_steps=10000, plot_every=1000, cfl=0.15,
        wall="noslip", adv_blend=0.0, seed_perturbation=False,
        precision="double",     # validasi pakai fp64 untuk akurasi
    )

    solver = NavierStokesSolver(cfg)
    print(f"dt = {solver.dt:.6f} | nu = {cfg.nu:.6f}")
    print(f"Menjalankan {cfg.n_steps} langkah...")
    for step in range(cfg.n_steps):
        solver.step()
        if (step + 1) % cfg.plot_every == 0:
            solver.update_diagnostics()
            print(f"  Step {step+1:>6d} | Max|div u| = {solver.divergence_error:.2e}")

    ny, nx = cfg.ny, cfg.nx
    u_profile = backend.to_cpu(solver.d.u[1:ny+1, nx // 2])    # kolom di tengah domain
    y = np.array([(j - 0.5) * cfg.dy for j in range(1, ny + 1)])

    H = cfg.Ly
    u_max = np.max(u_profile)
    u_analytic = 4.0 * u_max / H**2 * y * (H - y)              # parabola Poiseuille

    error = np.abs(u_profile - u_analytic)
    max_error = np.max(error)
    rel_error = max_error / u_max * 100

    print("\n" + "=" * 50)
    print("  HASIL VALIDASI POISEUILLE")
    print("=" * 50)
    print(f"  Max error absolut : {max_error:.6f}")
    print(f"  Error relatif     : {rel_error:.2f}%")
    print(f"  Status            : {'LULUS' if rel_error < 5 else 'GAGAL'}")
    print("=" * 50)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(u_analytic, y, 'r-', linewidth=2, label='Analitik (Poiseuille)')
    ax1.plot(u_profile, y, 'bo', markersize=4, label='Numerik (FVM)')
    ax1.set_xlabel('u(y)'); ax1.set_ylabel('y')
    ax1.set_title('Profil Kecepatan', fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(error, y, 'k-', linewidth=1.5)
    ax2.set_xlabel('|Error|'); ax2.set_ylabel('y')
    ax2.set_title(f'Error Absolut (max = {max_error:.4e})', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Validasi: Aliran Poiseuille — Numerik vs Analitik',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("Hasil/validasi_poiseuille.png", dpi=150, bbox_inches='tight')
    plt.show()
    print("Plot disimpan ke: Hasil/validasi_poiseuille.png")


if __name__ == "__main__":
    os.makedirs("Hasil", exist_ok=True)
    run_poiseuille_validation()
