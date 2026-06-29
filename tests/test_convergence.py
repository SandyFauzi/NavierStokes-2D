import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# uji konvergensi grid (l2 error vs analitik poiseuille)

import os
import numpy as np
import matplotlib.pyplot as plt
from src.config import SimulationConfig
from src.solver import NavierStokesSolver
from src import backend

def run_simulation(nx, ny, method="fvm"):
    # setup poiseuille: saluran kosong
    cfg = SimulationConfig(
        Lx=10.0, Ly=2.0, nx=nx, ny=ny, Re=20.0, U_inf=1.0,
        obstacle_type="none",
        method=method, n_steps=99999, plot_every=99999, cfl=0.15,
        wall="noslip", adv_blend=0.0, precision="double",
    )
    
    solver = NavierStokesSolver(cfg)
    # hitung waktu fisis target untuk fully developed flow (h^2/nu)
    T_target = 80.0 
    
    while solver.time < T_target:
        solver.step()
        
    u_profile = backend.to_cpu(solver.d.u[1:ny+1, nx // 2])
    y = np.array([(j - 0.5) * cfg.dy for j in range(1, ny + 1)])
    
    H = cfg.Ly
    u_max = np.max(u_profile)
    u_analytic = 4.0 * u_max / H**2 * y * (H - y)
    
    error_L2 = np.sqrt(np.mean((u_profile - u_analytic)**2))
    return cfg.dy, error_L2

def main():
    resolutions = [(25, 10), (50, 20), (100, 40), (200, 80)]
    methods = ["fvm", "fdm"]
    
    results = {}
    print("=" * 50)
    print("  UJI KONVERGENSI GRID (Spatial Convergence)")
    print("=" * 50)
    
    for method in methods:
        print(f"Menguji metode: {method.upper()}")
        dys = []
        errors = []
        for nx, ny in resolutions:
            dy, err = run_simulation(nx, ny, method)
            dys.append(dy)
            errors.append(err)
            print(f"  Grid {nx}x{ny:2d} | dy = {dy:.4f} | L2 Error = {err:.2e}")
        results[method] = (dys, errors)
        
    # render plot log
    plt.figure(figsize=(8, 6))
    
    for method in methods:
        dys, errs = results[method]
        plt.loglog(dys, errs, marker='o', label=f'{method.upper()} Error')
        
        # kalkulasi orde akurasi (slope)
        slope, _ = np.polyfit(np.log(dys), np.log(errs), 1)
        print(f"Orde konvergensi {method.upper()}: O(dx^{slope:.2f})")
        
    # garis referensi o(dx) dan o(dx^2)
    dys_ref = np.array(results["fvm"][0])
    plt.loglog(dys_ref, dys_ref * (results["fvm"][1][0] / dys_ref[0]), 'k--', label='O(dx) (Orde 1)')
    plt.loglog(dys_ref, dys_ref**2 * (results["fdm"][1][0] / dys_ref[0]**2), 'r--', label='O(dx^2) (Orde 2)')
    
    plt.xlabel('Ukuran Grid (dy)')
    plt.ylabel('L2 Error (Kecepatan u)')
    plt.title('Uji Konvergensi Grid (Aliran Poiseuille)')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/test_convergence.png", dpi=150)
    print("Plot disimpan ke: results/test_convergence.png")
    
if __name__ == "__main__":
    main()
