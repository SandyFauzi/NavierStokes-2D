import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# uji konservasi massa & positivitas suhu
# pastikan divergensi 0 & tidak melanggar batas suhu

import os
import numpy as np
import matplotlib.pyplot as plt
from src.config import SimulationConfig
from src.solver import NavierStokesSolver

def run_positivity_test(method):
    # simulasi peclet sangat tinggi
    # uji osilasi fdm
    cfg = SimulationConfig(
        Lx=15.0, Ly=5.0, nx=150, ny=50, Re=200.0, U_inf=1.0,
        obstacle_type="square", obs_D=1.0,
        method=method, n_steps=1000, plot_every=1000,
        T_inf=0.0, T_obs=1.0, Pr=100.0  # pr tinggi = difusi termal kecil
    )
    
    solver = NavierStokesSolver(cfg)
    
    max_div_history = []
    T_max_history = []
    T_min_history = []
    
    for step in range(cfg.n_steps):
        solver.step()
        if step % 10 == 0:
            solver.update_diagnostics()
            T_field = solver.get_temperature()
            max_div_history.append(solver.div_err)
            T_max_history.append(float(np.max(T_field)))
            T_min_history.append(float(np.min(T_field)))
            
    return max_div_history, T_max_history, T_min_history, cfg.dt

def main():
    print("=" * 60)
    print("  UJI KONSERVASI MASSA & POSITIVITAS SUHU")
    print("=" * 60)
    
    methods = ["fvm", "fdm"]
    results = {}
    
    for method in methods:
        print(f"Menjalankan simulasi {method.upper()}...")
        divs, t_max, t_min, dt = run_positivity_test(method)
        results[method] = (divs, t_max, t_min, dt)
        
        print(f"  {method.upper()} - Max Divergence: {max(divs):.2e}")
        print(f"  {method.upper()} - Suhu Terendah : {min(t_min):.4f} (Seharusnya >= 0.0)")
        print(f"  {method.upper()} - Suhu Tertinggi: {max(t_max):.4f} (Seharusnya <= 1.0)")
        if min(t_min) < -1e-3 or max(t_max) > 1.001:
            print(f"  [!] {method.upper()} MELANGGAR kriteria Positivitas (Maximum Principle).")
        else:
            print(f"  [+] {method.upper()} MEMATUHI kriteria Positivitas.")
            
    # render grafik
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for method in methods:
        divs, t_max, t_min, dt = results[method]
        # Filter NaN or Inf
        valid_idx = np.isfinite(divs) & np.isfinite(t_max) & np.isfinite(t_min)
        divs = np.array(divs)[valid_idx]
        t_max = np.array(t_max)[valid_idx]
        t_min = np.array(t_min)[valid_idx]
        
        time_axis = np.arange(len(divs)) * dt * 10
        
        ax1.plot(time_axis, divs, label=f'{method.upper()}')
        ax2.plot(time_axis, t_max, label=f'{method.upper()} Max T', linestyle='-')
        ax2.plot(time_axis, t_min, label=f'{method.upper()} Min T', linestyle='--')
        
    ax1.set_yscale('log')
    ax1.set_ylim(1e-2, 1e4)  # Limit to see FVM clearly
    ax1.set_xlabel('Waktu Simulasi (s)')
    ax1.set_ylabel('Maksimum Absolut Divergensi')
    ax1.set_title('Uji Konservasi Massa (Divergensi ~ 0)')
    ax1.grid(True)
    ax1.legend()
    
    ax2.axhline(1.0, color='r', linestyle=':', label='Batas T_obs (1.0)')
    ax2.axhline(0.0, color='b', linestyle=':', label='Batas T_inf (0.0)')
    ax2.set_ylim(-0.2, 1.5)  # Limit to see positivity bounds
    ax2.set_xlabel('Waktu Simulasi (s)')
    ax2.set_ylabel('Suhu (Min & Max)')
    ax2.set_title('Uji Positivitas Suhu (Maximum Principle)')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/test_conservation.png", dpi=150)
    print("Plot disimpan ke: results/test_conservation.png")

if __name__ == "__main__":
    main()
