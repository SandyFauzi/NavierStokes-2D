import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# uji pengukuran bilangan nusselt dan koefisien drag
# bandingkan kinerja termal berbagai bentuk geometri

import numpy as np
from src.config import SimulationConfig
from src.solver import NavierStokesSolver

def run_nusselt_test(shape, angle, Re):
    # konfigurasi identik dengan batch_plot
    cfg = SimulationConfig(
        Lx=36.0, Ly=12.0, nx=360, ny=120, Re=Re,
        obstacle_type=shape, obs_angle=angle, obs_D=3.0,
        obs_cx=9.0, obs_cy=6.0,
        method="fvm", adv_blend=0.8,
        wall="freeslip", seed_perturbation=True
    )
    
    solver = NavierStokesSolver(cfg, "cpu")
    solver.aero_sample_interval = 25
    
    warmup_steps = 5000
    measure_steps = 2000
    
    # lewati masa transien agar wake berkembang
    for _ in range(warmup_steps):
        solver.step()
        
    nu_history = []
    cd_history = []
    
    # ukur rata-rata
    for step in range(measure_steps):
        solver.step()
        if step % 25 == 0:
            cd, _ = solver.compute_aerodynamics()
            nu_history.append(solver.compute_nusselt())
            cd_history.append(cd)
            
    return float(np.mean(nu_history)), float(np.mean(cd_history))

def main():
    print("=" * 60)
    print("  UJI BILANGAN NUSSELT & KOEFISIEN DRAG")
    print("=" * 60)
    print("  Catatan: Nilai dipengaruhi blockage ratio (D/Ly=0.25),")
    print("  boundary freeslip, dan skema adv_blend=0.8.")
    print("-" * 60)
    
    cases = [
        ("cylinder", 0.0, 140),
        ("square", 0.0, 140),
        ("triangle", 180.0, 140),
        ("ellipse", 0.0, 240),
    ]
    
    results = []
    
    for shape, angle, Re in cases:
        print(f"Menjalankan simulasi {shape.upper()}...")
        nu_avg, cd_avg = run_nusselt_test(shape, angle, Re)
        results.append((shape, Re, nu_avg, cd_avg))
        
        lit_note = ""
        if shape == "cylinder":
            # Literature approx for cylinder Re=140: CD ~1.3-1.4, St ~0.18-0.2
            lit_note = " (Lit CD≈1.3-1.4. Selisih karena blockage)"
            
        print(f"  {shape:10} Re={Re:<4} Nu={nu_avg:8.3f} CD={cd_avg:8.3f} {lit_note}")
        
    print("-" * 60)
    print("  RINGKASAN (Diurutkan dari Nu tertinggi):")
    for shape, Re, nu_avg, cd_avg in sorted(results, key=lambda r: -r[2]):
        print(f"    {shape:10} Re={Re:<4} Nu={nu_avg:7.3f} CD={cd_avg:7.3f}")
    print("=" * 60)

if __name__ == "__main__":
    main()
