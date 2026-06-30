import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ukur bilangan Nusselt rata-rata per geometri.
# config identik batch_plot.py (D=3, domain 36x12, grid 360x120, FVM) supaya
# nilai Nu konsisten dengan tabel CD/St di laporan.
#
# Nu dirata-rata pada jendela pengukuran setelah aliran + plume termal
# berkembang, jadi nilainya stabil walau vortex shedding membuatnya berosilasi.
#
# pakai: python utils/test_nusselt.py

import numpy as np
from src.config import SimulationConfig
from src.solver import NavierStokesSolver

# (bentuk, sudut, Re render) -> sama dengan kolom "Re render" tabel laporan
CASES = [
    ("cylinder", 0.0,   140),
    ("square",   0.0,   140),
    ("triangle", 180.0, 140),
    ("ellipse",  0.0,   240),
]
DOMAIN = dict(Lx=36.0, Ly=12.0, nx=360, ny=120, obs_cx=9.0, obs_cy=6.0)

WARMUP   = 5000   # lewati transien, biarkan wake + plume termal berkembang
MEASURE  = 2000   # jendela rata-rata
SAMPLE   = 25     # ambil sampel tiap sekian langkah


def _make(shape, angle, Re):
    return SimulationConfig(Re=Re, obstacle_type=shape, obs_angle=angle,
                            obs_D=3.0, method="fvm", adv_blend=0.8,
                            wall="freeslip", seed_perturbation=True, **DOMAIN)


def measure(shape, angle, Re):
    s = NavierStokesSolver(_make(shape, angle, Re), "cpu")
    s.aero_sample_interval = SAMPLE
    for _ in range(WARMUP):
        s.step()
    nus, cds = [], []
    for k in range(MEASURE):
        s.step()
        if k % SAMPLE == 0:
            cd, _ = s.compute_aerodynamics()
            nus.append(s.compute_nusselt())
            cds.append(cd)
    return float(np.mean(nus)), float(np.mean(cds))


def main():
    print("=" * 76)
    print("  Nu & CD RATA-RATA per geometri  (D=3, grid 360x120, FVM)")
    print(f"  warmup={WARMUP}  ukur={MEASURE} langkah")
    print("  Catatan: Nilai Nu/CD dipengaruhi oleh efek blockage ratio (D/Ly=0.25),")
    print("  boundary freeslip, dan skema adv_blend=0.8. Grid-convergence untuk Nu")
    print("  mungkin memerlukan resolusi lebih tinggi untuk presisi mutlak.")
    print("=" * 76)
    print(f"  {'Bentuk':10} {'Re':>4}  {'Nu_avg':>8}  {'CD_avg':>8} | {'Validasi Literatur (Silinder)'}")
    rows = []
    for shape, angle, Re in CASES:
        print(f"  [{shape}] jalan ...", flush=True)
        Nu, CD = measure(shape, angle, Re)
        rows.append((shape, Re, Nu, CD))
        
        lit_note = ""
        if shape == "cylinder":
            # Literature approx for cylinder Re=140: CD ~1.3-1.4, St ~0.18-0.2
            # Nu is highly dependent on blockage; free-stream Nu ~ 6.0 (Hilpert)
            # Our CD is higher (~1.7) due to blockage and freeslip walls.
            lit_note = " (Lit Re=140: CD≈1.3-1.4. Beda krn blockage 25%)"
            
        print(f"  {shape:10} {Re:>4}  {Nu:8.3f}  {CD:8.3f} |{lit_note}", flush=True)
        
    print("=" * 76)
    print("  RINGKASAN (urut Nu):")
    for shape, Re, Nu, CD in sorted(rows, key=lambda r: -r[2]):
        print(f"    {shape:10} Re={Re:<4}  Nu={Nu:7.3f}  CD={CD:7.3f}")
    print("=" * 76)

if __name__ == "__main__":
    main()
