import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""Batch untuk laporan: cari Re KRITIS (onset vortex shedding) tiap geometri,
lalu export PNG vortisitas di Re tersebut + legend (Bentuk, Re, CD_avg, St).

Onset dideteksi dari LAJU-TUMBUH osilasi gaya angkat (CL): di atas Re kritis
amplitudo tumbuh terhadap waktu, di bawahnya meluruh. Ini kriteria fisika yang
benar (instabilitas linear), dan tak perlu menunggu shedding jenuh.

Segitiga pakai sudut 180 derajat -> sisi tajam (apex) menghadap aliran.

Pakai: python batch_plot.py
Output: NavierStokes-2D/Hasil/batch_<bentuk>_Re<...>_vorticity.png
"""

import numpy as np
from src.config import SimulationConfig, hasil_path
from src.solver import NavierStokesSolver
from src.render import Renderer

SHAPES = [("cylinder", 0.0), ("square", 0.0), ("triangle", 180.0), ("ellipse", 0.0)]
RE_CANDIDATES = [50, 70, 100, 140, 190]
# domain cukup halus (dx=0.1) supaya difusi numerik tak menutupi shedding
DOMAIN = dict(Lx=36.0, Ly=12.0, nx=360, ny=120, obs_cx=9.0, obs_cy=6.0)


def _make(shape, angle, Re):
    return SimulationConfig(Re=Re, obstacle_type=shape, obs_angle=angle,
                            obs_D=3.0, method="fvm", adv_blend=0.8, wall="freeslip",
                            seed_perturbation=True, **DOMAIN)


def _amp(cl):
    cl = np.asarray(cl)
    return 0.5 * (cl.max() - cl.min()) if len(cl) else 0.0


def is_shedding(shape, angle, Re, warm=1500, win=800, gap=3000):
    """True bila osilasi CL tumbuh (di atas onset) atau sudah jenuh besar."""
    s = NavierStokesSolver(_make(shape, angle, Re), "cpu")
    s.aero_sample_interval = 1
    for _ in range(warm):                 # lewati transien awal non-modal
        s.step()
    s.history_CL.clear()
    for _ in range(win):
        s.step(); s.compute_aerodynamics()
    amp_a = _amp(s.history_CL[-win:])
    for _ in range(gap):
        s.step(); s.compute_aerodynamics()
    amp_b = _amp(s.history_CL[-win:])
    growing = amp_b > 1.5 * amp_a and amp_b > 1e-3
    saturated = amp_b > 0.02
    return (growing or saturated), amp_b


def find_critical_Re(shape, angle):
    for Re in RE_CANDIDATES:
        shed, amp = is_shedding(shape, angle, Re)
        print(f"    Re={Re:>4}: amp CL = {amp:.4f}  {'SHEDDING' if shed else 'stabil'}")
        if shed:
            return Re, True
    return RE_CANDIDATES[-1], False


def render_at(shape, angle, render_Re, re_onset):
    s = NavierStokesSolver(_make(shape, angle, render_Re), "cpu")
    s.aero_sample_interval = 1
    s.re_onset = re_onset                   # tampil di legend
    for _ in range(8000):                   # biarkan shedding berkembang utk gambar
        s.step(); s.compute_aerodynamics()
    s.update_diagnostics()
    out = hasil_path(f"batch_{shape}_Re{int(render_Re)}_vorticity.png")
    r = Renderer(s.cfg, "vorticity")
    r.draw(s); r.save_png(out); r.close()
    return out


def main():
    print("=" * 62)
    print("  BATCH: Re KRITIS onset shedding per bentuk  (D=3, domain 36x12)")
    print("=" * 62)
    summary = []
    for shape, angle in SHAPES:
        print(f"\n  [{shape}] sudut={angle:g} deg -> cari Re onset (laju-tumbuh CL) ...")
        Re_c, found = find_critical_Re(shape, angle)
        # render di Re cukup di atas onset supaya shedding jelas terlihat di gambar
        render_Re = min(max(round(Re_c * 2), 140), 240) if found else Re_c
        if found:
            print(f"    => Re KRITIS ~ {Re_c}  (render gambar di Re={render_Re})")
        else:
            print(f"    => tak shedding s/d Re={Re_c} (bentuk ramping); render di Re={Re_c}")
        out = render_at(shape, angle, render_Re, Re_c)
        summary.append((shape, angle, Re_c, found))
        print(f"    tersimpan: {out}")

    print("\n" + "=" * 62)
    print("  RINGKASAN Re KRITIS (onset vortex shedding):")
    for shape, angle, Re_c, found in summary:
        tag = f"Re ~ {Re_c}" if found else f"> {Re_c} (tak shedding)"
        print(f"    {shape:9} (sudut {angle:>3g}) : {tag}")
    print("=" * 62)


if __name__ == "__main__":
    main()
