# parameter simulasi dan konstanta fisika

import os
from dataclasses import dataclass

# set path output ke folder results
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def results_path(filename):
    # buat folder results jika belum ada
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return os.path.join(RESULTS_DIR, filename)


@dataclass
class SimulationConfig:
    # domain
    Lx: float = 25.0
    Ly: float = 10.0
    nx: int = 300
    ny: int = 120

    # fisika
    Re: float = 150.0          # bilangan Reynolds
    Pr: float = 0.71           # bilangan Prandtl untuk udara
    U_inf: float = 1.0
    rho: float = 1.0
    T_inf: float = 0.0
    T_obs: float = 1.0         # suhu permukaan penghalang

    # penghalang
    obstacle_type: str = "cylinder"   # jenis penghalang: cylinder, ellipse, square, diamond, hexagon, triangle, plate
    obs_cx: float = 6.0
    obs_cy: float = 5.0
    obs_D: float = 1.0         # ukuran karakteristik
    obs_angle: float = 0.0     # orientasi penghalang dalam derajat

    # numerik
    method: str = "fvm"        # pilih fdm atau fvm
    dt: float = 0.0            # nol berarti dihitung otomatis dari CFL
    cfl: float = 0.2
    adv_blend: float = 0.8     # porsi skema central pada adveksi FVM, nol berarti upwind penuh
    wall: str = "freeslip"     # pilih freeslip atau noslip
    seed_perturbation: bool = True
    precision: str = "single"  # presisi komputasi: float32 (single) atau float64 (double)
    n_steps: int = 20_000
    poisson_max_iter: int = 60
    poisson_tol: float = 1e-5

    # output
    plot_every: int = 25
    save_every: int = 500

    @property
    def dx(self): return self.Lx / self.nx
    @property
    def dy(self): return self.Ly / self.ny
    @property
    def nu(self): return self.U_inf * self.obs_D / self.Re
    @property
    def alpha(self): return self.nu / self.Pr
    @property
    def dtype_str(self): return "float32" if self.precision == "single" else "float64"

    def compute_dt(self):
        if self.dt > 0: return self.dt
        h = min(self.dx, self.dy)
        dt_adv  = self.cfl * h / self.U_inf            # syarat CFL adveksi
        dt_diff = self.cfl * 0.25 * h * h / self.nu    # syarat difusi 2d
        return min(dt_adv, dt_diff)
