"""Parameter simulasi & konstanta fisika (tak-berdimensi)."""

import os
from dataclasses import dataclass

# folder output selalu di dalam NavierStokes-2D/Hasil (anchor ke lokasi paket, bukan cwd)
HASIL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Hasil")


def hasil_path(filename: str) -> str:
    """Path absolut file output di NavierStokes-2D/Hasil (folder dibuat bila belum ada)."""
    os.makedirs(HASIL_DIR, exist_ok=True)
    return os.path.join(HASIL_DIR, filename)


@dataclass
class SimulationConfig:
    # Domain
    Lx: float = 25.0
    Ly: float = 10.0
    nx: int = 300
    ny: int = 120

    # Fisika
    Re: float = 150.0          # Reynolds = U*D/nu
    Pr: float = 0.71           # Prandtl = nu/alpha (udara)
    U_inf: float = 1.0
    rho: float = 1.0
    T_inf: float = 0.0
    T_obs: float = 1.0         # suhu permukaan penghalang

    # Penghalang
    obstacle_type: str = "cylinder"   # cylinder/ellipse/square/diamond/hexagon/triangle/plate
    obs_cx: float = 6.0
    obs_cy: float = 5.0
    obs_D: float = 1.0         # ukuran karakteristik
    obs_angle: float = 0.0     # orientasi penghalang (derajat)

    # Numerik
    method: str = "fvm"        # "fdm" atau "fvm"
    dt: float = 0.0            # 0 = otomatis dari CFL
    cfl: float = 0.2
    adv_blend: float = 0.8     # fraksi central pada adveksi FVM (0=upwind, ~0.8=shedding)
    wall: str = "freeslip"     # "freeslip" atau "noslip"
    seed_perturbation: bool = True
    precision: str = "single"  # "single" (float32, cepat di GPU) / "double" (float64)
    n_steps: int = 20_000
    poisson_max_iter: int = 60
    poisson_tol: float = 1e-5

    # Output
    plot_every: int = 25
    save_every: int = 500

    @property
    def dx(self) -> float:
        return self.Lx / self.nx

    @property
    def dy(self) -> float:
        return self.Ly / self.ny

    @property
    def nu(self) -> float:
        return self.U_inf * self.obs_D / self.Re

    @property
    def alpha(self) -> float:
        return self.nu / self.Pr

    @property
    def dtype_str(self) -> str:
        return "float32" if self.precision == "single" else "float64"

    def compute_dt(self) -> float:
        if self.dt > 0:
            return self.dt
        h = min(self.dx, self.dy)
        dt_adv = self.cfl * h / self.U_inf            # syarat CFL adveksi
        dt_diff = self.cfl * 0.25 * h * h / self.nu   # syarat difusi (2D)
        return min(dt_adv, dt_diff)
