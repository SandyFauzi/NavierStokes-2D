"""Solver Navier-Stokes 2D — metode proyeksi Chorin. Backend cpu/gpu per-instance."""

import numpy as np
from .config import SimulationConfig
from .grid import FieldArrays
from . import backend, kernels


class NavierStokesSolver:
    def __init__(self, cfg: SimulationConfig, mode: str = None):
        self.cfg = cfg
        self.mode = mode or backend.default_mode()
        if self.mode == "gpu" and not backend.GPU_PRESENT:
            self.mode = "cpu"                      # fallback aman

        if self.mode == "gpu":
            from . import gpu_ops as engine        # operasi array CuPy
        else:
            engine = kernels                       # kernel Numba @njit
        self.K = engine
        self.xp = backend.array_module(self.mode)

        self.d = FieldArrays(cfg, self.xp)
        self.dt = cfg.compute_dt()
        self.step_count = 0
        self.time = 0.0
        self.divergence_error = 0.0
        self.poisson_residual = 0.0

        dx, dy = cfg.dx, cfg.dy
        self.dx2 = dx * dx
        self.dy2 = dy * dy
        self.coeff = 2.0 * (1.0 / self.dx2 + 1.0 / self.dy2)

        self.b = cfg.adv_blend
        self.noslip = (cfg.wall == "noslip")

        # pilih implementasi suku adveksi
        if cfg.method == "fdm":
            self._adv_u, self._adv_v, self._adv_T = (
                self.K.advection_u_fdm, self.K.advection_v_fdm, self.K.advection_T_fdm)
        else:
            self._adv_u, self._adv_v, self._adv_T = (
                self.K.advection_u_fvm, self.K.advection_v_fvm, self.K.advection_T_fvm)
            # hybrid: upwind FVM lewat kernel @cuda.jit Numba di atas array CuPy
            if self.mode == "gpu" and backend.NUMBA_CUDA:
                from . import gpu_cuda_kernels as KC
                self._adv_u, self._adv_v, self._adv_T = (
                    KC.advection_u_fvm, KC.advection_v_fvm, KC.advection_T_fvm)

        if cfg.seed_perturbation:
            self._seed_shedding()

    def _seed_shedding(self):
        # perturbasi-v kecil asimetris di belakang penghalang -> memicu shedding
        cfg = self.cfg
        j = int(cfg.obs_cy / cfg.dy)
        i0 = int((cfg.obs_cx + cfg.obs_D) / cfg.dx)
        i1 = min(cfg.nx, i0 + int(2.0 / cfg.dx))
        if i0 >= 1 and i1 > i0 and 1 <= j < cfg.ny:
            pert = 0.2 * cfg.U_inf * np.sin(np.linspace(0.0, np.pi, i1 - i0))
            self.d.v[j, i0:i1] += self.xp.asarray(pert)

    def step(self):
        K, cfg, d, dt = self.K, self.cfg, self.d, self.dt
        nx, ny = cfg.nx, cfg.ny
        dx, dy = cfg.dx, cfg.dy

        # adveksi + difusi
        self._adv_u(d.u, d.v, d.Hu, nx, ny, dx, dy, self.b)
        self._adv_v(d.u, d.v, d.Hv, nx, ny, dx, dy, self.b)
        K.diffusion_u(d.u, d.Du, nx, ny, dx, dy, cfg.nu)
        K.diffusion_v(d.v, d.Dv, nx, ny, dx, dy, cfg.nu)

        # kecepatan sementara (abaikan tekanan)
        K.tentative_u(d.u, d.Hu, d.Du, d.u_star, d.mask_u, nx, ny, dt)
        K.tentative_v(d.v, d.Hv, d.Dv, d.v_star, d.mask_v, nx, ny, dt)
        K.bc_u(d.u_star, d.mask_u, nx, ny, cfg.U_inf, self.noslip)
        K.bc_v(d.v_star, d.mask_v, nx, ny)

        # Poisson tekanan
        K.divergence(d.u_star, d.v_star, d.div, nx, ny, dx, dy)
        d.rhs[1:ny+1, 1:nx+1] = (cfg.rho / dt) * d.div[1:ny+1, 1:nx+1]
        if hasattr(K, "poisson_solve"):           # GPU: loop di RawKernel
            d.p, d.p_new = K.poisson_solve(d.p, d.p_new, d.rhs, d.mask_p,
                                           nx, ny, self.dx2, self.dy2, self.coeff,
                                           cfg.poisson_max_iter)
        else:
            for _ in range(cfg.poisson_max_iter):
                K.poisson_jacobi(d.p, d.p_new, d.rhs, d.mask_p,
                                 nx, ny, self.dx2, self.dy2, self.coeff)
                K.pressure_bc(d.p_new, nx, ny)
                d.p, d.p_new = d.p_new, d.p

        # koreksi kecepatan (proyeksi)
        K.correct_u(d.u_star, d.p, d.u, d.mask_u, nx, ny, dt / (cfg.rho * dx))
        K.correct_v(d.v_star, d.p, d.v, d.mask_v, nx, ny, dt / (cfg.rho * dy))

        # transport suhu
        self._adv_T(d.T, d.u, d.v, d.HT, nx, ny, dx, dy, self.b)
        K.diffusion_scalar(d.T, d.DT, nx, ny, dx, dy, cfg.alpha)
        K.update_T(d.T, d.HT, d.DT, d.T_new, d.mask_p, nx, ny, dt, cfg.T_obs)
        d.T, d.T_new = d.T_new, d.T

        # kondisi batas
        K.bc_u(d.u, d.mask_u, nx, ny, cfg.U_inf, self.noslip)
        K.bc_v(d.v, d.mask_v, nx, ny)
        K.bc_T(d.T, d.mask_p, nx, ny, cfg.T_inf, cfg.T_obs)
        K.pressure_bc(d.p, nx, ny)

        self.step_count += 1
        self.time += dt

    def update_diagnostics(self):
        # max|div u| dan residual Poisson (dipanggil saat butuh saja)
        K, xp, cfg, d = self.K, self.xp, self.cfg, self.d
        nx, ny = cfg.nx, cfg.ny
        K.divergence(d.u, d.v, d.div, nx, ny, cfg.dx, cfg.dy)
        self.divergence_error = float(xp.abs(d.div[1:ny+1, 1:nx+1]).max())

        p = d.p
        lap = ((p[1:ny+1, 2:nx+2] + p[1:ny+1, 0:nx]) / self.dx2 +
               (p[2:ny+2, 1:nx+1] + p[0:ny, 1:nx+1]) / self.dy2 -
               self.coeff * p[1:ny+1, 1:nx+1])
        res = xp.where(d.mask_p[1:ny+1, 1:nx+1], 0.0, lap - d.rhs[1:ny+1, 1:nx+1])
        self.poisson_residual = float(xp.abs(res).max())

    # getter selalu kembalikan numpy (untuk GUI/plot)
    def get_vorticity(self):
        cfg = self.cfg
        self.K.compute_vorticity(self.d.u, self.d.v, self.d.omega, cfg.nx, cfg.ny, cfg.dx, cfg.dy)
        return backend.to_cpu(self.d.omega[1:cfg.ny+1, 1:cfg.nx+1])

    def get_temperature(self):
        return backend.to_cpu(self.d.T[1:self.cfg.ny+1, 1:self.cfg.nx+1])

    def get_velocity_magnitude(self):
        cfg = self.cfg
        self.K.compute_velocity_mag(self.d.u, self.d.v, self.d.vel_mag, cfg.nx, cfg.ny)
        return backend.to_cpu(self.d.vel_mag[1:cfg.ny+1, 1:cfg.nx+1])

    def get_pressure(self):
        return backend.to_cpu(self.d.p[1:self.cfg.ny+1, 1:self.cfg.nx+1])

    def get_obstacle_mask(self):
        return self.d.mask_p_host[1:self.cfg.ny+1, 1:self.cfg.nx+1]

    def reset(self, cfg: SimulationConfig = None):
        if cfg is not None:
            self.cfg = cfg
        self.__init__(self.cfg, self.mode)
