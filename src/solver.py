# solver navier-stokes chorin projection
# backend execution mode

import numpy as np
from .config import SimulationConfig
from .grid import FieldArrays
from . import backend, kernels


class NavierStokesSolver:
    def __init__(self, cfg: SimulationConfig, mode: str = None):
        self.cfg = cfg
        self.mode = mode or backend.default_mode()
        if self.mode == "gpu" and not backend.GPU_PRESENT:
            self.mode = "cpu"                      # cpu fallback

        if self.mode == "gpu":
            from . import gpu_ops as engine        # module cupy
        else:
            engine = kernels                       # module numba njit
        self.K = engine
        self.xp = backend.array_module(self.mode)

        self.d  = FieldArrays(cfg, self.xp)
        self.dt = cfg.compute_dt()
        self.step_count = 0
        self.time    = 0.0
        self.div_err = 0.0
        self.p_resid = 0.0

        # metrik fisika saat ini
        self.current_CD = 0.0
        self.current_CL = 0.0
        self.current_St = 0.0
        self.current_Nu = 0.0
        
        # array riwayat cl cd
        self.history_CL = []
        self.history_CD = []
        self.aero_sample_interval = cfg.plot_every  # interval fft cl

        dx, dy   = cfg.dx, cfg.dy
        self.dx2   = dx * dx
        self.dy2   = dy * dy
        self.coeff = 2.0 * (1.0 / self.dx2 + 1.0 / self.dy2)

        self.b      = cfg.adv_blend
        self.noslip = cfg.wall == "noslip"

        # fungsi adveksi
        if cfg.method == "fdm":
            self._adv_u, self._adv_v, self._adv_T = \
                self.K.advection_u_fdm, self.K.advection_v_fdm, self.K.advection_T_fdm
        else:
            self._adv_u, self._adv_v, self._adv_T = \
                self.K.advection_u_fvm, self.K.advection_v_fvm, self.K.advection_T_fvm

        if cfg.seed_perturbation: self._seed_shedding()

    def _seed_shedding(self):
        # trigger vortex shedding
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

        # adveksi dan difusi
        self._adv_u(d.u, d.v, d.Hu, nx, ny, dx, dy, self.b)
        self._adv_v(d.u, d.v, d.Hv, nx, ny, dx, dy, self.b)
        K.diffusion_u(d.u, d.Du, nx, ny, dx, dy, cfg.nu)
        K.diffusion_v(d.v, d.Dv, nx, ny, dx, dy, cfg.nu)

        # hitung kecepatan sementara
        K.tentative_u(d.u, d.Hu, d.Du, d.u_star, d.mask_u, nx, ny, dt)
        K.tentative_v(d.v, d.Hv, d.Dv, d.v_star, d.mask_v, nx, ny, dt)
        K.bc_u(d.u_star, d.mask_u, nx, ny, cfg.U_inf, self.noslip)
        K.bc_v(d.v_star, d.mask_v, nx, ny)

        # hitung tekanan poisson
        K.divergence(d.u_star, d.v_star, d.div, nx, ny, dx, dy)
        d.rhs[1:ny+1, 1:nx+1] = (cfg.rho / dt) * d.div[1:ny+1, 1:nx+1]
        if hasattr(K, "poisson_solve"):           # eksekusi rawkernel di gpu
            d.p, d.p_new = K.poisson_solve(d.p, d.p_new, d.rhs, d.mask_p,
                                           nx, ny, self.dx2, self.dy2, self.coeff,
                                           cfg.poisson_max_iter)
        else:
            for _ in range(cfg.poisson_max_iter):
                K.poisson_jacobi(d.p, d.p_new, d.rhs, d.mask_p,
                                 nx, ny, self.dx2, self.dy2, self.coeff)
                K.pressure_bc(d.p_new, nx, ny)
                d.p, d.p_new = d.p_new, d.p

        # koreksi kecepatan
        K.correct_u(d.u_star, d.p, d.u, d.mask_u, nx, ny, dt / (cfg.rho * dx))
        K.correct_v(d.v_star, d.p, d.v, d.mask_v, nx, ny, dt / (cfg.rho * dy))

        # update suhu
        self._adv_T(d.T, d.u, d.v, d.HT, nx, ny, dx, dy, self.b)
        K.diffusion_scalar(d.T, d.DT, nx, ny, dx, dy, cfg.alpha)
        K.update_T(d.T, d.HT, d.DT, d.T_new, d.mask_p, nx, ny, dt, cfg.T_obs)
        d.T, d.T_new = d.T_new, d.T

        # set kondisi batas
        K.bc_u(d.u, d.mask_u, nx, ny, cfg.U_inf, self.noslip)
        K.bc_v(d.v, d.mask_v, nx, ny)
        K.bc_T(d.T, d.mask_p, nx, ny, cfg.T_inf, cfg.T_obs)
        K.pressure_bc(d.p, nx, ny)

        # aerodinamika dihitung terpisah di update_diagnostics

        self.step_count += 1
        self.time += dt

    def update_diagnostics(self):
        # hitung divergensi, residual poisson, dan metrik fisika
        K, xp, cfg, d = self.K, self.xp, self.cfg, self.d
        nx, ny = cfg.nx, cfg.ny
        K.divergence(d.u, d.v, d.div, nx, ny, cfg.dx, cfg.dy)
        self.div_err = float(xp.abs(d.div[1:ny+1, 1:nx+1]).max())

        p = d.p
        lap = ((p[1:ny+1, 2:nx+2] + p[1:ny+1, 0:nx]) / self.dx2 +
               (p[2:ny+2, 1:nx+1] + p[0:ny, 1:nx+1]) / self.dy2 -
               self.coeff * p[1:ny+1, 1:nx+1])
        res = xp.where(d.mask_p[1:ny+1, 1:nx+1], 0.0, lap - d.rhs[1:ny+1, 1:nx+1])
        self.p_resid = float(xp.abs(res).max())
        
        # komputasi metrik fisika di device
        if cfg.obstacle_type != "none":
            self.current_CD, self.current_CL = self.compute_aerodynamics()
            self.current_St = self.compute_strouhal()
            self.current_Nu = self.compute_nusselt()

    # numpy array getter untuk gui
    def get_vorticity(self):
        cfg = self.cfg
        self.K.compute_vorticity(self.d.u, self.d.v, self.d.omega, cfg.nx, cfg.ny, cfg.dx, cfg.dy)
        return backend.to_cpu(self.d.omega[1:cfg.ny+1, 1:cfg.nx+1])

    def get_temperature(self):  return backend.to_cpu(self.d.T[1:self.cfg.ny+1, 1:self.cfg.nx+1])
    def get_pressure(self):     return backend.to_cpu(self.d.p[1:self.cfg.ny+1, 1:self.cfg.nx+1])
    def get_obstacle_mask(self): return self.d.mask_p_host[1:self.cfg.ny+1, 1:self.cfg.nx+1]

    def get_velocity_magnitude(self):
        cfg = self.cfg
        self.K.compute_velocity_mag(self.d.u, self.d.v, self.d.vel_mag, cfg.nx, cfg.ny)
        return backend.to_cpu(self.d.vel_mag[1:cfg.ny+1, 1:cfg.nx+1])

    # metrik fisika

    def compute_aerodynamics(self):
        # hitung koefisien cd dan cl
        xp = self.xp
        ny, nx = self.cfg.ny, self.cfg.nx
        dx, dy = self.cfg.dx, self.cfg.dy

        # array mask dan tekanan interior
        S = self.d.mask_p
        F = ~S
        pc = self.d.p[1:ny+1, 1:nx+1]
        fc = F[1:ny+1, 1:nx+1]

        # akumulasi gaya pada muka penghalang
        Fx = dy * ( (pc * (fc & S[1:ny+1, 2:nx+2])).sum()    # arah timur
                   -(pc * (fc & S[1:ny+1, 0:nx  ])).sum())   # arah barat
        Fy = dx * ( (pc * (fc & S[2:ny+2, 1:nx+1])).sum()    # arah utara
                   -(pc * (fc & S[0:ny,   1:nx+1])).sum())   # arah selatan
        
        q = 0.5 * self.cfg.rho * self.cfg.U_inf**2 * self.cfg.obs_D
        CD = float(Fx) / q if q else 0.0
        CL = float(Fy) / q if q else 0.0
        
        self.history_CD.append(CD)
        self.history_CL.append(CL)
        return CD, CL

    def compute_strouhal(self):
        # fft frekuensi vortex shedding dari cl
        if len(self.history_CL) < 100:
            return 0.0
            
        CL = np.array(self.history_CL)
        half = len(CL) // 2
        CL_steady = CL[half:].copy()
        CL_steady -= np.mean(CL_steady)
        
        fft_vals = np.abs(np.fft.rfft(CL_steady))
        # time delta per sampel
        sample_dt = self.dt * self.aero_sample_interval
        freqs = np.fft.rfftfreq(len(CL_steady), d=sample_dt)
        
        # abaikan komponen dc (idx 0)
        if len(fft_vals) > 1:
            idx = np.argmax(fft_vals[1:]) + 1
        else:
            return 0.0
        f_dom = freqs[idx]
        return f_dom * self.cfg.obs_D / self.cfg.U_inf

    def compute_nusselt(self):
        # vektorisasi perhitungan nusselt number
        xp = self.xp
        ny, nx = self.cfg.ny, self.cfg.nx
        dx, dy = self.cfg.dx, self.cfg.dy

        S = self.d.mask_p
        F = ~S
        Tc = self.d.T[1:ny+1, 1:nx+1]
        fc = F[1:ny+1, 1:nx+1]
        T_obs = self.cfg.T_obs

        # akumulasi fluks panas pada antarmuka fluida solid
        east  = fc & S[1:ny+1, 2:nx+2]
        west  = fc & S[1:ny+1, 0:nx]
        north = fc & S[2:ny+2, 1:nx+1]
        south = fc & S[0:ny,   1:nx+1]
        
        Q_ew = ((T_obs - Tc) * east).sum() / (0.5 * dx) * dy \
             + ((T_obs - Tc) * west).sum() / (0.5 * dx) * dy
        Q_ns = ((T_obs - Tc) * north).sum() / (0.5 * dy) * dx \
             + ((T_obs - Tc) * south).sum() / (0.5 * dy) * dx
        Q_total = float(Q_ew + Q_ns)
        
        A_tot = float(east.sum() + west.sum()) * dy \
              + float(north.sum() + south.sum()) * dx
        
        dT = T_obs - self.cfg.T_inf
        if A_tot == 0 or dT == 0:
            return 0.0
        return (Q_total / A_tot) * self.cfg.obs_D / dT

    def reset(self, cfg: SimulationConfig = None):
        if cfg: self.cfg = cfg
        self.__init__(self.cfg, self.mode)
