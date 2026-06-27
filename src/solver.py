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
        
        # metrik fisika terkini (diupdate saat diagnostik)
        self.current_CD = 0.0
        self.current_CL = 0.0
        self.current_St = 0.0
        self.current_Nu = 0.0
        
        # history untuk ekstraksi data fisika (CL, CD)
        self.history_CL = []
        self.history_CD = []
        self.aero_sample_interval = cfg.plot_every  # interval sampling untuk FFT

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

        # aerodinamika TIDAK dihitung di sini (dipindah ke update_diagnostics)

        self.step_count += 1
        self.time += dt

    def update_diagnostics(self):
        """Dipanggil tiap plot_every langkah: divergensi, residual, + metrik fisika."""
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
        
        # metrik fisika (vektorisasi, tetap di device)
        if cfg.obstacle_type != "none":
            self.current_CD, self.current_CL = self.compute_aerodynamics()
            self.current_St = self.compute_strouhal()
            self.current_Nu = self.compute_nusselt()

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

    # --- Metrik Fisika & Export Data ---
    
    def compute_aerodynamics(self):
        """Koefisien seret (CD) & angkat (CL) — vektorisasi penuh, tetap di device."""
        xp = self.xp
        ny, nx = self.cfg.ny, self.cfg.nx
        dx, dy = self.cfg.dx, self.cfg.dy
        
        # operasi langsung di device array (CuPy/NumPy), tanpa to_cpu
        S = self.d.mask_p          # solid mask (bool, device)
        F = ~S                     # fluid mask
        pc = self.d.p[1:ny+1, 1:nx+1]  # tekanan interior
        fc = F[1:ny+1, 1:nx+1]         # fluid interior
        
        # gaya tekanan pada obstacle: F = -∮ p·n dS  (n = normal keluar dari obstacle)
        # sel fluid dengan solid di TIMUR = muka depan/hulu obstacle -> tekan ke +x (drag)
        Fx = dy * ( (pc * (fc & S[1:ny+1, 2:nx+2])).sum()    # solid di timur  -> +x
                   -(pc * (fc & S[1:ny+1, 0:nx  ])).sum())   # solid di barat  -> -x
        Fy = dx * ( (pc * (fc & S[2:ny+2, 1:nx+1])).sum()    # solid di utara  -> +y
                   -(pc * (fc & S[0:ny,   1:nx+1])).sum())   # solid di selatan-> -y
        
        q = 0.5 * self.cfg.rho * self.cfg.U_inf**2 * self.cfg.obs_D
        CD = float(Fx) / q if q != 0 else 0.0
        CL = float(Fy) / q if q != 0 else 0.0
        
        self.history_CD.append(CD)
        self.history_CL.append(CL)
        return CD, CL

    def compute_strouhal(self):
        """Frekuensi vortex shedding dominan dari riwayat CL via FFT.
        
        Interval sampling = dt * aero_sample_interval karena CL hanya
        dicatat tiap plot_every langkah, bukan tiap langkah.
        """
        if len(self.history_CL) < 100:
            return 0.0
            
        CL = np.array(self.history_CL)
        half = len(CL) // 2
        CL_steady = CL[half:].copy()
        CL_steady -= np.mean(CL_steady)
        
        fft_vals = np.abs(np.fft.rfft(CL_steady))
        # interval sampling = dt * jumlah langkah antar-sampel
        sample_dt = self.dt * self.aero_sample_interval
        freqs = np.fft.rfftfreq(len(CL_steady), d=sample_dt)
        
        # abaikan DC (indeks 0)
        if len(fft_vals) > 1:
            idx = np.argmax(fft_vals[1:]) + 1
        else:
            return 0.0
        f_dom = freqs[idx]
        return f_dom * self.cfg.obs_D / self.cfg.U_inf

    def compute_nusselt(self):
        """Nusselt rata-rata di permukaan obstacle — vektorisasi penuh."""
        xp = self.xp
        ny, nx = self.cfg.ny, self.cfg.nx
        dx, dy = self.cfg.dx, self.cfg.dy
        
        S = self.d.mask_p
        F = ~S
        Tc = self.d.T[1:ny+1, 1:nx+1]
        fc = F[1:ny+1, 1:nx+1]
        T_obs = self.cfg.T_obs
        
        # fluks di masing-masing arah: (T_obs - T_fluid) / (0.5 * dn) * dl
        # timur/barat: dn=dx, dl=dy;  utara/selatan: dn=dy, dl=dx
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
        if cfg is not None:
            self.cfg = cfg
        self.__init__(self.cfg, self.mode)
