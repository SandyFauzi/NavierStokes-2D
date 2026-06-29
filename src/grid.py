# alokasi memori array dan mask

import numpy as np
from .config import SimulationConfig


def _obstacle_mask(cfg: SimulationConfig) -> np.ndarray:
    nx, ny = cfg.nx, cfg.ny
    ii = np.arange(nx + 2)
    jj = np.arange(ny + 2)
    X, Y = np.meshgrid((ii - 0.5) * cfg.dx, (jj - 0.5) * cfg.dy)

    # rotasi dan geser koordinat mask
    a = np.radians(cfg.obs_angle)
    px, py = X - cfg.obs_cx, Y - cfg.obs_cy
    xr = px * np.cos(a) + py * np.sin(a)
    yr = -px * np.sin(a) + py * np.cos(a)

    R = cfg.obs_D / 2.0
    t = cfg.obstacle_type
    if t == "cylinder":
        m = xr**2 + yr**2 <= R**2
    elif t == "ellipse":
        m = (xr / R)**2 + (yr / (0.55 * R))**2 <= 1.0
    elif t == "square":
        m = (np.abs(xr) <= R) & (np.abs(yr) <= R)
    elif t == "diamond":
        m = np.abs(xr) + np.abs(yr) <= R
    elif t == "hexagon":
        s = np.sqrt(3.0) / 2.0
        m = ((np.abs(xr) <= R) &
             (np.abs(0.5 * xr + s * yr) <= R) &
             (np.abs(0.5 * xr - s * yr) <= R))
    elif t == "triangle":
        half = (R - xr) / 2.0          # alas di kiri, apex di kanan
        m = (xr >= -R) & (xr <= R) & (np.abs(yr) <= half)
    elif t == "plate":
        th = max(cfg.dx * 2, cfg.obs_D * 0.1)
        m = (np.abs(xr) <= th / 2) & (np.abs(yr) <= R)
    else:
        m = np.zeros((ny + 2, nx + 2), dtype=bool)

    m[0, :] = m[ny + 1, :] = False     # abaikan sel halo
    m[:, 0] = m[:, nx + 1] = False
    return m


def obstacle_area(cfg: SimulationConfig) -> float:
    # luas penghalang
    return float(_obstacle_mask(cfg).sum()) * cfg.dx * cfg.dy


class FieldArrays:
    def __init__(self, cfg: SimulationConfig, xp):
        nx, ny = cfg.nx, cfg.ny
        dt = getattr(xp, cfg.dtype_str)     # float32 atau float64

        mp = _obstacle_mask(cfg)
        mu = mp[:, :nx + 1] | mp[:, 1:nx + 2]
        mv = mp[:ny + 1, :] | mp[1:ny + 2, :]

        self.mask_p_host = mp           # array numpy untuk render gui
        self.mask_p  = xp.asarray(mp)
        self.mask_u  = xp.asarray(mu)
        self.mask_v  = xp.asarray(mv)

        self.u = xp.full((ny + 2, nx + 1), cfg.U_inf, dtype=dt)
        self.u[self.mask_u] = 0.0
        self.v = xp.zeros((ny + 1, nx + 2), dtype=dt)
        self.p = xp.zeros((ny + 2, nx + 2), dtype=dt)
        self.T = xp.zeros((ny + 2, nx + 2), dtype=dt)
        self.T[self.mask_p] = cfg.T_obs

        # array alokasi bantu
        self.u_star = xp.zeros_like(self.u)
        self.v_star = xp.zeros_like(self.v)
        self.Hu = xp.zeros_like(self.u)
        self.Hv = xp.zeros_like(self.v)
        self.Du = xp.zeros_like(self.u)
        self.Dv = xp.zeros_like(self.v)
        self.div = xp.zeros_like(self.p)
        self.p_new = xp.zeros_like(self.p)
        self.rhs = xp.zeros_like(self.p)
        self.HT = xp.zeros_like(self.T)
        self.DT = xp.zeros_like(self.T)
        self.T_new = xp.zeros_like(self.T)
        self.omega = xp.zeros_like(self.p)
        self.vel_mag = xp.zeros_like(self.p)
