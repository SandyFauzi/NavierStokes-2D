"""Kernel CFD untuk GPU via operasi array CuPy. Tanda tangan = kernels (CPU).

Grid berselang-seling (halo 1 sel):
    u   (ny+2, nx+1)   v   (ny+1, nx+2)   p,T (ny+2, nx+2)
"""

from .backend import cp           # modul ini hanya dipakai pada mode GPU
xp = cp                           # operasi array memakai cupy

# RawKernel Poisson: 1 sweep Jacobi + BC Neumann (clamp indeks) + mask, satu launch.
# Memangkas ~10 launch kecil/iterasi jadi 1 (penyebab utama overhead di GPU).
# REAL = float (fp32) / double (fp64) — dikompilasi sesuai presisi medan.
_JACOBI_SRC = r'''
extern "C" __global__
void jacobi(const REAL* p, REAL* pn, const REAL* rhs,
            const char* mask, int nx, int ny,
            REAL dx2, REAL dy2, REAL coeff) {
    int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
    int j = blockDim.y * blockIdx.y + threadIdx.y + 1;
    if (i > nx || j > ny) return;
    int W = nx + 2, idx = j * W + i;
    if (mask[idx]) { pn[idx] = 0; return; }
    int ie = (i < nx) ? i + 1 : i;     // clamp = Neumann
    int iw = (i > 1)  ? i - 1 : i;
    int jn = (j < ny) ? j + 1 : j;
    int js = (j > 1)  ? j - 1 : j;
    pn[idx] = ((p[j*W + ie] + p[j*W + iw]) / dx2 +
               (p[jn*W + i] + p[js*W + i]) / dy2 - rhs[idx]) / coeff;
}
'''
_jacobi_cache = {}


def _jacobi_kernel(dtype):
    key = "float" if dtype == cp.float32 else "double"
    if key not in _jacobi_cache:
        _jacobi_cache[key] = cp.RawKernel(_JACOBI_SRC.replace("REAL", key), "jacobi")
    return _jacobi_cache[key]


def poisson_solve(p, p_new, rhs, mask_p, nx, ny, dx2, dy2, coeff, n_iter):
    # n_iter sweep Jacobi (1 launch/iter); kembalikan (solusi, buffer_lain)
    kern = _jacobi_kernel(p.dtype)
    rt = p.dtype.type
    sc = (rt(dx2), rt(dy2), rt(coeff))
    mask_c = mask_p.astype(cp.int8)
    block = (16, 16)
    grid = ((nx + 15) // 16, (ny + 15) // 16)
    a, b = p, p_new
    for _ in range(n_iter):
        kern(grid, block, (a, b, rhs, mask_c, nx, ny, *sc))
        a, b = b, a
    pressure_bc(a, nx, ny)
    return a, b


# --- Adveksi FDM (central) ---

def advection_u_fdm(u, v, Hu, nx, ny, dx, dy, b=0.0):
    C = u[1:ny+1, 1:nx]
    E = u[1:ny+1, 2:nx+1]
    W = u[1:ny+1, 0:nx-1]
    N = u[2:ny+2, 1:nx]
    S = u[0:ny, 1:nx]
    v_int = 0.25 * (v[0:ny, 1:nx] + v[0:ny, 2:nx+1] +
                    v[1:ny+1, 1:nx] + v[1:ny+1, 2:nx+1])
    Hu[1:ny+1, 1:nx] = C * (E - W) / (2.0 * dx) + v_int * (N - S) / (2.0 * dy)


def advection_v_fdm(u, v, Hv, nx, ny, dx, dy, b=0.0):
    vC = v[1:ny, 1:nx+1]
    vE = v[1:ny, 2:nx+2]
    vW = v[1:ny, 0:nx]
    vN = v[2:ny+1, 1:nx+1]
    vS = v[0:ny-1, 1:nx+1]
    u_int = 0.25 * (u[1:ny, 0:nx] + u[1:ny, 1:nx+1] +
                    u[2:ny+1, 0:nx] + u[2:ny+1, 1:nx+1])
    Hv[1:ny, 1:nx+1] = u_int * (vE - vW) / (2.0 * dx) + vC * (vN - vS) / (2.0 * dy)


# --- Adveksi FVM (blended) ---

def advection_u_fvm(u, v, Hu, nx, ny, dx, dy, b):
    inv_vol = 1.0 / (dx * dy)
    ub = 1.0 - b
    C = u[1:ny+1, 1:nx]
    E = u[1:ny+1, 2:nx+1]
    W = u[1:ny+1, 0:nx-1]
    N = u[2:ny+2, 1:nx]
    S = u[0:ny, 1:nx]
    ue = 0.5 * (C + E)
    uw = 0.5 * (W + C)
    vn = 0.5 * (v[1:ny+1, 1:nx] + v[1:ny+1, 2:nx+1])
    vs = 0.5 * (v[0:ny, 1:nx] + v[0:ny, 2:nx+1])
    pe = b*0.5*(C + E) + ub*xp.where(ue >= 0, C, E)
    pw = b*0.5*(W + C) + ub*xp.where(uw >= 0, W, C)
    pn = b*0.5*(C + N) + ub*xp.where(vn >= 0, C, N)
    ps = b*0.5*(S + C) + ub*xp.where(vs >= 0, S, C)
    Hu[1:ny+1, 1:nx] = ((ue*pe - uw*pw) * dy + (vn*pn - vs*ps) * dx) * inv_vol


def advection_v_fvm(u, v, Hv, nx, ny, dx, dy, b):
    inv_vol = 1.0 / (dx * dy)
    ub = 1.0 - b
    vC = v[1:ny, 1:nx+1]
    vE = v[1:ny, 2:nx+2]
    vW = v[1:ny, 0:nx]
    vN = v[2:ny+1, 1:nx+1]
    vS = v[0:ny-1, 1:nx+1]
    ue = 0.5 * (u[1:ny, 1:nx+1] + u[2:ny+1, 1:nx+1])
    uw = 0.5 * (u[1:ny, 0:nx] + u[2:ny+1, 0:nx])
    vn = 0.5 * (vC + vN)
    vs = 0.5 * (vS + vC)
    pe = b*0.5*(vC + vE) + ub*xp.where(ue >= 0, vC, vE)
    pw = b*0.5*(vW + vC) + ub*xp.where(uw >= 0, vW, vC)
    pn = b*0.5*(vC + vN) + ub*xp.where(vn >= 0, vC, vN)
    ps = b*0.5*(vS + vC) + ub*xp.where(vs >= 0, vS, vC)
    Hv[1:ny, 1:nx+1] = ((ue*pe - uw*pw) * dy + (vn*pn - vs*ps) * dx) * inv_vol


# --- Difusi ---

def diffusion_u(u, Du, nx, ny, dx, dy, nu):
    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)
    C = u[1:ny+1, 1:nx]
    Du[1:ny+1, 1:nx] = nu * (
        (u[1:ny+1, 2:nx+1] - 2.0*C + u[1:ny+1, 0:nx-1]) * idx2 +
        (u[2:ny+2, 1:nx] - 2.0*C + u[0:ny, 1:nx]) * idy2)


def diffusion_v(v, Dv, nx, ny, dx, dy, nu):
    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)
    C = v[1:ny, 1:nx+1]
    Dv[1:ny, 1:nx+1] = nu * (
        (v[1:ny, 2:nx+2] - 2.0*C + v[1:ny, 0:nx]) * idx2 +
        (v[2:ny+1, 1:nx+1] - 2.0*C + v[0:ny-1, 1:nx+1]) * idy2)


# --- Kecepatan sementara + koreksi ---

def tentative_u(u, Hu, Du, u_star, mask_u, nx, ny, dt):
    sl = (slice(1, ny+1), slice(1, nx))
    val = u[sl] + dt * (-Hu[sl] + Du[sl])
    u_star[sl] = xp.where(mask_u[sl], 0.0, val)


def tentative_v(v, Hv, Dv, v_star, mask_v, nx, ny, dt):
    sl = (slice(1, ny), slice(1, nx+1))
    val = v[sl] + dt * (-Hv[sl] + Dv[sl])
    v_star[sl] = xp.where(mask_v[sl], 0.0, val)


def correct_u(u_star, p, u_new, mask_u, nx, ny, dt_rho_dx):
    sl = (slice(1, ny+1), slice(1, nx))
    val = u_star[sl] - dt_rho_dx * (p[1:ny+1, 2:nx+1] - p[1:ny+1, 1:nx])
    u_new[sl] = xp.where(mask_u[sl], 0.0, val)


def correct_v(v_star, p, v_new, mask_v, nx, ny, dt_rho_dy):
    sl = (slice(1, ny), slice(1, nx+1))
    val = v_star[sl] - dt_rho_dy * (p[2:ny+1, 1:nx+1] - p[1:ny, 1:nx+1])
    v_new[sl] = xp.where(mask_v[sl], 0.0, val)


# --- Divergensi + Poisson (per-sweep, dipakai bila tanpa RawKernel) ---

def divergence(u, v, div, nx, ny, dx, dy):
    div[1:ny+1, 1:nx+1] = (
        (u[1:ny+1, 1:nx+1] - u[1:ny+1, 0:nx]) / dx +
        (v[1:ny+1, 1:nx+1] - v[0:ny, 1:nx+1]) / dy)


def poisson_jacobi(p, p_new, rhs, mask_p, nx, ny, dx2, dy2, coeff):
    val = ((p[1:ny+1, 2:nx+2] + p[1:ny+1, 0:nx]) / dx2 +
           (p[2:ny+2, 1:nx+1] + p[0:ny, 1:nx+1]) / dy2 -
           rhs[1:ny+1, 1:nx+1]) / coeff
    p_new[1:ny+1, 1:nx+1] = xp.where(mask_p[1:ny+1, 1:nx+1], 0.0, val)


def pressure_bc(p, nx, ny):
    p[:, 0] = p[:, 1]
    p[:, nx+1] = p[:, nx]
    p[0, :] = p[1, :]
    p[ny+1, :] = p[ny, :]


# --- Transport suhu ---

def advection_T_fdm(T, u, v, HT, nx, ny, dx, dy, b=0.0):
    uc = 0.5 * (u[1:ny+1, 0:nx] + u[1:ny+1, 1:nx+1])
    vc = 0.5 * (v[0:ny, 1:nx+1] + v[1:ny+1, 1:nx+1])
    HT[1:ny+1, 1:nx+1] = (
        uc * (T[1:ny+1, 2:nx+2] - T[1:ny+1, 0:nx]) / (2.0*dx) +
        vc * (T[2:ny+2, 1:nx+1] - T[0:ny, 1:nx+1]) / (2.0*dy))


def advection_T_fvm(T, u, v, HT, nx, ny, dx, dy, b):
    inv_vol = 1.0 / (dx * dy)
    ub = 1.0 - b
    ue = u[1:ny+1, 1:nx+1]
    uw = u[1:ny+1, 0:nx]
    vn = v[1:ny+1, 1:nx+1]
    vs = v[0:ny, 1:nx+1]
    TC = T[1:ny+1, 1:nx+1]
    TE = T[1:ny+1, 2:nx+2]
    TW = T[1:ny+1, 0:nx]
    TN = T[2:ny+2, 1:nx+1]
    TS = T[0:ny, 1:nx+1]
    Te = b*0.5*(TC + TE) + ub*xp.where(ue >= 0, TC, TE)
    Tw = b*0.5*(TW + TC) + ub*xp.where(uw >= 0, TW, TC)
    Tn = b*0.5*(TC + TN) + ub*xp.where(vn >= 0, TC, TN)
    Ts = b*0.5*(TS + TC) + ub*xp.where(vs >= 0, TS, TC)
    HT[1:ny+1, 1:nx+1] = ((ue*Te - uw*Tw) * dy + (vn*Tn - vs*Ts) * dx) * inv_vol


def diffusion_scalar(phi, D, nx, ny, dx, dy, kappa):
    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)
    C = phi[1:ny+1, 1:nx+1]
    D[1:ny+1, 1:nx+1] = kappa * (
        (phi[1:ny+1, 2:nx+2] - 2.0*C + phi[1:ny+1, 0:nx]) * idx2 +
        (phi[2:ny+2, 1:nx+1] - 2.0*C + phi[0:ny, 1:nx+1]) * idy2)


def update_T(T, HT, DT, T_new, mask_p, nx, ny, dt, T_obs):
    sl = (slice(1, ny+1), slice(1, nx+1))
    val = T[sl] + dt * (-HT[sl] + DT[sl])
    T_new[sl] = xp.where(mask_p[sl], T_obs, val)


# --- Kondisi batas ---

def bc_u(u, mask_u, nx, ny, U_inf, noslip):
    u[:, 0] = U_inf
    u[:, nx] = u[:, nx - 1]
    if noslip:
        u[0, :] = -u[1, :]
        u[ny+1, :] = -u[ny, :]
    else:
        u[0, :] = u[1, :]
        u[ny+1, :] = u[ny, :]
    u[mask_u] = 0.0


def bc_v(v, mask_v, nx, ny):
    v[0, :] = 0.0
    v[ny, :] = 0.0
    v[:, 0] = -v[:, 1]
    v[:, nx+1] = v[:, nx]
    v[mask_v] = 0.0


def bc_T(T, mask_p, nx, ny, T_inf, T_obs):
    T[:, 0] = T_inf
    T[:, nx+1] = T[:, nx]
    T[0, :] = T[1, :]
    T[ny+1, :] = T[ny, :]
    T[mask_p] = T_obs


# --- Post-processing ---

def compute_vorticity(u, v, omega, nx, ny, dx, dy):
    omega[1:ny+1, 1:nx+1] = (
        (v[1:ny+1, 2:nx+2] - v[1:ny+1, 1:nx+1]) / dx -
        (u[2:ny+2, 1:nx+1] - u[1:ny+1, 1:nx+1]) / dy)


def compute_velocity_mag(u, v, mag, nx, ny):
    uc = 0.5 * (u[1:ny+1, 0:nx] + u[1:ny+1, 1:nx+1])
    vc = 0.5 * (v[0:ny, 1:nx+1] + v[1:ny+1, 1:nx+1])
    mag[1:ny+1, 1:nx+1] = xp.sqrt(uc*uc + vc*vc)
