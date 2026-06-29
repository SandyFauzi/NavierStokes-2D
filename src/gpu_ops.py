# kernel cfd gpu (cupy)
# signature argumen setara cpu (numba)
# grid selang seling dengan sel halo

from .backend import cp           # modul cupy
xp = cp                           # mapping numpy ke cupy

# elementwise kernel cupy
# penggabungan array untuk meminimalisasi alokasi memori
_diff_kern = cp.ElementwiseKernel(
    'T C, T E, T W, T N, T S, T nu, T idx2, T idy2', 'T D',
    'D = nu * ((E - 2.0*C + W) * idx2 + (N - 2.0*C + S) * idy2)', 'diff_kern'
)

_tentative_kern = cp.ElementwiseKernel(
    'T u, T H, T D, T dt, bool mask', 'T u_star',
    'u_star = mask ? 0.0 : u + dt * (-H + D)', 'tentative_kern'
)

_correct_kern = cp.ElementwiseKernel(
    'T u_star, T pE, T pC, T dt_rho_d, bool mask', 'T u_new',
    'u_new = mask ? 0.0 : u_star - dt_rho_d * (pE - pC)', 'correct_kern'
)

_div_kern = cp.ElementwiseKernel(
    'T uE, T uC, T vN, T vC, T dx, T dy', 'T div',
    'div = (uE - uC) / dx + (vN - vC) / dy', 'div_kern'
)

_update_T_kern = cp.ElementwiseKernel(
    'T T_old, T H, T D, T dt, bool mask, T T_obs', 'T T_new',
    'T_new = mask ? (T)T_obs : T_old + dt * (-H + D)', 'update_T_kern'
)

_adv_u_fvm_kern = cp.ElementwiseKernel(
    'T C, T E, T W, T N, T S, T vn, T vs, T dy, T dx, T inv_vol, T b, T ub', 'T Hu',
    '''
    T ue = 0.5 * (C + E);
    T uw = 0.5 * (W + C);
    T pe = b * 0.5 * (C + E) + ub * (ue >= 0 ? C : E);
    T pw = b * 0.5 * (W + C) + ub * (uw >= 0 ? W : C);
    T pn = b * 0.5 * (C + N) + ub * (vn >= 0 ? C : N);
    T ps = b * 0.5 * (S + C) + ub * (vs >= 0 ? S : C);
    Hu = ((ue * pe - uw * pw) * dy + (vn * pn - vs * ps) * dx) * inv_vol;
    ''', 'adv_u_fvm_kern'
)

_adv_v_fvm_kern = cp.ElementwiseKernel(
    'T vC, T vE, T vW, T vN, T vS, T ue, T uw, T dy, T dx, T inv_vol, T b, T ub', 'T Hv',
    '''
    T vn = 0.5 * (vC + vN);
    T vs = 0.5 * (vS + vC);
    T pe = b * 0.5 * (vC + vE) + ub * (ue >= 0 ? vC : vE);
    T pw = b * 0.5 * (vW + vC) + ub * (uw >= 0 ? vW : vC);
    T pn = b * 0.5 * (vC + vN) + ub * (vn >= 0 ? vC : vN);
    T ps = b * 0.5 * (vS + vC) + ub * (vs >= 0 ? vS : vC);
    Hv = ((ue * pe - uw * pw) * dy + (vn * pn - vs * ps) * dx) * inv_vol;
    ''', 'adv_v_fvm_kern'
)

_adv_T_fvm_kern = cp.ElementwiseKernel(
    'T TC, T TE, T TW, T TN, T TS, T ue, T uw, T vn, T vs, T dy, T dx, T inv_vol, T b, T ub', 'T HT',
    '''
    T Te = b * 0.5 * (TC + TE) + ub * (ue >= 0 ? TC : TE);
    T Tw = b * 0.5 * (TW + TC) + ub * (uw >= 0 ? TW : TC);
    T Tn = b * 0.5 * (TC + TN) + ub * (vn >= 0 ? TC : TN);
    T Ts = b * 0.5 * (TS + TC) + ub * (vs >= 0 ? TS : TC);
    HT = ((ue * Te - uw * Tw) * dy + (vn * Tn - vs * Ts) * dx) * inv_vol;
    ''', 'adv_T_fvm_kern'
)

# rawkernel poisson: sapuan jacobi, kondisi batas neumann, dan mask dalam sekali launch
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
    int ie = (i < nx) ? i + 1 : i;     // clamp indeks untuk syarat Neumann
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

# adveksi fdm (beda pusat)
def advection_u_fdm(u, v, Hu, nx, ny, dx, dy, b=0.0):
    C, E, W = u[1:ny+1, 1:nx], u[1:ny+1, 2:nx+1], u[1:ny+1, 0:nx-1]
    N, S = u[2:ny+2, 1:nx], u[0:ny, 1:nx]
    v_int = 0.25 * (v[0:ny, 1:nx] + v[0:ny, 2:nx+1] + v[1:ny+1, 1:nx] + v[1:ny+1, 2:nx+1])
    Hu[1:ny+1, 1:nx] = C * (E - W) / (2.0 * dx) + v_int * (N - S) / (2.0 * dy)

def advection_v_fdm(u, v, Hv, nx, ny, dx, dy, b=0.0):
    vC, vE, vW = v[1:ny, 1:nx+1], v[1:ny, 2:nx+2], v[1:ny, 0:nx]
    vN, vS = v[2:ny+1, 1:nx+1], v[0:ny-1, 1:nx+1]
    u_int = 0.25 * (u[1:ny, 0:nx] + u[1:ny, 1:nx+1] + u[2:ny+1, 0:nx] + u[2:ny+1, 1:nx+1])
    Hv[1:ny, 1:nx+1] = u_int * (vE - vW) / (2.0 * dx) + vC * (vN - vS) / (2.0 * dy)

# adveksi fvm (hybrid beda pusat & upwind)
def advection_u_fvm(u, v, Hu, nx, ny, dx, dy, b):
    rt = u.dtype.type
    vn = 0.5 * (v[1:ny+1, 1:nx] + v[1:ny+1, 2:nx+1])
    vs = 0.5 * (v[0:ny, 1:nx] + v[0:ny, 2:nx+1])
    _adv_u_fvm_kern(
        u[1:ny+1, 1:nx], u[1:ny+1, 2:nx+1], u[1:ny+1, 0:nx-1],
        u[2:ny+2, 1:nx], u[0:ny, 1:nx], vn, vs,
        rt(dy), rt(dx), rt(1.0/(dx*dy)), rt(b), rt(1.0 - b),
        Hu[1:ny+1, 1:nx]
    )

def advection_v_fvm(u, v, Hv, nx, ny, dx, dy, b):
    rt = u.dtype.type
    ue = 0.5 * (u[1:ny, 1:nx+1] + u[2:ny+1, 1:nx+1])
    uw = 0.5 * (u[1:ny, 0:nx] + u[2:ny+1, 0:nx])
    _adv_v_fvm_kern(
        v[1:ny, 1:nx+1], v[1:ny, 2:nx+2], v[1:ny, 0:nx],
        v[2:ny+1, 1:nx+1], v[0:ny-1, 1:nx+1], ue, uw,
        rt(dy), rt(dx), rt(1.0/(dx*dy)), rt(b), rt(1.0 - b),
        Hv[1:ny, 1:nx+1]
    )

# difusi
def diffusion_u(u, Du, nx, ny, dx, dy, nu):
    rt = u.dtype.type
    _diff_kern(u[1:ny+1, 1:nx], u[1:ny+1, 2:nx+1], u[1:ny+1, 0:nx-1],
               u[2:ny+2, 1:nx], u[0:ny, 1:nx], rt(nu), rt(1.0/(dx*dx)), rt(1.0/(dy*dy)),
               Du[1:ny+1, 1:nx])

def diffusion_v(v, Dv, nx, ny, dx, dy, nu):
    rt = v.dtype.type
    _diff_kern(v[1:ny, 1:nx+1], v[1:ny, 2:nx+2], v[1:ny, 0:nx],
               v[2:ny+1, 1:nx+1], v[0:ny-1, 1:nx+1], rt(nu), rt(1.0/(dx*dx)), rt(1.0/(dy*dy)),
               Dv[1:ny, 1:nx+1])

# kalkulasi & koreksi kecepatan sementara
def tentative_u(u, Hu, Du, u_star, mask_u, nx, ny, dt):
    sl = (slice(1, ny+1), slice(1, nx))
    _tentative_kern(u[sl], Hu[sl], Du[sl], u.dtype.type(dt), mask_u[sl], u_star[sl])

def tentative_v(v, Hv, Dv, v_star, mask_v, nx, ny, dt):
    sl = (slice(1, ny), slice(1, nx+1))
    _tentative_kern(v[sl], Hv[sl], Dv[sl], v.dtype.type(dt), mask_v[sl], v_star[sl])

def correct_u(u_star, p, u_new, mask_u, nx, ny, dt_rho_dx):
    sl = (slice(1, ny+1), slice(1, nx))
    _correct_kern(u_star[sl], p[1:ny+1, 2:nx+1], p[1:ny+1, 1:nx], u_star.dtype.type(dt_rho_dx), mask_u[sl], u_new[sl])

def correct_v(v_star, p, v_new, mask_v, nx, ny, dt_rho_dy):
    sl = (slice(1, ny), slice(1, nx+1))
    _correct_kern(v_star[sl], p[2:ny+1, 1:nx+1], p[1:ny, 1:nx+1], v_star.dtype.type(dt_rho_dy), mask_v[sl], v_new[sl])

# divergensi dan persamaan Poisson
def divergence(u, v, div, nx, ny, dx, dy):
    rt = u.dtype.type
    _div_kern(u[1:ny+1, 1:nx+1], u[1:ny+1, 0:nx],
              v[1:ny+1, 1:nx+1], v[0:ny, 1:nx+1],
              rt(dx), rt(dy), div[1:ny+1, 1:nx+1])

def poisson_jacobi(p, p_new, rhs, mask_p, nx, ny, dx2, dy2, coeff):
    # fallback jacobi bila rawkernel gagal diluncurkan
    val = ((p[1:ny+1, 2:nx+2] + p[1:ny+1, 0:nx]) / dx2 +
           (p[2:ny+2, 1:nx+1] + p[0:ny, 1:nx+1]) / dy2 -
           rhs[1:ny+1, 1:nx+1]) / coeff
    p_new[1:ny+1, 1:nx+1] = xp.where(mask_p[1:ny+1, 1:nx+1], 0.0, val)

def pressure_bc(p, nx, ny):
    p[:, 0] = p[:, 1]
    p[:, nx+1] = p[:, nx]
    p[0, :] = p[1, :]
    p[ny+1, :] = p[ny, :]

# transport suhu
def advection_T_fdm(T, u, v, HT, nx, ny, dx, dy, b=0.0):
    uc = 0.5 * (u[1:ny+1, 0:nx] + u[1:ny+1, 1:nx+1])
    vc = 0.5 * (v[0:ny, 1:nx+1] + v[1:ny+1, 1:nx+1])
    HT[1:ny+1, 1:nx+1] = (uc * (T[1:ny+1, 2:nx+2] - T[1:ny+1, 0:nx]) / (2.0*dx) +
                          vc * (T[2:ny+2, 1:nx+1] - T[0:ny, 1:nx+1]) / (2.0*dy))

def advection_T_fvm(T, u, v, HT, nx, ny, dx, dy, b):
    rt = T.dtype.type
    ue, uw = u[1:ny+1, 1:nx+1], u[1:ny+1, 0:nx]
    vn, vs = v[1:ny+1, 1:nx+1], v[0:ny, 1:nx+1]
    _adv_T_fvm_kern(
        T[1:ny+1, 1:nx+1], T[1:ny+1, 2:nx+2], T[1:ny+1, 0:nx],
        T[2:ny+2, 1:nx+1], T[0:ny, 1:nx+1], ue, uw, vn, vs,
        rt(dy), rt(dx), rt(1.0/(dx*dy)), rt(b), rt(1.0 - b),
        HT[1:ny+1, 1:nx+1]
    )

def diffusion_scalar(phi, D, nx, ny, dx, dy, kappa):
    rt = phi.dtype.type
    _diff_kern(phi[1:ny+1, 1:nx+1], phi[1:ny+1, 2:nx+2], phi[1:ny+1, 0:nx],
               phi[2:ny+2, 1:nx+1], phi[0:ny, 1:nx+1], rt(kappa), rt(1.0/(dx*dx)), rt(1.0/(dy*dy)),
               D[1:ny+1, 1:nx+1])

def update_T(T, HT, DT, T_new, mask_p, nx, ny, dt, T_obs):
    sl = (slice(1, ny+1), slice(1, nx+1))
    rt = T.dtype.type
    _update_T_kern(T[sl], HT[sl], DT[sl], rt(dt), mask_p[sl], rt(T_obs), T_new[sl])

# kondisi batas
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

# post-processing
def compute_vorticity(u, v, omega, nx, ny, dx, dy):
    omega[1:ny+1, 1:nx+1] = ((v[1:ny+1, 2:nx+2] - v[1:ny+1, 1:nx+1]) / dx -
                             (u[2:ny+2, 1:nx+1] - u[1:ny+1, 1:nx+1]) / dy)

def compute_velocity_mag(u, v, mag, nx, ny):
    uc = 0.5 * (u[1:ny+1, 0:nx] + u[1:ny+1, 1:nx+1])
    vc = 0.5 * (v[0:ny, 1:nx+1] + v[1:ny+1, 1:nx+1])
    mag[1:ny+1, 1:nx+1] = xp.sqrt(uc*uc + vc*vc)
