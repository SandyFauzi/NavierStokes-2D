"""Kernel @cuda.jit Numba untuk adveksi FVM, jalan di atas array CuPy.

Bagian yang menggabungkan CuPy + Numba sekaligus. Hanya dipakai bila
backend.NUMBA_CUDA True; selain itu solver memakai gpu_ops (CuPy) yang identik.
"""

import warnings
from numba import cuda
from numba.core.errors import NumbaPerformanceWarning

# grid kecil -> occupancy rendah; warning ini tak relevan untuk kasus kita
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

_TPB = (16, 16)


def _grid(ni, nj):
    return ((ni + _TPB[0] - 1) // _TPB[0], (nj + _TPB[1] - 1) // _TPB[1])


@cuda.jit
def _k_adv_u_fvm(u, v, Hu, nx, ny, dx, dy, inv_vol, b, ub):
    i, j = cuda.grid(2)
    i += 1; j += 1
    if i < nx and j < ny + 1:
        ue = 0.5 * (u[j, i] + u[j, i+1])
        uw = 0.5 * (u[j, i-1] + u[j, i])
        vn = 0.5 * (v[j, i] + v[j, i+1])
        vs = 0.5 * (v[j-1, i] + v[j-1, i+1])
        pe = b*0.5*(u[j, i] + u[j, i+1]) + ub*(u[j, i] if ue >= 0 else u[j, i+1])
        pw = b*0.5*(u[j, i-1] + u[j, i]) + ub*(u[j, i-1] if uw >= 0 else u[j, i])
        pn = b*0.5*(u[j, i] + u[j+1, i]) + ub*(u[j, i] if vn >= 0 else u[j+1, i])
        ps = b*0.5*(u[j-1, i] + u[j, i]) + ub*(u[j-1, i] if vs >= 0 else u[j, i])
        Hu[j, i] = ((ue*pe - uw*pw)*dy + (vn*pn - vs*ps)*dx) * inv_vol


@cuda.jit
def _k_adv_v_fvm(u, v, Hv, nx, ny, dx, dy, inv_vol, b, ub):
    i, j = cuda.grid(2)
    i += 1; j += 1
    if i < nx + 1 and j < ny:
        ue = 0.5 * (u[j, i] + u[j+1, i])
        uw = 0.5 * (u[j, i-1] + u[j+1, i-1])
        vn = 0.5 * (v[j, i] + v[j+1, i])
        vs = 0.5 * (v[j-1, i] + v[j, i])
        pe = b*0.5*(v[j, i] + v[j, i+1]) + ub*(v[j, i] if ue >= 0 else v[j, i+1])
        pw = b*0.5*(v[j, i-1] + v[j, i]) + ub*(v[j, i-1] if uw >= 0 else v[j, i])
        pn = b*0.5*(v[j, i] + v[j+1, i]) + ub*(v[j, i] if vn >= 0 else v[j+1, i])
        ps = b*0.5*(v[j-1, i] + v[j, i]) + ub*(v[j-1, i] if vs >= 0 else v[j, i])
        Hv[j, i] = ((ue*pe - uw*pw)*dy + (vn*pn - vs*ps)*dx) * inv_vol


@cuda.jit
def _k_adv_T_fvm(T, u, v, HT, nx, ny, dx, dy, inv_vol, b, ub):
    i, j = cuda.grid(2)
    i += 1; j += 1
    if i < nx + 1 and j < ny + 1:
        ue, uw = u[j, i], u[j, i-1]
        vn, vs = v[j, i], v[j-1, i]
        Te = b*0.5*(T[j, i] + T[j, i+1]) + ub*(T[j, i] if ue >= 0 else T[j, i+1])
        Tw = b*0.5*(T[j, i-1] + T[j, i]) + ub*(T[j, i-1] if uw >= 0 else T[j, i])
        Tn = b*0.5*(T[j, i] + T[j+1, i]) + ub*(T[j, i] if vn >= 0 else T[j+1, i])
        Ts = b*0.5*(T[j-1, i] + T[j, i]) + ub*(T[j-1, i] if vs >= 0 else T[j, i])
        HT[j, i] = ((ue*Te - uw*Tw)*dy + (vn*Tn - vs*Ts)*dx) * inv_vol


def advection_u_fvm(u, v, Hu, nx, ny, dx, dy, b):
    iv = 1.0 / (dx * dy)
    _k_adv_u_fvm[_grid(nx - 1, ny), _TPB](u, v, Hu, nx, ny, dx, dy, iv, b, 1.0 - b)


def advection_v_fvm(u, v, Hv, nx, ny, dx, dy, b):
    iv = 1.0 / (dx * dy)
    _k_adv_v_fvm[_grid(nx, ny - 1), _TPB](u, v, Hv, nx, ny, dx, dy, iv, b, 1.0 - b)


def advection_T_fvm(T, u, v, HT, nx, ny, dx, dy, b):
    iv = 1.0 / (dx * dy)
    _k_adv_T_fvm[_grid(nx, ny), _TPB](T, u, v, HT, nx, ny, dx, dy, iv, b, 1.0 - b)
