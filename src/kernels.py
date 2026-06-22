"""Kernel CFD untuk CPU (Numba @njit paralel). Tanda tangan = gpu_ops."""

import math
import numpy as np
from numba import njit, prange


# --- Adveksi FDM (central) ---

@njit(parallel=True)
def advection_u_fdm(u, v, Hu, nx, ny, dx, dy, b):
    for j in prange(1, ny + 1):
        for i in range(1, nx):
            dudx = (u[j, i+1] - u[j, i-1]) / (2.0 * dx)
            v_int = 0.25 * (v[j-1, i] + v[j-1, i+1] + v[j, i] + v[j, i+1])
            dudy = (u[j+1, i] - u[j-1, i]) / (2.0 * dy)
            Hu[j, i] = u[j, i] * dudx + v_int * dudy

@njit(parallel=True)
def advection_v_fdm(u, v, Hv, nx, ny, dx, dy, b):
    for j in prange(1, ny):
        for i in range(1, nx + 1):
            u_int = 0.25 * (u[j, i-1] + u[j, i] + u[j+1, i-1] + u[j+1, i])
            dvdx = (v[j, i+1] - v[j, i-1]) / (2.0 * dx)
            dvdy = (v[j+1, i] - v[j-1, i]) / (2.0 * dy)
            Hv[j, i] = u_int * dvdx + v[j, i] * dvdy


# --- Adveksi FVM (blended: b*central + (1-b)*upwind) ---

@njit(parallel=True)
def advection_u_fvm(u, v, Hu, nx, ny, dx, dy, b):
    inv_vol = 1.0 / (dx * dy)
    ub = 1.0 - b
    for j in prange(1, ny + 1):
        for i in range(1, nx):
            ue = 0.5 * (u[j, i] + u[j, i+1])
            uw = 0.5 * (u[j, i-1] + u[j, i])
            vn = 0.5 * (v[j, i] + v[j, i+1])
            vs = 0.5 * (v[j-1, i] + v[j-1, i+1])
            pe = b*0.5*(u[j, i] + u[j, i+1]) + ub*(u[j, i] if ue >= 0 else u[j, i+1])
            pw = b*0.5*(u[j, i-1] + u[j, i]) + ub*(u[j, i-1] if uw >= 0 else u[j, i])
            pn = b*0.5*(u[j, i] + u[j+1, i]) + ub*(u[j, i] if vn >= 0 else u[j+1, i])
            ps = b*0.5*(u[j-1, i] + u[j, i]) + ub*(u[j-1, i] if vs >= 0 else u[j, i])
            Hu[j, i] = ((ue*pe - uw*pw)*dy + (vn*pn - vs*ps)*dx) * inv_vol

@njit(parallel=True)
def advection_v_fvm(u, v, Hv, nx, ny, dx, dy, b):
    inv_vol = 1.0 / (dx * dy)
    ub = 1.0 - b
    for j in prange(1, ny):
        for i in range(1, nx + 1):
            ue = 0.5 * (u[j, i] + u[j+1, i])
            uw = 0.5 * (u[j, i-1] + u[j+1, i-1])
            vn = 0.5 * (v[j, i] + v[j+1, i])
            vs = 0.5 * (v[j-1, i] + v[j, i])
            pe = b*0.5*(v[j, i] + v[j, i+1]) + ub*(v[j, i] if ue >= 0 else v[j, i+1])
            pw = b*0.5*(v[j, i-1] + v[j, i]) + ub*(v[j, i-1] if uw >= 0 else v[j, i])
            pn = b*0.5*(v[j, i] + v[j+1, i]) + ub*(v[j, i] if vn >= 0 else v[j+1, i])
            ps = b*0.5*(v[j-1, i] + v[j, i]) + ub*(v[j-1, i] if vs >= 0 else v[j, i])
            Hv[j, i] = ((ue*pe - uw*pw)*dy + (vn*pn - vs*ps)*dx) * inv_vol


# --- Difusi ---

@njit(parallel=True)
def diffusion_u(u, Du, nx, ny, dx, dy, nu):
    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)
    for j in prange(1, ny + 1):
        for i in range(1, nx):
            Du[j, i] = nu * ((u[j, i+1] - 2.0*u[j, i] + u[j, i-1]) * idx2 +
                             (u[j+1, i] - 2.0*u[j, i] + u[j-1, i]) * idy2)

@njit(parallel=True)
def diffusion_v(v, Dv, nx, ny, dx, dy, nu):
    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)
    for j in prange(1, ny):
        for i in range(1, nx + 1):
            Dv[j, i] = nu * ((v[j, i+1] - 2.0*v[j, i] + v[j, i-1]) * idx2 +
                             (v[j+1, i] - 2.0*v[j, i] + v[j-1, i]) * idy2)


# --- Kecepatan sementara + koreksi tekanan ---

@njit(parallel=True)
def tentative_u(u, Hu, Du, u_star, mask_u, nx, ny, dt):
    for j in prange(1, ny + 1):
        for i in range(1, nx):
            u_star[j, i] = 0.0 if mask_u[j, i] else u[j, i] + dt * (-Hu[j, i] + Du[j, i])

@njit(parallel=True)
def tentative_v(v, Hv, Dv, v_star, mask_v, nx, ny, dt):
    for j in prange(1, ny):
        for i in range(1, nx + 1):
            v_star[j, i] = 0.0 if mask_v[j, i] else v[j, i] + dt * (-Hv[j, i] + Dv[j, i])

@njit(parallel=True)
def correct_u(u_star, p, u_new, mask_u, nx, ny, dt_rho_dx):
    for j in prange(1, ny + 1):
        for i in range(1, nx):
            u_new[j, i] = 0.0 if mask_u[j, i] else u_star[j, i] - dt_rho_dx * (p[j, i+1] - p[j, i])

@njit(parallel=True)
def correct_v(v_star, p, v_new, mask_v, nx, ny, dt_rho_dy):
    for j in prange(1, ny):
        for i in range(1, nx + 1):
            v_new[j, i] = 0.0 if mask_v[j, i] else v_star[j, i] - dt_rho_dy * (p[j+1, i] - p[j, i])


# --- Divergensi + Poisson ---

@njit(parallel=True)
def divergence(u, v, div, nx, ny, dx, dy):
    for j in prange(1, ny + 1):
        for i in range(1, nx + 1):
            div[j, i] = (u[j, i] - u[j, i-1]) / dx + (v[j, i] - v[j-1, i]) / dy

@njit(parallel=True)
def poisson_jacobi(p, p_new, rhs, mask_p, nx, ny, dx2, dy2, coeff):
    for j in prange(1, ny + 1):
        for i in range(1, nx + 1):
            if mask_p[j, i]:
                p_new[j, i] = 0.0
            else:
                p_new[j, i] = ((p[j, i+1] + p[j, i-1]) / dx2 +
                               (p[j+1, i] + p[j-1, i]) / dy2 - rhs[j, i]) / coeff

@njit
def pressure_bc(p, nx, ny):
    for j in range(ny + 2):
        p[j, 0] = p[j, 1]
        p[j, nx+1] = p[j, nx]
    for i in range(nx + 2):
        p[0, i] = p[1, i]
        p[ny+1, i] = p[ny, i]


# --- Transport suhu ---

@njit(parallel=True)
def advection_T_fdm(T, u, v, HT, nx, ny, dx, dy, b):
    for j in prange(1, ny + 1):
        for i in range(1, nx + 1):
            uc = 0.5 * (u[j, i-1] + u[j, i])
            vc = 0.5 * (v[j-1, i] + v[j, i])
            HT[j, i] = (uc * (T[j, i+1] - T[j, i-1]) / (2.0*dx) +
                        vc * (T[j+1, i] - T[j-1, i]) / (2.0*dy))

@njit(parallel=True)
def advection_T_fvm(T, u, v, HT, nx, ny, dx, dy, b):
    inv_vol = 1.0 / (dx * dy)
    ub = 1.0 - b
    for j in prange(1, ny + 1):
        for i in range(1, nx + 1):
            ue, uw = u[j, i], u[j, i-1]
            vn, vs = v[j, i], v[j-1, i]
            Te = b*0.5*(T[j, i] + T[j, i+1]) + ub*(T[j, i] if ue >= 0 else T[j, i+1])
            Tw = b*0.5*(T[j, i-1] + T[j, i]) + ub*(T[j, i-1] if uw >= 0 else T[j, i])
            Tn = b*0.5*(T[j, i] + T[j+1, i]) + ub*(T[j, i] if vn >= 0 else T[j+1, i])
            Ts = b*0.5*(T[j-1, i] + T[j, i]) + ub*(T[j-1, i] if vs >= 0 else T[j, i])
            HT[j, i] = ((ue*Te - uw*Tw)*dy + (vn*Tn - vs*Ts)*dx) * inv_vol

@njit(parallel=True)
def diffusion_scalar(phi, D, nx, ny, dx, dy, kappa):
    idx2 = 1.0 / (dx * dx)
    idy2 = 1.0 / (dy * dy)
    for j in prange(1, ny + 1):
        for i in range(1, nx + 1):
            D[j, i] = kappa * ((phi[j, i+1] - 2.0*phi[j, i] + phi[j, i-1]) * idx2 +
                               (phi[j+1, i] - 2.0*phi[j, i] + phi[j-1, i]) * idy2)

@njit(parallel=True)
def update_T(T, HT, DT, T_new, mask_p, nx, ny, dt, T_obs):
    for j in prange(1, ny + 1):
        for i in range(1, nx + 1):
            T_new[j, i] = T_obs if mask_p[j, i] else T[j, i] + dt * (-HT[j, i] + DT[j, i])


# --- Kondisi batas ---

@njit
def bc_u(u, mask_u, nx, ny, U_inf, noslip):
    for j in range(ny + 2):
        u[j, 0] = U_inf            # inlet
        u[j, nx] = u[j, nx - 1]    # outlet (gradien nol)
    if noslip:
        for i in range(nx + 1):
            u[0, i] = -u[1, i]
            u[ny+1, i] = -u[ny, i]
    else:                          # free-slip: du/dy = 0
        for i in range(nx + 1):
            u[0, i] = u[1, i]
            u[ny+1, i] = u[ny, i]
    for j in range(ny + 2):
        for i in range(nx + 1):
            if mask_u[j, i]:
                u[j, i] = 0.0

@njit
def bc_v(v, mask_v, nx, ny):
    for i in range(nx + 2):
        v[0, i] = 0.0
        v[ny, i] = 0.0
    for j in range(ny + 1):
        v[j, 0] = -v[j, 1]
        v[j, nx+1] = v[j, nx]
    for j in range(ny + 1):
        for i in range(nx + 2):
            if mask_v[j, i]:
                v[j, i] = 0.0

@njit
def bc_T(T, mask_p, nx, ny, T_inf, T_obs):
    for j in range(ny + 2):
        T[j, 0] = T_inf
        T[j, nx+1] = T[j, nx]
    for i in range(nx + 2):
        T[0, i] = T[1, i]
        T[ny+1, i] = T[ny, i]
    for j in range(ny + 2):
        for i in range(nx + 2):
            if mask_p[j, i]:
                T[j, i] = T_obs


# --- Post-processing ---

@njit(parallel=True)
def compute_vorticity(u, v, omega, nx, ny, dx, dy):
    for j in prange(1, ny + 1):
        for i in range(1, nx + 1):
            omega[j, i] = (v[j, i+1] - v[j, i]) / dx - (u[j+1, i] - u[j, i]) / dy

@njit(parallel=True)
def compute_velocity_mag(u, v, mag, nx, ny):
    for j in prange(1, ny + 1):
        for i in range(1, nx + 1):
            uc = 0.5 * (u[j, i-1] + u[j, i])
            vc = 0.5 * (v[j-1, i] + v[j, i])
            mag[j, i] = math.sqrt(uc*uc + vc*vc)
