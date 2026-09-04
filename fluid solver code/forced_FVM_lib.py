#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Useful functions for FVM implementation. Currently can only solve a domain
where all boundaries are slip walls.

compute_tentative_velocity:
    Computes the tentative velocity field.
pressure_poisson:
    Solves the poisson pressure equation and thus the fluid pressure field.
correct_velocity:
    orrects the fluid velocity field using the solved pressure field.
set_BCs:
    Sets the fluid velocity boundary conditions.

Created on Mon Aug 31 12:00:21 2026
Author: Max Robbins
"""

import numpy as np
import numba as nb



#%%
@nb.jit(nopython=True, parallel=True, fastmath=True)
def compute_tentative_velocity(u, u_star, F, Nx, Ny, Nz, dx, dy, dz, dt, nu, rho):
    """
    Computes the tentative velocity field (i.e. without consideration of the 
    pressure field).

    Parameters
    ----------
    u : ndarray
        fluid velocity field. ndims=4, dtype=float
    u_star : ndarray
        tentative fluid velocity field. ndims=4, dtype=float
    F : ndarray
        fluid force density field. ndims=4, dtype=float
    Nx : int
        number of fluid cells in the x direction.
    Ny : int
        number of fluid cells in the y direction.
    Nz : int
        number of fluid cells in the z direction.
    dx : float
        lattice cell size (x direction).
    dy : float
        lattice cell size (y direction).
    dz : float
        lattice cell size (z direction).
    dt : float
        time step duration.
    nu : float
        fluid kinematic viscosity.
    rho : float
        fluid density.

    Returns
    -------
    None.
    """
    
    inv_dx_sq, inv_dy_sq, inv_dz_sq = 1/(dx**2), 1/(dy**2), 1/(dy**2)
    # inv_2dx, inv_2dy, inv_2dz = 1/(2*dx), 1/(2*dy), 1/(2*dz)
    inv_dx, inv_dy, inv_dz = 1/dx, 1/dy, 1/dz
    
    for i in nb.prange(1, Nx-1):
        np.int64(i)
        for j in range(1, Ny-1):
            for k in range(1, Nz-1):
                
                # Extract Current and Neighboring Velocities
                curr_ux, curr_uy, curr_uz = u[i, j, k, 0], u[i, j, k, 1], u[i, j, k, 2]
                prev_x_ux, prev_x_uy, prev_x_uz = u[i-1, j, k, 0], u[i-1, j, k, 1], u[i-1, j, k, 2]
                next_x_ux, next_x_uy, next_x_uz = u[i+1, j, k, 0], u[i+1, j, k, 1], u[i+1, j, k, 2]
                prev_y_ux, prev_y_uy, prev_y_uz = u[i, j-1, k, 0], u[i, j-1, k, 1], u[i, j-1, k, 2]
                next_y_ux, next_y_uy, next_y_uz = u[i, j+1, k, 0], u[i, j+1, k, 1], u[i, j+1, k, 2]
                prev_z_ux, prev_z_uy, prev_z_uz = u[i, j, k-1, 0], u[i, j, k-1, 1], u[i, j, k-1, 2]
                next_z_ux, next_z_uy, next_z_uz = u[i, j, k+1, 0], u[i, j, k+1, 1], u[i, j, k+1, 2]
                
                # Diffusion Terms
                d2u = (next_x_ux - 2*curr_ux + prev_x_ux)*inv_dx_sq + (next_y_ux - 2*curr_ux + prev_y_ux)*inv_dy_sq + (next_z_ux - 2*curr_ux + prev_z_ux)*inv_dz_sq
                d2v = (next_x_uy - 2*curr_uy + prev_x_uy)*inv_dx_sq + (next_y_uy - 2*curr_uy + prev_y_uy)*inv_dy_sq + (next_z_uy - 2*curr_uy + prev_z_uy)*inv_dz_sq
                d2w = (next_x_uz - 2*curr_uz + prev_x_uz)*inv_dx_sq + (next_y_uz - 2*curr_uz + prev_y_uz)*inv_dy_sq + (next_z_uz - 2*curr_uz + prev_z_uz)*inv_dz_sq
                
                # Advection Terms (using central differencing)
                # adv_u = curr_ux*(next_x_ux - prev_x_ux)*inv_2dx + curr_uy*(next_y_ux - prev_y_ux)*inv_2dy + curr_uz*(next_z_ux - prev_z_ux)*inv_2dz
                # adv_v = curr_ux*(next_x_uy - prev_x_uy)*inv_2dx + curr_uy*(next_y_uy - prev_y_uy)*inv_2dy + curr_uz*(next_z_uy - prev_z_uy)*inv_2dz
                # adv_w = curr_ux*(next_x_uz - prev_x_uz)*inv_2dx + curr_uy*(next_y_uz - prev_y_uz)*inv_2dy + curr_uz*(next_z_uz - prev_z_uz)*inv_2dz
                
                # Advection Terms (using forward differencing)
                if curr_ux >= 0: adv_u = curr_ux*(next_x_ux - curr_ux)*inv_dx + curr_uy*(next_y_ux - curr_ux)*inv_dy + curr_uz*(next_z_ux - curr_ux)*inv_dz
                else: adv_u = curr_ux*(curr_ux - prev_x_ux)*inv_dx + curr_uy*(curr_ux - prev_y_ux)*inv_dy + curr_uz*(curr_ux - prev_z_ux)*inv_dz
                
                if curr_uy >= 0: adv_v = curr_ux*(next_x_uy - curr_uy)*inv_dx + curr_uy*(next_y_uy - curr_uy)*inv_dy + curr_uz*(next_z_uy - curr_uy)*inv_dz
                else: adv_v = curr_ux*(curr_uy - prev_x_uy)*inv_dx + curr_uy*(curr_uy - prev_y_uy)*inv_dy + curr_uz*(curr_uy - prev_z_uy)*inv_dz
                
                if curr_uz >= 0: adv_w = curr_ux*(next_x_uz - curr_uz)*inv_dx + curr_uy*(next_y_uz - curr_uz)*inv_dy + curr_uz*(next_z_uz - curr_uz)*inv_dz
                else: adv_w = curr_ux*(curr_uz - prev_x_uz)*inv_dx + curr_uy*(curr_uz - prev_y_uz)*inv_dy + curr_uz*(curr_uz - prev_z_uz)*inv_dz
                
                # Extract force density components for this cell
                fx = F[i, j, k, 0]
                fy = F[i, j, k, 1]
                fz = F[i, j, k, 2]
                
                # Advance intermediate velocity 
                u_star[i, j, k, 0] = curr_ux + dt*(nu*d2u - adv_u + fx/rho)
                u_star[i, j, k, 1] = curr_uy + dt*(nu*d2v - adv_v + fy/rho)
                u_star[i, j, k, 2] = curr_uz + dt*(nu*d2w - adv_w + fz/rho)


@nb.jit(nopython=True, parallel=True, fastmath=True)
def pressure_poisson(p, u_star, u_star_div, Nx, Ny, Nz, dx, dy, dz, dt, n_lattice_internal, rho, Nit, omega=1.99, check_every=10, conv_tol=1e-3, abs_tol=1e-16):
    """
    Solves the poisson pressure equation and thus the fluid pressure field.
    Assumes all domain boundaries are slip walls (closed domain).

    Parameters
    ----------
    p : ndarray
        fluid pressure field. ndims=3, dtype=float
    u_star : ndarray
        tentative fluid velocity field. ndims=4, dtype=float
    u_star_div : ndarray
        divergence of the tentative velocity field. ndims=3, dtype=float
    Nx : int
        number of fluid cells in the x direction.
    Ny : int
        number of fluid cells in the y direction.
    Nz : int
        number of fluid cells in the z direction.
    dx : float
        lattice cell size (x direction).
    dy : float
        lattice cell size (y direction).
    dz : float
        lattice cell size (z direction).
    dt : float
        time step duration.
    n_lattice_internal : int
        number of fluid lattice points, excluding boundary nodes.
    rho : float
        fluid density.
    Nit : int
        maximum number of SOR iterations for the poisson pressure equation.
    omega : float, optional
        SOR over-relaxation factor. The default is 1.99.
    check_every : int, optional
        check for convergence every this many iterations. The default is 10.
    conv_tol : float, optional
        the system is considered to be converged if the relative residual falls
        below this value. The default is 1e-3.
    abs_tol : float, optional
        the system is considered to be converged if the absolute residual falls
        below this value. The default is 1e-16.

    Returns
    -------
    None.
    """
    
    # Calculate Constants
    omega_prime = 1.0-omega
    inv_2dx, inv_2dy, inv_2dz = 1/(2*dx), 1/(2*dy), 1/(2*dz)
    dy_sq_dz_sq, dx_sq_dz_sq, dx_sq_dy_sq, dx_sq_dy_sq_dz_sq = (dy**2)*(dz**2), (dx**2)*(dz**2), (dx**2)*(dy**2), (dx**2)*(dy**2)*(dz**2)
    denom = 2*(dy_sq_dz_sq + dx_sq_dz_sq + dx_sq_dy_sq)
    inv_denom = omega/denom
    dpdx_coeff, dpdy_coeff, dpdz_coeff = dy_sq_dz_sq*inv_denom, dx_sq_dz_sq*inv_denom, dx_sq_dy_sq*inv_denom
    div_coeff_num = (rho/dt)*dx_sq_dy_sq_dz_sq
    div_coeff = div_coeff_num*inv_denom
    
    # Divergence of Tentative Velocity
    b_l1_norm = 0.0
    for i in nb.prange(1, Nx-1):
        np.int64(i)
        for j in range(1, Ny-1):
            for k in range(1, Nz-1):
                u_star_div[i, j, k] = (u_star[i+1, j, k, 0] - u_star[i-1, j, k, 0])*inv_2dx + (u_star[i, j+1, k, 1] - u_star[i, j-1, k, 1])*inv_2dy + (u_star[i, j, k+1, 2] - u_star[i, j, k-1, 2])*inv_2dz
                b_l1_norm += abs(u_star_div[i, j, k]*div_coeff_num)
    if b_l1_norm == 0.0:
        b_l1_norm = 1.0
    
    # Solve the Pressure Poisson Equation using SOR Red-Black Checkerboarding
    for p_iter in range(Nit):
        # Red Pass (i+j+k is even)
        for i in nb.prange(1, Nx-1):
            np.int64(i)
            for j in range(1, Ny-1):
                # If i+j is odd, first even k is 1; if even, first even k is 2
                k_start = 1 if ((i+j)%2 != 0) else 2
                for k in range(k_start, Nz-1, 2):
                    p[i, j, k] = omega_prime*p[i, j, k] + (p[i+1, j, k] + p[i-1, j, k])*dpdx_coeff + (p[i, j+1, k] + p[i, j-1, k])*dpdy_coeff + (p[i, j, k+1] + p[i, j, k-1])*dpdz_coeff - u_star_div[i, j, k]*div_coeff
        # Black Pass (i+j+k is odd)
        for i in nb.prange(1, Nx-1):
            np.int64(i)
            for j in range(1, Ny-1):
                # If i+j is even, first odd k is 1; if odd, first odd k is 2
                k_start = 1 if ((i+j)%2 == 0) else 2
                for k in range(k_start, Nz-1, 2):
                    p[i, j, k] = omega_prime*p[i, j, k] + (p[i+1, j, k] + p[i-1, j, k])*dpdx_coeff + (p[i, j+1, k] + p[i, j-1, k])*dpdy_coeff + (p[i, j, k+1] + p[i, j, k-1])*dpdz_coeff - u_star_div[i, j, k]*div_coeff
        
        # Pressure Boundary Conditions - slip walls (zero gradient)
        p[0, :, :] = p[1, :, :]
        p[-1, :, :] = p[-2, :, :]
        p[:, 0, :] = p[:, 1, :]
        p[:, -1, :] = p[:, -2, :]
        p[:, :, 0] = p[:, :, 1]
        p[:, :, -1] = p[:, :, -2]
        
        # Calculate Residual
        if p_iter%check_every == 0:
            curr_residual = 0.0
            for i in nb.prange(1, Nx-1):
                np.int64(i)
                for j in range(1, Ny-1):
                    for k in range(1, Nz-1):
                        curr_residual += abs(((p[i+1, j, k] + p[i-1, j, k])*dy_sq_dz_sq + (p[i, j+1, k] + p[i, j-1, k])*dx_sq_dz_sq + (p[i, j, k+1] + p[i, j, k-1])*dx_sq_dy_sq - p[i, j, k]*denom) - u_star_div[i, j, k]*div_coeff_num)
            
            rel_residual = curr_residual/b_l1_norm
            if (rel_residual <= conv_tol) or (curr_residual <= abs_tol):
                break


@nb.jit(nopython=True, parallel=True, fastmath=True)
def correct_velocity(u, u_star, p, Nx, Ny, Nz, dx, dy, dz, dt, rho):
    """
    Corrects the fluid velocity field using the solved pressure field.

    Parameters
    ----------
    u : ndarray
        fluid velocity field. ndims=4, dtype=float
    u_star : ndarray
        tentative fluid velocity field. ndims=4, dtype=float
    p : ndarray
        fluid pressure field. ndims=3, dtype=float
    Nx : int
        number of fluid cells in the x direction.
    Ny : int
        number of fluid cells in the y direction.
    Nz : int
        number of fluid cells in the z direction.
    dx : float
        lattice cell size (x direction).
    dy : float
        lattice cell size (y direction).
    dz : float
        lattice cell size (z direction).
    dt : float
        time step duration.
    rho : float
        fluid density.

    Returns
    -------
    None.
    """
    
    for i in nb.prange(1, Nx-1):
        np.int64(i)
        for j in range(1, Ny-1):
            for k in range(1, Nz-1):
                u[i, j, k, 0] = u_star[i, j, k, 0] - (dt/rho)*(p[i+1, j, k] - p[i-1, j, k])/(2*dx)
                u[i, j, k, 1] = u_star[i, j, k, 1] - (dt/rho)*(p[i, j+1, k] - p[i, j-1, k])/(2*dy)
                u[i, j, k, 2] = u_star[i, j, k, 2] - (dt/rho)*(p[i, j, k+1] - p[i, j, k-1])/(2*dz)


def set_BCs(u):
    """
    Sets the fluid velocity boundary conditions. Assumes all domain boundaries 
    are slip walls (closed domain).

    Parameters
    ----------
    u : ndarray
        fluid velocity field. ndims=4, dtype=float

    Returns
    -------
    None.
    """
    
    # Z Boundary 
    ## X Velocity (zero velocity)
    u[0, :, :, 0] = 0.0
    u[-1, :, :, 0] = 0.0
    ## Y Velocity (zero gradient)
    u[0, :, :, 1] = u[1, :, :, 1]
    u[-1, :, :, 1] = u[-2, :, :, 1]
    ## Z Velocity (zero gradient)
    u[0, :, :, 2] = u[1, :, :, 2]
    u[-1, :, :, 2] = u[-2, :, :, 2]
    
    # Y Boundary 
    ## X Velocity (zero gradient)
    u[:, 0, :, 0] = u[:, 1, :, 0]
    u[:, -1, :, 0] = u[:, -2, :, 0]
    ## Y Velocity (zero velocity)
    u[:, 0, :, 1] = 0.0
    u[:, -1, :, 1] = 0.0
    ## Z Velocity (zero gradient)
    u[:, 0, :, 2] = u[:, 1, :, 2]
    u[:, -1, :, 2] = u[:, -2, :, 2]
    
    # Z Boundary 
    ## X Velocity (zero gradient)
    u[:, :, 0, 0] = u[:, :, 1, 0]
    u[:, :, -1, 0] = u[:, :, -2, 0]
    ## Y Velocity (zero gradient)
    u[:, :, 0, 1] = u[:, :, 1, 1]
    u[:, :, -1, 1] = u[:, :, -2, 1]
    ## Z Velocity (zero velocity)
    u[:, :, 0, 2] = 0.0
    u[:, :, -1, 2] = 0.0






