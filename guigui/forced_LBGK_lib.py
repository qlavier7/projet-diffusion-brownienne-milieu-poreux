#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Useful functions for LBM implementation. Currently employs the quasi-
compressible LB equation with BGK collisions and Guo forcing. Options for 1D
flow or a completely closed domain (both with slip walls).

get_LBM_consts: 
    Calculates useful constants and velocity/weighting arrays for the D3Q19 LBM.
initialise_pops:
    Initialises particle populations.
update_LBM_pops_1D_flow:
    Updates populations for one time step for a 1D flow domain with separate
    collision/streaming steps.
update_LBM_pops_closed:
    Updates populations for one time step for a closed domain with separate
    collision/streaming steps.

Created on Wed Aug 26 13:07:29 2026
Author: Max Robbins
"""

import numpy as np
import numba as nb



#%% Useful LBM Constants
def get_LBM_consts(nu):
    """
    Returns a dictionary of useful constants and velocity/weighting arrays for 
    the D3Q19 LBM. It is assumed that the regular lattice spacing (dx) and the
    time step (dt) are in standard lattice units and simply equal to 1.

    Parameters
    ----------
    nu : float
        fluid kinematic viscosity.

    Returns
    -------
    LBM_consts : dict
        'dx': regular lattice spacing
        'dt': time step
        'cs2': squared sonic velocity
        'inv_cs2': inverse squared sonic velocity
        'inv_cs4': inverse sonic velocity fourth power
        'inv_2cs2': half inverse squared sonic velocity
        'inv_2cs4': half inverse sonic velocity fourth power
        'tau': BGK relaxation factor
        'omega': inverse relaxation factor
        'omega_prime': "conjugate" inverse relaxation factor (1 - omega)
        'omega_S_coeff': "conjugate" half inverse relaxation factor
        'N_vels': number of discrete velocities (i.e. "lattice vectors")
        'w': D3Q19 discrete velocity weighting
        'c': D3Q19 discrete velocity set
        'inv_cx_indx': indexing array for specular reflection in the x direction
        'inv_cy_indx': indexing array for specular reflection in the y direction
        'inv_cz_indx': indexing array for specular reflection in the z direction
        'inv_c_indx': indexing array for inverted velocities
    """
    
    dx = 1 # regular lattice spacing
    dt = 1 # time step
    
    cs2 = 1/3 # squared speed of sound
    inv_cs2 = 1/cs2
    inv_cs4 = 1/(cs2**2)
    inv_2cs2 = 1/(2*cs2)
    inv_2cs4 = 1/(2*cs2**2)
    
    tau = nu/cs2 + 0.5 # relaxation parameter
    omega = 1/tau
    omega_prime = 1 - omega
    omega_S_coeff = (1 - 0.5*omega)
    
    # D3Q19 Velocity Set
    N_vels = 19
    w = np.array([1/3, 
                  1/18, 1/18, 1/18, 1/18, 1/18, 1/18, 
                  1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36, 1/36], 
                 dtype=np.float64) # weightings
    c = np.array([[ 0, 0, 0], # 0
                  [ 1, 0, 0], # 1
                  [-1, 0, 0], # 2
                  [ 0, 1, 0], # 3
                  [ 0,-1, 0], # 4
                  [ 0, 0, 1], # 5
                  [ 0, 0,-1], # 6
                  [ 1, 1, 0], # 7
                  [-1,-1, 0], # 8
                  [ 1, 0, 1], # 9
                  [-1, 0,-1], # 10
                  [ 0, 1, 1], # 11
                  [ 0,-1,-1], # 12
                  [ 1,-1, 0], # 13
                  [-1, 1, 0], # 14
                  [ 1, 0,-1], # 15
                  [-1, 0, 1], # 16
                  [ 0, 1,-1], # 17
                  [ 0,-1, 1]  # 18
                  ], dtype=np.int64) # velocities
    
    ## Inverted and Specularly-Reflected Velocity Arrays
    # Mapping Indices:     [0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    inv_cx_indx = np.array([0,  2,  1,  3,  4,  5,  6, 14, 13, 16, 15, 11, 12,  8,  7, 10,  9, 17, 18], 
                           dtype=np.int64) # velocity indices mirrored across x axis
    inv_cy_indx = np.array([0,  1,  2,  4,  3,  5,  6, 13, 14,  9, 10, 18, 17,  7,  8, 15, 16, 12, 11], 
                           dtype=np.int64) # velocity indices mirrored across y axis
    inv_cz_indx = np.array([0,  1,  2,  3,  4,  6,  5,  7,  8, 15, 16, 17, 18, 13, 14,  9, 10, 11, 12], 
                           dtype=np.int64) # velocity indices mirrored across z axis
    inv_c_indx  = np.array([0,  2,  1,  4,  3,  6,  5,  8,  7, 10,  9, 12, 11, 14, 13, 16, 15, 18, 17], 
                           dtype=np.int64) # inverted velocity indices
    
    LBM_consts = {'dx': dx, 
                  'dt': dt, 
                  'cs2': cs2, 
                  'inv_cs2': inv_cs2, 
                  'inv_cs4': inv_cs4, 
                  'inv_2cs2': inv_2cs2, 
                  'inv_2cs4': inv_2cs4, 
                  'tau': tau, 
                  'omega': omega, 
                  'omega_prime': omega_prime, 
                  'omega_S_coeff': omega_S_coeff, 
                  'N_vels': N_vels, 
                  'w': w, 
                  'c': c, 
                  'inv_cx_indx': inv_cx_indx, 
                  'inv_cy_indx': inv_cy_indx, 
                  'inv_cz_indx': inv_cz_indx, 
                  'inv_c_indx': inv_c_indx}
    
    return LBM_consts



#%% BGK Collision and Guo Forcing Expressions
@nb.jit(nopython=True, inline='always', fastmath=True)
def calc_f_eq_comp_BGK(cu, u2, rho, w, inv_cs2, inv_2cs2, inv_2cs4):
    """
    Calculates the current equilibrium population for BGK collisions with the
    quasi-compressible LB equation.

    Parameters
    ----------
    cu : float
        discrete velocity and fluid velocity dot product.
    u2 : float
        fluid velocity squared magnitude.
    rho : float
        fluid density.
    w : float
        discrete velocity weighting.
    inv_cs2 : float
        inverse squared sonic velocity.
    inv_2cs2 : float
        half inverse squared sonic velocity.
    inv_2cs4 : float
        half inverse sonic velocity fourth power.

    Returns
    -------
    f_eq : float
        equilibrium population.
    """
    f_eq = rho*w*(1.0 + cu*inv_cs2 + cu*cu*inv_2cs4 - u2*inv_2cs2)
    return f_eq


@nb.jit(nopython=True, inline='always', fastmath=True)
def calc_Fi_comp_BGK(Fx, Fy, Fz, ux, uy, uz, cu, cx, cy, cz, w, inv_cs2, inv_cs4):
    """
    Calculates the BGK Guo forcing term for the quasi-compressible LB equation.

    Parameters
    ----------
    Fx : float
        force density x component.
    Fy : float
        force density y component.
    Fz : float
        force density z component.
    ux : float
        fluid velocity x component.
    uy : float
        fluid velocity y component.
    uz : float
        fluid velocity z component.
    cu : float
        discrete velocity and fluid velocity dot product.
    cx : float
        discrete velocity x component.
    cy : float
        discrete velocity y component.
    cz : float
        discrete velocity z component.
    w : float
        discrete velocity weighting.
    inv_cs2 : float
        inverse squared sonic velocity.
    inv_cs4 : float
        inverse sonic velocity fourth power.

    Returns
    -------
    F_i : float
        BGK Guo forcing term.
    """
    Fc = Fx*cx + Fy*cy + Fz*cz
    Fu = Fx*ux + Fy*uy + Fz*uz
    F_i = w*(Fc*(inv_cs2 + cu*inv_cs4) - Fu*inv_cs2)
    return F_i



#%% BGK Forced Collision
@nb.jit(nopython=True, parallel=True, fastmath=True)
def collide_forced(pops_pre, pops_post, F, rho, u, u_mag2, Nx, Ny, Nz, 
                   inv_cs2, inv_2cs2, inv_cs4, inv_2cs4, omega_S_coeff, N_vels, w, c, nu, kB_T):
    """
    Calculates and saves the velocity and density fields from the current 
    populations and performs the BGK collision globally with Guo forcing.

    Parameters
    ----------
    pops_pre : ndarray
        pre-collision populations. ndims=4, dtype=float
    pops_post : ndarray
        post-collision populations. ndims=4, dtype=float
    F : ndarray
        fluid force density field. ndims=4, dtype=float
    rho : ndarray
        fluid density field. ndims=3, dtype=float
    u : ndarray
        fluid velocity field. ndims=4, dtype=float
    u_mag2 : ndarray
        fluid velocity squared magnitude field. ndims=3, dtype=float
    Nx : int
        number of fluid cells in the x direction.
    Ny : int
        number of fluid cells in the y direction.
    Nz : int
        number of fluid cells in the z direction.
    inv_cs2 : float
        inverse squared sonic velocity.
    inv_2cs2 : float
        half inverse squared sonic velocity.
    inv_cs4 : float
        inverse sonic velocity fourth power.
    inv_2cs4 : float
        half inverse sonic velocity fourth power.
    omega : float
        inverse relaxation factor.
    omega_prime : float
        "conjugate" inverse relaxation factor (1 - omega).
    omega_S_coeff : float
        "conjugate" half inverse relaxation factor.
    N_vels : int
        number of discrete velocities (i.e. "lattice vectors").
    w : ndarray
        discrete velocity weightings. ndims=1, dtype=float
    c : ndarray
        discrete velocity set. ndtims=2, dtype=int

    Returns
    -------
    None.
    """
    
    """
    Calculates and saves the velocity and density fields from the current 
    populations and performs the BGK collision globally with Guo forcing
    and thermal fluctuations.
    """
    
    # -------------------------------------------------------------------------
    # PRÉCALCULS GLOBAUX : Base des modes, paramètres de relaxation et bruit
    # -------------------------------------------------------------------------
    
    # Vecteurs de base e_ki pour le modèle D3Q19 (Table II)
    e_k = np.zeros((19, 19), dtype=np.float64)
    
    for q in range(19):
        cx, cy, cz = c[q, 0], c[q, 1], c[q, 2]
        c2 = cx*cx + cy*cy + cz*cz
        
        e_k[0, q] = 1.0
        e_k[1, q] = cx
        e_k[2, q] = cy
        e_k[3, q] = cz
        e_k[4, q] = c2 - 1.0
        e_k[5, q] = 3.0*cx*cx - c2
        e_k[6, q] = cy*cy - cz*cz
        e_k[7, q] = cx*cy
        e_k[8, q] = cy*cz
        e_k[9, q] = cz*cx
        e_k[10, q] = (3.0*c2 - 5.0)*cx
        e_k[11, q] = (3.0*c2 - 5.0)*cy
        e_k[12, q] = (3.0*c2 - 5.0)*cz
        e_k[13, q] = (cy*cy - cz*cz)*cx
        e_k[14, q] = (cz*cz - cx*cx)*cy
        e_k[15, q] = (cx*cx - cy*cy)*cz
        e_k[16, q] = 3.0*c2*c2 - 6.0*c2 + 1.0
        e_k[17, q] = (2.0*c2 - 3.0)*(3.0*cx*cx - c2)
        e_k[18, q] = (2.0*c2 - 3.0)*(cy*cy - cz*cz)

    w_k = np.array([1.0, 1.0/3.0, 1.0/3.0, 1.0/3.0, 2.0/3.0, 4.0/3.0, 4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 
                    2.0/3.0, 2.0/3.0, 2.0/3.0, 2.0/9.0, 2.0/9.0, 2.0/9.0, 2.0, 4.0/3.0, 4.0/9.0], dtype=np.float64)

    # Matrice de passage orthonormée e_hat
    e_hat = np.zeros((19, 19), dtype=np.float64)
    for mk in range(19):
        for q in range(19):
            e_hat[mk, q] = np.sqrt(w[q] / w_k[mk]) * e_k[mk, q]

    # Paramètres de relaxation (gamma)
    gamma = np.zeros(19, dtype=np.float64)
    eta_hat = nu*inv_cs2
    gamma[0:4] = 1.0                                # Modes conservés (masse et impulsion)
    gamma[4] = (3.0 * eta_hat - 1.0) / (3.0 * eta_hat + 1.0)  # Mode de contrainte volumique (viscosité = nu)
    gamma_s = (2.0 * eta_hat - 1.0) / (2.0 * eta_hat + 1.0)   # Modes de contrainte de cisaillement
    gamma[5:10] = gamma_s
    gamma[10:19] = 0.0                              # Modes cinétiques purement stochastiques

    # Amplitude du bruit garantissant le respect du bilan détaillé
    phi = np.sqrt(1.0 - gamma**2)
    
    # Contrôle de l'intensité des fluctuations
    mu = kB_T * inv_cs2  # Eq. 90: \mu = (kB_T * h^2) / (c_s^2 * b^{d+2}), ici ramené en unités réseaux (h=1, b=1)

    # -------------------------------------------------------------------------
    # DYNAMIQUE SUR GRILLE SPATIALE
    # -------------------------------------------------------------------------
    
    for i in nb.prange(Nx):
        i = np.int64(i)
        for j in range(Ny):
            for k in range(Nz):
                
                # Calcul de rho et u
                local_rho = 0.0
                local_rhou_x = 0.5*F[i, j, k, 0]
                local_rhou_y = 0.5*F[i, j, k, 1]
                local_rhou_z = 0.5*F[i, j, k, 2]
                
                for q in range(N_vels):
                    pop = pops_pre[i, j, k, q]
                    local_rho += pop
                    local_rhou_x += pop*c[q, 0]
                    local_rhou_y += pop*c[q, 1]
                    local_rhou_z += pop*c[q, 2]
                
                if local_rho < 1.0e-14: local_rho = 1.0e-14
                local_rho_inv = 1.0/local_rho
                local_ux = local_rhou_x*local_rho_inv
                local_uy = local_rhou_y*local_rho_inv
                local_uz = local_rhou_z*local_rho_inv
                local_u2 = local_ux*local_ux + local_uy*local_uy + local_uz*local_uz
                
                # Sauvegarde des données macroscopiques
                rho[i, j, k] = local_rho
                u[i, j, k, 0] = local_ux
                u[i, j, k, 1] = local_uy
                u[i, j, k, 2] = local_uz
                u_mag2[i, j, k] = local_u2
                
                # Tableaux temporaires pour les variables locales
                local_f_eq = np.zeros(19, dtype=np.float64)
                local_F_i = np.zeros(19, dtype=np.float64)
                local_x = np.zeros(19, dtype=np.float64)
                
                # 1 & 2. Calcul des variables hors-équilibre normalisées (x_i)
                for q in range(19):
                    cx = c[q, 0]
                    cy = c[q, 1]
                    cz = c[q, 2]
                    cu = cx*local_ux + cy*local_uy + cz*local_uz
                    
                    f_eq = calc_f_eq_comp_BGK(cu, local_u2, local_rho, w[q], inv_cs2, inv_2cs2, inv_2cs4)
                    F_i = calc_Fi_comp_BGK(F[i, j, k, 0], F[i, j, k, 1], F[i, j, k, 2], local_ux, local_uy, local_uz, 
                                           cu, cx, cy, cz, w[q], inv_cs2, inv_cs4)
                    
                    local_f_eq[q] = f_eq
                    local_F_i[q] = F_i
                    
                    # Normalisation adimensionnée de la fraction hors-équilibre
                    local_x[q] = (pops_pre[i, j, k, q] - f_eq) / np.sqrt(mu * local_rho * w[q])
                
                # 3 & 4. Projection sur l'espace des modes et Mise à jour stochastique
                local_m_star = np.zeros(19, dtype=np.float64)
                for mk in range(19):
                    m_k = 0.0
                    for q in range(19):
                        m_k += e_hat[mk, q] * local_x[q]
                    
                    r_k = 0.0
                    # Injection du bruit indépendant de distribution normale réduite N(0, 1) pour les modes non conservés
                    if mk > 3:
                        r_k = np.random.randn()
                    
                    # Relaxation de la partie déterministe et ajout de l'intensité stochastique
                    local_m_star[mk] = gamma[mk] * m_k + phi[mk] * r_k
                
                # 5 & 6. Rétro-projection linéaire et réintégration des forçages
                for q in range(19):
                    x_star = 0.0
                    for mk in range(19):
                        x_star += e_hat[mk, q] * local_m_star[mk]
                    
                    # Reconstruction de l'équation de Boltzmann sur réseaux (post-collision) + forçage scalaire de Guo
                    pops_post[i, j, k, q] = local_f_eq[q] + np.sqrt(mu * local_rho * w[q]) * x_star + local_F_i[q] * omega_S_coeff
    

@nb.jit(nopython=True, parallel=True, fastmath=True)
def collide_forced2(pops_pre, pops_post, F, rho, u, u_mag2, Nx, Ny, Nz, 
                   inv_cs2, inv_2cs2, inv_cs4, inv_2cs4, omega, omega_prime, omega_S_coeff, N_vels, w, c):
    """
    Calculates and saves the velocity and density fields from the current 
    populations and performs the BGK collision globally with Guo forcing.

    Parameters
    ----------
    pops_pre : ndarray
        pre-collision populations. ndims=4, dtype=float
    pops_post : ndarray
        post-collision populations. ndims=4, dtype=float
    F : ndarray
        fluid force density field. ndims=4, dtype=float
    rho : ndarray
        fluid density field. ndims=3, dtype=float
    u : ndarray
        fluid velocity field. ndims=4, dtype=float
    u_mag2 : ndarray
        fluid velocity squared magnitude field. ndims=3, dtype=float
    Nx : int
        number of fluid cells in the x direction.
    Ny : int
        number of fluid cells in the y direction.
    Nz : int
        number of fluid cells in the z direction.
    inv_cs2 : float
        inverse squared sonic velocity.
    inv_2cs2 : float
        half inverse squared sonic velocity.
    inv_cs4 : float
        inverse sonic velocity fourth power.
    inv_2cs4 : float
        half inverse sonic velocity fourth power.
    omega : float
        inverse relaxation factor.
    omega_prime : float
        "conjugate" inverse relaxation factor (1 - omega).
    omega_S_coeff : float
        "conjugate" half inverse relaxation factor.
    N_vels : int
        number of discrete velocities (i.e. "lattice vectors").
    w : ndarray
        discrete velocity weightings. ndims=1, dtype=float
    c : ndarray
        discrete velocity set. ndtims=2, dtype=int

    Returns
    -------
    None.
    """
    
    for i in nb.prange(Nx):
        i = np.int64(i)
        for j in range(Ny):
            for k in range(Nz):
                
                # Calculate rho and u
                local_rho = 0.0
                local_rhou_x = 0.5*F[i, j, k, 0]
                local_rhou_y = 0.5*F[i, j, k, 1]
                local_rhou_z = 0.5*F[i, j, k, 2]
                
                for q in range(N_vels):
                    pop = pops_pre[i, j, k, q]
                    local_rho += pop
                    local_rhou_x += pop*c[q, 0]
                    local_rhou_y += pop*c[q, 1]
                    local_rhou_z += pop*c[q, 2]
                
                if local_rho < 1.0e-14: local_rho = 1.0e-14
                local_rho_inv = 1.0/local_rho
                local_ux = local_rhou_x*local_rho_inv
                local_uy = local_rhou_y*local_rho_inv
                local_uz = local_rhou_z*local_rho_inv
                local_u2 = local_ux*local_ux + local_uy*local_uy + local_uz*local_uz
                
                # Save Data
                rho[i, j, k] = local_rho
                u[i, j, k, 0] = local_ux
                u[i, j, k, 1] = local_uy
                u[i, j, k, 2] = local_uz
                u_mag2[i, j, k] = local_u2
                
                # BGK Collision Loop
                for q in range(N_vels):
                    cx = c[q, 0]
                    cy = c[q, 1]
                    cz = c[q, 2]
                    cu = cx*local_ux + cy*local_uy + cz*local_uz
                    
                    f_eq = calc_f_eq_comp_BGK(cu, local_u2, local_rho, w[q], inv_cs2, inv_2cs2, inv_2cs4)
                    F_i = calc_Fi_comp_BGK(F[i, j, k, 0], F[i, j, k, 1], F[i, j, k, 2], local_ux, local_uy, local_uz, 
                                           cu, cx, cy, cz, w[q], inv_cs2, inv_cs4)
                    
                    ## Perform forced BGK collisions
                    
                    pops_post[i, j, k, q] = pops_pre[i, j, k, q]*omega_prime + f_eq*omega + F_i*omega_S_coeff # plus optional fluctuating stress term
                    

#%% Streaming
@nb.jit(nopython=True, parallel=True, fastmath=True)
def stream_1D_flow(pops_pre, pops_post, rho, Nx, Ny, Nz, rho_0, Ux_t, Uy_0, Uz_0, 
                   N_vels, c, inv_cy_indx, inv_cz_indx, alpha=0.99):
    """
    Streams particle populations, assuming a flow inlet at the x = 0 and outlet 
    at x = Nx with inlet velocity u = [Ux_t, Uy_0, Uz_0], where Uy_0 = Uz_0 = 0.
    Slip condition on all other boundaries.
    Constant 1D velocity inlet with x velocity = Ux_t.
    Zero-gradient density (pressure) outlet, with the density pinned to the
    desired fluid density rho_0. alpha controls the strength at which the fluid
    density is enforced to remain at rho_0.

    Parameters
    ----------
    pops_pre : ndarray
        pre-collision populations. ndims=4, dtype=float
    pops_post : ndarray
        post-collision populations. ndims=4, dtype=float
    rho : ndarray
        fluid density field. ndims=3, dtype=float
    Nx : int
        number of fluid cells in the x direction.
    Ny : int
        number of fluid cells in the y direction.
    Nz : int
        number of fluid cells in the z direction.
    rho_0 : float
        reference fluid density.
    Ux_t : float
        current fluid inlet x velocity.
    Uy_0 : float
       fluid inlet y velocity (0).
    Uz_0 : float
        fluid inlet z velocity (0).
    N_vels : int
        number of discrete velocities (i.e. "lattice vectors").
    c : ndarray
        discrete velocity set. ndtims=2, dtype=int
    inv_cy_indx : ndarray
        indexing array for specular reflection in the y direction. ndims=1, dtype=float
    inv_cz_indx : ndarray
        indexing array for specular reflection in the z direction. ndims=1, dtype=float
    alpha : float, optional
        reference density anchoring weighting factor. The default is 0.99.

    Returns
    -------
    None.
    """
    
    # Bulk Streaming - Internal Nodes
    for i in nb.prange(1, Nx-1):
        i = np.int64(i)
        for j in range(1, Ny-1):
            for k in range(1, Nz-1):
                for q in range(N_vels):
                    pull_i = i - c[q, 0]
                    pull_j = j - c[q, 1]
                    pull_k = k - c[q, 2]
                    pops_pre[i, j, k, q] = pops_post[pull_i, pull_j, pull_k, q]
    
    # X Bounds
    for i in [0, Nx-1]:
        i = np.int64(i)
        for j in nb.prange(Ny):
            j = np.int64(j)
            for k in range(Nz):
                for q in range(N_vels):
                    pull_i = i - c[q, 0]
                    pull_j = j - c[q, 1]
                    pull_k = k - c[q, 2]
                    q_write = q
                    ## Slip condition on y-walls (top/bottom)
                    if pull_j < 0 or pull_j >= Ny:
                        pull_j = j
                        q_write = inv_cy_indx[q]
                    ## Slip condition on z-walls (front/back)
                    if pull_k < 0 or pull_k >= Nz:
                        pull_k = k
                        q_write = inv_cz_indx[q]
                    ## Skip particles pulled from outside the x domain
                    if pull_i < 0 or pull_i >= Nx:
                        continue
                    pops_pre[i, j, k, q] = pops_post[pull_i, pull_j, pull_k, q_write]
    
    # Y Bounds
    for j in [0, Ny-1]:
        j = np.int64(j)
        for i in nb.prange(1, Nx-1):
            i = np.int64(i)
            for k in range(Nz):
                for q in range(N_vels):
                    pull_i = i - c[q, 0]
                    pull_j = j - c[q, 1]
                    pull_k = k - c[q, 2]
                    q_write = q
                    ## Slip condition on y-walls (top/bottom)
                    if pull_j < 0 or pull_j >= Ny:
                        pull_j = j
                        q_write = inv_cy_indx[q]
                    ## Slip condition on z-walls (front/back)
                    if pull_k < 0 or pull_k >= Nz:
                        pull_k = k
                        q_write = inv_cz_indx[q]
                    pops_pre[i, j, k, q] = pops_post[pull_i, pull_j, pull_k, q_write]
    
    # Z Bounds
    for k in [0, Nz-1]:
        k = np.int64(k)
        for i in nb.prange(1, Nx-1):
            i = np.int64(i)
            for j in range(1, Ny-1):
                for q in range(N_vels):
                    pull_i = i - c[q, 0]
                    pull_j = j - c[q, 1]
                    pull_k = k - c[q, 2]
                    q_write = q
                    ## Slip condition on z-walls (front/back)
                    if pull_k < 0 or pull_k >= Nz:
                        pull_k = k
                        q_write = inv_cz_indx[q]
                    pops_pre[i, j, k, q] = pops_post[pull_i, pull_j, pull_k, q_write]
    
    
    
    # Inlet - Zou-He Constant Velocity
    for j in nb.prange(Ny):
        j = np.int64(j)
        for k in range(Nz):
            pop0 = pops_pre[0, j, k, 0]
            pop2 = pops_pre[0, j, k, 2]
            pop3 = pops_pre[0, j, k, 3]
            pop4 = pops_pre[0, j, k, 4]
            pop5 = pops_pre[0, j, k, 5]
            pop6 = pops_pre[0, j, k, 6]
            pop8 = pops_pre[0, j, k, 8]
            pop10 = pops_pre[0, j, k, 10]
            pop11 = pops_pre[0, j, k, 11]
            pop12 = pops_pre[0, j, k, 12]
            pop14 = pops_pre[0, j, k, 14]
            pop16 = pops_pre[0, j, k, 16]
            pop17 = pops_pre[0, j, k, 17]
            pop18 = pops_pre[0, j, k, 18]
            
            local_rho = pop10 + pop16 + pop8 + pop14 + pop2
            local_rho = pop12 + pop17 + pop18 + pop11 + pop6 + pop5 + pop4 + pop3 + pop0 + local_rho*2.0
            local_rho_ux= Ux_t*local_rho/(1.0 - Ux_t)
            local_rho_ux_div6 = local_rho_ux/6.0
            
            Nxy = (-pop12 - pop18 - pop4 + pop17 + pop11 + pop3)/2.0
            Nxz = (-pop12 - pop17 - pop6 + pop18 + pop11 + pop5)/2.0
            
            pops_pre[0, j, k, 1] = pop2 + local_rho_ux/3.0
            pops_pre[0, j, k, 13] = pop14 + local_rho_ux_div6 + Nxy
            pops_pre[0, j, k, 7] = pop8 + local_rho_ux_div6 - Nxy
            pops_pre[0, j, k, 9] = pop10 + local_rho_ux_div6 - Nxz
            pops_pre[0, j, k, 15] = pop16 + local_rho_ux_div6 + Nxz
    
    
    # Outlet - Zou-He Constant Pressure (Density)
    alpha_prime = 1 - alpha
    for j in nb.prange(Ny):
        j = np.int64(j)
        for k in range(Nz):
            pop0 = pops_pre[-1, j, k, 0]
            pop1 = pops_pre[-1, j, k, 1]
            pop3 = pops_pre[-1, j, k, 3]
            pop4 = pops_pre[-1, j, k, 4]
            pop5 = pops_pre[-1, j, k, 5]
            pop6 = pops_pre[-1, j, k, 6]
            pop7 = pops_pre[-1, j, k, 7]
            pop9 = pops_pre[-1, j, k, 9]
            pop11 = pops_pre[-1, j, k, 11]
            pop12 = pops_pre[-1, j, k, 12]
            pop13 = pops_pre[-1, j, k, 13]
            pop15 = pops_pre[-1, j, k, 15]
            pop17 = pops_pre[-1, j, k, 17]
            pop18 = pops_pre[-1, j, k, 18]
            
            local_ux = pop15 + pop9 + pop13 + pop7 + pop1
            local_ux = pop12 + pop17 + pop18 + pop11 + pop6 + pop5 + pop4 + pop3 + pop0 + local_ux*2.0
            
            local_rho = alpha*rho[-2, j, k] + alpha_prime*rho_0
            
            local_ux_rho =  local_ux - local_rho
            local_ux_rho_div6 = local_ux_rho/6.0
            
            Nxy = (-pop12 - pop18 - pop4 + pop17 + pop11 + pop3)/2.0
            Nxz = (-pop12 - pop17 - pop6 + pop18 + pop11 + pop5)/2.0
            
            pops_pre[-1, j, k, 2] = pop1 - local_ux_rho/3.0
            pops_pre[-1, j, k, 14] = pop13 - local_ux_rho_div6 - Nxy
            pops_pre[-1, j, k, 8] = pop7 - local_ux_rho_div6 + Nxy
            pops_pre[-1, j, k, 10] = pop9 - local_ux_rho_div6 + Nxz
            pops_pre[-1, j, k, 16] = pop15 - local_ux_rho_div6 - Nxz


@nb.jit(nopython=True, parallel=True, fastmath=True)
def stream_closed(pops_pre, pops_post, Nx, Ny, Nz, N_vels, c, inv_cx_indx, inv_cy_indx, inv_cz_indx):
    """
    Streams particle populations, assuming all domain boundaries are slip walls.
    There are no external conditions which anchor fluid properties
    (specifically density); the fluid density may drift over long simulations.
    It may be necessary to anchor the fluid density globally via
    rho = alpha*rho + (1-alpha)*rho_0.

    Parameters
    ----------
    pops_pre : ndarray
        pre-collision populations. ndims=4, dtype=float
    pops_post : ndarray
        post-collision populations. ndims=4, dtype=float
    Nx : int
        number of fluid cells in the x direction.
    Ny : int
        number of fluid cells in the y direction.
    Nz : int
        number of fluid cells in the z direction.
    N_vels : int
        number of discrete velocities (i.e. "lattice vectors").
    c : ndarray
        discrete velocity set. ndtims=2, dtype=int
    inv_cx_indx : ndarray
        indexing array for specular reflection in the x direction. ndims=1, dtype=float
    inv_cy_indx : ndarray
        indexing array for specular reflection in the y direction. ndims=1, dtype=float
    inv_cz_indx : ndarray
        indexing array for specular reflection in the z direction. ndims=1, dtype=float

    Returns
    -------
    None.
    """
    
    # Bulk Streaming - Internal Nodes
    for i in nb.prange(1, Nx-1):
        i = np.int64(i)
        for j in range(1, Ny-1):
            for k in range(1, Nz-1):
                for q in range(N_vels):
                    pull_i = i - c[q, 0]
                    pull_j = j - c[q, 1]
                    pull_k = k - c[q, 2]
                    pops_pre[i, j, k, q] = pops_post[pull_i, pull_j, pull_k, q]
    
    # X Bounds
    for i in [0, Nx-1]:
        i = np.int64(i)
        for j in nb.prange(Ny):
            j = np.int64(j)
            for k in range(Nz):
                for q in range(N_vels):
                    pull_i = i - c[q, 0]
                    pull_j = j - c[q, 1]
                    pull_k = k - c[q, 2]
                    q_write = q
                    ## Slip condition on x-walls (left/right)
                    if pull_i < 0 or pull_i >= Nx:
                        pull_i = i
                        q_write = inv_cx_indx[q]
                    ## Slip condition on y-walls (top/bottom)
                    if pull_j < 0 or pull_j >= Ny:
                        pull_j = j
                        q_write = inv_cy_indx[q]
                    ## Slip condition on z-walls (front/back)
                    if pull_k < 0 or pull_k >= Nz:
                        pull_k = k
                        q_write = inv_cz_indx[q]
                    pops_pre[i, j, k, q] = pops_post[pull_i, pull_j, pull_k, q_write]
    
    # Y Bounds
    for j in [0, Ny-1]:
        j = np.int64(j)
        for i in nb.prange(1, Nx-1):
            i = np.int64(i)
            for k in range(Nz):
                for q in range(N_vels):
                    pull_i = i - c[q, 0]
                    pull_j = j - c[q, 1]
                    pull_k = k - c[q, 2]
                    q_write = q
                    ## Slip condition on y-walls (top/bottom)
                    if pull_j < 0 or pull_j >= Ny:
                        pull_j = j
                        q_write = inv_cy_indx[q]
                    ## Slip condition on z-walls (front/back)
                    if pull_k < 0 or pull_k >= Nz:
                        pull_k = k
                        q_write = inv_cz_indx[q]
                    pops_pre[i, j, k, q] = pops_post[pull_i, pull_j, pull_k, q_write]
    
    # Z Bounds
    for k in [0, Nz-1]:
        k = np.int64(k)
        for i in nb.prange(1, Nx-1):
            i = np.int64(i)
            for j in range(1, Ny-1):
                for q in range(N_vels):
                    pull_i = i - c[q, 0]
                    pull_j = j - c[q, 1]
                    pull_k = k - c[q, 2]
                    q_write = q
                    ## Slip condition on z-walls (front/back)
                    if pull_k < 0 or pull_k >= Nz:
                        pull_k = k
                        q_write = inv_cz_indx[q]
                    pops_pre[i, j, k, q] = pops_post[pull_i, pull_j, pull_k, q_write]



#%% Population Initialisation
@nb.jit(nopython=True, parallel=True, fastmath=True)
def initialise_pops(pops, F, u, u_mag2, Nx, Ny, Nz, rho_0, inv_cs2, inv_2cs2, inv_2cs4, N_vels, w, c):
    """
    Initialises particle populations with their equilibrium populations.

    Parameters
    ----------
    pops : ndarray
        lattice particle populations. ndims=4, dtype=float
    F : ndarray
        fluid force density field. ndims=4, dtype=float
    u : ndarray
        fluid velocity field. ndims=4, dtype=float
    u_mag2 : ndarray
        fluid velocity squared magnitude field. ndims=3, dtype=float
    Nx : int
        number of fluid cells in the x direction.
    Ny : int
        number of fluid cells in the y direction.
    Nz : int
        number of fluid cells in the z direction.
    rho_0 : float
        reference fluid density.
    inv_cs2 : float
        inverse squared sonic velocity.
    inv_2cs2 : float
        half inverse squared sonic velocity.
    inv_2cs4 : float
        half inverse sonic velocity fourth power.
    N_vels : int
        number of discrete velocities (i.e. "lattice vectors").
    w : ndarray
        discrete velocity weightings. ndims=1, dtype=float
    c : ndarray
        discrete velocity set. ndtims=2, dtype=int

    Returns
    -------
    None.
    """
    
    inv_2rho = 0.5/rho_0
    for i in nb.prange(Nx):
        i = np.int64(i)
        for j in range(Ny):
            for k in range(Nz):
                
                # Force Velocity Correction
                local_ux = u[i, j, k, 0] - F[i, j, k, 0]*inv_2rho
                local_uy = u[i, j, k, 1] - F[i, j, k, 1]*inv_2rho
                local_uz = u[i, j, k, 2] - F[i, j, k, 2]*inv_2rho
                local_u2 = local_ux*local_ux + local_uy*local_uy + local_uz*local_uz
                u[i, j, k, 0] = local_ux
                u[i, j, k, 1] = local_uy
                u[i, j, k, 2] = local_uz
                u_mag2[i, j, k] = local_u2
                
                # Calculate Equilibrium Populations
                for q in range(N_vels):
                    cu = c[q, 0]*local_ux + c[q, 1]*local_uy + c[q, 2]*local_uz
                    pops[i, j, k, q] = calc_f_eq_comp_BGK(cu, local_u2, rho_0, w[q], inv_cs2, inv_2cs2, inv_2cs4)



#%% Complete LBM Update Functions
def update_LBM_pops_1D_flow(pops_pre, pops_post, F, rho, u, u_mag2, Nx, Ny, Nz, rho_0, Ux_t, Uy_0, Uz_0, 
                            inv_cs2, inv_2cs2, inv_cs4, inv_2cs4, omega, omega_prime, omega_S_coeff, 
                            N_vels, w, c, inv_cy_indx, inv_cz_indx):
    """
    Updates the paricle populations for one time step using the quasi-
    compressible LB equaiton with BGK collisions and Guo forcing. First collides
    the particles, then streams the populations. 1D inlet flow.

    Parameters
    ----------
    pops_pre : ndarray
        pre-collision populations. ndims=4, dtype=float
    pops_post : ndarray
        post-collision populations. ndims=4, dtype=float
    F : ndarray
        fluid force density field. ndims=4, dtype=float
    rho : ndarray
        fluid density field. ndims=3, dtype=float
    u : ndarray
        fluid velocity field. ndims=4, dtype=float
    u_mag2 : ndarray
        fluid velocity squared magnitude field. ndims=3, dtype=float
    Nx : int
        number of fluid cells in the x direction.
    Ny : int
        number of fluid cells in the y direction.
    Nz : int
        number of fluid cells in the z direction.
    rho_0 : float
        reference fluid density.
    Ux_t : float
        current fluid inlet x velocity.
    Uy_0 : float
       fluid inlet y velocity (0).
    Uz_0 : float
        fluid inlet z velocity (0).
    inv_cs2 : float
        inverse squared sonic velocity.
    inv_2cs2 : float
        half inverse squared sonic velocity.
    inv_cs4 : float
        inverse sonic velocity fourth power.
    inv_2cs4 : float
        half inverse sonic velocity fourth power.
    omega : float
        inverse relaxation factor.
    omega_prime : float
        "conjugate" inverse relaxation factor (1 - omega).
    omega_S_coeff : float
        "conjugate" half inverse relaxation factor.
    N_vels : int
        number of discrete velocities (i.e. "lattice vectors").
    w : ndarray
        discrete velocity weightings. ndims=1, dtype=float
    c : ndarray
        discrete velocity set. ndtims=2, dtype=int
    inv_cy_indx : ndarray
        indexing array for specular reflection in the y direction. ndims=1, dtype=float
    inv_cz_indx : ndarray
        indexing array for specular reflection in the z direction. ndims=1, dtype=float

    Returns
    -------
    None.
    """
    
    # Calculate fluid properties and perform collisions
    collide_forced(pops_pre, pops_post, F, rho, u, u_mag2, Nx, Ny, Nz, 
                   inv_cs2, inv_2cs2, inv_cs4, inv_2cs4, omega, omega_prime, omega_S_coeff, N_vels, w, c)
    
    # Stream populations
    stream_1D_flow(pops_pre, pops_post, rho, Nx, Ny, Nz, rho_0, Ux_t, Uy_0, Uz_0, 
                   N_vels, c, inv_cy_indx, inv_cz_indx)


def update_LBM_pops_closed(pops_pre, pops_post, F, rho, u, u_mag2, Nx, Ny, Nz, 
                           inv_cs2, inv_2cs2, inv_cs4, inv_2cs4, omega, omega_prime, omega_S_coeff, 
                           N_vels, w, c, inv_cx_indx, inv_cy_indx, inv_cz_indx, nu, kB_T):
    """
    Updates the paricle populations for one time step using the quasi-
    compressible LB equaiton with BGK collisions and Guo forcing. First collides
    the particles, then streams the populations. Slip walls for all boundaries.

    Parameters
    ----------
    pops_pre : ndarray
        pre-collision populations. ndims=4, dtype=float
    pops_post : ndarray
        post-collision populations. ndims=4, dtype=float
    F : ndarray
        fluid force density field. ndims=4, dtype=float
    rho : ndarray
        fluid density field. ndims=3, dtype=float
    u : ndarray
        fluid velocity field. ndims=4, dtype=float
    u_mag2 : ndarray
        fluid velocity squared magnitude field. ndims=3, dtype=float
    Nx : int
        number of fluid cells in the x direction.
    Ny : int
        number of fluid cells in the y direction.
    Nz : int
        number of fluid cells in the z direction.
    inv_cs2 : float
        inverse squared sonic velocity.
    inv_2cs2 : float
        half inverse squared sonic velocity.
    inv_cs4 : float
        inverse sonic velocity fourth power.
    inv_2cs4 : float
        half inverse sonic velocity fourth power.
    omega : float
        inverse relaxation factor.
    omega_prime : float
        "conjugate" inverse relaxation factor (1 - omega).
    omega_S_coeff : float
        "conjugate" half inverse relaxation factor.
    N_vels : int
        number of discrete velocities (i.e. "lattice vectors").
    w : ndarray
        discrete velocity weightings. ndims=1, dtype=float
    c : ndarray
        discrete velocity set. ndtims=2, dtype=int
    inv_cx_indx : ndarray
        indexing array for specular reflection in the x direction. ndims=1, dtype=float
    inv_cy_indx : ndarray
        indexing array for specular reflection in the y direction. ndims=1, dtype=float
    inv_cz_indx : ndarray
        indexing array for specular reflection in the z direction. ndims=1, dtype=float

    Returns
    -------
    None.
    """
    
    # Calculate fluid properties and perform collisions
    collide_forced(pops_pre, pops_post, F, rho, u, u_mag2, Nx, Ny, Nz, inv_cs2, inv_2cs2, inv_cs4, inv_2cs4, omega_S_coeff, N_vels, w, c, nu, kB_T)
    #collide_forced2(pops_pre, pops_post, F, rho, u, u_mag2, Nx, Ny, Nz, inv_cs2, inv_2cs2, inv_cs4, inv_2cs4, omega, omega_prime, omega_S_coeff, N_vels, w, c)
    # Stream populations
    stream_closed(pops_pre, pops_post, Nx, Ny, Nz, 
                  N_vels, c, inv_cx_indx, inv_cy_indx, inv_cz_indx)
