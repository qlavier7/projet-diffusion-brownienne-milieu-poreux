#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A starting point for simulating the brownian motion of a spherical particle
based on the Langevin approach. 
For the fluctuating hydrodynamics approach, no force is applied to the particle; 
instead, it is simply carried along with the fluid. The fluctuating stress term 
will need to be added to the LBGK collision equation.

The boundaries of the simulation domain can be set to be periodic, slip, or no-
slip. An arbitrary no-slip obstacle can also be defined and included in the 
simulation.

Based on previous convergence analyses, the minimum accurate particle diameter
is 7 lattice units. Increasing it may help with stability (at the cost of
computation). Two force distribution functions are available; a standard
gaussian (standard Force Coupling Method) and a novel dual gaussian. For the
dual gaussian kernel, a distribution width of 2 is appropriate (smaller widths
cause substantial integration errors).
The goal of the novel kernel is to obtain a more accurate representation of a
spherical particle in Stokes flow - this may or may not be the case for the
Langevin or fluctuating hydrodynamics approaches. The y distribution of this
force distribution function (for only one particle) can be visualised if needed 
(it is radially symmetric in 3D - i.e. spherical).
If you want to run quicker simulations just for testing, use the standard
gaussian kernel with a smaller particle diameter (perhaps 4), but the results
will likely be inaccurate.

Substantial hydrodynamic hinderance is caused by the proximity of the slip walls
to the moving particle. Thus, it is expected that simulation results won't match 
analytical solutions. However, an approximation of the actual solution can 
usually be extrapolated from a series of simulations. 
Define x = (Nx/2)/r_particle (note that x=spacing_mutl). Assuming that the 
particle remains near the centre of the domain and the domain is cubic, a 
measured quantity of interest (y) can generally be approximated by the proposed 
function:
    y = (a/(x**b) + 1)*c
Thus, the value for y in a very large simulation domain can be estimated by 
performing a regression, and is equal to the asymptotic value c.

It is unecessary (and not recommended) to change the fluid parameters.

There may be stability issues, which may arise to due to large marker forcing
(or some other reasons). Tracking the total fluid mass through the domain over
time may help to highlight issues quickly.

The functionality exists to add additional particles, but nothing will happen
unless the starting positions are offset (beware of overlapping force 
distributions!).


Created on Wed Aug 26 13:26:02 2026
Author: Max Robbins
"""

import time
import tqdm
import warnings
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import imageio
from matplotlib.patches import Polygon
import matplotlib.cm as cm

from obs_forced_LBGK_lib import get_LBM_consts, initialise_pops, update_LBM_pops_closed#, update_LBM_pops_closed_combined
from multi_marker_IBM_lib_v2 import gaus_consts, gaus_dist, dual_gaus_consts, dual_gaus_dist, plot_gaus_dist, IB_force_density, interpolate_marker_vels

max_mem_avail = 12.0e9 # maximum available memory [bytes]



#%% Problem Input Parameters

# Graphing and Outputs
show_gaus_dist = False # plot the y distribution of the force distribution function
live_flow_plot = True # plot the flow field during the simulation
N_outputs = 10 # n.o. times to plot the solution field (only if live_flow_plot=True)
show_mass = False # plot the total fluid mass over the simulation duration - can be useful for identifying instabilities (should remain constant)


# Simulation Duration
sim_time = 1000 # simulation time [s] - adjust accordingly


# Fluid Domain Boundary Conditions
## BCs = [[x_low, x_high], [y_low, y_high], [z_low, z_high]]
## 0 = periodic boundary, 1 = slip boundary, 2 = no-slip boundary
## When using periodic boundaries, both i_low and i_high must == 0
BCs = [[0, 0], [0, 0], [0, 0]] # ex: all periodic
# BCs = [[1, 1], [1, 1], [1, 1]] # ex: all slip
# BCs = [[2, 2], [2, 2], [2, 2]] # ex: all no-slip
# BCs = [[0, 0], [2, 2], [2, 2]] # ex: x periodic, y/z no-slip
BCs = np.array(BCs, dtype=np.uint8)


# Geometry
D_particle1 = 7 # number of lattice points across the particle diameter
D_particle2 = 10
D_particle3 = 7
D_particles = [D_particle1, D_particle2, D_particle3] # list of particle diameters
r_particles = [D/2 for D in D_particles] # list of particle radii
spacing_mutl = 10 # control the spacing between the particle and the domain walls

Nx = int(spacing_mutl*D_particle1+1) # simulation domain length
Ny = Nx # simulation domain height
Nz = Nx # simulation domain depth
n_lattice = Nx*Ny*Nz # total n.o. fluid nodes

cx_particle = (Nx-1)/2 # particle initial x position
cy_particle = (Ny-1)/2 # particle initial y position
cz_particle = (Nz-1)/2 # particle initial z position
N_markers = 3 # number of particle markers - need to offset initial positions for anything to happen when increasing this from 1

stop_dist = D_particle1/2.0 # minimum distance from the particle to the wall before the simulation is stopped
stopping_lims = [[stop_dist, Nx-1-stop_dist], [stop_dist, Ny-1-stop_dist], [stop_dist, Nz-1-stop_dist]]


# IBM
IB_kernel = ['standard gaussian', 'dual gaussian'][1] # force distribution function to use
f_dist_width = 2 # width of the surface gaussian force distribution function [lattice points] (only for dual gaussian IB kernel)


# Fluid
nu = 1/50 # kinematic viscosity [m2 s-1]
rho_0 = 1.0 # initial density [kg m-3]
mu = rho_0*nu # dynamic viscosity [kg m-1 s-1]

# Particle volumic masses
rhos = np.ones(N_markers, dtype=np.float64) * 1.14 * rho_0 # list of particle densities (assumed to be the same for now)

# Diffusion - would probably be defined by a temperature
F_brownian_scale = 1 # sample parameter for example only - the simulation may go unstable if the marker force is too large

# Cylinder (fiber) properties
N_cylinders = 1
cyl_pos = np.array([[35.0, 35.0, 35.0]])   # Centers (X, Y, Z)
cyl_axis = np.array([[1.0, 1.0, 0.0]])      # Orientations (X, Y, Z)
cyl_axis = cyl_axis / np.linalg.norm(cyl_axis, axis=1)[:, np.newaxis]  # Normalize each axis vector
cyl_radius = [6.0]          # Radius of the cylinders
cyl_height = [40.0]         # Height of the cylinders

#%% Define Obstacle Geometry
@nb.jit(nopython=True, parallel=True, fastmath=True)
def initialise_obstacle(obstacle, Nx, Ny, Nz, cyl_pos, cyl_axis, cyl_radius, cyl_height):
    """
    Remplit le tableau booléen obstacle pour que le fluide LBM "voie" le cylindre.
    """
    obstacle.fill(False)
    N_cyl = cyl_pos.shape[0]
    
    for c in range(N_cyl):
        pos = cyl_pos[c]
        axis = cyl_axis[c] / np.linalg.norm(cyl_axis[c])
        r = cyl_radius[c]
        h_half = cyl_height[c] / 2.0
        
        for i in nb.prange(Nx):
            for j in range(Ny):
                for k in range(Nz):
                    dx = i - pos[0]
                    dy = j - pos[1]
                    dz = k - pos[2]
                    
                    h_proj = dx * axis[0] + dy * axis[1] + dz * axis[2]
                    if abs(h_proj) <= h_half:
                        perp_x = dx - h_proj * axis[0]
                        perp_y = dy - h_proj * axis[1]
                        perp_z = dz - h_proj * axis[2]
                        if (perp_x**2 + perp_y**2 + perp_z**2) <= r**2:
                            obstacle[i, j, k] = True



#%% Solver Parameters
Nt = int(sim_time) # number of time steps (since dt=1)
outevery = int(Nt/N_outputs) # generate an output every this many steps
# outevery = 1


domain_dims = [Nx, Ny, Nz]

stopping_lims = []
skip_stop_check = []
for dim in range(len(domain_dims)):
    
    if (BCs[dim, 0] == 0) ^ (BCs[dim, 1] == 0):
        raise ValueError(f'Periodic boundary conditions must be applied to both the upper and lower boundaries of a dimension, or neither (error for dimension {dim})')
    
    periodic_BC = (BCs[dim, 0] == 0) and (BCs[dim, 1] == 0)
    if periodic_BC:
        dim_lim = [None, None]
    else:
        dim_lim = [stop_dist, (domain_dims[dim]-1)-stop_dist]
    skip_stop_check.append(periodic_BC)
    stopping_lims.append(dim_lim.copy())


LBM_consts = get_LBM_consts(nu)
inv_cs2 = LBM_consts['inv_cs2']
inv_cs4 = LBM_consts['inv_cs4']
inv_2cs2 = LBM_consts['inv_2cs2']
inv_2cs4 = LBM_consts['inv_2cs4']
tau = LBM_consts['tau']
omega = LBM_consts['omega']
omega_prime = LBM_consts['omega_prime']
omega_S_coeff = LBM_consts['omega_S_coeff']
N_vels = LBM_consts['N_vels']
w = LBM_consts['w']
c = LBM_consts['c']
inv_cx_indx = LBM_consts['inv_cx_indx']
inv_cy_indx = LBM_consts['inv_cy_indx']
inv_cz_indx = LBM_consts['inv_cz_indx']
inv_c_indx = LBM_consts['inv_c_indx']

est_mem_req = 2.1*Nx*Ny*Nz*N_vels*8.0 # very rough estimation
if est_mem_req > max_mem_avail:
    cont = input(f'WARNING: Estimated memory requirements ({(est_mem_req*1e-9):.2f} GB) exceeds the maximum available memory specified ({(max_mem_avail*1e-9):.2f} GB). Continue? (y/n)')
    if cont != 'y':
        raise MemoryError()
else:
    print(f'Estimated Memory Requirements: {(est_mem_req*1e-9):.6f} GB\n')



#%% Brownian Motion
@nb.jit(nopython=True, parallel=True, fastmath=True)
def brownian_forcing(F_brownian_scale, N_markers, marker_f):
    """
    A sample marker forcing function for example only.
    
    Large marker forces can cause instabilities.
    """
    for m in nb.prange(N_markers):
        np.int64(m)
        marker_f[m, 0] = np.random.normal(0, 1)*F_brownian_scale
        marker_f[m, 1] = np.random.normal(0, 1)*F_brownian_scale
        marker_f[m, 2] = 0 # np.random.normal(0, 1)*F_brownian_scale



#%% Initialisation Functions
def initialise_fluid_arrays(Nx, Ny, Nz, rho_0, rho, u, u_mag_sq, F, pops_pre, pops_post):
    rho.fill(rho_0)
    u.fill(0.0)
    u_mag_sq.fill(0.0)
    F.fill(0.0)
    initialise_pops(pops_pre, F, u, u_mag_sq, Nx, Ny, Nz, rho_0, inv_cs2, inv_2cs2, inv_2cs4, N_vels, w, c)
    pops_post[:] = np.copy(pops_pre)


def initialise_IBM(init_marker_pos, marker_pos, marker_vel, marker_f):
    for dim in range(3):
        marker_pos[:, dim] = init_marker_pos[:, dim].copy()
        marker_vel[:, dim] = 0.0
        marker_f[:, dim] = 0.0



#%% Setup and Memory Allocation

# Fluid Arrays
rho = np.empty((Nx, Ny, Nz), dtype=np.float64) # densities
u = np.empty((Nx, Ny, Nz, 3), dtype=np.float64) # velocities
u_mag_sq = np.empty_like(rho) # squared velocity magnitudes
F = np.empty_like(u) # body forces

pops_pre = np.empty((Nx, Ny, Nz, N_vels), dtype=np.float64) # discrete velocity distribution functions
pops_post = np.empty_like(pops_pre) # second DVDF array for efficient data writing during streaming


# Obstacle Array
obstacle = np.zeros_like(rho, dtype=np.bool)

# IBM setup
sigma = np.empty(N_markers, dtype=np.float64)
A = np.empty(N_markers, dtype=np.float64)
r_cutoff = np.empty(N_markers, dtype=np.float64)
r_gaus = np.empty(N_markers, dtype=np.float64)

for m in range(N_markers):
    if IB_kernel == 'standard gaussian':
        r_gaus[m] = r_particles[m]
        sigma[m], A[m], r_cutoff[m] = gaus_consts(r_gaus[m])
        dist_func = gaus_dist
    elif IB_kernel == 'dual gaussian':
        r_gaus[m] = max(r_particles[m] - f_dist_width / 2.0, 0.0)
        sigma[m], A[m], r_cutoff[m] = dual_gaus_consts(f_dist_width, r_gaus[m])
        dist_func = dual_gaus_dist

if IB_kernel == 'standard gaussian':
    r_cutoff_outer = np.max(r_cutoff)
    r_cutoff_inner = 0.0
else:
    r_cutoff_outer = np.max(r_gaus + r_cutoff)
    r_cutoff_inner = max(np.max(r_gaus - r_cutoff), 0.0)

r_cutoff_outer_sq = r_cutoff_outer ** 2
r_cutoff_inner_sq = r_cutoff_inner ** 2

# IBM Arrays
init_marker_pos = np.empty((N_markers, 3), dtype=np.float64) # initial marker positions

# marker 1
init_marker_pos[0, 0] = cx_particle - 2.5*D_particle1
init_marker_pos[0, 1] = cy_particle
init_marker_pos[0, 2] = cz_particle

# marker 2
init_marker_pos[1, 0] = cx_particle + 2*D_particle2
init_marker_pos[1, 1] = cy_particle
init_marker_pos[1, 2] = cz_particle

# marker 3
init_marker_pos[2, 0] = cx_particle 
init_marker_pos[2, 1] = cy_particle - 2.5*D_particle3
init_marker_pos[2, 2] = cz_particle

marker_pos = np.empty_like(init_marker_pos) # Lagrangian boundary marker positions
marker_vel = np.empty_like(marker_pos) # Lagrangian boundary marker velocities
marker_f = np.empty_like(marker_pos) # Lagrangian boundary marker forces


# Allocate marker neighbourhood arrays
marker_nh_size = np.empty(N_markers, dtype=np.int64)
max_marker_neighbours = int((2*(np.ceil(r_cutoff_outer)+1))**3)
marker_nh = np.empty((N_markers, 5, max_marker_neighbours), dtype=np.float64) # x, y, z, mag, weighting


# Initialise Arrays
initialise_fluid_arrays(Nx, Ny, Nz, rho_0, rho, u, u_mag_sq, F, pops_pre, pops_post)
initialise_IBM(init_marker_pos, marker_pos, marker_vel, marker_f)
initialise_obstacle(obstacle, Nx, Ny, Nz, cyl_pos, cyl_axis, cyl_radius, cyl_height)



#%% Force Distribution Function Analysis
if show_gaus_dist:
    plot_gaus_dist(Ny, dist_func, r_particle2, f_dist_width, r_gaus, sigma, A, r_cutoff, r_cutoff_outer, IB_kernel, N_markers, marker_pos)



#%% Solver Loop
print(f'Particle 1 Diameter: {D_particle1}\nSpacing Multiplier: {spacing_mutl}\nNx, Ny, Nz: ({Nx}, {Ny}, {Nz})\nNumber of Lattice Points: {n_lattice}')
print(f'Boundary Representation: immersed boundary method ({IB_kernel})')
if IB_kernel == 'dual gaussian':
    print(f'Distribution Width: {f_dist_width}')
print(f'Kinematic Viscosity: {nu:.4f}\nInitial Fluid Density: {rho_0}\nForcing Scale: {F_brownian_scale}')
print(f'Relaxation Factor (BGK): {tau:.4f}\n')


@nb.jit(nopython=True, parallel=True, fastmath=True)
def save_marker_data(step, marker_pos_hist, marker_vel_hist, marker_f_hist, N_markers, marker_pos, marker_vel, marker_f):
    for m in nb.prange(N_markers):
        np.int64(m)
        marker_pos_hist[m, step, 0] = marker_pos[m, 0]
        marker_pos_hist[m, step, 1] = marker_pos[m, 1]
        marker_pos_hist[m, step, 2] = marker_pos[m, 2]
        marker_vel_hist[m, step, 0] = marker_vel[m, 0]
        marker_vel_hist[m, step, 1] = marker_vel[m, 1]
        marker_vel_hist[m, step, 2] = marker_vel[m, 2]
        marker_f_hist[m, step, 0] = marker_f[m, 0]
        marker_f_hist[m, step, 1] = marker_f[m, 1]
        marker_f_hist[m, step, 2] = marker_f[m, 2]

# @nb.jit(nopython=True, fastmath=True)
# def resolve_particle_collisions(marker_pos, marker_vel, D_particles, rhos):
#     """ This function computes the 3D collision response between two spherical particles.
#     It works for N  particles : for whatever mass, radius, initial speed.
#     It doesn't work when there is more than 1 collision at a time for the same particle.
#     """

#     collided = np.zeros(N_markers, dtype=nb.boolean)
    
#     for i in range(N_markers):
        
#         if collided[i]:
            
#             continue
        
#         for j in range(i+1, N_markers):
            
#             if collided[j]:
#                 continue
            
#             mi = 4/3 * np.pi * (D_particles[i] / 2.0) ** 3 * rhos[i] # mass of particle i
#             mj = 4/3 * np.pi * (D_particles[j] / 2.0) ** 3 * rhos[j] # mass of particle j
#             Ri = D_particles[i] / 2.0 # radius of particle i
#             Rj = D_particles[j] / 2.0 # radius of particle j
#             min_dist_sq = (Ri + Rj) ** 2
            
#             rx = marker_pos[i, 0] - marker_pos[j, 0]
#             ry = marker_pos[i, 1] - marker_pos[j, 1]
#             rz = marker_pos[i, 2] - marker_pos[j, 2]
            
#             vx = marker_vel[i, 0] - marker_vel[j, 0]
#             vy = marker_vel[i, 1] - marker_vel[j, 1]
#             vz = marker_vel[i, 2] - marker_vel[j, 2]
            
#             dist_sq = rx**2 + ry**2 + rz**2
#             a = vx**2 + vy**2 + vz**2
#             b = 2.0 * (rx*vx + ry*vy + rz*vz)
#             c = dist_sq - min_dist_sq
            
#             if c < 0.0:
#                 raise ValueError("Particles are overlapping at the start of the time step.")
            
#             if a > 0.0 and b < 0.0 and (b**2 - 4.0*a*c) >= 0.0:
#                 delta_t = (-b - np.sqrt(b**2 - 4.0*a*c)) / (2.0 * a)
                
#                 if 0.0 <= delta_t <= 1.0:
#                     rx_impact = rx + vx * delta_t
#                     ry_impact = ry + vy * delta_t
#                     rz_impact = rz + vz * delta_t
#                     dist_impact = np.sqrt(rx_impact**2 + ry_impact**2 + rz_impact**2)
                    
#                     if dist_impact > 0.0:
#                         nx, ny, nz = rx_impact / dist_impact, ry_impact / dist_impact, rz_impact / dist_impact
#                         v_dot_n = vx * nx + vy * ny + vz * nz
                        
#                         v0_x, v0_y, v0_z = marker_vel[i, 0], marker_vel[i, 1], marker_vel[i, 2]
#                         v1_x, v1_y, v1_z = marker_vel[j, 0], marker_vel[j, 1], marker_vel[j, 2]
                        
#                         marker_vel[i, 0] -= 2 * mj / (mi + mj) * v_dot_n * nx
#                         marker_vel[i, 1] -= 2 * mj / (mi + mj) * v_dot_n * ny
#                         marker_vel[i, 2] -= 2 * mj / (mi + mj) * v_dot_n * nz
                        
#                         marker_vel[j, 0] += 2 * mi / (mi + mj) * v_dot_n * nx
#                         marker_vel[j, 1] += 2 * mi / (mi + mj) * v_dot_n * ny
#                         marker_vel[j, 2] += 2 * mi / (mi + mj) * v_dot_n * nz

#                         marker_pos[i, 0] += v0_x * delta_t + marker_vel[i, 0] * (1.0 - delta_t)
#                         marker_pos[i, 1] += v0_y * delta_t + marker_vel[i, 1] * (1.0 - delta_t)
#                         marker_pos[i, 2] += v0_z * delta_t + marker_vel[i, 2] * (1.0 - delta_t)
                        
#                         marker_pos[j, 0] += v1_x * delta_t + marker_vel[j, 0] * (1.0 - delta_t)
#                         marker_pos[j, 1] += v1_y * delta_t + marker_vel[j, 1] * (1.0 - delta_t)
#                         marker_pos[j, 2] += v1_z * delta_t + marker_vel[j, 2] * (1.0 - delta_t)
                        
#                         collided[i] = True
#                         collided[j] = True
#                         break
    
#     for i in range(N_markers):
#         if not collided[i]:
#             marker_pos[i, 0] += marker_vel[i, 0]
#             marker_pos[i, 1] += marker_vel[i, 1]
#             marker_pos[i, 2] += marker_vel[i, 2]

@nb.jit(nopython=True, fastmath=True)
def resolve_all_collisions(marker_pos, marker_vel, marker_f, D_particles, rhos, 
                           cyl_pos, cyl_axis, cyl_radius, cyl_height):
    """
    Gestion unifiée des collisions 3D par redirection du vecteur force applique (marker_f) :
      - Cas 1 : Collision particule-particule (redirection/transfert de force)
      - Cas 2 : Collision particule-cylindre/wall (réflexion de la force sur la normale)
      - Cas 3 : Déplacement libre si aucune collision (marker_pos += marker_vel)
    """
    N_markers = marker_pos.shape[0]
    N_cyl = cyl_pos.shape[0]
    
    # Marqueur pour suivre les particules ayant déjà subi un choc durant ce step
    collided = np.zeros(N_markers, dtype=nb.boolean)
    
    # -------------------------------------------------------------------------
    # 1. TEST ET RÉSOLUTIONS DES COLLISIONS PARTICULE - PARTICULE
    # -------------------------------------------------------------------------
    for i in range(N_markers):
        if collided[i]:
            continue
            
        for j in range(i + 1, N_markers):
            if collided[j]:
                continue
                
            mi = (4.0 / 3.0) * np.pi * (D_particles[i] / 2.0)**3 * rhos[i]
            mj = (4.0 / 3.0) * np.pi * (D_particles[j] / 2.0)**3 * rhos[j]
            Ri = D_particles[i] / 2.0
            Rj = D_particles[j] / 2.0
            min_dist_sq = (Ri + Rj)**2
            
            rx = marker_pos[i, 0] - marker_pos[j, 0]
            ry = marker_pos[i, 1] - marker_pos[j, 1]
            rz = marker_pos[i, 2] - marker_pos[j, 2]
            
            vx = marker_vel[i, 0] - marker_vel[j, 0]
            vy = marker_vel[i, 1] - marker_vel[j, 1]
            vz = marker_vel[i, 2] - marker_vel[j, 2]
            
            dist_sq = rx**2 + ry**2 + rz**2
            a = vx**2 + vy**2 + vz**2
            b = 2.0 * (rx * vx + ry * vy + rz * vz)
            c = dist_sq - min_dist_sq
            
            if a > 0.0 and b < 0.0 and (b**2 - 4.0 * a * c) >= 0.0:
                delta_t = (-b - np.sqrt(b**2 - 4.0 * a * c)) / (2.0 * a)
                
                if 0.0 <= delta_t <= 1.0:
                    rx_imp = rx + vx * delta_t
                    ry_imp = ry + vy * delta_t
                    rz_imp = rz + vz * delta_t
                    dist_imp = np.sqrt(rx_imp**2 + ry_imp**2 + rz_imp**2)
                    
                    if dist_imp > 0.0:
                        nx, ny, nz = rx_imp / dist_imp, ry_imp / dist_imp, rz_imp / dist_imp
                        
                        # Projection de la force relative sur la normale
                        fx_rel = marker_f[i, 0] - marker_f[j, 0]
                        fy_rel = marker_f[i, 1] - marker_f[j, 1]
                        fz_rel = marker_f[i, 2] - marker_f[j, 2]
                        f_dot_n = fx_rel * nx + fy_rel * ny + fz_rel * nz
                        
                        # Redirection élastique de la force entre les 2 sphères
                        if f_dot_n < 0.0:
                            marker_f[i, 0] -= 2.0 * (mj / (mi + mj)) * f_dot_n * nx
                            marker_f[i, 1] -= 2.0 * (mj / (mi + mj)) * f_dot_n * ny
                            marker_f[i, 2] -= 2.0 * (mj / (mi + mj)) * f_dot_n * nz
                            
                            marker_f[j, 0] += 2.0 * (mi / (mi + mj)) * f_dot_n * nx
                            marker_f[j, 1] += 2.0 * (mi / (mi + mj)) * f_dot_n * ny
                            marker_f[j, 2] += 2.0 * (mi / (mi + mj)) * f_dot_n * nz
                        
                        # Avancement jusqu'au point de contact
                        marker_pos[i, 0] += marker_vel[i, 0] * delta_t
                        marker_pos[i, 1] += marker_vel[i, 1] * delta_t
                        marker_pos[i, 2] += marker_vel[i, 2] * delta_t
                        
                        marker_pos[j, 0] += marker_vel[j, 0] * delta_t
                        marker_pos[j, 1] += marker_vel[j, 1] * delta_t
                        marker_pos[j, 2] += marker_vel[j, 2] * delta_t
                        
                        collided[i] = True
                        collided[j] = True
                        break

    # -------------------------------------------------------------------------
    # 2. TEST ET RÉSOLUTIONS DES COLLISIONS PARTICULE - CYLINDRE (WALL)
    # -------------------------------------------------------------------------
    for i in range(N_markers):
        if collided[i]:
            continue
            
        Ri = D_particles[i] / 2.0
        
        for c in range(N_cyl):
            pos_c = cyl_pos[c]
            axis_c = cyl_axis[c]  # Supposé unitaire
            R_cyl = cyl_radius[c]
            H_half = cyl_height[c] / 2.0
            
            min_dist = Ri + R_cyl
            min_dist_sq = min_dist**2
            
            rx = marker_pos[i, 0] - pos_c[0]
            ry = marker_pos[i, 1] - pos_c[1]
            rz = marker_pos[i, 2] - pos_c[2]
            
            vx = marker_vel[i, 0]
            vy = marker_vel[i, 1]
            vz = marker_vel[i, 2]
            
            h_proj = rx * axis_c[0] + ry * axis_c[1] + rz * axis_c[2]
            
            # Projection orthogonale
            perp_x = rx - h_proj * axis_c[0]
            perp_y = ry - h_proj * axis_c[1]
            perp_z = rz - h_proj * axis_c[2]
            
            dist_perp_sq = perp_x**2 + perp_y**2 + perp_z**2
            
            if abs(h_proj) <= (H_half + Ri):
                vh = vx * axis_c[0] + vy * axis_c[1] + vz * axis_c[2]
                v_perp_x = vx - vh * axis_c[0]
                v_perp_y = vy - vh * axis_c[1]
                v_perp_z = vz - vh * axis_c[2]
                
                a = v_perp_x**2 + v_perp_y**2 + v_perp_z**2
                b = 2.0 * (perp_x * v_perp_x + perp_y * v_perp_y + perp_z * v_perp_z)
                c_eq = dist_perp_sq - min_dist_sq
                
                if a > 0.0 and b < 0.0 and (b**2 - 4.0 * a * c_eq) >= 0.0:
                    delta_t = (-b - np.sqrt(b**2 - 4.0 * a * c_eq)) / (2.0 * a)
                    
                    if 0.0 <= delta_t <= 1.0:
                        perp_x_imp = perp_x + v_perp_x * delta_t
                        perp_y_imp = perp_y + v_perp_y * delta_t
                        perp_z_imp = perp_z + v_perp_z * delta_t
                        dist_imp = np.sqrt(perp_x_imp**2 + perp_y_imp**2 + perp_z_imp**2)
                        
                        if dist_imp > 0.0:
                            nx = perp_x_imp / dist_imp
                            ny = perp_y_imp / dist_imp
                            nz = perp_z_imp / dist_imp
                            
                            v_dot_n = marker_vel[i, 0] * nx + marker_vel[i, 1] * ny + marker_vel[i, 2] * nz
                            
                            # Réflexion du vecteur force applique s'il pointe vers l'obstacle
                            if v_dot_n < 0.0:
                                vx_impact = marker_vel[i, 0]
                                vy_impact = marker_vel[i, 1]
                                vz_impact = marker_vel[i, 2]
                                
                                vx_impact -= 2.0 * v_dot_n * nx
                                vy_impact -= 2.0 * v_dot_n * ny
                                vz_impact -= 2.0 * v_dot_n * nz

                                # Déplacement arrêté au point d'impact + marge de sécurité
                                marker_pos[i, 0] += vx * delta_t + vx_impact * (1.0 - delta_t)
                                marker_pos[i, 1] += vy * delta_t + vy_impact * (1.0 - delta_t)
                                marker_pos[i, 2] += vz * delta_t + vz_impact * (1.0 - delta_t)
                            
                                marker_f[i, 0] -= 2 * mi * v_dot_n * nx
                                marker_f[i, 1] -= 2 * mi * v_dot_n * ny
                                marker_f[i, 2] -= 2 * mi * v_dot_n * nz
                            
                            collided[i] = True
                            break

    # -------------------------------------------------------------------------
    # 3. CAS AUCUNE COLLISION : DÉPLACEMENT LIBRE
    # -------------------------------------------------------------------------
    for i in range(N_markers):
        if not collided[i]:
            marker_pos[i, 0] += marker_vel[i, 0]
            marker_pos[i, 1] += marker_vel[i, 1]
            marker_pos[i, 2] += marker_vel[i, 2]

def compute_cylinder_z_thickness(cyl_pos, cyl_axis, cyl_radius, cyl_height, grid_x, grid_y):
    """
    Computes the exact thickness traversed along the Z-axis by a 3D cylinder.
    `grid_x` and `grid_y` are 2D grids.
    """
    axis = cyl_axis / np.linalg.norm(cyl_axis)
    u_x, u_y, u_z = axis
    
    dx = grid_x - cyl_pos[0]
    dy = grid_y - cyl_pos[1]
    
    thickness = np.zeros_like(grid_x, dtype=float)
    
    # Case 1 : Cylinder strictly horizontal (parallel to the XY plane, u_z = 0)
    if abs(u_z) < 1e-6:
        h_proj = dx * u_x + dy * u_y
        d_perp_sq = (dx**2 + dy**2) - h_proj**2
        inside = (d_perp_sq <= cyl_radius**2) & (np.abs(h_proj) <= cyl_height / 2.0)
        thickness[inside] = 2.0 * np.sqrt(np.maximum(0, cyl_radius**2 - d_perp_sq[inside]))
        return thickness

    # Case 2 : Cylinder not strictly horizontal (u_z != 0)
    A = 1.0 - u_z**2
    B = -2.0 * u_z * (dx * u_x + dy * u_y)
    C = (dx**2 + dy**2) - (dx * u_x + dy * u_y)**2 - cyl_radius**2
    
    delta = B**2 - 4.0 * A * C
    valid = delta >= 0
    
    if np.any(valid):
        sqrt_delta = np.sqrt(np.maximum(0, delta[valid]))
        dz1 = (-B[valid] - sqrt_delta) / (2.0 * A)
        dz2 = (-B[valid] + sqrt_delta) / (2.0 * A)
        
        h1 = (dx[valid] * u_x + dy[valid] * u_y) + dz1 * u_z
        h2 = (dx[valid] * u_x + dy[valid] * u_y) + dz2 * u_z
        
        h_max = cyl_height / 2.0
        
        dz_top = np.where(np.abs(h2) <= h_max, dz2, np.sign(u_z) * h_max - (dx[valid]*u_x + dy[valid]*u_y) / u_z)
        dz_bot = np.where(np.abs(h1) <= h_max, dz1, -np.sign(u_z) * h_max - (dx[valid]*u_x + dy[valid]*u_y) / u_z)
        
        thick_vals = dz_top - dz_bot
        thickness[valid] = np.maximum(0.0, thick_vals)
        
    return thickness

def run_diff_sim(pops_pre, pops_post, F, rho, u, u_mag_sq, obstacle, N_markers, marker_pos, marker_vel, marker_f, marker_nh, marker_nh_size, 
                 Nt, F_brownian_scale, Nx, Ny, Nz, n_lattice, r_cutoff_outer, r_cutoff_outer_sq, r_cutoff_inner_sq, dist_func, r_gaus, sigma, A, stopping_lims, 
                 inv_cs2, inv_2cs2, inv_cs4, inv_2cs4, omega, omega_prime, omega_S_coeff, N_vels, w, c, inv_cx_indx, inv_cy_indx, inv_cz_indx, inv_c_indx, BCs, skip_stop_check, 
                 live_flow_plot, outevery):
    
    break_cond = False
    int_err = 0.0
    u_mag_sq_max = 0.0
    
    marker_pos_hist = np.empty((N_markers, Nt, 3), dtype=np.float64)
    marker_vel_hist = np.empty_like(marker_pos_hist)
    marker_f_hist = np.empty_like(marker_pos_hist)
    fluid_mass_hist = np.empty(Nt, dtype=np.float64)

    # add initial force to particles
    marker_f[0, 0] = 0.5 # initial force marker 1
    # marker_f[0, 0] = -0.5 # initial force marker 1
    marker_f[2, 1] = 0.5 # initial force marker 2
    # marker_f[2, 1] = 1.5 # initial force marker 3
    
    iterations = tqdm.tqdm(range(Nt)) # initialise progress bar
    start_time = time.perf_counter()
    
    frames = []
    for t in iterations:
        
        # remove force after 100 steps (frame 2)
        if t>=100:
            marker_f[0, 0] = 0 # update force marker 1
            marker_f[1, 0] = 0 # update force marker 2
            marker_f[2, 1] = 0 # update force marker 3
        
        if np.isnan(u_mag_sq).any():
            raise RuntimeError(f'Unrealistic velocities: t={t}')
        
        # Calculate marker forces
        brownian_forcing(F_brownian_scale, N_markers, marker_f)
        # marker_f[0, 0] = 0.5 # initial force marker 1
        # # marker_f[0, 0] = -0.5 # initial force marker 1
        # marker_f[2, 1] = 0.5 # initial force marker 2
        resolve_all_collisions(marker_pos, marker_vel, marker_f, D_particles, rhos, cyl_pos, cyl_axis, cyl_radius, cyl_height)
        
        # Calculate forcing due to IB markers
        int_err = IB_force_density(Nx, Ny, Nz, r_cutoff_outer, r_cutoff_outer_sq, r_cutoff_inner_sq, F, 
                                   dist_func, r_gaus, sigma, A, N_markers, marker_pos, marker_f, marker_nh, marker_nh_size, int_err)
        
        # Calculate fluid properties, perform collisions, and stream populations
        # update_LBM_pops_closed_combined(t, pops_pre, pops_post, F, rho, u, u_mag_sq, Nx, Ny, Nz, 
        #                                 inv_cs2, inv_2cs2, inv_cs4, inv_2cs4, omega, omega_prime, omega_S_coeff, 
        #                                 N_vels, w, c, inv_cx_indx, inv_cy_indx, inv_cz_indx)
        update_LBM_pops_closed(pops_pre, pops_post, F, rho, u, u_mag_sq, obstacle, Nx, Ny, Nz, 
                               inv_cs2, inv_2cs2, inv_cs4, inv_2cs4, omega, omega_prime, omega_S_coeff, 
                               N_vels, w, c, inv_cx_indx, inv_cy_indx, inv_cz_indx, inv_c_indx, BCs)
        
        # Interpolate boundary marker velocities
        interpolate_marker_vels(u, N_markers, marker_vel, marker_nh, marker_nh_size)
        
        # Integrate boundary markers
        # marker_pos += marker_vel # since dt=1
        
        for dim in range(len(domain_dims)):
            if skip_stop_check[dim]:
                marker_pos[:, dim] %= (domain_dims[dim]-1)
        
        
        # Save marker data
        save_marker_data(t, marker_pos_hist, marker_vel_hist, marker_f_hist, N_markers, marker_pos, marker_vel, marker_f)
        
        ## Integrate and save fluid mass
        fluid_mass_hist[t] = np.sum(rho)
        
        
        # Check stopping criteria
        for m in range(N_markers):
            np.int64(m)
            inbounds_x = skip_stop_check[0] or (stopping_lims[0][0] <= marker_pos[m, 0] <= stopping_lims[0][1])
            inbounds_y = skip_stop_check[1] or (stopping_lims[1][0] <= marker_pos[m, 1] <= stopping_lims[1][1])
            inbounds_z = skip_stop_check[2] or (stopping_lims[2][0] <= marker_pos[m, 2] <= stopping_lims[2][1])
            # if not (inbounds_x and inbounds_y and inbounds_z):
            #     break_cond = True
            #     save_data = True
            #     break
            if m == N_markers-1:
                save_data = (t%outevery == 0) or (t == Nt-1)
        
        
        # Output flow field at particle cross-section
        # if save_data and live_flow_plot:
        #     u_mag_sq_curr = np.max(u_mag_sq)
        #     if u_mag_sq_curr > u_mag_sq_max:
        #         u_mag_sq_max = u_mag_sq_curr
            
        #     m = 0 # only plot the first marker
        #     z_slice = min(max(int(round(marker_pos[m, 2])), 0), Nz-1)
            
        #     plt.figure(figsize=(5, 4))
        #     im = plt.imshow(np.sqrt(u_mag_sq[:, :, z_slice]).T, cmap='viridis', origin='lower', vmin=0, vmax=u_mag_sq_max**0.5)
        #     # im = plt.imshow(u[:, :, z_slice, 0].T, cmap='viridis', origin='lower')
        #     # im = plt.imshow(rho[:, :, z_slice].T, cmap='viridis', origin='lower')
        #     # im = plt.imshow(F[:, :, z_slice, 0].T, cmap='viridis', origin='lower')
        #     plt.colorbar(im, label='Velocity Magnitude')
            
        #     for m in range(N_markers):
        #         plt.plot(marker_pos_hist[m, :t+1, 0], marker_pos_hist[m, :t+1, 1], 'r', alpha=0.5)
        #         plt.plot(init_marker_pos[m, 0], init_marker_pos[m, 1], 'xr')
                
        #         circle = plt.Circle((marker_pos[m, 0], marker_pos[m, 1]), r_particles[m], color='red', fill=False, linewidth=1.5)
        #         plt.gca().add_patch(circle)
            
        #     plt.xlim([0, Nx-1])
        #     plt.ylim([0, Ny-1])
            
        #     plt.title(f'3D Particle Diffusion - {IB_kernel} IBM\nZ Position = {z_slice}, t = {t}')
        #     plt.xlabel('X Position')
        #     plt.ylabel('Y Position')
        #     plt.tight_layout()
        #     # plt.savefig(f'{t}_diff_ani.png')
        #     plt.show()
        
        # Output flow field at particle cross-section
        if save_data and live_flow_plot:
            u_mag_sq_curr = np.max(u_mag_sq)
            if u_mag_sq_curr > u_mag_sq_max:
                u_mag_sq_max = u_mag_sq_curr
            
            m = 0
            z_slice = min(max(int(round(marker_pos[m, 2])), 0), Nz-1)
            
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(np.sqrt(u_mag_sq[:, :, z_slice]).T, cmap='viridis', origin='lower', vmin=0, vmax=u_mag_sq_max**0.5)
            plt.colorbar(im, label='Velocity Magnitude')
            
            for m_idx in range(N_markers):
                ax.plot(marker_pos_hist[m_idx, :t+1, 0], marker_pos_hist[m_idx, :t+1, 1], 'r', alpha=0.5)
                ax.plot(init_marker_pos[m_idx, 0], init_marker_pos[m_idx, 1], 'xr')
                circle = plt.Circle((marker_pos[m_idx, 0], marker_pos[m_idx, 1]), r_particles[m_idx], color='red', fill=False, linewidth=1.5)
                ax.add_patch(circle)
            
            for c_idx in range(N_cylinders):
                pos = cyl_pos[c_idx]
                axis = cyl_axis[c_idx] / np.linalg.norm(cyl_axis[c_idx]) 
                r = cyl_radius[c_idx]
                h = cyl_height[c_idx]
                
                sorted_pts = None
                
                # Case 1 : Cylinder parallel to the XY plane (u_z = 0) -> Rectangular cross-section
                if np.abs(axis[2]) < 1e-4:
                    dz = abs(z_slice - pos[2])
                    if dz <= r:
                        w_eff = np.sqrt(r**2 - dz**2)
                        axis_2d = axis[:2] / np.linalg.norm(axis[:2])
                        perp_2d = np.array([-axis_2d[1], axis_2d[0]])
                        
                        p1 = pos[:2] - (h/2)*axis_2d - w_eff*perp_2d
                        p2 = pos[:2] + (h/2)*axis_2d - w_eff*perp_2d
                        p3 = pos[:2] + (h/2)*axis_2d + w_eff*perp_2d
                        p4 = pos[:2] - (h/2)*axis_2d + w_eff*perp_2d
                        
                        sorted_pts = np.array([p1, p2, p3, p4])
                
                # Case 2 : Cylinder not parallel to the XY plane (u_z != 0) -> Elliptical cross-section
                else:
                    if np.allclose(axis[:2], 0):
                        v1 = np.array([1.0, 0.0, 0.0])
                    else:
                        v1 = np.array([-axis[1], axis[0], 0.0])
                        v1 /= np.linalg.norm(v1)
                    v2 = np.cross(axis, v1)
                    
                    t_vals = np.linspace(-h/2, h/2, 200)
                    pts_intersection = []
                    
                    for t_val in t_vals:
                        center_t = pos + t_val * axis
                        z_diff = z_slice - center_t[2]
                        denom = np.sqrt((r * v1[2])**2 + (r * v2[2])**2)
                        
                        if denom > 1e-6 and abs(z_diff) <= denom:
                            phi0 = np.arctan2(v2[2], v1[2])
                            arg = z_diff / denom
                            delta_phi = np.arccos(np.clip(arg, -1.0, 1.0))
                            
                            for angle in [phi0 + delta_phi, phi0 - delta_phi]:
                                pt = center_t + r * np.cos(angle) * v1 + r * np.sin(angle) * v2
                                pts_intersection.append(pt[:2])
                    
                    if len(pts_intersection) > 3:
                        pts_intersection = np.array(pts_intersection)
                        center_2d = np.mean(pts_intersection, axis=0)
                        angles = np.arctan2(pts_intersection[:, 1] - center_2d[1], pts_intersection[:, 0] - center_2d[0])
                        sorted_pts = pts_intersection[np.argsort(angles)]
                
                # Colormap for cylinder thickness
                if sorted_pts is not None and len(sorted_pts) > 3:
                    x_min, y_min = np.min(sorted_pts[:, 0]), np.min(sorted_pts[:, 1])
                    x_max, y_max = np.max(sorted_pts[:, 0]), np.max(sorted_pts[:, 1])
                    
                    res = 100
                    gx, gy = np.meshgrid(np.linspace(x_min, x_max, res), 
                                         np.linspace(y_min, y_max, res))
                    
                    thick_map = compute_cylinder_z_thickness(cyl_pos[c_idx], cyl_axis[c_idx], 
                                                            cyl_radius[c_idx], cyl_height[c_idx], gx, gy)
                    
                    from matplotlib.path import Path
                    path = Path(sorted_pts)
                    points = np.column_stack((gx.flatten(), gy.flatten()))
                    mask = path.contains_points(points).reshape(gx.shape)
                    
                    thick_map_masked = np.ma.masked_where(~mask | (thick_map <= 1e-5), thick_map)
                    
                    im_cyl = ax.imshow(thick_map_masked, origin='lower', 
                                       extent=[x_min, x_max, y_min, y_max],
                                       cmap='plasma', zorder=5, alpha=0.95)
                    
                    polygon_outline = Polygon(sorted_pts, fill=False, edgecolor='none', linewidth=0.0, zorder=6)
                    ax.add_patch(polygon_outline)
            
            ax.set_xlim([0, Nx-1])
            ax.set_ylim([0, Ny-1])
            ax.set_title(f'3D Particle Diffusion - {IB_kernel} IBM\nZ Position = {z_slice}, t = {t}')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            plt.tight_layout()
            
            # Save the current frame for GIF creation
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
            frames.append(image)
            plt.close(fig)
        
        if break_cond:
            break
    
    if live_flow_plot and len(frames) > 0:
        imageio.mimsave('particle_diffusion.gif', frames, fps=5, loop=0)
    end_time = time.perf_counter()
    loop_wt = end_time - start_time
    cell_updates = n_lattice*(t+1)
    print(f'\nLoop Wall Time: {loop_wt:.6f} s\nEfficiency: {(1e-6*cell_updates/loop_wt):.4f} MCUPS')

    if int_err > 1.0e-2:
        warnings.warn(f'Subtantial kernel integration error (likely due to unrefined mesh): {(int_err*100):.4f} %', RuntimeWarning)

    if t == Nt-1:
        simtime_reached = True
    else:
        simtime_reached = False
    
    marker_pos_hist = marker_pos_hist[:, :(t+1), :]
    marker_vel_hist = marker_vel_hist[:, :(t+1), :]
    marker_f_hist = marker_f_hist[:, :(t+1), :]
    fluid_mass_hist = fluid_mass_hist[:(t+1)]
    
    return marker_pos_hist, marker_vel_hist, marker_f_hist, fluid_mass_hist, simtime_reached



sim_res = run_diff_sim(pops_pre, pops_post, F, rho, u, u_mag_sq, obstacle, N_markers, marker_pos, marker_vel, marker_f, marker_nh, marker_nh_size, 
                       Nt, F_brownian_scale, Nx, Ny, Nz, n_lattice, r_cutoff_outer, r_cutoff_outer_sq, r_cutoff_inner_sq, dist_func, r_gaus, sigma, A, stopping_lims, 
                       inv_cs2, inv_2cs2, inv_cs4, inv_2cs4, omega, omega_prime, omega_S_coeff, N_vels, w, c, inv_cx_indx, inv_cy_indx, inv_cz_indx, inv_c_indx, BCs, skip_stop_check, 
                       live_flow_plot, outevery)

marker_pos_hist, marker_vel_hist, marker_f_hist, fluid_mass_hist, simtime_reached = sim_res



if show_mass:
    # Plot Fluid Mass
    N_steps = fluid_mass_hist.size
    time_hist = np.arange(0, N_steps)
    init_mass = fluid_mass_hist[0]
    rel_mass_change = 100*(fluid_mass_hist-init_mass)/init_mass
    
    fig_mass = plt.figure(figsize=(6, 4))
    plt.plot(time_hist, rel_mass_change, 'b-')
    plt.title('Domain Mass Integral')
    plt.xlabel('Time')
    plt.ylabel('Relative Mass Change (%)')
    plt.grid()
    plt.show()








