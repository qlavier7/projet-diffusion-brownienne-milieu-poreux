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
import sys
import os

base_path = os.path.dirname(os.path.dirname(__file__))
lib_path = os.path.join(base_path, "fluid solver code")

sys.path.append(lib_path)

from obs_forced_LBGK_lib import get_LBM_consts, initialise_pops, update_LBM_pops_closed
from multi_marker_IBM_lib import gaus_consts, gaus_dist, dual_gaus_consts, dual_gaus_dist, plot_gaus_dist, IB_force_density, interpolate_marker_vels

max_mem_avail = 12.0e9 # maximum available memory [bytes]



#%% Problem Input Parameters

# Graphing and Outputs
show_gaus_dist = False # plot the y distribution of the force distribution function
live_flow_plot = True # plot the flow field during the simulation
N_outputs = 10 # n.o. times to plot the solution field (only if live_flow_plot=True)
show_mass = False # plot the total fluid mass over the simulation duration - can be useful for identifying instabilities (should remain constant)
output_gif_filename = 'martinou/gif/fig8_sim2'
output_figname = 'martinou/fig8_sim2.png'

# Simulation Duration
sim_time = 1000 # simulation time [s] - adjust accordingly


# Fluid Domain Boundary Conditions
## BCs = [[x_low, x_high], [y_low, y_high], [z_low, z_high]]
## 0 = periodic boundary, 1 = slip boundary, 2 = no-slip boundary
## When using periodic boundaries, both i_low and i_high must == 0
#BCs = [[0, 0], [0, 0], [0, 0]] # ex: all periodic
#BCs = [[1, 1], [1, 1], [1, 1]] # ex: all slip
# BCs = [[2, 2], [2, 2], [2, 2]] # ex: all no-slip
# BCs = [[0, 0], [2, 2], [2, 2]] # ex: x periodic, y/z no-slip
BCs = [[0, 0], [1, 1], [0, 0]]
BCs = np.array(BCs, dtype=np.uint8)
wall_y = -0.5


# Geometry
D_particle1 = 9.6 # number of lattice points across the particle diameter
D_particle2 = 9.6
D_particles = [D_particle1, D_particle2] # list of particle diameters

r_particles = [D/2 for D in D_particles] # list of particle radii
marker_radii = np.array(r_particles, dtype=np.float64) # array of particle radii

spacing_mutl = 10 # control the spacing between the particle and the domain walls

Nx = int(spacing_mutl*D_particles[0]+1) # simulation domain length
Ny = Nx # simulation domain height
Nz = Nx # simulation domain depth

cx_particle = (Nx-1)/2 # particle initial x position
cy_particle = 1.2*r_particles[0] + wall_y # particle initial y position
cz_particle = (Nz-1)/2 # particle initial z position

N_markers = 1 # number of particle markers - need to offset initial positions for anything to happen when increasing this from 1

stop_dist = r_particles[0] # minimum distance from the particle to the wall before the simulation is stopped


# IBM
IB_kernel = ['standard gaussian', 'dual gaussian'][1] # force distribution function to use
f_dist_width = 2 # width of the surface gaussian force distribution function [lattice points] (only for dual gaussian IB kernel)


# Fluid
nu = 1/6 # kinematic viscosity [m2 s-1]
rho_0 = 1.0 # initial density [kg m-3]
mu = rho_0*nu # dynamic viscosity [kg m-1 s-1]


# Diffusion - would probably be defined by a temperature
F_gravity = 0.3 # gravity force magnitude
lubrication_threshold = 2/3


#%% Define Obstacle Geometry
@nb.jit(nopython=True, parallel=True, fastmath=True)
def initialise_obstacle(obstacle, Nx, Ny, Nz):
    """
    If you'd like a solid obstacle within the domain, change this function to 
    define it. There can be multiple obstacles. Remember that only the fluid 
    sees the obstacle; the obstacle-particle interactions will have to be 
    handled by something else.
    
    Fluid cells where obstacle == True act as solid objects; the fluid is not
    updated at these points and they act as no-slip boundaries.
    
    Note that this is a simplistic True/False obstacle representation (curved 
    obstacles will thus be approximated by a stepped surface) - increase the 
    mesh resolution for a more accurate representation.
    
    Haven't tested with placing the obstacle on the domain boundaries.
    """
    
    # Random rectangular obstacle
    for i in nb.prange(Nx):
        i = np.int64(i)
        if 0.1*(Nx-1) <= i <= 0.4*(Nx-1):
            for j in range(Ny):
                if 0.3*(Ny-1) <= j <= 0.7*(Ny-1):
                    for k in range(Nz):
                        if 0.2*(Nz-1) <= k <= 0.8*(Nz-1):
                            obstacle[i, j, k] = True
    
    # No obstacle/s
    # for i in nb.prange(Nx):
    #     i = np.int64(i)
    #     for j in range(Ny):
    #         for k in range(Nz):
    #             obstacle[i, j, k] = False
    
    # Do nothing
    return None



#%% Solver Parameters
Nt = int(sim_time) # number of time steps (since dt=1)
outevery = int(Nt/N_outputs) # generate an output every this many steps
outevery = 10
# outevery = 1


domain_dims = [Nx, Ny, Nz]
n_lattice = Nx*Ny*Nz # total n.o. fluid nodes

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
        marker_f[m, 0] += np.random.normal(0, 1)*F_brownian_scale
        marker_f[m, 1] += np.random.normal(0, 1)*F_brownian_scale
        marker_f[m, 2] += np.random.normal(0, 1)*F_brownian_scale

        
# Lubrication corrrection
@nb.jit(nopython=True, fastmath=True)
def lubrication_correction_forcing_ini(N_markers, marker_pos, marker_vel, marker_radii, 
                            marker_f, wall_y, lubrication_threshold, F_gravity, h_floor=1e-4):
    """
    Applique la gravité (poids apparent, direction -z) et la force de répulsion 
    à courte portée entre particules (Glowinski et al. via Cao et al., Eq. 16).
    
    Applique également une correction de lubrification particule-paroi pour le
    mur situé en y = wall_y (paroi inférieure, normale +y). Le mur est traité
    comme une sphère de rayon infini (R_eff -> R_particule) et de vitesse nulle.
    """

    # We apply the force for each pair i-j and j-i of particles
    for m in range(N_markers):
        for j in range(N_markers):
            if m == j:
                continue
            
            dx = marker_pos[m, 0] - marker_pos[j, 0]
            dy = marker_pos[m, 1] - marker_pos[j, 1]
            dz = marker_pos[m, 2] - marker_pos[j, 2]
            dist = (dx*dx + dy*dy + dz*dz)**0.5
            
            h = dist - marker_radii[m] - marker_radii[j]  # h is the gap between the 2 surfaces
            
            if h <= lubrication_threshold * 1:
                R_eff = marker_radii[m]*marker_radii[j]/(marker_radii[m]+marker_radii[j])
                U12 = marker_vel[m, :] - marker_vel[j, :]
                UR = np.dot(U12, np.array([dx, dy, dz])) / dist
                coef = -6 * np.pi * nu * R_eff**2 * (1/h - 1/lubrication_threshold) * UR
                
                marker_f[m, 0] += coef * dx/dist
                marker_f[m, 1] += coef * dy/dist
                marker_f[m, 2] += coef * dz/dist
    
    # Wall lubrication correction (bottom wall, y = wall_y, unit normal (0, 1, 0))
    for m in range(N_markers):
        h_wall = (marker_pos[m, 1] - marker_radii[m]) - wall_y # gap between the particle surface and the wall
        h_min = 0.01

        if h_wall <= h_min:
            # Spring force to keep the particle near the wall
            marker_f[m, 1] += F_gravity - 0.4*h_wall
            print("elastic : ", h_wall)

        elif h_wall <= lubrication_threshold:
            R_eff = marker_radii[m]  # sphere-wall: the wall's "radius" -> infinity, so R_eff -> marker_radii[i]
            UR = marker_vel[m, 1]  # the wall is fixed, so the relative velocity along the normal is just marker_vel[i, 1]
            coef = -6 * np.pi * nu * R_eff**2 * (1/h_wall - 1/lubrication_threshold)
            marker_f[m, 1] += coef  # force is directed along the wall normal (0, 1, 0)
            print(h_wall, coef, UR)

    return h_wall

@nb.jit(nopython=True, fastmath=True)
def lubrication_correction_forcing(N_markers, marker_pos, marker_vel, marker_radii,
                            marker_f, wall_y, lubrication_threshold,
                            rho_0, nu, dt=1.0, h_floor=1e-4):
    """
    Lubrication correction, force-based, avec saturation analogue à un update
    implicite (backward Euler) mais exprimée entièrement via marker_f.
    """

    # --- Particle-particle lubrication ---
    for m in range(N_markers):
        for j in range(N_markers):
            if m == j:
                continue

            dx = marker_pos[m, 0] - marker_pos[j, 0]
            dy = marker_pos[m, 1] - marker_pos[j, 1]
            dz = marker_pos[m, 2] - marker_pos[j, 2]
            dist = (dx*dx + dy*dy + dz*dz)**0.5

            h = dist - marker_radii[m] - marker_radii[j]
            h_eff = max(h, h_floor)

            if h_eff <= lubrication_threshold:
                a1, a2 = marker_radii[m], marker_radii[j]
                R_eff = a1*a2/(a1+a2)
                # masse réduite effective (approx du volume de fluide déplacé)
                m1 = (4.0/3.0)*np.pi*rho_0*a1**3
                m2 = (4.0/3.0)*np.pi*rho_0*a2**3
                m_eff = m1*m2/(m1+m2)

                U12 = marker_vel[m, :] - marker_vel[j, :]
                UR = np.dot(U12, np.array([dx, dy, dz])) / dist

                xi = 6*np.pi*nu*rho_0*R_eff**2 * (1.0/h_eff - 1.0/lubrication_threshold)
                xi_eff = xi / (1.0 + xi*dt/m_eff)   # <-- saturation implicite
                coef = -xi_eff * UR

                marker_f[m, 0] += coef * dx/dist
                marker_f[m, 1] += coef * dy/dist
                marker_f[m, 2] += coef * dz/dist

    # --- Wall lubrication (bottom wall, normal +y) ---
    h_wall_out = 0.0
    for m in range(N_markers):
        h_wall = (marker_pos[m, 1] - marker_radii[m]) - wall_y
        h_eff = max(h_wall, h_floor)
        h_wall_out = h_wall

        if h_eff <= lubrication_threshold:
            R_eff = marker_radii[m]
            m_eff = (4.0/3.0)*np.pi*rho_0*R_eff**3

            UR = marker_vel[m, 1]  # mur fixe

            xi = 6*np.pi*nu*rho_0*R_eff**2 * (1.0/h_eff - 1.0/lubrication_threshold)
            xi_eff = xi / (1.0 + xi*dt/m_eff)   # <-- saturation implicite

            marker_f[m, 1] += -xi_eff * UR

    return h_wall_out


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


"""# IBM Setup
if IB_kernel == 'standard gaussian':
    r_gaus = r_particle
    sigma, A, r_cutoff = gaus_consts(r_gaus)
    r_cutoff_outer = r_cutoff
    r_cutoff_inner = 0
    dist_func = gaus_dist
elif IB_kernel == 'dual gaussian':
    r_gaus = max(r_particle-f_dist_width/2, 0) # surface gaussian force distribution function radial location
    sigma, A, r_cutoff = dual_gaus_consts(f_dist_width, r_gaus)
    r_cutoff_outer = r_gaus+r_cutoff
    r_cutoff_inner = max(r_gaus-r_cutoff, 0)
    dist_func = dual_gaus_dist
r_cutoff_outer_sq = r_cutoff_outer*r_cutoff_outer
r_cutoff_inner_sq = r_cutoff_inner*r_cutoff_inner"""

# --- IBM Setup pour particules hétérogènes ---

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

# Rayon de coupure maximal pour la réservation du voisinage (tableau 3D)
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
init_marker_pos[:, 0] = cx_particle
init_marker_pos[:, 1] = cy_particle
init_marker_pos[:, 2] = cz_particle
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
initialise_obstacle(obstacle, Nx, Ny, Nz)



#%% Force Distribution Function Analysis
if show_gaus_dist:
    plot_gaus_dist(Ny, dist_func, r_particles[0], f_dist_width, r_gaus, sigma, A, r_cutoff, r_cutoff_outer, IB_kernel, N_markers, marker_pos)



"""#%% Solver Loop
print(f'Particle Diameter: {D_particle}\nSpacing Multiplier: {spacing_mutl}\nNx, Ny, Nz: ({Nx}, {Ny}, {Nz})\nNumber of Lattice Points: {n_lattice}')
print(f'Boundary Representation: immersed boundary method ({IB_kernel})')
if IB_kernel == 'dual gaussian':
    print(f'Distribution Width: {f_dist_width}')
print(f'Kinematic Viscosity: {nu:.4f}\nInitial Fluid Density: {rho_0}\nForcing Scale: {F_brownian_scale}')
print(f'Relaxation Factor (BGK): {tau:.4f}\n')"""


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

def plot_settling_fig8(h_walls, a_hy, nu, dt=1.0, step=5):
    """
    Reproduit un graphique semblable à la Fig. 8 de Nguyen & Ladd (2002).

    Parameters
    ----------
    h_walls : array-like
        Gap particule-mur h(t) à chaque itération (mêmes unités que a_hy, ex. Δx).
    a_hy : float
        Rayon hydrodynamique de la particule (même unité que h_walls).
    nu : float
        Viscosité cinématique de la simulation (en unités lattice, Δx²/Δt).
    dt : float, optional
        Pas de temps entre deux valeurs successives de h_walls (défaut 1, en Δt).
    step : int, optional
        On ne garde qu'une valeur toutes les `step` itérations (défaut 5).
    """
    h_walls = np.asarray(h_walls, dtype=float)
    n_iter = len(h_walls)
    t = np.arange(n_iter) * dt

    # Adimensionnement
    h_nondim = h_walls / a_hy
    t_nondim = nu * t / a_hy**2

    # Sous-échantillonnage : une valeur tous les steps
    h_plot = h_nondim[::step]
    t_plot = t_nondim[::step]

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(t_plot, h_plot, 'o', mfc='none', mec='black', label='Simulation')

    ax.set_xlabel(r'$\nu t / a_{hy}^{2}$')
    ax.set_ylabel(r'$h / a_{hy}$')
    ax.set_title('Settling of a sphere onto a wall')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_figname, dpi=300)
    plt.show()

    return fig, ax

def run_diff_sim(pops_pre, pops_post, F, rho, u, u_mag_sq, obstacle, N_markers, marker_pos, marker_vel, marker_f, marker_nh, marker_nh_size, 
                 Nt, F_gravity, Nx, Ny, Nz, n_lattice, r_cutoff_outer, r_cutoff_outer_sq, r_cutoff_inner_sq, dist_func, r_gaus, sigma, A, stopping_lims, 
                 inv_cs2, inv_2cs2, inv_cs4, inv_2cs4, omega, omega_prime, omega_S_coeff, N_vels, w, c, inv_cx_indx, inv_cy_indx, inv_cz_indx, inv_c_indx, BCs, skip_stop_check, 
                 live_flow_plot, outevery):
    
    break_cond = False
    int_err = 0.0
    u_mag_sq_max = 0.0
    
    marker_pos_hist = np.empty((N_markers, Nt, 3), dtype=np.float64)
    marker_vel_hist = np.empty_like(marker_pos_hist)
    marker_f_hist = np.empty_like(marker_pos_hist)
    fluid_mass_hist = np.empty(Nt, dtype=np.float64)

    iterations = tqdm.tqdm(range(Nt)) # initialise progress bar
    start_time = time.perf_counter()

    frames = []
    h_walls = []
    for t in iterations:
        
        if np.isnan(u_mag_sq).any():
            raise RuntimeError(f'Unrealistic velocities: t={t}')


        # Gravity for the markers
        for m in range(N_markers):
            marker_f[m, 0] = 0.0
            marker_f[m, 1] = -F_gravity # apply gravity in the -y direction
            marker_f[m, 2] = 0.0 

        # Calculate marker forces
        #brownian_forcing(F_brownian_scale, N_markers, marker_f)
        h_wall = lubrication_correction_forcing(N_markers, marker_pos, marker_vel, marker_radii, marker_f, 
                                                wall_y, lubrication_threshold, rho_0, nu, dt=1.0, h_floor=1e-4)
        h_walls.append(h_wall)
        
        # Calculate forcing due to IB markers
        int_err = IB_force_density(Nx, Ny, Nz, r_cutoff_outer, r_cutoff_outer_sq, r_cutoff_inner_sq, F, 
                                  dist_func, r_gaus, sigma, A, N_markers, marker_pos, marker_f, marker_nh, marker_nh_size, int_err)
        
        # Calculate fluid properties, perform collisions, and stream populations
        update_LBM_pops_closed(pops_pre, pops_post, F, rho, u, u_mag_sq, obstacle, Nx, Ny, Nz, 
                               inv_cs2, inv_2cs2, inv_cs4, inv_2cs4, omega, omega_prime, omega_S_coeff, 
                               N_vels, w, c, inv_cx_indx, inv_cy_indx, inv_cz_indx, inv_c_indx, BCs)
        
        # Interpolate boundary marker velocities
        interpolate_marker_vels(u, N_markers, marker_vel, marker_nh, marker_nh_size)
        
        # Integrate boundary markers
        marker_pos += marker_vel # since dt=1
        
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
            #if not (inbounds_x and inbounds_y and inbounds_z):
            #    break_cond = True
            #    save_data = True
            #    break
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
            
            ax.set_xlim([0, Nx-1])
            ax.set_ylim([0, Ny-1])
            ax.set_title(f'3D Particle Diffusion - {IB_kernel} IBM\nZ Position = {z_slice}, t = {t}')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            plt.tight_layout()
            
            # Enregistrement de l'image en mémoire pour le GIF
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
            frames.append(image)
            plt.close(fig)
        
        if break_cond:
            break
    
    if live_flow_plot and len(frames) > 0:
            imageio.mimsave(f'{output_gif_filename}.gif', frames, fps=25, loop=0)

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
    
    plot_settling_fig8(h_walls, r_particles[0], nu, dt=1.0, step=20)

    return marker_pos_hist, marker_vel_hist, marker_f_hist, fluid_mass_hist, simtime_reached



sim_res = run_diff_sim(pops_pre, pops_post, F, rho, u, u_mag_sq, obstacle, N_markers, marker_pos, marker_vel, marker_f, marker_nh, marker_nh_size, 
                       Nt, F_gravity, Nx, Ny, Nz, n_lattice, r_cutoff_outer, r_cutoff_outer_sq, r_cutoff_inner_sq, dist_func, r_gaus, sigma, A, stopping_lims, 
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








