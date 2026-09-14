#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A starting point for simulating the brownian motion of a spherical particle
based on the Langevin approach. 
For the fluctuating hydrodynamics approach, no force is applied to the particle; 
instead, it is simply carried along with the fluid. The fluctuating stress term 
will need to be added to the LBGK collision equation.

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
import sys
import os

base_path = os.path.dirname(os.path.dirname(__file__))
lib_path = os.path.join(base_path, "fluid solver code")

sys.path.append(lib_path)

from forced_LBGK_lib import get_LBM_consts, initialise_pops, update_LBM_pops_closed#, update_LBM_pops_closed_combined
from multi_marker_IBM_lib import gaus_consts, gaus_dist, dual_gaus_consts, dual_gaus_dist, plot_gaus_dist, IB_force_density, interpolate_marker_vels

max_mem_avail = 12.0e9 # maximum available memory [bytes]



#%% Problem Input Parameters

# Graphing and Outputs
show_gaus_dist = True # plot the y distribution of the force distribution function
live_flow_plot = True # plot the flow field during the simulation
N_outputs = 200 # n.o. times to plot the solution field (only if live_flow_plot=True)
show_mass = True # plot the total fluid mass over the simulation duration - can be useful for identifying instabilities (should remain constant)


# Geometry
D_particle = 7 # number of lattice points across the particle diameter
r_particle = D_particle/2 # particle radius
spacing_mutl = 10 # control the spacing between the particle and the domain walls

Nx = int(spacing_mutl*D_particle+1) # simulation domain length
Ny = Nx # simulation domain height
Nz = Nx # simulation domain depth
n_lattice = Nx*Ny*Nz # total n.o. fluid nodes

cx_particle = (Nx-1)/2 # particle initial x position
cy_particle = (Ny-1)/2 # particle initial y position
cz_particle = (Nz-1)/2 # particle initial z position
N_markers = 2 # number of particle markers - need to offset initial positions for anything to happen when increasing this from 1

stop_dist = D_particle # minimum distance from the particle to the wall before the simulation is stopped
stopping_lims = [[stop_dist, Nx-1-stop_dist], [stop_dist, Ny-1-stop_dist], [stop_dist, Nz-1-stop_dist]]


# IBM
IB_kernel = ['standard gaussian', 'dual gaussian'][1] # force distribution function to use
f_dist_width = 2 # width of the surface gaussian force distribution function [lattice points] (only for dual gaussian IB kernel)


# Fluid
nu = 1/6 # kinematic viscosity [m2 s-1]
rho_0 = 1.0 # initial density [kg m-3]
mu = rho_0*nu # dynamic viscosity [kg m-1 s-1]


# Gravité et répulsion 
rho_p_over_rho_f = 1.14  # rapport de densité particule/fluide (Cao et al.)
rho_p = rho_p_over_rho_f * rho_0

g_z = -2.0e-3  # accélération de gravité [lattice units], négative = vers le bas (-z)


#%% Solver Parameters
sim_time = 1000 # simulation time [s] - adjust accordingly
Nt = int(sim_time) # number of time steps (since dt=1)

outevery = int(Nt/N_outputs) # generate an output every this many steps
# outevery = 2

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

est_mem_req = 2.1*Nx*Ny*Nz*N_vels*8.0 # very rough estimation
if est_mem_req > max_mem_avail:
    cont = input(f'WARNING: Estimated memory requirements ({(est_mem_req*1e-9):.2f} GB) exceeds the maximum available memory specified ({(max_mem_avail*1e-9):.2f} GB). Continue? (y/n)')
    if cont != 'y':
        raise MemoryError()
else:
    print(f'Estimated Memory Requirements: {(est_mem_req*1e-9):.6f} GB\n')


# Repulsion
@nb.jit(nopython=True, fastmath=True)
def lubrication_correction_forcing(t, N_markers, marker_pos, marker_vel, marker_radii, F_gravity_y, 
                              lubrication_threshold, epsilon_stiff, c_ij, marker_f):
    """
    Applique la gravité (poids apparent, direction -z) et la force de répulsion 
    à courte portée entre particules (Glowinski et al. via Cao et al., Eq. 16).
    """
    
    if t < 50:
        marker_f[0, 0] = -10*F_gravity_y
        marker_f[0, 1] = 0.0 #F_gravity_y
        marker_f[0, 2] = 0.0 
    else:
        marker_f[0, 0] = 0.0
        marker_f[0, 1] = 0.0 #F_gravity_y
        marker_f[0, 2] = 0.0 

    """# Initialization : gravity only
    for m in range(1, N_markers):
        marker_f[m, 0] = 0.0
        marker_f[m, 1] = 0.0 #F_gravity_y
        marker_f[m, 2] = 0.0
        #print("gravity : ", F_gravity_y)
    
    # Répulsion par paire (N_markers petit -> boucle simple, pas de race condition)
    for i in range(N_markers):
        for j in range(N_markers):
            if i == j:
                continue
            
            dx = marker_pos[i, 0] - marker_pos[j, 0]
            dy = marker_pos[i, 1] - marker_pos[j, 1]
            dz = marker_pos[i, 2] - marker_pos[j, 2]
            dist = (dx*dx + dy*dy + dz*dz)**0.5
            
            h = dist - marker_radii[i] - marker_radii[j]  # h is the gap between the 2 surfaces
            
            if h <= lubrication_threshold * 1:
                R_eff = marker_radii[i]*marker_radii[j]/(marker_radii[i]+marker_radii[j])
                U12 = marker_vel[i, :] - marker_vel[j, :]
                UR = np.dot(U12, np.array([dx, dy, dz])) / dist
                coef = -6 * np.pi * nu * R_eff**2 * (1/h - 1/lubrication_threshold) * UR
                #print(coef)
                #print("h: ", {h})
                
                marker_f[i, 0] += coef * dx/dist
                marker_f[i, 1] += coef * dy/dist
                marker_f[i, 2] += coef * dz/dist"""





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


# IBM Setup
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
r_cutoff_inner_sq = r_cutoff_inner*r_cutoff_inner


# IBM Arrays
init_marker_pos = np.empty((N_markers, 3), dtype=np.float64) # initial marker positions
"""init_marker_pos[:, 0] = cx_particle
init_marker_pos[:, 1] = cy_particle
init_marker_pos[:, 2] = cz_particle"""
offset = 3 * D_particle  # à ajuster : doit être assez grand pour éviter le recouvrement des noyaux gaussiens

init_marker_pos[0, 0] = cx_particle - offset/2
init_marker_pos[0, 1] = cy_particle
init_marker_pos[0, 2] = cz_particle

init_marker_pos[1, 0] = cx_particle + offset/2
init_marker_pos[1, 1] = cy_particle #+ 1 * D_particle
init_marker_pos[1, 2] = cz_particle
"""offset = 3 * D_particle  # à ajuster : doit être assez grand pour éviter le recouvrement des noyaux gaussiens

init_marker_pos[0, 0] = cx_particle
init_marker_pos[0, 1] = cy_particle
init_marker_pos[0, 2] = cz_particle

init_marker_pos[1, 0] = cx_particle + 1.1 * D_particle
init_marker_pos[1, 1] = cy_particle + 2 * D_particle
init_marker_pos[1, 2] = cz_particle"""
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



#%% Force Distribution Function Analysis
if show_gaus_dist:
    plot_gaus_dist(Ny, dist_func, r_particle, f_dist_width, r_gaus, sigma, A, r_cutoff, r_cutoff_outer, IB_kernel, N_markers, marker_pos)



#%% Solver Loop
print(f'Particle Diameter: {D_particle}\nSpacing Multiplier: {spacing_mutl}\nNx, Ny, Nz: ({Nx}, {Ny}, {Nz})\nNumber of Lattice Points: {n_lattice}')
print(f'Boundary Representation: immersed boundary method ({IB_kernel})')
if IB_kernel == 'dual gaussian':
    print(f'Distribution Width: {f_dist_width}')
#print(f'Kinematic Viscosity: {nu:.4f}\nInitial Fluid Density: {rho_0}\nForcing Scale: {F_brownian_scale}')
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



def run_diff_sim(pops_pre, pops_post, F, rho, u, u_mag_sq, N_markers, marker_pos, marker_vel, marker_f, marker_nh, marker_nh_size, 
                 Nt, Nx, Ny, Nz, n_lattice, r_cutoff_outer, r_cutoff_outer_sq, r_cutoff_inner_sq, dist_func, r_gaus, sigma, A, stopping_lims, 
                 inv_cs2, inv_2cs2, inv_cs4, inv_2cs4, omega, omega_prime, omega_S_coeff, N_vels, w, c, inv_cx_indx, inv_cy_indx, inv_cz_indx, live_flow_plot, outevery):
    
    break_cond = False
    int_err = 0.0
    u_mag_sq_max = 0.0
    
    marker_pos_hist = np.empty((N_markers, Nt, 3), dtype=np.float64)
    marker_vel_hist = np.empty_like(marker_pos_hist)
    marker_f_hist = np.empty_like(marker_pos_hist)
    fluid_mass_hist = np.empty(Nt, dtype=np.float64)

    V_particle = (4.0/3.0) * np.pi * r_particle**3  # volume de la particule (sphère)
    F_gravity_y = rho_p * V_particle * g_z  # force nette (poids - Archimède), par particule
    
    lubrication_threshold = 0.67  # for nu*=1/6 , Ladd suggests 2/3
    epsilon_stiff = 1.0  # rigidité ε = (dx)^2 = 1
    c_ij = abs(F_gravity_y)  # échelle = force de flottabilité, comme dans l'article

    marker_radii = np.full(N_markers, r_particle, dtype=np.float64)  # rayon physique de chaque particule

    iterations = tqdm.tqdm(range(Nt)) # initialise progress bar
    start_time = time.perf_counter()
    for t in iterations:
        
        if np.isnan(u_mag_sq).any():
            raise RuntimeError(f'Unrealistic velocities: t={t}')
        
        
        # Calculate marker forces
        lubrication_correction_forcing(t, N_markers, marker_pos, marker_vel, marker_radii, F_gravity_y, 
                            lubrication_threshold, epsilon_stiff, c_ij, marker_f)
        
        # Calculate forcing due to IB markers
        int_err = IB_force_density(Nx, Ny, Nz, r_cutoff_outer, r_cutoff_outer_sq, r_cutoff_inner_sq, F, 
                                   dist_func, r_gaus, sigma, A, N_markers, marker_pos, marker_f, marker_nh, marker_nh_size, int_err)
        
        # Calculate fluid properties, perform collisions, and stream populations
        # update_LBM_pops_closed_combined(t, pops_pre, pops_post, F, rho, u, u_mag_sq, Nx, Ny, Nz, 
        #                                 inv_cs2, inv_2cs2, inv_cs4, inv_2cs4, omega, omega_prime, omega_S_coeff, 
        #                                 N_vels, w, c, inv_cx_indx, inv_cy_indx, inv_cz_indx)
        update_LBM_pops_closed(pops_pre, pops_post, F, rho, u, u_mag_sq, Nx, Ny, Nz, 
                               inv_cs2, inv_2cs2, inv_cs4, inv_2cs4, omega, omega_prime, omega_S_coeff, 
                               N_vels, w, c, inv_cx_indx, inv_cy_indx, inv_cz_indx)
        
        # Interpolate boundary marker velocities
        interpolate_marker_vels(u, N_markers, marker_vel, marker_nh, marker_nh_size)
        
        # Integrate boundary markers
        """marker_vel[0, 0] = 0.0
        marker_vel[0, 1] = 0.0
        marker_vel[0, 2] = 0.0"""
        marker_pos += marker_vel # since dt=1

        #marker_pos[0, :] = init_marker_pos[0, :] # keep the first marker fixed in space (for testing)
        
        
        # Save marker data
        save_marker_data(t, marker_pos_hist, marker_vel_hist, marker_f_hist, N_markers, marker_pos, marker_vel, marker_f)
        
        ## Integrate and save fluid mass
        fluid_mass_hist[t] = np.sum(rho)
        
        
        # Check stopping criteria
        for m in range(N_markers):
            np.int64(m)
            inbounds_x = stopping_lims[0][0] <= marker_pos[m, 0] <= stopping_lims[0][1]
            inbounds_y = stopping_lims[1][0] <= marker_pos[m, 1] <= stopping_lims[1][1]
            inbounds_z = stopping_lims[2][0] <= marker_pos[m, 2] <= stopping_lims[2][1]
            if not (inbounds_x and inbounds_y and inbounds_z):
                break_cond = True
                save_data = True
                break
            elif m == N_markers-1:
                save_data = (t%outevery == 0) or (t == Nt-1)
        
        
        # Output flow field at particle cross-section
        if save_data and live_flow_plot:
            u_mag_sq_curr = np.max(u_mag_sq)
            if u_mag_sq_curr > u_mag_sq_max:
                u_mag_sq_max = u_mag_sq_curr
            
            m = 0 # only plot the first marker
            z_slice = min(max(int(round(marker_pos[m, 2])), 0), Nz-1)
            
            plt.figure(figsize=(5, 4))
            im = plt.imshow(np.sqrt(u_mag_sq[:, :, z_slice]).T, cmap='viridis', origin='lower', vmin=0, vmax=u_mag_sq_max**0.5)
            # im = plt.imshow(u[:, :, z_slice, 0].T, cmap='viridis', origin='lower')
            # im = plt.imshow(rho[:, :, z_slice].T, cmap='viridis', origin='lower')
            # im = plt.imshow(F[:, :, z_slice, 0].T, cmap='viridis', origin='lower')
            plt.colorbar(im, label='Velocity Magnitude')
            
            for m in range(N_markers):
                plt.plot(marker_pos_hist[m, :t+1, 0], marker_pos_hist[m, :t+1, 1], 'r', alpha=0.5)
                plt.plot(init_marker_pos[m, 0], init_marker_pos[m, 1], 'xr')
                
                circle = plt.Circle((marker_pos[m, 0], marker_pos[m, 1]), r_particle, color='red', fill=False, linewidth=1.5)
                plt.gca().add_patch(circle)
            
            plt.xlim([0, Nx-1])
            plt.ylim([0, Ny-1])
            
            plt.title(f'3D Particle Diffusion - {IB_kernel} IBM\nZ Position = {z_slice}, t = {t}')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.tight_layout()
            # plt.savefig(f'{t}_diff_ani.png')
            plt.show()
        
        if break_cond:
            break

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



sim_res = run_diff_sim(pops_pre, pops_post, F, rho, u, u_mag_sq, N_markers, marker_pos, marker_vel, marker_f, marker_nh, marker_nh_size, 
                       Nt, Nx, Ny, Nz, n_lattice, r_cutoff_outer, r_cutoff_outer_sq, r_cutoff_inner_sq, dist_func, r_gaus, sigma, A, stopping_lims, 
                       inv_cs2, inv_2cs2, inv_cs4, inv_2cs4, omega, omega_prime, omega_S_coeff, N_vels, w, c, inv_cx_indx, inv_cy_indx, inv_cz_indx, live_flow_plot, outevery)

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








