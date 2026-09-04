#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A companion script to IB_LBM_particle_diff_example.py, but using the finite 
volume method instead of the lattice Boltzmann method. 
The LBM is generally much faster and better suited to the problem, so the 
investigation should be carried out using the LBM (and also so that the 
resulting methods can be easily implemented with existing code). However, the 
LBM can be non-intuitive and more prone to instabilities, so this script enables 
the testing of diffusion methods before LB implementation if needed.

The script assumes that diffusion dominates advection (and thus the flow is
approximately Stokes flow). The only area where this assumption is used is when
calculating the time step duration for stability (currently only considers the 
Fourier condition - will need to also consider the CFL condition for greater
advection terms).
The maximum number of poisson pressure iterations (Nit) can be decreased to
gain computational efficiency at the expense of accuracy. Since the simulation
does not reach a steady state, there are many poisson iterations for every step
(unlike for steady simulations, where the number of iterations decreases
dramatically when steady state is reached) - this makes the FVM very inefficient
for transient simulations such as this.


Created on Mon Aug 31 11:21:56 2026
Author: Max Robbins
"""

import time
import tqdm
import warnings
import numpy as np
import numba as nb
import matplotlib.pyplot as plt

from forced_FVM_lib import compute_tentative_velocity, pressure_poisson, correct_velocity, set_BCs
from multi_marker_IBM_lib import gaus_consts, gaus_dist, dual_gaus_consts, dual_gaus_dist, plot_gaus_dist, IB_force_density, interpolate_marker_vels

max_mem_avail = 12.0e9 # maximum available memory [bytes]



#%% Problem Input Parameters

# Graphing and Outputs
show_gaus_dist = True # plot the y distribution of the force distribution function
live_flow_plot = True # plot the flow field during the simulation
N_outputs = 10 # n.o. times to plot the solution field (only if live_flow_plot=True)


# Particle/Fluid Geometry and Discretisation
dx = 1.0 # cell size [m]
dy, dz = dx, dx # assume dx=dy=dz [m]
cell_vol = dx*dy*dz # cell volume [m3]

N_D_particle = 7 # number of lattice points across the particle diameter
D_particle = N_D_particle*dx # particle diameter [m]
r_particle = D_particle/2 # particle radius [m]
spacing_mutl = 10 # control the spacing between the particle and the domain walls

Nx = int(spacing_mutl*N_D_particle+1) # simulation domain length [lattice points]
Ny = Nx # simulation domain height [lattice points]
Nz = Nx # simulation domain depth [lattice points]
n_lattice = Nx*Ny*Nz # total n.o. fluid nodes
n_lattice_internal = (Nx-2)*(Ny-2)*(Nz-2) # number of fluid lattice points, excluding boundary nodes

L_x = (Nx-1)*dx # domain x length [m]
L_y = (Ny-1)*dy # domain y length [m]
L_z = (Nz-1)*dz # domain z length [m]

cx_particle = (Nx-1)/2 # particle initial x position [lattice points]
cy_particle = (Ny-1)/2 # particle initial y position [lattice points]
cz_particle = (Nz-1)/2 # particle initial z position [lattice points]
N_markers = 1 # number of particle markers - need to offset initial positions for anything to happen when increasing this from 1

stop_dist = N_D_particle # minimum distance from the particle to the wall before the simulation is stopped [lattice points]
stopping_lims = [[stop_dist, Nx-1-stop_dist], [stop_dist, Ny-1-stop_dist], [stop_dist, Nz-1-stop_dist]] # [lattice points]


# IBM
IB_kernel = ['standard gaussian', 'dual gaussian'][1] # force distribution function to use
f_dist_width = 2 # width of the surface gaussian force distribution function [lattice points] (only for dual gaussian IB kernel)


# Fluid
nu = 1/6 # kinematic viscosity [m2 s-1]
rho_0 = 1.0 # initial density [kg m-3]
mu = rho_0*nu # dynamic viscosity [kg m-1 s-1]


# Diffusion - would probably be defined by a temperature
F_brownian_scale = 5 # sample parameter for example only



#%% Solver Parameters

# Fluid diffusion (Fourier) constraint
Fo = 0.25
dt_d = Fo/(nu*(1/(dx**2) + 1/(dy**2) + 1/(dz**2))) # time step - Fourier condition

dt = dt_d # assumes Stokes flow, i.e. diffusion dominates advection

sim_time = 1000 # simulation time [s] - adjust accordingly
Nt = int(sim_time/dt) # number of time steps
Nit = 500 # maximum iterations for pressure poisson equation - can decrease for faster computation

outevery = int(Nt/N_outputs) # generate an output every this many steps
# outevery = 20


est_mem_req = 9*n_lattice*8.0 # very rough estimation
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
    """
    for m in nb.prange(N_markers):
        np.int64(m)
        marker_f[m, 0] = np.random.normal(0, 1)*F_brownian_scale
        marker_f[m, 1] = np.random.normal(0, 1)*F_brownian_scale
        marker_f[m, 2] = np.random.normal(0, 1)*F_brownian_scale



#%% Initialisation Functions
def initialise_fluid_arrays(p, u, u_star, u_star_div, u_mag_sq, F):
    p.fill(0.0)
    u.fill(0.0)
    u_star.fill(0.0)
    u_star_div.fill(0.0)
    u_mag_sq.fill(0.0)
    F.fill(0.0)


def initialise_IBM(init_marker_pos, marker_pos, marker_vel, marker_f):
    for dim in range(3):
        marker_pos[:, dim] = init_marker_pos[:, dim].copy()
        marker_vel[:, dim] = 0.0
        marker_f[:, dim] = 0.0



#%% Setup and Memory Allocation

# Fluid Arrays
p = np.empty((Nx, Ny, Nz), dtype=np.float64) # pressures [Pa]
u = np.empty((Nx, Ny, Nz, 3), dtype=np.float64) # velocities [m s-1]
u_star = np.empty_like(u) # tentative velocities [m s-1]
u_mag_sq = np.empty_like(p) # squared velocity magnitudes [m2 s-2]
u_star_div = np.empty_like(u_mag_sq) # divergence of the tentative velocity field
F = np.empty_like(u) # body force densities [N m-3]


# IBM Setup
if IB_kernel == 'standard gaussian':
    r_gaus = r_particle/dx
    sigma, A, r_cutoff = gaus_consts(r_gaus)
    r_cutoff_outer = r_cutoff
    r_cutoff_inner = 0
    dist_func = gaus_dist
elif IB_kernel == 'dual gaussian':
    r_gaus = max(r_particle/dx-f_dist_width/2, 0) # surface gaussian force distribution function radial location
    sigma, A, r_cutoff = dual_gaus_consts(f_dist_width, r_gaus)
    r_cutoff_outer = r_gaus+r_cutoff
    r_cutoff_inner = max(r_gaus-r_cutoff, 0)
    dist_func = dual_gaus_dist
r_cutoff_outer_sq = r_cutoff_outer*r_cutoff_outer
r_cutoff_inner_sq = r_cutoff_inner*r_cutoff_inner


# IBM Arrays
init_marker_pos = np.empty((N_markers, 3), dtype=np.float64) # initial marker positions [lattice points]
init_marker_pos[:, 0] = cx_particle
init_marker_pos[:, 1] = cy_particle
init_marker_pos[:, 2] = cz_particle
marker_pos = np.empty_like(init_marker_pos) # Lagrangian boundary marker positions [lattice points]
marker_vel = np.empty_like(marker_pos) # Lagrangian boundary marker velocities [m s-1]
marker_f = np.empty_like(marker_pos) # Lagrangian boundary marker forces [N]


# Allocate marker neighbourhood arrays
marker_nh_size = np.empty(N_markers, dtype=np.int64)
max_marker_neighbours = int((2*(np.ceil(r_cutoff_outer)+1))**3)
marker_nh = np.empty((N_markers, 5, max_marker_neighbours), dtype=np.float64) # x, y, z, mag, weighting


# Initialise Arrays
initialise_fluid_arrays(p, u, u_star, u_star_div, u_mag_sq, F)
initialise_IBM(init_marker_pos, marker_pos, marker_vel, marker_f)



#%% Force Distribution Function Analysis
if show_gaus_dist:
    plot_gaus_dist(Ny, dist_func, r_particle/dx, f_dist_width, r_gaus, sigma, A, r_cutoff, r_cutoff_outer, IB_kernel, N_markers, marker_pos)



#%% Solver Loop
print(f'Particle Diameter: {N_D_particle} lattice points, {D_particle} m\nSpacing Multiplier: {spacing_mutl}\nNx, Ny, Nz: ({Nx}, {Ny}, {Nz})\nNumber of Lattice Points: {n_lattice}')
print(f'Boundary Representation: immersed boundary method ({IB_kernel})')
if IB_kernel == 'dual gaussian':
    print(f'Distribution Width: {f_dist_width} lattice points')
print(f'Kinematic Viscosity: {nu:.4f} m2 s-1\nInitial Fluid Density: {rho_0} kg m-3\nForcing Scale: {F_brownian_scale}')
print(f'Time step: {dt:.4f} s\n')


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



def run_diff_sim(F, p, u, u_star, u_star_div, u_mag_sq, N_markers, marker_pos, marker_vel, marker_f, marker_nh, marker_nh_size, 
                 Nt, Nit, rho_0, F_brownian_scale, Nx, Ny, Nz, dx, dy, dz, cell_vol, dt, n_lattice, n_lattice_internal, 
                 r_cutoff_outer, r_cutoff_outer_sq, r_cutoff_inner_sq, dist_func, r_gaus, sigma, A, stopping_lims, 
                 live_flow_plot, outevery):
    
    break_cond = False
    int_err = 0.0
    u_mag_sq_max = 0.0
    
    marker_pos_hist = np.empty((N_markers, Nt, 3), dtype=np.float64)
    marker_vel_hist = np.empty_like(marker_pos_hist)
    marker_f_hist = np.empty_like(marker_pos_hist)

    iterations = tqdm.tqdm(range(Nt)) # initialise progress bar
    start_time = time.perf_counter()
    for step in iterations:
        
        if np.isnan(u_mag_sq).any():
            raise RuntimeError(f'Unrealistic velocities: step {step} (t={step*dt})')
        
        
        # Calculate marker forces
        brownian_forcing(F_brownian_scale, N_markers, marker_f)
        
        # Calculate forcing due to IB markers
        int_err = IB_force_density(Nx, Ny, Nz, r_cutoff_outer, r_cutoff_outer_sq, r_cutoff_inner_sq, F, 
                                   dist_func, r_gaus, sigma, A, N_markers, marker_pos, marker_f, marker_nh, marker_nh_size, int_err, cell_vol=cell_vol)
        
        # Apply FVM
        compute_tentative_velocity(u, u_star, F, Nx, Ny, Nz, dx, dy, dz, dt, nu, rho_0)
        set_BCs(u_star)
        
        pressure_poisson(p, u_star, u_star_div, Nx, Ny, Nz, dx, dy, dz, dt, n_lattice_internal, rho_0, Nit)
        
        correct_velocity(u, u_star, p, Nx, Ny, Nz, dx, dy, dz, dt, rho_0)
        set_BCs(u)
        
        
        # Interpolate boundary marker velocities
        interpolate_marker_vels(u, N_markers, marker_vel, marker_nh, marker_nh_size)
        
        # Integrate boundary markers
        marker_pos += (marker_vel*dt)/dx # [lattice points]
        
        
        # Save marker data
        save_marker_data(step, marker_pos_hist, marker_vel_hist, marker_f_hist, N_markers, marker_pos, marker_vel, marker_f)
        
        
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
                save_data = (step%outevery == 0) or (step == Nt-1)
        
        
        # Output flow field at particle cross-section
        if save_data and live_flow_plot:
            u_mag_sq = u[:, :, :, 0]*u[:, :, :, 0] + u[:, :, :, 1]*u[:, :, :, 1] + u[:, :, :, 2]*u[:, :, :, 2]
            
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
                plt.plot(marker_pos_hist[m, :step+1, 0], marker_pos_hist[m, :step+1, 1], 'r', alpha=0.5)
                plt.plot(init_marker_pos[m, 0], init_marker_pos[m, 1], 'xr')
                
                circle = plt.Circle((marker_pos[m, 0], marker_pos[m, 1]), r_particle, color='red', fill=False, linewidth=1.5)
                plt.gca().add_patch(circle)
            
            plt.xlim([0, Nx-1])
            plt.ylim([0, Ny-1])
            
            plt.title(f'3D Particle Diffusion - {IB_kernel} IBM\nZ Position = {z_slice} lattice points, t = {step*dt:.0f}s')
            plt.xlabel('X Position [lattice points]')
            plt.ylabel('Y Position [lattice points]')
            plt.tight_layout()
            # plt.savefig(f'{t}_diff_ani.png')
            plt.show()
        
        if break_cond:
            break

    end_time = time.perf_counter()
    loop_wt = end_time - start_time
    cell_updates = n_lattice*(step+1)
    print(f'\nLoop Wall Time: {loop_wt:.6f} s\nEfficiency: {(1e-6*cell_updates/loop_wt):.4f} MCUPS')

    if int_err > 1.0e-2:
        warnings.warn(f'Subtantial kernel integration error (likely due to unrefined mesh): {(int_err*100):.4f} %', RuntimeWarning)

    if step == Nt-1:
        simtime_reached = True
    else:
        simtime_reached = False
    
    marker_pos_hist = marker_pos_hist[:, :(step+1), :]
    marker_vel_hist = marker_vel_hist[:, :(step+1), :]
    marker_f_hist = marker_f_hist[:, :(step+1), :]
    
    return marker_pos_hist, marker_vel_hist, marker_f_hist, simtime_reached



sim_res = run_diff_sim(F, p, u, u_star, u_star_div, u_mag_sq, N_markers, marker_pos, marker_vel, marker_f, marker_nh, marker_nh_size, 
                 Nt, Nit, rho_0, F_brownian_scale, Nx, Ny, Nz, dx, dy, dz, cell_vol, dt, n_lattice, n_lattice_internal, 
                 r_cutoff_outer, r_cutoff_outer_sq, r_cutoff_inner_sq, dist_func, r_gaus, sigma, A, stopping_lims, 
                 live_flow_plot, outevery)

marker_pos_hist, marker_vel_hist, marker_f_hist, simtime_reached = sim_res









