#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Some useful immersed boundary method functions.

Created on Thu Aug 20 11:31:20 2026
Author: Max Robbins
"""

import numpy as np
import numba as nb
import matplotlib.pyplot as plt


# not parallelised by default - race condition present mean that the solution may not be accurate
# (different markers may try to spread force to the same fluid node at the same time)
@nb.jit(nopython=True, parallel=False, fastmath=True)
def IB_force_density(Nx, Ny, Nz, r_cutoff_outer, r_cutoff_outer_sq, r_cutoff_inner_sq, F, dist_func, r_gaus, sigma, A, 
                     N_markers, marker_pos, marker_f, marker_nh, marker_nh_size, int_err, cell_vol=1.0):
    """
    Spreads the Lagrangian marker force to the Eulerian fluid lattice using a
    specified kernel.

    Parameters
    ----------
    Nx : int
        number of fluid cells in the x direction.
    Ny : int
        number of fluid cells in the y direction.
    Nz : int
        number of fluid cells in the z direction.
    r_cutoff_outer : float
        force distribution function outer cutoff distance.
    r_cutoff_outer_sq : float
        force distribution function squared outer cutoff distance.
    r_cutoff_inner_sq : float
        force distribution function squared inner cutoff distance.
    F : ndarray
        fluid force density field. ndims=4, dtype=float
    dist_func : function
        force distribution function.
    r_gaus : float
        gaussian function radial offset (dual gaussian kernel only).
    sigma : float
        gaussian standard deviation.
    A : float
        force distribution function normalisation coefficient.
    N_markers : int
        number of Lagrangian boundary markers
    marker_pos : ndarray
        lagrangian marker position. ndims=2, dtype=float
    marker_f : ndarray
        lagrangian marker force. ndims=2, dtype=float
    marker_nh : ndarray
        lagrangian marker fluid lattice neighborhood points, distances, and 
        weightings. ndims=3, dtype=float
    marker_nh_size : ndarray
        lagrangian marker fluid lattice neighborhood size. ndims=1, dtype=int
    int_err : float
        force distribution function integration error.
    cell_vol : float, optional
        fluid lattice cell volume. The default is 1.0.

    Returns
    -------
    int_err : float
        force distribution function integration error.
    """
    
    F[:] = 0.0 # reset fluid body force density
    inv_cell_vol = 1/cell_vol
    
    for m in nb.prange(N_markers):
        np.int64(m)
        curr_pos_x, curr_pos_y, curr_pos_z = marker_pos[m, 0], marker_pos[m, 1], marker_pos[m, 2]
        
        # Only loop over lattice points which are near the boundary markers
        x_min, x_max = np.int64(max(np.floor(curr_pos_x-r_cutoff_outer), 0)), np.int64(min(np.ceil(curr_pos_x+r_cutoff_outer), Nx-1))
        y_min, y_max = np.int64(max(np.floor(curr_pos_y-r_cutoff_outer), 0)), np.int64(min(np.ceil(curr_pos_y+r_cutoff_outer), Ny-1))
        z_min, z_max = np.int64(max(np.floor(curr_pos_z-r_cutoff_outer), 0)), np.int64(min(np.ceil(curr_pos_z+r_cutoff_outer), Nz-1))
        
        weighting_sum = 0.0
        count = 0
        for x_i in range(x_min, x_max+1):
            rel_x = x_i-curr_pos_x
            rel_x_sq = rel_x*rel_x
            if rel_x_sq > r_cutoff_outer_sq:
                continue
            for y_i in range(y_min, y_max+1):
                rel_y = y_i-curr_pos_y
                rel_xy_sq = rel_y*rel_y + rel_x_sq
                if rel_xy_sq > r_cutoff_outer_sq:
                    continue
                for z_i in range(z_min, z_max+1):
                    rel_z = z_i-curr_pos_z
                    rel_xyz_sq = rel_z*rel_z + rel_xy_sq
                    if r_cutoff_inner_sq <= rel_xyz_sq <= r_cutoff_outer_sq:
                        # Calculate the force distribution weighting
                        r_mag = rel_xyz_sq**0.5
                        f_dist = dist_func(r_mag, r_gaus, sigma, A)
                        weighting_sum += f_dist
                        f_dist_div_cell_vol = f_dist*inv_cell_vol
                        
                        # Spread the boundary marker force to the fluid
                        F[x_i, y_i, z_i, 0] += marker_f[m, 0]*f_dist_div_cell_vol # race condition if run in parallel - be careful
                        F[x_i, y_i, z_i, 1] += marker_f[m, 1]*f_dist_div_cell_vol
                        F[x_i, y_i, z_i, 2] += marker_f[m, 2]*f_dist_div_cell_vol
                        
                        # Save the current marker neighborhood data
                        marker_nh[m, 0, count] = x_i
                        marker_nh[m, 1, count] = y_i
                        marker_nh[m, 2, count] = z_i
                        marker_nh[m, 3, count] = r_mag
                        marker_nh[m, 4, count] = f_dist
                        count += 1
        marker_nh_size[m] = count
        
        sum_err = abs(1.0-weighting_sum)
        if (sum_err > int_err) and (1 <= m <= N_markers-2):
            int_err = sum_err
    
    return int_err


@nb.jit(nopython=True, parallel=True, fastmath=True)
def interpolate_marker_vels(u, N_markers, marker_vel, marker_nh, marker_nh_size):
    """
    Interpolates the local fluid velocity from the Eulerian lattice to the 
    Lagrangian marker.

    Parameters
    ----------
    u : ndarray
        fluid velocity field. ndims=4, dtype=float
    N_markers : int
        number of Lagrangian boundary markers
    marker_vel : ndarray
        lagrangian marker position. ndims=2, dtype=float
    marker_nh : ndarray
        lagrangian marker fluid lattice neighborhood points, distances, and 
        weightings. ndims=3, dtype=float
    marker_nh_size : ndarray
        lagrangian marker fluid lattice neighborhood size. ndims=1, dtype=int
    cell_vol : float, optional
        fluid lattice cell volume. The default is 1.0.

    Returns
    -------
    None.
    """
    
    for m in nb.prange(N_markers):
        np.int64(m)
        
        marker_u = 0.0
        marker_v = 0.0
        marker_w = 0.0
        for lattice_node in range(marker_nh_size[m]):
            x_i = int(marker_nh[m, 0, lattice_node])
            y_i = int(marker_nh[m, 1, lattice_node])
            z_i = int(marker_nh[m, 2, lattice_node])
            f_dist = marker_nh[m, 4, lattice_node]
            
            # Interpolate the fluid velocity at the boundary marker
            marker_u += u[x_i, y_i, z_i, 0]*f_dist
            marker_v += u[x_i, y_i, z_i, 1]*f_dist
            marker_w += u[x_i, y_i, z_i, 2]*f_dist
        
        marker_vel[m, 0] = marker_u
        marker_vel[m, 1] = marker_v
        marker_vel[m, 2] = marker_w



@nb.jit(nopython=True, fastmath=True)
def dual_gaus_consts(width, R_offset, tol=1.0e-8):
    """
    Calculates the constants used to describe the dual gaussian kernel.
    The width is defined such that integral[w, inf](f) = int[0, w](A-f)

    Parameters
    ----------
    width : float
        gaussian distribution width.
    R_offset : float
        surface gaussian force distribution function radial offset.
    tol : float, optional
        cutoff tolerance. The default is 1.0e-8.

    Returns
    -------
    sigma : float
        gaussian standard deviation.
    A : float
        force distribution function normalisation coefficient.
    r_cutoff : float
        cutoff distance from the centre of the offset gaussian.
    """
    
    sigma = width/((2*np.pi)**0.5) # force distribution function (FDF) standard deviation
    A = 1/(2*(2**0.5)*(np.pi**(3/2))*sigma*(sigma**2 + R_offset**2)) # FDF normalisation coefficient
    r_cutoff = sigma*((2*np.log(1/tol))**0.5) # value from the centre of the distribution for which the FDF will be below some cutoff tolerance
    return sigma, A, r_cutoff

@nb.jit(nopython=True, fastmath=True)
def dual_gaus_dist(r, R_offset, sigma, A):
    """
    Calculates the weighting at a given distance using the dual gaussian kernel.

    Parameters
    ----------
    r : float
        distance from marker centre.
    R_offset : float
        surface gaussian force distribution function radial offset.
    sigma : float
        gaussian standard deviation.
    A : float
        force distribution function normalisation coefficient.

    Returns
    -------
    float
        weighting factor.
    """
    
    return 0.5*A*(np.exp(-(0.5*((r-R_offset)/sigma)**2)) + np.exp(-(0.5*((r+R_offset)/sigma)**2)))


@nb.jit(nopython=True, fastmath=True)
def gaus_consts(radius, tol=1.0e-8):
    """
    Calculates the constants used to describe the standard gaussian kernel.

    Parameters
    ----------
    radius : float
        marker radius.
    tol : float, optional
        cutoff tolerance. The default is 1.0e-8.

    Returns
    -------
    sigma : float
        gaussian standard deviation.
    A : float
        force distribution function normalisation coefficient.
    r_cutoff : float
        cutoff distance from the centre of the gaussian.
    """
    
    sigma = radius*((1/np.pi)**0.5)
    A = (2*np.pi*(sigma**2))**(-3/2)
    r_cutoff = sigma*((2*(np.log(1/tol)))**0.5) # value for which the FDF will be below some cutoff tolerance
    return sigma, A, r_cutoff

@nb.jit(nopython=True, fastmath=True)
def gaus_dist(r, R_offset, sigma, A):
    """
    Calculates the weighting at a given distance using the standard gaussian 
    kernel.

    Parameters
    ----------
    r : float
        distance from marker centre.
    R_offset : float
        unused.
    sigma : float
        gaussian standard deviation.
    A : float
        force distribution function normalisation coefficient.

    Returns
    -------
    float
        weighting factor.
    """
    
    return A*np.exp(-(0.5*(r/sigma)**2))



def plot_gaus_dist(Ny, dist_func, r_particle, f_dist_width, r_gaus, sigma, A, r_cutoff, r_cutoff_outer, IB_kernel, N_markers, marker_pos):
    """
    Plot the y distribution of the force distribution function.

    Parameters
    ----------
    Ny : int
        number of fluid cells in the y direction..
    dist_func : function
        force distribution function.
    r_particle : float
        marker radius.
    f_dist_width : float
        gaussian distribution width (dual gaussian only).
    r_gaus : float
        surface gaussian force distribution function radial offset (dual 
        gaussian) OR marker radius (standard gaussian).
    sigma : float
        gaussian standard deviation.
    A : float
        force distribution function normalisation coefficient.
    r_cutoff : float
        cutoff distance from the centre of the gaussian.
    r_cutoff_outer : float
        force distribution function outer cutoff distance.
    N_markers : int
        number of Lagrangian boundary markers
    IB_kernel : str
        type of immersed boundary kernel to use (standard or dual gaussian).
    marker_pos : ndarray
        lagrangian marker position. ndims=2, dtype=float

    Returns
    -------
    None.
    """
    
    fig_dist = plt.figure(figsize=(8, 5))
    ax_dist = fig_dist.add_subplot()
    
    line_reach = 2
    
    loc_plot = np.linspace(0, Ny-1, 1000)
    dist_mag = np.zeros_like(loc_plot)
    
    for m in range(N_markers):
    
        m_ypos = marker_pos[m, 1]
        for i in range(loc_plot.size):
            loc = loc_plot[i]
            r_mag = abs(loc-m_ypos)
            if r_mag <= r_cutoff_outer:
                dist_mag[i] += dist_func(r_mag, r_gaus, sigma, A)
        dist_max = np.max(dist_mag)
        
        if m == 0:
            ax_dist.vlines(m_ypos, 0, dist_max*line_reach, linestyle='-', color='r', alpha=0.5, label='Marker Centre')
            
            ax_dist.vlines(m_ypos-r_particle, 0, dist_max*line_reach, linestyle='--', color='r', alpha=0.5, label=f'Marker Surface (Physical Radius = {r_particle:.2f})')
            ax_dist.vlines(m_ypos+r_particle, 0, dist_max*line_reach, linestyle='--', color='r', alpha=0.5)
            
            if IB_kernel == 'dual gaussian':
                ax_dist.vlines(m_ypos-(r_gaus+f_dist_width/2), 0, dist_max*line_reach, linestyle=':', color='k', alpha=0.5, label=f'Width = {f_dist_width:.2f}')
                ax_dist.vlines(m_ypos-(r_gaus-f_dist_width/2), 0, dist_max*line_reach, linestyle=':', color='k', alpha=0.5)
                ax_dist.vlines(m_ypos+(r_gaus+f_dist_width/2), 0, dist_max*line_reach, linestyle=':', color='k', alpha=0.5)
                ax_dist.vlines(m_ypos+(r_gaus-f_dist_width/2), 0, dist_max*line_reach, linestyle=':', color='k', alpha=0.5)
                
                ax_dist.vlines(m_ypos-(r_gaus+r_cutoff), 0, dist_max*line_reach, linestyle=':', color='g', alpha=0.5, label=f'Function Limit (Cutoff Distance = {r_cutoff:.2f})')
                ax_dist.vlines(m_ypos+(r_gaus+r_cutoff), 0, dist_max*line_reach, linestyle=':', color='g', alpha=0.5)
                if r_cutoff < r_gaus:
                    ax_dist.vlines(m_ypos-(r_gaus-r_cutoff), 0, dist_max*line_reach, linestyle=':', color='g', alpha=0.5)
                    ax_dist.vlines(m_ypos+(r_gaus-r_cutoff), 0, dist_max*line_reach, linestyle=':', color='g', alpha=0.5)
            else:
                ax_dist.vlines(m_ypos-r_cutoff, 0, dist_max*line_reach, linestyle=':', color='g', alpha=0.5, label=f'Function Limit (Cutoff Radius = {r_cutoff:.2f})')
                ax_dist.vlines(m_ypos+r_cutoff, 0, dist_max*line_reach, linestyle=':', color='g', alpha=0.5)
        
        else:
            ax_dist.vlines(m_ypos, 0, dist_max*line_reach, linestyle='-', color='r', alpha=0.5)
            
            ax_dist.vlines(m_ypos-r_particle, 0, dist_max*line_reach, linestyle='--', color='r', alpha=0.5)
            ax_dist.vlines(m_ypos+r_particle, 0, dist_max*line_reach, linestyle='--', color='r', alpha=0.5)
            
            if IB_kernel == 'dual gaussian':
                ax_dist.vlines(m_ypos-(r_gaus+f_dist_width/2), 0, dist_max*line_reach, linestyle=':', color='k', alpha=0.5)
                ax_dist.vlines(m_ypos-(r_gaus-f_dist_width/2), 0, dist_max*line_reach, linestyle=':', color='k', alpha=0.5)
                ax_dist.vlines(m_ypos+(r_gaus+f_dist_width/2), 0, dist_max*line_reach, linestyle=':', color='k', alpha=0.5)
                ax_dist.vlines(m_ypos+(r_gaus-f_dist_width/2), 0, dist_max*line_reach, linestyle=':', color='k', alpha=0.5)
                
                ax_dist.vlines(m_ypos-(r_gaus+r_cutoff), 0, dist_max*line_reach, linestyle=':', color='g', alpha=0.5)
                ax_dist.vlines(m_ypos+(r_gaus+r_cutoff), 0, dist_max*line_reach, linestyle=':', color='g', alpha=0.5)
                if r_cutoff < r_gaus:
                    ax_dist.vlines(m_ypos-(r_gaus-r_cutoff), 0, dist_max*line_reach, linestyle=':', color='g', alpha=0.5)
                    ax_dist.vlines(m_ypos+(r_gaus-r_cutoff), 0, dist_max*line_reach, linestyle=':', color='g', alpha=0.5)
            else:
                ax_dist.vlines(m_ypos-r_cutoff, 0, dist_max*line_reach, linestyle=':', color='g', alpha=0.5)
                ax_dist.vlines(m_ypos+r_cutoff, 0, dist_max*line_reach, linestyle=':', color='g', alpha=0.5)
        
    if IB_kernel == 'standard gaussian':
        ax_dist.plot(loc_plot, dist_mag, 'b-', label=f'Distribution Function (Std Deviation = {sigma:.2f}')
        plt.title('Standard Gaussian Distribution Function - Y Distribution')
    elif IB_kernel == 'dual gaussian':
        ax_dist.plot(loc_plot, dist_mag, 'b-', label=f'Dual Distribution Function (Std Deviation = {sigma:.2f}, Radius = {r_gaus:.2f})')
        plt.title('Dual Gaussian Distribution Function - Y Distribution')
    
    plt.xlabel('Y Position')
    plt.ylabel('Magnitude')
    if IB_kernel == 'standard gaussian':
        plt.ylim([-0.4*dist_max, 1.1*dist_max])
    elif IB_kernel == 'dual gaussian':
        plt.ylim([-0.5*dist_max, 1.1*dist_max])
    # plt.xlim([0, Ny-1])
    plt.legend(loc='lower right')
    plt.grid()
    plt.tight_layout()
    plt.show()




