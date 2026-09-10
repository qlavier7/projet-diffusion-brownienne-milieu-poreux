#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import warnings
import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import imageio
from matplotlib.patches import Polygon
import tqdm

from forced_LBGK_lib import get_LBM_consts, initialise_pops, update_LBM_pops_closed
from multi_marker_IBM_lib import gaus_consts, gaus_dist, dual_gaus_consts, dual_gaus_dist, IB_force_density, interpolate_marker_vels

max_mem_avail = 12.0e9 

#%% Paramètres du problème

show_gaus_dist = False 
live_flow_plot = True 
N_outputs = 10 
show_mass = False 

# Géométrie du domaine et des particules
D_particle1 = 7 
D_particle2 = 10
D_particle3 = 4
D_particles = [D_particle1, D_particle2, D_particle3] 
r_particles = [D/2 for D in D_particles] 
spacing_mutl = 10 

Nx = int(spacing_mutl*D_particle1+1) 
Ny = Nx 
Nz = Nx 
n_lattice = Nx*Ny*Nz 

cx_particle = (Nx-1)/2 
cy_particle = (Ny-1)/2 
cz_particle = (Nz-1)/2 
N_markers = 3 

stop_dist = D_particle1 
stopping_lims = [[stop_dist, Nx-1-stop_dist], [stop_dist, Ny-1-stop_dist], [stop_dist, Nz-1-stop_dist]]

# Paramètres IBM
IB_kernel = ['standard gaussian', 'dual gaussian'][1] 
f_dist_width = 2 

# Propriétés du fluide
nu = 1/50 
rho_0 = 1.0 
mu = rho_0*nu 

# Masses volumiques des particules
rhos = np.ones(N_markers, dtype=np.float64) * 1.14 * rho_0 

# Propriétés du cylindre (fibre)
N_cylinders = 1
cyl_pos = np.array([[35.0, 35.0, 35.0]], dtype=np.float64)   # Centre (X, Y, Z)
cyl_axis = np.array([[1.0, 1.0, 0.0]], dtype=np.float64)      # Orientation (X, Y, Z)
cyl_axis = cyl_axis / np.linalg.norm(cyl_axis, axis=1)[:, np.newaxis]  
cyl_radius = np.array([6.0], dtype=np.float64)                # Rayon
cyl_height = np.array([40.0], dtype=np.float64)               # Hauteur

#%% Paramètres du solveur
sim_time = 1000 
Nt = int(sim_time) 
outevery = 10

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


#%% Fonctions Physiques IBM & Collisions

def generate_cylinder_ibm_markers(cyl_pos, cyl_axis, cyl_radius, cyl_height, ds=0.8):
    """Génère un maillage de points sur la surface du cylindre pour l'IBM fixe."""
    markers = []
    axis = cyl_axis / np.linalg.norm(cyl_axis)
    
    if np.allclose(axis[:2], 0):
        v1 = np.array([1.0, 0.0, 0.0])
    else:
        v1 = np.array([-axis[1], axis[0], 0.0])
        v1 /= np.linalg.norm(v1)
    v2 = np.cross(axis, v1)
    
    n_h = max(int(np.ceil(cyl_height / ds)), 1)
    n_theta = max(int(np.ceil(2 * np.pi * cyl_radius / ds)), 6)
    
    h_vals = np.linspace(-cyl_height/2, cyl_height/2, n_h)
    theta_vals = np.linspace(0, 2*np.pi, n_theta, endpoint=False)
    
    for h in h_vals:
        center_h = cyl_pos + h * axis
        for theta in theta_vals:
            pt = center_h + cyl_radius * (np.cos(theta) * v1 + np.sin(theta) * v2)
            markers.append(pt)
            
    return np.array(markers, dtype=np.float64)


@nb.jit(nopython=True, fastmath=True)
def apply_fixed_cylinder_ibm_force(Nx, Ny, Nz, cyl_markers, u, F, r_cutoff=1.5):
    """Impose une vitesse fluide nulle (u = 0) sur la surface du cylindre fixe."""
    N_pts = cyl_markers.shape[0]
    for m in range(N_pts):
        mx, my, mz = cyl_markers[m, 0], cyl_markers[m, 1], cyl_markers[m, 2]
        
        x_min = max(int(np.floor(mx - r_cutoff)), 0)
        x_max = min(int(np.ceil(mx + r_cutoff)), Nx - 1)
        y_min = max(int(np.floor(my - r_cutoff)), 0)
        y_max = min(int(np.ceil(my + r_cutoff)), Ny - 1)
        z_min = max(int(np.floor(mz - r_cutoff)), 0)
        z_max = min(int(np.ceil(mz + r_cutoff)), Nz - 1)
        
        u_m_x, u_m_y, u_m_z = 0.0, 0.0, 0.0
        weight_sum = 0.0
        
        for ix in range(x_min, x_max + 1):
            for iy in range(y_min, y_max + 1):
                for iz in range(z_min, z_max + 1):
                    dist_sq = (ix - mx)**2 + (iy - my)**2 + (iz - mz)**2
                    if dist_sq <= r_cutoff**2:
                        w_val = np.exp(-dist_sq / (2.0 * 0.5**2))
                        u_m_x += u[ix, iy, iz, 0] * w_val
                        u_m_y += u[ix, iy, iz, 1] * w_val
                        u_m_z += u[ix, iy, iz, 2] * w_val
                        weight_sum += w_val
        
        if weight_sum > 0.0:
            u_m_x /= weight_sum
            u_m_y /= weight_sum
            u_m_z /= weight_sum
            
            f_x = -u_m_x
            f_y = -u_m_y
            f_z = -u_m_z
            
            for ix in range(x_min, x_max + 1):
                for iy in range(y_min, y_max + 1):
                    for iz in range(z_min, z_max + 1):
                        dist_sq = (ix - mx)**2 + (iy - my)**2 + (iz - mz)**2
                        if dist_sq <= r_cutoff**2:
                            w_val = np.exp(-dist_sq / (2.0 * 0.5**2)) / weight_sum
                            F[ix, iy, iz, 0] += f_x * w_val
                            F[ix, iy, iz, 1] += f_y * w_val
                            F[ix, iy, iz, 2] += f_z * w_val


@nb.jit(nopython=True, fastmath=True)
def resolve_all_collisions(marker_pos, marker_vel, D_particles, rhos, 
                          cyl_pos, cyl_axis, cyl_radius, cyl_height):
    """
    Gestion unifiée des collisions 3D sous l'hypothèse d'au plus 1 collision par particule par dt :
      - Cas 1 : Collision particule-particule (choc élastique)
      - Cas 2 : Collision particule-cylindre/wall (masse infinie, choc élastique)
      - Cas 3 : Déplacement libre (pas de collision)
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
                        v_dot_n = vx * nx + vy * ny + vz * nz
                        
                        v0_i = marker_vel[i].copy()
                        v0_j = marker_vel[j].copy()
                        
                        # Choc élastique entre 2 sphères
                        marker_vel[i, 0] -= 2.0 * mj / (mi + mj) * v_dot_n * nx
                        marker_vel[i, 1] -= 2.0 * mj / (mi + mj) * v_dot_n * ny
                        marker_vel[i, 2] -= 2.0 * mj / (mi + mj) * v_dot_n * nz
                        
                        marker_vel[j, 0] += 2.0 * mi / (mi + mj) * v_dot_n * nx
                        marker_vel[j, 1] += 2.0 * mi / (mi + mj) * v_dot_n * ny
                        marker_vel[j, 2] += 2.0 * mi / (mi + mj) * v_dot_n * nz
                        
                        # Mise à jour position sur le pas splité (delta_t)
                        marker_pos[i, 0] += v0_i[0] * delta_t + marker_vel[i, 0] * (1.0 - delta_t)
                        marker_pos[i, 1] += v0_i[1] * delta_t + marker_vel[i, 1] * (1.0 - delta_t)
                        marker_pos[i, 2] += v0_i[2] * delta_t + marker_vel[i, 2] * (1.0 - delta_t)
                        
                        marker_pos[j, 0] += v0_j[0] * delta_t + marker_vel[j, 0] * (1.0 - delta_t)
                        marker_pos[j, 1] += v0_j[1] * delta_t + marker_vel[j, 1] * (1.0 - delta_t)
                        marker_pos[j, 2] += v0_j[2] * delta_t + marker_vel[j, 2] * (1.0 - delta_t)
                        
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
                            
                            v_dot_n = vx * nx + vy * ny + vz * nz
                            v0_i = marker_vel[i].copy()
                            
                            # Réflexion élastique (M_cyl -> infini)
                            marker_vel[i, 0] -= 2.0 * v_dot_n * nx
                            marker_vel[i, 1] -= 2.0 * v_dot_n * ny
                            marker_vel[i, 2] -= 2.0 * v_dot_n * nz
                            
                            # Déplacement sous-pas de temps
                            marker_pos[i, 0] += v0_i[0] * delta_t + marker_vel[i, 0] * (1.0 - delta_t)
                            marker_pos[i, 1] += v0_i[1] * delta_t + marker_vel[i, 1] * (1.0 - delta_t)
                            marker_pos[i, 2] += v0_i[2] * delta_t + marker_vel[i, 2] * (1.0 - delta_t)
                            
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
    """Calcul d'épaisseur 3D pour la visualisation Matplotlib."""
    axis = cyl_axis / np.linalg.norm(cyl_axis)
    u_x, u_y, u_z = axis
    dx = grid_x - cyl_pos[0]
    dy = grid_y - cyl_pos[1]
    thickness = np.zeros_like(grid_x, dtype=float)
    
    if abs(u_z) < 1e-6:
        h_proj = dx * u_x + dy * u_y
        d_perp_sq = (dx**2 + dy**2) - h_proj**2
        inside = (d_perp_sq <= cyl_radius**2) & (np.abs(h_proj) <= cyl_height / 2.0)
        thickness[inside] = 2.0 * np.sqrt(np.maximum(0, cyl_radius**2 - d_perp_sq[inside]))
        return thickness

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
        
        thickness[valid] = np.maximum(0.0, dz_top - dz_bot)
        
    return thickness


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


#%% Setup des Grilles et Tableaux
rho = np.empty((Nx, Ny, Nz), dtype=np.float64) 
u = np.empty((Nx, Ny, Nz, 3), dtype=np.float64) 
u_mag_sq = np.empty_like(rho) 
F = np.empty_like(u) 

pops_pre = np.empty((Nx, Ny, Nz, N_vels), dtype=np.float64) 
pops_post = np.empty_like(pops_pre) 

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

r_cutoff_outer = np.max(r_gaus + r_cutoff) if IB_kernel != 'standard gaussian' else np.max(r_cutoff)
r_cutoff_inner = max(np.max(r_gaus - r_cutoff), 0.0) if IB_kernel != 'standard gaussian' else 0.0

r_cutoff_outer_sq = r_cutoff_outer ** 2
r_cutoff_inner_sq = r_cutoff_inner ** 2

init_marker_pos = np.empty((N_markers, 3), dtype=np.float64)
init_marker_pos[0] = [cx_particle - 4*D_particle1, cy_particle, cz_particle]
init_marker_pos[1] = [cx_particle + 2*D_particle2, cy_particle, cz_particle]
init_marker_pos[2] = [cx_particle, cy_particle - 5*D_particle3, cz_particle]

marker_pos = np.empty_like(init_marker_pos) 
marker_vel = np.empty_like(marker_pos) 
marker_f = np.empty_like(marker_pos) 

marker_nh_size = np.empty(N_markers, dtype=np.int64)
max_marker_neighbours = int((2*(np.ceil(r_cutoff_outer)+1))**3)
marker_nh = np.empty((N_markers, 5, max_marker_neighbours), dtype=np.float64) 

initialise_fluid_arrays(Nx, Ny, Nz, rho_0, rho, u, u_mag_sq, F, pops_pre, pops_post)
initialise_IBM(init_marker_pos, marker_pos, marker_vel, marker_f)

# Marqueurs IBM du cylindre fixe
cyl_ibm_markers = generate_cylinder_ibm_markers(cyl_pos[0], cyl_axis[0], cyl_radius[0], cyl_height[0])

@nb.jit(nopython=True, parallel=True, fastmath=True)
def save_marker_data(step, marker_pos_hist, marker_vel_hist, marker_f_hist, N_markers, marker_pos, marker_vel, marker_f):
    for m in nb.prange(N_markers):
        marker_pos_hist[m, step] = marker_pos[m]
        marker_vel_hist[m, step] = marker_vel[m]
        marker_f_hist[m, step] = marker_f[m]


#%% Boucle Solver Principale

def run_diff_sim():
    int_err = 0.0
    u_mag_sq_max = 0.0
    
    marker_pos_hist = np.empty((N_markers, Nt, 3), dtype=np.float64)
    marker_vel_hist = np.empty_like(marker_pos_hist)
    marker_f_hist = np.empty_like(marker_pos_hist)
    fluid_mass_hist = np.empty(Nt, dtype=np.float64)

    # Impulsion initiale modérée pour éviter l'effet de saut numérique
    marker_f[0, 0] = 1
    marker_f[2, 1] = 1
    
    frames = []
    for t in tqdm.tqdm(range(Nt)):
        if t >= 100:
            marker_f.fill(0.0)
        
        # 1. Forçage IBM du cylindre immobile sur le fluide
        apply_fixed_cylinder_ibm_force(Nx, Ny, Nz, cyl_ibm_markers, u, F)
        
        # 2. Forçage IBM des particules mobiles
        int_err = IB_force_density(Nx, Ny, Nz, r_cutoff_outer, r_cutoff_outer_sq, r_cutoff_inner_sq, F, 
                                   dist_func, r_gaus, sigma, A, N_markers, marker_pos, marker_f, marker_nh, marker_nh_size, int_err)
        
        # 3. Étape de streaming & collision LBM
        update_LBM_pops_closed(pops_pre, pops_post, F, rho, u, u_mag_sq, Nx, Ny, Nz, 
                               inv_cs2, inv_2cs2, inv_cs4, inv_2cs4, omega, omega_prime, omega_S_coeff, 
                               N_vels, w, c, inv_cx_indx, inv_cy_indx, inv_cz_indx)
        
        # 4. Interpolation des vitesses
        interpolate_marker_vels(u, N_markers, marker_vel, marker_nh, marker_nh_size)
        
        # 1. Traitement unifié des 3 cas de collision
        resolve_all_collisions(marker_pos, marker_vel, D_particles, rhos, 
                      cyl_pos, cyl_axis, cyl_radius, cyl_height)
        
        save_marker_data(t, marker_pos_hist, marker_vel_hist, marker_f_hist, N_markers, marker_pos, marker_vel, marker_f)
        fluid_mass_hist[t] = np.sum(rho)
        
        # Rendus graphiques
        if (t % outevery == 0 or t == Nt - 1) and live_flow_plot:
            u_mag_sq_max = max(u_mag_sq_max, np.max(u_mag_sq))
            z_slice = min(max(int(round(marker_pos[0, 2])), 0), Nz-1)
            
            fig, ax = plt.subplots(figsize=(5, 4))
            im = ax.imshow(np.sqrt(u_mag_sq[:, :, z_slice]).T, cmap='viridis', origin='lower', vmin=0, vmax=u_mag_sq_max**0.5)
            plt.colorbar(im, label='Velocity Magnitude')
            
            for m_idx in range(N_markers):
                ax.plot(marker_pos_hist[m_idx, :t+1, 0], marker_pos_hist[m_idx, :t+1, 1], 'r', alpha=0.5)
                circle = plt.Circle((marker_pos[m_idx, 0], marker_pos[m_idx, 1]), r_particles[m_idx], color='red', fill=False, linewidth=1.5)
                ax.add_patch(circle)
            
            # Affichage cylindre
            for c_idx in range(N_cylinders):
                pos, axis, r, h = cyl_pos[c_idx], cyl_axis[c_idx], cyl_radius[c_idx], cyl_height[c_idx]
                sorted_pts = None
                
                if np.abs(axis[2]) < 1e-4:
                    dz = abs(z_slice - pos[2])
                    if dz <= r:
                        w_eff = np.sqrt(r**2 - dz**2)
                        axis_2d = axis[:2] / np.linalg.norm(axis[:2])
                        perp_2d = np.array([-axis_2d[1], axis_2d[0]])
                        sorted_pts = np.array([pos[:2] - (h/2)*axis_2d - w_eff*perp_2d,
                                               pos[:2] + (h/2)*axis_2d - w_eff*perp_2d,
                                               pos[:2] + (h/2)*axis_2d + w_eff*perp_2d,
                                               pos[:2] - (h/2)*axis_2d + w_eff*perp_2d])
                
                if sorted_pts is not None and len(sorted_pts) > 3:
                    x_min, y_min = np.min(sorted_pts[:, 0]), np.min(sorted_pts[:, 1])
                    x_max, y_max = np.max(sorted_pts[:, 0]), np.max(sorted_pts[:, 1])
                    gx, gy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
                    thick_map = compute_cylinder_z_thickness(pos, axis, r, h, gx, gy)
                    
                    from matplotlib.path import Path
                    mask = Path(sorted_pts).contains_points(np.column_stack((gx.flatten(), gy.flatten()))).reshape(gx.shape)
                    thick_map_masked = np.ma.masked_where(~mask | (thick_map <= 1e-5), thick_map)
                    
                    ax.imshow(thick_map_masked, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='plasma', zorder=5, alpha=0.95)
                    ax.add_patch(Polygon(sorted_pts, fill=False, linewidth=0, zorder=6, edgecolor='none'))
            
            ax.set_xlim([0, Nx-1]); ax.set_ylim([0, Ny-1])
            ax.set_title(f'3D IB-LBM - Fixed Cylinder & Bounce\nt = {t}')
            plt.tight_layout()
            
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
            frames.append(image)
            plt.close(fig)

    if live_flow_plot and len(frames) > 0:
        imageio.mimsave('particle_cylinder_interaction.gif', frames, fps=5, loop=0)


if __name__ == '__main__':
    run_diff_sim()