import copy
import numpy as np
from obs_IB_LBM_particle_diff_example import run_diff_sim, initialise_fluid_arrays, initialise_IBM, initialise_obstacle
from obs_forced_LBGK_lib import get_LBM_consts
from multi_marker_IBM_lib import gaus_consts, gaus_dist, dual_gaus_consts, dual_gaus_dist, plot_gaus_dist, IB_force_density, interpolate_marker_vels
import matplotlib.pyplot as plt

#%% Initial conditions
show_gaus_dist = False # plot the y distribution of the force distribution function
live_flow_plot = False # plot the flow field during the simulation
N_outputs = 100 # n.o. times to plot the solution field (only if live_flow_plot=True)
show_mass = False # plot the total fluid mass over the simulation duration - can be useful for identifying instabilities (should remain constant)
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
N_markers = 1 # number of particle markers - need to offset initial positions for anything to happen when increasing this from 1
stop_dist = D_particle # minimum distance from the particle to the wall before the simulation is stopped
stopping_lims = [[stop_dist, Nx-1-stop_dist], [stop_dist, Ny-1-stop_dist], [stop_dist, Nz-1-stop_dist]]
IB_kernel = ['standard gaussian', 'dual gaussian'][0] # force distribution function to use
f_dist_width = 2 # width of the surface gaussian force distribution function [lattice points] (only for dual gaussian IB kernel)
nu = 1/6 # kinematic viscosity [m2 s-1]
rho_0 = 1.0 # initial density [kg m-3]
mu = rho_0*nu # dynamic viscosity [kg m-1 s-1]
kB_T = 0.005 # Diffusion - would probably be defined by a temperature
sim_time = 1000 # simulation time [s] - adjust accordingly
Nt = int(sim_time) # number of time steps (since dt=1)
outevery = int(Nt/N_outputs) # generate an output every this many steps
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
rho = np.empty((Nx, Ny, Nz), dtype=np.float64) # densities
u = np.empty((Nx, Ny, Nz, 3), dtype=np.float64) # velocities
u_mag_sq = np.empty_like(rho) # squared velocity magnitudes
F = np.empty_like(u) # body forces
pops_pre = np.empty((Nx, Ny, Nz, N_vels), dtype=np.float64) # discrete velocity distribution functions
pops_post = np.empty_like(pops_pre) # second DVDF array for efficient data writing during streaming
obstacle = np.zeros_like(rho, dtype=bool)

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

collide_forced = ['initial', 'fluctuation'][0]

# Fluid Domain Boundary Conditions
## BCs = [[x_low, x_high], [y_low, y_high], [z_low, z_high]]
## 0 = periodic boundary, 1 = slip boundary, 2 = no-slip boundary
## When using periodic boundaries, both i_low and i_high must == 0
#BCs = [[0, 0], [0, 0], [0, 0]] # ex: all periodic
BCs = [[1, 1], [1, 1], [1, 1]] # ex: all slip
# BCs = [[2, 2], [2, 2], [2, 2]] # ex: all no-slip
# BCs = [[0, 0], [2, 2], [2, 2]] # ex: x periodic, y/z no-slip
BCs = np.array(BCs, dtype=np.uint8)
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

#%% 


initial_state = {
    'pops_pre': pops_pre,
    'pops_post': pops_post,
    'marker_pos': marker_pos,
    'marker_vel': marker_vel,
    'marker_f': marker_f,
    'marker_nh': marker_nh,
    'rho': rho,
    'u': u,
}

def get_fresh_params(state):
    """Crée une copie indépendante de chaque variable d'état."""
    return {
        key: val.copy() if isinstance(val, np.ndarray) else copy.deepcopy(val)
        for key, val in state.items()
    }


Matrix = []
N_simulations = 20

for i in range(N_simulations):
    print(f"Étape {i}")

    # Réinitialisation des tableaux d'état en place avant chaque itération
    initialise_fluid_arrays(
        Nx, Ny, Nz, rho_0, rho, u, u_mag_sq, F, pops_pre, pops_post
    )
    initialise_IBM(init_marker_pos, marker_pos, marker_vel, marker_f)

    sim_res = run_diff_sim(pops_pre, pops_post, F, rho, u, u_mag_sq, obstacle, N_markers, marker_pos, marker_vel, marker_f, marker_nh, marker_nh_size, 
                 Nt, Nx, Ny, Nz, n_lattice, r_cutoff_outer, r_cutoff_outer_sq, r_cutoff_inner_sq, dist_func, r_gaus, sigma, A, stopping_lims, 
                 inv_cs2, inv_2cs2, inv_cs4, inv_2cs4, omega, omega_prime, omega_S_coeff, N_vels, w, c, inv_cx_indx, inv_cy_indx, inv_cz_indx, BCs, skip_stop_check, 
                live_flow_plot, outevery, collide_forced)

    marker_pos_hist = sim_res[0]

    # Extraction sous forme de liste
    hist_list = (
        marker_pos_hist.tolist()
        if isinstance(marker_pos_hist, np.ndarray)
        else marker_pos_hist
    )
    Matrix.append(hist_list[0])
 


def process_and_plot_msd(matrix, dt=1.0, txt_filename="msd_matrix.txt"):
    min_len = min(len(traj) for traj in matrix)

    # Tronquer chaque trajectoire à min_len
    Matrix_aligned = [traj[:min_len] for traj in matrix]

# Conversion NumPy sans erreur
    Matrix_np = np.asarray(Matrix_aligned)
    # Conversion en tableau numpy 3D de forme (M, N, 3)
    data = Matrix_np
    M, N, _ = data.shape
    print(data.shape)
    
    # Calcul du déplacement par rapport à t = 0 : pos(t) - pos(0)
    displacements = data - data[:, 0:1, :]
    
    # MSD individuel pour chaque simulation : dx² + dy² + dz² (forme : M, N)
    msd_individual = np.sum(displacements**2, axis=2)
    
    # MSD moyen sur l'ensemble des M simulations (forme : N)
    msd_mean = np.mean(msd_individual, axis=0)
    
    # Temps de simulation
    time = np.arange(N) * dt
    
    # Sauvegarde avec np.savetxt
    # np.savetxt gère les matrices 2D : on formate (N lignes, M + 2 colonnes)
    export_matrix = np.column_stack([time, msd_mean, msd_individual.T])
    header = "Temps\tMSD_moyen\t" + "\t".join([f"Sim_{i}" for i in range(M)])
    np.savetxt(txt_filename, export_matrix, fmt="%.6e", delimiter="\t", header=header)
    
    # Tracé du graphique
    plt.figure(figsize=(8, 5))
    
    # Courbes individuelles en gris clair
    plt.plot(time, msd_individual.T, color="lightgray", alpha=0.8)
    
    # Courbe moyenne en rouge au-dessus
    plt.plot(time, msd_mean, color="red", linewidth=2.5, label="MSD")
    
    # Entrée factice pour la légende du gris clair
    plt.plot([], [], color="lightgray", label=f"Simulations ({M})")
    
    plt.xlabel("Simulation time")
    plt.xlim(0)
    plt.ylim(0)
    plt.ylabel("MSD")
    plt.title("Évolution du MSD en fonction du temps")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

    return msd_individual, msd_mean

# À la fin de ta boucle for :
msd_ind, msd_moy = process_and_plot_msd(Matrix, dt=1.0, txt_filename="resultats_msd.txt")
    
    
