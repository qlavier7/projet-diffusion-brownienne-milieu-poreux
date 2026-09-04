import bibliotheque_generation_visualisation
import bibliotheque_etude_milieu
import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
import os
import tarfile
import json
import tempfile
import shutil



#########################
###### PARAMÉTRAGE ######
#########################

#paramètres géométriques du milieu généré

num_points_per_fiber = 20 #size of the fiber
point_spacing = 3 #spacing between neighbors at rest
num_fibers = 700
cluster_radius = 100 # Radius of the spherical cluster
min_start_distance = 0 # Minimum distance from center for starting points to avoid clustering at center


# Physical parameters
rest_length = point_spacing
damping = 1.0 # Increasee damping for stability
bending_stiffness = 2.0 # Reduce bending stiffness for stability

# EM parameters
em_strength = 0.3 # Reduce EM strength for stability
em_cutoff_distance = 2.0 #cutoff saves computation time

# Crosslinking parameters
label_probabilities = (0.8, 0.1, 0.1) # (None, A, B) A for acid, B for base, None for nothing, probabilities

# A-B bond parameters
crosslink_distance = 1.0
crosslink_break_distance = 3
crosslink_stiffness = 1.5
crosslink_rest_factor = 1.5 #rest length factor

# B-B bond parameters
bb_bond_distance = 0.8
bb_bond_break_distance = 1.8
bb_bond_stiffness = 3.0
bb_bond_rest_factor = 1


# Simulation parameters
dt = 0.01 #time step
num_steps = 400 #number of time steps

fiber_radius=em_cutoff_distance/2
kappa=5

######################################
###### INITIALISATION DU MILIEU ######
######################################
fiber_positions, fiber_velocities, fiber_lengths, anchor_configs = \
    bibliotheque_generation_visualisation.generate_anisotropic_fiber_positions_2(kappa,
        num_fibers, num_points_per_fiber, point_spacing,
        cluster_radius)
    






######################################
###### STABILISATION DU MILIEU ######
######################################

total_points = sum(fiber_lengths)
stiffnesses = np.array([2.0] * total_points, dtype=np.float64)

# Run optimized simulation
import time
start_time = time.time()

(position_data, velocity_data, fiber_ids, point_in_fiber,
labels, crosslink_partner, bb_partners, bb_partner_count,
final_crosslinks, final_bb_bonds) = \
    bibliotheque_generation_visualisation.compute_multi_fiber_dynamics_3d_optimized(
    fiber_positions, fiber_velocities, fiber_lengths,
    rest_length, stiffnesses, damping,
    bending_stiffness, em_strength, em_cutoff_distance,
    crosslink_distance, crosslink_break_distance,
    crosslink_stiffness, crosslink_rest_factor,
    bb_bond_distance, bb_bond_break_distance,
    bb_bond_stiffness, bb_bond_rest_factor,
    label_probabilities,
    dt, num_steps, anchor_configs
    )

elapsed_time = time.time() - start_time
print(f"\nSimulation completed in {elapsed_time:.2f} seconds")
print(f"Performance: {num_steps * total_points / elapsed_time:.0f} point-updates/second")



#################################################
###### SUIVI DE L'ÉNERGIE TOTALE DU MILIEU ######
#################################################



# Calculate energy
initial_energy = np.sum(velocity_data[:, 0, :]**2) / 2
final_energy = np.sum(velocity_data[:, -1, :]**2) / 2
print(f"\nEnergy analysis:")
print(f"- Initial kinetic energy: {initial_energy:.6f}")
print(f"- Final kinetic energy: {final_energy:.6f}")
print(f"- Energy dissipated: {initial_energy - final_energy:.6f}")
print(len(position_data))

# Calcul de l'énergie cinétique à chaque étape
kinetic_energy = 0.5 * np.sum(velocity_data**2, axis=2)  # somme sur x, y, z
energy_per_step = np.sum(kinetic_energy, axis=0)        # somme sur tous les points

# Affichage final
print(f"\nEnergy analysis:")
print(f"- Initial kinetic energy: {energy_per_step[0]:.6f}")
print(f"- Final kinetic energy: {energy_per_step[-1]:.6f}")
print(f"- Energy dissipated: {energy_per_step[0] - energy_per_step[-1]:.6f}")

# Création du graphique
plt.figure(figsize=(10, 6))
plt.plot(range(len(energy_per_step)), energy_per_step, marker='o', linestyle='-')
plt.title("Evolution of kinetic energy over simulation")
plt.xlabel("Time step")
plt.ylabel("Kinetic energy")
plt.grid(True)
plt.show()


#################################################
###### ENREGISTREMENT DU MILIEU FINAL ##########
#################################################

np.save("position_data.npy", position_data)
np.save("velocity_data.npy", velocity_data)
np.save("fiber_ids.npy", fiber_ids)
np.save("point_in_fiber.npy", point_in_fiber)
np.save("labels.npy", labels)
np.save("crosslink_partner.npy", crosslink_partner)
np.save("bb_partners.npy", bb_partners)
np.save("bb_partner_count.npy", bb_partner_count)
np.save("final_crosslinks.npy", final_crosslinks)
np.save("final_bb_bonds.npy", final_bb_bonds)


####################################################
###### CALCUL ET AFFICHAGE DE LA POROSITÉ ##########
####################################################


#porosité 
longueurs= np.arange(1, 100, 1)
porosites = {longueur: None for longueur in longueurs}
position_data_f=position_data[:,-1,:]


for longueur in longueurs:
    porosite=bibliotheque_etude_milieu.compute_sphere_porosity(longueur, fiber_ids, position_data_f, fiber_radius)
    porosites[longueur] = porosite


# Conversion du dictionnaire en listes pour le tracé
x = list(porosites.keys())       # longueurs
y = list(porosites.values())     # porosités

r_lim = cluster_radius - (num_points_per_fiber -1)* point_spacing / 2

# Création du graphique
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='-', markersize=3, label="Porosité")

# Ligne verticale rouge (limite de validité)
plt.axvline(r_lim, linestyle='--', linewidth=2, label="Limite biais")

plt.title("Porosité en fonction de la longueur")
plt.xlabel("Longueur (µm)")
plt.ylabel("Porosité")
plt.grid(True)
plt.ylim(0.95, 1)

plt.legend()
plt.show()





####################################################
###### AFFICHAGE DU MILIEU EN 3D ###################
####################################################
bibliotheque_generation_visualisation.create_interactive_3d_view(position_data, fiber_ids, point_in_fiber, labels,
                               crosslink_partner, bb_partners, bb_partner_count,
                               timestep=-1)

