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
import time

##########################################################
###### PARAMÉTRAGE GÉOMÉTRIQUE À MODIFIER ET ADAPTER######
##########################################################

num_points_per_fiber = 20    # Nombre de points par fibre (taille de la fibre)
point_spacing = 3            # Espacement entre les points voisins au repos
num_fibers = 700             # Nombre total de fibres dans le milieu
cluster_radius = 100         # Rayon de l'amas (cluster) sphérique
min_start_distance = 0       # Distance minimale au centre pour éviter l'accumulation centrale
fiber_radius = 1             # Rayon physique d'une fibre (pour calcul de volume/porosité)

################################################
###### PARAMÉTRAGE PHYSIQUE À CONSERVER ########
################################################

# Paramètres mécaniques
rest_length = point_spacing
damping = 1.0                # Amortissement (stabilité du système)
bending_stiffness = 2.0      # Rigidité en flexion des fibres

# Paramètres Électromagnétiques (EM)
em_strength = 0.3            # Force de l'interaction EM entre fibres
em_cutoff_distance = 2.0     # Distance de coupure pour les calculs d'interaction

# Paramètres de réticulation (Crosslinking)
label_probabilities = (0.8, 0.1, 0.1)  # Probabilités des étiquettes (Aucune, Acide A, Base B)

# Paramètres des liaisons A-B (Liaisons chimiques)
crosslink_distance = 1.0
crosslink_break_distance = 3
crosslink_stiffness = 1.5
crosslink_rest_factor = 1.5  # Facteur de longueur au repos

# Paramètres des liaisons B-B
bb_bond_distance = 0.8
bb_bond_break_distance = 1.8
bb_bond_stiffness = 3.0
bb_bond_rest_factor = 1

################################################
###### PARAMÉTRAGE DE LA SIMULATION ############
################################################

# Il est fondamental de choisir un num_step pour stabiliser au maximum le milieu 
# et éviter l'interpénétration des fibres au maximum.
# On trace l'évolution de l'énergie cinétique totale pour justifier cette valeur.
dt = 0.01                    # Pas de temps
num_steps = 400              # Nombre d'itérations pour la stabilisation

######################################
###### INITIALISATION DU MILIEU ######
######################################

# Génération des positions initiales selon une distribution anisotrope
fiber_positions, fiber_velocities, fiber_lengths, anchor_configs = \
    bibliotheque_generation_visualisation.generate.isotropic_fiber_positions(
         num_fibers, num_points_per_fiber, point_spacing, cluster_radius)

######################################
###### STABILISATION DU MILIEU ######
######################################

total_points = sum(fiber_lengths)
stiffnesses = np.array([2.0] * total_points, dtype=np.float64)

# Lancement de la simulation dynamique optimisée
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

#################################################
###### SUIVI DE L'ÉNERGIE TOTALE DU MILIEU ######
#################################################

# Calcul de l'énergie cinétique (Ec = 1/2 * m * v²) à chaque étape
# On considère une masse unitaire pour chaque point
kinetic_energy = 0.5 * np.sum(velocity_data**2, axis=2)  # Somme sur les composantes x, y, z
energy_per_step = np.sum(kinetic_energy, axis=0)        # Somme sur tous les points du milieu

print(f"\nEnergy analysis:")
print(f"- Initial kinetic energy: {energy_per_step[0]:.6f}")
print(f"- Final kinetic energy: {energy_per_step[-1]:.6f}")
print(f"- Energy dissipated: {energy_per_step[0] - energy_per_step[-1]:.6f}")

# Visualisation de la dissipation d'énergie pour vérifier la stabilisation
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

# Sauvegarde des données finales pour une utilisation ultérieure (fichiers .npy)
np.save("position_data.npy", position_data)
np.save("velocity_data.npy", velocity_data)
np.save("fiber_ids.npy", fiber_ids)
np.save("point_in_fiber.npy", point_in_fiber)
np.save("labels.npy", labels)
np.save("final_crosslinks.npy", final_crosslinks)
np.save("final_bb_bonds.npy", final_bb_bonds)

####################################################
###### CALCUL ET AFFICHAGE DE LA POROSITÉ ##########
####################################################

# Calcul de la porosité locale dans des sphères de rayons croissants
longueurs = np.arange(1, 100, 1)
porosites = {longueur: None for longueur in longueurs}
position_data_f = position_data[:, -1, :] # Positions au dernier pas de temps

for longueur in longueurs:
    # Calcul précis de la porosité à l'aide de la bibliothèque dédiée
    porosite = bibliotheque_etude_milieu.compute_sphere_porosity(
        longueur, fiber_ids, position_data_f, fiber_radius)
    porosites[longueur] = porosite

# Préparation des données pour le graphique
x = list(porosites.keys())       # Rayons des sphères
y = list(porosites.values())     # Valeurs de porosité calculées

# Calcul de la limite théorique de validité pour éviter les effets de bord (biais)
r_lim = cluster_radius - (num_points_per_fiber - 1) * point_spacing / 2

plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='-', markersize=3, label="Porosité")
plt.axvline(r_lim, color='red', linestyle='--', linewidth=2, label="Limite biais")

plt.title("Porosité en fonction du rayon de la sphère de mesure")
plt.xlabel("Rayon (µm)")
plt.ylabel("Porosité")
plt.grid(True)
plt.ylim(0.95, 1.0)
plt.legend()
plt.show()

####################################################
###### AFFICHAGE DU MILIEU EN 3D ###################
####################################################

# Création d'une vue 3D interactive du milieu stabilisé au dernier pas de temps
bibliotheque_generation_visualisation.create_interactive_3d_view(
    position_data, fiber_ids, point_in_fiber, labels,
    crosslink_partner, bb_partners, bb_partner_count,
    timestep=-1
)
