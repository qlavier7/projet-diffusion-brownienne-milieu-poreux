"""
=============================================================================
MODULE : Simulation de milieux fibreux 3D avec liaisons dynamiques
=============================================================================
 
Ce module implémente une simulation de dynamique moléculaire simplifiée pour
un réseau de fibres en 3D. Il est organisé en trois couches fonctionnelles :
 
─────────────────────────────────────────────────────────────────────────────
1. COUCHE INFRASTRUCTRE (fonctions @njit bas niveau)
─────────────────────────────────────────────────────────────────────────────
 
   build_neighbor_lookup()          build_spatial_hash()
         │                                 │
         │  connectivité                   │  localisation spatiale
         │  intra-fibre                    │  inter-points
         └──────────────┬──────────────────┘
                        │
                        ▼
           get_neighbor_cells()
           (cellules voisines dans la grille 3D)
 
─────────────────────────────────────────────────────────────────────────────
2. COUCHE SIMULATION (cœur du calcul)
─────────────────────────────────────────────────────────────────────────────
 
   compute_multi_fiber_dynamics_3d_optimized()
     ├── appelle build_neighbor_lookup()   → liaisons ressort / flexion
     ├── appelle build_spatial_hash()      → construction de la grille
     ├── appelle get_neighbor_cells()      → voisinage local de chaque point
     └── retourne : positions, vitesses, labels, liaisons A-B et B-B
 
─────────────────────────────────────────────────────────────────────────────
3. COUCHE GÉNÉRATION DE GÉOMÉTRIE (initialisation des fibres)
─────────────────────────────────────────────────────────────────────────────
 
   Milieu isotrope :
     
     generate_isotropic_fiber_positions_2()  ← version (r^(1/3), uniforme en volume)
                                               + paramètre kappa (von Mises) pour le contrôle angulaire
 
   Milieu anisotrope :
     
     generate_anisotropic_fiber_positions_2()← distribution volumique + von Mises directionnel
 
   Ces 2 fonctions produisent le même type de sortie :
     (fiber_positions, fiber_velocities, fiber_lengths, anchor_configs)
   → Prêt à être passé directement à compute_multi_fiber_dynamics_3d_optimized()
 
─────────────────────────────────────────────────────────────────────────────
4. COUCHE VISUALISATION & SAUVEGARDE
─────────────────────────────────────────────────────────────────────────────
 
   create_3d_animation()            → animation MP4 / GIF
   create_interactive_3d_view()     → vue matplotlib interactive
   save_simulation_to_tar()         → archivage de l'état final (.tar)
 
   Ces fonctions consomment les sorties de compute_multi_fiber_dynamics_3d_optimized()
   et n'ont aucune dépendance entre elles.
 
=============================================================================
"""
 
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
from scipy.stats import vonmises
 
 
# =============================================================================
# COUCHE 1 : INFRASTRUCTURE
# =============================================================================




 ####################
 ####FONCTION 1.1####
 ####################
 
@njit
def build_neighbor_lookup(fiber_ids, point_in_fiber, fiber_lengths):
    """
    Construit les tableaux de voisinage immédiat (précédent / suivant) pour
    chaque point le long des fibres.
 
    Les points d'une même fibre sont stockés de manière contiguë en mémoire.
    Cette fonction exploite cette propriété pour relier chaque point à ses
    deux voisins directs (i-1 et i+1) le long de la fibre.
 
    Ces tableaux sont utilisés plus tard dans compute_multi_fiber_dynamics_3d_optimized()
    pour calculer :
      - les forces de ressort (élasticité longitudinale)
      - les forces de flexion (rigidité en courbure)
 
    Paramètres
    ----------
    fiber_ids : ndarray (int32)
        Indice de la fibre à laquelle appartient chaque point global.
    point_in_fiber : ndarray (int32)
        Position locale du point au sein de sa fibre (0 = premier point).
    fiber_lengths : ndarray (int32)
        Nombre de points dans chaque fibre.
 
    Retourne
    --------
    prev_neighbors : ndarray (int32)
        Pour chaque point, l'indice global du point précédent dans la fibre.
        Vaut -1 pour le premier point de chaque fibre (pas de voisin précédent).
    next_neighbors : ndarray (int32)
        Pour chaque point, l'indice global du point suivant dans la fibre.
        Vaut -1 pour le dernier point de chaque fibre.
    """
    total_points = len(fiber_ids)
    prev_neighbors = np.zeros(total_points, dtype=np.int32) - 1
    next_neighbors = np.zeros(total_points, dtype=np.int32) - 1
    
    # Les points sont stockés séquentiellement : fibre 0 puis fibre 1, etc.
    idx = 0
    for f in range(len(fiber_lengths)):
        fiber_len = fiber_lengths[f]
        
        # Connexion de chaque point au suivant au sein de la fibre
        for i in range(fiber_len - 1):
            next_neighbors[idx + i] = idx + i + 1
            prev_neighbors[idx + i + 1] = idx + i
        
        idx += fiber_len
    
    return prev_neighbors, next_neighbors
 
 
 
 
 ####################
 ####FONCTION 1.2####
 ####################
 
@njit
def build_spatial_hash(positions, cell_size, grid_size=20):
    """
    Construit une grille spatiale (hash spatial) pour accélérer la recherche
    de voisins dans le volume 3D.
 
    Principe : le volume est subdivisé en cellules cubiques de côté `cell_size`.
    Chaque point est affecté à la cellule qui contient sa position. Pour trouver
    les voisins d'un point, il suffit ensuite d'examiner les 27 cellules adjacentes
    (via get_neighbor_cells) plutôt que l'ensemble des N points.
 
    Cette grille est reconstruite tous les 10 pas de temps dans la boucle principale
    de compute_multi_fiber_dynamics_3d_optimized(), ce qui suppose implicitement que
    les points se déplacent peu entre deux reconstructions.
 
    Paramètres
    ----------
    positions : ndarray (N, 3)
        Positions courantes de tous les points.
    cell_size : float
        Taille d'une cellule (doit être ≥ à la distance d'interaction maximale).
    grid_size : int
        Nombre de cellules par dimension (grille grid_size³).
 
    Retourne
    --------
    cell_indices : ndarray (int32)
        Indice de cellule (linéarisé) pour chaque point.
    cell_lists : ndarray (grid_size³ × max_points_per_cell, int32)
        Pour chaque cellule, liste des indices des points qu'elle contient.
    cell_counts : ndarray (int32)
        Nombre de points dans chaque cellule.
    min_pos : ndarray (3,)
        Coin inférieur de la boîte englobante (avec marge), nécessaire pour
        recalculer les coordonnées de cellule à partir des positions réelles.
    grid_size : int
        Renvoyé tel quel pour être utilisé par l'appelant.
    """
    total_points = len(positions)
    max_points_per_cell = 1000
 
    cell_indices = np.zeros(total_points, dtype=np.int32)
    cell_lists = np.zeros((grid_size**3, max_points_per_cell), dtype=np.int32) - 1
    cell_counts = np.zeros(grid_size**3, dtype=np.int32)
    
    # Calcul de la boîte englobante avec marge pour éviter les cas limites
    min_pos = np.zeros(3)
    max_pos = np.zeros(3)
    for dim in range(3):
        min_pos[dim] = np.min(positions[:, dim])
        max_pos[dim] = np.max(positions[:, dim])
    
    min_pos -= cell_size
    max_pos += cell_size
    
    # Affectation de chaque point à sa cellule
    for i in range(total_points):
        # Coordonnées entières dans la grille
        cx = int((positions[i, 0] - min_pos[0]) / cell_size)
        cy = int((positions[i, 1] - min_pos[1]) / cell_size)
        cz = int((positions[i, 2] - min_pos[2]) / cell_size)
        
        # Clampage aux limites de la grille
        cx = max(0, min(cx, grid_size - 1))
        cy = max(0, min(cy, grid_size - 1))
        cz = max(0, min(cz, grid_size - 1))
        
        # Indice linéaire de la cellule (cx + cy*N + cz*N²)
        cell_idx = cx + cy * grid_size + cz * grid_size * grid_size
        cell_indices[i] = cell_idx
        
        count = cell_counts[cell_idx]
        if count < max_points_per_cell:
            cell_lists[cell_idx, count] = i
            cell_counts[cell_idx] += 1
    
    return cell_indices, cell_lists, cell_counts, min_pos, grid_size
 
 
 
 ####################
 ####FONCTION 1.3####
 ####################
 
@njit
def get_neighbor_cells(cell_x, cell_y, cell_z, grid_size):
    """
    Retourne les indices linéaires des 27 cellules voisines (y compris la cellule
    elle-même) d'une cellule identifiée par ses coordonnées (cell_x, cell_y, cell_z).
 
    Les cellules hors des limites de la grille sont signalées par l'indice -1
    et sont ignorées par l'appelant.
 
    Cette fonction est appelée dans compute_multi_fiber_dynamics_3d_optimized()
    lors de la construction de la liste de voisins de chaque point.
 
    Paramètres
    ----------
    cell_x, cell_y, cell_z : int
        Coordonnées entières de la cellule cible dans la grille.
    grid_size : int
        Taille de la grille (même valeur dans les trois dimensions).
 
    Retourne
    --------
    neighbors : ndarray (27, int32)
        Indices linéaires des cellules voisines (-1 si hors grille).
    """
    neighbors = np.zeros(27, dtype=np.int32)
    count = 0
    
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            for dz in range(-1, 2):
                nx = cell_x + dx
                ny = cell_y + dy
                nz = cell_z + dz
                
                if 0 <= nx < grid_size and 0 <= ny < grid_size and 0 <= nz < grid_size:
                    neighbors[count] = nx + ny * grid_size + nz * grid_size * grid_size
                else:
                    neighbors[count] = -1
                count += 1
    
    return neighbors
 
 
 
 
 
 
 
# =============================================================================
# COUCHE 2 : SIMULATION PRINCIPALE
# =============================================================================
 
 
 
 ####################
 ####FONCTION 2.1####
 ####################
 
@njit(parallel=True)
def compute_multi_fiber_dynamics_3d_optimized(
        fiber_positions, fiber_velocities, fiber_lengths, 
        rest_length, stiffnesses, damping, 
        bending_stiffness, em_strength, em_cutoff_distance,
        crosslink_distance, crosslink_break_distance,
        crosslink_stiffness, crosslink_rest_factor,
        bb_bond_distance, bb_bond_break_distance,
        bb_bond_stiffness, bb_bond_rest_factor,
        label_probabilities,
        dt, num_steps, anchor_configs):
    """
    Moteur principal de la simulation : intègre les équations du mouvement pour
    un réseau de fibres en 3D avec formation/rupture dynamique de liaisons.
 
    ─── Modèle physique ──────────────────────────────────────────────────────
 
    Chaque fibre est discrétisée en `N` points reliés par des ressorts
    (élasticité longitudinale) et soumis à une résistance à la courbure
    (rigidité de flexion). Trois types d'interactions inter-fibres sont modélisés :
 
      1. Force EM (électromagnétique ou stérique) : répulsion/attraction à
         courte portée entre tous les points non liés (coupure à em_cutoff_distance).
 
      2. Liaisons A-B (crosslinks physiques) : se forment si un point de type A
         (label=1) et un point de type B (label=2) sont à distance < crosslink_distance.
         Chaque point ne peut avoir qu'un seul partenaire A-B simultanément.
         La liaison se rompt si la distance dépasse crosslink_break_distance.
 
      3. Liaisons B-B (covalentes chimiques) : se forment entre deux points B
         à distance < bb_bond_distance. Chaque point B peut avoir jusqu'à 2
         partenaires B-B. Rupture si distance > bb_bond_break_distance.
 
    ─── Optimisations numériques ─────────────────────────────────────────────
 
    - Hash spatial reconstruit tous les 10 pas de temps (via build_spatial_hash
      et get_neighbor_cells) pour limiter la recherche de voisins à O(N).
    - Cache de distances pré-calculées réutilisé pour toutes les interactions.
    - Parallélisation de la mise à jour des positions (prange sur les points).
    - Les points "ancrés" (extrémités fixées) ne sont jamais mis à jour.
 
    ─── Intégration temporelle ───────────────────────────────────────────────
 
    Schéma d'Euler explicite du premier ordre :
        v(t+dt) = v(t) + F_total(t) * dt
        x(t+dt) = x(t) + v(t+dt) * dt
    avec F_total = F_ressort + F_flexion + F_EM + F_crosslink_AB + F_covalent_BB
                + F_amortissement
 
    Paramètres
    ----------
    fiber_positions : list of ndarray (N_i, 3)
        Positions initiales des points de chaque fibre.
    fiber_velocities : list of ndarray (N_i, 3)
        Vitesses initiales des points de chaque fibre.
    fiber_lengths : list of int
        Nombre de points dans chaque fibre.
    rest_length : float
        Longueur à repos des ressorts inter-points.
    stiffnesses : ndarray (total_points,)
        Raideur du ressort pour chaque point.
    damping : float
        Coefficient d'amortissement visqueux (force = -damping * vitesse).
    bending_stiffness : float
        Rigidité en flexion (résistance au changement de courbure).
    em_strength : float
        Amplitude de la force EM (>0 répulsif, <0 attractif). 0 = désactivée.
    em_cutoff_distance : float
        Distance de coupure pour la force EM.
    crosslink_distance : float
        Distance maximale de formation d'une liaison A-B.
    crosslink_break_distance : float
        Distance de rupture d'une liaison A-B existante.
    crosslink_stiffness : float
        Raideur des liaisons A-B.
    crosslink_rest_factor : float
        Facteur appliqué à la distance de formation pour définir la longueur
        à repos d'une liaison A-B (rest_len = distance_formation * facteur).
    bb_bond_distance : float
        Distance maximale de formation d'une liaison B-B.
    bb_bond_break_distance : float
        Distance de rupture d'une liaison B-B existante.
    bb_bond_stiffness : float
        Raideur des liaisons B-B.
    bb_bond_rest_factor : float
        Facteur pour la longueur à repos des liaisons B-B.
    label_probabilities : tuple (p_0, p_A, p_B)
        Probabilités d'assigner à chaque point le label neutre (0), A (1) ou B (2).
        Doit satisfaire p_0 + p_A + p_B = 1.
    dt : float
        Pas de temps de l'intégration (en secondes).
    num_steps : int
        Nombre de pas de temps à simuler.
    anchor_configs : list of tuple (bool, bool)
        Pour chaque fibre, indique si le premier et/ou le dernier point sont ancrés
        (immobiles pendant toute la simulation).
 
    Retourne
    --------
    position_data : ndarray (total_points, num_steps+1, 3)
        Trajectoires complètes de tous les points.
    velocity_data : ndarray (total_points, num_steps+1, 3)
        Vitesses complètes de tous les points.
    fiber_ids : ndarray (int32)
        Indice de fibre pour chaque point global.
    point_in_fiber : ndarray (int32)
        Indice local de chaque point dans sa fibre.
    labels : ndarray (int32)
        Labels (0, 1=A, 2=B) assignés aléatoirement à chaque point.
    crosslink_partner : ndarray (int32)
        Pour chaque point, indice de son partenaire A-B (-1 si aucun).
    bb_partners : ndarray (total_points, 2, int32)
        Pour chaque point B, indices de ses partenaires B-B (-1 si absent).
    bb_partner_count : ndarray (int32)
        Nombre de liaisons B-B actives par point.
    final_crosslinks : ndarray (M_AB, 2, int32)
        Liste des paires (i, j) liées par crosslink A-B à la fin de la simulation.
    final_bb_bonds : ndarray (M_BB, 2, int32)
        Liste des paires (i, j) liées par liaison B-B à la fin de la simulation.
    """
    
    num_fibers = len(fiber_lengths)
    total_points = sum(fiber_lengths)
    
    positions = np.zeros((total_points, 3), dtype=np.float64)
    velocities = np.zeros((total_points, 3), dtype=np.float64)
    fiber_ids = np.zeros(total_points, dtype=np.int32)
    point_in_fiber = np.zeros(total_points, dtype=np.int32)
    labels = np.zeros(total_points, dtype=np.int32)
    
    # Attribution aléatoire des labels : 0 (neutre), 1 (A), 2 (B)
    p_0, p_A, p_B = label_probabilities
    for i in prange(total_points):
        rand = np.random.random()
        if rand < p_0:
            labels[i] = 0
        elif rand < p_0 + p_A:
            labels[i] = 1
        else:
            labels[i] = 2
    
    # Initialisation des tableaux plats à partir des listes par fibre
    idx = 0
    for f in range(num_fibers):
        for p in range(fiber_lengths[f]):
            positions[idx] = fiber_positions[f][p]
            velocities[idx] = fiber_velocities[f][p]
            fiber_ids[idx] = f
            point_in_fiber[idx] = p
            idx += 1
    
    # Construction du voisinage intra-fibre (utilisé pour ressort + flexion)
    prev_neighbors, next_neighbors = build_neighbor_lookup(fiber_ids, point_in_fiber, fiber_lengths)
    
    # Stockage de toutes les trajectoires (mémoire : total_points × num_steps × 3 × 8 octets)
    position_data = np.zeros((total_points, num_steps + 1, 3))
    velocity_data = np.zeros((total_points, num_steps + 1, 3))
    position_data[:, 0, :] = positions
    velocity_data[:, 0, :] = velocities
    
    # ── État des liaisons A-B (crosslinks physiques) ──────────────────────
    # Un point ne peut avoir qu'un seul partenaire A-B à la fois
    crosslink_partner = np.zeros(total_points, dtype=np.int32) - 1
    crosslink_rest_lengths = np.zeros(total_points, dtype=np.float64)
    
    # ── État des liaisons B-B (covalentes chimiques) ──────────────────────
    # Un point B peut avoir jusqu'à 2 partenaires B-B
    bb_partners = np.zeros((total_points, 2), dtype=np.int32) - 1
    bb_rest_lengths = np.zeros((total_points, 2), dtype=np.float64)
    bb_partner_count = np.zeros(total_points, dtype=np.int32)
 
    # Compteurs statistiques pour le suivi des événements de liaison
    crosslinks_formed = 0
    crosslinks_broken = 0
    bb_bonds_formed = 0
    bb_bonds_broken = 0
    
    # ── Identification des points ancrés ──────────────────────────────────
    # Les points extrêmes de certaines fibres sont maintenus fixes : ils ne
    # reçoivent aucune mise à jour de position ni de vitesse.
    is_anchored = np.zeros(total_points, dtype=np.bool_)
    for i in range(total_points):
        fiber_id = fiber_ids[i]
        local_idx = point_in_fiber[i]
        fiber_len = fiber_lengths[fiber_id]
        anchor_start, anchor_end = anchor_configs[fiber_id]
        
        is_anchored[i] = (local_idx == 0 and anchor_start) or \
                        (local_idx == fiber_len - 1 and anchor_end)
    
    # ── Paramètres du hash spatial ────────────────────────────────────────
    # La taille de cellule est choisie légèrement supérieure à la distance
    # d'interaction maximale, garantissant que tous les voisins influents
    # se trouvent dans les 27 cellules adjacentes.
    max_interaction_dist = max(em_cutoff_distance, crosslink_distance, bb_bond_distance)
    cell_size = max_interaction_dist * 1.1
 
    # Cache de voisinage : pour chaque point, jusqu'à max_neighbors voisins
    # avec leurs distances pré-calculées. Évite de recalculer les distances
    # à chaque type de force.
    max_neighbors = 200
    neighbor_cache = np.zeros(total_points * max_neighbors, dtype=np.int32) - 1
    distance_cache = np.zeros(total_points * max_neighbors, dtype=np.float64)
    neighbor_counts = np.zeros(total_points, dtype=np.int32)
    
    # ═════════════════════════════════════════════════════════════════════
    # BOUCLE PRINCIPALE DE SIMULATION
    # ═════════════════════════════════════════════════════════════════════
    for step in range(num_steps):
        current_positions = position_data[:, step, :]
        current_velocities = velocity_data[:, step, :]
        
        # ── Mise à jour du hash spatial tous les 10 pas de temps ──────────
        # Hypothèse : les points bougent peu entre deux reconstructions.
        # Reconstruire à chaque pas serait trop coûteux en temps de calcul.
        if step % 10 == 0:
            cell_indices, cell_lists, cell_counts, grid_min, grid_size = \
                build_spatial_hash(current_positions, cell_size)
            
            # Remise à zéro du cache de voisins
            neighbor_counts[:] = 0
            
            # Construction de la liste de voisins pour chaque point
            for i in range(total_points):
                cx = int((current_positions[i, 0] - grid_min[0]) / cell_size)
                cy = int((current_positions[i, 1] - grid_min[1]) / cell_size)
                cz = int((current_positions[i, 2] - grid_min[2]) / cell_size)
                
                cx = max(0, min(cx, grid_size - 1))
                cy = max(0, min(cy, grid_size - 1))
                cz = max(0, min(cz, grid_size - 1))
                
                neighbor_cells = get_neighbor_cells(cx, cy, cz, grid_size)
                
                count = 0
                base_idx = i * max_neighbors
                
                for nc in neighbor_cells:
                    if nc < 0:
                        continue
                    
                    for idx in range(cell_counts[nc]):
                        j = cell_lists[nc, idx]
                        # On ne traite chaque paire (i,j) qu'une seule fois (j > i)
                        if j < 0 or j <= i:
                            continue
                        
                        displacement = current_positions[i] - current_positions[j]
                        distance = np.sqrt(np.sum(displacement**2))
                        
                        # On ne stocke que les voisins potentiellement influents
                        if distance < max_interaction_dist and count < max_neighbors:
                            neighbor_cache[base_idx + count] = j
                            distance_cache[base_idx + count] = distance
                            count += 1
                
                neighbor_counts[i] = count
            
            # ── Rupture des liaisons existantes trop étirées ──────────────
 
            for i in range(total_points):
                # Rupture des crosslinks A-B
                if crosslink_partner[i] >= 0 and i < crosslink_partner[i]:
                    j = crosslink_partner[i]
                    displacement = current_positions[i] - current_positions[j]
                    distance = np.sqrt(np.sum(displacement**2))
                    
                    if distance > crosslink_break_distance:
                        crosslink_partner[i] = -1
                        crosslink_partner[j] = -1
                        crosslink_rest_lengths[i] = 0.0
                        crosslink_rest_lengths[j] = 0.0
                        crosslinks_broken += 1
                
                # Rupture des liaisons B-B
                if labels[i] == 2:
                    for bond_idx in range(bb_partner_count[i]):
                        j = bb_partners[i, bond_idx]
                        if j >= 0 and i < j:
                            displacement = current_positions[i] - current_positions[j]
                            distance = np.sqrt(np.sum(displacement**2))
                            
                            if distance > bb_bond_break_distance:
                                # Suppression en décalant les entrées restantes
                                for k in range(bond_idx, bb_partner_count[i] - 1):
                                    bb_partners[i, k] = bb_partners[i, k + 1]
                                    bb_rest_lengths[i, k] = bb_rest_lengths[i, k + 1]
                                bb_partners[i, bb_partner_count[i] - 1] = -1
                                bb_partner_count[i] -= 1
                                
                                # Suppression symétrique chez le partenaire
                                for k in range(bb_partner_count[j]):
                                    if bb_partners[j, k] == i:
                                        for m in range(k, bb_partner_count[j] - 1):
                                            bb_partners[j, m] = bb_partners[j, m + 1]
                                            bb_rest_lengths[j, m] = bb_rest_lengths[j, m + 1]
                                        bb_partners[j, bb_partner_count[j] - 1] = -1
                                        bb_partner_count[j] -= 1
                                        break
                                
                                bb_bonds_broken += 1
            
            # ── Formation de nouvelles liaisons ───────────────────────────
            # On parcourt les paires voisines pré-calculées dans le cache
            for i in range(total_points):
                base_idx = i * max_neighbors
                
                for n_idx in range(neighbor_counts[i]):
                    j = neighbor_cache[base_idx + n_idx]
                    distance = distance_cache[base_idx + n_idx]
                    
                    if j < 0:
                        continue
                    
                    # Ignorer les voisins proches sur la même fibre
                    # (seuil de 4 positions arbitraire, évite l'auto-pontage)
                    if fiber_ids[i] == fiber_ids[j]:
                        if abs(point_in_fiber[i] - point_in_fiber[j]) <= 4:
                            continue
                    
                    # Tentative de formation d'un crosslink A-B
                    # Condition : les deux points sont libres ET de types opposés
                    if crosslink_partner[i] < 0 and crosslink_partner[j] < 0:
                        if (labels[i] == 1 and labels[j] == 2) or (labels[i] == 2 and labels[j] == 1):
                            if distance < crosslink_distance:
                                crosslink_partner[i] = j
                                crosslink_partner[j] = i
                                rest_len = distance * crosslink_rest_factor
                                crosslink_rest_lengths[i] = rest_len
                                crosslink_rest_lengths[j] = rest_len
                                crosslinks_formed += 1
                    
                    # Tentative de formation d'une liaison B-B
                    # Condition : les deux points sont B et ont encore de la capacité (< 2 partenaires)
                    if labels[i] == 2 and labels[j] == 2:
                        if bb_partner_count[i] < 2 and bb_partner_count[j] < 2:
                            already_bonded = False
                            for k in range(bb_partner_count[i]):
                                if bb_partners[i, k] == j:
                                    already_bonded = True
                                    break
                            
                            if not already_bonded and distance < bb_bond_distance:
                                bb_partners[i, bb_partner_count[i]] = j
                                bb_partners[j, bb_partner_count[j]] = i
                                rest_len = distance * bb_bond_rest_factor
                                bb_rest_lengths[i, bb_partner_count[i]] = rest_len
                                bb_rest_lengths[j, bb_partner_count[j]] = rest_len
                                bb_partner_count[i] += 1
                                bb_partner_count[j] += 1
                                bb_bonds_formed += 1
 
        # ── Mise à jour des positions et vitesses (Euler explicite) ───────
        for i in prange(total_points):
            # Les points ancrés restent immobiles
            if is_anchored[i]:
                position_data[i, step + 1, :] = position_data[i, step, :]
                velocity_data[i, step + 1, :] = velocity_data[i, step, :]
            else:
                position = current_positions[i]
                velocity = current_velocities[i]
                
                spring_force = np.zeros(3)
                bending_force = np.zeros(3)
                em_force = np.zeros(3)
                crosslink_force = np.zeros(3)
                bb_bond_force = np.zeros(3)
                
                # ── Force de ressort (élasticité longitudinale) ───────────
                # Appliquée aux deux liaisons voisines le long de la fibre
                prev_idx = prev_neighbors[i]
                next_idx = next_neighbors[i]
                
                if prev_idx >= 0:
                    position_before = current_positions[prev_idx]
                    displacement_before = position - position_before
                    length_before = np.sqrt(np.sum(displacement_before**2))
                    
                    if length_before > 1e-10:
                        unit_before = displacement_before / length_before
                        spring_force += -stiffnesses[i] * (length_before - rest_length) * unit_before
                
                if next_idx >= 0:
                    position_after = current_positions[next_idx]
                    displacement_after = position_after - position
                    length_after = np.sqrt(np.sum(displacement_after**2))
                    
                    if length_after > 1e-10:
                        unit_after = displacement_after / length_after
                        spring_force += stiffnesses[i] * (length_after - rest_length) * unit_after
                
                # ── Force de flexion (résistance à la courbure) ───────────
                # Calculée à partir du changement de tangente entre les deux
                # segments adjacents. Nécessite d'avoir un voisin de chaque côté.
                if prev_idx >= 0 and next_idx >= 0:
                    position_before = current_positions[prev_idx]
                    position_after = current_positions[next_idx]
                    
                    vec_before = position - position_before
                    vec_after = position_after - position
                    
                    len_before = np.sqrt(np.sum(vec_before**2))
                    len_after = np.sqrt(np.sum(vec_after**2))
                    
                    if len_before > 1e-10 and len_after > 1e-10:
                        unit_before = vec_before / len_before
                        unit_after = vec_after / len_after
                        tangent_change = unit_after - unit_before
                        avg_length = (len_before + len_after) / 2.0
                        # Vecteur courbure ≈ Δtangente / longueur_moyenne
                        curvature_vector = tangent_change / avg_length
                        bending_force = +bending_stiffness * curvature_vector
                
                # ── Force du crosslink A-B ────────────────────────────────
                if crosslink_partner[i] >= 0:
                    partner_idx = crosslink_partner[i]
                    partner_position = current_positions[partner_idx]
                    displacement = partner_position - position
                    distance = np.sqrt(np.sum(displacement**2))
                    
                    if distance > 1e-10:
                        unit_displacement = displacement / distance
                        rest_len = crosslink_rest_lengths[i]
                        crosslink_force += crosslink_stiffness * (distance - rest_len) * unit_displacement
 
                # ── Forces des liaisons covalentes B-B ────────────────────
                if labels[i] == 2:
                    for bond_idx in range(bb_partner_count[i]):
                        partner_idx = bb_partners[i, bond_idx]
                        if partner_idx >= 0:
                            partner_position = current_positions[partner_idx]
                            displacement = partner_position - position
                            distance = np.sqrt(np.sum(displacement**2))
                            
                            if distance > 1e-10:
                                unit_displacement = displacement / distance
                                rest_len = bb_rest_lengths[i, bond_idx]
                                bb_bond_force += bb_bond_stiffness * (distance - rest_len) * unit_displacement
                
                # ── Force EM (répulsion/attraction à courte portée) ────────
                # Désactivée si em_strength == 0.
                # Utilise le cache de voisins pré-calculé (mis à jour tous les 10 pas).
                # En dehors des pas de mise à jour, revient au calcul brut sur tous
                # les points (fallback O(N²), coûteux mais cohérent).
                if em_strength != 0.0:
                    if step % 10 == 0 and neighbor_counts[i] > 0:
                        base_idx = i * max_neighbors
                        
                        for n_idx in range(neighbor_counts[i]):
                            j = neighbor_cache[base_idx + n_idx]
                            distance = distance_cache[base_idx + n_idx]
                            
                            if j < 0:
                                continue
                            
                            # Ignorer les voisins directs sur la même fibre
                            if fiber_ids[j] == fiber_ids[i]:
                                if abs(point_in_fiber[j] - point_in_fiber[i]) <= 1:
                                    continue
                            
                            # Ignorer les points déjà liés (crosslink ou covalent)
                            if crosslink_partner[i] == j:
                                continue
                            
                            is_bb_bonded = False
                            for k in range(bb_partner_count[i]):
                                if bb_partners[i, k] == j:
                                    is_bb_bonded = True
                                    break
                            if is_bb_bonded:
                                continue
                            
                            if 1e-10 < distance < em_cutoff_distance:
                                other_position = current_positions[j]
                                displacement = position - other_position
                                unit_displacement = displacement / distance
                                # Loi en 1/r² (analogue coulombien / stérique)
                                force_magnitude = em_strength / (distance * distance)
                                em_force += force_magnitude * unit_displacement
                    else:
                        # Fallback : calcul brut O(N²) pour les pas sans mise à jour du cache
                        for j in range(total_points):
                            if j == i:
                                continue
                            
                            if fiber_ids[j] == fiber_ids[i]:
                                if abs(point_in_fiber[j] - point_in_fiber[i]) <= 1:
                                    continue
                            
                            if crosslink_partner[i] == j:
                                continue
                            
                            is_bb_bonded = False
                            for k in range(bb_partner_count[i]):
                                if bb_partners[i, k] == j:
                                    is_bb_bonded = True
                                    break
                            if is_bb_bonded:
                                continue
                            
                            other_position = current_positions[j]
                            displacement = position - other_position
                            distance = np.sqrt(np.sum(displacement**2))
                            
                            if 1e-10 < distance < em_cutoff_distance:
                                unit_displacement = displacement / distance
                                force_magnitude = em_strength / (distance * distance)
                                em_force += force_magnitude * unit_displacement
                
                # ── Intégration d'Euler explicite ─────────────────────────
                conservative_force = spring_force + bending_force + em_force + crosslink_force + bb_bond_force
                damping_force = -damping * velocity
                total_force = conservative_force + damping_force
                
                new_velocity = velocity + total_force * dt
                new_position = position + new_velocity * dt
                
                velocity_data[i, step + 1, :] = new_velocity
                position_data[i, step + 1, :] = new_position
    
    # ── Extraction des liaisons actives en fin de simulation ──────────────
    final_crosslinks = []
    for i in range(total_points):
        if crosslink_partner[i] >= 0 and i < crosslink_partner[i]:
            final_crosslinks.append([i, crosslink_partner[i]])
    
    final_bb_bonds = []
    for i in range(total_points):
        if labels[i] == 2:
            for bond_idx in range(bb_partner_count[i]):
                j = bb_partners[i, bond_idx]
                if j >= 0 and i < j:
                    final_bb_bonds.append([i, j])
    
    if len(final_crosslinks) > 0:
        final_crosslinks = np.array(final_crosslinks, dtype=np.int32)
    else:
        final_crosslinks = np.zeros((0, 2), dtype=np.int32)
    
    if len(final_bb_bonds) > 0:
        final_bb_bonds = np.array(final_bb_bonds, dtype=np.int32)
    else:
        final_bb_bonds = np.zeros((0, 2), dtype=np.int32)
    
    print(f"Crosslink statistics (A-B):")
    print(f"  - Total formed: {crosslinks_formed}")
    print(f"  - Total broken: {crosslinks_broken}")
    print(f"  - Currently active: {len(final_crosslinks)}")
    
    print(f"B-B Covalent bond statistics:")
    print(f"  - Total formed: {bb_bonds_formed}")
    print(f"  - Total broken: {bb_bonds_broken}")
    print(f"  - Currently active: {len(final_bb_bonds)}")
    
    return (position_data, velocity_data, fiber_ids, point_in_fiber, 
            labels, crosslink_partner, bb_partners, bb_partner_count,
            final_crosslinks, final_bb_bonds)
 
 
 
# =============================================================================
# COUCHE 3 : GÉNÉRATION DE GÉOMÉTRIE
# =============================================================================


 ####################
 ####FONCTION 3.1####
 ####################

def generate_isotropic_fiber_positions(num_fibers, num_points_per_fiber, point_spacing,
                                       cluster_radius):
    """
    Génère un milieu fibreux homogène et isotrope dans une sphère.
 
    ─── Homogénéité volumique ────────────────────────────────────────────────
    Les centres de fibre sont tirés uniformément EN VOLUME dans une sphère de
    rayon cluster_radius grâce à la transformation :
        r = cluster_radius * u^(1/3),  u ~ Uniforme[0, 1]
    Sans ce facteur 1/3, la densité serait proportionnelle à 1/r² (les zones
    centrales seraient sur-représentées). Ce tirage garantit que la probabilité
    d'avoir un centre dans un volume dV est proportionnelle à dV.
 
    ─── Isotropie du volume ──────────────────────────────
    Loi uniforme pour phi et cos theta
    
 
    Les fibres sont centrées sur leur point médian (pas d'ancrage) et reçoivent
    une légère perturbation gaussienne sur chaque point pour simuler la courbure
    naturelle des fibres biologiques.
 
    Paramètres
    ----------
    num_fibers : int
        Nombre de fibres à générer.
    num_points_per_fiber : int
        Nombre de points par fibre.
    point_spacing : float
        Distance entre deux points consécutifs d'une même fibre.
    cluster_radius : float
        Rayon de la sphère dans laquelle les centres sont distribués.
    
 
    Retourne
    --------
    fiber_positions : list of ndarray (num_points_per_fiber, 3)
    fiber_velocities : list of ndarray (num_points_per_fiber, 3)
    fiber_lengths : list of int
    anchor_configs : list of tuple (False, False)
        Aucun point ancré (fibres libres aux deux extrémités).
    """
    fiber_positions = []
    fiber_velocities = []
    fiber_lengths = []
    anchor_configs = []
 
    # Demi-longueur d'une fibre (pour centrer chaque fibre sur son point médian)
    half_length = (num_points_per_fiber - 1) * point_spacing / 2
 
    for fiber_idx in range(num_fibers):
 
        # ── Tirage uniforme en volume dans la sphère ──────────────────────
        u = np.random.rand()
        r = cluster_radius * u**(1/3)  # Correction volumique : r ~ r² dr
        cos_theta = 1 - 2 * np.random.rand()
        sin_theta = np.sqrt(1 - cos_theta**2)
        phi = 2 * np.pi * np.random.rand()
 
        x_c = r * sin_theta * np.cos(phi)
        y_c = r * sin_theta * np.sin(phi)
        z_c = r * cos_theta
 
        # ── Orientation de la fibre (von Mises autour de µ = π/2) ─────────
        # cos_theta pour la latitude (uniforme sur [-1,1])
        cos_theta = 1 - 2 * np.random.rand()
        sin_theta = np.sqrt(1 - cos_theta**2)
        mu = np.pi / 2
        
        # phi tiré selon von Mises, puis tronqué à [0, π] pour rester physique
        phi = 2 * np.pi * np.random.rand()
 
        dx = sin_theta * np.cos(phi)
        dy = sin_theta * np.sin(phi)
        dz = cos_theta
 
        positions = np.zeros((num_points_per_fiber, 3), dtype=np.float64)
 
        for i in range(num_points_per_fiber):
            # Paramètre curviligne centré sur 0 (de -half_length à +half_length)
            s = (i * point_spacing) - half_length
            x = x_c + s * dx
            y = y_c + s * dy
            z = z_c + s * dz
 
            # Perturbation gaussienne pour simuler la courbure naturelle
            curve_amplitude = 0.05 * point_spacing
            x += curve_amplitude * np.random.randn()
            y += curve_amplitude * np.random.randn()
            z += curve_amplitude * np.random.randn()
 
            positions[i] = [x, y, z]
 
        velocities = np.random.randn(num_points_per_fiber, 3) * 0.1
 
        fiber_positions.append(positions)
        fiber_velocities.append(velocities)
        fiber_lengths.append(num_points_per_fiber)
        anchor_configs.append((False, False))  # Aucun point ancré
 
    return fiber_positions, fiber_velocities, fiber_lengths, anchor_configs





 ####################
 ####FONCTION 3.2####
 ####################


def generate_anisotropic_fiber_positions(kappa, num_fibers, num_points_per_fiber,
                                           point_spacing, cluster_radius):
    """
    Génère un milieu fibreux anisotrope à distribution spatiale homogène en volume.
 
    Cette fonction combine les deux approches précédentes :
      - Distribution spatiale homogène en volume (r^(1/3)) de generate_isotropic_fiber_positions_2()
      - Orientation anisotrope pilotée par une distribution de von Mises
 
    ─── Différence par rapport à generate_isotropic_fiber_positions_2() ─────
    Ici, la distribution de von Mises est calculée via scipy.stats.vonmises
    (au lieu de np.random.vonmises) avec un repliement modulo π, ce qui peut
    donner une distribution légèrement différente en bord de domaine angulaire.
    La direction privilégiée est µ = π/2 (axe Y dans le plan XY).
 
    Pour un kappa élevé, les fibres s'orientent préférentiellement autour de
    l'axe Y, ce qui simule un tissu conjonctif ou un milieu fibreux orienté
    (tendons, muscles, etc.).
 
    Paramètres
    ----------
    kappa : float
        Paramètre de concentration de von Mises. 0 = isotrope, grand = très anisotrope.
    num_fibers : int
        Nombre de fibres à générer.
    num_points_per_fiber : int
        Nombre de points par fibre.
    point_spacing : float
        Distance entre deux points consécutifs d'une même fibre.
    cluster_radius : float
        Rayon de la sphère dans laquelle les centres sont distribués.
 
    Retourne
    --------
    fiber_positions : list of ndarray (num_points_per_fiber, 3)
    fiber_velocities : list of ndarray (num_points_per_fiber, 3)
    fiber_lengths : list of int
    anchor_configs : list of tuple (False, False)
        Aucun point ancré (fibres libres aux deux extrémités).
 
    Notes
    -----
    Contrairement à generate_anisotropic_fiber_positions() qui place les fibres
    sur une grille régulière, cette fonction réalise un tirage aléatoire des
    positions et orientations, ce qui est plus adapté à la simulation de milieux
    désordonnés anisotropes comme les hydrogels chargés ou les réseaux de
    collagène sous contrainte mécanique.
    """
    fiber_positions = []
    fiber_velocities = []
    fiber_lengths = []
    anchor_configs = []
 
    half_length = (num_points_per_fiber - 1) * point_spacing / 2
 
    for fiber_idx in range(num_fibers):
 
        # ── Tirage uniforme en volume dans la sphère ──────────────────────
        u = np.random.rand()
        r = cluster_radius * u**(1/3)  # Correction volumique
        cos_theta = 1 - 2 * np.random.rand()
        sin_theta = np.sqrt(1 - cos_theta**2)
        phi = 2 * np.pi * np.random.rand()
 
        x_c = r * sin_theta * np.cos(phi)
        y_c = r * sin_theta * np.sin(phi)
        z_c = r * cos_theta
 
        # ── Orientation anisotrope via von Mises (scipy) ───────────────────
        cos_theta = 1 - 2 * np.random.rand()
        sin_theta = np.sqrt(1 - cos_theta**2)
        mu = np.pi / 2  # Direction privilégiée : axe Y (phi = π/2)
        
        # vonmises.rvs de scipy + repliement dans [0, π] via modulo
        phi = np.mod(vonmises.rvs(kappa, loc=mu), np.pi)
 
        dx = sin_theta * np.cos(phi)
        dy = sin_theta * np.sin(phi)
        dz = cos_theta
 
        positions = np.zeros((num_points_per_fiber, 3), dtype=np.float64)
 
        for i in range(num_points_per_fiber):
            s = (i * point_spacing) - half_length
            x = x_c + s * dx
            y = y_c + s * dy
            z = z_c + s * dz
 
            # Perturbation gaussienne pour la courbure naturelle
            curve_amplitude = 0.05 * point_spacing
            x += curve_amplitude * np.random.randn()
            y += curve_amplitude * np.random.randn()
            z += curve_amplitude * np.random.randn()
 
            positions[i] = [x, y, z]
 
        velocities = np.random.randn(num_points_per_fiber, 3) * 0.1
 
        fiber_positions.append(positions)
        fiber_velocities.append(velocities)
        fiber_lengths.append(num_points_per_fiber)
        anchor_configs.append((False, False))  # Aucun point ancré
 
    return fiber_positions, fiber_velocities, fiber_lengths, anchor_configs
 
 
 
# =============================================================================
# COUCHE 4 : VISUALISATION ET SAUVEGARDE
# =============================================================================
 
 
 
 
 
 
 ####################
 ####FONCTION 4.1####
 ####################
 
def create_3d_animation(position_data, fiber_ids, point_in_fiber, labels,
                        crosslink_partner, bb_partners, bb_partner_count,
                        dt, fps=60, duration=10, 
                        filename="3d_fiber_animation.mp4"):
    """
    Génère une animation 3D de la simulation et la sauvegarde sur disque.
 
    La fonction sélectionne un sous-ensemble régulier de pas de temps parmi
    toutes les données simulées afin d'obtenir exactement `fps * duration` images,
    ce qui évite de créer des animations trop volumineuses.
 
    Les fibres sont tracées en noir, les crosslinks A-B en rouge et les liaisons
    B-B en bleu (plus épais). La caméra effectue une rotation lente autour de la
    scène pour améliorer la lisibilité 3D.
 
    La sauvegarde tente d'abord le codec H.264 (libx264) via ffmpeg ; en cas
    d'échec, elle replie sur un GIF Pillow à fps réduit.
 
    Paramètres
    ----------
    position_data : ndarray (total_points, num_steps+1, 3)
        Trajectoires complètes issues de compute_multi_fiber_dynamics_3d_optimized().
    fiber_ids : ndarray (int32)
        Indice de fibre de chaque point.
    point_in_fiber : ndarray (int32)
        Indice local de chaque point dans sa fibre.
    labels : ndarray (int32)
        Labels des points (0, 1=A, 2=B).
    crosslink_partner : ndarray (int32)
        Partenaire A-B de chaque point (-1 si aucun).
    bb_partners : ndarray (total_points, 2)
        Partenaires B-B de chaque point.
    bb_partner_count : ndarray (int32)
        Nombre de liaisons B-B par point.
    dt : float
        Pas de temps utilisé lors de la simulation (pour l'affichage du temps).
    fps : int
        Images par seconde de l'animation de sortie.
    duration : int
        Durée cible de l'animation en secondes.
    filename : str
        Chemin du fichier de sortie (.mp4 ou .gif en fallback).
 
    Retourne
    --------
    anim : matplotlib.animation.FuncAnimation
        Objet animation (peut être réutilisé ou affiché dans un notebook).
    """
    
    total_points, num_timesteps, _ = position_data.shape
    num_fibers = np.max(fiber_ids) + 1
    
    # Sélection des frames à afficher (sous-échantillonnage temporel)
    total_frames = fps * duration
    frame_step = max(1, num_timesteps // total_frames)
    selected_frames = np.arange(0, num_timesteps, frame_step)
    
    print(f"Creating 3D animation with {len(selected_frames)} frames at {fps} fps")
    
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calcul des limites de la scène à partir de toutes les positions
    all_positions = position_data.reshape(-1, 3)
    x_min, x_max = all_positions[:, 0].min() - 1, all_positions[:, 0].max() + 1
    y_min, y_max = all_positions[:, 1].min() - 1, all_positions[:, 1].max() + 1
    z_min, z_max = all_positions[:, 2].min() - 1, all_positions[:, 2].max() + 1
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min, z_max)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('3D Multi-Fiber Dynamics with A-B Crosslinking and B-B Covalent Bonds')
    
    # Création des objets graphiques (une ligne par fibre, puis pour les liaisons)
    fiber_lines = []
    for f in range(num_fibers):
        line, = ax.plot([], [], [], '-', color='black', linewidth=1.5, alpha=0.7)
        fiber_lines.append(line)
    
    # Lignes pour les crosslinks A-B (rouge)
    max_crosslinks = 1000
    crosslink_lines = []
    for _ in range(max_crosslinks):
        cl_line, = ax.plot([], [], [], 'r-', linewidth=1, alpha=0.5)
        crosslink_lines.append(cl_line)
    
    # Lignes pour les liaisons B-B (bleu, plus épaisses)
    max_bb_bonds = 1000
    bb_bond_lines = []
    for _ in range(max_bb_bonds):
        bb_line, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.7)
        bb_bond_lines.append(bb_line)
    
    # Texte d'information en incrustation
    time_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def animate(frame_idx):
        """Callback appelé par FuncAnimation pour chaque image."""
        timestep = selected_frames[frame_idx]
        current_positions = position_data[:, timestep, :]
        
        # Mise à jour des lignes de fibres
        for f in range(num_fibers):
            fiber_indices = np.where(fiber_ids == f)[0]
            sorted_indices = sorted(fiber_indices, key=lambda i: point_in_fiber[i])
            
            if len(sorted_indices) > 0:
                x_coords = current_positions[sorted_indices, 0]
                y_coords = current_positions[sorted_indices, 1]
                z_coords = current_positions[sorted_indices, 2]
                fiber_lines[f].set_data_3d(x_coords, y_coords, z_coords)
        
        # Mise à jour des crosslinks A-B
        crosslink_pairs = []
        for i in range(total_points):
            if crosslink_partner[i] >= 0 and i < crosslink_partner[i]:
                crosslink_pairs.append((i, crosslink_partner[i]))
        
        for i, (idx1, idx2) in enumerate(crosslink_pairs):
            if i < len(crosslink_lines):
                x_coords = [current_positions[idx1, 0], current_positions[idx2, 0]]
                y_coords = [current_positions[idx1, 1], current_positions[idx2, 1]]
                z_coords = [current_positions[idx1, 2], current_positions[idx2, 2]]
                crosslink_lines[i].set_data_3d(x_coords, y_coords, z_coords)
        
        # Effacement des lignes de crosslinks non utilisées
        for i in range(len(crosslink_pairs), len(crosslink_lines)):
            crosslink_lines[i].set_data_3d([], [], [])
        
        # Mise à jour des liaisons B-B
        bb_pairs = []
        for i in range(total_points):
            if labels[i] == 2:
                for bond_idx in range(bb_partner_count[i]):
                    j = bb_partners[i, bond_idx]
                    if j >= 0 and i < j:
                        bb_pairs.append((i, j))
        
        for i, (idx1, idx2) in enumerate(bb_pairs):
            if i < len(bb_bond_lines):
                x_coords = [current_positions[idx1, 0], current_positions[idx2, 0]]
                y_coords = [current_positions[idx1, 1], current_positions[idx2, 1]]
                z_coords = [current_positions[idx1, 2], current_positions[idx2, 2]]
                bb_bond_lines[i].set_data_3d(x_coords, y_coords, z_coords)
        
        for i in range(len(bb_pairs), len(bb_bond_lines)):
            bb_bond_lines[i].set_data_3d([], [], [])
        
        time_text.set_text(f'Time: {timestep * dt:.3f} s\n'
                          f'A-B Crosslinks: {len(crosslink_pairs)}\n'
                          f'B-B Covalent Bonds: {len(bb_pairs)}')
        
        # Rotation lente de la caméra pour une meilleure visualisation 3D
        ax.view_init(elev=20, azim=30 + frame_idx * 0.5)
        
        return fiber_lines + crosslink_lines + bb_bond_lines + [time_text]
    
    anim = animation.FuncAnimation(
        fig, animate, frames=len(selected_frames), 
        interval=1000/fps, blit=False, repeat=True
    )
    
    print(f"Saving 3D animation to {filename}...")
    try:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=fps, metadata=dict(artist='3D Fiber Simulation'), 
                       bitrate=6000, extra_args=['-vcodec', 'libx264'])
        anim.save(filename, writer=writer, dpi=300)
        print(f"Animation saved successfully as {filename}")
    except Exception as e:
        print(f"Error saving animation: {e}")
        print("Trying to save as GIF...")
        gif_filename = filename.replace('.mp4', '.gif')
        anim.save(gif_filename, writer='pillow', fps=fps//2)
        print(f"Saved as GIF: {gif_filename}")
    
    plt.close(fig)
    return anim






 ####################
 ####FONCTION 4.2####
 ####################
 
 
def create_interactive_3d_view(position_data, fiber_ids, point_in_fiber, labels,
                               crosslink_partner, bb_partners, bb_partner_count,
                               timestep=-1):
    """
    Affiche une vue 3D interactive (matplotlib) du réseau de fibres à un instant donné.
 
    Contrairement à create_3d_animation(), cette fonction génère une figure
    statique interactive que l'utilisateur peut faire pivoter à la souris.
    Elle est utile pour inspecter rapidement l'état final (timestep=-1 par défaut)
    ou n'importe quel pas de temps intermédiaire.
 
    Les fibres sont affichées en noir, les crosslinks A-B en rouge et les liaisons
    B-B en bleu. La légende n'est affichée que si au moins un type de liaison est
    présent.
 
    Paramètres
    ----------
    position_data : ndarray (total_points, num_steps+1, 3)
        Trajectoires complètes issues de compute_multi_fiber_dynamics_3d_optimized().
    fiber_ids : ndarray (int32)
        Indice de fibre de chaque point.
    point_in_fiber : ndarray (int32)
        Indice local de chaque point dans sa fibre.
    labels : ndarray (int32)
        Labels des points (0, 1=A, 2=B).
    crosslink_partner : ndarray (int32)
        Partenaire A-B de chaque point (-1 si aucun).
    bb_partners : ndarray (total_points, 2)
        Partenaires B-B de chaque point.
    bb_partner_count : ndarray (int32)
        Nombre de liaisons B-B par point.
    timestep : int
        Indice du pas de temps à visualiser. -1 = dernier pas de temps.
    """
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    current_positions = position_data[:, timestep, :]
    num_fibers = np.max(fiber_ids) + 1
    total_points = len(fiber_ids)
    
    colors = plt.cm.rainbow(np.linspace(0, 1, min(num_fibers, 20)))
    
    for f in range(num_fibers):
        fiber_indices = np.where(fiber_ids == f)[0]
        sorted_indices = sorted(fiber_indices, key=lambda i: point_in_fiber[i])
        
        if len(sorted_indices) > 0:
            x = current_positions[sorted_indices, 0]
            y = current_positions[sorted_indices, 1]
            z = current_positions[sorted_indices, 2]
            ax.plot(x, y, z, '-', color='black', linewidth=2, alpha=0.7)
    
    # Tracé des crosslinks A-B
    ab_count = 0
    for i in range(total_points):
        if crosslink_partner[i] >= 0 and i < crosslink_partner[i]:
            j = crosslink_partner[i]
            x = [current_positions[i, 0], current_positions[j, 0]]
            y = [current_positions[i, 1], current_positions[j, 1]]
            z = [current_positions[i, 2], current_positions[j, 2]]
            ax.plot(x, y, z, 'r-', linewidth=1, alpha=0.5, label='A-B' if ab_count == 0 else "")
            ab_count += 1
    
    # Tracé des liaisons B-B
    bb_count = 0
    for i in range(total_points):
        if labels[i] == 2:
            for bond_idx in range(bb_partner_count[i]):
                j = bb_partners[i, bond_idx]
                if j >= 0 and i < j:
                    x = [current_positions[i, 0], current_positions[j, 0]]
                    y = [current_positions[i, 1], current_positions[j, 1]]
                    z = [current_positions[i, 2], current_positions[j, 2]]
                    ax.plot(x, y, z, 'b-', linewidth=2, alpha=0.7, label='B-B' if bb_count == 0 else "")
                    bb_count += 1
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title(f'3D Fiber Network - Interactive View\n'
                f'A-B Crosslinks: {ab_count}, B-B Covalent Bonds: {bb_count}')
    
    if ab_count > 0 or bb_count > 0:
        ax.legend()
    
    plt.show()
 
 
 
 ####################
 ####FONCTION 4.3####
 ####################
 
def save_simulation_to_tar(position_data, velocity_data, fiber_ids, point_in_fiber,
                          labels, crosslink_partner, bb_partners, bb_partner_count,
                          fiber_lengths, simulation_params, filename="simulation_results.tar"):
    """
    Sauvegarde l'état final de la simulation dans une archive tar compressée.
 
    Seul le dernier pas de temps est archivé (et non l'intégralité des trajectoires),
    ce qui réduit drastiquement la taille sur disque tout en conservant les
    informations nécessaires pour une reprise de simulation ou une analyse post-hoc.
 
    Contenu de l'archive :
      - final_positions.npz      : positions à t_final
      - final_velocities.npz     : vitesses à t_final
      - fiber_ids.npz            : appartenance fibre de chaque point
      - point_in_fiber.npz       : indice local de chaque point
      - labels.npz               : labels A/B/neutre
      - fiber_lengths.npz        : longueurs des fibres
      - final_ab_crosslinks.npz  : liste des paires liées par crosslink A-B
      - final_bb_bonds.npz       : liste des paires liées par liaison B-B
      - parameters.json          : paramètres de simulation + statistiques finales
 
    Paramètres
    ----------
    position_data : ndarray (total_points, num_steps+1, 3)
        Trajectoires complètes (seul le dernier pas est extrait).
    velocity_data : ndarray (total_points, num_steps+1, 3)
        Vitesses complètes (seul le dernier pas est extrait).
    fiber_ids : ndarray (int32)
        Indice de fibre de chaque point.
    point_in_fiber : ndarray (int32)
        Indice local de chaque point dans sa fibre.
    labels : ndarray (int32)
        Labels des points.
    crosslink_partner : ndarray (int32)
        Partenaires A-B de chaque point.
    bb_partners : ndarray (total_points, 2)
        Partenaires B-B de chaque point.
    bb_partner_count : ndarray (int32)
        Nombre de liaisons B-B par point.
    fiber_lengths : list of int
        Longueurs des fibres.
    simulation_params : dict
        Dictionnaire des paramètres de simulation (sérialisable en JSON).
    filename : str
        Chemin du fichier tar de sortie.
    """
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        final_positions = position_data[:, -1, :]
        final_velocities = velocity_data[:, -1, :]
        
        np.savez_compressed(os.path.join(temp_dir, "final_positions.npz"), 
                           data=final_positions)
        np.savez_compressed(os.path.join(temp_dir, "final_velocities.npz"), 
                           data=final_velocities)
        np.savez_compressed(os.path.join(temp_dir, "fiber_ids.npz"), 
                           data=fiber_ids)
        np.savez_compressed(os.path.join(temp_dir, "point_in_fiber.npz"), 
                           data=point_in_fiber)
        np.savez_compressed(os.path.join(temp_dir, "labels.npz"), 
                           data=labels)
        np.savez_compressed(os.path.join(temp_dir, "fiber_lengths.npz"), 
                           data=np.array(fiber_lengths))
        
        # Construction des listes de liaisons finales pour l'archivage
        ab_bonds = []
        for i in range(len(crosslink_partner)):
            if crosslink_partner[i] >= 0 and i < crosslink_partner[i]:
                ab_bonds.append([i, crosslink_partner[i]])
        ab_bonds = np.array(ab_bonds, dtype=np.int32) if len(ab_bonds) > 0 else np.zeros((0, 2), dtype=np.int32)
        np.savez_compressed(os.path.join(temp_dir, "final_ab_crosslinks.npz"), 
                           data=ab_bonds)
        
        bb_bonds = []
        for i in range(len(bb_partner_count)):
            if labels[i] == 2:
                for bond_idx in range(bb_partner_count[i]):
                    j = bb_partners[i, bond_idx]
                    if j >= 0 and i < j:
                        bb_bonds.append([i, j])
        bb_bonds = np.array(bb_bonds, dtype=np.int32) if len(bb_bonds) > 0 else np.zeros((0, 2), dtype=np.int32)
        np.savez_compressed(os.path.join(temp_dir, "final_bb_bonds.npz"), 
                           data=bb_bonds)
        
        # Ajout des statistiques finales dans le JSON de paramètres
        simulation_params["final_state_stats"] = {
            "total_ab_crosslinks": len(ab_bonds),
            "total_bb_bonds": len(bb_bonds),
            "timestep_saved": position_data.shape[1] - 1
        }
        
        with open(os.path.join(temp_dir, "parameters.json"), 'w') as f:
            json.dump(simulation_params, f, indent=2)
        
        with tarfile.open(filename, 'w') as tar:
            for item in os.listdir(temp_dir):
                tar.add(os.path.join(temp_dir, item), arcname=item)
        
        print(f"Final state simulation data saved to {filename}")
        file_size = os.path.getsize(filename) / (1024 * 1024)
        print(f"Archive size: {file_size:.2f} MB")
        print(f"Final state contains:")
        print(f"  - {len(ab_bonds)} A-B crosslinks")
        print(f"  - {len(bb_bonds)} B-B covalent bonds")
        
    finally:
        shutil.rmtree(temp_dir)
        
        
