import numpy as np
import matplotlib.pyplot as plt
import bibliotheque_etude_milieu as bem

# ===================================================
# 1. CHARGEMENT DES DONNÉES ET PARAMÈTRES
# ===================================================

try:
    position_data = np.load("positions.npy")
    fiber_ids = np.load("fiber_ids.npy")
    # On travaille par défaut sur le dernier pas de temps pour l'analyse statique
    pos_final = position_data[:, -1, :] 
except FileNotFoundError:
    print("Erreur : Fichiers de données introuvables.")
    # Simulation de données vides pour éviter les erreurs de syntaxe si test sans fichiers
    pos_final = np.zeros((100, 3))
    fiber_ids = np.zeros(100)

# Paramètres physiques
em_cutoff_distance = 2.0
fiber_radius = em_cutoff_distance / 2
cluster_radius = 100

# ===================================================
# 2. ANALYSE DE L'ISOTROPIE (ORIENTATIONS)
# ===================================================
if True:
    print("--- Analyse de l'orientation des fibres ---")
    directions, theta, phi, cos_theta = bem.compute_fiber_orientation_distribution(
        position_data, 
        fiber_ids
    )
    
    bem.plot_orientation_distribution(theta, phi, cos_theta)


# ===================================================
# 3. ANALYSE DE LA POROSITÉ RADIALE (PROFIL)
# ===================================================
if False:
    print("--- Analyse du profil radial de porosité ---")
    # Épaisseur de la coquille de mesure (µm)
    ecart_mesure = 40 
    
    bem.plot_radial_analysis(
        ecart_mesure, 
        fiber_ids, 
        pos_final, 
        fiber_radius, 
        cluster_radius
    )


# ===================================================
# 4. ÉTUDE DE CONVERGENCE (STABILITÉ DE LA MESURE)
# ===================================================
if False:
    print("--- Analyse de la convergence de l'épaisseur ---")
    # Teste les épaisseurs de coquille de 1 à 50 µm
    bem.plot_convergence_analysis(
        max_ecart=50, 
        fiber_ids=fiber_ids, 
        position_data_f=pos_final, 
        fiber_radius=fiber_radius, 
        cluster_radius=cluster_radius
    )


# ===================================================
# 5. CARTOGRAPHIE CARTÉSIENNE (HEATMAPS)
# ===================================================
if False:
    print("--- Génération des cartes de porosité par fenêtres glissantes (Plan X=0) ---")
    
    # On définit directement le nombre de points de mesure (résolution de la map)
    # Plus le nombre est élevé, plus la carte sera "lisse" (mais calcul long)
    resolutions = [10] 
    
    # On fixe la taille du carré de mesure (ex: 70µm pour avoir une bonne statistique)
    taille_fenetre = 70
    
    for nb_pts in resolutions:
        print(f"\nCalcul en cours : Grille de {nb_pts}x{nb_pts} points")
        print(f"Chaque point mesure la porosité sur un carré de {taille_fenetre}x{taille_fenetre} µm")
        
        # Utilisation de la version avec chevauchement (overlapping)
        bem.color_map_overlapping_grids(
            nb_points=nb_pts,
            cote_carre=taille_fenetre,
            fiber_ids=fiber_ids, 
            position_data_f=pos_final, 
            fiber_radius=fiber_radius, 
            area_size=90
        )

# ===================================================
# 6. CALCUL DU RAYON Rm POUR UN Vmin CONSTANT
# ===================================================
if False:
    print("--- Calcul de Rm en fonction de l'écart x ---")
    
    Rmin=40
    # Définition du volume minimum souhaité (ex: volume d'une sphère de rayon 20)
    V_min_target = (4/3) * np.pi * (Rmin**3) 
    
    # Gamme d'écarts (x) de 0.1 à 40
    x_range = np.linspace(0.1, 40, 100)
    
    # Calcul des Rm
    Rm_results = bem.compute_Rm_from_Vmin(V_min_target, x_range)
    
    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x_range, Rm_results, color='purple', lw=2, label=f"$R_{{min}}$ = {Rmin:.0f} µm")
    
    plt.title("Rayon interne $R_m$ nécessaire pour maintenir un volume de contrôle représentatif pour un écart fixé")
    plt.xlabel("Épaisseur de la coquille $x$ (µm)")
    plt.ylabel("Rayon interne $R_m$ (µm)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Note : Si x augmente, Rm doit diminuer pour que le volume reste identique
    plt.show()