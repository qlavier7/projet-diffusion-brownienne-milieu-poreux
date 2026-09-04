import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import circmean, circstd

# ===================================================
# FONCTIONS DE CALCUL DE POROSITE
# ===================================================

def compute_sphere_porosity_at_0(radius, fiber_ids, position_data, fiber_radius):
    """
    Calcule la porosité à l'intérieur d'une sphère centrée à l'origine.
    Note : position_data doit être en 2D (N, 3) pour un instant T donné.
    """
    sphere_volume = (4/3) * np.pi * radius**3
    fiber_volume = 0.0
    
    # Précision : On s'assure que les données sont des arrays
    fiber_ids = np.asanyarray(fiber_ids)
    position_data = np.asanyarray(position_data)

    def segment_length_in_sphere(p1, p2, R):
        d = p2 - p1
        a = np.dot(d, d)
        if a == 0: return 0.0
        b = 2 * np.dot(p1, d)
        c = np.dot(p1, p1) - R**2
        discriminant = b**2 - 4 * a * c

        if discriminant < 0: return 0.0
        sqrt_disc = np.sqrt(discriminant)
        t1, t2 = (-b - sqrt_disc) / (2 * a), (-b + sqrt_disc) / (2 * a)
        t_min, t_max = max(0.0, min(t1, t2)), min(1.0, max(t1, t2))
        
        if t_max <= t_min: return 0.0
        return np.linalg.norm((p1 + t_max * d) - (p1 + t_min * d))

    for fid in np.unique(fiber_ids):
        indices = np.where(fiber_ids == fid)[0]
        for i in range(len(indices) - 1):
            p1, p2 = position_data[indices[i]], position_data[indices[i + 1]]
            length = segment_length_in_sphere(p1, p2, radius)
            fiber_volume += np.pi * fiber_radius**2 * length

    porosity = max(1.0 - fiber_volume / sphere_volume, 0.0)
    return porosity, fiber_volume, sphere_volume


def compute_cube_porosity(center, cote, fiber_ids, position_data, fiber_radius):
    """Calcule la porosité dans un volume cubique aligné sur les axes."""
    cube_volume = cote**3
    fiber_volume = 0.0
    center = np.array(center)

    def segment_length_in_cube(p1, p2, L, C):
        xmin, xmax = C[0] - L/2, C[0] + L/2
        ymin, ymax = C[1] - L/2, C[1] + L/2
        zmin, zmax = C[2] - L/2, C[2] + L/2
        d = p2 - p1
        t0, t1 = 0.0, 1.0
        bounds = [(xmin, xmax), (ymin, ymax), (zmin, zmax)]

        for i in range(3):
            if abs(d[i]) < 1e-12:
                if p1[i] < bounds[i][0] or p1[i] > bounds[i][1]: return 0.0
            else:
                inv_d = 1.0 / d[i]
                t_near = (bounds[i][0] - p1[i]) * inv_d
                t_far = (bounds[i][1] - p1[i]) * inv_d
                t0 = max(t0, min(t_near, t_far))
                t1 = min(t1, max(t_near, t_far))
                if t0 > t1: return 0.0
        return np.linalg.norm(d) * (t1 - t0)

    for fid in np.unique(fiber_ids):
        indices = np.where(fiber_ids == fid)[0]
        for i in range(len(indices) - 1):
            length = segment_length_in_cube(position_data[indices[i]], position_data[indices[i+1]], cote, center)
            fiber_volume += np.pi * fiber_radius**2 * length

    return max(1.0 - fiber_volume / cube_volume, 0.0)


# ===================================================
# ANALYSE RADIALE
# ===================================================

def calcul_porosite_rad(ecart, fiber_ids, position_data_f, fiber_radius, cluster_radius):
    """
    Calcule la porosité par couches sphériques concentriques (coquilles).
    """
    porosite_rad, radius_ext, radius_int = [], [], []
    
    for r in range(ecart, int(cluster_radius), 1):
        # Sphère externe
        _, vol_fib_ext, vol_sph_ext = compute_sphere_porosity_at_0(r, fiber_ids, position_data_f, fiber_radius)
        
        # Sphère interne
        if (r - ecart) <= 0:
            vol_fib_int, vol_sph_int = 0.0, 0.0
        else:
            _, vol_fib_int, vol_sph_int = compute_sphere_porosity_at_0(r - ecart, fiber_ids, position_data_f, fiber_radius)
            
        shell_fiber_vol = vol_fib_ext - vol_fib_int
        shell_total_vol = vol_sph_ext - vol_sph_int
        
        porosite = max(1.0 - shell_fiber_vol / shell_total_vol, 0.0)
        porosite_rad.append(porosite)
        radius_ext.append(r)
        radius_int.append(r - ecart)
        
    return np.array(porosite_rad), np.array(radius_ext), np.array(radius_int)


def plot_radial_analysis(ecart, fiber_ids, position_data_f, fiber_radius, cluster_radius):
    """Génère le graphique de la porosité radiale pour un écart donné."""
    p_rad, r_ext, r_int = calcul_porosite_rad(ecart, fiber_ids, position_data_f, fiber_radius, cluster_radius)
    
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax1.plot(r_ext, p_rad, color='tab:blue', lw=2, label="Porosité locale")
    ax1.set_xlabel("Rayon extérieur (µm)")
    ax1.set_ylabel("Porosité")
    ax1.set_title(f"Profil radial de porosité (Épaisseur coquille : {ecart} µm)")
    ax1.grid(True, alpha=0.3)

    # Statistiques
    moy, std_rel = np.mean(p_rad), np.std(p_rad)
    stats_text = f"Moyenne : {moy:.4f}\nÉcart-type réel. : {std_rel:.5f}%"
    ax1.text(0.95, 0.05, stats_text, transform=ax1.transAxes, ha='right', bbox=dict(facecolor='white', alpha=0.8))
    plt.show()


def plot_convergence_analysis(max_ecart, fiber_ids, position_data_f, fiber_radius, cluster_radius):
    """Analyse l'influence de l'épaisseur de mesure (ecart) sur la stabilité des résultats."""
    moyennes, et_reels, ecarts = [], [], range(1, max_ecart)
    
    for e in ecarts:
        p_rad, _, _ = calcul_porosite_rad(e, fiber_ids, position_data_f, fiber_radius, cluster_radius)
        m = np.mean(p_rad)
        moyennes.append(m)
        et_reels.append(np.std(p_rad))
        
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(ecarts, moyennes, marker='o', markersize=3)
    plt.title("Convergence de la Porosité Moyenne")
    plt.xlabel("Épaisseur de la coquille (µm)")
    plt.ylabel("Moyenne")

    plt.subplot(1, 2, 2)
    plt.plot(ecarts, et_reels, color='red', marker='s', markersize=3)
    plt.title("Stabilité de la mesure (ET réel)")
    plt.xlabel("Épaisseur de la coquille (µm)")
    plt.ylabel("Incertitude (%)")
    plt.tight_layout()
    plt.show()


# ===================================================
# ANALYSE CARTÉSIENNE
# ===================================================
def color_map_overlapping_grids(nb_points, cote_carre, fiber_ids, position_data_f, fiber_radius, area_size=90):
    """
    Génère 3 heatmaps en limitant les centres des cubes pour ne jamais déborder
    de la zone area_size (évite les effets de bord).
    """
    # Rayon du cube de mesure
    r_cube = cote_carre / 2
    
    # La zone où le centre peut se déplacer sans que le bord du cube ne sorte :
    # Si area_size = 90 et cote = 40, la limite est 45 - 20 = 25.
    # Les centres iront de -25 à +25.
    limite = (area_size / 2) - r_cube
    
    if limite <= 0:
        print("Erreur : Le côté du cube est plus grand que la zone d'étude !")
        return
        
    # On répartit les nb_points entre -limite et +limite
    offsets = np.linspace(-limite, limite, nb_points)
    
    # Initialisation des grilles
    grid_x = np.zeros((nb_points, nb_points))
    grid_y = np.zeros((nb_points, nb_points))
    grid_z = np.zeros((nb_points, nb_points))

    print(f"Analyse sécurisée sur la zone [{-limite:.1f}, {limite:.1f}]")

    # Remplissage (Cubes centrés dans la zone de sécurité)
    for i, a1 in enumerate(offsets):
        for j, a2 in enumerate(offsets):
            grid_x[i, j] = compute_cube_porosity([0, a1, a2], cote_carre, fiber_ids, position_data_f, fiber_radius)
            grid_y[i, j] = compute_cube_porosity([a1, 0, a2], cote_carre, fiber_ids, position_data_f, fiber_radius)
            grid_z[i, j] = compute_cube_porosity([a1, a2, 0], cote_carre, fiber_ids, position_data_f, fiber_radius)

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    grids = [grid_x, grid_y, grid_z]
    titles = ["Plan YZ (X=0)", "Plan XZ (Y=0)", "Plan XY (Z=0)"]
    
    # On ajuste 'extent' pour refléter la zone RÉELLEMENT couverte par les centres
    # Ou par les bords des cubes (ici on affiche la zone couverte par les bords)
    display_extent = [-limite - r_cube, limite + r_cube, -limite - r_cube, limite + r_cube]

    for k in range(3):
        im = axes[k].imshow(grids[k], origin="lower", extent=display_extent,
                            cmap='viridis', interpolation='none')
        
        moy = np.mean(grids[k])
        sigma = np.std(grids[k])
        
        stats_text = f"Moyenne : {moy:.4f}\nSigma : {sigma:.4f}"
        axes[k].text(0.05, 0.95, stats_text, transform=axes[k].transAxes, 
                     fontsize=10, fontweight='bold', verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[k].set_title(titles[k])

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Porosité volumique")
    plt.suptitle(f"Heatmaps Isotropes (Zone de sécurité, Cube : {cote_carre}µm)", fontsize=15)
    plt.show()

# ===================================================
# ORIENTATION ET ISOTROPIE
# ===================================================

def compute_fiber_orientation_distribution(position_data, fiber_ids):
    
    # calcule vecteurs directeurs et angles des fibres
    pos = position_data[:, -1, :]
    
    unique_fibers = np.unique(fiber_ids)
    directions = []
    
    for fid in unique_fibers:
        indices = np.where(fiber_ids == fid)[0]
        fiber_points = pos[indices]
        
        start = fiber_points[0]
        end = fiber_points[-1]
        
        direction = end - start
        norm = np.linalg.norm(direction)
        
        if norm > 0:
            direction /= norm
            directions.append(direction)
    
    directions = np.array(directions)
    
    # Coordonnées sphériques
    theta = np.arccos(directions[:,2])     # angle polaire
    phi = np.arctan2(directions[:,1], directions[:,0])
    
    cos_theta = directions[:,2]
    
    return directions, theta, phi, cos_theta

def plot_orientation_distribution(theta, phi, cos_theta):
    #affiche les distributions angulaires de phi cos theta et theta
    # ---- Statistiques ----
    phi_2pi = np.mod(phi, 2*np.pi)
    mean_cos = np.mean(cos_theta)
    std_cos  = np.std(cos_theta)

    mean_theta = np.mean(theta)
    std_theta  = np.std(theta)

    # Statistiques circulaires pour phi
    mean_phi = circmean(phi_2pi, high=2*np.pi, low=0)
    std_phi  = circstd(phi_2pi, high=2*np.pi, low=0)

    print("Statistiques sur l’orientation :")
    print(f"cos(θ)  -> moyenne = {mean_cos:.4f}, écart-type = {std_cos:.4f}")
    print(f"θ       -> moyenne = {mean_theta:.4f}, écart-type = {std_theta:.4f}")
    print(f"φ       -> moyenne = {mean_phi:.4f}, écart-type circulaire = {std_phi:.4f}")

    # ---- Graphiques ----
    plt.figure(figsize=(12,4))

    # Distribution phi
    plt.subplot(1,3,1)
    plt.hist(phi_2pi, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(mean_phi, color='red', linestyle='--', label=f"moy={mean_phi:.2f}")
    plt.title("Distribution de φ")
    plt.legend()

    # Distribution theta
    plt.subplot(1,3,2)
    plt.hist(theta, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(mean_theta, color='red', linestyle='--', label=f"moy={mean_theta:.2f}")
    plt.title("Distribution de θ")
    plt.legend()

    # Distribution cos(theta)
    plt.subplot(1,3,3)
    plt.hist(cos_theta, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(mean_cos, color='red', linestyle='--', label=f"moy={mean_cos:.2f}")
    plt.title("Distribution de cos(θ)")
    plt.legend()

    # ---- Ajouter encadré global avec toutes les stats ----
    stats_text = (
        f"cos(θ) -> moy={mean_cos:.4f}, σ={std_cos:.4f}\n"
        f"θ      -> moy={mean_theta:.4f}, σ={std_theta:.4f}\n"
        f"φ      -> moy={mean_phi:.4f}, σ_circ={std_phi:.4f}"
    )
    # ---- Ajouter encadré global avec toutes les stats ----
    fig = plt.gcf()
    # Coordonnées relatives à la figure (0-1), en dehors des sousplots
    fig.text(
        0.98, 0.995, stats_text,
        fontsize=10,
        verticalalignment='top',
        horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='black')
            )


    plt.tight_layout()
    plt.show()

    return mean_cos, std_cos, mean_theta, std_theta, mean_phi, std_phi
    
    
def compute_Rm_from_Vmin(Vmin, ecart_range):
    """
    Calcule le rayon interne Rm pour un volume de coquille constant Vmin
    en fonction de l'épaisseur 'x' (ecart).
    
    Résout 3x*Rm² + 3x²*Rm + (x³ - 3Vmin/4pi) = 0
    """
    Rm_values = []
    
    for x in ecart_range:
        if x <= 0:
            Rm_values.append(np.nan)
            continue
            
        # Coefficients de l'équation quadratique aR² + bR + c = 0
        a = 3 * x
        b = 3 * (x**2)
        c = (x**3) - (3 * Vmin) / (4 * np.pi)
        
        delta = b**2 - 4 * a * c
        
        if delta < 0:
            Rm_values.append(0.0) # Ou nan si le volume est impossible
        else:
            # On prend la racine positive
            sol = (-b + np.sqrt(delta)) / (2 * a)
            Rm_values.append(max(0, sol))
            
    return np.array(Rm_values)

