import numpy as np
import bibliotheque_diffusion_milieu_poreux as bib

# --- 1. CONFIGURATION ---
ACTIVER_VIDEO = True 
VOIR_PHOTO = True
L = 90
n_div = 5
step = L / n_div
half_L = L / 2

R_FIBRE = 0.5
R_PART = 1
R_COLLISION = R_FIBRE + R_PART

# Paramètres Simulation
n_iterations = 10000
dt = 0.005 # Pas de temps plus petit = meilleure précision de collision
sigma = 2.0
COEFF_RESTITUTION = 0.8 # Dissipation : 1.0 = élastique, < 1.0 = réaliste (perte d'énergie)

# Initialisation
traj = np.zeros((n_iterations + 1, 3))
traj_libre = np.zeros((n_iterations + 1, 3))
traj[0] = [0, 0, 0]
traj_libre[0] = [0, 0, 0]

dists_log = [] 
coll_count = 0
stop_reason = "Limite d'itérations atteinte"

# Chargement des données
try:
    position_data = np.load("positions.npy")
    fiber_ids = np.load("fiber_ids.npy")
    pos_fibers_raw = position_data[:, -1, :]
except:
    pos_fibers_raw = np.random.uniform(-half_L, half_L, (1000, 3))
    fiber_ids = np.repeat(np.arange(100), 10)

voxels_fibres = bib.preparer_voxels(fiber_ids, pos_fibers_raw, n_div, L)

# --- 2. CALCUL ---

print("Calcul de la trajectoire...")

for i in range(n_iterations):
    pas_brownien = np.random.normal(0, sigma * np.sqrt(dt), size=3)
    
    # 1. Trajectoire LIBRE (Sert de référence)
    traj_libre[i+1] = traj_libre[i] + pas_brownien
    
    # 2. Trajectoire AVEC FIBRES
    new_p = traj[i] + pas_brownien
    
    # --- CONDITION D'ARRÊT : Sortie du cube ---
    if np.any(np.abs(new_p) > half_L):
        stop_reason = "Sortie du domaine (Cube L)"
        break # On quitte la boucle immédiatement
    
    # Recherche de collision
    v_idx = tuple(np.clip(((new_p + half_L) / step).astype(int), 0, n_div - 1))
    segs_data = voxels_fibres.get(v_idx, [])
    
    d_min, best_proj, hit_now = 1e10, None, False
    for (P1, P2, A, B) in segs_data:
        proj = bib.projection_sur_droite(new_p, P1, P2)
        if bib.est_sur_segment(proj, A, B):
            dist = np.linalg.norm(new_p - proj)
            if dist < d_min:
                d_min, best_proj = dist, proj
    
    # --- REBOND AVEC DISSIPATION ---
    if best_proj is not None and d_min < R_COLLISION:
        coll_count += 1
        hit_now = True
        normal = (new_p - best_proj) / (d_min + 1e-12)
        penetration = R_COLLISION - d_min
        
        # Réflexion dissipative : on ramène la particule avec un facteur (1 + e)
        # Cela réduit la "vitesse" de sortie après l'impact
        new_p = new_p + (1 + COEFF_RESTITUTION) * penetration * normal
        d_min = R_COLLISION
        
    traj[i+1] = new_p
    dists_log.append({'d': d_min, 'n': coll_count, 'hit': hit_now, 't': (i+1)*dt})

# --- 3. ANALYSE ---

# On tronque les trajectoires au moment du break
traj_reelle = traj[:len(dists_log)+1]
traj_libre_reelle = traj_libre[:len(dists_log)+1]

D_reel = bib.calculer_diffusion(traj_reelle, dt)
D_libre = bib.calculer_diffusion(traj_libre_reelle, dt)

print(f"\nBilan : {stop_reason}")
print(f"D (Libre) : {D_libre:.4f} | D (Milieu) : {D_reel:.4f}")
print(f"Tortuosité (D_libre/D_milieu) : {D_libre/max(D_reel, 1e-10):.2f}")
print(f"Nombre d'itérations : {n_iterations} | Nombre de collisioons : {coll_count}")

# --- 4. AFFICHAGE ---
t_total_max = n_iterations * dt
if ACTIVER_VIDEO:
    bib.lancer_animation_3d(traj, dists_log, voxels_fibres, L, step, half_L, R_FIBRE, t_total_max)

if VOIR_PHOTO:
    bib.afficher_photo_finale(traj_reelle, voxels_fibres, half_L, t_total_max, dt, coll_count, n_div, stop_reason)