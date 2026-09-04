import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# --- GÉOMÉTRIE ET PROJECTION ---

def projection_sur_droite(P, A, B):
    AB = B - A
    mag2 = np.dot(AB, AB)
    if mag2 < 1e-12: return A
    t = np.dot(P - A, AB) / mag2
    return A + t * AB

def est_sur_segment(Proj, A, B):
    # Vérifie si Proj est entre A et B
    return np.dot(Proj - A, B - A) >= 0 and np.dot(Proj - B, A - B) >= 0

def get_segment_cube_intersection(p1, p2, v_min, v_max):
    t_min, t_max = 0.0, 1.0
    d = p2 - p1
    for i in range(3):
        if abs(d[i]) < 1e-10:
            if p1[i] < v_min[i] or p1[i] > v_max[i]: return None
        else:
            t1 = (v_min[i] - p1[i]) / d[i]
            t2 = (v_max[i] - p1[i]) / d[i]
            t_min = max(t_min, min(t1, t2))
            t_max = min(t_max, max(t1, t2))
    return (p1 + t_min * d, p1 + t_max * d) if t_min < t_max else None

def preparer_voxels(fiber_ids, pos, n_div, L):
    vox_dict = {}
    h_L, s_step = L/2, L/n_div
    for i in range(n_div):
        for j in range(n_div):
            for k in range(n_div):
                v_min = -h_L + np.array([i, j, k]) * s_step
                v_max = v_min + s_step
                segs_data = []
                for fid in np.unique(fiber_ids):
                    pts = pos[fiber_ids == fid]
                    for idx_p in range(len(pts)-1):
                        A, B = pts[idx_p], pts[idx_p+1]
                        inter = get_segment_cube_intersection(A, B, v_min, v_max)
                        if inter:
                            segs_data.append((inter[0], inter[1], A, B))
                if segs_data: vox_dict[(i,j,k)] = segs_data
    return vox_dict

# --- DIFFUSION ---

def calculer_diffusion(trajectoire, dt):
    if len(trajectoire) < 2: return 0
    msd = np.sum((trajectoire - trajectoire[0])**2, axis=1)
    temps = np.linspace(0, len(trajectoire)*dt, len(trajectoire))
    pente, _ = np.polyfit(temps, msd, 1)
    return pente / 6

# --- VISUALISATION ---

def draw_cube_outline(ax, v_min, s):
    pts = [v_min + np.array([x, y, z])*s for x in [0,1] for y in [0,1] for z in [0,1]]
    edges = [[0,1],[1,3],[3,2],[2,0],[4,5],[5,7],[7,6],[6,4],[0,4],[1,5],[2,6],[3,7]]
    for e in edges:
        ax.plot([pts[e[0]][0], pts[e[1]][0]], [pts[e[0]][1], pts[e[1]][1]], 
                [pts[e[0]][2], pts[e[1]][2]], color='white', lw=0.5, alpha=0.3)

def lancer_animation_3d(traj, dists_log, voxels_fibres, L, step, half_L, R_FIBRE, t_total_max):
    fig = plt.figure(figsize=(10, 8), facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    dot, = ax.plot([], [], [], 'wo', ms=10, zorder=15) # Point plus gros (R_PART=3)
    current_v = [None]

    def update(frame):
        nonlocal dot
        if frame >= len(dists_log) or frame == 0: return dot,
        p = traj[frame]
        info = dists_log[frame-1]
        v_idx = tuple(np.clip(((p + half_L) / step).astype(int), 0, int(L/step) - 1))
        
        if v_idx != current_v[0]:
            current_v[0] = v_idx
            ax.clear()
            ax.set_facecolor('black')
            v_min = -half_L + np.array(v_idx) * step
            ax.set_xlim(v_min[0], v_min[0]+step); ax.set_ylim(v_min[1], v_min[1]+step); ax.set_zlim(v_min[2], v_min[2]+step)
            draw_cube_outline(ax, v_min, step)
            segs = voxels_fibres.get(v_idx, [])
            if segs:
                # On n'affiche que P1-P2 pour la vidéo
                lines = [[s[0], s[1]] for s in segs]
                lc = Line3DCollection(lines, colors='#ff4444', linewidths=1.5, alpha=0.6)
                ax.add_collection3d(lc)
            ax.set_axis_off()
            dot, = ax.plot([], [], [], 'wo', ms=10, zorder=15)

        color = '#ff3333' if info['hit'] else '#00d4ff'
        ax.plot(traj[frame-1:frame+1, 0], traj[frame-1:frame+1, 1], traj[frame-1:frame+1, 2], color=color, lw=2)
        dot.set_data([p[0]], [p[1]])
        dot.set_3d_properties([p[2]])
        dot.set_color('#ffcc00' if info['hit'] else 'white')
        ax.set_title(f"T: {info['t']:.2f}s | COLL: {info['n']}", color='white')
        return dot,

    ani = FuncAnimation(fig, update, frames=len(traj), interval=30, blit=False, repeat=False)
    plt.show()

def afficher_photo_finale(traj_reelle, voxels_fibres, half_L, t_total_max, dt, coll_count, n_div, stop_reason): 
    """ 
    Affiche le rendu statique final avec cage, bouton de masquage et statistiques détaillées. 
    """ 
    import matplotlib.pyplot as plt 
    from matplotlib.widgets import Button 
    from mpl_toolkits.mplot3d.art3d import Line3DCollection 
    import numpy as np

    # Configuration du style sombre
    plt.style.use('dark_background') 
    fig_f = plt.figure(figsize=(12, 10), facecolor='#121212') 
    ax_f = fig_f.add_subplot(111, projection='3d') 
    ax_f.set_facecolor('#121212') 

    # --- 1. DESSIN DE LA TRAJECTOIRE (Dégradé de couleurs) ---
    # On crée des segments pour pouvoir appliquer un dégradé (cmap)
    pts = traj_reelle.reshape(-1, 1, 3) 
    segments_traj = np.concatenate([pts[:-1], pts[1:]], axis=1) 
    lc_traj = Line3DCollection(segments_traj, cmap='cool', norm=plt.Normalize(0, len(segments_traj)), 
                               alpha=0.9, lw=2.5, zorder=10) 
    lc_traj.set_array(np.arange(len(segments_traj))) 
    ax_f.add_collection3d(lc_traj) 

    # --- 2. DESSIN DES FIBRES (Transparentes) ---
    all_segs = [] 
    for s_list in voxels_fibres.values(): 
        for s in s_list:
            # s[0] et s[1] sont les points d'intersection P1 et P2
            all_segs.append([s[0], s[1]]) 
            
    fibres_obj = Line3DCollection(all_segs, colors='#ff4444', alpha=0.15, lw=1.0, zorder=1) 
    ax_f.add_collection3d(fibres_obj) 

    # --- 3. POINTS DE DÉPART ET DE FIN ---
    sc_start = ax_f.scatter(traj_reelle[0,0], traj_reelle[0,1], traj_reelle[0,2], 
                color='#00FF00', s=200, label='Départ (0,0,0)', edgecolors='white', linewidth=2, zorder=20) 
    sc_end = ax_f.scatter(traj_reelle[-1,0], traj_reelle[-1,1], traj_reelle[-1,2], 
                color='#FF0055', s=200, label='Position Finale', edgecolors='white', linewidth=2, zorder=20) 

    # --- 4. DESSIN DE LA CAGE (CADRE DU MILIEU) ---
    r = [-half_L, half_L] 
    for x in r: 
        for y in r: 
            # Arêtes selon Z
            ax_f.plot([x, x], [y, y], [r[0], r[1]], color='white', lw=1, alpha=0.4) 
            # Arêtes selon Y
            ax_f.plot([x, x], [r[0], r[1]], [y, y], color='white', lw=1, alpha=0.4) 
            # Arêtes selon X
            ax_f.plot([r[0], r[1]], [x, x], [y, y], color='white', lw=1, alpha=0.4) 

    # --- 5. ENCADRÉ DE STATISTIQUES (TOP LEFT) ---
    stats_text = ( 
        f"--- CONFIGURATION ---\n" 
        f"Temps total : {t_total_max:.2f} s\n" 
        f"Pas de temps : {dt} s\n" 
        f"Itérations : {len(traj_reelle)-1}\n\n" 
        f"--- RÉSULTATS ---\n" 
        f"Collisions : {coll_count}\n" 
        f"Voxelisation : {n_div}^3\n"
        f"Statut : {stop_reason}"
    ) 
    props = dict(boxstyle='round', facecolor='#1e1e1e', alpha=0.8, edgecolor='white') 
    ax_f.text2D(0.02, 0.95, stats_text, transform=ax_f.transAxes, fontsize=10, 
                verticalalignment='top', bbox=props, color='white', family='monospace') 

    # --- 6. BOUTON INTERACTIF (BOTTOM RIGHT) ---
    ax_button = plt.axes([0.7, 0.05, 0.25, 0.05]) 
    # On utilise une variable globale pour éviter que le bouton soit désactivé par le ramasse-miettes
    global btn_toggle 
    btn_toggle = Button(ax_button, 'Afficher/Masquer Fibres', color='#2c2c2c', hovercolor='#444444') 
    btn_toggle.label.set_color('white') 

    def toggle_fibres(event): 
        is_visible = fibres_obj.get_visible()
        fibres_obj.set_visible(not is_visible) 
        plt.draw() 
    
    btn_toggle.on_clicked(toggle_fibres) 

    # Ajustements finaux des axes
    ax_f.set_axis_off() 
    ax_f.set_xlim(-half_L, half_L)
    ax_f.set_ylim(-half_L, half_L)
    ax_f.set_zlim(-half_L, half_L) 
    ax_f.set_title(f"ANALYSE FINALE - MILIEU FIBREUX", color='white', fontsize=14, fontweight='bold') 
    
    # Légende pour les points
    ax_f.legend(handles=[sc_start, sc_end], loc='lower left', facecolor='#1e1e1e', 
                edgecolor='white', labelcolor='white', bbox_to_anchor=(0.02, 0.05)) 
    
    plt.show()