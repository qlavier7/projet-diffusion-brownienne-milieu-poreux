### Use the following configs in order to reproduce the different figures from the report

### Figure 6a (particle-obstacle collisions, case nu=1/50):
Script: 'IB-LBM_particle_diff_vF.py'
- N_outputs = 10
- sim_time = 500
- BCs = [[0, 0], [0, 0], [0, 0]]
- D_particle1 = 7
- D_particle2 = 10
- D_particle3 = 4
- D_particle4 = 4
- N_markers = 4
- nu = 1/50
- rhos = np.ones(N_markers, dtype=np.float64) * 1.14 * rho_0
- brownian_method = ['none', 'force', 'fluctuation'][0]
- collision_time_model = "hertzian"
- N_cylinders = 3

- copy in line 798 (if not already in the script) :
# marker 1
init_marker_pos[0, 0] = cx_particle - 3 * D_particle1
init_marker_pos[0, 1] = cy_particle
init_marker_pos[0, 2] = cz_particle

# marker 2
init_marker_pos[1, 0] = cx_particle + 2 * D_particle2
init_marker_pos[1, 1] = cy_particle + 0.5 * D_particle2
init_marker_pos[1, 2] = cz_particle

# marker 3
init_marker_pos[2, 0] = cx_particle 
init_marker_pos[2, 1] = cy_particle - 4 * D_particle3
init_marker_pos[2, 2] = cz_particle

# marker 4
init_marker_pos[3, 0] = cx_particle + 3 * D_particle4
init_marker_pos[3, 1] = cy_particle - 3.5 * D_particle4
init_marker_pos[3, 2] = cz_particle

- copy in line 935 (if not already in the script) :
# Initial force distribution for the first 100 steps
if t < 100:
    marker_f[0, 0] = 1 # initial force marker 1. Use 1 for nu = 1/50, 3 for nu = 1/6
    marker_f[0, 1] = 1 # Use 1 for nu = 1/50, 3 for nu = 1/6
    marker_f[1, 1] = 2 # initial force marker 2. Use 2 for nu = 1/50, 8 for nu = 1/6
    marker_f[2, 0] = 0.3 # initial force marker 3. Use 0.3 for nu = 1/50, 1.5 for nu = 1/6
    marker_f[2, 1] = 0.3 # Use 0.3 for nu = 1/50, 1.5 for nu = 1/6

# Remove force after 100 steps
if t>=100:
    marker_f[0, 0] = 0 # update force marker 1
    marker_f[0, 1] = 0
    marker_f[1, 1] = 0 # update force marker 2
    marker_f[2, 0] = 0 # update force marker 3
    marker_f[2, 1] = 0


##########################################################


### Figure 6b (particle-obstacle collisions, case nu=1/6):
Script: 'IB-LBM_particle_diff_vF.py'
- nu = 1/6

- copy in line 935 (if not already in the script) :
# Initial force distribution for the first 100 steps
if t < 100:
    marker_f[0, 0] = 3 # initial force marker 1. Use 1 for nu = 1/50, 3 for nu = 1/6
    marker_f[0, 1] = 3 # Use 1 for nu = 1/50, 3 for nu = 1/6
    marker_f[1, 1] = 8 # initial force marker 2. Use 2 for nu = 1/50, 8 for nu = 1/6
    marker_f[2, 0] = 1.5 # initial force marker 3. Use 0.3 for nu = 1/50, 1.5 for nu = 1/6
    marker_f[2, 1] = 1.5 # Use 0.3 for nu = 1/50, 1.5 for nu = 1/6

# Remove force after 100 steps
if t>=100:
    marker_f[0, 0] = 0 # update force marker 1
    marker_f[0, 1] = 0
    marker_f[1, 1] = 0 # update force marker 2
    marker_f[2, 0] = 0 # update force marker 3
    marker_f[2, 1] = 0

- same as fig. 6a for other parameters


##########################################################


### Figure 11a (amplified brownian motion):
Script: 'IB-LBM_particle_diff_3d_vF.py'
- nu = 1/6
- rhos = np.ones(N_markers, dtype=np.float64) * rho_0
- kB_T = 1.0
- brownian_method = ['none', 'force', 'fluctuation'][1]
- collision_time_model = "normal"
- comment lines 1142-1158


##########################################################


### Figure 11b (fluctuating LBM brownian motion):
Script: 'IB-LBM_particle_diff_3d_vF.py'
- nu = 1/6
- rhos = np.ones(N_markers, dtype=np.float64) * rho_0
- kB_T = 0.005
- brownian_method = ['none', 'force', 'fluctuation'][2]
- collision_time_model = "normal"
- comment lines 1142-1158

##########################################################


### Figure 11c (3D collisions, not in the report):
Script: 'IB-LBM_particle_diff_3d_vF.py'
- nu = 1/50
- rhos = np.ones(N_markers, dtype=np.float64) * 1.14 * rho_0
- brownian_method = ['none', 'force', 'fluctuation'][0]
- collision_time_model = "hertzian"

- copy in line 1142 (if not already in the script):
# Initial force distribution for the first 100 steps
if t < 100:
    marker_f[0, 0] = 1 # initial force marker 1
    marker_f[1, 1] = -2 # initial force marker 2
    marker_f[2, 0] = 0.3 # initial force marker 3
    marker_f[2, 1] = 0.3
    marker_f[3, 0] = -0.3 # initial force marker 4
    marker_f[3, 1] = 0.3

# Remove force after 100 steps
if t>=100:
    marker_f[0, 0] = 0 # update force marker 1
    marker_f[1, 1] = 0 # update force marker 2
    marker_f[2, 0] = 0 # update force marker 3
    marker_f[2, 1] = 0
    marker_f[3, 0] = 0 # update force marker 4
    marker_f[3, 1] = 0