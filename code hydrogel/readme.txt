Readme for the hydrogel.py file 

# Numba 

The script uses Numba, it will compile any function that has the decoration @njit or @njit(parallel=True). 

Numba compiles Python to get speed closer to C++, parallel=True will run the loops that have the 'prange' in parallel 

Some Python tools (dictionaries...) wont work with Numba, for more informations, go to : 

https://numba.pydata.org/


# Script 

## build_neighbor_lookup() 

This function will build the partition with the neighbor information for each fiber. 
With a cleaner fiber structure it may be possible to avoid this for later optimization.

## build_spatial_hash() 

Spatial hashing algorithm to avoid looking for all points when computing interactions. 
It's often done in many body gravity simulations to save computational power. 
It will assign each mass to a cell


## build_neighbor_cells()

Get the nearest neighbor cells so when the dynamic of a mass is computed, 
we only look at the distance for the mass and its close neighborood. 


## compute_multi_fiber_dynamics_3d_optimized()

Compute the dynamics of the system with Euler Integration. 
Spring force
Electromagnetic force
Bending force 
Crosslinks check 



## create_3d_animation(), create_interactive_3d_view(), save_simulation_to_tar() 

Self explanatory, has comments on the .py 
Mostly generated with copilote, which can easily changed the parameters for you if needed (no risk, it's vizualisation). 

## generate_isotropic_fiber_positions()

randomly generate fibers in a sphere area, completely isotropely

## generate_anisotropic_fiber_positions()

Generate sine like fibers starting from the xz plane, and extending along the x axis. 

# Main 

This is where the parameters of the simulations must be configured. Everything is explained and commented directly on the script. 

# How to run the script 

Currently the code can simply be ran through a terminal or directly within VScode or any other editor 

# Output file and visuals : 

.tar with the last configuration data 
.mp4 of the evolution 
matplotlib plot of the final configuration (can be reopen using the .tar)
