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


@njit
def build_neighbor_lookup(fiber_ids, point_in_fiber, fiber_lengths):
    """This function builds previous and next neighbor lookups for each point in fibers.
    It is later used in force calculations."""
    total_points = len(fiber_ids)
    prev_neighbors = np.zeros(total_points, dtype=np.int32) - 1
    next_neighbors = np.zeros(total_points, dtype=np.int32) - 1
    
    # Since points are stored sequentially 
    idx = 0
    for f in range(len(fiber_lengths)):
        fiber_len = fiber_lengths[f]
        
        # Connect consecutive points in this fiber
        for i in range(fiber_len - 1):
            next_neighbors[idx + i] = idx + i + 1
            prev_neighbors[idx + i + 1] = idx + i
        
        idx += fiber_len
    
    return prev_neighbors, next_neighbors

@njit
def build_spatial_hash(positions, cell_size, grid_size=20):
    """Build spatial hash for efficient neighbor searching.
    
    Returns:
    - cell_indices: which cell each point belongs to
    - cell_lists: list of points in each cell
    - cell_counts: number of points in each cell
    """
    total_points = len(positions)
    max_points_per_cell = 1000

    # Calculate cell indices for each point
    cell_indices = np.zeros(total_points, dtype=np.int32)
    cell_lists = np.zeros((grid_size**3, max_points_per_cell), dtype=np.int32) - 1
    cell_counts = np.zeros(grid_size**3, dtype=np.int32)
    
    # Find bounding box
    min_pos = np.zeros(3)
    max_pos = np.zeros(3)
    for dim in range(3):
        min_pos[dim] = np.min(positions[:, dim])
        max_pos[dim] = np.max(positions[:, dim])
    
    # Add small margin to avoid edge cases
    min_pos -= cell_size
    max_pos += cell_size
    
    # Assign points to cells
    for i in range(total_points):
        # Calculate cell coordinates
        cx = int((positions[i, 0] - min_pos[0]) / cell_size)
        cy = int((positions[i, 1] - min_pos[1]) / cell_size)
        cz = int((positions[i, 2] - min_pos[2]) / cell_size)
        
        # Clamp to grid bounds
        cx = max(0, min(cx, grid_size - 1))
        cy = max(0, min(cy, grid_size - 1))
        cz = max(0, min(cz, grid_size - 1))
        
        # Linear cell index
        cell_idx = cx + cy * grid_size + cz * grid_size * grid_size
        cell_indices[i] = cell_idx
        
        # Add to cell list
        count = cell_counts[cell_idx]
        if count < max_points_per_cell:
            cell_lists[cell_idx, count] = i
            cell_counts[cell_idx] += 1
    
    return cell_indices, cell_lists, cell_counts, min_pos, grid_size

@njit
def get_neighbor_cells(cell_x, cell_y, cell_z, grid_size):
    """Get indices of the 27 neighboring cells (including self)."""
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
    """Optimized 3D fiber dynamics with spatial hashing and combined distance checks."""
    
    num_fibers = len(fiber_lengths)
    total_points = sum(fiber_lengths)
    
    positions = np.zeros((total_points, 3), dtype=np.float64)
    velocities = np.zeros((total_points, 3), dtype=np.float64)
    fiber_ids = np.zeros(total_points, dtype=np.int32)
    point_in_fiber = np.zeros(total_points, dtype=np.int32)
    labels = np.zeros(total_points, dtype=np.int32)
    
    # Assign labels A, B and 0
    p_0, p_A, p_B = label_probabilities
    for i in prange(total_points):
        rand = np.random.random()
        if rand < p_0:
            labels[i] = 0
        elif rand < p_0 + p_A:
            labels[i] = 1
        else:
            labels[i] = 2
    
    # Initialize
    idx = 0
    for f in range(num_fibers):
        for p in range(fiber_lengths[f]):
            positions[idx] = fiber_positions[f][p]
            velocities[idx] = fiber_velocities[f][p]
            fiber_ids[idx] = f
            point_in_fiber[idx] = p
            idx += 1
    
    prev_neighbors, next_neighbors = build_neighbor_lookup(fiber_ids, point_in_fiber, fiber_lengths)
    
    position_data = np.zeros((total_points, num_steps + 1, 3))
    velocity_data = np.zeros((total_points, num_steps + 1, 3))
    position_data[:, 0, :] = positions
    velocity_data[:, 0, :] = velocities
    
    # A-B physical crosslinks
    crosslink_partner = np.zeros(total_points, dtype=np.int32) - 1
    crosslink_rest_lengths = np.zeros(total_points, dtype=np.float64)
    
    # B-B chemical crosslinks
    bb_partners = np.zeros((total_points, 2), dtype=np.int32) - 1
    bb_rest_lengths = np.zeros((total_points, 2), dtype=np.float64)
    bb_partner_count = np.zeros(total_points, dtype=np.int32)

    # count tracking
    crosslinks_formed = 0
    crosslinks_broken = 0
    bb_bonds_formed = 0
    bb_bonds_broken = 0
    
    # Anchored points tracking
    is_anchored = np.zeros(total_points, dtype=np.bool_)
    for i in range(total_points):
        fiber_id = fiber_ids[i]
        local_idx = point_in_fiber[i]
        fiber_len = fiber_lengths[fiber_id]
        anchor_start, anchor_end = anchor_configs[fiber_id]
        
        is_anchored[i] = (local_idx == 0 and anchor_start) or \
                        (local_idx == fiber_len - 1 and anchor_end)
    
    # Determine maximum interaction distance 
    max_interaction_dist = max(em_cutoff_distance, crosslink_distance, bb_bond_distance)
    cell_size = max_interaction_dist * 1.1  # Slightly larger than max distance
    
    # Cache for distance calculations (reused across force types)
    max_neighbors = 200  # Maximum neighbors to consider per point
    neighbor_cache = np.zeros(total_points * max_neighbors, dtype=np.int32) - 1
    distance_cache = np.zeros(total_points * max_neighbors, dtype=np.float64)
    neighbor_counts = np.zeros(total_points, dtype=np.int32)
    
    # Main simulation loop
    for step in range(num_steps):
        current_positions = position_data[:, step, :]
        current_velocities = velocity_data[:, step, :]
        
        # rebuild the spatial hash and neighbor lists every 10 steps to save computation time
        if step % 10 == 0:
            # Build spatial hash table
            cell_indices, cell_lists, cell_counts, grid_min, grid_size = \
                build_spatial_hash(current_positions, cell_size)
            
            # Clear neighbor cache
            neighbor_counts[:] = 0
            
            # Build neighbor lists using spatial hash
            for i in range(total_points):
                # Get cell coordinates
                cx = int((current_positions[i, 0] - grid_min[0]) / cell_size)
                cy = int((current_positions[i, 1] - grid_min[1]) / cell_size)
                cz = int((current_positions[i, 2] - grid_min[2]) / cell_size)
                
                cx = max(0, min(cx, grid_size - 1))
                cy = max(0, min(cy, grid_size - 1))
                cz = max(0, min(cz, grid_size - 1))
                
                # Check neighboring cells
                neighbor_cells = get_neighbor_cells(cx, cy, cz, grid_size)
                
                count = 0
                base_idx = i * max_neighbors
                
                for nc in neighbor_cells:
                    if nc < 0:
                        continue
                    
                    # Check all points in this cell
                    for idx in range(cell_counts[nc]):
                        j = cell_lists[nc, idx]
                        if j < 0 or j <= i:  # Only check each pair once
                            continue
                        
                        # Calculate distance once and cache it
                        displacement = current_positions[i] - current_positions[j]
                        distance = np.sqrt(np.sum(displacement**2))
                        
                        # Only store if within maximum interaction distance
                        if distance < max_interaction_dist and count < max_neighbors:
                            neighbor_cache[base_idx + count] = j
                            distance_cache[base_idx + count] = distance
                            count += 1
                
                neighbor_counts[i] = count
            
            # Process bond formation/breaking using cached distances
            # Break existing bonds first if the threshold is exceeded
            for i in range(total_points):
                # Break A-B crosslinks
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
                
                # Break B-B bonds
                if labels[i] == 2:
                    for bond_idx in range(bb_partner_count[i]):
                        j = bb_partners[i, bond_idx]
                        if j >= 0 and i < j:
                            displacement = current_positions[i] - current_positions[j]
                            distance = np.sqrt(np.sum(displacement**2))
                            
                            if distance > bb_bond_break_distance:
                                # Remove bond
                                for k in range(bond_idx, bb_partner_count[i] - 1):
                                    bb_partners[i, k] = bb_partners[i, k + 1]
                                    bb_rest_lengths[i, k] = bb_rest_lengths[i, k + 1]
                                bb_partners[i, bb_partner_count[i] - 1] = -1
                                bb_partner_count[i] -= 1
                                
                                # Remove from partner
                                for k in range(bb_partner_count[j]):
                                    if bb_partners[j, k] == i:
                                        for m in range(k, bb_partner_count[j] - 1):
                                            bb_partners[j, m] = bb_partners[j, m + 1]
                                            bb_rest_lengths[j, m] = bb_rest_lengths[j, m + 1]
                                        bb_partners[j, bb_partner_count[j] - 1] = -1
                                        bb_partner_count[j] -= 1
                                        break
                                
                                bb_bonds_broken += 1
            
            # Form new bonds using cached distances
            for i in range(total_points):
                base_idx = i * max_neighbors
                
                for n_idx in range(neighbor_counts[i]):
                    j = neighbor_cache[base_idx + n_idx]
                    distance = distance_cache[base_idx + n_idx]
                    
                    if j < 0:
                        continue
                    
                    # Skip if on same fiber and too close to avoid clustering
                    if fiber_ids[i] == fiber_ids[j]:
                        if abs(point_in_fiber[i] - point_in_fiber[j]) <= 4:
                            continue
                    
                    # Try A-B crosslink formation
                    if crosslink_partner[i] < 0 and crosslink_partner[j] < 0:
                        if (labels[i] == 1 and labels[j] == 2) or (labels[i] == 2 and labels[j] == 1):
                            if distance < crosslink_distance:
                                crosslink_partner[i] = j
                                crosslink_partner[j] = i
                                rest_len = distance * crosslink_rest_factor
                                crosslink_rest_lengths[i] = rest_len
                                crosslink_rest_lengths[j] = rest_len
                                crosslinks_formed += 1
                    

                    ### might want to add the B-B formation before the A-B to prioritize B-B bonds in the future ###
                    # Try B-B bond formation
                    if labels[i] == 2 and labels[j] == 2:
                        if bb_partner_count[i] < 2 and bb_partner_count[j] < 2:
                            # Check if not already bonded
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
        
        # Update dynamics with cached neighbor lists
        for i in prange(total_points):
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
                
                # Spring forces along fiber
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
                
                # Bending force 
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
                        curvature_vector = tangent_change / avg_length
                        bending_force = +bending_stiffness * curvature_vector
                
                # A-B Crosslink force
                if crosslink_partner[i] >= 0:
                    partner_idx = crosslink_partner[i]
                    partner_position = current_positions[partner_idx]
                    displacement = partner_position - position
                    distance = np.sqrt(np.sum(displacement**2))
                    
                    if distance > 1e-10:
                        unit_displacement = displacement / distance
                        rest_len = crosslink_rest_lengths[i]
                        crosslink_force += crosslink_stiffness * (distance - rest_len) * unit_displacement

                # B-B chemical crosslink forces
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
                
                # EM forces using cached neighbors (when available)
                if em_strength != 0.0:
                    if step % 10 == 0 and neighbor_counts[i] > 0:
                        # Use cached neighbors
                        base_idx = i * max_neighbors
                        
                        for n_idx in range(neighbor_counts[i]):
                            j = neighbor_cache[base_idx + n_idx]
                            distance = distance_cache[base_idx + n_idx]
                            
                            if j < 0:
                                continue
                            
                            # Skip if on same fiber and adjacent
                            if fiber_ids[j] == fiber_ids[i]:
                                if abs(point_in_fiber[j] - point_in_fiber[i]) <= 1:
                                    continue
                            
                            # Skip if connected
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
                                force_magnitude = em_strength / (distance * distance)
                                em_force += force_magnitude * unit_displacement
                    else:
                        # Fallback to traditional calculation
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
                
                # Total forces and update
                conservative_force = spring_force + bending_force + em_force + crosslink_force + bb_bond_force
                damping_force = -damping * velocity
                total_force = conservative_force + damping_force
                
                new_velocity = velocity + total_force * dt
                new_position = position + new_velocity * dt
                
                velocity_data[i, step + 1, :] = new_velocity
                position_data[i, step + 1, :] = new_position
    
    # Prepare output
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


def create_3d_animation(position_data, fiber_ids, point_in_fiber, labels,
                        crosslink_partner, bb_partners, bb_partner_count,
                        dt, fps=60, duration=10, 
                        filename="3d_fiber_animation.mp4"):
    """Create 3D animation for multi-fiber simulation with B-B bonds."""
    
    total_points, num_timesteps, _ = position_data.shape
    num_fibers = np.max(fiber_ids) + 1
    
    # Frame selection
    total_frames = fps * duration
    frame_step = max(1, num_timesteps // total_frames)
    selected_frames = np.arange(0, num_timesteps, frame_step)
    
    print(f"Creating 3D animation with {len(selected_frames)} frames at {fps} fps")
    
    # Setup figure
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Calculate plot limits
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
    
    # Create plot elements
    fiber_lines = []

    
    for f in range(num_fibers):
        line, = ax.plot([], [], [], '-', color='black', linewidth=1.5, alpha=0.7)
        fiber_lines.append(line)
    
    # A-B Crosslink lines (red)
    max_crosslinks = 1000
    crosslink_lines = []
    for _ in range(max_crosslinks):
        cl_line, = ax.plot([], [], [], 'r-', linewidth=1, alpha=0.5)
        crosslink_lines.append(cl_line)
    
    # B-B covalent bond lines (blue, thicker)
    max_bb_bonds = 1000
    bb_bond_lines = []
    for _ in range(max_bb_bonds):
        bb_line, = ax.plot([], [], [], 'b-', linewidth=2, alpha=0.7)
        bb_bond_lines.append(bb_line)
    
    # Time text
    time_text = ax.text2D(0.02, 0.98, '', transform=ax.transAxes, fontsize=12,
                          bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def animate(frame_idx):
        timestep = selected_frames[frame_idx]
        current_positions = position_data[:, timestep, :]
        
        # Update each fiber
        for f in range(num_fibers):
            fiber_indices = np.where(fiber_ids == f)[0]
            sorted_indices = sorted(fiber_indices, key=lambda i: point_in_fiber[i])
            
            if len(sorted_indices) > 0:
                x_coords = current_positions[sorted_indices, 0]
                y_coords = current_positions[sorted_indices, 1]
                z_coords = current_positions[sorted_indices, 2]
                fiber_lines[f].set_data_3d(x_coords, y_coords, z_coords)
        
        # Update A-B crosslinks
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
        
        for i in range(len(crosslink_pairs), len(crosslink_lines)):
            crosslink_lines[i].set_data_3d([], [], [])
        
        # Update B-B bonds
        bb_pairs = []
        for i in range(total_points):
            if labels[i] == 2:  # B label
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
        
        # Update text
        time_text.set_text(f'Time: {timestep * dt:.3f} s\n'
                          f'A-B Crosslinks: {len(crosslink_pairs)}\n'
                          f'B-B Covalent Bonds: {len(bb_pairs)}')
        
        # Rotate view for better visualization
        ax.view_init(elev=20, azim=30 + frame_idx * 0.5)
        
        return fiber_lines + crosslink_lines + bb_bond_lines + [time_text]
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, frames=len(selected_frames), 
        interval=1000/fps, blit=False, repeat=True
    )
    
    # Save
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


def create_interactive_3d_view(position_data, fiber_ids, point_in_fiber, labels,
                               crosslink_partner, bb_partners, bb_partner_count,
                               timestep=-1):
    """Create an interactive 3D view with B-B bonds visualized."""
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    current_positions = position_data[:, timestep, :]
    num_fibers = np.max(fiber_ids) + 1
    total_points = len(fiber_ids)
    
    # Plot fibers
    colors = plt.cm.rainbow(np.linspace(0, 1, min(num_fibers, 20)))
    
    for f in range(num_fibers):
        fiber_indices = np.where(fiber_ids == f)[0]
        sorted_indices = sorted(fiber_indices, key=lambda i: point_in_fiber[i])
        
        if len(sorted_indices) > 0:
            x = current_positions[sorted_indices, 0]
            y = current_positions[sorted_indices, 1]
            z = current_positions[sorted_indices, 2]
            color = colors[f % len(colors)]
            ax.plot(x, y, z, '-', color='black', linewidth=2, alpha=0.7)
    
    # Plot A-B crosslinks (red)
    ab_count = 0
    for i in range(total_points):
        if crosslink_partner[i] >= 0 and i < crosslink_partner[i]:
            j = crosslink_partner[i]
            x = [current_positions[i, 0], current_positions[j, 0]]
            y = [current_positions[i, 1], current_positions[j, 1]]
            z = [current_positions[i, 2], current_positions[j, 2]]
            ax.plot(x, y, z, 'r-', linewidth=1, alpha=0.5, label='A-B' if ab_count == 0 else "")
            ab_count += 1
    
    # Plot B-B covalent bonds (blue, thicker)
    bb_count = 0
    for i in range(total_points):
        if labels[i] == 2:  # B label
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


def save_simulation_to_tar(position_data, velocity_data, fiber_ids, point_in_fiber,
                          labels, crosslink_partner, bb_partners, bb_partner_count,
                          fiber_lengths, simulation_params, filename="simulation_results.tar"):
    """Save only final state simulation results to a tar archive."""
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Extract only the final timestep
        final_positions = position_data[:, -1, :]  # Last timestep positions
        final_velocities = velocity_data[:, -1, :]  # Last timestep velocities
        
        # Save only final state arrays as .npz files
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
        
        # Save final bond states
        # A-B crosslinks - create list of bonded pairs
        ab_bonds = []
        for i in range(len(crosslink_partner)):
            if crosslink_partner[i] >= 0 and i < crosslink_partner[i]:
                ab_bonds.append([i, crosslink_partner[i]])
        ab_bonds = np.array(ab_bonds, dtype=np.int32) if len(ab_bonds) > 0 else np.zeros((0, 2), dtype=np.int32)
        np.savez_compressed(os.path.join(temp_dir, "final_ab_crosslinks.npz"), 
                           data=ab_bonds)
        
        # B-B bonds - create list of bonded pairs
        bb_bonds = []
        for i in range(len(bb_partner_count)):
            if labels[i] == 2:  # B label
                for bond_idx in range(bb_partner_count[i]):
                    j = bb_partners[i, bond_idx]
                    if j >= 0 and i < j:
                        bb_bonds.append([i, j])
        bb_bonds = np.array(bb_bonds, dtype=np.int32) if len(bb_bonds) > 0 else np.zeros((0, 2), dtype=np.int32)
        np.savez_compressed(os.path.join(temp_dir, "final_bb_bonds.npz"), 
                           data=bb_bonds)
        
        # Add summary statistics to parameters
        simulation_params["final_state_stats"] = {
            "total_ab_crosslinks": len(ab_bonds),
            "total_bb_bonds": len(bb_bonds),
            "timestep_saved": position_data.shape[1] - 1
        }
        
        # Save simulation parameters as JSON
        with open(os.path.join(temp_dir, "parameters.json"), 'w') as f:
            json.dump(simulation_params, f, indent=2)
        
        # Create tar archive
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
        # Clean up temp directory
        shutil.rmtree(temp_dir)


def generate_isotropic_fiber_positions(num_fibers, num_points_per_fiber, point_spacing,
                                       cluster_radius, min_start_distance):
    """
    Generate isotropic distribution of fibers with random orientations.
    
    Parameters:
    - num_fibers: Number of fibers to generate
    - num_points_per_fiber: Points per fiber
    - point_spacing: Distance between consecutive points
    - cluster_radius: Maximum radius for fiber starting points
    - min_start_distance: Minimum distance from center for starting points
    """
    fiber_positions = []
    fiber_velocities = []
    fiber_lengths = []
    anchor_configs = []
    
    for fiber_idx in range(num_fibers):
        # Generate random starting point in spherical shell
        # Avoid center by using minimum distance
        while True:
            # Random point in sphere
            theta = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle
            phi = np.random.uniform(0, np.pi)  # Polar angle
            r = np.random.uniform(min_start_distance, cluster_radius)
            
            # Convert to Cartesian
            x_start = r * np.sin(phi) * np.cos(theta)
            y_start = r * np.sin(phi) * np.sin(theta)
            z_start = r * np.cos(phi)
            
            # Check minimum distance from origin
            dist_from_origin = np.sqrt(x_start**2 + y_start**2 + z_start**2)
            if dist_from_origin >= min_start_distance:
                break
        
        # Generate random direction for fiber extension
        # Use different angles to avoid all fibers pointing to/from center
        dir_theta = np.random.uniform(0, 2 * np.pi)
        dir_phi = np.random.uniform(0, np.pi)
        
        # Direction unit vector
        dx = np.sin(dir_phi) * np.cos(dir_theta)
        dy = np.sin(dir_phi) * np.sin(dir_theta)
        dz = np.cos(dir_phi)
        
        # Generate fiber points along this direction
        positions = np.zeros((num_points_per_fiber, 3), dtype=np.float64)
        
        for i in range(num_points_per_fiber):
            # Base position along straight line
            x = x_start + i * point_spacing * dx
            y = y_start + i * point_spacing * dy
            z = z_start + i * point_spacing * dz
            
            # Add small random curvature for realism
            if i > 0:  # Don't perturb the anchored first point
                curve_amplitude = 0.05 * point_spacing
                x += curve_amplitude * np.sin(i * np.pi / 15) * np.random.randn()
                y += curve_amplitude * np.cos(i * np.pi / 15) * np.random.randn()
                z += curve_amplitude * np.sin(i * np.pi / 20) * np.random.randn()
            
            positions[i] = [x, y, z]
        
        # Random initial velocities
        velocities = np.random.randn(num_points_per_fiber, 3) * 0.1
        
        fiber_positions.append(positions)
        fiber_velocities.append(velocities)
        fiber_lengths.append(num_points_per_fiber)
        
        # Anchor configuration - anchor first point
        anchor_configs.append((True, False))
    
    return fiber_positions, fiber_velocities, fiber_lengths, anchor_configs


def generate_anisotropic_fiber_positions(grid_size, num_points_per_fiber, 
                                        x_spacing, y_spacing, z_spacing):
    """
    Generate anisotropic distribution of fibers arranged in a grid on the YZ plane,
    all extending in the positive X direction.
    
    Parameters:
    - grid_size: Number of fibers in each dimension of the YZ grid (total fibers = grid_size^2)
    - num_points_per_fiber: Points per fiber
    - x_spacing: Distance between consecutive points along X
    - y_spacing: Grid spacing in Y direction
    - z_spacing: Grid spacing in Z direction
    """
    fiber_positions = []
    fiber_velocities = []
    fiber_lengths = []
    anchor_configs = []
    
    fiber_count = 0
    for yi in range(grid_size):
        for zi in range(grid_size):
            # Starting position on YZ plane
            y_start = (yi - grid_size/2) * y_spacing
            z_start = (zi - grid_size/2) * z_spacing
            
            # Generate fiber extending in positive x direction
            positions = np.zeros((num_points_per_fiber, 3), dtype=np.float64)
            
            for i in range(num_points_per_fiber):
                x = i * x_spacing
                
                # Add some curvature for visual interest
                if (yi + zi) % 3 == 0:
                    # Slight upward curve in z
                    z = z_start + 0.1 * np.sin(i * np.pi / 10)
                    y = y_start
                elif (yi + zi) % 3 == 1:
                    # Slight curve in y
                    y = y_start + 0.1 * np.sin(i * np.pi / 10)
                    z = z_start
                else:
                    # Diagonal curve
                    y = y_start + 0.05 * np.cos(i * np.pi / 8)
                    z = z_start + 0.05 * np.sin(i * np.pi / 8)
                
                positions[i] = [x, y, z]
            
            # Random initial velocities (small)
            velocities = np.random.randn(num_points_per_fiber, 3) * 0.1
            
            fiber_positions.append(positions)
            fiber_velocities.append(velocities)
            fiber_lengths.append(num_points_per_fiber)
            
            # Anchor configuration - anchor first point at YZ plane
            anchor_configs.append((True, False))
            
            fiber_count += 1
    
    print(f"Generated {fiber_count} fibers in {grid_size}x{grid_size} grid")
    return fiber_positions, fiber_velocities, fiber_lengths, anchor_configs


if __name__ == "__main__":

    ############## USER INPUT SECTION ##############
    # Choose generation type
    generation_type = "isotropic"  # Change to "isotropic" or "anisotropic" depending on what you want

    # Common parameters
    num_points_per_fiber = 30 #size of the fiber
    point_spacing = 0.5 #spacing between neighbors at rest
    
    if generation_type == "isotropic":
        # Isotropic configuration
        num_fibers = 75 
        cluster_radius = 8.0 # Radius of the spherical cluster
        min_start_distance = 1 # Minimum distance from center for starting points to avoid clustering at center
        
        print(f"Generating ISOTROPIC fiber distribution...")
        print(f"- Number of fibers: {num_fibers}")
        print(f"- Points per fiber: {num_points_per_fiber}")
        print(f"- Total points: {num_fibers * num_points_per_fiber}")
        
        fiber_positions, fiber_velocities, fiber_lengths, anchor_configs = \
            generate_isotropic_fiber_positions(
                num_fibers, num_points_per_fiber, point_spacing,
                cluster_radius, min_start_distance
            )
    
    elif generation_type == "anisotropic":
        # Anisotropic configuration
        grid_size = 10  # 10x10 grid = 100 fibers for the hashing
        x_spacing = point_spacing
        y_spacing = 1  # Spacing between fibers in Y
        z_spacing = 1  # Spacing between fibers in Z
        
        print(f"Generating ANISOTROPIC fiber distribution...")
        fiber_positions, fiber_velocities, fiber_lengths, anchor_configs = \
            generate_anisotropic_fiber_positions(
                grid_size, num_points_per_fiber,
                x_spacing, y_spacing, z_spacing
            )
        num_fibers = grid_size * grid_size
        
        print(f"- Grid size: {grid_size}x{grid_size}")
        print(f"- Number of fibers: {num_fibers}")
        print(f"- Points per fiber: {num_points_per_fiber}")
        print(f"- Total points: {num_fibers * num_points_per_fiber}")
    
    else:
        raise ValueError(f"Unknown generation_type: {generation_type}")
    
    # Physical parameters
    rest_length = point_spacing
    damping = 0.7 # Increasee damping for stability
    bending_stiffness = 1 # Reduce bending stiffness for stability
    
    # EM parameters
    em_strength = 0.05 # Reduce EM strength for stability
    em_cutoff_distance = 1.2 #cutoff saves computation time
    
    # Crosslinking parameters
    label_probabilities = (0.8, 0.1, 0.1) # (None, A, B) A for acid, B for base, None for nothing, probabilities

    # A-B bond parameters
    crosslink_distance = 1.5 
    crosslink_break_distance = 3
    crosslink_stiffness = 1.5
    crosslink_rest_factor = 1.5 #rest length factor
    
    # B-B bond parameters
    bb_bond_distance = 1.3
    bb_bond_break_distance = 1.8
    bb_bond_stiffness = 3.0
    bb_bond_rest_factor = 1
    
    # Simulation parameters
    dt = 0.01 #time step
    num_steps = 5000 #number of time steps

############## END OF USER INPUT ##############
    



    print(f"\nRunning OPTIMIZED simulation with spatial hashing...")
    print(f"- Distribution type: {generation_type.upper()}")
    
    # Prepare arrays
    total_points = sum(fiber_lengths)
    stiffnesses = np.array([2.0] * total_points, dtype=np.float64)
    
    # Run optimized simulation
    import time
    start_time = time.time()
    
    (position_data, velocity_data, fiber_ids, point_in_fiber, 
     labels, crosslink_partner, bb_partners, bb_partner_count,
     final_crosslinks, final_bb_bonds) = \
        compute_multi_fiber_dynamics_3d_optimized(
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
    
    # Calculate energy
    initial_energy = np.sum(velocity_data[:, 0, :]**2) / 2
    final_energy = np.sum(velocity_data[:, -1, :]**2) / 2
    print(f"\nEnergy analysis:")
    print(f"- Initial kinetic energy: {initial_energy:.6f}")
    print(f"- Final kinetic energy: {final_energy:.6f}")
    print(f"- Energy dissipated: {initial_energy - final_energy:.6f}")
    
    # Save simulation data to tar archive
    simulation_params = {
        "num_fibers": num_fibers,
        "num_points_per_fiber": num_points_per_fiber,
        "total_points": total_points,
        "rest_length": rest_length,
        "damping": damping,
        "bending_stiffness": bending_stiffness,
        "em_strength": em_strength,
        "em_cutoff_distance": em_cutoff_distance,
        "crosslink_distance": crosslink_distance,
        "crosslink_break_distance": crosslink_break_distance,
        "crosslink_stiffness": crosslink_stiffness,
        "crosslink_rest_factor": crosslink_rest_factor,
        "bb_bond_distance": bb_bond_distance,
        "bb_bond_break_distance": bb_bond_break_distance,
        "bb_bond_stiffness": bb_bond_stiffness,
        "bb_bond_rest_factor": bb_bond_rest_factor,
        "label_probabilities": label_probabilities,
        "dt": dt,
        "num_steps": num_steps,
        "point_spacing": point_spacing,
        "distribution_type": generation_type
    }
    
    # Add generation-specific parameters
    if generation_type == "isotropic":
        simulation_params.update({
            "cluster_radius": cluster_radius,
            "min_start_distance": min_start_distance
        })
    elif generation_type == "anisotropic":
        simulation_params.update({
            "grid_size": grid_size,
            "x_spacing": x_spacing,
            "y_spacing": y_spacing,
            "z_spacing": z_spacing
        })
    
    print("\nSaving simulation data to tar archive...")
    save_simulation_to_tar(
        position_data, velocity_data, fiber_ids, point_in_fiber,
        labels, crosslink_partner, bb_partners, bb_partner_count,
        fiber_lengths, simulation_params, 
        filename=f"3d_{generation_type}_bb_simulation.tar"
    )
    
    # Create animation
    print("\nCreating 3D animation...")
    create_3d_animation(
        position_data, fiber_ids, point_in_fiber, labels,
        crosslink_partner, bb_partners, bb_partner_count,
        dt, fps=60, duration=10, 
        filename=f"3d_{generation_type}_bb_fiber_animation.mp4"
    )
    
    # Create interactive view of final state
    print("\nCreating interactive 3D view of final configuration...")
    create_interactive_3d_view(
        position_data, fiber_ids, point_in_fiber, labels,
        crosslink_partner, bb_partners, bb_partner_count,
        timestep=-1
    )