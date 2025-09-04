"""
Power Line Extraction from LAS Point Cloud Files - Python Version
Based on demo_extract_powerline.py, modified to process single LAS files

This script processes single LAS point cloud files for power line extraction:
1. LAS file loading and ground removal
2. Candidate power line point extraction
3. Euclidean clustering
4. Power line modeling and refinement

Original algorithm by:
Zhenwei Shi, Yi Lin, and Hui Li
"Extraction of urban power lines and potential hazard analysis from mobile laser scanning point clouds."
International Journal of Remote Sensing 41, no. 9 (2020): 3411-3428.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import laspy
from utils import extract_pls, insert_3d
from clustering import (euclidean_clustering, create_power_line_structure, 
                       merge_power_lines, filter_short_power_lines, 
                       calculate_total_length)

def load_las_point_cloud_data(las_path):
    """
    Load and process LAS point cloud file with ground filtering
    
    Args:
        las_path: Path to the LAS file
        
    Returns:
        non_ground_points: Filtered non-ground points (N x 3 array)
    """
    print(f"Loading LAS point cloud data from: {las_path}")
    
    if not os.path.exists(las_path):
        raise FileNotFoundError(f"LAS file not found: {las_path}")
    
    # Load LAS file
    try:
        las_file = laspy.read(las_path)
        print(f"Successfully loaded LAS file with {len(las_file.points)} points")
    except Exception as e:
        raise RuntimeError(f"Failed to load LAS file: {e}")
    
    # Extract XYZ coordinates
    x = las_file.x
    y = las_file.y  
    z = las_file.z
    
    # Combine into point cloud array
    point_cloud = np.column_stack((x, y, z))
    print(f"Point cloud shape: {point_cloud.shape}")
    print(f"X range: [{np.min(x):.2f}, {np.max(x):.2f}]")
    print(f"Y range: [{np.min(y):.2f}, {np.max(y):.2f}]")
    print(f"Z range: [{np.min(z):.2f}, {np.max(z):.2f}]")
    
    # Ground filtering using histogram analysis (same as original MATLAB/Python)
    z_coords = point_cloud[:, 2]
    
    # Create histogram with 50 bins
    counts, bin_edges = np.histogram(z_coords, bins=50)
    centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find bin with maximum count (ground level estimation)
    max_idx = np.argmax(counts)
    
    # Ground threshold: 3 bins above the maximum density height
    ground_height = centers[min(max_idx + 3, len(centers) - 1)]
    print(f"Estimated ground height: {ground_height:.2f} m")
    
    # Extract non-ground points
    non_ground_mask = point_cloud[:, 2] > ground_height
    non_ground_points = point_cloud[non_ground_mask]
    
    print(f"Non-ground points: {non_ground_points.shape[0]} ({100*non_ground_points.shape[0]/point_cloud.shape[0]:.1f}%)")
    
    return non_ground_points

def visualize_points_3d(points, title="Point Cloud", colors=None, save_path=None):
    """
    Visualize 3D point cloud with proper aspect ratio matching MATLAB pcshow
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    if colors is None:
        ax.scatter(points[:, 1], points[:, 0], points[:, 2], c=points[:, 2], s=0.5, alpha=0.6, cmap='viridis')
    else:
        ax.scatter(points[:, 1], points[:, 0], points[:, 2], c=colors, s=0.5, alpha=0.6)
    
    # Calculate data ranges (original axes)
    x_range = points[:, 0].max() - points[:, 0].min()
    y_range = points[:, 1].max() - points[:, 1].min()
    z_range = points[:, 2].max() - points[:, 2].min()
    
    # Set axis limits with some padding (swapped for display)
    padding = 0.1
    ax.set_xlim(points[:, 1].max() + y_range * padding, points[:, 1].min() - x_range * padding)
    ax.set_ylim(points[:, 0].min() - x_range * padding, points[:, 0].max() + x_range * padding)
    ax.set_zlim(points[:, 2].min() - z_range * padding, points[:, 2].max() + z_range * padding)
    
    # Set equal aspect ratio for better visualization
    max_range = max(x_range, y_range, z_range)
    ax.set_box_aspect([y_range/max_range, x_range/max_range, z_range/max_range])
    
    ax.set_xlabel('Y')
    ax.set_ylabel('X')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.view_init(elev=25, azim=60)  # Same view as MATLAB
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization: {save_path}")
    
    plt.show()
    plt.close()

def show_segs(points, labels, show_plot=True):
    """
    Show different clusters with different colors
    Equivalent to show_segs.m function
    """
    if not show_plot:
        return
    
    # Generate colors for each cluster
    unique_labels = np.unique(labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    point_colors = np.zeros((len(points), 3))
    for i, label in enumerate(unique_labels):
        mask = labels == label
        point_colors[mask] = colors[i][:3]
    
    return point_colors

def save_results_to_las(power_lines_final, output_path):
    """
    Save the final power line results to a new LAS file
    
    Args:
        power_lines_final: List of PowerLine objects
        output_path: Output LAS file path
    """
    if not power_lines_final:
        print("No power lines to save")
        return
        
    # Combine all power line points
    all_points = np.vstack([pl.Location for pl in power_lines_final])
    
    # Create classification labels for each power line
    all_classifications = []
    for i, pl in enumerate(power_lines_final):
        # Use classification value 14 (wire - conductor) + power line index
        classifications = np.full(pl.Location.shape[0], 14 + i, dtype=np.uint8)
        all_classifications.append(classifications)
    all_classifications = np.concatenate(all_classifications)
    
    # Create new LAS file
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.x_scale = 0.01
    header.y_scale = 0.01 
    header.z_scale = 0.01
    header.x_offset = np.min(all_points[:, 0])
    header.y_offset = np.min(all_points[:, 1])
    header.z_offset = np.min(all_points[:, 2])
    
    las = laspy.LasData(header)
    las.x = all_points[:, 0]
    las.y = all_points[:, 1]
    las.z = all_points[:, 2]
    las.classification = all_classifications
    
    # Set intensity based on power line index
    intensity = []
    for i, pl in enumerate(power_lines_final):
        pl_intensity = np.full(pl.Location.shape[0], (i + 1) * 1000, dtype=np.uint16)
        intensity.append(pl_intensity)
    las.intensity = np.concatenate(intensity)
    
    las.write(output_path)
    print(f"Saved power line results to: {output_path}")
    print(f"  - {len(power_lines_final)} power lines")
    print(f"  - {all_points.shape[0]} total points")

def main(las_file_path, output_dir=None):
    """
    Main power line extraction pipeline for LAS files
    
    Args:
        las_file_path: Path to input LAS file
        output_dir: Output directory for results (optional)
    """
    print("=== Power Line Extraction from LAS Point Cloud ===")
    print("Python version - processing single LAS file")
    print()
    
    # Configuration - Same parameters as original
    radius = 0.5           # Neighborhood search radius  
    angle_thr = 10         # Angle threshold in degrees
    l_thr = 0.98          # Linearity threshold
    min_distance_1 = 2.0   # First clustering distance
    min_distance_2 = 0.3   # Second clustering distance
    min_cluster_size = 15  # Minimum points per cluster
    min_length = 1.5       # Minimum power line length
    insert_resolution = 0.1 # Point insertion resolution
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(las_file_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base filename for outputs
    base_name = os.path.splitext(os.path.basename(las_file_path))[0]
    
    try:
        # Step 1: LAS file loading and ground filtering
        print("Step 1: LAS file loading and ground filtering...")
        start_time = time.time()
        
        non_ground_points = load_las_point_cloud_data(las_file_path)
        
        end_time = time.time()
        print(f"Ground filtering completed in {end_time - start_time:.2f} seconds")
        
        # Visualize non-ground points
        visualize_points_3d(non_ground_points, 
                          "Non-ground Points",
                          save_path=os.path.join(output_dir, f"{base_name}_1_nonGroundPoints.png"))
        
        # Step 2: Extract candidate power line points
        print("\nStep 2: Extracting candidate power line points...")
        start_time = time.time()
        
        is_pl_index = extract_pls(non_ground_points, radius, angle_thr, l_thr)
        candidate_points = non_ground_points[is_pl_index]
        
        end_time = time.time()
        print(f"Power line extraction completed in {end_time - start_time:.2f} seconds")
        print(f"Found {candidate_points.shape[0]} candidate power line points")
        
        if candidate_points.shape[0] == 0:
            print("No candidate power line points found. Exiting.")
            return None
        
        # Visualize candidate points
        visualize_points_3d(candidate_points,
                          "Candidate Power Line Points", 
                          save_path=os.path.join(output_dir, f"{base_name}_2_candidate_powerline_points.png"))
        
        # Step 3: Euclidean clustering
        print("\nStep 3: Euclidean clustering...")
        
        # First clustering with larger distance
        labels, num_clusters = euclidean_clustering(candidate_points, min_distance_1)
        print(f"Initial clustering found {num_clusters} clusters")
        
        # Visualize initial clusters
        cluster_colors = show_segs(candidate_points, labels)
        visualize_points_3d(candidate_points,
                          "Candidate Power Line Clusters",
                          colors=cluster_colors,
                          save_path=os.path.join(output_dir, f"{base_name}_3_candidate_powerline_clusters.png"))
        
        # Filter clusters by minimum size
        power_lines_raw = create_power_line_structure(candidate_points, labels, 
                                                    num_clusters, min_cluster_size)
        
        # Combine filtered power lines
        if power_lines_raw:
            filtered_points = np.vstack([pl.Location for pl in power_lines_raw])
            print(f"After filtering: {len(power_lines_raw)} clusters, {filtered_points.shape[0]} points")
            
            visualize_points_3d(filtered_points,
                              "Power Line Clusters",
                              save_path=os.path.join(output_dir, f"{base_name}_4_powerline_clusters.png"))
        else:
            print("No valid power line clusters found")
            return None
        
        # Step 4: Power line modeling
        print("\nStep 4: Power line modeling...")
        
        # Fine clustering with smaller distance
        labels_fine, num_clusters_fine = euclidean_clustering(filtered_points, min_distance_2)
        print(f"Fine clustering found {num_clusters_fine} clusters")
        
        # Create power line structures
        power_lines = create_power_line_structure(filtered_points, labels_fine,
                                                num_clusters_fine, min_points=1)
        
        # Visualize with different colors
        if power_lines:
            all_points = np.vstack([pl.Location for pl in power_lines])
            all_colors = []
            for i, pl in enumerate(power_lines):
                color = plt.cm.tab20(i % 20)[:3]
                pl_colors = np.tile(color, (pl.Location.shape[0], 1))
                all_colors.append(pl_colors)
            all_colors = np.vstack(all_colors)
            
            visualize_points_3d(all_points,
                              "Different Clusters with Different Colors",
                              colors=all_colors,
                              save_path=os.path.join(output_dir, f"{base_name}_5_colorization_clusters.png"))
        
        # Sort power lines by count
        counts = [pl.Count for pl in power_lines]
        sorted_indices = np.argsort(counts)[::-1]  # Descending order
        
        # Multiple rounds of merging (same as original)
        print("Performing power line merging...")
        for round_num in range(3):
            power_lines, sorted_indices = merge_power_lines(power_lines, sorted_indices)
            print(f"  Round {round_num + 1}: {len(power_lines)} power lines")
        
        # Visualize merged power lines
        if power_lines:
            merged_points = np.vstack([pl.Location for pl in power_lines])
            merged_colors = []
            for i, pl in enumerate(power_lines):
                color = np.random.rand(3)  # Random color for each power line
                pl_colors = np.tile(color, (pl.Location.shape[0], 1))
                merged_colors.append(pl_colors)
            merged_colors = np.vstack(merged_colors)
            
            visualize_points_3d(merged_points,
                              "Power Line Clusters",
                              colors=merged_colors,
                              save_path=os.path.join(output_dir, f"{base_name}_6_powerLines_clusters.png"))
        
        # Filter short power lines
        power_lines = filter_short_power_lines(power_lines, min_length)
        print(f"After length filtering: {len(power_lines)} power lines")
        
        if not power_lines:
            print("No power lines remaining after length filtering")
            return None
        
        # Calculate total length
        total_length = calculate_total_length(power_lines)
        print(f"Total power line length: {total_length:.2f} meters")
        
        # Insert points to sparse power lines
        print("Inserting points to sparse power lines...")
        power_lines_final = []
        
        for pl in power_lines:
            dense_points = insert_3d(pl.Location, insert_resolution)
            pl.Location = dense_points
            power_lines_final.append(pl)
        
        # Final visualization
        if power_lines_final:
            final_points = np.vstack([pl.Location for pl in power_lines_final])
            final_colors = []
            for i, pl in enumerate(power_lines_final):
                color = np.random.rand(3)
                pl_colors = np.tile(color, (pl.Location.shape[0], 1))
                final_colors.append(pl_colors)
            final_colors = np.vstack(final_colors)
            
            visualize_points_3d(final_points,
                              "Power Line Modeling",
                              colors=final_colors,
                              save_path=os.path.join(output_dir, f"{base_name}_7_Power_line_model.png"))
        
        # Save results to LAS file
        output_las_path = os.path.join(output_dir, f"{base_name}_powerlines.las")
        save_results_to_las(power_lines_final, output_las_path)
        
        print(f"\n=== Power Line Extraction Complete ===")
        print(f"Final results:")
        print(f"  - Number of power lines: {len(power_lines_final)}")
        print(f"  - Total length: {total_length:.2f} meters")
        print(f"  - Total points in model: {sum([pl.Location.shape[0] for pl in power_lines_final])}")
        print(f"  - Results saved to: {output_dir}")
        
        return power_lines_final
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extract_powerlines_las.py <las_file_path> [output_dir]")
        print("Example: python extract_powerlines_las.py ../convert_cloudpoint/Tile_62.las ./output")
        sys.exit(1)
    
    las_file_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = main(las_file_path, output_dir)
    
    if result:
        print("Power line extraction completed successfully!")
    else:
        print("Power line extraction failed.")