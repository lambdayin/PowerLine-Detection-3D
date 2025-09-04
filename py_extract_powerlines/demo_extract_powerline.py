"""
Power Line Extraction from Mobile LiDAR Point Cloud - Python Version
Converted from MATLAB code with identical parameters and processing logic

This script demonstrates the complete power line extraction pipeline:
1. Mobile LiDAR filtering (ground removal)
2. Candidate power line point extraction
3. Euclidean clustering
4. Power line modeling and refinement

Original MATLAB implementation by:
Zhenwei Shi, Yi Lin, and Hui Li
"Extraction of urban power lines and potential hazard analysis from mobile laser scanning point clouds."
International Journal of Remote Sensing 41, no. 9 (2020): 3411-3428.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
import time
import os
from utils import extract_pls, insert_3d
from clustering import (euclidean_clustering, create_power_line_structure, 
                       merge_power_lines, filter_short_power_lines, 
                       calculate_total_length)

def load_point_cloud_data(data_path, file_numbers=[8, 9, 10]):
    """
    Load and combine multiple point cloud files
    Equivalent to MATLAB's loading section
    """
    print("Loading point cloud data...")
    non_ground_points = []
    
    for i in file_numbers:
        filename = f'L037_Sens1_600x250_cloud_{i}.mat'
        filepath = os.path.join(data_path, filename)
        
        if not os.path.exists(filepath):
            print(f"Warning: File {filepath} not found, skipping...")
            continue
            
        print(f"Loading {filename}...")
        
        # Load MATLAB .mat file
        mat_data = scipy.io.loadmat(filepath)
        
        # Extract point cloud data (assuming structure similar to MATLAB)
        # May need adjustment based on actual .mat file structure
        if 'ptCloudA' in mat_data:
            pt_data = mat_data['ptCloudA']['data'][0][0]
        else:
            # Try alternative structures
            keys = [k for k in mat_data.keys() if not k.startswith('__')]
            if keys:
                pt_data = mat_data[keys[0]]
            else:
                print(f"Could not find point cloud data in {filename}")
                continue
        
        # Ground filtering using histogram analysis (same as MATLAB)
        z_coords = pt_data[:, 2] # extract the Z coordinates (elevation) of all points
        
        # Create histogram with 50 bins
        # z_coords are the Z-coordinate values (elevation values) for all points. Divides the range of Z-coordinates into 50 equally spaced intervals.
        counts, bin_edges = np.histogram(z_coords, bins=50)  # counts: The number of points within each interval. bin_edges: The boundary values that define the intervals.
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Find bin with maximum count
        # 找到点数最多的那个区间，核心假设：地面是场景中面积最大的连续表面，因此地面点在某个高度上会特别集中。
        max_idx = np.argmax(counts)
        
        # Ground threshold: 3 bins above the maximum density height
        # 确定地面高度阈值，以点数最密集的区间的中心高度为基准，再向上浮动3个区间的高度。
        ground_height = centers[min(max_idx + 3, len(centers) - 1)]
        
        # Extract non-ground points
        non_ground_mask = pt_data[:, 2] > ground_height
        non_ground_data = pt_data[non_ground_mask, :3]  # Take only x,y,z
        
        non_ground_points.append(non_ground_data)
        print(f"  Loaded {non_ground_data.shape[0]} non-ground points")
    
    # Combine all non-ground points
    if non_ground_points:
        combined_points = np.vstack(non_ground_points)
        print(f"Total non-ground points: {combined_points.shape[0]}")
        return combined_points
    else:
        raise ValueError("No point cloud data could be loaded")

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
    # ax.set_xlim(points[:, 1].min() - y_range * padding, points[:, 1].max() + y_range * padding)
    ax.set_xlim(points[:, 1].max() + y_range * padding, points[:, 1].min() - x_range * padding)
    ax.set_ylim(points[:, 0].min() - x_range * padding, points[:, 0].max() + x_range * padding)
    ax.set_zlim(points[:, 2].min() - z_range * padding, points[:, 2].max() + z_range * padding)
    
    # Set equal aspect ratio for better visualization
    # Method 1: Force equal aspect ratio (swapped for display)
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
    # 获取所有唯一的簇标签ID
    unique_labels = np.unique(labels)
    # - plt.cm.tab20是一个颜色映射表，包含20中视觉上区分度比较高的颜色
    # - np.linspace 会生成一个从0到1的等差数列，数量与簇的数量相同
    # - 为每个唯一的簇标签，挑选一个独一无二的颜色
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    point_colors = np.zeros((len(points), 3))
    # 遍历每个簇，为其所有点分配颜色
    for i, label in enumerate(unique_labels):
        # 创建一个布尔掩码(mask)，用于选中所有属于当前簇(label)的点
        mask = labels == label
        point_colors[mask] = colors[i][:3]
    
    return point_colors

def main():
    """
    Main power line extraction pipeline
    """
    print("=== Power Line Extraction from Mobile LiDAR Point Cloud ===")
    print("Python version - maintaining identical parameters and logic")
    print()
    
    # Configuration - Same parameters as MATLAB
    data_path = "./pointcloud_files"
    radius = 0.5           # Neighborhood search radius  
    angle_thr = 10         # Angle threshold in degrees
    l_thr = 0.98          # Linearity threshold
    min_distance_1 = 2.0   # First clustering distance
    min_distance_2 = 0.3   # Second clustering distance
    min_cluster_size = 15  # Minimum points per cluster
    min_length = 1.5       # Minimum power line length
    insert_resolution = 0.1 # Point insertion resolution
    
    try:
        # Step 1: Mobile LiDAR filtering
        print("Step 1: Mobile LiDAR filtering...")
        start_time = time.time()
        
        non_ground_points = load_point_cloud_data(data_path)
        
        end_time = time.time()
        print(f"Ground filtering completed in {end_time - start_time:.2f} seconds")
        
        # Visualize non-ground points
        visualize_points_3d(non_ground_points, 
                          "Non-ground Points",
                          save_path="f1_nonGroundPoints.png")
        
        # Step 2: Extract candidate power line points
        print("\nStep 2: Extracting candidate power line points...")
        start_time = time.time()
        
        is_pl_index = extract_pls(non_ground_points, radius, angle_thr, l_thr)
        candidate_points = non_ground_points[is_pl_index]
        
        end_time = time.time()
        print(f"Power line extraction completed in {end_time - start_time:.2f} seconds")
        print(f"Found {candidate_points.shape[0]} candidate power line points")
        
        # Visualize candidate points
        visualize_points_3d(candidate_points,
                          "Candidate Power Line Points", 
                          save_path="f2_candidate_powerline_points.png")
        
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
                          save_path="f3_candidate_powerline_points_clusters.png")
        
        # Filter clusters by minimum size
        power_lines_raw = create_power_line_structure(candidate_points, labels, 
                                                    num_clusters, min_cluster_size)
        
        # Combine filtered power lines
        if power_lines_raw:
            filtered_points = np.vstack([pl.Location for pl in power_lines_raw])
            print(f"After filtering: {len(power_lines_raw)} clusters, {filtered_points.shape[0]} points")
            
            visualize_points_3d(filtered_points,
                              "Power Line Clusters",
                              save_path="f4_powerline_points_clusters.png")
        else:
            print("No valid power line clusters found")
            return
        
        # Step 4: Power line modeling
        print("\nStep 4: Power line modeling...")
        
        # Fine clustering with smaller distance
        labels_fine, num_clusters_fine = euclidean_clustering(filtered_points, min_distance_2)
        print(f"Fine clustering found {num_clusters_fine} clusters")
        
        # Create power line structures 精细聚类结果
        # 为精细聚类结果创建PowerLine结构体
        # 这里的min_points=1意味着不过滤任何簇，保留所有精细分割结果
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
                              save_path="f5_colorization_clusters.png")
        
        # Sort power lines by count
        # 按照每个电力线簇的点数进行降序排序，获得排序索引
        # 点数多的簇排在前面，用于后续的合并优化策略
        counts = [pl.Count for pl in power_lines]
        sorted_indices = np.argsort(counts)[::-1]  # Descending order
        
        # Multiple rounds of merging (same as MATLAB)
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
                              save_path="f6_powerLines_clusters.png")
        
        # Filter short power lines
        power_lines = filter_short_power_lines(power_lines, min_length)
        print(f"After length filtering: {len(power_lines)} power lines")
        
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
                              save_path="f7_Power_line_model.png")
        
        print(f"\n=== Power Line Extraction Complete ===")
        print(f"Final results:")
        print(f"  - Number of power lines: {len(power_lines_final)}")
        print(f"  - Total length: {total_length:.2f} meters")
        print(f"  - Total points in model: {sum([pl.Location.shape[0] for pl in power_lines_final])}")
        
        return power_lines_final
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = main()