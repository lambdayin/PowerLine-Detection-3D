"""
Quick Start Demo for Power Line Extraction
A simplified version for quick testing and demonstration
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
import os
from utils import extract_pls
from clustering import euclidean_clustering

def load_single_cloud(file_path):
    """
    Load a single point cloud file for quick testing
    """
    print(f"Loading {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    # Load MATLAB .mat file
    mat_data = scipy.io.loadmat(file_path)
    
    # Extract point cloud data
    if 'ptCloudA' in mat_data:
        pt_data = mat_data['ptCloudA']['data'][0][0]
    else:
        # Try alternative structures
        keys = [k for k in mat_data.keys() if not k.startswith('__')]
        if keys:
            pt_data = mat_data[keys[0]]
        else:
            print(f"Could not find point cloud data")
            return None
    
    # Simple ground filtering
    z_coords = pt_data[:, 2]
    z_median = np.median(z_coords)
    non_ground_mask = pt_data[:, 2] > z_median + 2  # Simple threshold
    non_ground_points = pt_data[non_ground_mask, :3]
    
    print(f"Loaded {non_ground_points.shape[0]} non-ground points")
    return non_ground_points

def quick_visualize(points, title="Point Cloud", sample_size=5000):
    """
    Quick 3D visualization with sampling for performance
    """
    # Sample points for faster visualization
    if points.shape[0] > sample_size:
        indices = np.random.choice(points.shape[0], sample_size, replace=False)
        points_sample = points[indices]
    else:
        points_sample = points
    
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(points_sample[:, 0], points_sample[:, 1], points_sample[:, 2], 
               s=1, alpha=0.6, c=points_sample[:, 2], cmap='viridis')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.view_init(elev=25, azim=60)
    
    plt.tight_layout()
    plt.show()
    plt.close()

def quick_demo():
    """
    Quick demonstration of power line extraction
    """
    print("=== Quick Start Power Line Extraction Demo ===")
    print()
    
    # Try to load a single point cloud file
    data_path = "../pointcloud_files"
    test_files = [
        "L037_Sens1_600x250_cloud_8.mat",
        "L037_Sens1_600x250_cloud_9.mat", 
        "L037_Sens1_600x250_cloud_10.mat"
    ]
    
    points = None
    for filename in test_files:
        filepath = os.path.join(data_path, filename)
        points = load_single_cloud(filepath)
        if points is not None:
            print(f"Successfully loaded: {filename}")
            break
    
    if points is None:
        print("Could not load any point cloud files.")
        print("Please ensure the .mat files are in the ../pointcloud_files directory")
        return
    
    # Quick visualization of raw points
    print("\nVisualizing non-ground points...")
    quick_visualize(points, "Non-ground Points")
    
    # Extract candidate power line points
    print("Extracting candidate power line points...")
    
    # Use same parameters as original MATLAB
    radius = 0.5
    angle_thr = 10  
    l_thr = 0.98
    
    try:
        is_pl_index = extract_pls(points, radius, angle_thr, l_thr)
        candidate_points = points[is_pl_index]
        
        print(f"Found {candidate_points.shape[0]} candidate power line points")
        
        if candidate_points.shape[0] > 0:
            # Visualize candidate points
            print("Visualizing candidate power line points...")
            quick_visualize(candidate_points, "Candidate Power Line Points")
            
            # Simple clustering
            if candidate_points.shape[0] > 10:
                print("Performing simple clustering...")
                labels, num_clusters = euclidean_clustering(candidate_points, 2.0)
                print(f"Found {num_clusters} potential power line clusters")
                
                # Show clusters with colors
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                
                # Sample for visualization
                sample_size = min(2000, candidate_points.shape[0])
                indices = np.random.choice(candidate_points.shape[0], sample_size, replace=False)
                sample_points = candidate_points[indices]
                sample_labels = labels[indices]
                
                scatter = ax.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2],
                                   c=sample_labels, cmap='tab20', s=2, alpha=0.7)
                
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z') 
                ax.set_title('Power Line Clusters')
                ax.view_init(elev=25, azim=60)
                
                plt.colorbar(scatter)
                plt.tight_layout()
                plt.show()
                plt.close()
                
        else:
            print("No candidate power line points found.")
            print("You may need to adjust the parameters (radius, angle_thr, l_thr)")
    
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== Quick Demo Complete ===")
    print("For full processing, run: python demo_extract_powerline.py")

if __name__ == "__main__":
    quick_demo()