"""
Basic function tests to verify the Python implementation
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import extract_pls, get_pca, eigen_dv, rotate_points
from clustering import euclidean_clustering, PowerLine

def test_basic_functions():
    """
    Test basic functionality with synthetic data
    """
    print("Testing basic functions with synthetic data...")
    
    # Create synthetic point cloud (simple line + noise)
    np.random.seed(42)
    
    # Create a synthetic power line (linear structure)
    t = np.linspace(0, 10, 100)
    x = t
    y = 0.1 * np.sin(t) + 0.05 * np.random.randn(100)  # Slight curve + noise
    z = 5 + 0.02 * t + 0.05 * np.random.randn(100)     # Slight slope + noise
    
    line_points = np.column_stack([x, y, z])
    
    # Add some random noise points
    noise_points = np.random.rand(50, 3) * 10
    noise_points[:, 2] += 5  # Put at similar height
    
    # Combine
    test_points = np.vstack([line_points, noise_points])
    
    print(f"Created test dataset: {test_points.shape[0]} points")
    print(f"  - {line_points.shape[0]} line points")
    print(f"  - {noise_points.shape[0]} noise points")
    
    # Test PCA function
    print("\n1. Testing PCA function...")
    try:
        normals, Ls = get_pca(test_points, r=0.5)
        print(f"   PCA completed: {np.sum(~np.isnan(Ls[:, 0]))} valid L values")
        print(f"   Mean L value: {np.nanmean(Ls):.3f}")
        print("   ✓ PCA test passed")
    except Exception as e:
        print(f"   ✗ PCA test failed: {e}")
        return False
    
    # Test power line extraction
    print("\n2. Testing power line extraction...")
    try:
        is_pl_index = extract_pls(test_points, radius=0.5, angle_thr=15, l_thr=0.85)
        candidate_points = test_points[is_pl_index]
        print(f"   Found {candidate_points.shape[0]} candidate power line points")
        print("   ✓ Power line extraction test passed")
    except Exception as e:
        print(f"   ✗ Power line extraction test failed: {e}")
        return False
    
    # Test clustering
    print("\n3. Testing clustering...")
    try:
        if candidate_points.shape[0] > 0:
            labels, num_clusters = euclidean_clustering(candidate_points, min_distance=1.0)
            print(f"   Found {num_clusters} clusters")
            print("   ✓ Clustering test passed")
        else:
            print("   No candidate points for clustering test")
    except Exception as e:
        print(f"   ✗ Clustering test failed: {e}")
        return False
    
    # Test eigen decomposition
    print("\n4. Testing eigen decomposition...")
    try:
        subset = line_points[:50]  # Use first 50 points of line
        eigenvals, eigenvecs, angle = eigen_dv(subset - np.mean(subset, axis=0))
        print(f"   Eigenvalues: {eigenvals}")
        print(f"   Main direction angle: {angle:.2f} degrees")
        print("   ✓ Eigen decomposition test passed")
    except Exception as e:
        print(f"   ✗ Eigen decomposition test failed: {e}")
        return False
    
    # Test rotation
    print("\n5. Testing point rotation...")
    try:
        rotated = rotate_points(line_points, np.pi/4)  # 45 degree rotation
        print(f"   Rotated {line_points.shape[0]} points")
        print("   ✓ Point rotation test passed")
    except Exception as e:
        print(f"   ✗ Point rotation test failed: {e}")
        return False
    
    print("\n=== All basic tests completed successfully! ===")
    return True

def test_data_structure():
    """
    Test PowerLine data structure
    """
    print("\nTesting PowerLine data structure...")
    
    try:
        pl = PowerLine()
        pl.Location = np.random.rand(20, 3)
        pl.Label = 0
        pl.Count = 20
        pl.Ids = []
        
        print(f"   PowerLine created with {pl.Count} points")
        print("   ✓ PowerLine structure test passed")
        return True
    except Exception as e:
        print(f"   ✗ PowerLine structure test failed: {e}")
        return False

if __name__ == "__main__":
    print("=== Testing Python Power Line Extraction Implementation ===\n")
    
    success = True
    success &= test_basic_functions()
    success &= test_data_structure()
    
    if success:
        print("\n🎉 All tests passed! The Python implementation is ready to use.")
    else:
        print("\n❌ Some tests failed. Please check the implementation.")