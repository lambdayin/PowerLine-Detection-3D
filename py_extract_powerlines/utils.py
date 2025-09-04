"""
Utility functions for power line extraction from point cloud data
Converted from MATLAB code with identical parameters and logic
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eig
import warnings

def get_pca(xyz, r):
    """
    Compute PCA features for each point in the point cloud
    Equivalent to getPCA.m function
    
    Args:
        xyz: Point cloud data (N x 3 numpy array)
        r: Search radius for neighborhood
    
    Returns:
        normals: Normal vectors (N x 3)
        Ls: Linearity values (N x 1)
    """
    n_points = xyz.shape[0]
    normals = np.full((n_points, 3), np.nan)
    Ls = np.full((n_points, 1), np.nan)
    
    # Use NearestNeighbors to find points within radius
    nbrs = NearestNeighbors(radius=r, algorithm='ball_tree').fit(xyz)
    distances, indices = nbrs.radius_neighbors(xyz)
    
    for i in range(n_points):
        neighbors_idx = indices[i]
        
        if len(neighbors_idx) < 3:
            continue
            
        # Get neighborhood points
        XNN = xyz[neighbors_idx, :]
        
        # Compute covariance matrix
        # 对于每个点的邻域点集，计算其三维坐标的协方差矩阵
        covm = np.cov(XNN.T)
        
        # Compute eigenvalues and eigenvectors
        # 特征向量给出了点集分布的三个主方向。最小特征值对应的特征向量就是该点邻域点集的法向量
        # 特征值表示点集在三个主方向上的离散程度(方差)
        eigenvalues, eigenvectors = eig(covm)
        
        # Convert to real values (eigenvalues should be real for covariance matrix)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        # Sort eigenvalues in descending order 降序排列 λ1最大、λ3最小
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Normal is the eigenvector corresponding to smallest eigenvalue
        # 将最大特征值对应的特征向量（即点集的主方向）存入normals数组
        normals[i, :] = eigenvectors[:, 0]
        
        # Compute geometric features
        lmbda1 = eigenvalues[0]
        lmbda2 = eigenvalues[1] 
        lmbda3 = eigenvalues[2]
        
        # Linearity feature (same as MATLAB)
        # 如果点集呈线状分布，则 λ1 会远大于 λ2 和 λ3，L 的值会趋近于 1
        # 如果点集呈面状分布，则 λ1 和 λ2 会比较接近，L 的值会趋近于 0
        # 如果点集呈球状分布，则三个特征值都会很接近，L 的值也趋近于 0
        L = (lmbda1 - lmbda2) / lmbda1
        Ls[i, 0] = L
    
    return normals, Ls

def extract_pls(non_ground_points, radius, angle_thr, l_thr):
    """
    Extract candidate power line points based on linear feature analysis
    Equivalent to extractPLs.m function
    
    Args:
        non_ground_points: Non-ground point cloud (N x 3)
        radius: Neighborhood search radius
        angle_thr: Angle threshold in degrees
        l_thr: Linearity threshold
        
    Returns:
        is_pl_index: Boolean array indicating power line points
    """
    # Get PCA features, 所有非地面点的法向量和线性度
    # normals: 每个点邻域的主方向向量，Ls是每个点邻域点集的线性度
    normals, Ls = get_pca(non_ground_points, radius)
    
    # Calculate angle between normal and vertical direction (0,0,1)
    # Equivalent to: angle = acosd(normals(:,3)./sqrt(sum(normals.^2,2)))
    # 首先计算每个主方向向量的模长（归一化），然后计算每个点的主方向向量与绝对垂直向量之间的夹角
    normal_magnitudes = np.sqrt(np.sum(normals**2, axis=1))
    angles = np.arccos(normals[:, 2] / normal_magnitudes) * 180 / np.pi
    
    # Apply filters: angle close to 90 degrees and high linearity
    # Equivalent to: isPLIndex = abs(angle - 90) < angleThr & Ls>LThr
    # 双重条件过滤，找出最有可能是电力线的点
    # 条件A：点的线性度 条件B：主方向向量必须与垂直方向接近90度
    is_pl_index = (np.abs(angles - 90) < angle_thr) & (Ls[:, 0] > l_thr)
    
    # Handle NaN values
    is_pl_index = np.nan_to_num(is_pl_index, nan=False)
    
    return is_pl_index

def eigen_dv(points):
    """
    Compute eigen decomposition for directional analysis
    Equivalent to eigenDV.m function
    """
    if points.shape[0] < 3:
        return 0, np.eye(3), 0
    
    # Compute covariance matrix
    covm = np.cov(points.T)
    
    # Eigen decomposition
    eigenvalues, eigenvectors = eig(covm)
    
    # Convert to real values
    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)
    
    # Sort by eigenvalues (descending)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Main direction is first eigenvector
    main_vector = eigenvectors[:, 0]
    
    # Calculate angle in degrees (projection to XY plane)
    angle = np.arctan2(main_vector[1], main_vector[0]) * 180 / np.pi
    
    return eigenvalues, eigenvectors, angle

def rotate_points(points, angle_rad):
    """
    Rotate points around Z axis
    Equivalent to rotate.m function
    
    Args:
        points: Points to rotate (N x 3)
        angle_rad: Rotation angle in radians
        
    Returns:
        rotated_points: Rotated points
    """
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Rotation matrix around Z axis
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    
    return points @ rotation_matrix.T

def get_dist(power_lines):
    """
    Calculate power line length
    Equivalent to getDist.m function
    """
    if power_lines.shape[0] < 3:
        return 0
    
    # Center the points
    shift = power_lines - np.mean(power_lines, axis=0)
    
    # Get main direction
    eigenvalues, eigenvectors, angle = eigen_dv(shift)
    
    # Rotate to align with main direction
    rotated = rotate_points(shift, -angle * np.pi / 180.0)
    shift_x = rotated[:, 0]
    
    # Length is the range in main direction
    dist = np.max(shift_x) - np.min(shift_x)
    
    return dist

def insert_3d(cluster_raw, resolution):
    """
    Insert points to sparse power lines with fixed resolution
    Equivalent to insert_3D.m function
    """
    if cluster_raw.shape[0] < 3:
        return cluster_raw
        
    # Center the cluster
    cluster_shift = cluster_raw - np.mean(cluster_raw, axis=0)
    
    # Get main direction using eigen decomposition
    eigenvalues, eigenvectors, angle = eigen_dv(cluster_shift)
    
    # Rotate to align with main axis
    rotated = rotate_points(cluster_shift, -angle * np.pi / 180.0)
    x = rotated[:, 0]
    z = rotated[:, 2]
    
    # Fit polynomial (degree 2, same as MATLAB polyfit)
    p = np.polyfit(x, z, 2)
    
    # Insert points with given resolution
    x_new = insert_points_1d(x, z, p, resolution)
    
    if x_new is None:
        return cluster_raw
        
    # Create new points (y=0 as in MATLAB code)
    xyz_new = np.column_stack([x_new, np.zeros(len(x_new)), np.polyval(p, x_new)])
    
    # Rotate back and translate
    xyz_new = rotate_points(xyz_new, angle * np.pi / 180.0) + np.mean(cluster_raw, axis=0)
    
    return xyz_new

def insert_points_1d(x, z, poly_coeffs, resolution):
    """
    Insert points along 1D curve with given resolution
    Equivalent to insert.m function logic
    """
    x_min, x_max = np.min(x), np.max(x)
    
    # Generate new x coordinates with given resolution
    num_points = int((x_max - x_min) / resolution) + 1
    x_new = np.linspace(x_min, x_max, num_points)
    
    return x_new

def fit_catenary(x, z):
    """
    Fit catenary model to power line points
    Equivalent to catenary.m function
    Note: This is a simplified version. Full implementation would need
    nonlinear optimization similar to MATLAB's curve fitting.
    """
    # For now, use polynomial fit as fallback
    # Full catenary implementation would require scipy.optimize
    return np.polyfit(x, z, 2)