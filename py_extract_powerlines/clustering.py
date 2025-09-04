"""
Clustering and merging functions for power line extraction
Converted from MATLAB code with identical parameters and logic
"""

import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
import copy
from utils import eigen_dv, rotate_points, get_dist

class PowerLine:
    """
    Power line structure equivalent to MATLAB struct
    """
    def __init__(self):
        self.Location = np.array([])
        self.Label = 0
        self.Count = 0
        self.Ids = []

def euclidean_clustering(points, min_distance):
    """
    Perform Euclidean clustering equivalent to MATLAB's pcsegdist
    
    Args:
        points: Point cloud data (N x 3)
        min_distance: Minimum distance for clustering
        
    Returns:
        labels: Cluster labels for each point
        num_clusters: Number of clusters found
    """
    # Use DBSCAN for clustering (eps = min_distance, min_samples = 1)
    # min_distance：聚类的最小距离阈值，如果两点之间的距离小于这个值，它们可能被分到同一簇
    # min_samples：确保每个点都会被分配到一个簇，即使是离群点
    # labels[0] 表示第0个点所属的簇标签
    clustering = DBSCAN(eps=min_distance, min_samples=1).fit(points)
    labels = clustering.labels_
    
    # Convert -1 (noise) to separate clusters
    # 备用逻辑：处理DBSCAN可能产生的噪声点(标签为-1)
    # 尽管因为min_samples=1理论上不会产生噪声点，但依然保留此逻辑增强代码鲁棒性
    unique_labels = np.unique(labels)
    if -1 in unique_labels:
        # Reassign noise points to individual clusters
        # 如果存在标签为-1的噪声点，则将每个噪声点重新分配为一个独立的簇
        noise_mask = labels == -1
        noise_indices = np.where(noise_mask)[0]
        # 找到下一个可用的簇标签ID
        next_label = max(labels) + 1
        # 便利所有噪声点，并为它们分配新的、唯一的簇标签
        for idx in noise_indices:
            labels[idx] = next_label
            next_label += 1
    
    # Make labels 1-based to match MATLAB
    labels = labels + 1
    num_clusters = len(np.unique(labels))
    
    return labels, num_clusters

def find_merge(power_lines, index, begin):
    """
    Find power line segments to merge
    Equivalent to findMerge.m function
    
    Args:
        power_lines: List of PowerLine objects
        index: Sorted indices 
        begin: Starting index for merging
        
    Returns:
        ids: List of indices to merge
    """
    ids = []
    A = power_lines[index[begin]].Location
    
    if A.shape[0] < 10:
        return ids
    
    mean_Az = np.mean(A[:, 2])
    
    # Center points A
    A_shift = A - np.mean(A, axis=0)
    eValue, eVector, angle = eigen_dv(A_shift)
    
    # Rotate to align with main direction
    A_rotated = rotate_points(A_shift, -angle * np.pi / 180.0)
    A_shift_x = A_rotated[:, 0]
    A_shift_z = A_rotated[:, 2]
    
    # Fit polynomial (degree 2)
    p = np.polyfit(A_shift_x, A_shift_z, 2)
    
    # Fit line in XY plane
    pl = np.polyfit(A[:, 0], A[:, 1], 1)
    
    # Check other power lines for merging
    for i in range(begin + 1, len(index)):
        if power_lines[index[i]].Label == 1:
            continue
            
        B = power_lines[index[i]].Location
        mean_Bz = np.mean(B[:, 2])
        
        # Transform B using A's coordinate system
        PLB_shift = B - np.mean(A, axis=0)  # Center B relative to A
        B_rotated = rotate_points(PLB_shift, -angle * np.pi / 180.0)
        B_shift_x = B_rotated[:, 0]
        B_shift_z = B_rotated[:, 2]
        
        # Predict Z values using polynomial fit
        B_Pz = np.polyval(p, B_shift_x)
        
        # Predict Y values using line fit
        B_Py = np.polyval(pl, B[:, 0])
        
        # Calculate deviations
        delta_By = np.abs(np.mean(B_Py - B[:, 1]))
        delta_Bz = np.abs(mean_Az - mean_Bz)
        mean_Bd = np.max(np.abs(B_Pz - B_shift_z))
        
        # Merging criteria (same as MATLAB)
        if delta_By < 0.5 and mean_Bd < 0.2 and delta_Bz < 4:
            result = 1
        else:
            result = 0
            
        if result == 1:
            ids.append(index[i])
            # Update polynomial with combined data
            A_shift_x = np.concatenate([A_shift_x, B_shift_x])
            A_shift_z = np.concatenate([A_shift_z, B_shift_z])
            p = np.polyfit(A_shift_x, A_shift_z, 2)
    
    return ids

def merge_power_lines(power_lines, ind):
    """
    Merge power line segments
    Equivalent to merge.m function
    
    Args:
        power_lines: List of PowerLine objects
        ind: Sorted indices
        
    Returns:
        power_lines_pro: Processed power lines after merging
        index: Updated indices
    """
    # Mark power lines to be merged
    for i in range(len(ind)):
        if power_lines[ind[i]].Label == 1:
            continue
        
        ids = find_merge(power_lines, ind, i)
        power_lines[ind[i]].Ids = ids
        
        # Mark merged power lines
        for j in ids:
            power_lines[j].Label = 1
    
    # Create new list excluding marked power lines
    power_lines_pro = []
    for pl in power_lines:
        if pl.Label == 0:
            power_lines_pro.append(copy.deepcopy(pl))
    
    # Merge locations from marked power lines
    for i in range(len(power_lines_pro)):
        for j in power_lines_pro[i].Ids:
            power_lines_pro[i].Location = np.vstack([
                power_lines_pro[i].Location, 
                power_lines[j].Location
            ])
    
    # Update counts and reset labels
    counts = []
    for i in range(len(power_lines_pro)):
        power_lines_pro[i].Label = 0
        power_lines_pro[i].Count = power_lines_pro[i].Location.shape[0]
        power_lines_pro[i].Ids = []
        counts.append(power_lines_pro[i].Count)
    
    # Sort by count (descending)
    if counts:
        index = np.argsort(counts)[::-1]
    else:
        index = np.array([])
    
    return power_lines_pro, index

def filter_short_power_lines(power_lines_pro, min_length=1.5):
    """
    Filter out short power lines
    
    Args:
        power_lines_pro: List of PowerLine objects
        min_length: Minimum length threshold
        
    Returns:
        filtered_power_lines: Power lines after filtering
    """
    filtered_power_lines = []
    
    for pl in power_lines_pro:
        dist = get_dist(pl.Location)
        if dist >= min_length:
            filtered_power_lines.append(pl)
    
    return filtered_power_lines

def calculate_total_length(power_lines_pro):
    """
    Calculate total length of all power lines
    
    Args:
        power_lines_pro: List of PowerLine objects
        
    Returns:
        total_length: Sum of all power line lengths
    """
    total_length = 0
    
    for pl in power_lines_pro:
        dist = get_dist(pl.Location)
        total_length += dist
    
    return total_length

def create_power_line_structure(points_list, labels, num_clusters, min_points=15):
    """
    Create PowerLine structures from clustered points
    
    Args:
        points_list: List of point arrays
        labels: Cluster labels
        num_clusters: Number of clusters
        min_points: Minimum points per cluster
        
    Returns:
        power_lines: List of PowerLine objects
    """
    power_lines = []
    
    for i in range(1, num_clusters + 1):  # 1-based labels
        # 创建一个布尔掩码(mask)，用于选中所有属于当前簇(label)的点
        cluster_mask = labels == i
        # 根据掩码选中原始电云中属于当前簇的点
        cluster_points = points_list[cluster_mask]
        
        # 检查当前簇的点数是否满足最小点数要求
        if cluster_points.shape[0] >= min_points:
            # 如果满足条件，则创建一个新的PowerLine对象
            pl = PowerLine()
            pl.Location = cluster_points
            pl.Label = 0
            pl.Count = cluster_points.shape[0]
            pl.Ids = []
            power_lines.append(pl)
    
    return power_lines