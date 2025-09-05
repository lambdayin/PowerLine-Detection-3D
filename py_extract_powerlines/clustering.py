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
    # min_samples=1：确保每个点都会被分配到一个簇，即使是离群点
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
    # 获取基准电力线段‘A’的点云数据
    A = power_lines[index[begin]].Location
    
    # 如果基准电力线段‘A’的点数小于10个，则认为其不可靠，不进行合并操作
    if A.shape[0] < 10:
        return ids
    
    #计算基准电力线A的平均高度（Z值）
    mean_Az = np.mean(A[:, 2])
    
    # --- 步骤1：对基准线段A进行主成分分析和模型你和 ---

    # Center points A
    # 将A的点云数据中心化（减去均值），使其质心位于原点
    A_shift = A - np.mean(A, axis=0)
    # 计算特征值 特征向量 角度
    eValue, eVector, angle = eigen_dv(A_shift)
    
    # Rotate to align with main direction
    # 将中心化后的点云旋转，使其主方向与X轴对齐
    # 将三维问题简化为二维，便于拟合悬链线（电力线弧垂）
    A_rotated = rotate_points(A_shift, -angle * np.pi / 180.0)
    A_shift_x = A_rotated[:, 0]  # 旋转后的X坐标
    A_shift_z = A_rotated[:, 2]  # 旋转后的Z坐标
    
    # Fit polynomial (degree 2)
    # 使用二次多项式拟合旋转后的XZ数据，以模拟电力线的弧垂（悬链线形状）
    p = np.polyfit(A_shift_x, A_shift_z, 2)
    
    # Fit line in XY plane
    # 使用一次多项式（直线）你和原始的XY平面，以模拟电力线的水平走向
    pl = np.polyfit(A[:, 0], A[:, 1], 1)
    
    # --- 步骤2：遍历其他线段，判断是否可以合并 --- 

    # Check other power lines for merging
    # 从基准线段的下一个开始遍历
    for i in range(begin + 1, len(index)):
        # 如果候选线段已被标记为合并（Label=1），则跳过
        if power_lines[index[i]].Label == 1:
            continue
            
        # 获取候选线段‘B’的点云数据
        B = power_lines[index[i]].Location
        # 计算候选线段B的平均高度
        mean_Bz = np.mean(B[:, 2])
        
        # --- 步骤3：在基准线段A的坐标系下变换和评估候选线段B ---

        # Transform B using A's coordinate system
        # 使用A的均值对B进行中心化，并应用与A相同的旋转
        PLB_shift = B - np.mean(A, axis=0)  # Center B relative to A
        B_rotated = rotate_points(PLB_shift, -angle * np.pi / 180.0)
        B_shift_x = B_rotated[:, 0]
        B_shift_z = B_rotated[:, 2]
        
        # Predict Z values using polynomial fit
        # 使用A的弧垂模型(p)预测B在A坐标系下Z值
        B_Pz = np.polyval(p, B_shift_x)
        
        # Predict Y values using line fit
        # 使用A的水平走向模型(pl)预测B在原始坐标系下的Y值
        B_Py = np.polyval(pl, B[:, 0])
        
        # --- 步骤4：计算偏差并应用合并准则 ---
        
        # Calculate deviations
        # 计算B在XY平面上的平均偏差
        delta_By = np.abs(np.mean(B_Py - B[:, 1]))
        # 计算A和B的平均高度差
        delta_Bz = np.abs(mean_Az - mean_Bz)
        # 计算B与A的弧垂模型的最大偏差
        mean_Bd = np.max(np.abs(B_Pz - B_shift_z))
        
        # Merging criteria (same as MATLAB)
        # dlta_By < 0.5: 水平走向偏差小
        # mean_By < 0.2: 弧垂形状偏差小
        # delta_Bz < 4: 整体高度差异不大
        if delta_By < 0.5 and mean_Bd < 0.2 and delta_Bz < 4:
            result = 1
        else:
            result = 0
            
        if result == 1:
            # 如果决定合并，将B的索引添加到待合并列表ids中
            ids.append(index[i])
            # Update polynomial with combined data
            # **关键步骤**：将B的点云数据加入到A的模型中，并更新弧垂模型
            # 这样做可以使模型随着合并的进行而变得更加鲁邦，更好的判断后续的线段
            A_shift_x = np.concatenate([A_shift_x, B_shift_x])
            A_shift_z = np.concatenate([A_shift_z, B_shift_z])
            p = np.polyfit(A_shift_x, A_shift_z, 2)
    
    # 返回所有被确定可以合并到基准线段A的线段索引列表
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
    # 步骤1：标记待合并的电力线段
    for i in range(len(ind)):
        # 如果一个电力线段的Label为1，说明它已经被合并到其他线段中，直接跳过
        if power_lines[ind[i]].Label == 1:
            continue
        
        # 调用find_merge函数，查找可以与当前电力线段合并的其他线段
        # ind[i]是当前作为基准的电力线段
        ids = find_merge(power_lines, ind, i)
        # 将找到的待合并线段的索引列表(ids)存入当前基准线段的Ids属性中
        power_lines[ind[i]].Ids = ids
        
        # Mark merged power lines
        # 将被合并的线段的Label标记为1，避免它们在后续循环中成为新的合并基准
        for j in ids:
            power_lines[j].Label = 1
    
    # Create new list excluding marked power lines
    # 步骤2：创建新的电力线列表，仅包含未被合并的（Label为0）的基准线段
    power_lines_pro = []
    for pl in power_lines:
        if pl.Label == 0:
            power_lines_pro.append(copy.deepcopy(pl))
    
    # Merge locations from marked power lines
    # 步骤3：将被合并线段的点云数据合并到基准线段中
    for i in range(len(power_lines_pro)):
        # 遍历每个基准线段中记录的待合并线段索引(Ids)
        for j in power_lines_pro[i].Ids:
            # 将被合并线段(power_lines[j])的点云位置(Location)数据
            power_lines_pro[i].Location = np.vstack([
                power_lines_pro[i].Location, 
                power_lines[j].Location
            ])
    
    # Update counts and reset labels
    # 步骤4：更新合并后线段的属性并准备重新排序
    counts = []
    for i in range(len(power_lines_pro)):
        power_lines_pro[i].Label = 0
        power_lines_pro[i].Count = power_lines_pro[i].Location.shape[0]
        power_lines_pro[i].Ids = []
        counts.append(power_lines_pro[i].Count)
    
    # Sort by count (descending)
    # 步骤5：根据新的点数对合并后的电力线段进行降序排序
    if counts:
        # 降序排列
        index = np.argsort(counts)[::-1]
    else:
        # 如果列表为空，返回空索引数组
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
        # 调用get_dist 函数计算当前电力线段的长度
        # get_dist 通过主成分分析计算电云在其方向上的跨度作为其长度
        dist = get_dist(pl.Location)

        # 判断该线段的长度是否大于或等于设定的最小长度阈值
        if dist >= min_length:
            # 如果长度符合要求，则将其添加到过滤后的列表中
            filtered_power_lines.append(pl)
    
    # 返回只包含合格长度电力线段的新列表
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