#!/usr/bin/env python3
"""
合并多个点云文件为一个点云文件

支持合并.mat格式的点云文件，并输出为.mat或.pcd格式
"""

import sys
import os
import argparse
import scipy.io
import numpy as np
from pathlib import Path


def load_mat_pointcloud(mat_file):
    """从.mat文件加载点云数据"""
    try:
        data = scipy.io.loadmat(mat_file)
        if 'ptCloudA' not in data:
            raise ValueError(f"文件 {mat_file} 中未找到 'ptCloudA' 键")
        
        ptCloud = data['ptCloudA'][0, 0]
        points = ptCloud[0]  # 实际点云数据
        headers = ptCloud[1] if len(ptCloud) > 1 else None
        
        print(f"已加载 {mat_file}: {points.shape[0]} 个点")
        return points, headers
    
    except Exception as e:
        print(f"加载 {mat_file} 失败: {str(e)}")
        raise


def save_merged_mat(points, headers, output_file):
    """保存合并后的点云为.mat格式"""
    try:
        # 构建与原始格式相同的数据结构
        if headers is not None:
            ptCloudA = np.array([[points, headers]], dtype=object)
        else:
            # 如果没有headers，创建默认的headers
            default_headers = np.array([[
                np.array(['//X'], dtype='<U3'),
                np.array(['Y'], dtype='<U1'), 
                np.array(['Z'], dtype='<U1'),
                np.array(['Point_Source_ID'], dtype='<U15'),
                np.array(['Scan_Angle_Rank'], dtype='<U15'),
                np.array(['Scan_Direction'], dtype='<U14'),
                np.array(['Number_of_Returns'], dtype='<U17'),
                np.array(['Return_Number'], dtype='<U13'),
                np.array(['Time'], dtype='<U4'),
                np.array(['Intensity'], dtype='<U9')
            ]], dtype=object)
            ptCloudA = np.array([[points, default_headers]], dtype=object)
        
        # 保存为.mat文件
        scipy.io.savemat(output_file, {'ptCloudA': ptCloudA})
        print(f"合并点云已保存为: {output_file}")
        
    except Exception as e:
        print(f"保存 {output_file} 失败: {str(e)}")
        raise


def write_pcd_header(f, num_points, has_intensity=False):
    """写入PCD文件头"""
    if has_intensity:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z intensity\n")
        f.write("SIZE 4 4 4 4\n")
        f.write("TYPE F F F F\n")
        f.write("COUNT 1 1 1 1\n")
    else:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
    
    f.write(f"WIDTH {num_points}\n")
    f.write("HEIGHT 1\n")
    f.write(f"VIEWPOINT 0 0 0 1 0 0 0\n")
    f.write(f"POINTS {num_points}\n")
    f.write("DATA ascii\n")


def save_merged_pcd(points, output_file, include_intensity=True):
    """保存合并后的点云为.pcd格式"""
    try:
        x = points[:, 0]
        y = points[:, 1] 
        z = points[:, 2]
        
        # 检查是否有强度数据
        intensity = None
        if include_intensity and points.shape[1] >= 10:
            intensity = points[:, 9]
        
        with open(output_file, 'w') as f:
            write_pcd_header(f, points.shape[0], intensity is not None)
            
            # 写入点云数据
            if intensity is not None:
                for i in range(points.shape[0]):
                    f.write(f"{x[i]:.6f} {y[i]:.6f} {z[i]:.6f} {intensity[i]:.6f}\n")
            else:
                for i in range(points.shape[0]):
                    f.write(f"{x[i]:.6f} {y[i]:.6f} {z[i]:.6f}\n")
        
        print(f"合并点云已保存为: {output_file}")
        
    except Exception as e:
        print(f"保存 {output_file} 失败: {str(e)}")
        raise


def merge_point_clouds(input_files, output_file, output_format='auto'):
    """
    合并多个点云文件
    
    Args:
        input_files: 输入文件列表
        output_file: 输出文件路径
        output_format: 输出格式 ('mat', 'pcd', 'auto')
    """
    print(f"开始合并 {len(input_files)} 个点云文件...")
    
    all_points = []
    headers = None
    total_points = 0
    
    # 加载所有点云文件
    for i, file_path in enumerate(input_files):
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在，跳过")
            continue
        
        points, file_headers = load_mat_pointcloud(file_path)
        all_points.append(points)
        total_points += points.shape[0]
        
        # 使用第一个文件的headers作为合并后的headers
        if headers is None:
            headers = file_headers
    
    if not all_points:
        raise ValueError("没有成功加载任何点云文件")
    
    # 合并所有点云数据
    print(f"正在合并 {total_points} 个点...")
    merged_points = np.vstack(all_points)
    print(f"合并完成，总共 {merged_points.shape[0]} 个点")
    
    # 确定输出格式
    if output_format == 'auto':
        output_format = 'pcd' if output_file.lower().endswith('.pcd') else 'mat'
    
    # 保存合并后的点云
    if output_format == 'mat':
        save_merged_mat(merged_points, headers, output_file)
    elif output_format == 'pcd':
        save_merged_pcd(merged_points, output_file)
    else:
        raise ValueError(f"不支持的输出格式: {output_format}")
    
    # 显示统计信息
    print(f"\n合并统计信息:")
    print(f"  输入文件数: {len(input_files)}")
    print(f"  总点数: {merged_points.shape[0]}")
    print(f"  特征数: {merged_points.shape[1]}")
    print(f"  X范围: {merged_points[:, 0].min():.2f} - {merged_points[:, 0].max():.2f}")
    print(f"  Y范围: {merged_points[:, 1].min():.2f} - {merged_points[:, 1].max():.2f}")
    print(f"  Z范围: {merged_points[:, 2].min():.2f} - {merged_points[:, 2].max():.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="合并多个点云文件为一个点云文件"
    )
    parser.add_argument(
        "input_files",
        nargs="+",
        help="输入的.mat点云文件路径（可以是多个文件或目录）"
    )
    parser.add_argument(
        "-o", "--output",
        required=True,
        help="输出文件路径（.mat或.pcd格式）"
    )
    parser.add_argument(
        "--format",
        choices=['mat', 'pcd', 'auto'],
        default='auto',
        help="输出格式（默认根据文件扩展名自动判断）"
    )
    
    args = parser.parse_args()
    
    # 处理输入文件列表
    input_files = []
    for input_path in args.input_files:
        path = Path(input_path)
        if path.is_file() and path.suffix.lower() == '.mat':
            input_files.append(str(path))
        elif path.is_dir():
            # 如果是目录，添加目录中的所有.mat文件
            mat_files = list(path.glob("*.mat"))
            input_files.extend([str(f) for f in mat_files])
        else:
            print(f"警告: {input_path} 不是有效的.mat文件或目录")
    
    if not input_files:
        print("错误: 没有找到有效的.mat文件")
        return 1
    
    # 排序文件列表以确保一致的合并顺序
    input_files.sort()
    
    print(f"找到 {len(input_files)} 个.mat文件:")
    for f in input_files:
        print(f"  - {f}")
    
    try:
        merge_point_clouds(input_files, args.output, args.format)
        print("\n合并完成！")
        return 0
    
    except Exception as e:
        print(f"合并失败: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())