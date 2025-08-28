#!/usr/bin/env python3
"""
Temporary script to examine .mat file structure
"""
import scipy.io

# Load and examine the structure of a sample .mat file
mat_file = '../pointcloud_files/L037_Sens1_600x250_cloud_10.mat'
data = scipy.io.loadmat(mat_file)

print("Keys in .mat file:", list(data.keys()))

# Get the point cloud data
ptCloud = data['ptCloudA'][0, 0]
points = ptCloud[0]  # The actual point data
headers = ptCloud[1] if len(ptCloud) > 1 else None

print(f"\nPoint cloud shape: {points.shape}")
print(f"Number of points: {points.shape[0]}")
print(f"Number of features: {points.shape[1]}")

print(f"\nFirst few points:")
for i in range(min(3, points.shape[0])):
    print(f"Point {i}: {points[i]}")

if headers is not None:
    print(f"\nHeaders shape: {headers.shape}")
    print("Column headers:")
    for i, header in enumerate(headers[0]):
        print(f"  {i}: {header[0] if len(header) > 0 else 'Unknown'}")

# Check if we have XYZ coordinates (first 3 columns typically)
if points.shape[1] >= 3:
    print(f"\nX range: {points[:, 0].min():.2f} to {points[:, 0].max():.2f}")
    print(f"Y range: {points[:, 1].min():.2f} to {points[:, 1].max():.2f}")
    print(f"Z range: {points[:, 2].min():.2f} to {points[:, 2].max():.2f}")