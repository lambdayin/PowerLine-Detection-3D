#!/usr/bin/env python3
"""
Convert .mat format point cloud files to .pcd format for PCL viewer support.

This script converts ExtractPowerLine .mat files containing point cloud data
to PCL-compatible .pcd format files.

Usage:
    python mat_to_pcd_converter.py input.mat [output.pcd]
    python mat_to_pcd_converter.py input_directory/ [output_directory/]
"""

import sys
import os
import argparse
import scipy.io
import numpy as np
from pathlib import Path


def write_pcd_header(f, num_points, has_intensity=False):
    """Write PCD file header."""
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


def convert_mat_to_pcd(mat_file_path, pcd_file_path, include_intensity=True):
    """
    Convert a single .mat file to .pcd format.
    
    Args:
        mat_file_path (str): Path to input .mat file
        pcd_file_path (str): Path to output .pcd file
        include_intensity (bool): Whether to include intensity data
    """
    try:
        # Load .mat file
        print(f"Loading {mat_file_path}...")
        data = scipy.io.loadmat(mat_file_path)
        
        # Extract point cloud data
        if 'ptCloudA' not in data:
            raise ValueError("Expected 'ptCloudA' key not found in .mat file")
        
        ptCloud = data['ptCloudA'][0, 0]
        points = ptCloud[0]  # The actual point data
        
        print(f"Found {points.shape[0]} points with {points.shape[1]} features")
        
        # Validate data structure
        if points.shape[1] < 3:
            raise ValueError("Point cloud must have at least X, Y, Z coordinates")
        
        # Extract coordinates
        x = points[:, 0]
        y = points[:, 1] 
        z = points[:, 2]
        
        # Extract intensity if available and requested
        intensity = None
        if include_intensity and points.shape[1] >= 10:
            intensity = points[:, 9]  # Intensity is the last column
        
        # Write PCD file
        print(f"Writing {pcd_file_path}...")
        with open(pcd_file_path, 'w') as f:
            write_pcd_header(f, points.shape[0], intensity is not None)
            
            # Write point data
            if intensity is not None:
                for i in range(points.shape[0]):
                    f.write(f"{x[i]:.6f} {y[i]:.6f} {z[i]:.6f} {intensity[i]:.6f}\n")
            else:
                for i in range(points.shape[0]):
                    f.write(f"{x[i]:.6f} {y[i]:.6f} {z[i]:.6f}\n")
        
        print(f"Successfully converted {mat_file_path} to {pcd_file_path}")
        
    except Exception as e:
        print(f"Error converting {mat_file_path}: {str(e)}")
        raise


def convert_directory(input_dir, output_dir, include_intensity=True):
    """Convert all .mat files in a directory."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .mat files
    mat_files = list(input_path.glob("*.mat"))
    
    if not mat_files:
        print(f"No .mat files found in {input_dir}")
        return
    
    print(f"Found {len(mat_files)} .mat files to convert")
    
    for mat_file in mat_files:
        pcd_file = output_path / (mat_file.stem + ".pcd")
        try:
            convert_mat_to_pcd(str(mat_file), str(pcd_file), include_intensity)
        except Exception as e:
            print(f"Skipping {mat_file.name}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert .mat point cloud files to .pcd format"
    )
    parser.add_argument(
        "input", 
        help="Input .mat file or directory containing .mat files"
    )
    parser.add_argument(
        "output", 
        nargs="?", 
        help="Output .pcd file or directory (optional)"
    )
    parser.add_argument(
        "--no-intensity", 
        action="store_true",
        help="Exclude intensity data from output"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    include_intensity = not args.no_intensity
    
    if not input_path.exists():
        print(f"Error: Input path {args.input} does not exist")
        return 1
    
    if input_path.is_file():
        # Single file conversion
        if not input_path.suffix.lower() == '.mat':
            print(f"Error: Input file must have .mat extension")
            return 1
        
        if args.output:
            output_file = args.output
        else:
            output_file = input_path.with_suffix('.pcd')
        
        try:
            convert_mat_to_pcd(str(input_path), output_file, include_intensity)
        except Exception as e:
            print(f"Conversion failed: {str(e)}")
            return 1
            
    elif input_path.is_dir():
        # Directory conversion
        if args.output:
            output_dir = args.output
        else:
            output_dir = input_path / "pcd_output"
        
        try:
            convert_directory(str(input_path), output_dir, include_intensity)
        except Exception as e:
            print(f"Directory conversion failed: {str(e)}")
            return 1
    else:
        print(f"Error: {args.input} is neither a file nor a directory")
        return 1
    
    print("Conversion completed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())