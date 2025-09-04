#!/usr/bin/env python3
"""
Convert .mat format point cloud files to .txt format.

This script converts ExtractPowerLine .mat files containing point cloud data
to simple .txt files (X Y Z [intensity]).

Usage:
    python mat_to_txt_converter.py input.mat [output.txt]
    python mat_to_txt_converter.py input_directory/ [output_directory/]
"""

import sys
import os
import argparse
import scipy.io
import numpy as np
from pathlib import Path


def convert_mat_to_txt(mat_file_path, txt_file_path, include_intensity=True):
    """
    Convert a single .mat file to .txt format.
    
    Args:
        mat_file_path (str): Path to input .mat file
        txt_file_path (str): Path to output .txt file
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
        
        # Write TXT file
        print(f"Writing {txt_file_path}...")
        with open(txt_file_path, 'w') as f:
            # Write point data
            if intensity is not None:
                # Using numpy for faster writing
                data_to_write = np.vstack((x, y, z, intensity)).T
                np.savetxt(f, data_to_write, fmt='%.6f')
            else:
                data_to_write = np.vstack((x, y, z)).T
                np.savetxt(f, data_to_write, fmt='%.6f')
        
        print(f"Successfully converted {mat_file_path} to {txt_file_path}")
        
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
        txt_file = output_path / (mat_file.stem + ".txt")
        try:
            convert_mat_to_txt(str(mat_file), str(txt_file), include_intensity)
        except Exception as e:
            print(f"Skipping {mat_file.name}: {str(e)}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert .mat point cloud files to .txt format"
    )
    parser.add_argument(
        "input", 
        help="Input .mat file or directory containing .mat files"
    )
    parser.add_argument(
        "output", 
        nargs="?", 
        help="Output .txt file or directory (optional)"
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
            output_file = input_path.with_suffix('.txt')
        
        try:
            convert_mat_to_txt(str(input_path), output_file, include_intensity)
        except Exception as e:
            print(f"Conversion failed: {str(e)}")
            return 1
            
    elif input_path.is_dir():
        # Directory conversion
        if args.output:
            output_dir = args.output
        else:
            output_dir = input_path / "txt_output"
        
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
