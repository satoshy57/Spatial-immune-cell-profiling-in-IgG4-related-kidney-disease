#!/usr/bin/env python3
"""
Running script for cell annotation analysis with TAG-based deduplication
and cell type filtering options.
"""

import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Import the cell_annotation_analysis module
# Assuming the revised script is saved as cell_annotation_analysis.py
from cell_annotation_analysis import main

# Define file paths
ground_truth_path = '/project/mclark/SCAMPI_datasets/IgG4/classify_by_dilation-sameTH/with-MPI/groundtruth_converted051225.csv'

# Define multiple annotation files with different thresholds
annotation_paths = [
    '/project/mclark/SCAMPI_datasets/IgG4/classify_by_dilation-sameTH/with-MPI/combined_SAM_scores_filtered0_0woMono_memcatname2.csv',
    '/project/mclark/SCAMPI_datasets/IgG4/classify_by_dilation-sameTH/with-MPI/combined_SAM_scores_filtered5_10woMono_memcatname2.csv',
    '/project/mclark/SCAMPI_datasets/IgG4/classify_by_dilation-sameTH/with-MPI/combined_SAM_scores_filtered7_10woMono_memcatname2.csv',
    '/project/mclark/SCAMPI_datasets/IgG4/classify_by_dilation-sameTH/with-MPI/combined_SAM_scores_filtered10_10woMono_memcatname2.csv'
]

# Define names for each annotation setting
names = ['memth 0', 'memth 5', 'memth 7', 'memth 10']

# Define output directory
output_dir = '/project/mclark/SCAMPI_datasets/IgG4/classify_by_dilation-sameTH/with-MPI/annotation_analysis051225woMono'

# Define which cell types to include in the analysis
# Uncomment and modify the line below to filter specific cell types
include_cell_types = ['CD4T', 'CD8T', 'γδT', 'NK', 'B','Plasma', 'Macrophages', 'cDC2', 'pDC','Neutrophils', 'SMC/MyoF', 'Endothelial', 'Tubules']
# Or set to None to include all cell types
#include_cell_types = None

# Run the main function
if __name__ == "__main__":
    print(f"Starting cell annotation analysis...")
    print(f"Ground truth file: {ground_truth_path}")
    print(f"Annotation files: {len(annotation_paths)}")
    print(f"Output directory: {output_dir}")
    
    if include_cell_types:
        print(f"Filtering to include only these cell types: {include_cell_types}")
    else:
        print("Including all cell types in analysis")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Run the main analysis
    main(ground_truth_path, annotation_paths, names, output_dir, include_cell_types)
    
    print("Analysis complete!")

