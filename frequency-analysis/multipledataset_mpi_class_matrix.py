#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Dataset MPI Analysis

This script processes cellular marker intensity data from multiple datasets 
(IgG4, AAV, NK2, etc.) and generates Z-scored mean marker intensity (MPI) matrices
for cell type analysis across datasets.

Uses existing cell classifications from input data.
Modified to use specific marker-location combinations.
Added membrane-nucleus class relationship heatmaps.
"""

import os
import argparse
import pandas as pd
import numpy as np

# Set matplotlib backend to Agg (non-interactive) before importing pyplot
import matplotlib
matplotlib.use('Agg')  # Use the Agg backend which doesn't require a display
import matplotlib.pyplot as plt
import matplotlib as mpl

def hex_to_rgb(hex):
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0],rgb[1],rgb[2])

def create_membrane_nucleus_heatmaps(df, output_dir, datasets):
    """
    Create heatmaps showing the relationship between membrane classes and nucleus classes
    Shows the percentage of each nucleus class within each membrane class
    """
    print("\nCreating membrane-nucleus class relationship heatmaps...")
    
    # Define the specific membrane classes to visualize (in specified order)
    target_membrane_classes = [
        'NaiveCD4T', 'CD4T', 'ICOS+Treg', 'CD103+CD4Treg', 'HLAII+CD27+Treg',
        'Tfh', 'TIGIT+Tfh', 'HLAII+GZMK+Tfh', 'GZMA+GZMK+CD4CTL', 'GZMB+CD4CTL','CD4Tcm-Tem',
        'NaiveCD8T', 'CD8T', 'GZMA+GZMK+CD8CTL', 'GZMB+CD8CTL','CD103+CD8Treg', 'CD8Trm',
        'CD103+CD8Trm', 'TIGIT+CD8T', 'CD8Tcm-Tcm', 'γδT','ILC', 'ImmatureNK', 'MatureNK', 'NKT', 
        'NaiveB', 'TransitionalB', 'B', 'CD27+B', 'DN2B', 'AgeAsscB'
    ]
    
    # Define the specific nucleus class order (as requested)
    ordered_nucleus_classes = [
        'GenericNucleus', 'Foxp3+Nucleus', 'Foxp3+TCF1+Nucleus', 'Foxp3+TCF1+TOX+Nucleus', 
        'Foxp3+TOX+Nucleus','Tbet+Nucleus','Tbet+TCF1+Nucleus', 'Tbet+TCF1+TOX+Nucleus', 
        'Tbet+TOX+Nucleus', 'Ki67+Nucleus', 'TCF1+Nucleus', 'Ki67+TCF1+Nucleus', 
        'TCF1+TOX+Nucleus', 'Ki67+TCF1+TOX+Nucleus','TOX+Nucleus', 'ApoptoticNucleus','NucBackground'
    ]
    
    # Filter to only include nucleus classes that exist in the data (maintaining order)
    all_nucleus_classes_in_data = df['NucCatName'].unique()
    all_nucleus_classes = [cls for cls in ordered_nucleus_classes if cls in all_nucleus_classes_in_data]
    
    # Add any nucleus classes that exist in data but weren't in our predefined order
    missing_classes = [cls for cls in all_nucleus_classes_in_data if cls not in all_nucleus_classes]
    if missing_classes:
        print(f"Found additional nucleus classes not in predefined order: {missing_classes}")
        all_nucleus_classes.extend(sorted(missing_classes))  # Add them at the end, sorted
    
    # Filter membrane classes to only those present in the data (maintaining order)
    present_membrane_classes = [cls for cls in target_membrane_classes if cls in df['MemCatName'].unique()]
    
    print(f"Found {len(present_membrane_classes)} target membrane classes in data:")
    for cls in present_membrane_classes:
        count = len(df[df['MemCatName'] == cls])
        print(f"  {cls}: {count} cells")
    
    print(f"\nNucleus classes found: {', '.join(all_nucleus_classes)}")
    
    def calculate_membrane_nucleus_percentages(data_subset, dataset_name=""):
        """Calculate percentage matrix for membrane-nucleus relationships"""
        percentage_matrix = np.zeros((len(present_membrane_classes), len(all_nucleus_classes)))
        
        for i, mem_class in enumerate(present_membrane_classes):
            mem_cells = data_subset[data_subset['MemCatName'] == mem_class]
            if not mem_cells.empty:
                total_cells = len(mem_cells)
                for j, nuc_class in enumerate(all_nucleus_classes):
                    count = len(mem_cells[mem_cells['NucCatName'] == nuc_class])
                    percentage_matrix[i, j] = (count / total_cells) * 100
                    
                # Print some stats for GenericNucleus to understand the distribution
                if dataset_name and 'GenericNucleus' in all_nucleus_classes:
                    generic_idx = all_nucleus_classes.index('GenericNucleus')
                    generic_pct = percentage_matrix[i, generic_idx]
                    if generic_pct > 50:  # Only report high GenericNucleus percentages
                        print(f"  {dataset_name} - {mem_class}: {generic_pct:.1f}% GenericNucleus")
        
        return percentage_matrix
    
    def create_heatmap(matrix, title, filename, show_values=False):
        """Create a heatmap with consistent formatting"""
        plt.figure(figsize=(16, 12), dpi=600)  # Increased size for better annotation visibility
        
        # Cap values at 30% for better visualization of smaller percentages
        matrix_capped = np.clip(matrix, 0, 30)
        
        # Use a colormap that works well for percentages
        im = plt.imshow(matrix_capped, vmin=0, vmax=30, cmap='viridis', aspect='auto')
        
        # Add grid lines
        ax = plt.gca()
        ax.set_xticks(np.arange(-0.5, len(all_nucleus_classes), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(present_membrane_classes), 1), minor=True)
        ax.grid(which='minor', color='white', linestyle='-', linewidth=0.5)
        
        # Set ticks and labels
        plt.yticks(ticks=np.arange(0, len(present_membrane_classes)), 
                  labels=present_membrane_classes, fontsize=12)
        plt.xticks(ticks=np.arange(0, len(all_nucleus_classes)), 
                  labels=all_nucleus_classes, rotation=45, ha='right', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(im, shrink=0.8)
        cbar.set_label('Percentage (%)', fontsize=10)
        # Add note about values ≥30%
        cbar.ax.text(0.5, 1.02, '≥30%', transform=cbar.ax.transAxes, 
                    ha='center', fontsize=8, weight='bold')
        
        # Add percentage values as text annotations
        if show_values:
            for i in range(len(present_membrane_classes)):
                for j in range(len(all_nucleus_classes)):
                    value = matrix[i, j]
                    
                    # Show numbers with % for all values >= 1%
                    if value >= 1:
                        # Swapped colors: white text on dark backgrounds, black text on light backgrounds
                        text_color = 'black' if matrix_capped[i, j] > 15 else 'white'
                        plt.text(j, i, f'{value:.0f}%', ha='center', va='center', 
                                fontsize=12, color=text_color, weight='bold')
        
        plt.xlabel('Nucleus Class', fontsize=10)
        plt.ylabel('Membrane Class', fontsize=10)
        plt.title(title, fontsize=12, pad=20)
        
        # Add annotation legend as subtitle
        if show_values:
            plt.figtext(0.5, 0.02, 'Annotations: Numbers show percentages ≥1% (black text on light colors, white text on dark colors)', 
                       ha='center', fontsize=9, style='italic', color='gray')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=600, bbox_inches='tight')
        plt.close()
    
    # Create heatmap for all datasets combined
    print("Processing combined dataset...")
    combined_matrix = calculate_membrane_nucleus_percentages(df, "Combined")
    create_heatmap(combined_matrix, 
                  'Membrane-Nucleus Class Relationships - All Datasets Combined',
                  'Combined_Membrane_Nucleus_Relationships.png',
                  show_values=True)
    
    # Save the combined data as CSV
    combined_df = pd.DataFrame(combined_matrix, 
                              index=present_membrane_classes, 
                              columns=all_nucleus_classes)
    combined_df.to_csv(os.path.join(output_dir, 'Combined_Membrane_Nucleus_Percentages.csv'))
    
    # Create heatmaps for individual datasets
    for dataset_name in datasets:
        print(f"Processing {dataset_name} dataset...")
        dataset_df = df[df['Dataset'] == dataset_name]
        
        if dataset_df.empty:
            print(f"No data found for dataset {dataset_name}, skipping...")
            continue
        
        # Check if any of our target membrane classes exist in this dataset
        dataset_mem_classes = dataset_df['MemCatName'].unique()
        target_in_dataset = [cls for cls in target_membrane_classes if cls in dataset_mem_classes]
        
        if not target_in_dataset:
            print(f"No target membrane classes found in {dataset_name}, skipping...")
            continue
            
        dataset_matrix = calculate_membrane_nucleus_percentages(dataset_df, dataset_name)
        create_heatmap(dataset_matrix,
                      f'Membrane-Nucleus Class Relationships - {dataset_name} Dataset',
                      f'{dataset_name}_Membrane_Nucleus_Relationships.png',
                      show_values=True)
        
        # Save individual dataset data as CSV
        dataset_df_save = pd.DataFrame(dataset_matrix, 
                                      index=present_membrane_classes, 
                                      columns=all_nucleus_classes)
        dataset_df_save.to_csv(os.path.join(output_dir, f'{dataset_name}_Membrane_Nucleus_Percentages.csv'))
    
    # Create a summary report
    print("\nCreating summary report...")
    with open(os.path.join(output_dir, 'Membrane_Nucleus_Summary.txt'), 'w') as f:
        f.write("Membrane-Nucleus Class Relationship Analysis Summary\n")
        f.write("=" * 55 + "\n\n")
        
        f.write(f"Total datasets analyzed: {len(datasets)}\n")
        f.write(f"Datasets: {', '.join(datasets)}\n")
        f.write(f"Target membrane classes: {len(target_membrane_classes)}\n")
        f.write(f"Present membrane classes: {len(present_membrane_classes)}\n")
        f.write(f"Nucleus classes found: {len(all_nucleus_classes)}\n\n")
        
        f.write("Membrane classes analyzed:\n")
        for i, cls in enumerate(present_membrane_classes, 1):
            count = len(df[df['MemCatName'] == cls])
            f.write(f"  {i:2d}. {cls}: {count} cells\n")
        
        f.write(f"\nNucleus classes: {', '.join(all_nucleus_classes)}\n\n")
        
        # Find classes with high GenericNucleus percentage
        if 'GenericNucleus' in all_nucleus_classes:
            generic_idx = all_nucleus_classes.index('GenericNucleus')
            f.write("Classes with >70% GenericNucleus (combined data):\n")
            for i, mem_class in enumerate(present_membrane_classes):
                pct = combined_matrix[i, generic_idx]
                if pct > 70:
                    f.write(f"  {mem_class}: {pct:.1f}%\n")
        
        f.write("\nNote: Heatmap color scale is capped at 30% to better visualize\n")
        f.write("smaller percentages. Values ≥30% are shown in the darkest color.\n\n")
        f.write("Annotation system:\n")
        f.write("  - Numbers with %: All values ≥1%\n")
        f.write("  - Text color: Black on light backgrounds, white on dark backgrounds\n")
        f.write("  - No annotation: Values <1%\n")
    
    print("Membrane-nucleus relationship analysis completed!")
    return combined_matrix

def process_multi_dataset_mpi(df, output_dir):
    """
    Process MPI data for multiple datasets and create combined Z-score plots
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame with 'Dataset' column to distinguish between datasets
        and existing cell classifications in 'MemCatName' and 'NucCatName'
    output_dir : str
        Directory to save output files
    
    Returns:
    --------
    df : pandas DataFrame
        DataFrame with added Z-score columns
    """
    print("\nProcessing MPI data for multiple datasets...")
    
    # Define the specific marker-location combinations to use
    requested_markers = [
        "TCRd_Int-mean_Mem", "CD86_Int-mean_WC", "Ki67_Int-mean_Nuc", "CD45RA_Int-mean_Mem",
        "MERTK_Int-mean_WC", "CD127_Int-mean_Mem", "IgG4_Int-mean_WC", "Foxp3_Int-mean_Nuc", 
        "cCasp3_Int-mean_WC", "BDCA1_Int-mean_WC", "CD90_Int-mean_WC", "SLAMF7_Int-mean_WC", 
        "CD34_Int-mean_WC", "CD56_Int-mean_Mem", "Claudin1_Int-mean_WC", "BDCA2_Int-mean_WC", 
        "PDPN_Int-mean_WC", "TCF1_Int-mean_Nuc", "CD163_Int-mean_WC", "ICOS_Int-mean_Mem", 
        "CD11c_Int-mean_WC", "MUC1_Int-mean_WC", "CD141_Int-mean_WC", "GZMB_Int-mean_WC",
        "CD103_Int-mean_Mem", "CD138_Int-mean_WC", "CD68_Int-mean_WC", "PD-1_Int-mean_Mem", 
        "CD20_Int-mean_Mem", "GZMK_Int-mean_WC", "CD69_Int-mean_WC", 
        "CLEC9A_Int-mean_WC", "GPNMB_Int-mean_WC", "CD27_Int-mean_Mem", 
        "CD8_Int-mean_Mem", "HIF1a_Int-mean_WC", "CD3_Int-mean_Mem", "iNOS_Int-mean_WC", 
        "TIGIT_Int-mean_Mem", "CD11b_Int-mean_WC", "CD4_Int-mean_Mem",
        "GZMA_Int-mean_WC", "CD10_Int-mean_WC", "MPO_Int-mean_WC", "HLAII_Int-mean_WC", 
        "TOX_Int-mean_Nuc", "aSMA_Int-mean_WC", "T-bet_Int-mean_Nuc", "CD16_Int-mean_WC", "DAPI_Int-mean_Nuc", "MXA_Int-mean_WC", 
    ]
    
    # Filter for columns that exist in the dataframe
    mpi_cols = []
    missing_markers = []
    
    for marker in requested_markers:
        if marker in df.columns:
            mpi_cols.append(marker)
        else:
            missing_markers.append(marker)
    
    if missing_markers:
        print("Warning: The following requested markers are missing from the data:")
        for marker in missing_markers:
            print(f"  - {marker}")
    
    # Sort the columns for consistent visualization
    mpi_cols.sort()
    
    # Print the list of markers being used
    print("Markers being analyzed:")
    for i, col in enumerate(mpi_cols):
        marker_name = col.split('_Int-mean_')[0]
        location = col.split('_Int-mean_')[1]
        print(f"  {i+1}. {marker_name} ({location})")
    
    if not mpi_cols:
        print("Error: No requested markers found in the data. Cannot proceed.")
        return df
    
    # Extract unique datasets
    datasets = df['Dataset'].unique()
    print(f"Found datasets: {', '.join(datasets)}")
    
    # Use existing cell classifications - verify they exist
    if 'MemCatName' not in df.columns or 'NucCatName' not in df.columns:
        print("Error: Expected cell classifications 'MemCatName' and 'NucCatName' not found in data.")
        return df
    
    # Check if NaiveCD4T cells exist in the data
    if 'NaiveCD4T' in df['MemCatName'].unique():
        naive_cd4t_cells = df[df['MemCatName'] == 'NaiveCD4T']
        print(f"\nFound {len(naive_cd4t_cells)} NaiveCD4T cells in the data")
        
        # Check key markers for NaiveCD4T cells
        key_markers = ['CD3', 'CD4', 'CD45RA', 'CD127']
        for marker in key_markers:
            # Look for this marker in any location
            marker_cols = [col for col in mpi_cols if marker in col.split('_Int-mean_')[0]]
            if marker_cols:
                mean_val = naive_cd4t_cells[marker_cols[0]].mean()
                print(f"  {marker} ({marker_cols[0].split('_Int-mean_')[1]}): Mean value = {mean_val:.4f}")
            else:
                print(f"  {marker}: Not found in available markers")
    
    # Check if specific cell types exist in the data
    print("\nChecking for specific cell types of interest:")
    for cell_type in ['TIGIT+Tfh', 'TIGIT+CD8T']:
        count = df[df['MemCatName'] == cell_type].shape[0]
        print(f"  {cell_type} cells count: {count}")
        # If cells exist, show some marker info
        if count > 0:
            cells = df[df['MemCatName'] == cell_type]
            # Show TIGIT marker if available
            tigit_cols = [col for col in mpi_cols if 'TIGIT' in col.split('_Int-mean_')[0]]
            if tigit_cols:
                mean_val = cells[tigit_cols[0]].mean()
                print(f"    TIGIT ({tigit_cols[0].split('_Int-mean_')[1]}) mean value: {mean_val:.4f}")
    
    for nuc_type in ['TOX+Nucleus', 'TCF1+TOX+Nucleus']:
        count = df[df['NucCatName'] == nuc_type].shape[0]
        print(f"  {nuc_type} count: {count}")
        # If cells exist, show some marker info
        if count > 0:
            cells = df[df['NucCatName'] == nuc_type]
            # Show TOX marker if available
            tox_cols = [col for col in mpi_cols if 'TOX' in col.split('_Int-mean_')[0]]
            if tox_cols:
                mean_val = cells[tox_cols[0]].mean()
                print(f"    TOX ({tox_cols[0].split('_Int-mean_')[1]}) mean value: {mean_val:.4f}")
    
    # Define ordered categories (only used for visualization ordering)
    ordered_memcats = [
        # T cells
        'NaiveCD4T', 'CD4T', 'ICOS+Treg', 'CD103+CD4Treg', 'HLAII+CD27+Treg',
        'Tfh', 'TIGIT+Tfh', 'HLAII+GZMK+Tfh', 'GZMA+GZMK+CD4CTL', 'GZMB+CD4CTL','CD4Tcm-Tem',
        'NaiveCD8T', 'CD8T', 'GZMA+GZMK+CD8CTL', 'GZMB+CD8CTL','CD103+CD8Treg', 'CD8Trm',
        'CD103+CD8Trm', 'TIGIT+CD8T', 'CD8Tcm-Tcm', 'γδT','ILC', 'ImmatureNK', 'MatureNK', 'NKT', 
        # Humoral cells
        'NaiveB', 'TransitionalB', 'B', 'CD27+B', 'DN2B', 'AgeAsscB',
        'Plasmablast', 'IgG4+Plasmablast', 'Plasma', 'IgG4+Plasma',
        # Myeloid cells
        'M1Macrophage', 'MPO+Macrophage',
        'M2Macrophage', 'MerTK+M1Macrophage', 'MerTK+M2Macrophage',
        'GPNMB+M1Macrophage', 'GPNMB+M2Macrophage', 'cDC1', 'cDC2', 'pDC',
        'Neutrophil',
        # Stromal and structural cells
        'Myofibroblast-SmoothMuscle', 'CD90+Myofibroblast', 'MMT',
        'ProximalTubular', 'DistalTubular', 'StressedTubular',
        'Endothelial', 'LymphaticEndothelial', 'Podocyte', 'PEC',
        # Other cells
        'Apoptotic','RBC','TubuleBackground','Unclassified'
    ]
    
    ordered_nuccats = [
        'GenericNucleus', 'Foxp3+Nucleus', 'Foxp3+TCF1+Nucleus', 'Foxp3+TCF1+TOX+Nucleus', 'Foxp3+TOX+Nucleus','Tbet+Nucleus','Tbet+TCF1+Nucleus', 'Tbet+TCF1+TOX+Nucleus', 'Tbet+TOX+Nucleus', 'Ki67+Nucleus', 'TCF1+Nucleus', 'Ki67+TCF1+Nucleus', 'TCF1+TOX+Nucleus','Ki67+TCF1+TOX+Nucleus', 'TOX+Nucleus', 'ApoptoticNucleus','NucBackground'
    ]
    
    # Get actual categories from data
    all_memcats = df['MemCatName'].unique()
    all_nuccats = df['NucCatName'].unique()
    
    # Create new filtered lists that preserve the original order
    # but only include categories that are in the data
    filtered_memcats = []
    for cat in ordered_memcats:
        if cat in all_memcats:
            filtered_memcats.append(cat)
    
    filtered_nuccats = []
    for cat in ordered_nuccats:
        if cat in all_nuccats:
            filtered_nuccats.append(cat)
    
    # Use the filtered lists that maintain original order
    ordered_memcats = filtered_memcats
    ordered_nuccats = filtered_nuccats
    
    cats = ordered_memcats + ordered_nuccats
    
    # Print cell type counts
    print("\nCell type counts in data:")
    mem_counts = df['MemCatName'].value_counts()
    for cell_type, count in mem_counts.items():
        print(f"  {cell_type}: {count} cells")
    
    # Replace NaN values with zeros before Z-score calculation
    print("\nReplacing NaN values with zeros for marker columns...")
    df_markers = df[mpi_cols].fillna(0)
    
    # Pre-calculate Z-scores across the entire combined dataset
    print("Calculating Z-scores...")
    z_scores = pd.DataFrame(index=df.index)
    for col in mpi_cols:
        mean_val = df_markers[col].mean()
        std_val = df_markers[col].std()
        if std_val == 0:  # Handle case where std is zero to avoid division by zero
            z_scores[col+'_Z'] = 0
        else:
            z_scores[col+'_Z'] = (df_markers[col] - mean_val) / std_val
    
    # Add Z-scores to the original dataframe
    df = pd.concat([df, z_scores], axis=1)
    
    # Prepare arrays for storing results
    mu_array_all = np.zeros([len(cats), len(mpi_cols)])
    z_array_all = np.zeros([len(cats), len(mpi_cols)])
    
    # Store arrays per dataset for comparison
    mu_arrays_per_dataset = {name: np.zeros([len(cats), len(mpi_cols)]) for name in datasets}
    z_arrays_per_dataset = {name: np.zeros([len(cats), len(mpi_cols)]) for name in datasets}
    
    print(f"Processing combined data with {len(mpi_cols)} markers and {len(cats)} cell categories")
    
    # Process combined dataset for all cells - membrane categories
    for i, cat in enumerate(ordered_memcats):
        cat_data = df[df['MemCatName']==cat]
        if not cat_data.empty:
            for j, col in enumerate(mpi_cols):
                # Calculate without dropping NaN values (they are now zeros)
                mu_array_all[i,j] = cat_data[col].mean()
                z_array_all[i,j] = cat_data[col+'_Z'].mean()
        else:
            # If category has no cells, explicitly set to zeros
            print(f"Setting zeros for empty category: {cat}")
            mu_array_all[i,:] = 0
            z_array_all[i,:] = 0
    
    # Process combined dataset for all cells - nuclear categories
    for ii, cat in enumerate(ordered_nuccats):
        idx = ii + len(ordered_memcats)
        cat_data = df[df['NucCatName']==cat]
        if not cat_data.empty:
            for j, col in enumerate(mpi_cols):
                # Calculate without dropping NaN values (they are now zeros)
                mu_array_all[idx,j] = cat_data[col].mean()
                z_array_all[idx,j] = cat_data[col+'_Z'].mean()
        else:
            # If category has no cells, explicitly set to zeros
            print(f"Setting zeros for empty category: {cat}")
            mu_array_all[idx,:] = 0
            z_array_all[idx,:] = 0
    
    # Process each dataset separately
    for dataset_name in datasets:
        print(f"Processing dataset: {dataset_name}")
        dataset_df = df[df['Dataset']==dataset_name]
        
        # Get actual categories from this dataset
        dataset_memcats = dataset_df['MemCatName'].unique()
        dataset_nuccats = dataset_df['NucCatName'].unique()
        
        # Process membrane categories
        for i, cat in enumerate(ordered_memcats):
            if cat in dataset_memcats:
                cat_data = dataset_df[dataset_df['MemCatName']==cat]
                if not cat_data.empty:
                    for j, col in enumerate(mpi_cols):
                        # Calculate without dropping NaN values (they are now zeros)
                        mu_arrays_per_dataset[dataset_name][i,j] = cat_data[col].mean()
                        z_arrays_per_dataset[dataset_name][i,j] = cat_data[col+'_Z'].mean()
        
        # Process nuclear categories
        for ii, cat in enumerate(ordered_nuccats):
            idx = ii + len(ordered_memcats)
            if cat in dataset_nuccats:
                cat_data = dataset_df[dataset_df['NucCatName']==cat]
                if not cat_data.empty:
                    for j, col in enumerate(mpi_cols):
                        # Calculate without dropping NaN values (they are now zeros)
                        mu_arrays_per_dataset[dataset_name][idx,j] = cat_data[col].mean()
                        z_arrays_per_dataset[dataset_name][idx,j] = cat_data[col+'_Z'].mean()
    
    # Process special cell types that might have no cells assigned
    # Ensure arrays are properly filled even for empty categories
    epsilon = 1e-6
    monitored_cell_types = ['TIGIT+Tfh', 'TIGIT+CD8T', 'TOX+Nucleus', 'TCF1+TOX+Nucleus']
    
    for cat in monitored_cell_types:
        if cat in ordered_memcats:
            cat_idx = ordered_memcats.index(cat)
            # Check if this category has any cells
            cat_data = df[df['MemCatName']==cat]
            
            if cat_data.empty:
                print(f"Note: No cells found for {cat}, but keeping in visualization with zero values")
                # Fill with zeros to ensure it appears in visualization
                mu_array_all[cat_idx,:] = 0
                z_array_all[cat_idx,:] = 0
                
                # Also update per-dataset arrays
                for dataset_name in datasets:
                    mu_arrays_per_dataset[dataset_name][cat_idx,:] = 0
                    z_arrays_per_dataset[dataset_name][cat_idx,:] = 0
            elif np.all(np.abs(z_array_all[cat_idx,:]) < epsilon):
                print(f"Note: {cat} has very small or zero values, but keeping actual values as requested")
                # Count cells of this type
                print(f"  Found {len(cat_data)} cells of type {cat}")
                
        elif cat in ordered_nuccats:
            cat_idx = len(ordered_memcats) + ordered_nuccats.index(cat)
            # Check if this category has any cells
            cat_data = df[df['NucCatName']==cat]
            
            if cat_data.empty:
                print(f"Note: No cells found for {cat}, but keeping in visualization with zero values")
                # Fill with zeros to ensure it appears in visualization
                mu_array_all[cat_idx,:] = 0
                z_array_all[cat_idx,:] = 0
                
                # Also update per-dataset arrays
                for dataset_name in datasets:
                    mu_arrays_per_dataset[dataset_name][cat_idx,:] = 0
                    z_arrays_per_dataset[dataset_name][cat_idx,:] = 0
            elif np.all(np.abs(z_array_all[cat_idx,:]) < epsilon):
                print(f"Note: {cat} has very small or zero values, but keeping actual values as requested")
                # Count cells of this type
                print(f"  Found {len(cat_data)} cells of type {cat}")
    
    # Create plots - visualize both raw (averaged) values and Z-scores
    print("Generating visualization plots...")
    
    # Create clean marker names for plotting
    markers = [x.split('_Int-mean_')[0] for x in mpi_cols]
    # Create a list of marker locations for tooltips or labels if needed
    marker_locations = [x.split('_Int-mean_')[1] for x in mpi_cols]
    
    # Plot mean MPI for combined data
    plt.figure(figsize=(8,8), dpi=600)
    plt.imshow(mu_array_all, cmap='jet')
    
    # Add thin black borders around each cell
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, len(mpi_cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cats), 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    
    plt.yticks(ticks=np.arange(0,len(cats)), labels=cats, fontsize=5)
    plt.xticks(ticks=np.arange(0,len(mpi_cols)), labels=markers, rotation=90, fontsize=5)
    plt.colorbar()
    plt.xlabel('Marker', fontsize=8)
    plt.ylabel('Predicted Cell Class', fontsize=8)
    plt.title('Mean MPI - All Datasets Combined', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Combined_MeanMPI-perClass.png'), dpi=600)
    plt.close()

    # Update markers for special characters
    markers = [x.replace('gamma','\u03B3') for x in markers]  
    markers = [x.replace('delta','\u03B4') for x in markers]
    
    # Plot Z-scores for combined data
    mpl.rcParams['font.size'] = 8 
    plt.figure(figsize=(10,8), dpi=600)
    plt.imshow(z_array_all, vmin=-3, vmax=3, cmap='seismic')
    
    # Add thin black borders around each cell
    ax = plt.gca()
    ax.set_xticks(np.arange(-0.5, len(mpi_cols), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(cats), 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
    
    plt.yticks(ticks=np.arange(0,len(cats)), labels=cats, fontsize=6)
    plt.xticks(ticks=np.arange(0,len(mpi_cols)), labels=markers, fontsize=6, rotation=90)
    plt.colorbar()
    plt.xlabel('Marker', fontsize=8)
    plt.ylabel('Predicted Cell or Nucleus Class', fontsize=8)
    plt.title('Z-Scored Mean MPI - All Datasets Combined', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'Combined_MeanMPIzscore-perClass.png'), dpi=600)
    plt.close()
    
    # Create Z-score plots for each dataset
    for dataset_name in datasets:
        plt.figure(figsize=(10,8), dpi=600)
        plt.imshow(z_arrays_per_dataset[dataset_name], vmin=-3, vmax=3, cmap='seismic')
        
        # Add thin black borders around each cell
        ax = plt.gca()
        ax.set_xticks(np.arange(-0.5, len(mpi_cols), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(cats), 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        
        plt.yticks(ticks=np.arange(0,len(cats)), labels=cats, fontsize=6)
        plt.xticks(ticks=np.arange(0,len(mpi_cols)), labels=markers, fontsize=6, rotation=90)
        plt.colorbar()
        plt.xlabel('Marker', fontsize=8)
        plt.ylabel('Predicted Cell or Nucleus Class', fontsize=8)
        plt.title(f'Z-Scored Mean MPI - {dataset_name} Dataset', fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{dataset_name}_MeanMPIzscore-perClass.png'), dpi=600)
        plt.close()
    
    # Create difference plots between datasets if more than one dataset
    if len(datasets) > 1:
        for i, name_i in enumerate(datasets):
            for j, name_j in enumerate(datasets):
                if i < j:  # Avoid duplicates
                    diff_array = z_arrays_per_dataset[name_i] - z_arrays_per_dataset[name_j]
                    
                    plt.figure(figsize=(10,8), dpi=600)
                    plt.imshow(diff_array, vmin=-3, vmax=3, cmap='seismic')
                    plt.yticks(ticks=np.arange(0,len(cats)), labels=cats, fontsize=6)
                    plt.xticks(ticks=np.arange(0,len(mpi_cols)), labels=markers, fontsize=6, rotation=90)
                    plt.colorbar()
                    plt.xlabel('Marker', fontsize=8)
                    plt.ylabel('Predicted Cell or Nucleus Class', fontsize=8)
                    plt.title(f'Z-Score Difference: {name_i} - {name_j}', fontsize=10)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'Diff_{name_i}_vs_{name_j}_MeanMPIzscore.png'), dpi=600)
                    plt.close()
    
    # Save data arrays to CSV for further analysis
    print("Saving Z-score data to CSV files...")
    # Also save the marker locations in the output
    combined_markers = [f"{markers[i]} ({marker_locations[i]})" for i in range(len(markers))]
    z_df = pd.DataFrame(z_array_all, index=cats, columns=combined_markers)
    z_df.to_csv(os.path.join(output_dir, 'Combined_ZScores.csv'))
    
    for dataset_name in datasets:
        z_df = pd.DataFrame(z_arrays_per_dataset[dataset_name], index=cats, columns=combined_markers)
        z_df.to_csv(os.path.join(output_dir, f'{dataset_name}_ZScores.csv'))
    
    # NEW: Create membrane-nucleus relationship heatmaps
    create_membrane_nucleus_heatmaps(df, output_dir, datasets)
    
    return df

def main():
    # Hard-coded input and output directories
    DEFAULT_INPUT_FILE = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/combined_SAM_scores_filtered_alldataset5_10woMonointerstitial.csv"
    DEFAULT_OUTPUT_DIR = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/cellclass_mpi_matrix_alldataset5_10woMonointerstitial"
    
    # Parse command-line arguments if provided (these will override the defaults)
    parser = argparse.ArgumentParser(description='Process multi-dataset MPI data and generate Z-score matrices')
    parser.add_argument('--input', type=str, default=DEFAULT_INPUT_FILE, 
                        help=f'Input CSV file or directory (default: {DEFAULT_INPUT_FILE})')
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR, 
                        help=f'Directory to save output files (default: {DEFAULT_OUTPUT_DIR})')
    
    args = parser.parse_args()
    
    input_path = args.input
    output_dir = args.output_dir
    
    print(f"Using input path: {input_path}")
    print(f"Using output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Debug file path issue
    print(f"\n=== Debugging File Path ===")
    print(f"Input path: {input_path}")
    print(f"Absolute path: {os.path.abspath(input_path)}")
    print(f"os.path.exists(): {os.path.exists(input_path)}")
    print(f"os.path.isfile(): {os.path.isfile(input_path)}")
    print(f"os.path.isdir(): {os.path.isdir(input_path)}")
    
    # Try to list the parent directory
    parent_dir = os.path.dirname(input_path)
    print(f"\nParent directory: {parent_dir}")
    print(f"Parent exists: {os.path.exists(parent_dir)}")
    
    if os.path.exists(parent_dir):
        try:
            print(f"\nListing files in parent directory:")
            files = os.listdir(parent_dir)
            for file in files:
                if "filterd" in file:  # Look for files with similar names
                    print(f"  {file}")
        except Exception as e:
            print(f"Error listing directory: {str(e)}")
    
    # Try to directly read the file regardless of path checks
    print(f"\n=== Attempting Direct File Read ===")
    try:
        print(f"Trying to read file directly: {input_path}")
        df = pd.read_csv(input_path, low_memory=False)
        print(f"Successfully read CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # Convert 'Sample' column to string if it exists
        if 'Sample' in df.columns:
            df['Sample'] = df['Sample'].astype(str)
        
        # Check if 'Dataset' column exists
        if 'Dataset' not in df.columns:
            print("Warning: 'Dataset' column not found in the CSV. Attempting to infer from filename...")
            
            # Extract dataset name from filename
            filename = os.path.basename(input_path).lower()
            if 'igg4' in filename:
                df['Dataset'] = 'IgG4'
            elif 'aav' in filename:
                df['Dataset'] = 'AAV'
            elif 'nk2' in filename:
                df['Dataset'] = 'NK2'
            elif 'combined' in filename or 'all' in filename:
                # If it's a combined file, let's make a combined dataset
                print("Using 'Combined' as dataset name for combined file")
                df['Dataset'] = 'Combined'
            else:
                print("Could not infer dataset name. Using 'Unknown'")
                df['Dataset'] = 'Unknown'
                
            print(f"Set Dataset column to: {df['Dataset'].iloc[0]}")
        
        # Skip cell filtering - use existing classifications from the data
        print("Using existing cell classifications from data...")
        print(f"Found classifications for {df['MemCatName'].nunique()} membrane cell types and {df['NucCatName'].nunique()} nuclear types")
        
        # Process MPI data for multiple datasets
        df = process_multi_dataset_mpi(df, output_dir)
        
        # Save processed data
        output_file = os.path.join(output_dir, 'processed_data.csv')
        df.to_csv(output_file, index=False)
        print(f"Saved processed data to {output_file}")
        
        print("\nProcessing complete. Generated Z-scored meanMPI matrices and membrane-nucleus relationship heatmaps for all datasets.")
        return
        
    except Exception as e:
        print(f"Error reading file directly: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\nFailed to process the data. Please check the file path and permissions.")

if __name__ == "__main__":
    main()
