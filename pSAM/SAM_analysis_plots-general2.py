import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def hex_to_rgb(hex):
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0],rgb[1],rgb[2])

def filter_cells(df, memth=0.0, nucth=0.0):
    # Define membrane and nuclear markers we want to keep
    membrane_markers = [
 # T cells
    'NaiveCD4T', 'CD4T', 'ICOS+Treg', 'CD103+CD4Treg', 'HLAII+CD27+Treg',
    'Tfh', 'TIGIT+Tfh', 'HLAII+GZMK+Tfh', 'GZMA+GZMK+CD4CTL', 'GZMB+CD4CTL', 'CD4Tcm-Tem',
    'NaiveCD8T', 'CD8T', 'GZMA+GZMK+CD8CTL', 'GZMB+CD8CTL', 'CD103+CD8Treg', 'CD8Trm',
    'CD103+CD8Trm', 'TIGIT+CD8T', 'CD8Tcm-Tem', 'γδT', 'ILC', 'ImmatureNK', 'MatureNK', 'NKT', 
    
    # Humoral cells
    'NaiveB', 'TransitionalB','B', 'CD27+B', 'DN2B', 'AgeAsscB',
    'Plasmablast', 'IgG4+Plasmablast', 'Plasma', 'IgG4+Plasma',
    
    # Myeloid cells
    'NonclassicalMonocyte', 'M1Macrophage', 'MPO+Macrophage',
    'M2Macrophage', 'MerTK+M1Macrophage', 'MerTK+M2Macrophage',
    'GPNMB+M1Macrophage', 'GPNMB+M2Macrophage', 'cDC1', 'cDC2', 'pDC',
    'Neutrophil',
    
    # Stromal and structural cells
    'Myofibroblast-SmoothMuscle', 'CD90+Myofibroblast', 'MMT', 
    'ProximalTubular', 'DistalTubular', 'StressedTubular',
    'Endothelial', 'LymphaticEndothelial', 'Podocyte', 'PEC',
    
    # Other
    'Apoptotic', 'RBC', 'TubuleBackground'
    ]
    
    nuclear_markers = ['GenericNucleus', 'Foxp3+Nucleus', 'Foxp3+TCF1+Nucleus', 'Foxp3+TOX+Nucleus',     'Foxp3+TCF1+TOX+Nucleus', 'Tbet+Nucleus', 'Tbet+TCF1+Nucleus', 'Tbet+TOX+Nucleus',     'Tbet+TCF1+TOX+Nucleus', 'Ki67+Nucleus', 'TCF1+Nucleus', 'Ki67+TCF1+Nucleus',    'TCF1+TOX+Nucleus', 'Ki67+TCF1+TOX+Nucleus', 'TOX+Nucleus', 'ApoptoticNucleus', 'NucBackground'
    ]
    
    # Get available membrane and nuclear markers from the data
    available_memcats = [x for x in membrane_markers if x in df.columns]
    available_nuccats = [x for x in nuclear_markers if x in df.columns]
    
    if len(available_memcats) == 0 or len(available_nuccats) == 0:
        print("Warning: No membrane or nuclear categories found")
        return df
    
    # Calculate maximum scores for membrane and nuclear markers
    memmax = df[available_memcats].max(axis=1)
    nucmax = df[available_nuccats].max(axis=1)
    
    # Get the category names with maximum scores
    memcatname = df[available_memcats].idxmax(axis=1)
    nuccatname = df[available_nuccats].idxmax(axis=1)
    
    # Get category indices
    memcat = [available_memcats.index(x) for x in memcatname]
    nuccat = [available_nuccats.index(x) for x in nuccatname]
    
    # Set category names based on scores using provided thresholds
    memcatname = [x if memmax[i] > memth else 'unclassified' for i, x in enumerate(memcatname)]
    nuccatname = [x if nucmax[i] > nucth else 'GenericNucleus' for i, x in enumerate(nuccatname)]
    
    # Add new columns to dataframe
    df['MemCat'] = memcat
    df['NucCat'] = nuccat
    df['MemCatName'] = memcatname
    df['NucCatName'] = nuccatname
    df['MemCatScore'] = memmax
    df['NucCatScore'] = nucmax
    
    return df

def process_mpi_data(df, mpidf, mpicols, sdir):
    # Filter for only WC markers
    mpicols = [x for x in mpicols if ('TAF' not in x) and ('DIC' not in x)]
    mpicols = [x for x in mpicols if 'WC' in x]  # Keep only WC markers
    
    # Remove specified markers
    excluded_markers = ['CD45', 'CD298', 'COLIII', 'COLIV', 'IFN-γ','RORgt', 'mTOC']
    mpicols = [x for x in mpicols if not any(marker in x for marker in excluded_markers)]
    mpicols.sort()
    
    # Define ordered categories
    ordered_memcats = [
 # T cells
    'NaiveCD4T', 'CD4T', 'ICOS+Treg', 'CD103+CD4Treg', 'HLAII+CD27+Treg',
    'Tfh', 'TIGIT+Tfh', 'HLAII+GZMK+Tfh', 'GZMA+GZMK+CD4CTL', 'GZMB+CD4CTL', 'CD4Tcm-Tem',
    'NaiveCD8T', 'CD8T', 'GZMA+GZMK+CD8CTL', 'GZMB+CD8CTL', 'CD103+CD8Treg', 'CD8Trm',
    'CD103+CD8Trm', 'TIGIT+CD8T', 'CD8Tcm-Tem', 'γδT', 'ILC', 'ImmatureNK', 'MatureNK', 'NKT', 
    
    # Humoral cells
    'NaiveB', 'TransitionalB', 'B', 'CD27+B', 'DN2B', 'AgeAsscB',
    'Plasmablast', 'IgG4+Plasmablast', 'Plasma', 'IgG4+Plasma',
    
    # Myeloid cells
    'NonclassicalMonocyte', 'M1Macrophage', 'MPO+Macrophage',
    'M2Macrophage', 'MerTK+M1Macrophage', 'MerTK+M2Macrophage',
    'GPNMB+M1Macrophage', 'GPNMB+M2Macrophage', 'cDC1', 'cDC2', 'pDC',
    'Neutrophil',
    
    # Stromal and structural cells
    'Myofibroblast-SmoothMuscle', 'CD90+Myofibroblast', 'MMT', 
    'ProximalTubular', 'DistalTubular', 'StressedTubular',
    'Endothelial', 'LymphaticEndothelial', 'Podocyte', 'PEC',
    
    # Other
    'Apoptotic', 'RBC', 'TubuleBackground'
    ]
    
    ordered_nuccats = ['GenericNucleus', 'Foxp3+Nucleus', 'Foxp3+TCF1+Nucleus', 'Foxp3+TOX+Nucleus',     'Foxp3+TCF1+TOX+Nucleus', 'Tbet+Nucleus', 'Tbet+TCF1+Nucleus', 'Tbet+TOX+Nucleus',     'Tbet+TCF1+TOX+Nucleus', 'Ki67+Nucleus', 'TCF1+Nucleus', 'Ki67+TCF1+Nucleus',    'TCF1+TOX+Nucleus', 'Ki67+TCF1+TOX+Nucleus', 'TOX+Nucleus', 'ApoptoticNucleus', 'NucBackground'
    ]
    
    # Get actual categories from data
    actual_memcats = np.unique(df['MemCatName'])
    actual_nuccats = np.unique(df['NucCatName'])
    
    # Use all ordered categories, even if they're not in the data
    cats = ordered_memcats + ordered_nuccats
    
    # Pre-calculate all Z-scores at once to avoid fragmentation
    z_scores = pd.DataFrame()
    for col in mpicols:
        mean_val = mpidf[col].mean()
        std_val = mpidf[col].std()
        z_scores[col+'_Z'] = (mpidf[col] - mean_val) / std_val
    
    # Concatenate the original data with Z-scores
    df = pd.concat([df, z_scores], axis=1)
    
    mpi_cols = [x for x in df.columns if 'Int-mean' in x and 'WC' in x]
    mpi_cols = [x for x in mpi_cols if not any(marker in x for marker in excluded_markers)]
    mpi_cols.sort()

    mu_array = np.zeros([len(cats), len(mpicols)])
    z_array = np.zeros([len(cats), len(mpicols)])
    
    # Process membrane categories in order
    for i, cat in enumerate(ordered_memcats):
        if cat in actual_memcats:
            mpi_df = df[df['MemCatName']==cat][mpi_cols]
            for j, col in enumerate(mpicols):
                mu = mpi_df[col].dropna().mean() if not mpi_df.empty else 0
                z = mpi_df[col+'_Z'].dropna().mean() if not mpi_df.empty else 0
                mu_array[i,j] = mu
                z_array[i,j] = z
    
    # Process nuclear categories in order
    for ii, cat in enumerate(ordered_nuccats):
        idx = ii + len(ordered_memcats)
        if cat in actual_nuccats:
            mpi_df = df[df['NucCatName']==cat][mpi_cols]
            for j, col in enumerate(mpicols):
                mu = mpi_df[col].dropna().mean() if not mpi_df.empty else 0
                z = mpi_df[col+'_Z'].dropna().mean() if not mpi_df.empty else 0
                mu_array[idx,j] = mu
                z_array[idx,j] = z

    # Create plots
    markers = [x.split('_Int')[0].split('_WC')[0] for x in mpicols]  # Remove _WC suffix
    
    # Plot mean MPI
    plt.figure(figsize=(4,4), dpi=600)
    plt.imshow(mu_array, cmap='jet')
    plt.yticks(ticks=np.arange(0,len(cats)), labels=cats, fontsize=5)
    plt.xticks(ticks=np.arange(0,len(mpicols)), labels=markers, rotation=90, fontsize=5)
    plt.colorbar()
    plt.xlabel('Marker', fontsize=8)
    plt.ylabel('Predicted Cell Class', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(sdir, 'MeanMPI-perClass.png'), dpi=600)
    plt.close()

    # Update markers for special characters
    markers = [x.replace('gamma','\u03B3') for x in markers]  
    markers = [x.replace('delta','\u03B4') for x in markers]
    
    # Plot Z-scores
    import matplotlib as mpl
    mpl.rcParams['font.size'] = 8 
    plt.figure(figsize=(10,8), dpi=600)
    plt.imshow(z_array, vmin=-3, vmax=3, cmap='seismic')
    plt.yticks(ticks=np.arange(0,len(cats)), labels=cats, fontsize=6)
    plt.xticks(ticks=np.arange(0,len(mpicols)), labels=markers, fontsize=6, rotation=90)
    plt.colorbar()
    plt.xlabel('Marker', fontsize=8)
    plt.ylabel('Predicted Cell or Nucleus Class', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(sdir, 'MeanMPIzscore-perClass.png'), dpi=600)
    plt.close()
    
    return df

def visualize_celltype_nucleus_relationship(df, ordered_memcats, ordered_nuccats, memcolor_dict, nuccolor_dict, sdir, csvname):
    """
    Visualize the relationship between cell types (membrane categories) and nucleus classes.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The filtered data containing both 'MemCatName' and 'NucCatName' columns
    ordered_memcats : list
        List of ordered membrane category names
    ordered_nuccats : list
        List of ordered nucleus category names
    memcolor_dict : dict
        Dictionary mapping membrane categories to RGB color values
    nuccolor_dict : dict
        Dictionary mapping nucleus categories to RGB color values
    sdir : str
        Directory to save output files
    csvname : str
        Name of the CSV file being processed
    """
    print("\nAnalyzing relationship between cell types and nucleus classes...")
    
    # Check if required columns exist
    if 'MemCatName' not in df.columns or 'NucCatName' not in df.columns:
        print("Error: Required columns 'MemCatName' or 'NucCatName' not found in DataFrame")
        return
    
    # Filter out cells with 'unclassified' membrane category for cleaner visualization
    filtered_df = df[df['MemCatName'] != 'unclassified'].copy()
    
    # Define structural cells to potentially exclude
    structural_cells = [
        'ProximalTubular', 'DistalTubular', 'StressedTubular',
        'Endothelial', 'LymphaticEndothelial', 'Myofibroblast-SmoothMuscle',
        'CD90+Myofibroblast', 'MMT',  
        'Podocyte', 'PEC', 'Apoptotic', 'RBC', 'TubuleBackground'
    ]
    
    # Get actual categories from data
    actual_memcats = [cat for cat in ordered_memcats if cat in filtered_df['MemCatName'].unique() and cat != 'unclassified']
    actual_nuccats = [cat for cat in ordered_nuccats if cat in filtered_df['NucCatName'].unique()]
    
    # Create a cross-tabulation of cell types vs nucleus classes
    cell_nucleus_matrix = pd.crosstab(
        filtered_df['MemCatName'], 
        filtered_df['NucCatName'],
        normalize='index'  # Normalize by row (cell type) to get percentage within each cell type
    ) * 100  # Convert to percentage
    
    # Ensure all categories are included (fill with zeros if missing)
    for cat in actual_nuccats:
        if cat not in cell_nucleus_matrix.columns:
            cell_nucleus_matrix[cat] = 0
    
    # Reorder columns (nucleus categories) based on ordered_nuccats
    cell_nucleus_matrix = cell_nucleus_matrix.reindex(columns=actual_nuccats, fill_value=0)
    
    # Reorder rows (membrane categories) based on ordered_memcats
    ordered_rows = [cat for cat in actual_memcats if cat in cell_nucleus_matrix.index]
    cell_nucleus_matrix = cell_nucleus_matrix.reindex(index=ordered_rows, fill_value=0)
    
    # Save raw data matrix to CSV
    matrix_file = os.path.join(sdir, f'CellType_NucleusClass_Matrix_{csvname}.csv')
    cell_nucleus_matrix.to_csv(matrix_file)
    print(f"Saved cell type vs nucleus class matrix to {matrix_file}")
    
    # Also create a version with raw counts (not normalized)
    count_matrix = pd.crosstab(filtered_df['MemCatName'], filtered_df['NucCatName'])
    count_matrix = count_matrix.reindex(index=ordered_rows, columns=actual_nuccats, fill_value=0)
    count_file = os.path.join(sdir, f'CellType_NucleusClass_Counts_{csvname}.csv')
    count_matrix.to_csv(count_file)
    print(f"Saved cell type vs nucleus class counts to {count_file}")
    
    # Create heatmap for percentages (within each cell type)
    plt.figure(figsize=(12, 14), dpi=600)
    
    # Plot the heatmap
    im = plt.imshow(cell_nucleus_matrix.values, cmap='viridis', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Percentage of Cell Type (%)', rotation=270, labelpad=20)
    
    # Set axis labels and title
    plt.xlabel('Nucleus Class', fontsize=12)
    plt.ylabel('Cell Type', fontsize=12)
    plt.title(f'Distribution of Nucleus Classes Within Each Cell Type', fontsize=14)
    
    # Set tick labels
    plt.xticks(range(len(actual_nuccats)), actual_nuccats, rotation=90, fontsize=10)
    plt.yticks(range(len(ordered_rows)), ordered_rows, fontsize=10)
    
    # Add text annotations in the heatmap cells
    for i in range(len(ordered_rows)):
        for j in range(len(actual_nuccats)):
            value = cell_nucleus_matrix.values[i, j]
            text_color = "white" if value > 30 else "black"
            plt.text(j, i, f"{value:.1f}", ha="center", va="center", color=text_color, fontsize=8)
    
    plt.tight_layout()
    
    # Save the heatmap
    heatmap_file = os.path.join(sdir, f'CellType_NucleusClass_Heatmap_{csvname}.png')
    plt.savefig(heatmap_file, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved cell type vs nucleus class heatmap to {heatmap_file}")
    
    # Create a second heatmap with alternate normalization - percentage of each nucleus class in each cell type
    alt_matrix = pd.crosstab(
        filtered_df['MemCatName'], 
        filtered_df['NucCatName'],
        normalize='columns'  # Normalize by column (nucleus class) 
    ) * 100  # Convert to percentage
    
    # Reorder rows and columns
    alt_matrix = alt_matrix.reindex(index=ordered_rows, columns=actual_nuccats, fill_value=0)
    
    # Save alternate normalization matrix
    alt_file = os.path.join(sdir, f'NucleusClass_CellType_Distribution_{csvname}.csv')
    alt_matrix.to_csv(alt_file)
    print(f"Saved nucleus class distribution across cell types to {alt_file}")
    
    # Create alternate heatmap
    plt.figure(figsize=(12, 14), dpi=600)
    
    # Plot the heatmap
    im = plt.imshow(alt_matrix.values, cmap='plasma', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('Percentage of Nucleus Class (%)', rotation=270, labelpad=20)
    
    # Set axis labels and title
    plt.xlabel('Nucleus Class', fontsize=12)
    plt.ylabel('Cell Type', fontsize=12)
    plt.title(f'Distribution of Each Nucleus Class Across Cell Types', fontsize=14)
    
    # Set tick labels
    plt.xticks(range(len(actual_nuccats)), actual_nuccats, rotation=90, fontsize=10)
    plt.yticks(range(len(ordered_rows)), ordered_rows, fontsize=10)
    
    # Add text annotations in the heatmap cells
    for i in range(len(ordered_rows)):
        for j in range(len(actual_nuccats)):
            value = alt_matrix.values[i, j]
            text_color = "white" if value > 30 else "black"
            plt.text(j, i, f"{value:.1f}", ha="center", va="center", color=text_color, fontsize=8)
    
    plt.tight_layout()
    
    # Save the alternate heatmap
    alt_heatmap_file = os.path.join(sdir, f'NucleusClass_Distribution_Heatmap_{csvname}.png')
    plt.savefig(alt_heatmap_file, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved nucleus class distribution heatmap to {alt_heatmap_file}")
    
    return cell_nucleus_matrix, alt_matrix

def plot_cell_fractions(df, sample_col, cat_col, color_dict, ordered_cats, sdir, title_prefix, filename_prefix, csvname, exclude_structural=False, exclude_specific=False):
    """
    Plot the fractions of different cell types for each sample.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The filtered data
    sample_col : str
        Column name for sample identifiers
    cat_col : str
        Column name for cell categories (e.g., 'MemCatName', 'NucCatName')
    color_dict : dict
        Dictionary mapping categories to RGB color values
    ordered_cats : list
        List of ordered category names
    sdir : str
        Directory to save output files
    title_prefix : str
        Prefix for plot title
    filename_prefix : str
        Prefix for output filename
    csvname : str
        Name of the CSV file being processed
    exclude_structural : bool, optional
        Whether to exclude structural cells and unclassified
    exclude_specific : bool, optional
        Whether to exclude specific cells (RBC, TubuleBackground, unclassified)
    """
    print(f"\n=== Debug Information ===")
    if cat_col not in df.columns:
        print(f"Warning: {cat_col} not found in DataFrame columns")
        return
        
    # Define structural cell types
    structural_cells = [
        'ProximalTubular', 'DistalTubular', 'StressedTubular',
        'Endothelial', 'LymphaticEndothelial', 'Myofibroblast-SmoothMuscle',
        'CD90+Myofibroblast', 'MMT', 
        'Podocyte', 'PEC', 'Apoptotic', 'RBC', 'TubuleBackground'
    ]
    
    # Define cells to exclude for the new option
    exclude_cells = ['RBC', 'TubuleBackground', 'unclassified']
    
    # Create a copy of the dataframe to avoid modifying the original
    plot_df = df.copy()
    
    # Filter out cells based on different exclusion criteria
    if exclude_structural and cat_col == 'MemCatName':
        plot_df = plot_df[~plot_df[cat_col].isin(structural_cells + ['unclassified'])]
    elif exclude_specific and cat_col == 'MemCatName':
        plot_df = plot_df[~plot_df[cat_col].isin(exclude_cells)]
    
    # THIS IS THE FIXED PART: We now use the filtered categories directly
    # We no longer override the categories by finding max values again
    
    samples = np.unique(plot_df[sample_col])
    if len(samples) == 0:
        print(f"Warning: No samples found in {sample_col}")
        return
    
    cell_fractions_data = []
    for sample in samples:
        subdf = plot_df[plot_df[sample_col] == sample]
        if len(subdf) == 0:
            continue
            
        # Calculate proportions for this sample
        value_counts = subdf[cat_col].value_counts()
        total_cells = len(subdf)
        proportions = value_counts / total_cells
        
        # For membrane categories, ensure all categories are present
        if cat_col == 'MemCatName':
            if exclude_structural:
                all_categories = [cat for cat in ordered_cats if cat not in structural_cells and cat != 'unclassified']
            elif exclude_specific:
                all_categories = [cat for cat in ordered_cats if cat not in exclude_cells]
            else:
                all_categories = ordered_cats
                if 'unclassified' not in all_categories:
                    all_categories = list(all_categories) + ['unclassified']
            
            # Create a complete dictionary with all categories
            complete_proportions = {cat: 0.0 for cat in all_categories}
            complete_proportions.update(proportions)
            proportions = complete_proportions
        
        cell_fractions_data.append({
            'sample': sample,
            **proportions
        })
    
    if not cell_fractions_data:
        print(f"Warning: No valid data for {title_prefix}")
        return
        
    cell_fractions = pd.DataFrame(cell_fractions_data)
    cell_fractions = cell_fractions.fillna(0)
    
    # Get categories for plotting
    if cat_col == 'MemCatName':
        if exclude_structural:
            available_cats = [cat for cat in ordered_cats if cat not in structural_cells and cat != 'unclassified']
        elif exclude_specific:
            available_cats = [cat for cat in ordered_cats if cat not in exclude_cells]
        else:
            available_cats = ordered_cats
            if 'unclassified' not in ordered_cats:
                available_cats = list(available_cats) + ['unclassified']
                if 'unclassified' not in color_dict:
                    color_dict['unclassified'] = [128, 128, 128]
    else:
        # For nuclear categories, use consistent ordering from ordered_cats
        available_cats = [cat for cat in ordered_cats if cat in cell_fractions.columns and cat != 'sample']
    
    print(f"Final categories for plotting: {available_cats}")
    print(f"Number of categories to plot: {len(available_cats)}")
    
    # Convert color dictionary values to tuples
    color_dict = {k: tuple(v) for k, v in color_dict.items()}
    
    plt.figure(figsize=(15, 9))
    bottom = np.zeros(len(samples))
    
    for cat in available_cats:
        if cat in color_dict:
            color = rgb_to_hex(color_dict[cat])
        else:
            print(f"Warning: No color defined for category '{cat}', using default gray")
            color = '#808080'
        
        values = cell_fractions[cat].values if cat in cell_fractions else np.zeros(len(samples))
        plt.bar(range(len(samples)), values, bottom=bottom, color=color, label=cat)
        bottom += values
    
    plt.xlabel('Samples')
    plt.ylabel('Proportion')
    
    # Set appropriate suffix based on exclusion
    if exclude_structural:
        suffix = ' (Excluding Structural Cells & Unclassified)'
    elif exclude_specific:
        suffix = ' (Excluding RBC, TubuleBackground & Unclassified)'
    else:
        suffix = ''
        
    plt.title(f'{title_prefix}{suffix}')
    plt.xticks(range(len(samples)), samples, rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # Set appropriate filename suffix
    if exclude_structural:
        filename_suffix = '_noStructural_noUnclassified'
    elif exclude_specific:
        filename_suffix = '_noRBC_noTubuleBkg_noUnclassified'
    else:
        filename_suffix = ''
        
    plt.savefig(os.path.join(sdir, f'{filename_prefix}_{csvname}{filename_suffix}.png'), dpi=600, bbox_inches='tight')
    plt.close()

def generate_cell_proportion_tables(df, sample_col, cat_col, ordered_cats, sdir, filename_prefix, csvname, exclude_structural=False, exclude_specific=False):
    """
    Generate tables of cell proportions per patient and mean proportions across the dataset,
    sorted from highest to lowest proportion.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The filtered data
    sample_col : str
        Column name for patient/sample identifiers
    cat_col : str
        Column name for cell categories (e.g., 'MemCatName', 'NucCatName')
    ordered_cats : list
        List of ordered category names
    sdir : str
        Directory to save output files
    filename_prefix : str
        Prefix for output filenames
    csvname : str
        Name of the CSV file being processed
    exclude_structural : bool, optional
        Whether to exclude structural cells and unclassified cells
    exclude_specific : bool, optional
        Whether to exclude specific cells (RBC, TubuleBackground, unclassified)
    """
    print(f"\nGenerating cell proportion tables for {cat_col}...")
    
    # Define structural cell types
    structural_cells = [
        'ProximalTubular', 'DistalTubular', 'StressedTubular',
        'Endothelial', 'LymphaticEndothelial', 'Myofibroblast-SmoothMuscle',
        'CD90+Myofibroblast', 'MMT', 
        'Podocyte', 'PEC', 'Apoptotic', 'RBC', 'TubuleBackground'
    ]
    
    # Define cells to exclude for the new option
    exclude_cells = ['RBC', 'TubuleBackground', 'unclassified']
    
    # Create a copy of the dataframe to avoid modifying the original
    table_df = df.copy()
    
    # Filter out cells based on different exclusion criteria
    if exclude_structural and cat_col == 'MemCatName':
        table_df = table_df[~table_df[cat_col].isin(structural_cells + ['unclassified'])]
    elif exclude_specific and cat_col == 'MemCatName':
        table_df = table_df[~table_df[cat_col].isin(exclude_cells)]
    
    samples = np.unique(table_df[sample_col])
    if len(samples) == 0:
        print(f"Warning: No samples found in {sample_col}")
        return
    
    # Initialize dataframe for patient proportions
    patient_proportions = pd.DataFrame(index=samples)
    
    # Calculate proportions for each patient
    for sample in samples:
        subdf = table_df[table_df[sample_col] == sample]
        if len(subdf) == 0:
            continue
            
        # Calculate proportions for this sample
        value_counts = subdf[cat_col].value_counts()
        total_cells = len(subdf)
        proportions = value_counts / total_cells * 100  # Convert to percentage
        
        # Add to patient proportions dataframe
        for cat, prop in proportions.items():
            patient_proportions.loc[sample, cat] = prop
    
    # Fill NaN values with 0
    patient_proportions = patient_proportions.fillna(0)
    
    # Calculate mean proportions across all patients
    mean_proportions = patient_proportions.mean().reset_index()
    mean_proportions.columns = ['Cell_Class', 'Mean_Proportion_Percent']
    
    # Sort mean proportions from high to low
    mean_proportions = mean_proportions.sort_values(by='Mean_Proportion_Percent', ascending=False)
    
    # Get the ordered classes from highest to lowest mean proportion
    ordered_classes = mean_proportions['Cell_Class'].tolist()
    
    # Reorder the patient proportions based on mean proportions (high to low)
    patient_proportions = patient_proportions[ordered_classes]
    
    # Transpose patient proportions for a better view (cells as rows, patients as columns)
    patient_proportions_t = patient_proportions.T
    patient_proportions_t['Mean'] = patient_proportions_t.mean(axis=1)
    
    # Sort the transposed table by the Mean column (high to low)
    patient_proportions_t = patient_proportions_t.sort_values(by='Mean', ascending=False)
    
    # Format the tables
    patient_proportions_t = patient_proportions_t.round(2)
    mean_proportions['Mean_Proportion_Percent'] = mean_proportions['Mean_Proportion_Percent'].round(2)
    
    # Set appropriate suffix based on exclusion
    if exclude_structural:
        suffix = '_noStructural_noUnclassified'
    elif exclude_specific:
        suffix = '_noRBC_noTubuleBkg_noUnclassified'
    else:
        suffix = ''
    
    # Generate filenames
    patient_file = f'{filename_prefix}_{csvname}_per_patient{suffix}.csv'
    mean_file = f'{filename_prefix}_{csvname}_mean{suffix}.csv'
    
    # Save tables
    patient_proportions_t.to_csv(os.path.join(sdir, patient_file))
    mean_proportions.to_csv(os.path.join(sdir, mean_file), index=False)
    
    print(f"Saved patient proportions to {patient_file}")
    print(f"Saved mean proportions to {mean_file}")
    
    return patient_proportions_t, mean_proportions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir',type=str,default='/project/mclark/SCAMPI_datasets/',help='')
    parser.add_argument('--dataset',type=str,default='NK2',help='')
    parser.add_argument('--readdir',type=str,default='classify_by_dilation-sameTH/with-MPI',help='')
    parser.add_argument('--sample',type=str,default='all_samples',help='')
    parser.add_argument('--area',type=str,default='all_areas',help='')
    parser.add_argument('--memth',type=float,default=0.0,help='Threshold for membrane classification 0-10')
    parser.add_argument('--nucth',type=float,default=0.0,help='Threshold for nucleus classification 0-10')
    
    args,unparsed = parser.parse_known_args()
    
    rdir = os.path.join(args.rootdir,args.dataset,args.readdir)
    sdir = os.path.join(rdir,'Prevalence_Plots_filtered5_10glomeruli')
    
    if not os.path.exists(sdir):
        os.makedirs(sdir)

    if (args.sample != 'all_samples') and (args.area !='all_areas'):
        csv_names = os.listdir(os.path.join(rdir,args.sample,args.area))
        csv_names = ['/'.join([args.sample,args.area,x]) for x in csv_names]
        csv_names = [x for x in csv_names if 'cellMPI' not in x]
    else:
        csv_names = ['combined_SAM_scores_filtered5_10woMonoglomeruli.csv']

    for csv_name in csv_names:
        csvname = csv_name.split('_SAM')[0].split('/')[-1]
        
        print(f"Processing {csvname}...")
        df = pd.read_csv(os.path.join(rdir,csv_name), low_memory=False)
        if 'Sample' in df.columns:
            df['Sample'] = df['Sample'].astype(str)
        
        # Filter cells and get categories using the threshold parameters
        df = filter_cells(df, args.memth, args.nucth)
        
        # Save filtered data
        df.to_csv(os.path.join(rdir,csvname+'_filtered5_10.csv'))
        
        # Get unique categories
        memcats = df['MemCatName'].unique()
        nuccats = df['NucCatName'].unique()
        
        # Set up ordered nucleus categories to maintain consistent order across datasets
        ordered_nuccats = [
            'GenericNucleus', 'Foxp3+Nucleus', 'Foxp3+TCF1+Nucleus', 'Foxp3+TOX+Nucleus', 
    'Foxp3+TCF1+TOX+Nucleus', 'Tbet+Nucleus', 'Tbet+TCF1+Nucleus', 'Tbet+TOX+Nucleus', 
    'Tbet+TCF1+TOX+Nucleus', 'Ki67+Nucleus', 'TCF1+Nucleus', 'Ki67+TCF1+Nucleus', 
    'TCF1+TOX+Nucleus', 'Ki67+TCF1+TOX+Nucleus', 'TOX+Nucleus', 'ApoptoticNucleus', 'NucBackground'
        ]
        
        # Setup color dictionaries based on dataset
        if args.dataset in ['IgG4', 'AAV', 'NK2']:
            memcolor_dict = {
# T cell (Cool blue for CD4, Warm red for CD8, Purple for specialized)
'NaiveCD4T': [100, 150, 255],          # Light blue
'CD4T': [50, 100, 255],                # Medium blue
'ICOS+Treg': [0, 50, 255],             # Deep blue
'CD103+CD4Treg': [0, 0, 200],          # Navy blue
'HLAII+CD27+Treg': [65, 105, 225],     # Royal blue
'Tfh': [70, 130, 180],                 # Steel blue
'TIGIT+Tfh': [30, 144, 255],           # Dodger blue
'HLAII+GZMK+Tfh': [0, 191, 255],       # Deep sky blue
'GZMA+GZMK+CD4CTL': [135, 206, 250],   # Light sky blue
'GZMB+CD4CTL': [135, 206, 235],        # Sky blue
'CD4Tcm-Tem': [176, 196, 222],         # Light steel blue
'NaiveCD8T': [255, 160, 160],          # Light red
'CD8T': [255, 99, 71],                 # Tomato red
'GZMA+GZMK+CD8CTL': [220, 20, 60],     # Crimson
'GZMB+CD8CTL': [233, 150, 122],        # Dark salmon
'CD103+CD8Treg': [178, 34, 34],        # Fire brick red
'CD8Trm': [139, 0, 0],                 # Dark red
'CD103+CD8Trm': [165, 42, 42],         # Brown red
'TIGIT+CD8T': [128, 0, 0],             # Maroon
'CD8Tcm-Tem': [205, 92, 92],           # Indian red
'γδT': [147, 112, 219],                # Medium purple
'ILC': [138, 43, 226],                 # Blue violet
'ImmatureNK': [148, 0, 211],           # Dark violet
'MatureNK': [153, 50, 204],            # Dark orchid
'NKT': [186, 85, 211],                 # Medium orchid

# Humoral cell (Green gradient)
'NaiveB': [144, 238, 144],             # Light green
'TransitionalB': [124, 252, 0],        # Lawn green
'B': [34, 139, 34],                    # Forest green
'CD27+B': [0, 128, 0],                 # Green
'DN2B': [0, 100, 0],                   # Dark green
'AgeAsscB': [46, 139, 87],             # Sea green
'Plasmablast': [60, 179, 113],         # Medium sea green
'IgG4+Plasmablast': [32, 178, 170],    # Light sea green
'Plasma': [143, 188, 143],             # Dark sea green
'IgG4+Plasma': [152, 251, 152],        # Pale green

# Myeloid cell (Yellow-Brown gradient)
'NonclassicalMonocyte': [255, 215, 0],      # Gold
'M1Macrophage': [255, 193, 37],             # Golden yellow
'MPO+Macrophage': [255, 185, 15],           # Dark golden yellow
'M2Macrophage': [255, 165, 0],              # Orange
'MerTK+M1Macrophage': [255, 140, 0],        # Dark orange
'MerTK+M2Macrophage': [255, 127, 80],       # Coral
'GPNMB+M1Macrophage': [255, 99, 71],        # Tomato
'GPNMB+M2Macrophage': [255, 69, 0],         # Red-orange
'cDC1': [244, 164, 96],                     # Sandy brown
'cDC2': [218, 160, 33],                     # Golden yellow
'pDC': [250, 237, 125],                     # Pale yellow
'Neutrophil': [255, 228, 181],              # Moccasin

# Stromal cell (Pink shades)
'Myofibroblast-SmoothMuscle': [199, 21, 133],   # Medium violet red
'CD90+Myofibroblast': [219, 112, 147],          # Pale violet red
'MMT': [255, 20, 147],                          # Deep pink

# Structural cell (Gray-range colors)
'ProximalTubular': [105, 105, 105],           # Dim gray
'DistalTubular': [128, 128, 128],            # Gray
'StressedTubular': [169, 169, 169],          # Dark gray
'Endothelial': [190, 190, 190],              # Gray
'LymphaticEndothelial': [211, 211, 211],     # Light gray
'Podocyte': [220, 220, 220],                 # Gainsboro
'PEC': [245, 245, 245],                      # White smoke

# Other (Distinctive colors)
'Apoptotic': [102, 51, 153],                 # Dark purple
'RBC': [255, 0, 0],                          # Pure red
'TubuleBackground': [224, 255, 255],         # Light cyan
    
                 # Unclassified (Very bright white-gray)
                'unclassified': [245, 245, 245]            # Almost white gray
            }

            # Nuclear markers (Purple gradient with more distinction)
            nuccolor_dict = {  # Nucleus class (Highly distinct colors across spectrum)
    'GenericNucleus': [25, 25, 112],                # Midnight blue
    'Foxp3+Nucleus': [255, 0, 255],                 # Magenta
    'Foxp3+TCF1+Nucleus': [218, 112, 214],          # Orchid - new
    'Foxp3+TOX+Nucleus': [238, 130, 238],           # Violet - new
    'Foxp3+TCF1+TOX+Nucleus': [221, 160, 221],      # Plum - new
    'Tbet+Nucleus': [0, 0, 205],                    # Medium blue
    'Tbet+TCF1+Nucleus': [100, 149, 237],           # Cornflower blue - new
    'Tbet+TOX+Nucleus': [30, 144, 255],             # Dodger blue - new
    'Tbet+TCF1+TOX+Nucleus': [135, 206, 250],       # Light sky blue - new
    'Ki67+Nucleus': [220, 20, 60],                  # Crimson
    'TCF1+Nucleus': [0, 206, 209],                  # Dark turquoise
    'Ki67+TCF1+Nucleus': [255, 20, 147],            # Deep pink 
    'TCF1+TOX+Nucleus': [255, 140, 0],              # Dark orange
    'Ki67+TCF1+TOX+Nucleus': [255, 127, 80],        # Coral - new
    'TOX+Nucleus': [50, 205, 50],                   # Lime green
    'ApoptoticNucleus': [189, 183, 107],            # Dark khaki - renamed from ApoptoticNuc
    'NucBackground': [230, 230, 250]                # Lavender
            }
        else:
            memcolor_dict = {cat: [128,128,128] for cat in memcats}
            nuccolor_dict = {cat: [200,200,200] for cat in nuccats}
        
        # Process MPI data
        cols = df.columns
        mpicols = [x for x in cols if 'Int-mean' in x]
        mpidf = df[mpicols]
        
        # Process and add MPI data - now passing sdir
        df = process_mpi_data(df, mpidf, mpicols, sdir)
        
        # Visualize relationship between cell types and nucleus classes
        visualize_celltype_nucleus_relationship(
            df=df,
            ordered_memcats=list(memcolor_dict.keys()),
            ordered_nuccats=ordered_nuccats,  # Use consistent ordered nucleus categories
            memcolor_dict=memcolor_dict,
            nuccolor_dict=nuccolor_dict,
            sdir=sdir,
            csvname=csvname
        )
        
        # In the plotting section:
        for sample_col in ['CompName', 'Sample', 'Patient']:
            if sample_col in df.columns:
                # Generate proportion tables for membrane categories (all cells)
                generate_cell_proportion_tables(
                    df=df,
                    sample_col=sample_col,
                    cat_col='MemCatName',
                    ordered_cats=list(memcolor_dict.keys()),
                    sdir=sdir,
                    filename_prefix=f'{sample_col}_CellTypeProportions',
                    csvname=csvname,
                    exclude_structural=False
                )
                
                # Generate proportion tables for membrane categories (excluding structural cells)
                generate_cell_proportion_tables(
                    df=df,
                    sample_col=sample_col,
                    cat_col='MemCatName',
                    ordered_cats=list(memcolor_dict.keys()),
                    sdir=sdir,
                    filename_prefix=f'{sample_col}_CellTypeProportions',
                    csvname=csvname,
                    exclude_structural=True
                )
                
                # NEW: Generate proportion tables excluding RBC, TubuleBackground, and unclassified
                generate_cell_proportion_tables(
                    df=df,
                    sample_col=sample_col,
                    cat_col='MemCatName',
                    ordered_cats=list(memcolor_dict.keys()),
                    sdir=sdir,
                    filename_prefix=f'{sample_col}_CellTypeProportions',
                    csvname=csvname,
                    exclude_specific=True
                )
                
                # Generate proportion tables for nuclear categories
                generate_cell_proportion_tables(
                    df=df,
                    sample_col=sample_col,
                    cat_col='NucCatName',
                    ordered_cats=ordered_nuccats,  # Use consistent ordered nucleus categories
                    sdir=sdir,
                    filename_prefix=f'{sample_col}_NucleusTypeProportions',
                    csvname=csvname,
                    exclude_structural=False
                )
            
                # Original plot with all cells (including unclassified)
                plot_cell_fractions(
                    df=df,
                    sample_col=sample_col,
                    cat_col='MemCatName',
                    color_dict=memcolor_dict,
                    ordered_cats=list(memcolor_dict.keys()),
                    sdir=sdir,
                    title_prefix=f'Membrane Type by {sample_col}',
                    filename_prefix=f'{sample_col}_CellTypePrevalence',
                    csvname=csvname,
                    exclude_structural=False
                )
            
                # Plot excluding structural cells and unclassified
                plot_cell_fractions(
                    df=df,
                    sample_col=sample_col,
                    cat_col='MemCatName',
                    color_dict=memcolor_dict,
                    ordered_cats=list(memcolor_dict.keys()),
                    sdir=sdir,
                    title_prefix=f'Membrane Type by {sample_col}',
                    filename_prefix=f'{sample_col}_CellTypePrevalence',
                    csvname=csvname,
                    exclude_structural=True
                )
                
                # NEW: Plot excluding RBC, TubuleBackground, and unclassified
                plot_cell_fractions(
                    df=df,
                    sample_col=sample_col,
                    cat_col='MemCatName',
                    color_dict=memcolor_dict,
                    ordered_cats=list(memcolor_dict.keys()),
                    sdir=sdir,
                    title_prefix=f'Membrane Type by {sample_col}',
                    filename_prefix=f'{sample_col}_CellTypePrevalence',
                    csvname=csvname,
                    exclude_specific=True
                )
            
                # Nuclear plot with consistent ordering
                plot_cell_fractions(
                    df=df,
                    sample_col=sample_col,
                    cat_col='NucCatName',
                    color_dict=nuccolor_dict,
                    ordered_cats=ordered_nuccats,  # Use consistent ordered nucleus categories
                    sdir=sdir,
                    title_prefix=f'Nuclear Type by {sample_col}',
                    filename_prefix=f'{sample_col}_NucleusTypePrevalence',
                    csvname=csvname,
                    exclude_structural=False
                )

if __name__=="__main__":
    main()
