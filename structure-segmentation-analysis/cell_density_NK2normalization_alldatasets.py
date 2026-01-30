import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tifffile import imread
from scipy import stats

# Define the custom cell type order
CELL_TYPE_ORDER = [
    'NaiveCD4T', 'CD4T', 'ICOS+Treg', 'CD103+CD4Treg', 'HLAII+CD27+Treg',
    'Tfh', 'TIGIT+Tfh', 'HLAII+GZMK+Tfh', 'GZMA+GZMK+CD4CTL', 'GZMB+CD4CTL','CD4Tcm-Tem',
    'NaiveCD8T', 'CD8T', 'GZMA+GZMK+CD8CTL', 'GZMB+CD8CTL','CD103+CD8Treg', 'CD8Trm',
    'CD103+CD8Trm', 'TIGIT+CD8T', 'CD8Tcm-Tem', 'γδT', 'ILC', 'ImmatureNK', 'MatureNK', 'NKT', 
    'NaiveB', 'TransitionalB','B', 'CD27+B', 'DN2B', 'AgeAsscB',
    'Plasmablast', 'IgG4+Plasmablast', 'Plasma', 'IgG4+Plasma',
    'M1Macrophage', 'MPO+Macrophage',
    'M2Macrophage', 'MerTK+M1Macrophage', 'MerTK+M2Macrophage',
    'GPNMB+M1Macrophage', 'GPNMB+M2Macrophage', 'cDC1', 'cDC2', 'pDC',
    'Neutrophil',
    'Myofibroblast-SmoothMuscle', 'CD90+Myofibroblast', 'MMT',
    'ProximalTubular', 'DistalTubular', 'StressedTubular',
    'Endothelial', 'LymphaticEndothelial', 'Podocyte', 'PEC', 'Apoptotic','RBC','TubuleBackground','unclassified'
]

# Define the anatomical structure order
STRUCTURE_ORDER = [
    'glomeruli', 'interstitial', 'outside the tissue', 'tubules', 'vessels'
]

# Define cell types to exclude from proportional heatmaps
EXCLUDE_FROM_PROPORTIONAL = [
    'ProximalTubular', 'DistalTubular', 'StressedTubular', 
    'Endothelial', 'LymphaticEndothelial', 'Podocyte', 'PEC', 
    'RBC', 'TubuleBackground', 'unclassified'
]

# Conversion factor: 1 pixel = 0.1507 µm
PIXEL_TO_UM = 0.1507
PIXEL_TO_UM2 = PIXEL_TO_UM ** 2  # For area conversion


def filter_structures(pivot_table, exclude_outside_tissue=False):
    """
    Filter structures in pivot table based on options.
    
    Args:
        pivot_table: DataFrame with structures as columns or index
        exclude_outside_tissue: Whether to exclude 'outside the tissue' column/index
    
    Returns:
        Filtered pivot table
    """
    if exclude_outside_tissue:
        # Check if 'outside the tissue' is in columns
        if 'outside the tissue' in pivot_table.columns:
            pivot_table = pivot_table.drop('outside the tissue', axis=1)
            print("  Excluded 'outside the tissue' from columns")
        # Check if 'outside the tissue' is in index
        if 'outside the tissue' in pivot_table.index:
            pivot_table = pivot_table.drop('outside the tissue', axis=0)
            print("  Excluded 'outside the tissue' from index")
    
    return pivot_table


def filter_cell_types_from_proportional(pivot_table, exclude_cell_types=EXCLUDE_FROM_PROPORTIONAL):
    """
    Filter specific cell types from proportional analysis pivot tables.
    
    Args:
        pivot_table: DataFrame with cell types as index
        exclude_cell_types: List of cell types to exclude from proportional analysis
    
    Returns:
        Filtered pivot table
    """
    original_count = len(pivot_table.index)
    
    # Filter out excluded cell types from index
    if hasattr(pivot_table.index, 'levels'):  # MultiIndex
        # For MultiIndex, filter based on the cell type level
        mask = ~pivot_table.index.get_level_values(-1).isin(exclude_cell_types)
        pivot_table = pivot_table[mask]
    else:
        # For regular index
        mask = ~pivot_table.index.isin(exclude_cell_types)
        pivot_table = pivot_table[mask]
    
    excluded_count = original_count - len(pivot_table.index)
    if excluded_count > 0:
        print(f"  Excluded {excluded_count} cell types from proportional analysis: {[ct for ct in exclude_cell_types if ct in pivot_table.index.values or (hasattr(pivot_table.index, 'levels') and ct in pivot_table.index.get_level_values(-1))]}")
    
    return pivot_table


def apply_ordering(pivot_df, cell_type_order=CELL_TYPE_ORDER, structure_order=STRUCTURE_ORDER, exclude_outside_tissue=False):
    """
    Apply consistent ordering to cell types (rows) and structures (columns) in a pivot table.
    
    Args:
        pivot_df: DataFrame with cell types as index and structures as columns
        cell_type_order: List of cell types in desired order
        structure_order: List of structures in desired order
        exclude_outside_tissue: Whether to exclude 'outside the tissue' from structures
        
    Returns:
        Reordered DataFrame
    """
    # Apply structure filtering first
    pivot_df = filter_structures(pivot_df, exclude_outside_tissue)
    
    # Order cell types (rows)
    available_cells = pivot_df.index.tolist()
    ordered_rows = [cell for cell in cell_type_order if cell in available_cells]
    ordered_rows.extend([cell for cell in available_cells if cell not in cell_type_order])
    
    # Order structures (columns) - filter out 'outside the tissue' if requested
    available_structures = pivot_df.columns.tolist()
    structure_order_filtered = structure_order.copy()
    if exclude_outside_tissue and 'outside the tissue' in structure_order_filtered:
        structure_order_filtered.remove('outside the tissue')
    
    ordered_cols = [struct for struct in structure_order_filtered if struct in available_structures]
    ordered_cols.extend([struct for struct in available_structures if struct not in structure_order_filtered])
    
    # Reindex both dimensions
    return pivot_df.reindex(index=ordered_rows, columns=ordered_cols)


def calculate_proportional_contributions(difference_matrix):
    """
    Calculate the proportional contribution of each structure to the total difference for each cell type.
    
    For each cell type (row), calculates: structure_difference / total_absolute_difference_across_structures
    
    Args:
        difference_matrix: DataFrame with cell types as rows and structures as columns
        
    Returns:
        DataFrame with proportional contributions (values between -1 and 1)
    """
    proportional_contrib = difference_matrix.copy()
    
    for cell_type in difference_matrix.index:
        cell_differences = difference_matrix.loc[cell_type, :]
        
        # Calculate total absolute difference for this cell type across all structures
        total_abs_diff = cell_differences.abs().sum()
        
        if total_abs_diff > 0:
            # Calculate proportional contribution: difference / total_absolute_difference
            # This preserves the sign (positive/negative) while showing relative magnitude
            proportional_contrib.loc[cell_type, :] = cell_differences / total_abs_diff
        else:
            # If no differences, set all proportions to 0
            proportional_contrib.loc[cell_type, :] = 0
    
    return proportional_contrib


def calculate_tissue_areas(df):
    """
    Calculate tissue areas for each Dataset-CompName combination.
    Then aggregate to Patient-Dataset-structure level.
    """
    print("Loading tissue masks to calculate tissue areas...")
    
    # Get unique Dataset-CompName combinations
    unique_comps = df[['Dataset', 'CompName']].drop_duplicates()
    
    # Dictionary to store data for each Dataset-CompName
    comp_data = {}
    
    for _, row in unique_comps.iterrows():
        dataset = row['Dataset']
        comp_name = row['CompName']
        
        print(f"  Processing {comp_name}...")
        
        # Define paths to the tissue mask files
        tissue_mask_path1 = f'/project/mclark/SCAMPI_datasets/{dataset}/tissue_composite_masks/DAPI_UV_1_{dataset}_{comp_name}.tif'
        tissue_mask_path2 = f'/project/mclark/SCAMPI_datasets/{dataset}/tissue_composite_masks/DAPI_UV_2_{dataset}_{comp_name}.tif'
        
        tissue_mask = None
        
        # Try to load tissue mask
        if os.path.exists(tissue_mask_path1):
            try:
                tissue_mask = imread(tissue_mask_path1)
                print(f"    Loaded tissue mask 1 with shape {tissue_mask.shape}")
            except Exception as e:
                print(f"    Error loading tissue mask 1: {e}")
        
        if tissue_mask is None and os.path.exists(tissue_mask_path2):
            try:
                tissue_mask = imread(tissue_mask_path2)
                print(f"    Loaded tissue mask 2 with shape {tissue_mask.shape}")
            except Exception as e:
                print(f"    Error loading tissue mask 2: {e}")
        
        if tissue_mask is not None:
            # Count tissue pixels (non-zero pixels in tissue mask)
            tissue_pixels = np.sum(tissue_mask > 0)
            tissue_area_um2 = tissue_pixels * PIXEL_TO_UM2
            
            comp_data[(dataset, comp_name)] = {
                'tissue_area_pixels': tissue_pixels,
                'tissue_area_um2': tissue_area_um2
            }
            
            print(f"    {comp_name}: tissue={tissue_pixels} pixels ({tissue_area_um2:.2f} µm²)")
        else:
            print(f"    Warning: Could not load tissue mask for {comp_name}")
            # Use default values if masks not found
            comp_data[(dataset, comp_name)] = {
                'tissue_area_pixels': 1000000,  # Default 1M pixels
                'tissue_area_um2': 1000000 * PIXEL_TO_UM2
            }
    
    # Now create data for each Patient-Dataset-structure combination
    area_data = []
    
    for (dataset, comp_name), comp_info in comp_data.items():
        # Get all unique Patient-structure combinations for this Dataset-CompName
        comp_cells = df[(df['Dataset'] == dataset) & (df['CompName'] == comp_name)]
        if not comp_cells.empty:
            unique_patient_structures = comp_cells[['Patient', 'structure']].drop_duplicates()
            
            for _, ps_row in unique_patient_structures.iterrows():
                area_data.append({
                    'Dataset': dataset,
                    'CompName': comp_name,
                    'Patient': ps_row['Patient'],
                    'structure': ps_row['structure'],
                    'tissue_area_pixels': comp_info['tissue_area_pixels'],
                    'tissue_area_um2': comp_info['tissue_area_um2']
                })
    
    area_df = pd.DataFrame(area_data)
    
    # Aggregate by Patient-Dataset-structure (sum areas)
    if not area_df.empty:
        aggregated_data = area_df.groupby(['Dataset', 'Patient', 'structure']).agg({
            'tissue_area_pixels': 'sum',
            'tissue_area_um2': 'sum'
        }).reset_index()
    else:
        # Fallback if no data calculated
        print("Warning: No tissue area data calculated. Creating default data.")
        unique_combinations = df.groupby(['Dataset', 'Patient', 'structure']).size().reset_index(name='cell_count')
        aggregated_data = unique_combinations.copy()
        aggregated_data['tissue_area_pixels'] = 1000000
        aggregated_data['tissue_area_um2'] = 1000000 * PIXEL_TO_UM2
    
    return aggregated_data


def analyze_cell_density_differences(file_path, output_dir='cell_density_difference_results', exclude_outside_tissue=False):
    """
    Analyzes cell densities and calculates differences between disease datasets and NK2 baseline.
    
    Cell density = cell_count / tissue_area_um2
    
    Args:
        file_path: Path to the CSV file containing the data
        output_dir: Directory where results will be saved
        exclude_outside_tissue: Whether to exclude 'outside the tissue' from heatmaps
    """
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)
    
    # Ensure required columns exist
    required_cols = ['Patient', 'Dataset', 'structure', 'MemCatName']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{col} column not found in the data. Please check your CSV file.")
    
    # Calculate tissue areas
    print("Calculating tissue areas...")
    tissue_data = calculate_tissue_areas(df)
    
    # Calculate cell counts by Patient-Dataset-structure-MemCatName
    print("Calculating cell counts and densities...")
    cell_counts = df.groupby(['Dataset', 'Patient', 'structure', 'MemCatName']).size().reset_index(name='count')
    
    # Merge with tissue area data
    density_data = pd.merge(cell_counts, tissue_data, on=['Dataset', 'Patient', 'structure'], how='left')
    
    # Handle cases where tissue area data might be missing
    if density_data['tissue_area_um2'].isna().any():
        print("Warning: Some Patient-Dataset-structure combinations missing tissue area data. Using default.")
        density_data['tissue_area_um2'] = density_data['tissue_area_um2'].fillna(1000000 * PIXEL_TO_UM2)
    
    # Calculate cell density (cells per µm²)
    density_data['cell_density'] = density_data['count'] / density_data['tissue_area_um2']
    
    # Convert to cells per mm² for more reasonable numbers
    density_data['cell_density_per_mm2'] = density_data['cell_density'] * 1e6  # 1 mm² = 1e6 µm²
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    detailed_csv = os.path.join(output_dir, 'cell_density_detailed.csv')
    print(f"Saving detailed results to {detailed_csv}...")
    density_data.to_csv(detailed_csv, index=False)
    
    # Calculate patient-level averages for each dataset-structure-celltype combination
    print("Calculating patient-level averages...")
    patient_averages = density_data.groupby(['Dataset', 'structure', 'MemCatName']).agg({
        'cell_density_per_mm2': ['mean', 'std', 'median', 'count'],
        'count': 'sum',
        'tissue_area_um2': 'mean'
    }).reset_index()
    
    # Flatten column names
    patient_averages.columns = ['_'.join(col).strip('_') for col in patient_averages.columns]
    patient_averages = patient_averages.rename(columns={
        'Dataset_': 'Dataset',
        'structure_': 'structure',
        'MemCatName_': 'MemCatName'
    })
    
    # Save patient averages
    averages_csv = os.path.join(output_dir, 'cell_density_patient_averages.csv')
    patient_averages.to_csv(averages_csv, index=False)
    
    # Create pivot tables for each dataset
    datasets = ['NK2', 'IgG4', 'AAV']
    dataset_pivots = {}
    
    for dataset in datasets:
        if dataset in patient_averages['Dataset'].values:
            print(f"Creating pivot table for {dataset}...")
            dataset_data = patient_averages[patient_averages['Dataset'] == dataset]
            
            pivot = dataset_data.pivot_table(
                values='cell_density_per_mm2_mean',
                index='MemCatName',
                columns='structure',
                fill_value=0
            )
            
            # Apply consistent ordering and filtering
            pivot = apply_ordering(pivot, exclude_outside_tissue=exclude_outside_tissue)
            
            dataset_pivots[dataset] = pivot
            
            # Save individual dataset pivot
            dataset_csv = os.path.join(output_dir, f'cell_density_{dataset}.csv')
            pivot.to_csv(dataset_csv)
    
    # Calculate difference matrices and proportional contributions if NK2 baseline exists
    proportional_data = {}
    if 'NK2' in dataset_pivots:
        nk2_baseline = dataset_pivots['NK2']
        
        # Calculate differences: Disease - NK2
        for disease in ['IgG4', 'AAV']:
            if disease in dataset_pivots:
                print(f"Calculating {disease} - NK2 differences...")
                
                disease_data = dataset_pivots[disease]
                
                # Align structures and cell types
                common_cells = set(nk2_baseline.index) & set(disease_data.index)
                common_structures = set(nk2_baseline.columns) & set(disease_data.columns)
                
                if common_structures and common_cells:
                    # Create aligned dataframes
                    nk2_aligned = nk2_baseline.loc[list(common_cells), list(common_structures)]
                    disease_aligned = disease_data.loc[list(common_cells), list(common_structures)]
                    
                    # Calculate difference
                    difference = disease_aligned - nk2_aligned
                    
                    # Apply consistent ordering to difference matrix
                    difference = apply_ordering(difference, exclude_outside_tissue=exclude_outside_tissue)
                    
                    # Save difference matrix
                    diff_csv = os.path.join(output_dir, f'cell_density_difference_{disease}_minus_NK2.csv')
                    difference.to_csv(diff_csv)
                    
                    print(f"Saved {disease} - NK2 difference matrix to {diff_csv}")
                    
                    # Store for visualization
                    dataset_pivots[f'{disease}_minus_NK2'] = difference
                    
                    # Calculate proportional contributions
                    print(f"Calculating proportional contributions for {disease}...")
                    proportional_contrib = calculate_proportional_contributions(difference)
                    
                    # Apply consistent ordering to proportional contributions
                    proportional_contrib = apply_ordering(proportional_contrib, exclude_outside_tissue=exclude_outside_tissue)
                    
                    # Filter out excluded cell types from proportional analysis
                    proportional_contrib_filtered = filter_cell_types_from_proportional(proportional_contrib)
                    
                    proportional_data[f'{disease}_proportion'] = proportional_contrib_filtered
                    
                    # Save proportional contribution matrix (filtered version)
                    prop_csv = os.path.join(output_dir, f'cell_density_proportion_{disease}_minus_NK2.csv')
                    proportional_contrib_filtered.to_csv(prop_csv)
                    
                    print(f"Saved {disease} proportional contributions to {prop_csv}")
    
    # Calculate global min/max for consistent scaling
    all_values = []
    for name, pivot in dataset_pivots.items():
        if not name.endswith('_minus_NK2'):  # Only use original datasets for scaling
            all_values.extend(pivot.values.flatten())
    
    if all_values:
        global_min = np.min(all_values)
        global_max = np.max(all_values)
        global_log_min = np.log10(global_min + 1) if global_min >= 0 else 0
        global_log_max = np.log10(global_max + 1)
        
        print(f"Global cell density range: {global_min:.4f} to {global_max:.4f} cells/mm²")
        print(f"Global log range: {global_log_min:.3f} to {global_log_max:.3f}")
    else:
        global_min, global_max = 0, 1
        global_log_min, global_log_max = 0, 1
    
    # Calculate difference scaling
    diff_values = []
    for name, pivot in dataset_pivots.items():
        if name.endswith('_minus_NK2'):
            diff_values.extend(pivot.values.flatten())
    
    if diff_values:
        diff_min = np.min(diff_values)
        diff_max = np.max(diff_values)
        diff_abs_max = max(abs(diff_min), abs(diff_max))
        print(f"Difference range: {diff_min:.4f} to {diff_max:.4f} cells/mm²")
    else:
        diff_abs_max = 1
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Individual dataset heatmaps
    for dataset_name, pivot in dataset_pivots.items():
        if not dataset_name.endswith('_minus_NK2'):
            print(f"Creating heatmap for {dataset_name}...")
            
            fig_width = max(8, len(pivot.columns) * 1.5)
            fig_height = max(12, min(30, len(pivot.index) * 1.0))
            plt.figure(figsize=(fig_width, fig_height))
            
            # Use log scale for better visualization
            pivot_log = np.log10(pivot + 1)
            
            sns.heatmap(pivot_log, annot=False, cmap='viridis', linewidths=0.5, linecolor='black',
                       cbar_kws={'shrink': 0.8, 'label': 'log10(Cell Density + 1) [cells/mm²]'},
                       vmin=global_log_min, vmax=global_log_max)
            
            plt.title(f'Cell Density in {dataset_name} Dataset (cells/mm²)', fontsize=16)
            plt.ylabel('Cell Type', fontsize=14)
            plt.xlabel('Structure', fontsize=14)
            plt.xticks(rotation=45, fontsize=12)
            plt.yticks(fontsize=10)
            plt.tight_layout()
            
            png_path = os.path.join(output_dir, f'cell_density_{dataset_name}.png')
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    # Difference heatmaps
    for dataset_name, pivot in dataset_pivots.items():
        if dataset_name.endswith('_minus_NK2'):
            print(f"Creating difference heatmap for {dataset_name}...")
            
            fig_width = max(8, len(pivot.columns) * 1.5)
            fig_height = max(12, min(30, len(pivot.index) * 1.0))
            plt.figure(figsize=(fig_width, fig_height))
            
            # Use diverging colormap for differences
            sns.heatmap(pivot, annot=False, cmap='RdBu_r', center=0, linewidths=0.5, linecolor='black',
                       cbar_kws={'shrink': 0.8, 'label': 'Difference in Cell Density [cells/mm²]'},
                       vmin=-diff_abs_max, vmax=diff_abs_max)
            
            plt.title(f'Cell Density Difference: {dataset_name.replace("_minus_", " - ")}', fontsize=16)
            plt.ylabel('Cell Type', fontsize=14)
            plt.xlabel('Structure', fontsize=14)
            plt.xticks(rotation=45, fontsize=12)
            plt.yticks(fontsize=10)
            plt.tight_layout()
            
            png_path = os.path.join(output_dir, f'cell_density_difference_{dataset_name}.png')
            plt.savefig(png_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    # Proportional contribution heatmaps (now filtered)
    print("Creating proportional contribution heatmaps...")
    for dataset_name, pivot in proportional_data.items():
        print(f"Creating proportional heatmap for {dataset_name} (filtered)...")
        
        fig_width = max(8, len(pivot.columns) * 1.5)
        fig_height = max(12, min(30, len(pivot.index) * 1.0))
        plt.figure(figsize=(fig_width, fig_height))
        
        # Use diverging colormap for proportions (values between -1 and 1)
        sns.heatmap(pivot, annot=False, cmap='RdBu_r', center=0, linewidths=0.5, linecolor='black',
                   cbar_kws={'shrink': 0.8, 'label': 'Proportional Contribution'},
                   vmin=-1, vmax=1)
        
        disease_name = dataset_name.replace('_proportion', '').replace('_minus_NK2', '')
        plt.title(f'Proportional Contribution by Structure: {disease_name} vs NK2\n'
                 f'(Structure Difference / Total Absolute Difference per Cell Type)\n'
                 f'Filtered: Excludes {len(EXCLUDE_FROM_PROPORTIONAL)} cell types', fontsize=16)
        plt.ylabel('Cell Type', fontsize=14)
        plt.xlabel('Structure', fontsize=14)
        plt.xticks(rotation=45, fontsize=12)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        
        png_path = os.path.join(output_dir, f'cell_density_proportion_{dataset_name}_filtered.png')
        plt.savefig(png_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved filtered proportional contribution heatmap: {png_path}")
    
    # Create combined proportional comparison if we have both diseases (filtered)
    if len(proportional_data) > 1:
        print("Creating combined proportional comparison (filtered)...")
        
        # Combine proportional data for comparison
        combined_prop_data = []
        for dataset_name, pivot in proportional_data.items():
            disease_name = dataset_name.replace('_proportion', '')
            for cell_type in pivot.index:
                for structure in pivot.columns:
                    combined_prop_data.append({
                        'Disease': disease_name,
                        'CellType': cell_type,
                        'Structure': structure,
                        'Proportion': pivot.loc[cell_type, structure]
                    })
        
        if combined_prop_data:
            combined_prop_df = pd.DataFrame(combined_prop_data)
            combined_prop_pivot = combined_prop_df.pivot_table(
                values='Proportion',
                index=['Disease', 'CellType'],
                columns='Structure',
                fill_value=0
            )
            
            # Apply structure filtering and ordering to columns
            combined_prop_pivot = filter_structures(combined_prop_pivot, exclude_outside_tissue)
            
            available_prop_structures = combined_prop_pivot.columns.tolist()
            structure_order_filtered = STRUCTURE_ORDER.copy()
            if exclude_outside_tissue and 'outside the tissue' in structure_order_filtered:
                structure_order_filtered.remove('outside the tissue')
            
            ordered_prop_structures = [struct for struct in structure_order_filtered if struct in available_prop_structures]
            ordered_prop_structures.extend([struct for struct in available_prop_structures if struct not in structure_order_filtered])
            combined_prop_pivot = combined_prop_pivot[ordered_prop_structures]
            
            # Apply cell type ordering within each disease
            all_prop_cell_types = [ct for (ds, ct) in combined_prop_pivot.index]
            available_prop_cells = list(set(all_prop_cell_types))
            # Filter out excluded cell types
            available_prop_cells = [cell for cell in available_prop_cells if cell not in EXCLUDE_FROM_PROPORTIONAL]
            ordered_prop_cells = [cell for cell in CELL_TYPE_ORDER if cell in available_prop_cells]
            ordered_prop_cells.extend([cell for cell in available_prop_cells if cell not in CELL_TYPE_ORDER])
            
            # Reorder the combined proportional pivot
            new_prop_index = []
            for disease in ['IgG4', 'AAV']:  # Only diseases, not NK2
                if disease in [ds for (ds, ct) in combined_prop_pivot.index]:
                    for cell_type in ordered_prop_cells:
                        if (disease, cell_type) in combined_prop_pivot.index:
                            new_prop_index.append((disease, cell_type))
            
            combined_prop_pivot = combined_prop_pivot.reindex(new_prop_index)
            
            # Save combined proportional data
            combined_prop_csv = os.path.join(output_dir, 'cell_density_proportions_combined_filtered.csv')
            combined_prop_pivot.to_csv(combined_prop_csv)
            
            # Create combined proportional heatmap
            fig_width = max(10, len(combined_prop_pivot.columns) * 1.2)
            fig_height = max(16, min(36, len(combined_prop_pivot.index) * 0.8))
            plt.figure(figsize=(fig_width, fig_height))
            
            sns.heatmap(combined_prop_pivot, annot=False, cmap='RdBu_r', center=0, linewidths=0.5, linecolor='black',
                       cbar_kws={'shrink': 0.8, 'label': 'Proportional Contribution'},
                       vmin=-1, vmax=1)
            
            plt.title('Proportional Contribution Comparison: Structure Differences as Fraction of Total\n'
                     f'(Red = Positive Contribution, Blue = Negative Contribution)\n'
                     f'Filtered: Excludes {len(EXCLUDE_FROM_PROPORTIONAL)} cell types', fontsize=16)
            plt.ylabel('Disease - Cell Type', fontsize=14)
            plt.xlabel('Structure', fontsize=14)
            plt.xticks(rotation=45, fontsize=12)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            
            combined_prop_png = os.path.join(output_dir, 'cell_density_proportions_combined_filtered.png')
            plt.savefig(combined_prop_png, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Saved combined filtered proportional heatmap: {combined_prop_png}")
    
    # Create combined comparison heatmap
    if len(dataset_pivots) > 1:
        print("Creating combined comparison heatmap...")
        
        # Combine all datasets for comparison
        combined_data = []
        for dataset_name, pivot in dataset_pivots.items():
            if not dataset_name.endswith('_minus_NK2'):
                for cell_type in pivot.index:
                    for structure in pivot.columns:
                        combined_data.append({
                            'Dataset': dataset_name,
                            'CellType': cell_type,
                            'Structure': structure,
                            'Density': pivot.loc[cell_type, structure]
                        })
        
        if combined_data:
            combined_df = pd.DataFrame(combined_data)
            combined_pivot = combined_df.pivot_table(
                values='Density',
                index=['Dataset', 'CellType'],
                columns='Structure',
                fill_value=0
            )
            
            # Apply structure filtering and ordering to columns
            combined_pivot = filter_structures(combined_pivot, exclude_outside_tissue)
            
            available_structures = combined_pivot.columns.tolist()
            structure_order_filtered = STRUCTURE_ORDER.copy()
            if exclude_outside_tissue and 'outside the tissue' in structure_order_filtered:
                structure_order_filtered.remove('outside the tissue')
            
            ordered_structures = [struct for struct in structure_order_filtered if struct in available_structures]
            ordered_structures.extend([struct for struct in available_structures if struct not in structure_order_filtered])
            combined_pivot = combined_pivot[ordered_structures]
            
            # Apply cell type ordering within each dataset
            # First get all unique cell types
            all_cell_types = [ct for (ds, ct) in combined_pivot.index]
            available_cells = list(set(all_cell_types))
            ordered_cells = [cell for cell in CELL_TYPE_ORDER if cell in available_cells]
            ordered_cells.extend([cell for cell in available_cells if cell not in CELL_TYPE_ORDER])
            
            # Reorder the combined pivot by creating new index order
            new_index = []
            for dataset in ['NK2', 'IgG4', 'AAV']:  # Maintain dataset order
                if dataset in [ds for (ds, ct) in combined_pivot.index]:
                    for cell_type in ordered_cells:
                        if (dataset, cell_type) in combined_pivot.index:
                            new_index.append((dataset, cell_type))
            
            combined_pivot = combined_pivot.reindex(new_index)
            
            # Save combined data
            combined_csv = os.path.join(output_dir, 'cell_density_combined.csv')
            combined_pivot.to_csv(combined_csv)
            
            # Create combined heatmap
            fig_width = max(10, len(combined_pivot.columns) * 1.2)
            fig_height = max(16, min(36, len(combined_pivot.index) * 0.8))
            plt.figure(figsize=(fig_width, fig_height))
            
            # Use log scale
            combined_log = np.log10(combined_pivot + 1)
            
            sns.heatmap(combined_log, annot=False, cmap='viridis', linewidths=0.5, linecolor='black',
                       cbar_kws={'shrink': 0.8, 'label': 'log10(Cell Density + 1) [cells/mm²]'},
                       vmin=global_log_min, vmax=global_log_max)
            
            plt.title('Cell Density Comparison Across Datasets and Cell Types', fontsize=16)
            plt.ylabel('Dataset - Cell Type', fontsize=14)
            plt.xlabel('Structure', fontsize=14)
            plt.xticks(rotation=45, fontsize=12)
            plt.yticks(fontsize=8)
            plt.tight_layout()
            
            combined_png = os.path.join(output_dir, 'cell_density_combined.png')
            plt.savefig(combined_png, dpi=300, bbox_inches='tight')
            plt.close()
    
    # Generate summary statistics including proportional analysis
    print("Generating summary statistics...")
    create_summary_statistics(density_data, output_dir, proportional_data, exclude_outside_tissue)
    
    print(f"\nCell density difference analysis complete!")
    print(f"Results saved to {output_dir} directory")
    print("\nGenerated files:")
    print("- cell_density_detailed.csv: Raw density calculations")
    print("- cell_density_patient_averages.csv: Patient-level averages")
    print("- cell_density_[dataset].csv: Density matrices for each dataset")
    print("- cell_density_difference_[disease]_minus_NK2.csv: Difference matrices")
    print("- cell_density_proportion_[disease]_minus_NK2.csv: Proportional contribution matrices (filtered)")
    print("- cell_density_proportions_combined_filtered.csv: Combined proportional analysis (filtered)")
    print("- Various .png files: Heatmap visualizations including filtered proportional contributions")
    print("- summary_statistics.csv: Top changes and statistics including proportional analysis")
    print(f"\nNote: Proportional heatmaps exclude {len(EXCLUDE_FROM_PROPORTIONAL)} cell types: {EXCLUDE_FROM_PROPORTIONAL}")


def create_summary_statistics(density_data, output_dir, proportional_data=None, exclude_outside_tissue=False):
    """
    Create summary statistics highlighting the most significant changes.
    """
    print("Creating summary statistics...")
    
    # Calculate dataset-level averages
    dataset_stats = density_data.groupby(['Dataset', 'structure', 'MemCatName']).agg({
        'cell_density_per_mm2': ['mean', 'std', 'count']
    }).reset_index()
    
    # Flatten column names
    dataset_stats.columns = ['_'.join(col).strip('_') for col in dataset_stats.columns]
    dataset_stats = dataset_stats.rename(columns={
        'Dataset_': 'Dataset',
        'structure_': 'structure',
        'MemCatName_': 'MemCatName'
    })
    
    # Filter out 'outside the tissue' if requested
    if exclude_outside_tissue:
        dataset_stats = dataset_stats[dataset_stats['structure'] != 'outside the tissue']
        print("  Excluded 'outside the tissue' from summary statistics")
    
    # Create summary for significant differences
    summary_data = []
    
    if 'NK2' in dataset_stats['Dataset'].values:
        nk2_data = dataset_stats[dataset_stats['Dataset'] == 'NK2'].copy()
        nk2_data = nk2_data.set_index(['structure', 'MemCatName'])
        
        for disease in ['IgG4', 'AAV']:
            if disease in dataset_stats['Dataset'].values:
                disease_data = dataset_stats[dataset_stats['Dataset'] == disease].copy()
                disease_data = disease_data.set_index(['structure', 'MemCatName'])
                
                # Find common indices
                common_indices = set(nk2_data.index) & set(disease_data.index)
                
                for (structure, cell_type) in common_indices:
                    nk2_mean = nk2_data.loc[(structure, cell_type), 'cell_density_per_mm2_mean']
                    disease_mean = disease_data.loc[(structure, cell_type), 'cell_density_per_mm2_mean']
                    
                    difference = disease_mean - nk2_mean
                    fold_change = disease_mean / nk2_mean if nk2_mean > 0 else np.inf
                    
                    # Get proportional contribution if available
                    proportional_contrib = 0
                    if proportional_data and f'{disease}_proportion' in proportional_data:
                        prop_matrix = proportional_data[f'{disease}_proportion']
                        if cell_type in prop_matrix.index and structure in prop_matrix.columns:
                            proportional_contrib = prop_matrix.loc[cell_type, structure]
                    
                    # Mark if cell type was excluded from proportional analysis
                    excluded_from_prop = cell_type in EXCLUDE_FROM_PROPORTIONAL
                    
                    summary_data.append({
                        'Comparison': f'{disease} vs NK2',
                        'Structure': structure,
                        'Cell_Type': cell_type,
                        'NK2_Density': nk2_mean,
                        'Disease_Density': disease_mean,
                        'Difference': difference,
                        'Fold_Change': fold_change,
                        'Percent_Change': ((disease_mean - nk2_mean) / nk2_mean * 100) if nk2_mean > 0 else 0,
                        'Proportional_Contribution': proportional_contrib,
                        'Excluded_From_Proportional': excluded_from_prop
                    })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by absolute difference to find most significant changes
        summary_df['Abs_Difference'] = abs(summary_df['Difference'])
        summary_df = summary_df.sort_values('Abs_Difference', ascending=False)
        
        # Save summary
        summary_csv = os.path.join(output_dir, 'summary_statistics.csv')
        summary_df.to_csv(summary_csv, index=False)
        
        # Create top changes summary
        top_increases = summary_df.nlargest(20, 'Difference')
        top_decreases = summary_df.nsmallest(20, 'Difference')
        
        print("\nTop 10 increases (Disease vs NK2):")
        for _, row in top_increases.head(10).iterrows():
            excluded_note = " [EXCLUDED]" if row['Excluded_From_Proportional'] else ""
            print(f"  {row['Comparison']}: {row['Cell_Type']}{excluded_note} in {row['Structure']} "
                  f"(+{row['Difference']:.4f} cells/mm², {row['Percent_Change']:.1f}%, "
                  f"prop: {row['Proportional_Contribution']:.3f})")
        
        print("\nTop 10 decreases (Disease vs NK2):")
        for _, row in top_decreases.head(10).iterrows():
            excluded_note = " [EXCLUDED]" if row['Excluded_From_Proportional'] else ""
            print(f"  {row['Comparison']}: {row['Cell_Type']}{excluded_note} in {row['Structure']} "
                  f"({row['Difference']:.4f} cells/mm², {row['Percent_Change']:.1f}%, "
                  f"prop: {row['Proportional_Contribution']:.3f})")
        
        # Create proportional contribution summary if available (only for non-excluded cell types)
        if proportional_data:
            print("\nTop structural contributors by proportional contribution (filtered):")
            
            # Find highest absolute proportional contributions among non-excluded cell types
            filtered_summary = summary_df[~summary_df['Excluded_From_Proportional']]
            if not filtered_summary.empty:
                top_proportional = filtered_summary.loc[filtered_summary['Proportional_Contribution'].abs().nlargest(10).index]
                
                for _, row in top_proportional.iterrows():
                    contribution_type = "positive" if row['Proportional_Contribution'] > 0 else "negative"
                    print(f"  {row['Comparison']}: {row['Cell_Type']} in {row['Structure']} "
                          f"({contribution_type} contrib: {row['Proportional_Contribution']:.3f}, "
                          f"diff: {row['Difference']:.4f} cells/mm²)")
        
        # Create cell type-specific proportional summary
        if proportional_data:
            create_proportional_summary(summary_df, output_dir)


def create_proportional_summary(summary_df, output_dir):
    """
    Create detailed proportional contribution summaries by cell type.
    """
    print("Creating proportional contribution summaries...")
    
    # Cell type-specific proportional analysis (only for non-excluded cell types)
    proportional_summaries = []
    
    # Filter out excluded cell types
    filtered_summary = summary_df[~summary_df['Excluded_From_Proportional']]
    
    for cell_type in filtered_summary['Cell_Type'].unique():
        cell_data = filtered_summary[filtered_summary['Cell_Type'] == cell_type].copy()
        
        for comparison in cell_data['Comparison'].unique():
            comp_data = cell_data[cell_data['Comparison'] == comparison]
            
            if not comp_data.empty:
                # Sort by proportional contribution
                comp_data_sorted = comp_data.sort_values('Proportional_Contribution', 
                                                       key=abs, ascending=False)
                
                # Get top contributing structures
                top_structures = comp_data_sorted.head(3)
                
                total_abs_diff = comp_data['Difference'].abs().sum()
                
                for i, (_, row) in enumerate(top_structures.iterrows(), 1):
                    proportional_summaries.append({
                        'Cell_Type': cell_type,
                        'Comparison': comparison,
                        'Rank': i,
                        'Structure': row['Structure'],
                        'Proportional_Contribution': row['Proportional_Contribution'],
                        'Difference': row['Difference'],
                        'Total_Cell_Type_Change': total_abs_diff
                    })
    
    if proportional_summaries:
        prop_summary_df = pd.DataFrame(proportional_summaries)
        prop_summary_csv = os.path.join(output_dir, 'proportional_contribution_summary_filtered.csv')
        prop_summary_df.to_csv(prop_summary_csv, index=False)
        print(f"Saved filtered proportional contribution summary to {prop_summary_csv}")


if __name__ == "__main__":
    # Set your input file path and output directory here
    input_path = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/combined_SAM_scores_filtered_alldataset5_10woMonostructure.csv"
    output_dir = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/cell_density_NK2normalization_structural_difference_analysis"
    
    # Run the analysis with option to exclude 'outside the tissue'
    # Set exclude_outside_tissue=True to exclude 'outside the tissue' from all heatmaps
    exclude_outside_tissue = True  # Change to True to exclude 'outside the tissue'
    
    analyze_cell_density_differences(input_path, output_dir, exclude_outside_tissue=exclude_outside_tissue)
