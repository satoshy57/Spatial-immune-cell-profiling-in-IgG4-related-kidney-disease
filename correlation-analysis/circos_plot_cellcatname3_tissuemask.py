#!/usr/bin/env python3
"""
Simplified Script: Circular Circos Plots for Disease-Specific Cell Co-occurrence
Creates circular Circos plots with ribbons for immune cell interactions
Only analyzes IgG4-RKD and AAV datasets (excludes KC and Overall)
No centrality analysis or network visualizations
Updated to use TISSUE masks for cell density calculations
REVISED: Updated cell categories and colors to match all_cell_types groupings
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from PIL import Image
import os
import warnings
from statsmodels.stats.multitest import multipletests
import matplotlib.patches as patches

warnings.filterwarnings('ignore')

# Ensure PIL can handle large images
Image.MAX_IMAGE_PIXELS = 1100000000

def calculate_tissue_area(dataset, comp_name, base_path):
    """
    Calculate the tissue area from tissue mask
    
    Args:
        dataset: Dataset name (IgG4-RKD, AAV, KC)
        comp_name: Component name (sample identifier)
        base_path: Base directory path
    Returns:
        tissue_area: Number of pixels in the tissue area
    """
    # Map dataset name for path construction
    original_dataset = dataset
    if dataset == "IgG4-RKD":
        original_dataset = "IgG4"
    elif dataset == "KC":
        original_dataset = "NK2"
    elif dataset == "AAV":
        original_dataset = "AAV"
    
    # Construct tissue mask paths
    tissue_mask_path1 = f'{base_path}/{original_dataset}/tissue_composite_masks/DAPI_UV_1_{original_dataset}_{comp_name}.tif'
    tissue_mask_path2 = f'{base_path}/{original_dataset}/tissue_composite_masks/DAPI_UV_2_{original_dataset}_{comp_name}.tif'
    tissue_mask_path1_alt = f'{base_path}/{original_dataset}/tissue_composite_masks/DAPI_UV_1_{original_dataset}_{comp_name}.tiff'
    tissue_mask_path2_alt = f'{base_path}/{original_dataset}/tissue_composite_masks/DAPI_UV_2_{original_dataset}_{comp_name}.tiff'
    
    tissue_mask_path = None
    for path in [tissue_mask_path1, tissue_mask_path2, tissue_mask_path1_alt, tissue_mask_path2_alt]:
        if os.path.exists(path):
            tissue_mask_path = path
            break
    
    if not tissue_mask_path:
        print(f"Tissue mask not found for {comp_name} in dataset {dataset}")
        return None
    
    try:
        tissue_mask = Image.open(tissue_mask_path)
        tissue_array = np.array(tissue_mask) > 0
        tissue_area = np.sum(tissue_array)
        
        if tissue_area == 0:
            print(f"Warning: No tissue pixels found in {comp_name}")
        
        return tissue_area
        
    except Exception as e:
        print(f"Error calculating tissue area for {comp_name}: {str(e)}")
        return None

def get_cell_counts(df, comp_name, category_column='CellCatName3', exclude_categories=None):
    """
    Get cell counts for each cell type in a specific component
    """
    comp_df = df[df['CompName'] == comp_name]
    
    if comp_df.empty:
        return {}
    
    if category_column not in comp_df.columns:
        print(f"Warning: {category_column} column not found in the dataframe")
        return {}
    
    cell_categories = comp_df[category_column].unique()
    
    if exclude_categories:
        cell_categories = [cat for cat in cell_categories if cat not in exclude_categories]
    
    counts = {}
    for cell_type in cell_categories:
        if pd.isna(cell_type):
            continue
        count = np.sum(comp_df[category_column] == cell_type)
        if count > 0:
            counts[cell_type] = count
    
    return counts

def calculate_cell_proportions(total_cell_counts):
    """Calculate cell proportions (cell number of each class/total cell number)"""
    total_cells = sum(total_cell_counts.values())
    proportions = {}
    for cell_type, count in total_cell_counts.items():
        proportions[cell_type] = count / total_cells if total_cells > 0 else 0
    return proportions

def calculate_densities(cell_counts, tissue_area, pixels_per_mm2):
    """Calculate cell densities (cells per mm² of tissue area)"""
    area_mm2 = tissue_area / pixels_per_mm2
    densities = {}
    for cell_type, count in cell_counts.items():
        densities[cell_type] = count / area_mm2 if area_mm2 > 0 else 0
    return densities

def load_and_prepare_data(file_path, base_path, all_cell_types, pixel_size_um=0.1507):
    """
    Load and prepare patient-level cell density and proportion data using tissue masks
    """
    pixel_size_mm = pixel_size_um * 0.001
    pixels_per_mm2 = 1 / (pixel_size_mm ** 2)
    
    print(f"Loading data from {file_path}...")
    print(f"Using pixel size: {pixel_size_um} μm ({pixel_size_mm} mm)")
    print(f"Pixels per mm²: {pixels_per_mm2:.0f}")
    
    df = pd.read_csv(file_path, low_memory=False)
    
    required_columns = ['CompName', 'CellCatName3', 'Patient', 'Dataset']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing required columns: {missing_columns}")
        return None, None
    
    # Create mapping of component names to datasets
    comp_to_dataset = {}
    for comp_name in df['CompName'].unique():
        dataset = df[df['CompName'] == comp_name]['Dataset'].iloc[0]
        if dataset == "IgG4":
            dataset = "IgG4-RKD"
        elif dataset == "NK2":
            dataset = "KC"
        comp_to_dataset[comp_name] = dataset
    
    # Aggregate data by patient
    print("Aggregating data by patient...")
    patient_data = []
    
    unique_patients = df['Patient'].unique()
    print(f"Found {len(unique_patients)} unique patients")
    
    for patient_id in unique_patients:
        print(f"Processing patient: {patient_id}")
        
        patient_df = df[df['Patient'] == patient_id]
        patient_components = patient_df['CompName'].unique()
        
        if len(patient_components) == 0:
            continue
        
        dataset = comp_to_dataset[patient_components[0]]
        
        total_cell_counts = {}
        total_tissue_area = 0
        valid_component_count = 0
        
        for comp_name in patient_components:
            try:
                tissue_area = calculate_tissue_area(dataset, comp_name, base_path)
                
                if tissue_area is None or tissue_area == 0:
                    print(f"Invalid tissue area for {comp_name}, skipping")
                    continue
                
                cell_counts = get_cell_counts(df, comp_name, category_column='CellCatName3')
                
                if not cell_counts:
                    print(f"No valid cell counts for {comp_name}, skipping")
                    continue
                
                for cell_type, count in cell_counts.items():
                    if cell_type in total_cell_counts:
                        total_cell_counts[cell_type] += count
                    else:
                        total_cell_counts[cell_type] = count
                
                total_tissue_area += tissue_area
                valid_component_count += 1
                
            except Exception as e:
                print(f"Error processing component {comp_name}: {str(e)}")
        
        if valid_component_count == 0 or total_tissue_area == 0:
            print(f"No valid components for patient {patient_id}, skipping")
            continue
        
        patient_densities = calculate_densities(total_cell_counts, total_tissue_area, pixels_per_mm2)
        patient_proportions = calculate_cell_proportions(total_cell_counts)
        
        patient_record = {
            'PatientID': patient_id,
            'Dataset': dataset,
            'ComponentCount': valid_component_count,
            'TotalTissueArea': total_tissue_area
        }
        
        for cell_type in all_cell_types:
            patient_record[f"{cell_type}_density"] = patient_densities.get(cell_type, 0)
            patient_record[f"{cell_type}_proportion"] = patient_proportions.get(cell_type, 0)
            patient_record[f"{cell_type}_count"] = total_cell_counts.get(cell_type, 0)
        
        patient_data.append(patient_record)
    
    if not patient_data:
        print("No patients were successfully processed")
        return None, None
    
    patient_df = pd.DataFrame(patient_data)
    
    # Create separate DataFrames for density and proportion analysis
    density_df = pd.DataFrame()
    density_df['PatientID'] = patient_df['PatientID']
    density_df['Dataset'] = patient_df['Dataset']
    
    proportion_df = pd.DataFrame()
    proportion_df['PatientID'] = patient_df['PatientID']
    proportion_df['Dataset'] = patient_df['Dataset']
    
    for cell_type in all_cell_types:
        density_df[cell_type] = patient_df[f"{cell_type}_density"]
        proportion_df[cell_type] = patient_df[f"{cell_type}_proportion"]
    
    return density_df, proportion_df

def compute_correlation_matrix(df, all_cell_types, method='spearman'):
    """
    Compute cell-to-cell correlation matrix with statistical testing
    """
    # Ensure all cell types are present
    for cell_type in all_cell_types:
        if cell_type not in df.columns:
            df[cell_type] = 0.0
    
    cell_data = df[all_cell_types]
    
    if len(cell_data) < 3:
        print(f"Not enough samples ({len(cell_data)}) for correlation analysis")
        return None, None, None
    
    n_types = len(all_cell_types)
    corr_matrix = np.zeros((n_types, n_types))
    pval_matrix = np.zeros((n_types, n_types))
    
    for i, cell1 in enumerate(all_cell_types):
        for j, cell2 in enumerate(all_cell_types):
            if i == j:
                corr_matrix[i, j] = 1.0
                pval_matrix[i, j] = 0.0
            else:
                try:
                    data1 = cell_data[cell1].dropna()
                    data2 = cell_data[cell2].dropna()
                    
                    common_idx = data1.index.intersection(data2.index)
                    if len(common_idx) < 3:
                        corr_matrix[i, j] = np.nan
                        pval_matrix[i, j] = np.nan
                        continue
                    
                    corr, pval = spearmanr(data1.loc[common_idx], data2.loc[common_idx])
                    corr_matrix[i, j] = corr
                    pval_matrix[i, j] = pval
                    
                except Exception as e:
                    print(f"Error calculating correlation between {cell1} and {cell2}: {str(e)}")
                    corr_matrix[i, j] = np.nan
                    pval_matrix[i, j] = np.nan
    
    # Multiple testing correction
    valid_pvals = pval_matrix[~np.isnan(pval_matrix)]
    adj_pval_matrix = np.full_like(pval_matrix, np.nan)
    
    if len(valid_pvals) > 0:
        try:
            valid_indices = []
            valid_pvals_list = []
            
            for i in range(n_types):
                for j in range(n_types):
                    if not np.isnan(pval_matrix[i, j]) and i != j:
                        valid_indices.append((i, j))
                        valid_pvals_list.append(pval_matrix[i, j])
            
            if valid_pvals_list:
                _, adj_pvals, _, _ = multipletests(valid_pvals_list, alpha=0.05, method='fdr_bh')
                
                for idx, (i, j) in enumerate(valid_indices):
                    adj_pval_matrix[i, j] = adj_pvals[idx]
                
                for i in range(n_types):
                    adj_pval_matrix[i, i] = 0.0
                    
                print(f"Applied FDR correction to {len(valid_pvals_list)} p-values")
        except Exception as e:
            print(f"Error in multiple testing correction: {str(e)}")
            adj_pval_matrix = pval_matrix.copy()
    
    # Convert to DataFrames
    corr_df = pd.DataFrame(corr_matrix, index=all_cell_types, columns=all_cell_types)
    pval_df = pd.DataFrame(pval_matrix, index=all_cell_types, columns=all_cell_types)
    adj_pval_df = pd.DataFrame(adj_pval_matrix, index=all_cell_types, columns=all_cell_types)
    
    return corr_df, pval_df, adj_pval_df

def assign_cell_categories(cell_types):
    """
    Assign cell types to functional categories matching all_cell_types groupings
    
    Categories:
    - CD4T: CD4+ T cells
    - CD8T: CD8+ T cells
    - Other_Lymphoid: γδT, ILC, NK, NKT cells
    - Humoral: B cells, Plasma cells, Plasmablasts
    - Myeloid: Macrophages, DCs, Neutrophils
    - Stromal: Myofibroblasts, Smooth Muscle, Tubular, etc.
    """
    categories = {}
    colors = {
        'CD4T': '#4169E1',           # Blue
        'CD8T': '#FF8E8E',           # Light red
        'Other_Lymphoid': '#9370DB', # Purple
        'Humoral': '#52C41A',        # Green
        'Myeloid': '#FFA500',        # Orange
        'Stromal': '#A9A9A9'         # Gray
    }
    
    # CD4T cell types - based on all_cell_types grouping
    cd4t_types = [
        'NaiveCD4T', 'OtherCD4T', 'CD4Treg', 'Th1', 'MKI+CD4T',
        'CD4Tcm-Tem', 'CD4+iTreg', 'Th1-EM',
        'ICOS+Treg', 'CD4Trm', 'Th1-RM',
        'CD103+CD4T', 'CD103+CD4Treg', 'CD103+Th1-like', 'ProliferatingCD103+CD4T',
        'HLAII+CD27+CD4T', 'HLAII+CD27+Treg', 'HLAII+CD27+Th1-like', 'ProliferatingHLAII+CD27+CD4T',
        'Tfh', 'Tfr', 'Tfh1', 'MKI+Tfh',
        'TIGIT+Tfh', 'TIGIT+Tfr', 'TIGIT+Tfh1', 'MKI+TIGIT+Tfh',
        'HLAII+GZMK+Tfh', 'GZMK+CD4Treg', 'GZMK+Th1-EM', 'MKI+HLAII+GZMK+Tfh',
        'GZMA+GZMK+CD4CTL', 'GZMA+GZMK+CD4Treg', 'GZMA+GZMK+Th1-EM', 'MKI+GZMA+GZMK+CD4CTL',
        'GZMB+CD4CTL', 'GZMB+CD4Treg', 'GZMB+Th1-EM', 'MKI+GZMB+CD4CTL'
    ]
    
    # CD8T cell types
    cd8t_types = [
        'NaiveCD8T', 'otherCD8T', 'CD8Treg', 'Tc1', 'MKI+CD8T',
        'GZMA+GZMK+CD8CTL', 'GZMA+GZMK+CD8Treg', 'GZMA+GZMK+Tc1-EM', 'MKI+GZMA+GZMK+CD8CTL',
        'GZMB+CD8CTL', 'GZMB+CD8Treg', 'GZMB+Tc1-EM', 'MKI+GZMB+CD8CTL',
        'CD103+CD8Treg', 'CD103+Tc1-like', 'ProliferatingCD103+CD8Treg',
        'CD8Trm', 'CD69+CD8Treg', 'Tbet+CD8Trm', 'ProliferatingCD8Trm',
        'CD103+CD8Trm', 'CD8Trm-Treg', 'Tbet+CD103+CD8Trm', 'ProliferatingCD103+CD8Trm',
        'TIGIT+CD8T', 'TIGIT+CD8Treg', 'TIGIT+Tc1', 'MKI+TIGIT+CD8T',
        'CD8Tcm-Tem', 'CD8+iTreg', 'Tc1-EM', 'ProliferatingCD8Tcm-Tem'
    ]
    
    # Other Lymphoid cell types
    other_lymphoid_types = [
        'γδT', 'γδTreg', 'γδT1', 'MKI+γδT',
        'ILC', 'ImmatureNK', 'MatureNK', 'NKT'
    ]
    
    # Humoral cell types (B cells and Plasma cells)
    humoral_types = [
        'NaiveB', 'TransitionalB', 'otherB', 'Breg', 'MKI+B',
        'CD27+B', 'GCB', 'MemoryB', 'Tbet+MemoryB',
        'DN2B', 'DN3B', 'CD11c+B', 'AgeAsscB', 'CD11c+CD27+B',
        'Plasmablast', 'IgG4+Plasmablast', 'Plasma', 'IgG4+Plasma'
    ]
    
    # Myeloid cell types
    myeloid_types = [
        'M1Macrophage', 'MPO+Macrophage', 'M2Macrophage',
        'MerTK+M1Macrophage', 'MerTK+M2Macrophage', 'GPNMB+M1Macrophage',
        'GPNMB+M2Macrophage', 'cDC1', 'cDC2', 'pDC', 'Neutrophil'
    ]
    
    # Stromal and structural cell types
    stromal_types = [
        'Myofibroblast-SmoothMuscle', 'CD90+Myofibroblast', 'MMT',
        'StressedTubular', 'Apoptotic'
    ]
    
    # Assign categories based on exact matches
    for cell in cell_types:
        if cell in cd4t_types:
            categories[cell] = 'CD4T'
        elif cell in cd8t_types:
            categories[cell] = 'CD8T'
        elif cell in other_lymphoid_types:
            categories[cell] = 'Other_Lymphoid'
        elif cell in humoral_types:
            categories[cell] = 'Humoral'
        elif cell in myeloid_types:
            categories[cell] = 'Myeloid'
        elif cell in stromal_types:
            categories[cell] = 'Stromal'
        else:
            # Fallback to keyword matching if exact match fails
            cell_lower = cell.lower()
            if 'cd4' in cell_lower or 'th1' in cell_lower or 'th2' in cell_lower or 'tfh' in cell_lower or 'tfr' in cell_lower or 'treg' in cell_lower:
                categories[cell] = 'CD4T'
            elif 'cd8' in cell_lower or 'tc1' in cell_lower or 'trm' in cell_lower:
                categories[cell] = 'CD8T'
            elif any(x in cell_lower for x in ['plasma', 'plasmablast', 'naiveb', 'otherb', 'breg', 'transitionalb', 'memoryb', 'gcb', 'cd27+b', 'dn2b', 'dn3b', 'cd11c+b', 'ageasscb']):
                categories[cell] = 'Humoral'
            elif any(x in cell_lower for x in ['macrophage', 'monocyte', 'neutrophil', 'eosinophil', 'cdc1', 'cdc2', 'pdc', 'm1', 'm2', 'mertk', 'gpnmb', 'mpo']):
                categories[cell] = 'Myeloid'
            elif any(x in cell_lower for x in ['nk', 'nkt', 'ilc', 'γδt']):
                categories[cell] = 'Other_Lymphoid'
            elif any(x in cell_lower for x in ['myofibroblast', 'mmt', 'cd90', 'tubular', 'stressed', 'apoptotic']):
                categories[cell] = 'Stromal'
            else:
                print(f"Warning: Cell type '{cell}' not categorized, defaulting to Stromal")
                categories[cell] = 'Stromal'
    
    return categories, colors

def get_connection_color(cell1, cell2, categories, colors):
    """
    Determine connection color based on hierarchical priority of cell categories
    """
    cat1 = categories.get(cell1, 'Stromal')
    cat2 = categories.get(cell2, 'Stromal')
    
    # Priority order for color selection
    priority_order = [
        'CD4T',
        'CD8T',
        'Humoral',
        'Other_Lymphoid',
        'Myeloid',
        'Stromal'
    ]
    
    categories_involved = {cat1, cat2}
    
    for priority_cat in priority_order:
        if priority_cat in categories_involved:
            return colors.get(priority_cat, colors['Stromal'])
    
    return colors['Stromal']

def create_improved_circular_circos_with_visible_segments(corr_matrix, pval_matrix, categories, colors, 
                                                         threshold=0.4, sig_threshold=0.05, title_suffix="", 
                                                         figsize=(20, 20)):
    """
    Create circular Circos plot with visible segments and black-bordered ribbons
    """
    
    # Filter for significant correlations above threshold
    significant_pairs = []
    cell_types = corr_matrix.columns.tolist()
    
    for i, cell1 in enumerate(cell_types):
        for j, cell2 in enumerate(cell_types):
            if i < j:  # Avoid duplicates
                corr_val = corr_matrix.loc[cell1, cell2]
                pval = pval_matrix.loc[cell1, cell2]
                
                if (not pd.isna(corr_val) and not pd.isna(pval) and 
                    abs(corr_val) >= threshold and pval <= sig_threshold):
                    significant_pairs.append({
                        'cell1': cell1,
                        'cell2': cell2,
                        'correlation': corr_val,
                        'pvalue': pval,
                        'abs_corr': abs(corr_val)
                    })
    
    if not significant_pairs:
        print("No significant correlations found for Circos plot")
        return None
    
    # Get unique cell types involved in significant correlations
    involved_cells_set = set()
    for pair in significant_pairs:
        involved_cells_set.add(pair['cell1'])
        involved_cells_set.add(pair['cell2'])
    
    if len(involved_cells_set) == 0:
        return None
    
    # Group cells by category and sort
    category_order = ['CD4T', 'CD8T', 'Other_Lymphoid', 'Humoral', 'Myeloid', 'Stromal']
    
    involved_cells_by_category = {}
    for cell in involved_cells_set:
        cat = categories.get(cell, 'Stromal')
        if cat not in involved_cells_by_category:
            involved_cells_by_category[cat] = []
        involved_cells_by_category[cat].append(cell)
    
    # Sort cells within each category and arrange by category order
    involved_cells = []
    for cat in category_order:
        if cat in involved_cells_by_category:
            sorted_cells = sorted(involved_cells_by_category[cat])
            involved_cells.extend(sorted_cells)
    
    n_cells = len(involved_cells)
    print(f"Creating Circos plot with {n_cells} cell types and {len(significant_pairs)} connections")
    
    # Create figure with regular axes (not polar)
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect('equal')
    
    # Define circle parameters
    center = (0, 0)
    radius_outer = 1.0
    radius_inner = 0.85
    segment_thickness = radius_outer - radius_inner
    
    print(f"Segment parameters: inner={radius_inner}, outer={radius_outer}, thickness={segment_thickness}")
    
    # Calculate positions for each cell type around the circle
    angles = np.linspace(0, 2*np.pi, n_cells, endpoint=False)
    
    # Create cell position mapping
    cell_info = {}
    base_segment_width = 2 * np.pi / n_cells
    
    for i, cell in enumerate(involved_cells):
        angle = angles[i]
        cell_info[cell] = {
            'angle': angle,
            'index': i,
            'x_outer': radius_outer * np.cos(angle),
            'y_outer': radius_outer * np.sin(angle),
            'x_inner': radius_inner * np.cos(angle),
            'y_inner': radius_inner * np.sin(angle)
        }
    
    # DRAW COLORED SEGMENTS FIRST (so ribbons appear on top)
    print(f"Drawing {n_cells} colored cell category segments...")
    
    for i, cell in enumerate(involved_cells):
        cat = categories.get(cell, 'Stromal')
        color = colors.get(cat, colors['Stromal'])
        
        angle = angles[i]
        
        # Calculate segment angular width (leave small gaps between segments)
        segment_angular_width = base_segment_width * 0.9  # 90% of available space
        start_angle = angle - segment_angular_width/2
        end_angle = angle + segment_angular_width/2
        
        # Convert to degrees for matplotlib
        start_angle_deg = np.degrees(start_angle)
        end_angle_deg = np.degrees(end_angle)
        
        # Ensure angle is in correct range
        if end_angle_deg < start_angle_deg:
            end_angle_deg += 360
        
        # Create visible wedge segment
        wedge = patches.Wedge(
            center,                    # Center point (0, 0)
            radius_outer,              # Outer radius
            start_angle_deg,           # Start angle in degrees  
            end_angle_deg,             # End angle in degrees
            width=segment_thickness,   # Radial width of segment
            facecolor=color,           # Category color
            alpha=0.9,                 # High opacity
            edgecolor='black',         # Black border
            linewidth=1.0,             # Border width
            zorder=1                   # Behind ribbons
        )
        
        ax.add_patch(wedge)
        
        # Debug output for first few segments
        if i < 3:
            print(f"  Segment {i}: {cell} ({cat}) -> color {color}")
            print(f"    Angle range: {start_angle_deg:.1f}° to {end_angle_deg:.1f}°")
    
    # Add cell type labels positioned much further outside to avoid overlap
    label_radius = 1.55  # Even further from segments to avoid overlap
    
    print("Adding cell type labels...")
    for cell in involved_cells:
        angle = cell_info[cell]['angle']
        x_label = label_radius * np.cos(angle)
        y_label = label_radius * np.sin(angle)
        
        # Determine text alignment and rotation for readability
        if -np.pi/2 <= angle <= np.pi/2:
            rotation = np.degrees(angle)
            ha = 'left'
        else:
            rotation = np.degrees(angle) + 180
            ha = 'right'
        
        ax.text(x_label, y_label, cell,
               rotation=rotation, ha=ha, va='center',
               fontsize=10, fontweight='bold', color='black')
    
    # Organize connections by angular distance to minimize overlaps
    def angular_distance(cell1, cell2):
        angle1 = cell_info[cell1]['angle']
        angle2 = cell_info[cell2]['angle']
        diff = abs(angle2 - angle1)
        return min(diff, 2*np.pi - diff)
    
    # Group connections by distance to create layers
    short_distance = []
    medium_distance = []
    long_distance = []
    
    for pair in significant_pairs:
        dist = angular_distance(pair['cell1'], pair['cell2'])
        if dist < np.pi/3:
            short_distance.append(pair)
        elif dist < 2*np.pi/3:
            medium_distance.append(pair)
        else:
            long_distance.append(pair)
    
    # Sort each group by correlation strength
    short_distance.sort(key=lambda x: x['abs_corr'], reverse=True)
    medium_distance.sort(key=lambda x: x['abs_corr'], reverse=True)
    long_distance.sort(key=lambda x: x['abs_corr'], reverse=True)
    
    # Draw ribbons in layers with different radial depths
    positive_count = 0
    negative_count = 0
    
    layer_depths = {
        'short': 0.7,
        'medium': 0.5,  
        'long': 0.3
    }
    
    connection_groups = [
        ('short', short_distance),
        ('medium', medium_distance), 
        ('long', long_distance)
    ]
    
    print(f"Drawing ribbons in {len(connection_groups)} layers...")
    
    for layer_name, pairs in connection_groups:
        layer_depth = layer_depths[layer_name]
        
        for pair in pairs:
            cell1 = pair['cell1']
            cell2 = pair['cell2']
            corr_val = pair['correlation']
            abs_corr = pair['abs_corr']
            
            # Get connection color based on category hierarchy
            connection_color = get_connection_color(cell1, cell2, categories, colors)
            
            # Count positive/negative
            if corr_val > 0:
                positive_count += 1
            else:
                negative_count += 1
            
            # Get cell positions
            angle1 = cell_info[cell1]['angle']
            angle2 = cell_info[cell2]['angle']
            
            # Calculate ribbon width based on correlation strength
            base_width = 0.02
            max_width = 0.08
            ribbon_width = base_width + (max_width - base_width) * abs_corr
            
            # Create smooth Bezier curve ribbon
            n_points = 100
            t = np.linspace(0, 1, n_points)
            
            # Control points for the ribbon path
            start_x = radius_inner * np.cos(angle1)
            start_y = radius_inner * np.sin(angle1)
            end_x = radius_inner * np.cos(angle2)
            end_y = radius_inner * np.sin(angle2)
            
            # Control point depth based on layer and correlation strength
            control_radius = layer_depth * (0.6 + 0.4 * abs_corr)
            
            # Calculate control point (center of the curve)
            mid_angle = (angle1 + angle2) / 2
            # Handle angle wrapping
            if abs(angle2 - angle1) > np.pi:
                mid_angle += np.pi
            
            control_x = control_radius * np.cos(mid_angle)
            control_y = control_radius * np.sin(mid_angle)
            
            # Generate ribbon path using quadratic Bezier curve
            ribbon_center_x = []
            ribbon_center_y = []
            
            for t_val in t:
                # Quadratic Bezier curve formula
                x = (1-t_val)**2 * start_x + 2*(1-t_val)*t_val * control_x + t_val**2 * end_x
                y = (1-t_val)**2 * start_y + 2*(1-t_val)*t_val * control_y + t_val**2 * end_y
                ribbon_center_x.append(x)
                ribbon_center_y.append(y)
            
            # Create ribbon edges by offsetting the center path
            ribbon_edge1_x = []
            ribbon_edge1_y = []
            ribbon_edge2_x = []
            ribbon_edge2_y = []
            
            for i in range(len(ribbon_center_x)):
                if i == 0:
                    dx = ribbon_center_x[1] - ribbon_center_x[0]
                    dy = ribbon_center_y[1] - ribbon_center_y[0]
                elif i == len(ribbon_center_x) - 1:
                    dx = ribbon_center_x[i] - ribbon_center_x[i-1]
                    dy = ribbon_center_y[i] - ribbon_center_y[i-1]
                else:
                    dx = ribbon_center_x[i+1] - ribbon_center_x[i-1]
                    dy = ribbon_center_y[i+1] - ribbon_center_y[i-1]
                
                # Normalize direction vector
                length = np.sqrt(dx**2 + dy**2)
                if length > 0:
                    dx /= length
                    dy /= length
                
                # Calculate perpendicular direction for ribbon width
                perp_x = -dy
                perp_y = dx
                
                # Taper the ribbon
                taper = 1 - 0.3 * (2 * abs(i / (len(ribbon_center_x) - 1) - 0.5))
                current_width = ribbon_width * taper
                
                # Calculate ribbon edges
                offset = current_width / 2
                ribbon_edge1_x.append(ribbon_center_x[i] + perp_x * offset)
                ribbon_edge1_y.append(ribbon_center_y[i] + perp_y * offset)
                ribbon_edge2_x.append(ribbon_center_x[i] - perp_x * offset)
                ribbon_edge2_y.append(ribbon_center_y[i] - perp_y * offset)
            
            # Create closed polygon for the ribbon
            polygon_x = ribbon_edge1_x + ribbon_edge2_x[::-1]
            polygon_y = ribbon_edge1_y + ribbon_edge2_y[::-1]
            
            # Set ribbon appearance
            ribbon_alpha = 0.7 + 0.2 * abs_corr
            
            if corr_val > 0:
                # Solid ribbon for positive correlations with black border
                ribbon_patch = patches.Polygon(list(zip(polygon_x, polygon_y)), 
                                             facecolor=connection_color, 
                                             alpha=ribbon_alpha,
                                             edgecolor='black',  # Black border for visibility
                                             linewidth=1.0,     # Visible border width
                                             zorder=2)          # Above segments
            else:
                # Hatched pattern for negative correlations with black border
                ribbon_patch = patches.Polygon(list(zip(polygon_x, polygon_y)), 
                                             facecolor=connection_color, 
                                             alpha=ribbon_alpha * 0.8,
                                             edgecolor='black',  # Black border for visibility
                                             linewidth=1.2,     # Slightly thicker for hatched
                                             hatch='///',
                                             zorder=2)          # Above segments
            
            ax.add_patch(ribbon_patch)
    
    # Configure plot appearance with extended space for farther labels
    ax.set_xlim(-1.8, 1.8)  # Extended limits to accommodate farther labels
    ax.set_ylim(-1.8, 1.8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Count connections by grade
    thick_count = sum(1 for pair in significant_pairs if pair['abs_corr'] >= 0.8)
    medium_count = sum(1 for pair in significant_pairs if 0.6 <= pair['abs_corr'] < 0.8)
    thin_count = sum(1 for pair in significant_pairs if 0.4 <= pair['abs_corr'] < 0.6)
    
    # Add title
    plt.suptitle(f'Cell Co-occurrence Circular Circos Plot{title_suffix}\n'
                 f'Perfect circle with colored segments and black-bordered ribbons\n'
                 f'Close (n={len(short_distance)}), Medium (n={len(medium_distance)}), Far (n={len(long_distance)}) connections\n'
                 f'Solid (positive, n={positive_count}), Hatched (negative, n={negative_count})',
                 fontsize=14, fontweight='bold', y=0.95)
    
    # Create legend
    legend_elements = []
    
    # Category information  
    used_categories = []
    for cat in category_order:
        if cat in involved_cells_by_category:
            used_categories.append(cat)
            color = colors.get(cat, colors['Stromal'])
            cell_count = len(involved_cells_by_category[cat])
            # Format category name for display
            cat_display = cat.replace('_', ' ')
            legend_elements.append(patches.Patch(color=color, 
                                               label=f'{cat_display} ({cell_count})'))
    
    # Position legend
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), 
             fontsize=10)
    
    plt.tight_layout()
    
    print(f"Circos plot completed with {n_cells} visible colored segments and {len(significant_pairs)} ribbons")
    
    return fig

def analyze_dataset_circos_only(density_df, proportion_df, all_cell_types, save_path):
    """
    Perform Circos plot analysis for IgG4-RKD and AAV datasets only
    """
    print("\n=== CIRCOS PLOT ANALYSIS FOR IgG4-RKD AND AAV ONLY ===")
    
    datasets = density_df['Dataset'].unique()
    # Filter to only include IgG4-RKD and AAV
    target_datasets = [d for d in datasets if d in ['IgG4-RKD', 'AAV']]
    print(f"Target datasets: {target_datasets}")
    
    # Analysis types to run
    analysis_types = [
        ('density', density_df, 'Cell Density (Tissue Area)'),
        ('proportion', proportion_df, 'Cell Proportion')
    ]
    
    for data_type, data_df, data_label in analysis_types:
        print(f"\n🔬 Analyzing {data_label} Circos plots...")
        
        for dataset in target_datasets:
            print(f"\n🌐 Creating {dataset} {data_label.lower()} Circos plot...")
            
            # Filter data for this dataset
            dataset_data = data_df[data_df['Dataset'] == dataset]
            
            if len(dataset_data) < 3:
                print(f"Not enough samples in {dataset} dataset ({len(dataset_data)})")
                continue
            
            # Compute correlations
            corr_df, pval_df, adj_pval_df = compute_correlation_matrix(
                dataset_data, all_cell_types)
            
            if corr_df is not None:
                # Assign categories and colors
                categories, colors = assign_cell_categories(all_cell_types)
                
                # Create Circos plot
                print(f"Creating Circos plot for {dataset} {data_label.lower()}...")
                fig = create_improved_circular_circos_with_visible_segments(
                    corr_df, adj_pval_df, categories, colors,
                    threshold=0.4, sig_threshold=0.05,
                    title_suffix=f" - {dataset} ({data_label})")
                
                if fig:
                    plt.savefig(os.path.join(save_path, f"{dataset}_{data_type}_circos_plot.png"), 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"✅ Saved: {dataset}_{data_type}_circos_plot.png")
                
                # Save correlation data
                corr_df.to_csv(os.path.join(save_path, f'{dataset}_{data_type}_correlations.csv'))
                adj_pval_df.to_csv(os.path.join(save_path, f'{dataset}_{data_type}_pvalues.csv'))
                print(f"✅ Saved correlation data for {dataset} {data_type}")

def main():
    """
    Main analysis pipeline - Circos plots only for IgG4-RKD and AAV
    """
    print("🕸️  Simplified Cell Co-occurrence Analysis - Circos Plots Only")
    print("=" * 80)
    
    # Define paths - UPDATE THESE PATHS
    data_path = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/processed_combined_SAM_scores_filtered_alldataset5_10woMonointerstitial_memcatname3cellcatname23.csv"
    base_path = "/project/mclark/SCAMPI_datasets"
    save_path = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/circos_analysis_IgG4_AAV_cellcatname3_tissue_area"
    
    # Create output directory
    os.makedirs(save_path, exist_ok=True)
    
    # Define comprehensive cell types based on CellCatName3
    all_cell_types = [
        # CD4T cell
        'NaiveCD4T', 'OtherCD4T', 'CD4Treg', 'Th1', 'MKI+CD4T',
        'CD4Tcm-Tem', 'CD4+iTreg', 'Th1-EM',
        'ICOS+Treg', 'CD4Trm', 'Th1-RM',
        'CD103+CD4T', 'CD103+CD4Treg', 'CD103+Th1-like', 'ProliferatingCD103+CD4T',
        'HLAII+CD27+CD4T', 'HLAII+CD27+Treg', 'HLAII+CD27+Th1-like', 'ProliferatingHLAII+CD27+CD4T',
        'Tfh', 'Tfr', 'Tfh1', 'MKI+Tfh',
        'TIGIT+Tfh', 'TIGIT+Tfr', 'TIGIT+Tfh1', 'MKI+TIGIT+Tfh',
        'HLAII+GZMK+Tfh', 'GZMK+CD4Treg', 'GZMK+Th1-EM', 'MKI+HLAII+GZMK+Tfh',
        'GZMA+GZMK+CD4CTL', 'GZMA+GZMK+CD4Treg', 'GZMA+GZMK+Th1-EM', 'MKI+GZMA+GZMK+CD4CTL',
        'GZMB+CD4CTL', 'GZMB+CD4Treg', 'GZMB+Th1-EM', 'MKI+GZMB+CD4CTL',
        
        # CD8T cell
        'NaiveCD8T', 'otherCD8T', 'CD8Treg', 'Tc1', 'MKI+CD8T',
        'GZMA+GZMK+CD8CTL', 'GZMA+GZMK+CD8Treg', 'GZMA+GZMK+Tc1-EM', 'MKI+GZMA+GZMK+CD8CTL',
        'GZMB+CD8CTL', 'GZMB+CD8Treg', 'GZMB+Tc1-EM', 'MKI+GZMB+CD8CTL',
        'CD103+CD8Treg', 'CD103+Tc1-like', 'ProliferatingCD103+CD8Treg',
        'CD8Trm', 'CD69+CD8Treg', 'Tbet+CD8Trm', 'ProliferatingCD8Trm',
        'CD103+CD8Trm', 'CD8Trm-Treg', 'Tbet+CD103+CD8Trm', 'ProliferatingCD103+CD8Trm',
        'TIGIT+CD8T', 'TIGIT+CD8Treg', 'TIGIT+Tc1', 'MKI+TIGIT+CD8T',
        'CD8Tcm-Tem', 'CD8+iTreg', 'Tc1-EM', 'ProliferatingCD8Tcm-Tem',
        
        # Other Lymphoid cell
        'γδT', 'γδTreg', 'γδT1', 'MKI+γδT',
        'ILC', 'ImmatureNK', 'MatureNK', 'NKT',
        
        # Humoral cell
        'NaiveB', 'TransitionalB', 'otherB', 'Breg', 'MKI+B',
        'CD27+B', 'GCB', 'MemoryB', 'Tbet+MemoryB',
        'DN2B', 'DN3B', 'CD11c+B', 'AgeAsscB', 'CD11c+CD27+B',
        'Plasmablast', 'IgG4+Plasmablast', 'Plasma', 'IgG4+Plasma',
        
        # Myeloid cell
        'M1Macrophage', 'MPO+Macrophage', 'M2Macrophage',
        'MerTK+M1Macrophage', 'MerTK+M2Macrophage', 'GPNMB+M1Macrophage',
        'GPNMB+M2Macrophage', 'cDC1', 'cDC2', 'pDC', 'Neutrophil',
        
        # Stromal and structural cells
        'Myofibroblast-SmoothMuscle', 'CD90+Myofibroblast', 'MMT',
        'StressedTubular', 'Apoptotic'
    ]
    
    print(f"Analyzing {len(all_cell_types)} cell types")
    print(f"Target datasets: IgG4-RKD and AAV only (excluding KC and Overall)")
    print(f"Analysis: Circos plots only (no centrality or network visualizations)")
    print(f"Using TISSUE masks for density calculations")
    print(f"Pixel size: 0.1507 μm")
    print(f"\nUpdated color scheme:")
    print(f"  - CD4T: Blue (#4169E1)")
    print(f"  - CD8T: Light Red (#FF8E8E)")
    print(f"  - Other Lymphoid: Purple (#9370DB)")
    print(f"  - Humoral: Green (#52C41A)")
    print(f"  - Myeloid: Orange (#FFA500)")
    print(f"  - Stromal: Gray (#A9A9A9)")
    
    # Load and prepare data
    print("\n📊 Loading and preparing patient-level data...")
    density_df, proportion_df = load_and_prepare_data(
        data_path, base_path, all_cell_types, pixel_size_um=0.1507)
    
    if density_df is None or proportion_df is None:
        print("❌ Failed to load data. Check file paths and data format.")
        return
    
    print(f"✅ Successfully processed {len(density_df)} patients")
    print(f"   Datasets: {density_df['Dataset'].value_counts().to_dict()}")
    
    # Perform Circos plot analysis for IgG4-RKD and AAV only
    print("\n🎯 Creating Circos plots for IgG4-RKD and AAV...")
    analyze_dataset_circos_only(density_df, proportion_df, all_cell_types, save_path)
    
    print(f"\n✅ Analysis complete! Results saved to: {save_path}")
    print(f"📁 Output files:")
    print(f"   - Circular Circos plots for IgG4-RKD and AAV datasets")
    print(f"   - Both density (TISSUE AREA) and proportion analyses")
    print(f"   - Correlation matrices (.csv)")
    print(f"   - P-value matrices (.csv)")
    print(f"   - Perfect circular shape with visible colored segments")
    print(f"   - Black-bordered ribbons for clear visualization")
    print(f"   - Updated color scheme matching cell type categories")

if __name__ == "__main__":
    main()
