import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from PIL import Image
import os
import warnings
from statsmodels.stats.multitest import multipletests
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
import plotly.graph_objects as go

# Set plot style and parameters with Arial font
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 16,  # Increased base font size
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.titlesize': 22
})

# Set plot style
try:
    plt.style.use('seaborn-whitegrid')
except (OSError, ValueError):
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except (OSError, ValueError):
        plt.style.use('ggplot')

warnings.filterwarnings('ignore')

# Ensure PIL can handle large images
Image.MAX_IMAGE_PIXELS = 1100000000

#########################################################
# NEW TISSUE AREA CALCULATION FUNCTION
#########################################################

def calculate_tissue_area(dataset, comp_name, base_path):
    """
    Calculate the total tissue area from tissue masks
    
    Args:
        dataset: Dataset name (mapped name like IgG4-RKD)
        comp_name: Component name
        base_path: Base directory path
    Returns:
        tissue_area: Number of pixels in the tissue area
    """
    # Map dataset name to path component if needed
    dataset_name_in_path = dataset
    if dataset == "KC":
        dataset_name_in_path = "NK2"
    elif dataset == "IgG4-RKD":
        dataset_name_in_path = "IgG4"
    
    # Define paths to the tissue mask files
    tissue_mask_path1 = f'{base_path}/{dataset_name_in_path}/tissue_composite_masks/DAPI_UV_1_{dataset_name_in_path}_{comp_name}.tif'
    tissue_mask_path2 = f'{base_path}/{dataset_name_in_path}/tissue_composite_masks/DAPI_UV_2_{dataset_name_in_path}_{comp_name}.tif'
    
    # Also check with .tiff extension
    tissue_mask_path1_alt = f'{base_path}/{dataset_name_in_path}/tissue_composite_masks/DAPI_UV_1_{dataset_name_in_path}_{comp_name}.tiff'
    tissue_mask_path2_alt = f'{base_path}/{dataset_name_in_path}/tissue_composite_masks/DAPI_UV_2_{dataset_name_in_path}_{comp_name}.tiff'
    
    # Find tissue mask
    tissue_mask_path = None
    for path in [tissue_mask_path1, tissue_mask_path2, tissue_mask_path1_alt, tissue_mask_path2_alt]:
        if os.path.exists(path):
            tissue_mask_path = path
            break
    
    if not tissue_mask_path:
        print(f"No tissue mask found for {comp_name}")
        return None
    
    try:
        # Load tissue mask
        tissue_mask = Image.open(tissue_mask_path)
        tissue_array = np.array(tissue_mask) > 0
        
        # Count pixels in tissue area
        tissue_area = np.sum(tissue_array)
        
        print(f"Tissue area for {comp_name}: {tissue_area} pixels")
        print(f"Using tissue mask: {tissue_mask_path}")
        
        return tissue_area
        
    except Exception as e:
        print(f"Error loading tissue mask {tissue_mask_path}: {str(e)}")
        return None

#########################################################
# UPDATED FUNCTIONS TO USE TISSUE AREA
#########################################################

def get_cell_counts(df, comp_name, category_column='CellCatName2', exclude_categories=None):
    """
    Get cell counts for each cell type in a specific component
    
    Args:
        df: DataFrame containing cell data
        comp_name: Name of the component to analyze
        category_column: Column name containing cell type categories
        exclude_categories: List of categories to exclude from counting
    Returns:
        dict: Dictionary of cell type counts
    """
    # Filter data for the specific component
    comp_df = df[df['CompName'] == comp_name]
    
    if comp_df.empty:
        print(f"No data found for component {comp_name}")
        return {}
    
    if category_column not in comp_df.columns:
        print(f"Warning: {category_column} column not found in the dataframe")
        return {}
    
    # Get unique cell categories
    cell_categories = comp_df[category_column].unique()
    
    # Exclude categories if specified
    if exclude_categories:
        cell_categories = [cat for cat in cell_categories if cat not in exclude_categories]
    
    # Count cells of each type
    counts = {}
    for cell_type in cell_categories:
        # Skip nan values
        if pd.isna(cell_type):
            continue
        
        # Count cells where category matches this cell type
        count = np.sum(comp_df[category_column] == cell_type)
        if count > 0:  # Only include non-zero counts
            counts[cell_type] = count
    
    return counts

def calculate_cell_proportions(total_cell_counts):
    """
    Calculate cell proportions (cell number of each class/total cell number)
    
    Args:
        total_cell_counts: Dictionary of total cell counts
    Returns:
        dict: Dictionary of cell proportions
    """
    # Calculate total number of cells
    total_cells = sum(total_cell_counts.values())
    
    # Calculate proportions
    proportions = {}
    for cell_type, count in total_cell_counts.items():
        proportions[cell_type] = count / total_cells if total_cells > 0 else 0
    
    return proportions

def calculate_densities(cell_counts, tissue_area, pixels_per_mm2):
    """
    Calculate cell densities (cells per mm² of tissue area)
    
    Args:
        cell_counts: Dictionary of cell counts
        tissue_area: Area in pixels (pixels with value > 0)
        pixels_per_mm2: Conversion factor from pixels to mm²
    Returns:
        dict: Dictionary of cell densities
    """
    # Convert pixel area to mm²
    area_mm2 = tissue_area / pixels_per_mm2
    
    densities = {}
    for cell_type, count in cell_counts.items():
        densities[cell_type] = count / area_mm2 if area_mm2 > 0 else 0
        
    return densities

def aggregate_by_patient(df, patient_column, comp_to_dataset, pixels_per_mm2, base_path, exclude_categories=None, output_dir=None):
    """
    Aggregate data by patient instead of by component, using tissue area
    
    Args:
        df: DataFrame containing cell data
        patient_column: Column name containing patient identifier
        comp_to_dataset: Dictionary mapping component names to datasets
        pixels_per_mm2: Conversion factor from pixels to mm²
        base_path: Base directory path for tissue masks
        exclude_categories: List of categories to exclude from counting
        output_dir: Directory to save output files
    Returns:
        DataFrame with patient-level densities, DataFrame with patient-level proportions
    """
    print(f"Aggregating data by {patient_column} using tissue area...")
    
    # Check if patient column exists
    if patient_column not in df.columns:
        print(f"Error: {patient_column} column not found in the dataframe")
        return None, None
    
    patient_data = []
    
    # Get unique patients
    unique_patients = df[patient_column].unique()
    print(f"Found {len(unique_patients)} unique patients/samples")
    
    # Process each patient
    for patient_id in unique_patients:
        print(f"Processing patient/sample: {patient_id}")
        
        # Get all components for this patient
        patient_df = df[df[patient_column] == patient_id]
        patient_components = patient_df['CompName'].unique()
        
        if len(patient_components) == 0:
            print(f"No components found for patient {patient_id}, skipping")
            continue
        
        # Check if we have dataset information for at least one component
        sample_comps_with_dataset = [c for c in patient_components if c in comp_to_dataset]
        if not sample_comps_with_dataset:
            print(f"No dataset information for patient {patient_id}, skipping")
            continue
        
        # Determine dataset from the components (should be the same for all components of a patient)
        dataset = comp_to_dataset[sample_comps_with_dataset[0]]
        
        # Initialize total cell counts and total tissue area
        total_cell_counts = {}
        total_tissue_area = 0
        valid_component_count = 0
        
        # Process each component for this patient
        for comp_name in patient_components:
            if comp_name not in comp_to_dataset:
                print(f"Skipping component {comp_name} - dataset not determined")
                continue
            
            if comp_to_dataset[comp_name] != dataset:
                print(f"Warning: Component {comp_name} has different dataset ({comp_to_dataset[comp_name]}) than expected ({dataset})")
                continue
            
            try:
                # Calculate tissue area
                tissue_area = calculate_tissue_area(dataset, comp_name, base_path)
                
                if tissue_area is None or tissue_area == 0:
                    print(f"Invalid tissue area for {comp_name}, skipping")
                    continue
                
                # Get cell counts
                cell_counts = get_cell_counts(df, comp_name, category_column='CellCatName2', exclude_categories=exclude_categories)
                
                if not cell_counts:
                    print(f"No valid cell counts for {comp_name}, skipping")
                    continue
                
                # Add to totals
                for cell_type, count in cell_counts.items():
                    if cell_type in total_cell_counts:
                        total_cell_counts[cell_type] += count
                    else:
                        total_cell_counts[cell_type] = count
                
                total_tissue_area += tissue_area
                valid_component_count += 1
                
            except Exception as e:
                print(f"Error processing component {comp_name} for patient {patient_id}: {str(e)}")
        
        # Skip patient if no valid components were processed
        if valid_component_count == 0 or total_tissue_area == 0:
            print(f"No valid components for patient {patient_id}, skipping")
            continue
        
        # Calculate patient-level densities and proportions
        patient_densities = calculate_densities(total_cell_counts, total_tissue_area, pixels_per_mm2)
        patient_proportions = calculate_cell_proportions(total_cell_counts)
        
        # Create patient record with all information
        patient_record = {
            'PatientID': patient_id,
            'Dataset': dataset,
            'ComponentCount': valid_component_count,
            'TotalTissueArea': total_tissue_area,
            'TotalTissueAreaMM2': total_tissue_area / pixels_per_mm2
        }
        
        # Add density and proportion for each cell type
        for cell_type in total_cell_counts.keys():
            patient_record[f"{cell_type}_density"] = patient_densities[cell_type]
            patient_record[f"{cell_type}_proportion"] = patient_proportions[cell_type]
            patient_record[f"{cell_type}_count"] = total_cell_counts[cell_type]
        
        patient_data.append(patient_record)
    
    # Convert to DataFrame
    if not patient_data:
        print("No patients were successfully processed")
        return None, None
    
    patient_df = pd.DataFrame(patient_data)
    
    # Extract density and proportion DataFrames
    density_cols = [col for col in patient_df.columns if col.endswith('_density')]
    proportion_cols = [col for col in patient_df.columns if col.endswith('_proportion')]
    
    # Create separate DataFrames for density and proportion analysis
    density_df = pd.DataFrame()
    density_df['PatientID'] = patient_df['PatientID']
    density_df['Dataset'] = patient_df['Dataset']
    
    proportion_df = pd.DataFrame()
    proportion_df['PatientID'] = patient_df['PatientID']
    proportion_df['Dataset'] = patient_df['Dataset']
    
    for col in density_cols:
        cell_type = col.replace('_density', '')
        density_df[cell_type] = patient_df[col]
    
    for col in proportion_cols:
        cell_type = col.replace('_proportion', '')
        proportion_df[cell_type] = patient_df[col]
    
    # Save original patient data for reference
    if output_dir:
        patient_df.to_csv(os.path.join(output_dir, "patient_level_combined_data_tissue.csv"), index=False)
    
    return density_df, proportion_df

# Modified function to return correlation matrices with enhanced visibility
def create_correlation_matrix(df, all_cell_types, title, save_path, dataset_name=None):
    """
    Create and save correlation matrices for the given dataset with multiple testing correction
    Enhanced for better visibility with larger fonts and optimized layout
    
    Args:
        df: DataFrame with cell densities/proportions/counts
        all_cell_types: Complete list of cell type categories
        title: Base title for the plots
        save_path: Path to save the plots
        dataset_name: Name of the dataset to include in filenames
    Returns:
        tuple: Correlation, p-value, and adjusted p-value DataFrames
    """
    # Make a copy of the dataframe to avoid modifying the original
    df_copy = df.copy()
    
    # Print the columns before modification
    print(f"Columns before adding missing cell types: {df_copy.columns.tolist()}")
    
    # Ensure all cell types are in the DataFrame even if they have no values
    for cell_type in all_cell_types:
        if cell_type not in df_copy.columns:
            print(f"Adding missing cell type to matrix: {cell_type}")
            df_copy[cell_type] = 0.0  # Add the column with zeros
    
    # Make sure we have enough samples
    if len(df) < 3:
        print(f"Not enough samples ({len(df)}) for correlation analysis in {title} - {dataset_name}")
        return None, None, None
    
    # Calculate correlation matrix
    # First try with spearmanr directly on the DataFrame
    try:
        # Initialize matrices with proper dimensions
        corr_matrix = np.zeros((len(all_cell_types), len(all_cell_types)))
        pval_matrix = np.zeros((len(all_cell_types), len(all_cell_types)))
        
        # Fill matrices with NaN initially
        corr_matrix.fill(np.nan)
        pval_matrix.fill(np.nan)
        
        # Map cell types to indices
        cell_type_to_idx = {ct: i for i, ct in enumerate(all_cell_types)}
        
        # Check which cell types have actual data (non-zero variance) to avoid misleading correlations
        valid_cell_types = []
        min_observations = 3  # Minimum number of non-zero observations required
        
        for cell_type in all_cell_types:
            # Add column if missing
            if cell_type not in df_copy.columns:
                df_copy[cell_type] = 0.0
                continue
                
            # Check for sufficient non-zero data
            non_zero_count = np.sum(df_copy[cell_type] > 0)
            if non_zero_count >= min_observations and df_copy[cell_type].nunique() > 1:
                valid_cell_types.append(cell_type)
            else:
                print(f"WARNING: Cell type {cell_type} has insufficient data (only {non_zero_count} non-zero values)")
        
        print(f"Using {len(valid_cell_types)} out of {len(all_cell_types)} cell types for correlation analysis")
        
        # List to store all p-values for multiple testing correction
        all_pvals = []
        all_pval_indices = []
        
        for i, col1 in enumerate(all_cell_types):
            for j, col2 in enumerate(all_cell_types):
                try:
                    # Special case: diagonal elements (self-correlation)
                    if i == j:
                        corr_matrix[i, j] = 1.0
                        pval_matrix[i, j] = 0.0
                        continue
                        
                    # Skip computation if either cell type has insufficient data
                    if col1 not in valid_cell_types or col2 not in valid_cell_types:
                        corr_matrix[i, j] = np.nan
                        pval_matrix[i, j] = np.nan
                        continue
                    
                    # Verify columns exist
                    if col1 not in df_copy.columns or col2 not in df_copy.columns:
                        print(f"WARNING: Columns {col1} or {col2} missing from DataFrame")
                        corr_matrix[i, j] = np.nan
                        pval_matrix[i, j] = np.nan
                        continue
                    
                    # Get data for correlation
                    data1 = df_copy[col1]
                    data2 = df_copy[col2]
                    
                    # Remove rows where either value is NaN
                    mask = ~np.isnan(data1) & ~np.isnan(data2)
                    if np.sum(mask) < min_observations:
                        print(f"WARNING: Insufficient valid pairs for {col1} vs {col2}")
                        corr_matrix[i, j] = np.nan
                        pval_matrix[i, j] = np.nan
                        continue
                    
                    # Calculate Spearman correlation
                    corr, pval = spearmanr(data1[mask], data2[mask], nan_policy='omit')
                    
                    # Only store correlations if p-value is valid
                    if np.isnan(pval):
                        corr_matrix[i, j] = np.nan
                        pval_matrix[i, j] = np.nan
                    else:
                        corr_matrix[i, j] = corr
                        pval_matrix[i, j] = pval
                        
                        # Store p-value for multiple testing correction
                        # Only store if i < j to avoid duplicates (since correlation matrix is symmetric)
                        if i < j:
                            all_pvals.append(pval)
                            all_pval_indices.append((i, j))
                        
                except Exception as e:
                    print(f"Error calculating correlation between {col1} and {col2}: {str(e)}")
                    corr_matrix[i, j] = np.nan
                    pval_matrix[i, j] = np.nan
                    
        # Apply multiple testing correction to p-values (Benjamini-Hochberg FDR)
        
        # Initialize adjusted p-value matrix with NaNs
        adjusted_pval_matrix = np.full_like(pval_matrix, np.nan)
        
        if all_pvals:
            try:
                # Apply Benjamini-Hochberg correction
                _, adjusted_pvals, _, _ = multipletests(all_pvals, alpha=0.05, method='fdr_bh')
                
                # Fill in the adjusted p-value matrix
                for idx, (i, j) in enumerate(all_pval_indices):
                    adjusted_pval_matrix[i, j] = adjusted_pvals[idx]
                    # Symmetric matrix - fill in the other side too
                    adjusted_pval_matrix[j, i] = adjusted_pvals[idx]
                    
                print(f"Successfully applied Benjamini-Hochberg FDR correction to {len(all_pvals)} p-values")
            except Exception as e:
                print(f"Error applying multiple testing correction: {str(e)}")
                print("Using uncorrected p-values for visualization")
                adjusted_pval_matrix = pval_matrix.copy()
        else:
            print("No valid p-values found for multiple testing correction")
            adjusted_pval_matrix = pval_matrix.copy()
            
    except Exception as e:
        print(f"Error calculating correlation matrix: {str(e)}")
        # If direct method fails, try pandas correlation
        try:
            # Make sure all columns exist in the DataFrame
            for cell_type in all_cell_types:
                if cell_type not in df_copy.columns:
                    df_copy[cell_type] = 0.0
                    
            print("Attempting pandas correlation method as fallback")
            # Create the correlation matrix using pandas corr method
            corr_matrix_df = df_copy[all_cell_types].corr(method='spearman')
            
            # Convert to numpy array
            corr_matrix = corr_matrix_df.values
            
            # For p-values, we'll have to set them to NaN since pandas doesn't provide them
            pval_matrix = np.full_like(corr_matrix, np.nan)
            adjusted_pval_matrix = np.full_like(corr_matrix, np.nan)
            
            # Print correlation matrix shape
            print(f"Fallback correlation matrix shape: {corr_matrix.shape}")
        except Exception as e2:
            print(f"Fallback correlation calculation also failed: {str(e2)}")
            return None, None, None
    
    # Print the cell types that are actually being included in the matrix
    included_cell_types = [ct for ct in all_cell_types if ct in df_copy.columns]
    print(f"Including {len(included_cell_types)} cell types in correlation matrix out of {len(all_cell_types)} total types")
    
    # Verify all cell types are present before creating correlation matrix
    missing_after_addition = [ct for ct in all_cell_types if ct not in df_copy.columns]
    if missing_after_addition:
        print(f"WARNING: Still missing {len(missing_after_addition)} cell types after attempted addition")
        for ct in missing_after_addition[:5]:  # Print first 5 missing types
            print(f"  - Missing: {ct}")
        if len(missing_after_addition) > 5:
            print(f"  - ... and {len(missing_after_addition) - 5} more")
    
    # Convert matrices to DataFrames - ensure we explicitly use all_cell_types for both indices and columns
    # This is crucial to include all cell types in our visualization
    corr_df = pd.DataFrame(corr_matrix, index=all_cell_types, columns=all_cell_types)
    pval_df = pd.DataFrame(pval_matrix, index=all_cell_types, columns=all_cell_types)
    adj_pval_df = pd.DataFrame(adjusted_pval_matrix, index=all_cell_types, columns=all_cell_types)
    
    # Verify the correlation matrix has all cell types
    if len(corr_df.columns) != len(all_cell_types) or len(corr_df.index) != len(all_cell_types):
        print(f"WARNING: Correlation matrix dimensions ({len(corr_df.columns)}x{len(corr_df.index)}) do not match expected size ({len(all_cell_types)}x{len(all_cell_types)})")
        # Force the index and columns to match all_cell_types
        corr_df = pd.DataFrame(corr_matrix, index=all_cell_types, columns=all_cell_types)
        pval_df = pd.DataFrame(pval_matrix, index=all_cell_types, columns=all_cell_types)
        adj_pval_df = pd.DataFrame(adjusted_pval_matrix, index=all_cell_types, columns=all_cell_types)
    
    # Define filename prefix based on dataset
    prefix = f"{dataset_name}_" if dataset_name else ""
    
    # Ensure filename part of title is filesystem-safe (no spaces, etc.)
    safe_title = title.replace(" ", "_").replace("/", "_")
    
    # ENHANCED HEATMAP CREATION WITH LARGE VISIBLE LABELS
    # Calculate optimal figure size for half A4 (approximately 8.27 x 5.85 inches)
    n_types = len(all_cell_types)
    
    # For half A4, use larger figure size to accommodate all labels
    fig_width = max(16, n_types * 0.4)  # Minimum 16 inches width
    fig_height = max(12, n_types * 0.35)  # Minimum 12 inches height
    
    # Limit maximum size to prevent extremely large figures
    fig_width = min(fig_width, 24)
    fig_height = min(fig_height, 20)
    
    # Plot and save full correlation matrix with enhanced visibility
    plt.figure(figsize=(fig_width, fig_height))
    
    # Set font sizes based on number of cell types
    if n_types > 50:
        label_fontsize = 10
        tick_fontsize = 8
    elif n_types > 30:
        label_fontsize = 12
        tick_fontsize = 10
    else:
        label_fontsize = 14
        tick_fontsize = 12
    
    try:
        # Create heatmap with enhanced visibility
        ax = sns.heatmap(corr_df.loc[all_cell_types, all_cell_types], 
                        annot=False, 
                        cmap="coolwarm", 
                        fmt=".2f", 
                        linewidths=0.2,
                        vmin=-1,
                        vmax=1,
                        square=True,
                        cbar_kws={"shrink": 0.6, "aspect": 20})
        
        # Enhanced label formatting
        plt.xticks(rotation=90, ha='center', fontsize=tick_fontsize)
        plt.yticks(rotation=0, ha='right', fontsize=tick_fontsize)
        
        # Set title with larger font
        plt.title(f"{title} - {dataset_name}" if dataset_name else title, 
                 fontsize=label_fontsize + 4, pad=20, weight='bold')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Fine-tune margins to ensure all labels are visible
        plt.subplots_adjust(bottom=0.25, left=0.25, right=0.95, top=0.90)
        
    except Exception as e:
        print(f"Error in creating heatmap with all cell types: {str(e)}")
        print("Falling back to using correlation matrix as is")
        ax = sns.heatmap(corr_df, 
                        annot=False, 
                        cmap="coolwarm", 
                        fmt=".2f", 
                        linewidths=0.2,
                        vmin=-1,
                        vmax=1,
                        square=True)
        plt.xticks(rotation=90, ha='center', fontsize=tick_fontsize)
        plt.yticks(rotation=0, ha='right', fontsize=tick_fontsize)
    
    plt.savefig(os.path.join(save_path, f"{prefix}{safe_title}_full_correlation_matrix.tiff"), 
                format='tiff', 
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.close()
    
    # Create masks for non-significant correlations
    # 1. For raw p-values
    significance_threshold = 0.05
    raw_mask = pval_df > significance_threshold
    
    # 2. For adjusted p-values
    adj_mask = adj_pval_df > significance_threshold
    
    # For regular heatmap with raw p-values, we can use NaN values
    filtered_corr_df = corr_df.copy()
    filtered_corr_df[raw_mask] = np.nan
    
    # Plot and save significant correlation matrix (raw p-values) with enhanced visibility
    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.heatmap(filtered_corr_df, 
                    annot=False, 
                    cmap="coolwarm", 
                    fmt=".2f", 
                    linewidths=0.2, 
                    vmin=-1,
                    vmax=1,
                    square=True,
                    cbar_kws={"shrink": 0.6, "aspect": 20})
    
    plt.xticks(rotation=90, ha='center', fontsize=tick_fontsize)
    plt.yticks(rotation=0, ha='right', fontsize=tick_fontsize)
    plt.title(f"Significant {title} (raw p < 0.05) - {dataset_name}" if dataset_name else f"Significant {title} (raw p < 0.05)",
              fontsize=label_fontsize + 4, pad=20, weight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25, left=0.25, right=0.95, top=0.90)
    
    plt.savefig(os.path.join(save_path, f"{prefix}{safe_title}_significant_raw_pval_correlation_matrix.tiff"),
                format='tiff',
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.close()
    
    # Create significant correlation matrix with adjusted p-values
    filtered_adj_corr_df = corr_df.copy()
    filtered_adj_corr_df[adj_mask] = np.nan
    
    # Plot and save significant correlation matrix (adjusted p-values) with enhanced visibility
    plt.figure(figsize=(fig_width, fig_height))
    ax = sns.heatmap(filtered_adj_corr_df, 
                    annot=False, 
                    cmap="coolwarm", 
                    fmt=".2f", 
                    linewidths=0.2, 
                    vmin=-1,
                    vmax=1,
                    square=True,
                    cbar_kws={"shrink": 0.6, "aspect": 20})
    
    plt.xticks(rotation=90, ha='center', fontsize=tick_fontsize)
    plt.yticks(rotation=0, ha='right', fontsize=tick_fontsize)
    plt.title(f"Significant {title} (FDR-adjusted p < 0.05) - {dataset_name}" if dataset_name else f"Significant {title} (FDR-adjusted p < 0.05)",
              fontsize=label_fontsize + 4, pad=20, weight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25, left=0.25, right=0.95, top=0.90)
    
    plt.savefig(os.path.join(save_path, f"{prefix}{safe_title}_significant_adjusted_pval_correlation_matrix.tiff"),
                format='tiff',
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.close()
    
    # Save correlation data to CSV for further analysis
    corr_df.to_csv(os.path.join(save_path, f"{prefix}{safe_title}_correlation_values.csv"))
    pval_df.to_csv(os.path.join(save_path, f"{prefix}{safe_title}_correlation_pvalues.csv"))
    adj_pval_df.to_csv(os.path.join(save_path, f"{prefix}{safe_title}_correlation_adjusted_pvalues.csv"))
    
    # Create a clustermap for better visualization of patterns with enhanced visibility
    try:
        # Replace NaN values with 0 for clustering (to avoid clustering errors)
        cluster_df = corr_df.copy()
        cluster_df = cluster_df.fillna(0)
        
        # Make sure all cell types are included
        missing_types = [ct for ct in all_cell_types if ct not in cluster_df.columns]
        if missing_types:
            print(f"Adding {len(missing_types)} missing cell types to cluster matrix")
            for ct in missing_types:
                cluster_df[ct] = 0
                cluster_df.loc[ct] = 0
        
        # Sort columns and index to ensure they match all_cell_types
        try:
            cluster_df = cluster_df.reindex(index=all_cell_types, columns=all_cell_types)
        except Exception as e:
            print(f"Error reindexing cluster dataframe: {str(e)}")
        
        print(f"Clustermap matrix shape: {cluster_df.shape}")
        
        # Create clustermap with enhanced visibility
        cluster = sns.clustermap(cluster_df, 
                              cmap="coolwarm",
                              method="average",
                              metric="euclidean",
                              vmin=-1,
                              vmax=1,
                              figsize=(fig_width, fig_height),
                              linewidths=0.2,
                              xticklabels=True,
                              yticklabels=True,
                              cbar_kws={"shrink": 0.6})
        
        # Enhanced label formatting for clustermap
        plt.setp(cluster.ax_heatmap.xaxis.get_majorticklabels(), 
                rotation=90, ha='center', fontsize=tick_fontsize)
        plt.setp(cluster.ax_heatmap.yaxis.get_majorticklabels(), 
                rotation=0, ha='right', fontsize=tick_fontsize)
        
        cluster.fig.suptitle(f"Clustered {title} - {dataset_name}" if dataset_name else f"Clustered {title}",
                           fontsize=label_fontsize + 4, weight='bold')
        
        plt.savefig(os.path.join(save_path, f"{prefix}{safe_title}_clustered_correlation_matrix.tiff"),
                    format='tiff',
                    dpi=300,
                    bbox_inches='tight',
                    facecolor='white')
        plt.close()
        
    except Exception as e:
        print(f"Error creating clustered heatmap for {title}: {str(e)}")
    
    return corr_df, pval_df, adj_pval_df

#########################################################
# New visualization functions (Enhanced for visibility)
#########################################################

def create_focused_correlation_matrix(corr_df, pval_df, adj_pval_df, cell_types, save_path, title, 
                                    min_corr=0.4, use_adjusted=True):
    """
    Create a more readable correlation matrix by focusing only on significant
    correlations above a minimum correlation threshold, with enhanced visibility
    """
    print(f"Creating focused correlation matrix for {title}...")
    
    # Select the p-value DataFrame to use
    sig_df = adj_pval_df if use_adjusted else pval_df
    sig_type = "FDR-adjusted" if use_adjusted else "raw"
    
    # Create mask for non-significant correlations
    sig_threshold = 0.05
    mask = sig_df > sig_threshold
    
    # Also create a mask for correlations below the minimum threshold
    min_mask = abs(corr_df) < min_corr
    
    # Combine masks: exclude non-significant OR below threshold correlations
    combined_mask = mask | min_mask
    
    # Apply mask to correlation matrix
    filtered_corr_df = corr_df.copy()
    filtered_corr_df[combined_mask] = np.nan
    
    # Remove diagonal elements (self-correlations)
    for i in range(len(cell_types)):
        filtered_corr_df.iloc[i, i] = np.nan
    
    # Find rows and columns with at least one significant correlation
    has_sig_corr = ~filtered_corr_df.isna().all(axis=1) & ~filtered_corr_df.isna().all(axis=0)
    active_cells = filtered_corr_df.index[has_sig_corr].tolist()
    
    if len(active_cells) == 0:
        print(f"No significant correlations with |r| >= {min_corr} found for {title}.")
        return
    
    if len(active_cells) < 2:
        print(f"Not enough significant correlations with |r| >= {min_corr} found for {title}. Need at least 2 for clustering.")
        return
    
    print(f"Found {len(active_cells)} cell types with significant correlations (|r| >= {min_corr}).")
    
    # Filter to only include cell types with significant correlations
    focused_corr_df = filtered_corr_df.loc[active_cells, active_cells]
    
    # Calculate figure size based on number of active cells with enhanced visibility
    fig_width = max(12, len(active_cells) * 0.5)
    fig_height = max(10, len(active_cells) * 0.45)
    
    # Set font sizes based on number of active cells
    if len(active_cells) > 30:
        label_fontsize = 10
        tick_fontsize = 8
    elif len(active_cells) > 20:
        label_fontsize = 12
        tick_fontsize = 10
    else:
        label_fontsize = 14
        tick_fontsize = 12
    
    # Check if the focused matrix has enough finite values for clustering
    finite_values = np.isfinite(focused_corr_df.values)
    if np.sum(finite_values) < len(active_cells):
        print(f"Warning: Not enough finite values for clustering in {title}. Creating heatmap without clustering.")
        
        # Create a standard heatmap instead with enhanced visibility
        plt.figure(figsize=(fig_width, fig_height))
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        ax = sns.heatmap(focused_corr_df, cmap=cmap, vmin=-1, vmax=1, center=0, 
                        linewidths=0.2, square=True,
                        cbar_kws={"shrink": 0.6})
        
        plt.title(f"Focused {title} (|r| ≥ {min_corr}, {sig_type} p < 0.05)", 
                 fontsize=label_fontsize + 2, pad=15, weight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=tick_fontsize)
        plt.yticks(rotation=0, ha='right', fontsize=tick_fontsize)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25, left=0.25)
    else:
        try:
            # Fill NaN values with 0 for clustering purposes only
            cluster_data = focused_corr_df.fillna(0)
            
            # Creating a clustered heatmap with the reduced matrix and enhanced visibility
            g = sns.clustermap(
                cluster_data,  # Use the filled data for clustering
                cmap="coolwarm",
                vmin=-1,
                vmax=1,
                center=0,
                linewidths=0.2,
                figsize=(fig_width, fig_height),
                xticklabels=True,
                yticklabels=True,
                square=True,
                cbar_kws={"shrink": 0.6}
            )
            
            # Enhanced label formatting for clustermap
            plt.setp(g.ax_heatmap.xaxis.get_majorticklabels(), 
                    rotation=45, ha='right', fontsize=tick_fontsize)
            plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), 
                    rotation=0, ha='right', fontsize=tick_fontsize)
            
            g.fig.suptitle(f"Focused {title} (|r| ≥ {min_corr}, {sig_type} p < 0.05)", 
                          fontsize=label_fontsize + 2, weight='bold')
            
        except Exception as e:
            print(f"Error in clustering: {str(e)}. Creating heatmap without clustering.")
            
            # If clustering fails, create a standard heatmap with enhanced visibility
            plt.figure(figsize=(fig_width, fig_height))
            cmap = sns.diverging_palette(220, 10, as_cmap=True)
            ax = sns.heatmap(focused_corr_df, cmap=cmap, vmin=-1, vmax=1, center=0, 
                            linewidths=0.2, square=True,
                            cbar_kws={"shrink": 0.6})
            
            plt.title(f"Focused {title} (|r| ≥ {min_corr}, {sig_type} p < 0.05)", 
                     fontsize=label_fontsize + 2, pad=15, weight='bold')
            plt.xticks(rotation=45, ha='right', fontsize=tick_fontsize)
            plt.yticks(rotation=0, ha='right', fontsize=tick_fontsize)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.25, left=0.25)
    
    # Save the focused correlation matrix
    safe_title = title.replace(" ", "_").replace("/", "_")
    plt.savefig(os.path.join(save_path, f"{safe_title}_focused_{sig_type}_correlation_matrix.tiff"),
                format='tiff', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Save the focused correlation matrix as CSV
    focused_corr_df.to_csv(os.path.join(save_path, f"{safe_title}_focused_{sig_type}_correlation_values.csv"))
    
    print(f"Successfully created focused correlation matrix for {title}")

def create_correlation_network(corr_df, pval_df, adj_pval_df, save_path, title, 
                              cell_values=None, value_type="Density",
                              min_abs_corr=0.3, use_adjusted=True, 
                              max_nodes=60, filter_isolates=True):
    """
    Create a network graph visualization of the correlation matrix with enhanced label visibility
    """
    print(f"Creating correlation network for {title}...")
    
    # Select p-value matrix and significance type
    sig_df = adj_pval_df if use_adjusted else pval_df
    sig_type = "FDR-adjusted" if use_adjusted else "raw"
    
    # Create a new graph
    G = nx.Graph()
    
    # Create significance mask (True if significant)
    sig_mask = sig_df <= 0.05
    
    # Create minimum correlation mask (True if above threshold)
    corr_mask = abs(corr_df) >= min_abs_corr
    
    # Apply both masks to get significant correlations above threshold
    combined_mask = sig_mask & corr_mask
    
    # Track nodes connected to significant edges
    nodes_with_edges = set()
    all_edges = []
    
    # Collect all potential edges
    for i, cell1 in enumerate(corr_df.columns):
        for j, cell2 in enumerate(corr_df.columns):
            if i < j:  # Only process each pair once (lower triangle)
                if combined_mask.loc[cell1, cell2]:
                    corr_val = corr_df.loc[cell1, cell2]
                    abs_corr = abs(corr_val)
                    # Store as a tuple: (cell1, cell2, abs_correlation, correlation_value)
                    all_edges.append((cell1, cell2, abs_corr, corr_val))
                    nodes_with_edges.add(cell1)
                    nodes_with_edges.add(cell2)
    
    # If we have too many nodes, filter to include only the strongest connections
    if len(nodes_with_edges) > max_nodes or len(all_edges) > max_nodes * 3:
        # Sort edges by absolute correlation (highest first)
        all_edges.sort(key=lambda x: x[2], reverse=True)
        
        # Take top edges up to a reasonable number
        edge_limit = min(len(all_edges), max_nodes * 3)
        filtered_edges = all_edges[:edge_limit]
        
        # Recalculate nodes that remain after filtering
        nodes_with_edges = set()
        for edge in filtered_edges:
            nodes_with_edges.add(edge[0])
            nodes_with_edges.add(edge[1])
        
        print(f"Reduced network from {len(all_edges)} edges to {len(filtered_edges)} strongest edges")
        print(f"Reduced from {len(corr_df.columns)} nodes to {len(nodes_with_edges)} connected nodes")
        
        # Use the filtered edge list
        all_edges = filtered_edges
    
    # Create a subgraph with only the filtered nodes
    for node in nodes_with_edges:
        G.add_node(node)
    
    # Add edges between nodes with correlation values
    for source, target, abs_corr, corr_val in all_edges:
        G.add_edge(source, target, 
                  weight=abs_corr,  # Store absolute correlation value for edge thickness
                  color='red' if corr_val > 0 else 'blue',  # Red for positive, blue for negative
                  corr=corr_val,  # Store original correlation value
                  abs_corr=abs_corr)  # Store absolute correlation for reference
    
    # Remove isolates if requested
    if filter_isolates:
        isolated_nodes = list(nx.isolates(G))
        G.remove_nodes_from(isolated_nodes)
    
    # If graph is empty, exit
    if len(G.nodes()) == 0:
        print(f"No significant correlations above threshold {min_abs_corr} found for {title}.")
        return
    
    # Calculate node sizes based on cell values if provided, otherwise use node degrees
    if cell_values is not None and isinstance(cell_values, pd.DataFrame):
        # Extract the mean values for each cell type across patients
        cell_means = {}
        for node in G.nodes():
            if node in cell_values.columns:
                # Calculate mean excluding NaN values
                mean_val = cell_values[node].mean(skipna=True)
                cell_means[node] = mean_val if not pd.isna(mean_val) else 0.0
            else:
                cell_means[node] = 0.0
                
        # Normalize the cell values for node sizing
        min_val = min(cell_means.values()) if cell_means else 0
        max_val = max(cell_means.values()) if cell_means else 1
        
        # Scale node sizes based on cell values - SIGNIFICANTLY INCREASED SIZE
        node_sizes = {}
        for node in G.nodes():
            if max_val > min_val:
                normalized_val = (cell_means[node] - min_val) / (max_val - min_val)
                # MUCH larger nodes to accommodate larger labels
                node_sizes[node] = 2500 + 5000 * normalized_val  # Scale between 2500 and 7500
            else:
                node_sizes[node] = 4000  # Default large size
                
        # Store cell values in nodes for reference
        for node, value in cell_means.items():
            G.nodes[node]['value'] = value
    else:
        # Fallback to degree-based sizing if no cell values provided
        degrees = dict(nx.degree(G))
        min_degree = min(degrees.values()) if degrees else 1
        max_degree = max(degrees.values()) if degrees else 1
        
        # MUCH larger nodes to accommodate larger labels
        node_sizes = {node: 2500 + 3000 * (degrees[node] - min_degree) / (max_degree - min_degree) 
                     if max_degree > min_degree else 4000 for node in G.nodes()}
    
    # Get min and max absolute correlation values for scaling edge widths
    min_abs_corr_val = min([G[u][v]['abs_corr'] for u, v in G.edges()])
    max_abs_corr_val = max([G[u][v]['abs_corr'] for u, v in G.edges()])
    
    # Calculate edge properties
    # Scale edge thickness based on absolute correlation value
    edge_weights = []
    edge_alphas = []
    for u, v in G.edges():
        abs_corr = G[u][v]['abs_corr']
        # Scale width non-linearly to emphasize differences
        # Map from min_abs_corr to max_abs_corr -> 1.0 to 6.0 line width
        if max_abs_corr_val > min_abs_corr_val:
            normalized_weight = (abs_corr - min_abs_corr_val) / (max_abs_corr_val - min_abs_corr_val)
            edge_weights.append(1.0 + 5.0 * normalized_weight)
            
            # Also use correlation strength for edge opacity
            edge_alphas.append(0.5 + 0.5 * normalized_weight)
        else:
            edge_weights.append(3.0)  # Default width
            edge_alphas.append(0.7)   # Default alpha
    
    edge_colors = [G[u][v]['color'] for u, v in G.edges()]
    
    # Create a much larger figure size for enhanced visibility
    fig_size = min(32, max(20, len(G.nodes()) / 1.2))
    plt.figure(figsize=(fig_size, fig_size))
    
    # Use a force-directed layout algorithm with increased spacing
    pos = nx.spring_layout(G, k=2.0, iterations=300, seed=42)
    
    # Draw edges with thickness proportional to correlation strength
    for i, (u, v) in enumerate(G.edges()):
        # Draw each edge individually to control alpha
        nx.draw_networkx_edges(G, pos, 
                             edgelist=[(u, v)], 
                             width=edge_weights[i], 
                             edge_color=[edge_colors[i]], 
                             alpha=edge_alphas[i],
                             connectionstyle='arc3,rad=0.1')
    
    # Draw nodes with scaled sizes and custom colormap
    node_color_list = []
    for node in G.nodes():
        # Use different colors based on node degree for better visibility
        degree = G.degree(node)
        if degree > np.median(list(dict(G.degree()).values())):
            node_color_list.append('lightblue')  # Higher degree nodes
        else:
            node_color_list.append('lightgray')  # Lower degree nodes
    
    nodes = nx.draw_networkx_nodes(G, pos, 
                          node_size=[node_sizes[node] for node in G.nodes()],
                          node_color=node_color_list,
                          alpha=0.8,
                          linewidths=2,
                          edgecolors='black')
    
    # Add labels directly ON the nodes with MUCH LARGER font and full names
    for node, (x, y) in pos.items():
        # Determine text color based on node background
        label_color = 'black'
        
        # Calculate larger font size based on node size and network size
        node_size = node_sizes[node]
        # SIGNIFICANTLY increased font size range
        if len(G.nodes()) > 30:
            font_size = min(20, max(12, np.sqrt(node_size) / 10))
        elif len(G.nodes()) > 15:
            font_size = min(24, max(14, np.sqrt(node_size) / 9))
        else:
            font_size = min(28, max(16, np.sqrt(node_size) / 8))
        
        # Use the full cell type name - NO truncation
        label = node
        
        # Place label directly on the node with enhanced styling
        plt.text(x, y, label, 
                fontsize=font_size,
                ha='center', 
                va='center',
                color=label_color,
                fontweight='bold',
                family='Arial',
                bbox=dict(facecolor='white', 
                         alpha=0.9,  # More opaque for better readability
                         edgecolor='black',
                         linewidth=1,
                         boxstyle='round,pad=0.4'))  # More padding
    
    # Add a title with enhanced formatting
    if cell_values is not None:
        plt.title(f"Correlation Network - {title}\n({sig_type} p < 0.05, |r| ≥ {min_abs_corr:.1f})\nNode size proportional to cell {value_type}, edge thickness to |r|", 
                 fontsize=20, weight='bold', family='Arial', pad=25)
    else:
        plt.title(f"Correlation Network - {title}\n({sig_type} p < 0.05, |r| ≥ {min_abs_corr:.1f})\nNode size proportional to connections, edge thickness to |r|", 
                 fontsize=20, weight='bold', family='Arial', pad=25)
    
    # Add a comprehensive legend with larger fonts
    legend_elements = [
        mpatches.Patch(color='red', label='Positive Correlation'),
        mpatches.Patch(color='blue', label='Negative Correlation'),
        mpatches.Patch(color='lightblue', label='Higher Connectivity Nodes'),
        mpatches.Patch(color='lightgray', label='Lower Connectivity Nodes')
    ]
    
    # Add edge thickness legend
    if min_abs_corr_val < max_abs_corr_val:
        for val, width in [(min_abs_corr_val, 1.0), 
                           ((min_abs_corr_val + max_abs_corr_val)/2, 3.0), 
                           (max_abs_corr_val, 6.0)]:
            legend_elements.append(
                plt.Line2D([0], [0], color='gray', linewidth=width, 
                          label=f'|r| = {val:.2f}')
            )
    
    # Add size legend if using cell values
    if cell_values is not None:
        # Get actual min/max values for legend
        min_value = min(val for val in cell_means.values() if not pd.isna(val))
        max_value = max(val for val in cell_means.values() if not pd.isna(val))
        
        # Create scatter points for size legend
        small_circle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
                            markersize=8, label=f'Min {value_type}: {min_value:.2e}')
        large_circle = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
                            markersize=15, label=f'Max {value_type}: {max_value:.2e}')
        
        legend_elements.extend([small_circle, large_circle])
    
    # Position legend outside of network area with larger font
    plt.legend(handles=legend_elements, loc='upper right', 
              fontsize=16, frameon=True, 
              facecolor='white', edgecolor='black')
    
    # Remove axes
    plt.axis('off')
    
    # Set tighter layout
    plt.tight_layout()
    
    # Save the figure with controlled DPI
    safe_title = title.replace(" ", "_").replace("/", "_")
    value_suffix = f"_{value_type.lower()}" if cell_values is not None else ""
    
    try:
        # Save with high DPI for publication quality
        plt.savefig(os.path.join(save_path, f"{safe_title}_correlation_network{value_suffix}_{sig_type}.tiff"),
                    format='tiff', dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Successfully saved network visualization to {save_path}")
    except ValueError as e:
        print(f"Warning: Could not save with tight bounding box. Trying with standard bounds. Error: {str(e)}")
        # If that fails, try without tight bbox and lower DPI
        plt.savefig(os.path.join(save_path, f"{safe_title}_correlation_network{value_suffix}_{sig_type}.tiff"),
                    format='tiff', dpi=100, facecolor='white')
    
    plt.close()
    
    # Export network data to CSV for potential use in other programs
    edge_data = []
    for u, v, d in G.edges(data=True):
        edge_data.append({
            'Source': u,
            'Target': v,
            'Correlation': d['corr'],
            'AbsCorrelation': d['abs_corr'],
            'Type': 'Positive' if d['corr'] > 0 else 'Negative'
        })
    
    # Also export node data including cell values
    node_data = []
    for node in G.nodes():
        node_data.append({
            'CellType': node,
            'Degree': G.degree(node),
            'Value': G.nodes[node].get('value', 0),
            'Size': node_sizes[node]
        })
    
    # Save to CSV
    edge_df = pd.DataFrame(edge_data)
    node_df = pd.DataFrame(node_data)
    
    edge_df.to_csv(os.path.join(save_path, f"{safe_title}_correlation_network{value_suffix}_{sig_type}_edges.csv"), index=False)
    node_df.to_csv(os.path.join(save_path, f"{safe_title}_correlation_network{value_suffix}_{sig_type}_nodes.csv"), index=False)
    
    print(f"Successfully created correlation network for {title}")
    return G

def create_grouped_correlation_heatmap(corr_df, pval_df, adj_pval_df, cell_groups, save_path, title, use_adjusted=True):
    """
    Create a correlation heatmap at the group level with enhanced visibility
    """
    print(f"Creating grouped correlation heatmap for {title}...")
    
    # Select p-value matrix and significance type
    sig_df = adj_pval_df if use_adjusted else pval_df
    sig_type = "FDR-adjusted" if use_adjusted else "raw"
    
    # Create significance mask
    sig_threshold = 0.05
    sig_mask = sig_df <= sig_threshold
    
    # Initialize arrays for group-level analysis
    num_groups = len(cell_groups)
    
    # Create empty DataFrames for group-level metrics
    group_corr_mean = pd.DataFrame(0.0, index=range(num_groups), columns=range(num_groups))
    group_corr_median = pd.DataFrame(0.0, index=range(num_groups), columns=range(num_groups))
    group_sig_count = pd.DataFrame(0, index=range(num_groups), columns=range(num_groups))
    group_total_pairs = pd.DataFrame(0, index=range(num_groups), columns=range(num_groups))
    group_sig_percent = pd.DataFrame(0.0, index=range(num_groups), columns=range(num_groups))
    
    # For each pair of groups, calculate stats
    for i, group1 in enumerate(cell_groups):
        for j, group2 in enumerate(cell_groups):
            # Get all valid cell types in each group (those present in the correlation matrix)
            valid_cells1 = [cell for cell in group1 if cell in corr_df.columns]
            valid_cells2 = [cell for cell in group2 if cell in corr_df.columns]
            
            if not valid_cells1 or not valid_cells2:
                group_corr_mean.iloc[i, j] = np.nan
                group_corr_median.iloc[i, j] = np.nan
                continue
            
            # Get all pairs of cells between these groups
            corr_values = []
            sig_count = 0
            total_pairs = 0
            
            for cell1 in valid_cells1:
                for cell2 in valid_cells2:
                    # Skip self-correlations when groups are the same
                    if i == j and cell1 == cell2:
                        continue
                    
                    # Get correlation value
                    corr_val = corr_df.loc[cell1, cell2]
                    
                    # Skip NaN values
                    if pd.isna(corr_val):
                        continue
                    
                    corr_values.append(corr_val)
                    total_pairs += 1
                    
                    # Check if significant
                    if sig_mask.loc[cell1, cell2]:
                        sig_count += 1
            
            # Calculate statistics if we have valid correlations
            if corr_values:
                group_corr_mean.iloc[i, j] = np.mean(corr_values)
                group_corr_median.iloc[i, j] = np.median(corr_values)
                group_sig_count.iloc[i, j] = sig_count
                group_total_pairs.iloc[i, j] = total_pairs
                group_sig_percent.iloc[i, j] = 100 * sig_count / total_pairs if total_pairs > 0 else 0
    
    # Create descriptive labels for groups
    group_labels = [f"Group {i+1}\n({len(g)} types)" for i, g in enumerate(cell_groups)]
    
    # Set index and columns with more descriptive names
    group_corr_mean.index = group_labels
    group_corr_mean.columns = group_labels
    group_corr_median.index = group_labels
    group_corr_median.columns = group_labels
    group_sig_count.index = group_labels
    group_sig_count.columns = group_labels
    group_sig_percent.index = group_labels
    group_sig_percent.columns = group_labels
    
    # Calculate optimal figure size
    fig_width = max(12, num_groups * 2)
    fig_height = max(10, num_groups * 1.8)
    
    # Plot mean correlation heatmap with enhanced visibility
    plt.figure(figsize=(fig_width, fig_height))
    
    # Create annotation matrix (will display count/total and percentage)
    annot_matrix = np.empty_like(group_corr_mean.values, dtype=object)
    for i in range(num_groups):
        for j in range(num_groups):
            sig = group_sig_count.iloc[i, j]
            total = group_total_pairs.iloc[i, j]
            pct = group_sig_percent.iloc[i, j]
            
            # Only add annotation if we have pairs
            if total > 0:
                annot_matrix[i, j] = f"{sig}/{total}\n({pct:.1f}%)"
            else:
                annot_matrix[i, j] = ""
    
    # Determine font sizes based on number of groups
    if num_groups > 10:
        annot_fontsize = 10
        tick_fontsize = 12
        title_fontsize = 16
    else:
        annot_fontsize = 12
        tick_fontsize = 14
        title_fontsize = 18
    
    # Create the heatmap with annotations and enhanced visibility
    ax = sns.heatmap(
        group_corr_mean,
        cmap="coolwarm",
        linewidths=0.5,
        annot=annot_matrix,
        fmt="",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        annot_kws={'fontsize': annot_fontsize, 'family': 'Arial'},
        cbar_kws={"shrink": 0.6}
    )
    
    # Enhanced formatting
    plt.xticks(fontsize=tick_fontsize, family='Arial')
    plt.yticks(fontsize=tick_fontsize, family='Arial')
    plt.title(f"Group-Level Mean Correlation - {title}\n(Annotations: sig/total pairs (% sig), using {sig_type} p < 0.05)",
              fontsize=title_fontsize, weight='bold', family='Arial', pad=20)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15, left=0.15)
    
    # Save the figure
    safe_title = title.replace(" ", "_").replace("/", "_")
    plt.savefig(os.path.join(save_path, f"{safe_title}_group_correlation_mean_{sig_type}.tiff"),
                format='tiff', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Save group-level data to CSV for further analysis
    group_corr_mean.to_csv(os.path.join(save_path, f"{safe_title}_group_correlation_mean_{sig_type}.csv"))
    group_corr_median.to_csv(os.path.join(save_path, f"{safe_title}_group_correlation_median_{sig_type}.csv"))
    group_sig_count.to_csv(os.path.join(save_path, f"{safe_title}_group_significant_counts_{sig_type}.csv"))
    group_sig_percent.to_csv(os.path.join(save_path, f"{safe_title}_group_significant_percent_{sig_type}.csv"))
    
    print(f"Successfully created grouped correlation heatmap for {title}")

def create_top_correlations_plot(corr_df, pval_df, adj_pval_df, save_path, title, 
                                top_n=30, use_adjusted=True, min_abs_corr=0.3):
    """
    Create a bar chart of the top N strongest correlations with enhanced visibility
    """
    print(f"Creating top correlations plot for {title}...")
    
    # Select p-value matrix and significance type
    sig_df = adj_pval_df if use_adjusted else pval_df
    sig_type = "FDR-adjusted" if use_adjusted else "raw"
    
    # Create significance mask
    sig_threshold = 0.05
    sig_mask = sig_df <= sig_threshold
    
    # Create a list to store all valid correlation pairs
    corr_pairs = []
    
    # Loop through all cell type pairs
    for i, cell1 in enumerate(corr_df.columns):
        for j, cell2 in enumerate(corr_df.columns):
            # Only process each pair once and exclude self-correlations
            if i < j:
                corr_val = corr_df.loc[cell1, cell2]
                
                # Skip NaN values and values below minimum threshold
                if pd.isna(corr_val) or abs(corr_val) < min_abs_corr:
                    continue
                
                # Check if correlation is significant
                is_sig = sig_mask.loc[cell1, cell2] if not pd.isna(sig_mask.loc[cell1, cell2]) else False
                
                # Add to list of correlation pairs
                corr_pairs.append({
                    'Cell1': cell1,
                    'Cell2': cell2,
                    'Correlation': corr_val,
                    'AbsCorrelation': abs(corr_val),
                    'Significant': is_sig,
                    'Direction': 'Positive' if corr_val > 0 else 'Negative'
                })
    
    # Convert to DataFrame and sort by absolute correlation (descending)
    if not corr_pairs:
        print(f"No correlations above threshold {min_abs_corr} found for {title}.")
        return
    
    corr_df_pairs = pd.DataFrame(corr_pairs)
    corr_df_pairs = corr_df_pairs.sort_values('AbsCorrelation', ascending=False)
    
    # Take top N correlations
    top_corr = corr_df_pairs.head(top_n)
    
    # Create pair labels
    top_corr['PairLabel'] = top_corr.apply(lambda row: f"{row['Cell1']} - {row['Cell2']}", axis=1)
    
    # Create bar colors based on correlation sign and significance
    colors = []
    for _, row in top_corr.iterrows():
        if row['Significant']:
            # Darker colors for significant correlations
            colors.append('#991010' if row['Correlation'] > 0 else '#101099')  # Dark red/blue for significant
        else:
            # Lighter colors for non-significant correlations
            colors.append('#ffb6b6' if row['Correlation'] > 0 else '#b6b6ff')  # Light red/blue for non-significant
    
    # Calculate optimal figure size
    fig_width = 16
    fig_height = max(12, top_n * 0.5)
    
    # Create the plot with enhanced visibility
    plt.figure(figsize=(fig_width, fig_height))
    
    # Sort for better visualization (optional)
    # Here we sort by correlation value, so negatives come before positives
    top_corr = top_corr.sort_values('Correlation')
    
    # Determine font sizes based on number of correlations
    if top_n > 25:
        label_fontsize = 10
        tick_fontsize = 8
        title_fontsize = 14
    else:
        label_fontsize = 12
        tick_fontsize = 10
        title_fontsize = 16
    
    # Create horizontal bar chart
    bars = plt.barh(
        top_corr['PairLabel'],
        top_corr['Correlation'],
        color=colors
    )
    
    # Add a vertical line at x=0
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    # Add correlation values as text labels with enhanced visibility
    for i, bar in enumerate(bars):
        xval = bar.get_width()
        width = xval
        # Position text differently for positive vs negative bars
        if width < 0:
            plt.text(width - 0.05, bar.get_y() + bar.get_height()/2, 
                   f'{width:.2f}', ha='right', va='center', 
                   color='white', fontweight='bold', fontsize=label_fontsize-2, family='Arial')
        else:
            plt.text(width + 0.05, bar.get_y() + bar.get_height()/2, 
                   f'{width:.2f}', ha='left', va='center', 
                   color='black', fontweight='bold', fontsize=label_fontsize-2, family='Arial')
    
    # Add a legend for significance with larger fonts
    sig_patch = mpatches.Patch(color='#991010', label=f'Significant (p < 0.05, {sig_type})')
    non_sig_patch = mpatches.Patch(color='#ffb6b6', label='Non-significant')
    plt.legend(handles=[sig_patch, non_sig_patch], loc='lower right', fontsize=label_fontsize, frameon=True)
    
    # Set labels and title with enhanced formatting
    plt.xlabel('Correlation Coefficient (Spearman\'s ρ)', fontsize=label_fontsize+2, family='Arial', weight='bold')
    plt.ylabel('Cell Type Pair', fontsize=label_fontsize+2, family='Arial', weight='bold')
    plt.title(f"Top {top_n} Strongest Correlations - {title}", 
             fontsize=title_fontsize, family='Arial', weight='bold', pad=15)
    
    # Enhanced tick formatting
    plt.xticks(fontsize=tick_fontsize, family='Arial')
    plt.yticks(fontsize=tick_fontsize, family='Arial')
    
    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(left=0.35)  # Make room for long cell type names
    
    # Save the figure
    safe_title = title.replace(" ", "_").replace("/", "_")
    plt.savefig(os.path.join(save_path, f"{safe_title}_top_{top_n}_correlations_{sig_type}.tiff"),
                format='tiff', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Save the data to CSV
    top_corr.to_csv(os.path.join(save_path, f"{safe_title}_top_{top_n}_correlations_{sig_type}.csv"), index=False)
    
    print(f"Successfully created top correlations plot for {title}")

def create_interactive_correlation_viz(corr_df, pval_df, adj_pval_df, save_path, title, min_abs_corr=0.3):
    """
    Create an interactive HTML visualization of correlation data using Plotly.
    """
    print(f"Creating interactive correlation visualization for {title}...")
    
    # Create masks for significance
    raw_sig_mask = pval_df <= 0.05
    adj_sig_mask = adj_pval_df <= 0.05
    
    # Apply minimum correlation threshold
    min_corr_mask = abs(corr_df) >= min_abs_corr
    
    # Create filtered matrices
    raw_sig_corr = corr_df.copy()
    raw_sig_corr[~(raw_sig_mask & min_corr_mask)] = np.nan
    
    adj_sig_corr = corr_df.copy()
    adj_sig_corr[~(adj_sig_mask & min_corr_mask)] = np.nan
    
    # Get active cell types (have at least one significant correlation)
    active_raw = ~raw_sig_corr.isna().all(axis=1) & ~raw_sig_corr.isna().all(axis=0)
    active_adj = ~adj_sig_corr.isna().all(axis=1) & ~adj_sig_corr.isna().all(axis=0)
    
    # Get list of active cell types
    active_raw_cells = raw_sig_corr.index[active_raw].tolist()
    active_adj_cells = adj_sig_corr.index[active_adj].tolist()
    
    # Create focused correlation matrices
    raw_focused = raw_sig_corr.loc[active_raw_cells, active_raw_cells] if active_raw_cells else None
    adj_focused = adj_sig_corr.loc[active_adj_cells, active_adj_cells] if active_adj_cells else None
    
    # Create the interactive figure with enhanced styling
    fig = go.Figure()
    
    # Add all correlation heatmap
    heatmap_all = go.Heatmap(
        z=corr_df.values,
        x=corr_df.columns.tolist(),
        y=corr_df.index.tolist(),
        colorscale='RdBu_r',
        zmid=0,
        zmin=-1,
        zmax=1,
        colorbar=dict(title="Correlation", titlefont=dict(size=16, family='Arial')),
        hovertemplate='Cell Type 1: %{y}<br>Cell Type 2: %{x}<br>Correlation: %{z:.3f}<extra></extra>',
        name="All Correlations",
        visible=True
    )
    fig.add_trace(heatmap_all)
    
    # Add raw significant correlations heatmap
    if active_raw_cells:
        heatmap_raw_sig = go.Heatmap(
            z=raw_focused.values,
            x=raw_focused.columns.tolist(),
            y=raw_focused.index.tolist(),
            colorscale='RdBu_r',
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation", titlefont=dict(size=16, family='Arial')),
            hovertemplate='Cell Type 1: %{y}<br>Cell Type 2: %{x}<br>Correlation: %{z:.3f}<extra></extra>',
            name="Raw Significant",
            visible=False
        )
        fig.add_trace(heatmap_raw_sig)
    
    # Add adjusted significant correlations heatmap
    if active_adj_cells:
        heatmap_adj_sig = go.Heatmap(
            z=adj_focused.values,
            x=adj_focused.columns.tolist(),
            y=adj_focused.index.tolist(),
            colorscale='RdBu_r',
            zmid=0,
            zmin=-1,
            zmax=1,
            colorbar=dict(title="Correlation", titlefont=dict(size=16, family='Arial')),
            hovertemplate='Cell Type 1: %{y}<br>Cell Type 2: %{x}<br>Correlation: %{z:.3f}<extra></extra>',
            name="FDR Significant",
            visible=False
        )
        fig.add_trace(heatmap_adj_sig)
    
    # Create buttons for switching between views with enhanced styling
    updatemenus = [
        {
            'buttons': [
                {
                    'method': 'update',
                    'label': 'All Correlations',
                    'args': [{'visible': [True, False, False]}]
                },
                {
                    'method': 'update',
                    'label': 'Raw p < 0.05',
                    'args': [{'visible': [False, True, False]}]
                },
                {
                    'method': 'update',
                    'label': 'FDR p < 0.05',
                    'args': [{'visible': [False, False, True]}]
                }
            ],
            'direction': 'down',
            'showactive': True,
            'x': 0.1,
            'y': 1.15,
            'font': dict(size=14, family='Arial')
        }
    ]
    
    # Add the updatemenus to the figure layout with enhanced styling
    fig.update_layout(
        updatemenus=updatemenus,
        title=dict(
            text=f"Interactive Correlation Matrix - {title}",
            x=0.5,
            font=dict(size=20, family='Arial', color='black')
        ),
        width=1200,
        height=1000,
        xaxis=dict(
            tickangle=-45,
            tickfont=dict(size=12, family='Arial')
        ),
        yaxis=dict(
            autorange="reversed",
            tickfont=dict(size=12, family='Arial')
        ),
        font=dict(family='Arial', size=12)
    )
    
    # Add a note to the legend with enhanced styling
    fig.add_annotation(
        xref='paper',
        yref='paper',
        x=0.5,
        y=1.1,
        showarrow=False,
        text=f"Visualizing correlations with absolute value ≥ {min_abs_corr}",
        font=dict(size=14, family='Arial', color='black')
    )
    
    # Save the figure as an HTML file
    safe_title = title.replace(" ", "_").replace("/", "_")
    fig.write_html(os.path.join(save_path, f"{safe_title}_interactive.html"))
    
    print(f"Successfully created interactive visualization for {title}")

def create_comparison_visualization(corr_df1, pval_df1, adj_pval_df1, title1,
                                   corr_df2, pval_df2, adj_pval_df2, title2,
                                   save_path, common_title, min_corr=0.5, use_adjusted=True):
    """
    Create a visualization comparing correlation patterns between two datasets with enhanced visibility
    """
    print(f"Creating comparison visualization between {title1} and {title2}...")
    
    # Select p-value matrices and significance type
    sig_df1 = adj_pval_df1 if use_adjusted else pval_df1
    sig_df2 = adj_pval_df2 if use_adjusted else pval_df2
    sig_type = "FDR-adjusted" if use_adjusted else "raw"
    
    # Create significance masks
    sig_mask1 = sig_df1 <= 0.05
    sig_mask2 = sig_df2 <= 0.05
    
    # Apply minimum correlation threshold
    min_corr_mask1 = abs(corr_df1) >= min_corr
    min_corr_mask2 = abs(corr_df2) >= min_corr
    
    # Get sets of significant correlations
    sig_pairs1 = set()
    sig_pairs2 = set()
    
    # Get all significant pairs from dataset 1
    for i, cell1 in enumerate(corr_df1.columns):
        for j, cell2 in enumerate(corr_df1.columns):
            if i < j:  # Only process each pair once
                # Check if significant and above threshold
                is_sig = sig_mask1.loc[cell1, cell2] if not pd.isna(sig_mask1.loc[cell1, cell2]) else False
                above_threshold = min_corr_mask1.loc[cell1, cell2] if not pd.isna(min_corr_mask1.loc[cell1, cell2]) else False
                
                if is_sig and above_threshold:
                    corr_val = corr_df1.loc[cell1, cell2]
                    # Store as tuple: (cell1, cell2, correlation_value, is_positive)
                    sig_pairs1.add((cell1, cell2, corr_val, corr_val > 0))
    
    # Get all significant pairs from dataset 2
    for i, cell1 in enumerate(corr_df2.columns):
        for j, cell2 in enumerate(corr_df2.columns):
            if i < j:  # Only process each pair once
                # Check if significant and above threshold
                is_sig = sig_mask2.loc[cell1, cell2] if not pd.isna(sig_mask2.loc[cell1, cell2]) else False
                above_threshold = min_corr_mask2.loc[cell1, cell2] if not pd.isna(min_corr_mask2.loc[cell1, cell2]) else False
                
                if is_sig and above_threshold:
                    corr_val = corr_df2.loc[cell1, cell2]
                    # Store as tuple: (cell1, cell2, correlation_value, is_positive)
                    sig_pairs2.add((cell1, cell2, corr_val, corr_val > 0))
    
    # Convert to frozenset for cell pairs only (for easier comparison)
    sig_pairs_set1 = {frozenset([p[0], p[1]]) for p in sig_pairs1}
    sig_pairs_set2 = {frozenset([p[0], p[1]]) for p in sig_pairs2}
    
    # Find common and unique correlations
    common_pairs = sig_pairs_set1.intersection(sig_pairs_set2)
    unique_to_1 = sig_pairs_set1 - sig_pairs_set2
    unique_to_2 = sig_pairs_set2 - sig_pairs_set1
    
    # Create dictionary mapping from frozenset to correlation details for fast lookup
    pair_to_details1 = {frozenset([p[0], p[1]]): (p[2], p[3]) for p in sig_pairs1}
    pair_to_details2 = {frozenset([p[0], p[1]]): (p[2], p[3]) for p in sig_pairs2}
    
    print(f"Found {len(sig_pairs1)} significant correlations in {title1}")
    print(f"Found {len(sig_pairs2)} significant correlations in {title2}")
    print(f"Common correlations: {len(common_pairs)}")
    print(f"Unique to {title1}: {len(unique_to_1)}")
    print(f"Unique to {title2}: {len(unique_to_2)}")
    
    # Check for opposing correlations (positive in one, negative in the other)
    opposing = []
    for pair in common_pairs:
        corr_val1, is_pos1 = pair_to_details1[pair]
        corr_val2, is_pos2 = pair_to_details2[pair]
        
        if is_pos1 != is_pos2:
            # Get cell type names
            cell_types = list(pair)
            opposing.append((cell_types[0], cell_types[1], corr_val1, corr_val2))
    
    print(f"Opposing correlations (positive in one, negative in the other): {len(opposing)}")
    
    # 1. Create a scatter plot comparing common correlation values with enhanced visibility
    plt.figure(figsize=(12, 12))
    
    common_corr1 = []
    common_corr2 = []
    labels = []
    
    for pair in common_pairs:
        corr_val1, _ = pair_to_details1[pair]
        corr_val2, _ = pair_to_details2[pair]
        
        common_corr1.append(corr_val1)
        common_corr2.append(corr_val2)
        
        # Get cell type names for labels
        cell_types = list(pair)
        labels.append(f"{cell_types[0]} - {cell_types[1]}")
    
    # Calculate correlation between the two sets
    if common_corr1 and common_corr2:
        dataset_corr, dataset_p = spearmanr(common_corr1, common_corr2)
        dataset_corr_text = f"Correlation between datasets: ρ = {dataset_corr:.3f}, p = {dataset_p:.3e}"
    else:
        dataset_corr_text = "No common significant correlations"
    
    # Create the scatter plot with enhanced styling
    plt.scatter(common_corr1, common_corr2, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
    
    # Add labels for some points (only for strongest correlations to avoid crowding)
    for i, (x, y) in enumerate(zip(common_corr1, common_corr2)):
        if abs(x) > 0.7 or abs(y) > 0.7:
            plt.annotate(labels[i], (x, y), fontsize=10, family='Arial',
                        xytext=(5, 5), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Add diagonal reference line
    plt.plot([-1, 1], [-1, 1], 'k--', alpha=0.5, linewidth=2)
    
    # Add quadrant lines
    plt.axhline(y=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    plt.axvline(x=0, color='gray', linestyle='-', alpha=0.3, linewidth=1)
    
    # Enhanced labels and title
    plt.xlabel(f"{title1} Correlation", fontsize=16, family='Arial', weight='bold')
    plt.ylabel(f"{title2} Correlation", fontsize=16, family='Arial', weight='bold')
    plt.title(f"Comparison of Common Significant Correlations\n{dataset_corr_text}", 
             fontsize=18, family='Arial', weight='bold', pad=20)
    
    # Set axis limits and formatting
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xticks(fontsize=14, family='Arial')
    plt.yticks(fontsize=14, family='Arial')
    
    # Equal aspect ratio
    plt.axis('equal')
    plt.grid(True, alpha=0.3)
    
    # Enhanced layout
    plt.tight_layout()
    
    # Save the figure
    safe_title = common_title.replace(" ", "_").replace("/", "_")
    plt.savefig(os.path.join(save_path, f"{safe_title}_comparison_scatter.tiff"),
                format='tiff', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 2. Create a Venn diagram showing overlap between the two datasets
    try:
        from matplotlib_venn import venn2
        
        plt.figure(figsize=(14, 12))
        
        # Create sets for Venn diagram
        set1 = sig_pairs_set1
        set2 = sig_pairs_set2
        
        # Generate Venn diagram with enhanced styling
        v = venn2([set1, set2], (title1, title2))
        
        # Set colors with better visibility
        v.get_patch_by_id('10').set_color('lightcoral')  # Unique to dataset 1
        v.get_patch_by_id('01').set_color('lightskyblue')  # Unique to dataset 2
        v.get_patch_by_id('11').set_color('lightgreen')  # Common
        
        # Set alpha for better visibility
        v.get_patch_by_id('10').set_alpha(0.8)
        v.get_patch_by_id('01').set_alpha(0.8)
        v.get_patch_by_id('11').set_alpha(0.8)
        
        # Enhanced title and labels
        plt.title(f"Overlap of Significant Correlations (|r| ≥ {min_corr}, {sig_type} p < 0.05)",
                 fontsize=18, family='Arial', weight='bold', pad=20)
        
        # Save the figure
        plt.savefig(os.path.join(save_path, f"{safe_title}_venn_diagram.tiff"),
                    format='tiff', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    except ImportError:
        print("matplotlib_venn not available, skipping Venn diagram")
    
    # 3. Save detailed comparison data to CSV
    # Create DataFrames for the different correlation categories
    
    # Common correlations
    common_data = []
    for pair in common_pairs:
        cells = list(pair)
        corr_val1, is_pos1 = pair_to_details1[pair]
        corr_val2, is_pos2 = pair_to_details2[pair]
        
        common_data.append({
            'Cell1': cells[0],
            'Cell2': cells[1],
            f'{title1}_Correlation': corr_val1,
            f'{title2}_Correlation': corr_val2,
            'Direction_Match': 'Yes' if is_pos1 == is_pos2 else 'No',
            'Difference': abs(corr_val1 - corr_val2)
        })
    
    # Unique to dataset 1
    unique1_data = []
    for pair in unique_to_1:
        cells = list(pair)
        corr_val1, is_pos1 = pair_to_details1[pair]
        
        unique1_data.append({
            'Cell1': cells[0],
            'Cell2': cells[1],
            'Correlation': corr_val1,
            'Direction': 'Positive' if is_pos1 else 'Negative'
        })
    
    # Unique to dataset 2
    unique2_data = []
    for pair in unique_to_2:
        cells = list(pair)
        corr_val2, is_pos2 = pair_to_details2[pair]
        
        unique2_data.append({
            'Cell1': cells[0],
            'Cell2': cells[1],
            'Correlation': corr_val2,
            'Direction': 'Positive' if is_pos2 else 'Negative'
        })
    
    # Convert to DataFrames
    common_df = pd.DataFrame(common_data) if common_data else pd.DataFrame()
    unique1_df = pd.DataFrame(unique1_data) if unique1_data else pd.DataFrame()
    unique2_df = pd.DataFrame(unique2_data) if unique2_data else pd.DataFrame()
    
    # Sort by difference or correlation value
    if not common_df.empty:
        common_df = common_df.sort_values('Difference', ascending=False)
    if not unique1_df.empty:
        unique1_df = unique1_df.sort_values('Correlation', key=abs, ascending=False)
    if not unique2_df.empty:
        unique2_df = unique2_df.sort_values('Correlation', key=abs, ascending=False)
    
    # Save to CSV
    if not common_df.empty:
        common_df.to_csv(os.path.join(save_path, f"{safe_title}_common_correlations.csv"), index=False)
    if not unique1_df.empty:
        unique1_df.to_csv(os.path.join(save_path, f"{safe_title}_unique_to_{title1}.csv"), index=False)
    if not unique2_df.empty:
        unique2_df.to_csv(os.path.join(save_path, f"{safe_title}_unique_to_{title2}.csv"), index=False)
    
    print(f"Successfully created comparison visualization between {title1} and {title2}")
    
    return common_df, unique1_df, unique2_df

def create_dynamic_heatmap(corr_df, pval_df, adj_pval_df, save_path, title, 
                          min_corr_step=0.1, max_corr_step=0.9, step_size=0.1,
                          use_adjusted=True, compact_view=False):
    """
    Create a series of progressively filtered heatmaps with enhanced visibility
    """
    print(f"Creating dynamic heatmap for {title}...")
    
    # Select p-value matrix based on user preference
    sig_df = adj_pval_df if use_adjusted else pval_df
    sig_type = "FDR-adjusted" if use_adjusted else "raw"
    
    # Create significance mask
    sig_mask = sig_df <= 0.05
    
    # Create a list of correlation thresholds
    thresholds = np.arange(min_corr_step, max_corr_step + step_size, step_size)
    thresholds = np.round(thresholds, 2)  # Round to avoid floating point issues
    
    if compact_view:
        # Create a figure with subplots for each threshold with enhanced visibility
        n_thresholds = len(thresholds)
        n_cols = 3  # Number of columns in the subplot grid
        n_rows = (n_thresholds + n_cols - 1) // n_cols
        
        # Create a larger figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
        
        # Flatten axes array for easier indexing
        if n_rows > 1:
            axes = axes.flatten()
        
        # For each threshold, create a filtered correlation matrix
        for i, threshold in enumerate(thresholds):
            # Create mask for correlations below threshold (absolute value)
            corr_mask = abs(corr_df) < threshold
            
            # Apply both masks (significance and threshold)
            combined_mask = corr_mask | ~sig_mask
            
            # Create a copy with NaN for filtered values
            filtered_corr = corr_df.copy()
            filtered_corr[combined_mask] = np.nan
            
            # Plot heatmap on corresponding subplot
            if n_thresholds > 1:
                ax = axes[i]
            else:
                ax = axes
            
            sns.heatmap(filtered_corr, cmap="coolwarm", vmin=-1, vmax=1, 
                       ax=ax, cbar=False if i % n_cols != 0 else True, square=True,
                       xticklabels=True, yticklabels=True)
            
            ax.set_title(f"|r| ≥ {threshold}", fontsize=14, family='Arial', weight='bold')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10, family='Arial')
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10, family='Arial')
        
        # Remove empty subplots
        for i in range(n_thresholds, n_rows * n_cols):
            if n_rows > 1 or n_cols > 1:
                axes[i].axis('off')
        
        # Add a global title with enhanced formatting
        plt.suptitle(f"Dynamic Correlation Filtering - {title}\n(showing only {sig_type} p < 0.05)", 
                    fontsize=18, family='Arial', weight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.94])  # Adjust for suptitle
        
        # Save the figure
        safe_title = title.replace(" ", "_").replace("/", "_")
        plt.savefig(os.path.join(save_path, f"{safe_title}_dynamic_heatmap_compact.tiff"),
                    format='tiff', dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
    else:
        # Create a separate figure for each threshold with enhanced visibility
        for threshold in thresholds:
            # Create mask for correlations below threshold (absolute value)
            corr_mask = abs(corr_df) < threshold
            
            # Apply both masks (significance and threshold)
            combined_mask = corr_mask | ~sig_mask
            
            # Create a copy with NaN for filtered values
            filtered_corr = corr_df.copy()
            filtered_corr[combined_mask] = np.nan
            
            # Count remaining correlations and active cell types
            remaining_corrs = np.sum(~np.isnan(filtered_corr.values)) - len(filtered_corr)  # Subtract diagonal
            active_cells = np.sum(~np.isnan(filtered_corr).all(axis=1))
            
            # Calculate optimal figure size
            n_active = max(active_cells, 10)
            fig_width = max(14, n_active * 0.4)
            fig_height = max(12, n_active * 0.35)
            
            # Create a new figure with enhanced visibility
            plt.figure(figsize=(fig_width, fig_height))
            
            # Create the heatmap
            ax = sns.heatmap(filtered_corr, cmap="coolwarm", vmin=-1, vmax=1, 
                           linewidths=0.2, square=True,
                           cbar_kws={"shrink": 0.6})
            
            # Enhanced formatting
            plt.title(f"Correlation Matrix: |r| ≥ {threshold}, {sig_type} p < 0.05 - {title}\n" + 
                     f"({remaining_corrs//2} correlations, {active_cells} active cell types)",
                     fontsize=16, family='Arial', weight='bold', pad=20)
            plt.xticks(rotation=90, fontsize=12, family='Arial')
            plt.yticks(rotation=0, fontsize=12, family='Arial')
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.25, left=0.25)
            
            # Save the figure
            safe_title = title.replace(" ", "_").replace("/", "_")
            plt.savefig(os.path.join(save_path, f"{safe_title}_dynamic_heatmap_{threshold:.1f}.tiff"),
                        format='tiff', dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
    
    print(f"Successfully created dynamic heatmap for {title}")

def create_comparative_heatmap(df_igg4, df_aav, all_cell_types, title, save_path, metric_type):
    """
    Create a comparative heatmap with IgG4-RKD dataset on upper right half 
    and AAV dataset on bottom left half, with SIGNIFICANTLY ENHANCED VISIBILITY
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    from scipy.cluster import hierarchy
    from statsmodels.stats.multitest import multipletests
    from scipy.stats import spearmanr
    
    print(f"Creating comparative heatmap for {metric_type} between IgG4-RKD and AAV datasets...")
    
    # Make copies of the dataframes to avoid modifying the originals
    df_igg4_copy = df_igg4.copy()
    df_aav_copy = df_aav.copy()
    
    # Define minimum requirement for valid data
    min_observations = 3  # Minimum number of non-zero observations required
    
    # Identify valid cell types (those with enough non-zero observations in either dataset)
    valid_cell_types = []
    for cell_type in all_cell_types:
        # Add columns if missing
        if cell_type not in df_igg4_copy.columns:
            df_igg4_copy[cell_type] = 0.0
        if cell_type not in df_aav_copy.columns:
            df_aav_copy[cell_type] = 0.0
            
        # Check for sufficient data
        igg4_non_zero = np.sum(df_igg4_copy[cell_type] > 0) if cell_type in df_igg4_copy.columns else 0
        aav_non_zero = np.sum(df_aav_copy[cell_type] > 0) if cell_type in df_aav_copy.columns else 0
        
        # Consider a cell type valid if it has enough data in either dataset
        if ((igg4_non_zero >= min_observations and df_igg4_copy[cell_type].nunique() > 1) or
            (aav_non_zero >= min_observations and df_aav_copy[cell_type].nunique() > 1)):
            valid_cell_types.append(cell_type)
        else:
            print(f"WARNING: Excluding cell type {cell_type} from comparative analysis (insufficient data)")
    
    print(f"Using {len(valid_cell_types)} out of {len(all_cell_types)} cell types for comparative analysis")
    
    # Calculate correlation matrices for both datasets, but limit to valid cell types
    # Create correlation and p-value matrices for IgG4
    corr_igg4_values = np.zeros((len(all_cell_types), len(all_cell_types)))
    pval_igg4_values = np.zeros((len(all_cell_types), len(all_cell_types)))
    corr_igg4_values.fill(np.nan)
    pval_igg4_values.fill(np.nan)
    
    # Create correlation and p-value matrices for AAV
    corr_aav_values = np.zeros((len(all_cell_types), len(all_cell_types)))
    pval_aav_values = np.zeros((len(all_cell_types), len(all_cell_types)))
    corr_aav_values.fill(np.nan)
    pval_aav_values.fill(np.nan)
    
    # Calculate correlations and p-values for IgG4
    for i, col1 in enumerate(all_cell_types):
        for j, col2 in enumerate(all_cell_types):
            # Self-correlation (diagonal)
            if i == j:
                corr_igg4_values[i, j] = 1.0
                pval_igg4_values[i, j] = 0.0
                continue
                
            # Skip if either cell type doesn't have valid data
            if col1 not in valid_cell_types or col2 not in valid_cell_types:
                continue
                
            try:
                data1 = df_igg4_copy[col1]
                data2 = df_igg4_copy[col2]
                
                # Skip if either has insufficient non-zero values
                if np.sum(data1 > 0) < min_observations or np.sum(data2 > 0) < min_observations:
                    continue
                    
                # Skip if either has no variation
                if data1.nunique() <= 1 or data2.nunique() <= 1:
                    continue
                
                # Calculate correlation
                corr, pval = spearmanr(data1, data2, nan_policy='omit')
                
                if not np.isnan(corr) and not np.isnan(pval):
                    corr_igg4_values[i, j] = corr
                    pval_igg4_values[i, j] = pval
            except Exception as e:
                print(f"Error calculating IgG4 correlation for {col1} vs {col2}: {str(e)}")
    
    # Calculate correlations and p-values for AAV
    for i, col1 in enumerate(all_cell_types):
        for j, col2 in enumerate(all_cell_types):
            # Self-correlation (diagonal)
            if i == j:
                corr_aav_values[i, j] = 1.0
                pval_aav_values[i, j] = 0.0
                continue
                
            # Skip if either cell type doesn't have valid data
            if col1 not in valid_cell_types or col2 not in valid_cell_types:
                continue
                
            try:
                data1 = df_aav_copy[col1]
                data2 = df_aav_copy[col2]
                
                # Skip if either has insufficient non-zero values
                if np.sum(data1 > 0) < min_observations or np.sum(data2 > 0) < min_observations:
                    continue
                    
                # Skip if either has no variation
                if data1.nunique() <= 1 or data2.nunique() <= 1:
                    continue
                
                # Calculate correlation
                corr, pval = spearmanr(data1, data2, nan_policy='omit')
                
                if not np.isnan(corr) and not np.isnan(pval):
                    corr_aav_values[i, j] = corr
                    pval_aav_values[i, j] = pval
            except Exception as e:
                print(f"Error calculating AAV correlation for {col1} vs {col2}: {str(e)}")
    
    # Convert to dataframes
    corr_igg4_df = pd.DataFrame(corr_igg4_values, index=all_cell_types, columns=all_cell_types)
    pval_igg4_df = pd.DataFrame(pval_igg4_values, index=all_cell_types, columns=all_cell_types)
    
    corr_aav_df = pd.DataFrame(corr_aav_values, index=all_cell_types, columns=all_cell_types)
    pval_aav_df = pd.DataFrame(pval_aav_values, index=all_cell_types, columns=all_cell_types)
    
    # Apply multiple testing correction
    # Extract p-values (excluding NaNs) for correction
    igg4_pvals = pval_igg4_df.values.flatten()
    igg4_pvals = igg4_pvals[~np.isnan(igg4_pvals)]
    
    aav_pvals = pval_aav_df.values.flatten()
    aav_pvals = aav_pvals[~np.isnan(aav_pvals)]
    
    # Initialize adjusted p-value matrices with NaNs
    adj_pval_igg4 = np.full_like(pval_igg4_values, np.nan)
    adj_pval_aav = np.full_like(pval_aav_values, np.nan)
    
    # Apply Benjamini-Hochberg correction for IgG4
    if len(igg4_pvals) > 0:
        try:
            # Create a flattened index mapping to track positions
            valid_indices_igg4 = []
            valid_pvals_igg4 = []
            
            for i in range(len(all_cell_types)):
                for j in range(len(all_cell_types)):
                    if not np.isnan(pval_igg4_values[i, j]):
                        valid_indices_igg4.append((i, j))
                        valid_pvals_igg4.append(pval_igg4_values[i, j])
            
            # Apply correction
            if valid_pvals_igg4:
                _, adj_pvals_igg4, _, _ = multipletests(valid_pvals_igg4, alpha=0.05, method='fdr_bh')
                
                # Put adjusted p-values back into the matrix
                for idx, (i, j) in enumerate(valid_indices_igg4):
                    adj_pval_igg4[i, j] = adj_pvals_igg4[idx]
                
                print(f"Successfully applied FDR correction to {len(valid_pvals_igg4)} p-values for IgG4")
        except Exception as e:
            print(f"Error applying multiple testing correction for IgG4: {str(e)}")
    
    # Apply Benjamini-Hochberg correction for AAV
    if len(aav_pvals) > 0:
        try:
            # Create a flattened index mapping to track positions
            valid_indices_aav = []
            valid_pvals_aav = []
            
            for i in range(len(all_cell_types)):
                for j in range(len(all_cell_types)):
                    if not np.isnan(pval_aav_values[i, j]):
                        valid_indices_aav.append((i, j))
                        valid_pvals_aav.append(pval_aav_values[i, j])
            
            # Apply correction
            if valid_pvals_aav:
                _, adj_pvals_aav, _, _ = multipletests(valid_pvals_aav, alpha=0.05, method='fdr_bh')
                
                # Put adjusted p-values back into the matrix
                for idx, (i, j) in enumerate(valid_indices_aav):
                    adj_pval_aav[i, j] = adj_pvals_aav[idx]
                
                print(f"Successfully applied FDR correction to {len(valid_pvals_aav)} p-values for AAV")
        except Exception as e:
            print(f"Error applying multiple testing correction for AAV: {str(e)}")
    
    # Convert to dataframes
    adj_pval_igg4_df = pd.DataFrame(adj_pval_igg4, index=all_cell_types, columns=all_cell_types)
    adj_pval_aav_df = pd.DataFrame(adj_pval_aav, index=all_cell_types, columns=all_cell_types)
    
    # Create additional enhanced visualizations
    # 1. Compare the two datasets
    create_comparison_visualization(
        corr_igg4_df, pval_igg4_df, adj_pval_igg4_df, "IgG4-RKD",
        corr_aav_df, pval_aav_df, adj_pval_aav_df, "AAV",
        save_path, f"{metric_type}_Comparison", min_corr=0.5
    )
    
    # 2. Create focused matrix for each dataset
    create_focused_correlation_matrix(
        corr_igg4_df, pval_igg4_df, adj_pval_igg4_df, all_cell_types, 
        save_path, f"IgG4-RKD_{metric_type}", min_corr=0.4
    )
    
    create_focused_correlation_matrix(
        corr_aav_df, pval_aav_df, adj_pval_aav_df, all_cell_types, 
        save_path, f"AAV_{metric_type}", min_corr=0.4
    )
    
    # 3. Create correlation networks for each dataset
    create_correlation_network(
        corr_igg4_df, pval_igg4_df, adj_pval_igg4_df, 
        save_path, f"IgG4-RKD_{metric_type}", min_abs_corr=0.6
    )
    
    create_correlation_network(
        corr_aav_df, pval_aav_df, adj_pval_aav_df, 
        save_path, f"AAV_{metric_type}", min_abs_corr=0.6
    )
    
    # 4. Create top correlations plots
    create_top_correlations_plot(
        corr_igg4_df, pval_igg4_df, adj_pval_igg4_df, 
        save_path, f"IgG4-RKD_{metric_type}", top_n=20
    )
    
    create_top_correlations_plot(
        corr_aav_df, pval_aav_df, adj_pval_aav_df, 
        save_path, f"AAV_{metric_type}", top_n=20
    )
    
    # Ensure both correlation matrices have the same index and columns
    corr_igg4_df = corr_igg4_df.reindex(index=all_cell_types, columns=all_cell_types)
    corr_aav_df = corr_aav_df.reindex(index=all_cell_types, columns=all_cell_types)
    
    # Perform clustering using IgG4 data (for cell type ordering in combined heatmap)
    # Fill in missing values with zeros for clustering
    cluster_df = corr_igg4_df.copy().fillna(0)
    
    # Generate dendrograms to get clustered ordering
    try:
        # Compute linkage matrix
        linkage = hierarchy.linkage(cluster_df.values, method='average', metric='euclidean')
        
        # Get dendrogram
        dendrogram = hierarchy.dendrogram(linkage, no_plot=True)
        
        # Get the order of cell types for the clustered heatmap
        clustered_indices = dendrogram['leaves']
        clustered_cell_types = [all_cell_types[i] for i in clustered_indices]
        
        print(f"Successfully performed clustering for {len(clustered_cell_types)} cell types")
    except Exception as e:
        print(f"Error during clustering: {str(e)}. Using original cell type order.")
        clustered_cell_types = all_cell_types
    
    # Create a combined matrix where:
    # - Upper right triangle contains IgG4 correlations
    # - Lower left triangle contains AAV correlations
    combined_matrix = np.zeros((len(clustered_cell_types), len(clustered_cell_types)))
    combined_matrix.fill(np.nan)  # Fill with NaN initially
    
    # Fill in the combined matrix
    for i in range(len(clustered_cell_types)):
        for j in range(len(clustered_cell_types)):
            ci = all_cell_types.index(clustered_cell_types[i])
            cj = all_cell_types.index(clustered_cell_types[j])
            
            if i <= j:  # Upper triangle (including diagonal) - IgG4
                combined_matrix[i, j] = corr_igg4_values[ci, cj]
            else:  # Lower triangle - AAV
                combined_matrix[i, j] = corr_aav_values[ci, cj]
    
    # Convert to DataFrame for easier plotting
    combined_df = pd.DataFrame(combined_matrix, index=clustered_cell_types, columns=clustered_cell_types)
    
    # Create a mask to add a visual divider along the diagonal
    mask = np.zeros_like(combined_matrix, dtype=bool)
    
    # =====================================================================
    # MASSIVELY ENHANCED HEATMAP CREATION FOR MAXIMUM VISIBILITY
    # =====================================================================
    
    # Calculate optimal figure size for half A4 with MAXIMUM label visibility
    n_types = len(clustered_cell_types)
    
    # SIGNIFICANTLY INCREASED figure size for maximum label visibility
    fig_width = max(24, n_types * 0.6)  # Much larger width
    fig_height = max(20, n_types * 0.55)  # Much larger height
    
    # Limit maximum size but allow larger figures for better readability
    fig_width = min(fig_width, 36)
    fig_height = min(fig_height, 32)
    
    # Set font sizes optimized for visibility
    if n_types > 60:
        tick_fontsize = 8
        title_fontsize = 16
    elif n_types > 40:
        tick_fontsize = 10
        title_fontsize = 18
    elif n_types > 25:
        tick_fontsize = 12
        title_fontsize = 20
    else:
        tick_fontsize = 14
        title_fontsize = 22
    
    # Plot and save full correlation matrix with MAXIMUM VISIBILITY
    plt.figure(figsize=(fig_width, fig_height))
    
    # Set Arial font for all text
    plt.rcParams.update({'font.family': 'Arial'})
    
    try:
        # Create the heatmap with enhanced visibility settings
        cmap = sns.diverging_palette(220, 10, as_cmap=True)  # Red-Blue diverging colormap
        
        # Create the heatmap with optimized settings for label visibility
        ax = sns.heatmap(combined_df, 
                         cmap=cmap,
                         mask=mask,  # Use mask to control what's shown
                         vmin=-1, 
                         vmax=1,
                         square=True,  # Make cells square
                         linewidths=0.1,  # Very thin lines to maximize space for labels
                         cbar_kws={"shrink": 0.4, "label": "Spearman Correlation",
                                  "label_kws": {"fontsize": tick_fontsize+2, "family": "Arial"}},
                         xticklabels=True,
                         yticklabels=True)
        
        # CRITICAL: Enhanced tick parameter management for MAXIMUM visibility
        plt.xticks(rotation=90, fontsize=tick_fontsize, family='Arial', ha='center')
        plt.yticks(rotation=0, fontsize=tick_fontsize, family='Arial', ha='right')
        
        # CRITICAL: Adjust tick parameters to ensure ALL labels are visible
        ax.tick_params(axis='x', which='major', pad=2, length=2, width=1)
        ax.tick_params(axis='y', which='major', pad=2, length=2, width=1)
        
        # Ensure all tick labels are visible and not overlapping
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', 
                          fontsize=tick_fontsize, family='Arial')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right', 
                          fontsize=tick_fontsize, family='Arial')
        
        # Add diagonal line to separate halves
        ax.plot([0, len(clustered_cell_types)], [len(clustered_cell_types), 0], 'k-', linewidth=2)
        
        # Add dataset labels with enhanced visibility
        plt.text(len(clustered_cell_types)*0.75, len(clustered_cell_types)*0.25, 'IgG4-RKD', 
                 fontsize=title_fontsize, ha='center', va='center', fontweight='bold', family='Arial',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='black'))
        plt.text(len(clustered_cell_types)*0.25, len(clustered_cell_types)*0.75, 'AAV', 
                 fontsize=title_fontsize, ha='center', va='center', fontweight='bold', family='Arial',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='black'))
        
        # Set title with enhanced formatting
        plt.title(f"Comparative {title} - IgG4-RKD vs AAV (Tissue-based)", 
                 fontsize=title_fontsize+2, family='Arial', weight='bold', pad=25)
        
        # CRITICAL: Enhanced layout management to prevent label cutoff
        plt.tight_layout(pad=1.0)
        
        # CRITICAL: Manual subplot adjustment to ensure ALL labels are visible
        plt.subplots_adjust(bottom=0.3, left=0.3, right=0.92, top=0.85)
        
    except Exception as e:
        print(f"Error in creating enhanced heatmap: {str(e)}")
        # Fallback with basic settings if enhanced version fails
        ax = sns.heatmap(combined_df, cmap="coolwarm", vmin=-1, vmax=1, square=True)
        plt.xticks(rotation=90, fontsize=tick_fontsize, family='Arial')
        plt.yticks(rotation=0, fontsize=tick_fontsize, family='Arial')
    
    # Save the figure with maximum quality
    safe_title = title.replace(" ", "_").replace("/", "_")
    plt.savefig(os.path.join(save_path, f"Comparative_{safe_title}_IgG4_vs_AAV_tissue.tiff"),
                format='tiff',
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.close()
    
    # Reset font parameters
    plt.rcParams.update({'font.family': 'Arial'})
    
    print(f"Comparative heatmaps saved successfully")

# Add this new function to visualize cell type groups with enhanced visibility
def visualize_cell_type_groups(cell_groups, save_path, title="Cell_Type_Groups"):
    """
    Create visualizations showing which cell types belong to which groups with enhanced visibility
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    import numpy as np
    
    # Create a binary matrix where rows are cell types and columns are groups
    # Flatten the list of lists to get all unique cell types
    flat_cell_types = [item for sublist in cell_groups for item in sublist]
    unique_cell_types = list(dict.fromkeys(flat_cell_types))  # Preserve order while removing duplicates
    
    # Create the binary membership matrix
    membership_matrix = np.zeros((len(unique_cell_types), len(cell_groups)))
    
    for j, group in enumerate(cell_groups):
        for cell_type in group:
            if cell_type in unique_cell_types:  # Check if cell type exists in the list
                i = unique_cell_types.index(cell_type)
                membership_matrix[i, j] = 1
    
    # Convert to DataFrame for easier manipulation
    membership_df = pd.DataFrame(
        membership_matrix,
        index=unique_cell_types,
        columns=[f"Group {i+1}" for i in range(len(cell_groups))]
    )
    
    # Calculate optimal figure size with enhanced visibility
    fig_width = max(12, len(cell_groups) * 1.5)
    fig_height = max(16, len(unique_cell_types) * 0.4)
    
    # Create the heatmap with enhanced visibility
    plt.figure(figsize=(fig_width, fig_height))
    
    # Determine font sizes based on number of cell types
    if len(unique_cell_types) > 50:
        tick_fontsize = 10
        title_fontsize = 16
    else:
        tick_fontsize = 12
        title_fontsize = 18
    
    sns.heatmap(
        membership_df,
        cmap=sns.color_palette("Greens", 2),
        cbar=False,
        linewidths=0.5,
        square=False,
        xticklabels=True,
        yticklabels=True
    )
    
    # Enhanced formatting
    plt.xticks(fontsize=tick_fontsize+2, family='Arial', weight='bold')
    plt.yticks(fontsize=tick_fontsize, family='Arial')
    plt.title(f"{title} - Cell Type Group Membership", 
             fontsize=title_fontsize, family='Arial', weight='bold', pad=20)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.35, bottom=0.1)
    
    # Save the figure
    safe_title = title.replace(" ", "_").replace("/", "_")
    plt.savefig(os.path.join(save_path, f"{safe_title}_group_membership.tiff"),
                format='tiff', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Also save a group sizes bar chart with enhanced visibility
    group_sizes = [len(group) for group in cell_groups]
    group_labels = [f"Group {i+1}" for i in range(len(cell_groups))]
    
    plt.figure(figsize=(14, 8))
    bars = plt.bar(group_labels, group_sizes, color='skyblue', edgecolor='black', linewidth=1)
    
    # Add labels on bars with enhanced formatting
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 str(int(height)), ha='center', va='bottom', 
                 fontsize=14, family='Arial', weight='bold')
    
    # Enhanced formatting
    plt.title(f"{title} - Number of Cell Types per Group", 
             fontsize=18, family='Arial', weight='bold', pad=20)
    plt.xlabel("Group", fontsize=16, family='Arial', weight='bold')
    plt.ylabel("Number of Cell Types", fontsize=16, family='Arial', weight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=14, family='Arial')
    plt.yticks(fontsize=14, family='Arial')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_path, f"{safe_title}_group_sizes.tiff"),
                format='tiff', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Create a detailed text file
    with open(os.path.join(save_path, f"{safe_title}_group_details.txt"), 'w') as f:
        for i, group in enumerate(cell_groups):
            f.write(f"Group {i+1} ({len(group)} cell types):\n")
            for cell_type in group:
                f.write(f"  - {cell_type}\n")
            f.write("\n")
    
    print(f"Cell type group visualizations saved to {save_path}")

def main():
    """Main function to run the analysis with enhanced visibility using tissue area"""
    # Define paths
    data_path = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/processed_combined_SAM_scores_filtered_alldataset5_10woMonointerstitial_memcatname3cellcatname2.csv"
    base_path = "/project/mclark/SCAMPI_datasets"
    save_path = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/enhanced_correlation_analysis_tissue_cellcatname2"
    
    # Create save directory if it doesn't exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Define all cell type categories (same as original)
    all_cell_types = [
        # CD4+ T cell lineage (Blues)
        # Naive CD4+ T cells and variants
        ['NaiveCD4T'],
        
        # General CD4+ T cells
        ['OtherCD4T', 'CD4Treg', 'Th1', 'MKI+CD4T'],
        
        # CD4+ Central-Effector Memory T cells
        ['CD4Tcm-Tem'],
        
        # ICOS+ Treg variants
        ['CD4Trm'],
        
        # CD103+ CD4+ T cells
        ['CD103+CD4T'],
        
        # HLAII+CD27+ T cells
        ['HLAII+CD27+CD4T'],
        
        # T follicular helper (Tfh) cells
        ['Tfh', 'Tfr', 'Tfh1'],
        
        # TIGIT+ Tfh cells
        [],
        
        # HLAII+GZMK+ Tfh cells
        [],
        
        # GZMA+GZMK+ CD4 CTL
        ['CD4CTL'],
        
        # GZMB+ CD4 CTL
        [],
        
        # CD8+ T cell lineage (Reds)
        # Naive CD8+ T cells
        ['NaiveCD8T'],
        
        # General CD8+ T cells
        ['otherCD8T', 'CD8Treg', 'Tc1', 'MKI+CD8T'],
        
        # GZMA+GZMK+ CD8 CTL
        ['CD8CTL'],
        
        # GZMB+ CD8 CTL
        [],
        
        # CD103+ CD8+ T cells
        [],
        
        # CD8+ tissue-resident memory T cells
        ['CD8Trm'],
        
        # CD103+ CD8+ tissue-resident memory T cells
        [],
        
        # TIGIT+ CD8+ T cells
        [],
        
        # CD8+ central-effector memory T cells
        ['CD8Tcm-Tem'],
        
        # Other T and similar cells (Purples)
        # γδT cells
        ['γδT'],
        
        # Innate lymphoid cells
        ['ILC'],
        
        # NK and NKT cells
        ['ImmatureNK', 'MatureNK'],
        
        # NKT cells
        ['NKT'],
        
        # B cell lineage (Greens)
        # Naive B cells
        ['NaiveB', 'TransitionalB'],
        
        # General B cells
        ['otherB', 'Breg', 'MKI+B'],
        
        # Memory and specialized B cells
        ['CD27+B', 'GCB', 'MemoryB'],
        
        # DN B cells and Age-associated B cells
        ['DN2B', 'DN3B', 'CD11c+B', 'AgeAsscB', 'CD11c+CD27+B'],
        
        # Plasma cells (Yellow-greens)
        ['Plasmablast', 'IgG4+Plasmablast', 'Plasma', 'IgG4+Plasma'],
        
        # Myeloid lineage (Yellows and Browns)
        ['M1Macrophage', 'MPO+Macrophage', 'M2Macrophage',
         'MerTK+M1Macrophage', 'MerTK+M2Macrophage', 'GPNMB+M1Macrophage',
         'GPNMB+M2Macrophage', 'cDC1', 'cDC2', 'pDC', 'Neutrophil'],
        
        # Stromal cells (Pinks)
        ['Myofibroblast-SmoothMuscle', 'CD90+Myofibroblast', 'MMT'],
        
        # Structural cells (Grays)
        ['StressedTubular'],
        
        # Other cells
        ['Apoptotic']
    ]
    
    # Add this call to visualize cell type groups after defining all_cell_types
    print("\n=== VISUALIZING CELL TYPE GROUPS ===")
    visualize_cell_type_groups(all_cell_types, save_path, "Cell_Types")
    
    # Flatten the list of lists into a single list of cell types
    flat_all_cell_types = [item for sublist in all_cell_types for item in sublist]
    
    # Print total number of cell types for clarity
    print(f"Total number of cell types to include in matrices: {len(flat_all_cell_types)}")
    
    # Calculate pixel to mm² conversion factor based on 0.1507 μm/pixel
    # 1 pixel = 0.1507 μm = 0.0001507 mm
    # 1 pixel² = (0.0001507)² mm² = 2.2710449 × 10^-8 mm²
    # pixels_per_mm² = 1 / (2.2710449 × 10^-8) = 44,022,416.89
    pixels_per_mm2 = 44022417  # Updated conversion factor for 0.1507 μm/pixel
    
    print(f"Using pixel to mm² conversion factor: {pixels_per_mm2} pixels/mm²")
    print(f"This corresponds to {1/np.sqrt(pixels_per_mm2)*1000:.4f} μm/pixel")
    
    # Load data
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    
    # Check if CellCatName2 column exists
    if 'CellCatName2' not in df.columns:
        print("Error: CellCatName2 column not found in the dataframe")
        return
    
    # Check which cell type categories actually exist in the CellCatName2 column
    unique_cell_categories = df['CellCatName2'].unique()
    print(f"Found {len(unique_cell_categories)} unique cell categories in CellCatName2")
    
    # Compare each individual cell type against the unique categories
    existing_cell_types = [ct for ct in flat_all_cell_types if ct in unique_cell_categories]
    missing_cell_types = [ct for ct in flat_all_cell_types if ct not in unique_cell_categories]
    
    if missing_cell_types:
        print(f"Warning: The following {len(missing_cell_types)} cell types are not found in CellCatName2 values:")
        for ct in missing_cell_types[:10]:  # Print first 10 only to avoid flooding output
            print(f"  - {ct}")
        if len(missing_cell_types) > 10:
            print(f"  - ... and {len(missing_cell_types) - 10} more")
        
        print(f"Will proceed with {len(existing_cell_types)} valid cell types for counting, but all cell types will appear in matrices.")
    
    # Create a mapping of component names to datasets
    comp_to_dataset = {}
    for comp_name in df['CompName'].unique():
        # Extract dataset from the component name or from the dataframe
        if 'Dataset' in df.columns:
            original_dataset = df[df['CompName'] == comp_name]['Dataset'].iloc[0]
            # Map old dataset names to new ones
            if original_dataset == "IgG4":
                dataset = "IgG4-RKD"
            elif original_dataset == "AAV":
                dataset = "AAV"  # no change
            elif original_dataset == "NK2":
                dataset = "KC"
            else:
                dataset = original_dataset
        else:
            # Try to extract from component name
            if "IgG4" in comp_name:
                dataset = "IgG4-RKD"
            elif "AAV" in comp_name:
                dataset = "AAV"
            elif "NK2" in comp_name:
                dataset = "KC"
            else:
                print(f"Skipping {comp_name} - dataset not determined")
                continue
        
        comp_to_dataset[comp_name] = dataset
    
    # Check if 'Patient' column exists for patient-level aggregation
    if 'Patient' not in df.columns:
        print("Error: Patient column not found in the dataframe. Required for patient-level analysis.")
        return
    
    # 1. PATIENT-LEVEL ANALYSIS USING TISSUE AREA
    # ===========================================
    print("\n=== PATIENT-LEVEL ANALYSIS USING TISSUE AREA ===")
    
    # Aggregate data by patient using tissue area
    print("Aggregating data by patient using tissue area...")
    density_df_by_patient, proportion_df_by_patient = aggregate_by_patient(
        df, 
        patient_column='Patient',  # Use the Patient column for aggregation
        comp_to_dataset=comp_to_dataset,
        pixels_per_mm2=pixels_per_mm2,
        base_path=base_path,
        exclude_categories=None,
        output_dir=save_path
    )
    
    if density_df_by_patient is None or proportion_df_by_patient is None:
        print("Error: Could not aggregate data by patient using tissue area. Check the logs for details.")
        return
    
    # Save patient-level data
    density_df_by_patient.to_csv(os.path.join(save_path, "patient_level_densities_tissue.csv"), index=False)
    proportion_df_by_patient.to_csv(os.path.join(save_path, "patient_level_proportions_tissue.csv"), index=False)
    
    # 2. ENHANCED CORRELATION ANALYSIS ON PATIENT-LEVEL DATA (TISSUE-BASED)
    # ======================================================================
    
    # 2.1 Cell Proportions Correlation Analysis
    print("\n=== ENHANCED CELL PROPORTIONS CORRELATION ANALYSIS (PATIENT-LEVEL, TISSUE-BASED) ===")
    if not proportion_df_by_patient.empty:
        # Create correlation matrices for cell proportions in the entire dataset
        print("Creating correlation matrices for cell proportions across all patients...")
        corr_df_all, pval_df_all, adj_pval_df_all = create_correlation_matrix(
            proportion_df_by_patient, 
            flat_all_cell_types,
            "Patient_Level_Cell_Proportion_Correlation_Tissue", 
            save_path, 
            "All_Datasets"
        )
        
        # Create additional enhanced visualizations for the entire dataset
        if corr_df_all is not None:
            # Create focused matrices (both raw and adjusted p-values)
            create_focused_correlation_matrix(
                corr_df_all, pval_df_all, adj_pval_df_all, 
                flat_all_cell_types, save_path, "All_Datasets_Patient_Level_Cell_Proportion_Tissue",
                min_corr=0.4, use_adjusted=True
            )
            
            create_focused_correlation_matrix(
                corr_df_all, pval_df_all, adj_pval_df_all, 
                flat_all_cell_types, save_path, "All_Datasets_Patient_Level_Cell_Proportion_Tissue",
                min_corr=0.4, use_adjusted=False
            )
            
            # Create correlation network with FDR-adjusted p-values
            create_correlation_network(
                corr_df_all, pval_df_all, adj_pval_df_all, 
                save_path, "All_Datasets_Patient_Level_Cell_Proportion_Tissue",
                cell_values=proportion_df_by_patient,  # Pass cell proportion values
                value_type="Proportion",
                min_abs_corr=0.3,  # Lower threshold to include more correlations
                use_adjusted=True
            )
            
            # Add this call to create correlation network with raw p-values
            create_correlation_network(
                corr_df_all, pval_df_all, adj_pval_df_all, 
                save_path, "All_Datasets_Patient_Level_Cell_Proportion_Tissue",
                cell_values=proportion_df_by_patient,  # Pass cell proportion values
                value_type="Proportion",
                min_abs_corr=0.3,  # Lower threshold to include more correlations
                use_adjusted=False
            )
            
            # Create top correlations plots
            create_top_correlations_plot(
                corr_df_all, pval_df_all, adj_pval_df_all, 
                save_path, "All_Datasets_Patient_Level_Cell_Proportion_Tissue",
                top_n=30, use_adjusted=True
            )
            
            create_top_correlations_plot(
                corr_df_all, pval_df_all, adj_pval_df_all, 
                save_path, "All_Datasets_Patient_Level_Cell_Proportion_Tissue",
                top_n=30, use_adjusted=False
            )
            
            # Create group-level heatmap
            create_grouped_correlation_heatmap(
                corr_df_all, pval_df_all, adj_pval_df_all, 
                all_cell_types, save_path, "All_Datasets_Patient_Level_Cell_Proportion_Tissue",
                use_adjusted=True
            )
            
            create_grouped_correlation_heatmap(
                corr_df_all, pval_df_all, adj_pval_df_all, 
                all_cell_types, save_path, "All_Datasets_Patient_Level_Cell_Proportion_Tissue",
                use_adjusted=False
            )
            
            # Create dynamic heatmaps
            create_dynamic_heatmap(
                corr_df_all, pval_df_all, adj_pval_df_all, 
                save_path, "All_Datasets_Patient_Level_Cell_Proportion_Tissue",
                min_corr_step=0.3, max_corr_step=0.8, step_size=0.1,
                use_adjusted=True, compact_view=True
            )
            
            # Create interactive visualization
            create_interactive_correlation_viz(
                corr_df_all, pval_df_all, adj_pval_df_all, 
                save_path, "All_Datasets_Patient_Level_Cell_Proportion_Tissue"
            )
        
        # Create separate correlation matrices for each dataset of interest
        all_dataset_corrs_proportion = {}  # Dictionary to store correlation matrices by dataset
        
        for dataset in ["IgG4-RKD", "AAV", "KC"]:
            # Filter by dataset for proportion data
            dataset_proportion_df = proportion_df_by_patient[proportion_df_by_patient['Dataset'] == dataset]
            if not dataset_proportion_df.empty:
                print(f"Creating correlation matrices for cell proportions in {dataset} dataset...")
                corr_df, pval_df, adj_pval_df = create_correlation_matrix(
                    dataset_proportion_df, 
                    flat_all_cell_types,  # Use flattened cell types list
                    "Patient_Level_Cell_Proportion_Correlation_Tissue", 
                    save_path, 
                    dataset
                )
                
                # Store correlation matrices for later comparison
                if corr_df is not None:
                    all_dataset_corrs_proportion[dataset] = (corr_df, pval_df, adj_pval_df)
                    
                    # Create enhanced visualizations for each dataset
                    create_focused_correlation_matrix(
                        corr_df, pval_df, adj_pval_df, 
                        flat_all_cell_types, save_path, f"{dataset}_Patient_Level_Cell_Proportion_Tissue",
                        min_corr=0.4, use_adjusted=True
                    )
                    
                    create_focused_correlation_matrix(
                        corr_df, pval_df, adj_pval_df, 
                        flat_all_cell_types, save_path, f"{dataset}_Patient_Level_Cell_Proportion_Tissue",
                        min_corr=0.4, use_adjusted=False
                    )
                    
                    # Create correlation network with FDR-adjusted p-values
                    create_correlation_network(
                        corr_df, pval_df, adj_pval_df, 
                        save_path, f"{dataset}_Patient_Level_Cell_Proportion_Tissue",
                        cell_values=dataset_proportion_df,  # Pass the dataset-specific proportion values
                        value_type="Proportion",
                        min_abs_corr=0.3,  # Lower threshold to include more correlations
                        use_adjusted=True
                    )
                    
                    # Add this call to create correlation network with raw p-values
                    create_correlation_network(
                        corr_df, pval_df, adj_pval_df, 
                        save_path, f"{dataset}_Patient_Level_Cell_Proportion_Tissue",
                        cell_values=dataset_proportion_df,  # Pass the dataset-specific proportion values
                        value_type="Proportion",
                        min_abs_corr=0.3,  # Lower threshold to include more correlations
                        use_adjusted=False
                    )
                    
                    create_top_correlations_plot(
                        corr_df, pval_df, adj_pval_df, 
                        save_path, f"{dataset}_Patient_Level_Cell_Proportion_Tissue",
                        top_n=20, use_adjusted=True
                    )
                    
                    create_grouped_correlation_heatmap(
                        corr_df, pval_df, adj_pval_df, 
                        all_cell_types, save_path, f"{dataset}_Patient_Level_Cell_Proportion_Tissue",
                        use_adjusted=True
                    )
                    
                    # Add this call for grouped correlation heatmap with raw p-values
                    create_grouped_correlation_heatmap(
                        corr_df, pval_df, adj_pval_df, 
                        all_cell_types, save_path, f"{dataset}_Patient_Level_Cell_Proportion_Tissue",
                        use_adjusted=False
                    )
            else:
                print(f"No proportion data available for {dataset} dataset.")
        
        # Compare IgG4-RKD and AAV datasets if both are available
        if "IgG4-RKD" in all_dataset_corrs_proportion and "AAV" in all_dataset_corrs_proportion:
            igg4_corrs = all_dataset_corrs_proportion["IgG4-RKD"]
            aav_corrs = all_dataset_corrs_proportion["AAV"]
            
            print("Creating comparison visualization between IgG4-RKD and AAV cell proportions...")
            create_comparison_visualization(
                igg4_corrs[0], igg4_corrs[1], igg4_corrs[2], "IgG4-RKD",
                aav_corrs[0], aav_corrs[1], aav_corrs[2], "AAV",
                save_path, "Patient_Level_Cell_Proportion_Comparison_Tissue",
                min_corr=0.5, use_adjusted=True
            )
            
            create_comparison_visualization(
                igg4_corrs[0], igg4_corrs[1], igg4_corrs[2], "IgG4-RKD",
                aav_corrs[0], aav_corrs[1], aav_corrs[2], "AAV",
                save_path, "Patient_Level_Cell_Proportion_Comparison_Tissue",
                min_corr=0.5, use_adjusted=False
            )
    else:
        print("No patient-level proportion data available for correlation analysis.")
    
    # 2.2 Cell Densities Correlation Analysis (with same pattern as proportions)
    print("\n=== ENHANCED CELL DENSITIES CORRELATION ANALYSIS (PATIENT-LEVEL, TISSUE-BASED) ===")
    if not density_df_by_patient.empty:
        # Create correlation matrices for cell densities in the entire dataset
        print("Creating correlation matrices for cell densities across all patients...")
        corr_df_all, pval_df_all, adj_pval_df_all = create_correlation_matrix(
            density_df_by_patient, 
            flat_all_cell_types,  # Use flattened cell types list
            "Patient_Level_Cell_Density_Correlation_Tissue", 
            save_path, 
            "All_Datasets"
        )
        
        # Create additional enhanced visualizations for the entire dataset
        if corr_df_all is not None:
            # Create focused matrices (both raw and adjusted p-values)
            create_focused_correlation_matrix(
                corr_df_all, pval_df_all, adj_pval_df_all, 
                flat_all_cell_types, save_path, "All_Datasets_Patient_Level_Cell_Density_Tissue",
                min_corr=0.4, use_adjusted=True
            )
            
            create_focused_correlation_matrix(
                corr_df_all, pval_df_all, adj_pval_df_all, 
                flat_all_cell_types, save_path, "All_Datasets_Patient_Level_Cell_Density_Tissue",
                min_corr=0.4, use_adjusted=False
            )
            
            # Create correlation network with FDR-adjusted p-values
            create_correlation_network(
                corr_df_all, pval_df_all, adj_pval_df_all, 
                save_path, "All_Datasets_Patient_Level_Cell_Density_Tissue",
                cell_values=density_df_by_patient,  # Pass cell density values
                value_type="Density",
                min_abs_corr=0.3,  # Lower threshold to include more correlations
                use_adjusted=True
            )
            
            # Add this call to create correlation network with raw p-values
            create_correlation_network(
                corr_df_all, pval_df_all, adj_pval_df_all, 
                save_path, "All_Datasets_Patient_Level_Cell_Density_Tissue",
                cell_values=density_df_by_patient,  # Pass cell density values
                value_type="Density",
                min_abs_corr=0.3,  # Lower threshold to include more correlations
                use_adjusted=False
            )
            
            # Create top correlations plots
            create_top_correlations_plot(
                corr_df_all, pval_df_all, adj_pval_df_all, 
                save_path, "All_Datasets_Patient_Level_Cell_Density_Tissue",
                top_n=30, use_adjusted=True
            )
            
            create_top_correlations_plot(
                corr_df_all, pval_df_all, adj_pval_df_all, 
                save_path, "All_Datasets_Patient_Level_Cell_Density_Tissue",
                top_n=30, use_adjusted=False
            )
            
            # Create group-level heatmap
            create_grouped_correlation_heatmap(
                corr_df_all, pval_df_all, adj_pval_df_all, 
                all_cell_types, save_path, "All_Datasets_Patient_Level_Cell_Density_Tissue",
                use_adjusted=True
            )
            
            create_grouped_correlation_heatmap(
                corr_df_all, pval_df_all, adj_pval_df_all, 
                all_cell_types, save_path, "All_Datasets_Patient_Level_Cell_Density_Tissue",
                use_adjusted=False
            )
            
            # Create dynamic heatmaps
            create_dynamic_heatmap(
                corr_df_all, pval_df_all, adj_pval_df_all, 
                save_path, "All_Datasets_Patient_Level_Cell_Density_Tissue",
                min_corr_step=0.3, max_corr_step=0.8, step_size=0.1,
                use_adjusted=True, compact_view=True
            )
            
            # Create interactive visualization
            create_interactive_correlation_viz(
                corr_df_all, pval_df_all, adj_pval_df_all, 
                save_path, "All_Datasets_Patient_Level_Cell_Density_Tissue"
            )
        
        # Create separate correlation matrices for each dataset of interest
        all_dataset_corrs_density = {}  # Dictionary to store correlation matrices by dataset
        
        for dataset in ["IgG4-RKD", "AAV", "KC"]:
            # Filter by dataset for density data
            dataset_density_df = density_df_by_patient[density_df_by_patient['Dataset'] == dataset]
            if not dataset_density_df.empty:
                print(f"Creating correlation matrices for cell densities in {dataset} dataset...")
                corr_df, pval_df, adj_pval_df = create_correlation_matrix(
                    dataset_density_df, 
                    flat_all_cell_types,  # Use flattened cell types list
                    "Patient_Level_Cell_Density_Correlation_Tissue", 
                    save_path, 
                    dataset
                )
                
                # Store correlation matrices for later comparison
                if corr_df is not None:
                    all_dataset_corrs_density[dataset] = (corr_df, pval_df, adj_pval_df)
                    
                    # Create enhanced visualizations for each dataset
                    create_focused_correlation_matrix(
                        corr_df, pval_df, adj_pval_df, 
                        flat_all_cell_types, save_path, f"{dataset}_Patient_Level_Cell_Density_Tissue",
                        min_corr=0.4, use_adjusted=True
                    )
                    
                    create_focused_correlation_matrix(
                        corr_df, pval_df, adj_pval_df, 
                        flat_all_cell_types, save_path, f"{dataset}_Patient_Level_Cell_Density_Tissue",
                        min_corr=0.4, use_adjusted=False
                    )
                    
                    # Create correlation network with FDR-adjusted p-values
                    create_correlation_network(
                        corr_df, pval_df, adj_pval_df, 
                        save_path, f"{dataset}_Patient_Level_Cell_Density_Tissue",
                        cell_values=dataset_density_df,  # Pass the dataset-specific density values
                        value_type="Density",
                        min_abs_corr=0.3,  # Lower threshold to include more correlations
                        use_adjusted=True
                    )
                    
                    # Add this call to create correlation network with raw p-values
                    create_correlation_network(
                        corr_df, pval_df, adj_pval_df, 
                        save_path, f"{dataset}_Patient_Level_Cell_Density_Tissue",
                        cell_values=dataset_density_df,  # Pass the dataset-specific density values
                        value_type="Density",
                        min_abs_corr=0.3,  # Lower threshold to include more correlations
                        use_adjusted=False
                    )
                    
                    create_top_correlations_plot(
                        corr_df, pval_df, adj_pval_df, 
                        save_path, f"{dataset}_Patient_Level_Cell_Density_Tissue",
                        top_n=20, use_adjusted=True
                    )
                    
                    create_grouped_correlation_heatmap(
                        corr_df, pval_df, adj_pval_df, 
                        all_cell_types, save_path, f"{dataset}_Patient_Level_Cell_Density_Tissue",
                        use_adjusted=True
                    )
                    
                    # Add this call for grouped correlation heatmap with raw p-values
                    create_grouped_correlation_heatmap(
                        corr_df, pval_df, adj_pval_df, 
                        all_cell_types, save_path, f"{dataset}_Patient_Level_Cell_Density_Tissue",
                        use_adjusted=False
                    )
            else:
                print(f"No density data available for {dataset} dataset.")
        
        # Compare IgG4-RKD and AAV datasets if both are available
        if "IgG4-RKD" in all_dataset_corrs_density and "AAV" in all_dataset_corrs_density:
            igg4_corrs = all_dataset_corrs_density["IgG4-RKD"]
            aav_corrs = all_dataset_corrs_density["AAV"]
            
            print("Creating comparison visualization between IgG4-RKD and AAV cell densities...")
            create_comparison_visualization(
                igg4_corrs[0], igg4_corrs[1], igg4_corrs[2], "IgG4-RKD",
                aav_corrs[0], aav_corrs[1], aav_corrs[2], "AAV",
                save_path, "Patient_Level_Cell_Density_Comparison_Tissue",
                min_corr=0.5, use_adjusted=True
            )
            
            create_comparison_visualization(
                igg4_corrs[0], igg4_corrs[1], igg4_corrs[2], "IgG4-RKD",
                aav_corrs[0], aav_corrs[1], aav_corrs[2], "AAV",
                save_path, "Patient_Level_Cell_Density_Comparison_Tissue",
                min_corr=0.5, use_adjusted=False
            )
    else:
        print("No patient-level density data available for correlation analysis.")
    
    # 3. COMPARATIVE ANALYSIS BETWEEN IgG4-RKD AND AAV DATASETS (PATIENT-LEVEL, TISSUE-BASED)
    # =======================================================================================
    print("\n=== COMPARATIVE ANALYSIS BETWEEN IgG4-RKD AND AAV DATASETS (PATIENT-LEVEL, TISSUE-BASED) ===")
    
    # 3.1 Cell Proportions Comparative Analysis
    if not proportion_df_by_patient.empty:
        # Filter by dataset
        igg4_proportion_df = proportion_df_by_patient[proportion_df_by_patient['Dataset'] == "IgG4-RKD"]
        aav_proportion_df = proportion_df_by_patient[proportion_df_by_patient['Dataset'] == "AAV"]
        
        if not igg4_proportion_df.empty and not aav_proportion_df.empty:
            print("Creating comparative heatmap for patient-level cell proportions (tissue-based)...")
            create_comparative_heatmap(
                igg4_proportion_df,
                aav_proportion_df,
                flat_all_cell_types,
                "Patient_Level_Cell_Proportion_Correlation_Tissue",
                save_path,
                "Patient_Level_Cell_Proportion_Tissue"
            )
        else:
            print("Insufficient data for comparative proportion analysis at patient level")
    
    # 3.2 Cell Densities Comparative Analysis
    if not density_df_by_patient.empty:
        # Filter by dataset
        igg4_density_df = density_df_by_patient[density_df_by_patient['Dataset'] == "IgG4-RKD"]
        aav_density_df = density_df_by_patient[density_df_by_patient['Dataset'] == "AAV"]
        
        if not igg4_density_df.empty and not aav_density_df.empty:
            print("Creating comparative heatmap for patient-level cell densities (tissue-based)...")
            create_comparative_heatmap(
                igg4_density_df,
                aav_density_df,
                flat_all_cell_types,
                "Patient_Level_Cell_Density_Correlation_Tissue",
                save_path,
                "Patient_Level_Cell_Density_Tissue"
            )
        else:
            print("Insufficient data for comparative density analysis at patient level")
    
    print(f"\nTissue-based analysis complete. Results saved to {save_path}")


if __name__ == "__main__":
    main()
