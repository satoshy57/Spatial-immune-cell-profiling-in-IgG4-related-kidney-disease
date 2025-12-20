import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def hex_to_rgb(hex):
    """Convert hex color code to RGB tuple."""
    return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    """Convert RGB tuple to hex color code."""
    return '#{:02x}{:02x}{:02x}'.format(rgb[0],rgb[1],rgb[2])

def process_mpi_data(df, mpidf, mpicols, sdir, categories_to_use, category_column='CellCatName3'):
    """
    Process MPI data to create marker expression heatmaps for each cell category.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The filtered data
    mpidf : pandas DataFrame
        DataFrame containing MPI data
    mpicols : list
        List of MPI column names
    sdir : str
        Directory to save output files
    categories_to_use : list
        List of cell categories to include in analysis in the desired order
    category_column : str
        Column name containing cell categories (default: 'CellCatName3')
    """
    # Filter for only WC markers
    mpicols = [x for x in mpicols if ('TAF' not in x) and ('DIC' not in x)]
    mpicols = [x for x in mpicols if 'WC' in x]  # Keep only WC markers
    
    # Remove specified markers
    excluded_markers = ['CD45', 'CD298', 'COLIII', 'COLIV', 'IFN-gamma','RORgt', 'NUR77', 'mTOC']
    mpicols = [x for x in mpicols if not any(marker in x for marker in excluded_markers)]
    mpicols.sort()
    
    # Get actual categories from data that exist in the ordered list
    actual_categories = [cat for cat in categories_to_use if cat in df[category_column].unique()]
    
    # Pre-calculate all Z-scores at once
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

    mu_array = np.zeros([len(actual_categories), len(mpicols)])
    z_array = np.zeros([len(actual_categories), len(mpicols)])
    
    # Process cell categories in the specified order
    for i, cat in enumerate(actual_categories):
        mpi_df = df[df[category_column]==cat][mpi_cols]
        for j, col in enumerate(mpicols):
            mu = mpi_df[col].dropna().mean() if not mpi_df.empty else 0
            z = mpi_df[col+'_Z'].dropna().mean() if not mpi_df.empty else 0
            mu_array[i,j] = mu
            z_array[i,j] = z

    # Create plots
    markers = [x.split('_Int')[0].split('_WC')[0] for x in mpicols]  # Remove _WC suffix
    
    # Plot mean MPI
    plt.figure(figsize=(10,8), dpi=600)
    plt.imshow(mu_array, cmap='jet')
    plt.yticks(ticks=np.arange(0,len(actual_categories)), labels=actual_categories, fontsize=6)
    plt.xticks(ticks=np.arange(0,len(mpicols)), labels=markers, rotation=90, fontsize=6)
    plt.colorbar()
    plt.xlabel('Marker', fontsize=8)
    plt.ylabel(f'Cell Category ({category_column})', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(sdir, f'MeanMPI-per{category_column}.png'), dpi=600)
    plt.close()

    # Update markers for special characters
    markers = [x.replace('gamma','\u03B3') for x in markers]  
    markers = [x.replace('delta','\u03B4') for x in markers]
    
    # Plot Z-scores
    import matplotlib as mpl
    mpl.rcParams['font.size'] = 8 
    plt.figure(figsize=(10,8), dpi=600)
    plt.imshow(z_array, vmin=-3, vmax=3, cmap='seismic')
    plt.yticks(ticks=np.arange(0,len(actual_categories)), labels=actual_categories, fontsize=6)
    plt.xticks(ticks=np.arange(0,len(mpicols)), labels=markers, fontsize=6, rotation=90)
    plt.colorbar()
    plt.xlabel('Marker', fontsize=8)
    plt.ylabel(f'Cell Category ({category_column})', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(sdir, f'MeanMPIzscore-per{category_column}.png'), dpi=600)
    plt.close()
    
    return df


def plot_cell_fractions(df, sample_col, cat_col, color_dict, ordered_cats, sdir, title_prefix, filename_prefix, dataset_name, exclude_structural=False, exclude_specific=False):
    """
    Plot the fractions of different cell types for each sample.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The filtered data
    sample_col : str
        Column name for sample identifiers
    cat_col : str
        Column name for cell categories (e.g., 'CellCatName3')
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
    dataset_name : str
        Name of the dataset being processed
    exclude_structural : bool, optional
        Whether to exclude structural cells and unclassified
    exclude_specific : bool, optional
        Whether to exclude specific cells
    """
    print(f"\n=== Debug Information ===")
    if cat_col not in df.columns:
        print(f"Warning: {cat_col} not found in DataFrame columns")
        return
        
    # Define cells to exclude
    exclude_cells = ['Others']
    
    # Create a copy of the dataframe to avoid modifying the original
    plot_df = df.copy()
    
    # Filter out cells based on different exclusion criteria
    if exclude_specific and cat_col == 'CellCatName3':
        plot_df = plot_df[~plot_df[cat_col].isin(exclude_cells)]
    
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
        
        # Ensure all categories are present
        if cat_col == 'CellCatName3':
            if exclude_specific:
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
    
    # Get categories for plotting based on the ordered list
    if cat_col == 'CellCatName3':
        if exclude_specific:
            available_cats = [cat for cat in ordered_cats if cat not in exclude_cells and cat in cell_fractions.columns]
        else:
            available_cats = [cat for cat in ordered_cats if cat in cell_fractions.columns]
            if 'unclassified' in cell_fractions.columns and 'unclassified' not in available_cats:
                available_cats.append('unclassified')
    else:
        # Use consistent ordering from ordered_cats
        available_cats = [cat for cat in ordered_cats if cat in cell_fractions.columns and cat != 'sample']
    
    print(f"Final categories for plotting: {available_cats}")
    print(f"Number of categories to plot: {len(available_cats)}")
    
    # Convert color dictionary values to tuples
    color_dict = {k: tuple(v) for k, v in color_dict.items()}
    
    plt.figure(figsize=(10, 6))
    bottom = np.zeros(len(samples))
    
    # Use the ordered categories for plotting
    for cat in available_cats:
        if cat in color_dict:
            color = rgb_to_hex(color_dict[cat])
        else:
            print(f"Warning: No color defined for category '{cat}', using default gray")
            color = '#808080'
        
        if cat in cell_fractions.columns:
            values = cell_fractions[cat].values
            plt.bar(range(len(samples)), values, bottom=bottom, color=color, label=cat)
            bottom += values
    
    plt.xlabel('Samples')
    plt.ylabel('Proportion')
    
    # Set appropriate suffix based on exclusion
    suffix = ' (Excluding Others)' if exclude_specific else ''
        
    plt.title(f'{title_prefix}{suffix}')
    plt.xticks(range(len(samples)), samples, rotation=45, fontsize=8)
    
    # Create legend with the correct order
    handles, labels = plt.gca().get_legend_handles_labels()
    # Create a mapping to get the index of each label in available_cats
    label_order = {label: available_cats.index(label) for label in labels}
    # Sort handles and labels by their order in available_cats
    sorted_pairs = sorted(zip(handles, labels), key=lambda pair: label_order[pair[1]])
    sorted_handles, sorted_labels = zip(*sorted_pairs) if sorted_pairs else ([], [])
    
    plt.legend(sorted_handles, sorted_labels, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    
    # Set appropriate filename suffix
    filename_suffix = '_noOthers' if exclude_specific else ''
        
    plt.savefig(os.path.join(sdir, f'{filename_prefix}_{dataset_name}{filename_suffix}.png'), dpi=600, bbox_inches='tight')
    plt.close()


def generate_cell_proportion_tables(df, sample_col, cat_col, ordered_cats, sdir, filename_prefix, dataset_name, exclude_specific=False):
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
        Column name for cell categories (e.g., 'CellCatName3')
    ordered_cats : list
        List of ordered category names
    sdir : str
        Directory to save output files
    filename_prefix : str
        Prefix for output filenames
    dataset_name : str
        Name of the dataset being processed
    exclude_specific : bool, optional
        Whether to exclude specific cells
    """
    print(f"\nGenerating cell proportion tables for {cat_col}...")
    
    # Define cells to exclude
    exclude_cells = ['Others']
    
    # Create a copy of the dataframe to avoid modifying the original
    table_df = df.copy()
    
    # Filter out cells based on different exclusion criteria
    if exclude_specific and cat_col == 'CellCatName3':
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
    
    # Filter ordered categories based on exclusion criteria
    relevant_cats = []
    if cat_col == 'CellCatName3':
        if exclude_specific:
            relevant_cats = [cat for cat in ordered_cats if cat not in exclude_cells]
        else:
            relevant_cats = ordered_cats
    else:
        relevant_cats = ordered_cats
    
    # Calculate mean proportions across all patients
    mean_proportions = patient_proportions.mean().reset_index()
    mean_proportions.columns = ['Cell_Class', 'Mean_Proportion_Percent']
    
    # Reorder the cells according to the predefined order, but only for cells that exist in the data
    existing_cats = [cat for cat in relevant_cats if cat in patient_proportions.columns]
    
    # Add any categories that exist in the data but weren't in our ordered list
    missing_cats = [cat for cat in patient_proportions.columns if cat not in existing_cats]
    ordered_existing_cats = existing_cats + missing_cats
    
    # Reorder the patient proportions based on the predefined order
    patient_proportions = patient_proportions[ordered_existing_cats]
    
    # Transpose patient proportions for a better view (cells as rows, patients as columns)
    patient_proportions_t = patient_proportions.T
    patient_proportions_t['Mean'] = patient_proportions_t.mean(axis=1)
    
    # Format the tables
    patient_proportions_t = patient_proportions_t.round(2)
    
    # Also create a version sorted by mean
    patient_proportions_t_sorted = patient_proportions_t.sort_values(by='Mean', ascending=False)
    
    # For the mean proportions table, respect the predefined order
    ordered_mean = []
    for cat in ordered_existing_cats:
        row = mean_proportions[mean_proportions['Cell_Class'] == cat]
        if not row.empty:
            ordered_mean.append(row.iloc[0])
    
    if ordered_mean:
        ordered_mean_df = pd.DataFrame(ordered_mean)
        ordered_mean_df['Mean_Proportion_Percent'] = ordered_mean_df['Mean_Proportion_Percent'].round(2)
    else:
        # Create an empty dataframe if no categories match
        ordered_mean_df = pd.DataFrame(columns=['Cell_Class', 'Mean_Proportion_Percent'])
    
    # Set appropriate suffix based on exclusion
    suffix = '_noOthers' if exclude_specific else ''
    
    # Generate filenames
    patient_file = f'{filename_prefix}_{dataset_name}_per_patient{suffix}.csv'
    patient_sorted_file = f'{filename_prefix}_{dataset_name}_per_patient_sorted{suffix}.csv'
    mean_file = f'{filename_prefix}_{dataset_name}_mean{suffix}.csv'
    
    # Save tables
    patient_proportions_t.to_csv(os.path.join(sdir, patient_file))
    patient_proportions_t_sorted.to_csv(os.path.join(sdir, patient_sorted_file))
    ordered_mean_df.to_csv(os.path.join(sdir, mean_file), index=False)
    
    print(f"Saved patient proportions to {patient_file}")
    print(f"Saved sorted patient proportions to {patient_sorted_file}")
    print(f"Saved mean proportions to {mean_file}")
    
    return patient_proportions_t, ordered_mean_df


def create_color_dict_for_cd4t_categories(categories):
    """
    Create a color dictionary for CD4+ T cell subtypes.
    
    Parameters:
    -----------
    categories : list
        List of cell category names
    
    Returns:
    --------
    dict: Dictionary mapping categories to RGB color values, and ordered_categories list
    """
    # Define ordered CD4+ T cell categories
    ordered_categories = [
        'GZMA+GZMK+CD4CTL',
        'GZMB+CD4CTL'
    ]
    
    # Create a distinctive color palette for CD4 T cell subtypes
    # Using a range of blues, purples, and complementary colors (0-255 range)
    cell_color_dict = {
    # Vibrant primary/secondary colors
    'GZMA+GZMK+CD4CTL': [228, 26, 28],         # #E41A1C - Red
 
    # Distinct vibrant colors
    'GZMB+CD4CTL': [55, 126, 184],            # #377EB8 - Blue
    }
    
    # Generate colors for any missing categories
    import random
    missing_categories = []
    for cat in categories:
        if cat not in cell_color_dict:
            # Generate a random RGB color in blue-purple range (0-255)
            r = random.randint(50, 150)
            g = random.randint(50, 150)
            b = random.randint(170, 250)
            cell_color_dict[cat] = [r, g, b]
            missing_categories.append(cat)
            
            # Also add to the ordered categories list
            if cat not in ordered_categories:
                ordered_categories.append(cat)
    
    if missing_categories:
        print(f"Generated random colors for {len(missing_categories)} new categories:")
        for cat in missing_categories:
            print(f"  - {cat}: {cell_color_dict[cat]}")
    
    # Filter and reorder the ordered_categories list to only include categories that exist in the dataset
    ordered_categories = [cat for cat in ordered_categories if cat in categories]
    
    return cell_color_dict, ordered_categories

def analyze_cell_type_distribution(df, sdir, dataset_name, ordered_categories):
    """
    Analyze and visualize the distribution of cell categories across different markers.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The filtered data containing the CellCatName3 column
    sdir : str
        Directory to save output files
    dataset_name : str
        Name of the dataset being processed
    ordered_categories : list
        List of cell categories in the desired order
    """
    print("\nAnalyzing cell type distribution...")
    
    # Check if required column exists
    if 'CellCatName3' not in df.columns:
        print("Error: CellCatName3 column not found in DataFrame")
        return
    
    # Get markers (columns with Int-mean)
    marker_cols = [col for col in df.columns if 'Int-mean' in col and ('TAF' not in col) and ('DIC' not in col)]
    marker_cols = [col for col in marker_cols if 'WC' in col]  # Keep only WC markers
    
    # Remove specified markers
    excluded_markers = ['CD45', 'CD298', 'COLIII', 'COLIV', 'RORgt', 'IFN-gamma', 'mTOC']
    marker_cols = [col for col in marker_cols if not any(marker in col for marker in excluded_markers)]
    marker_cols.sort()
    
    # Simplified marker names for plotting
    marker_names = [col.split('_Int')[0].split('_WC')[0] for col in marker_cols]
    
    # Filter ordered categories to include only those present in the data
    cell_types = [cat for cat in ordered_categories if cat in df['CellCatName3'].unique()]
    
    # Create a dataframe to store the median marker intensity for each cell type
    intensity_data = pd.DataFrame(index=cell_types, columns=marker_names)
    
    # Calculate median intensity for each marker in each cell type
    for cell_type in cell_types:
        cell_df = df[df['CellCatName3'] == cell_type]
        for i, marker in enumerate(marker_cols):
            intensity_data.loc[cell_type, marker_names[i]] = cell_df[marker].median()
    
    # Convert to numeric data types to avoid type issues
    intensity_data = intensity_data.apply(pd.to_numeric, errors='coerce')
    
    # Normalize data (Z-score across cell types for each marker)
    import numpy as np
    normalized_data = intensity_data.copy()
    
    for marker in marker_names:
        # Extract the values as numpy array of floats
        marker_values = intensity_data[marker].values.astype(float)
        
        # Handle NaN values
        mask = ~np.isnan(marker_values)
        valid_values = marker_values[mask]
        
        # Check if we have enough valid values and if they're not all the same
        if len(valid_values) > 1 and np.std(valid_values) > 0:
            # Calculate z-scores manually
            mean_val = np.mean(valid_values)
            std_val = np.std(valid_values)
            
            # Initialize z_scores with NaN values
            z_scores = np.full_like(marker_values, np.nan)
            
            # Calculate z-score only for non-NaN values
            z_scores[mask] = (marker_values[mask] - mean_val) / std_val
            
            normalized_data[marker] = z_scores
        else:
            # If all values are the same or we have too few valid values, set z-scores to 0
            normalized_data[marker] = 0.0
            print(f"Warning: Skipping z-score calculation for marker '{marker}' - not enough valid data or zero variance")
    
    # Fill NaN values with 0 for plotting
    plot_data = normalized_data.fillna(0)
    
    # Create heatmap with cell types in the predefined order
    plt.figure(figsize=(14, 10), dpi=600)
    plt.imshow(plot_data.values, cmap='seismic', vmin=-2, vmax=2, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Z-Score', rotation=270, labelpad=20, fontsize=10)
    
    # Set tick labels
    plt.xticks(range(len(marker_names)), marker_names, rotation=90, fontsize=8)
    plt.yticks(range(len(cell_types)), cell_types, fontsize=8)
    
    # Add title and labels
    plt.title(f'Marker Expression (Z-Score) by CD4+ T Cell Subtype - {dataset_name}', fontsize=12)
    plt.xlabel('Markers', fontsize=10)
    plt.ylabel('CD4+ T Cell Subtypes', fontsize=10)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    heatmap_file = os.path.join(sdir, f'CD4TSubtype_MarkerExpression_{dataset_name}.png')
    plt.savefig(heatmap_file, dpi=600, bbox_inches='tight')
    plt.close()
    print(f"Saved CD4+ T cell subtype marker expression heatmap to {heatmap_file}")
    
    # Save the normalized expression data
    normalized_data.to_csv(os.path.join(sdir, f'CD4TSubtype_MarkerExpression_{dataset_name}.csv'))
    
    # Create hierarchical clustering of cell types based on marker expression
    try:
        from scipy.cluster.hierarchy import dendrogram, linkage
        
        # Prepare data for clustering
        cluster_data = normalized_data.fillna(0).values
        
        # Check if we have enough data for clustering
        if len(cell_types) > 1 and cluster_data.size > 0:
            try:
                # Calculate linkage
                linked = linkage(cluster_data, 'ward')
                
                # Create dendrogram figure
                plt.figure(figsize=(12, 8), dpi=600)
                dendro = dendrogram(
                    linked,
                    orientation='right',
                    labels=cell_types,
                    distance_sort='descending',
                    show_leaf_counts=True
                )
                
                plt.title(f'Hierarchical Clustering of CD4+ T Cell Subtypes - {dataset_name}', fontsize=12)
                plt.xlabel('Distance', fontsize=10)
                plt.tight_layout()
                
                # Save the dendrogram
                dendro_file = os.path.join(sdir, f'CD4TSubtype_Dendrogram_{dataset_name}.png')
                plt.savefig(dendro_file, dpi=600, bbox_inches='tight')
                plt.close()
                print(f"Saved CD4+ T cell subtype dendrogram to {dendro_file}")
                
                # Save the clustering information
                leaf_order = dendro['leaves']
                clustered_cell_types = [cell_types[i] for i in leaf_order]
                cluster_info = pd.DataFrame({
                    'Cell_Subtype': clustered_cell_types,
                    'Cluster_Order': range(len(clustered_cell_types))
                })
                cluster_file = os.path.join(sdir, f'CD4TSubtype_Clusters_{dataset_name}.csv')
                cluster_info.to_csv(cluster_file, index=False)
                print(f"Saved CD4+ T cell subtype clustering information to {cluster_file}")
            except Exception as e:
                print(f"Warning: Could not create dendrogram: {e}")
        else:
            print("Not enough cell types or data for clustering")
    except ImportError:
        print("scipy.cluster.hierarchy not available - skipping dendrogram creation")
    
    return normalized_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir', type=str, default='/project/mclark/SCAMPI_datasets/', help='Root directory')
    parser.add_argument('--outdir', type=str, default='/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/CD4CTL_CellCatName3.Analysis', help='Output directory')
    parser.add_argument('--input_file', type=str, 
                       default='/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/processed_combined_SAM_scores_filtered_alldataset5_10woMonointerstitial_memcatname3cellcatname23_CD4CTL.csv', 
                       help='Input CSV file')
    
    args, unparsed = parser.parse_known_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    
    # Load the input CSV file
    print(f"Loading input file: {args.input_file}")
    df = pd.read_csv(args.input_file, low_memory=False)
    
    # Check if Dataset column exists
    if 'Dataset' not in df.columns:
        print("Error: Dataset column not found in the input CSV. Trying to infer from Sample column...")
        if 'Sample' in df.columns:
            # Try to extract dataset from sample names
            df['Dataset'] = df['Sample'].str.extract(r'(IgG4|AAV|NK2)')
            print(f"Inferred datasets: {df['Dataset'].unique()}")
        else:
            print("Error: Cannot infer datasets. Using 'Unknown' as dataset name.")
            df['Dataset'] = 'Unknown'
    
    # Verify CellCatName3 column exists
    if 'CellCatName3' not in df.columns:
        print(f"Error: CellCatName3 column not found in the input file. Available columns: {df.columns.tolist()}")
        return
    
    # Convert Sample to string if it exists
    if 'Sample' in df.columns:
        df['Sample'] = df['Sample'].astype(str)
    
    # Process each dataset separately
    datasets = df['Dataset'].unique()
    print(f"Found {len(datasets)} datasets: {datasets}")
    
    for dataset in datasets:
        print(f"\n=== Processing dataset: {dataset} ===")
        
        # Create dataset-specific output directory
        dataset_dir = os.path.join(args.outdir, dataset)
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
        
        # Filter data for current dataset
        dataset_df = df[df['Dataset'] == dataset].copy()
        
        # Get unique cell categories for this dataset
        cell_categories = sorted(dataset_df['CellCatName3'].unique())
        print(f"Found {len(cell_categories)} unique CD4+ T cell subtypes in {dataset}: {cell_categories}")
        
        # Create color dictionary for the cell categories and get ordered categories
        cell_color_dict, ordered_categories = create_color_dict_for_cd4t_categories(cell_categories)
        
        # Process MPI data
        cols = dataset_df.columns
        mpicols = [x for x in cols if 'Int-mean' in x]
        mpidf = dataset_df[mpicols]
        
        # Process and add MPI data
        dataset_df = process_mpi_data(
            df=dataset_df, 
            mpidf=mpidf, 
            mpicols=mpicols, 
            sdir=dataset_dir, 
            categories_to_use=ordered_categories, 
            category_column='CellCatName3'
        )
        
        # Analyze cell type distribution
        analyze_cell_type_distribution(dataset_df, dataset_dir, dataset, ordered_categories)
        
        # Generate proportion tables and plots for different sample-related columns
        for sample_col in ['CompName', 'Sample', 'Patient']:
            if sample_col in dataset_df.columns and len(dataset_df[sample_col].unique()) > 1:
                print(f"\nGenerating analysis by {sample_col}")
                
                # Generate proportion tables
                generate_cell_proportion_tables(
                    df=dataset_df,
                    sample_col=sample_col,
                    cat_col='CellCatName3',
                    ordered_cats=ordered_categories,
                    sdir=dataset_dir,
                    filename_prefix=f'{sample_col}_CD4TSubtypeProportions',
                    dataset_name=dataset,
                    exclude_specific=False
                )
                
                # Generate proportion tables excluding Others
                generate_cell_proportion_tables(
                    df=dataset_df,
                    sample_col=sample_col,
                    cat_col='CellCatName3',
                    ordered_cats=ordered_categories,
                    sdir=dataset_dir,
                    filename_prefix=f'{sample_col}_CD4TSubtypeProportions',
                    dataset_name=dataset,
                    exclude_specific=True
                )
                
                # Create plots
                plot_cell_fractions(
                    df=dataset_df,
                    sample_col=sample_col,
                    cat_col='CellCatName3',
                    color_dict=cell_color_dict,
                    ordered_cats=ordered_categories,
                    sdir=dataset_dir,
                    title_prefix=f'CD4+ T Cell Subtypes by {sample_col} - {dataset}',
                    filename_prefix=f'{sample_col}_CD4TSubtypePrevalence',
                    dataset_name=dataset,
                    exclude_specific=False
                )
                
                # Create plots excluding Others
                plot_cell_fractions(
                    df=dataset_df,
                    sample_col=sample_col,
                    cat_col='CellCatName3',
                    color_dict=cell_color_dict,
                    ordered_cats=ordered_categories,
                    sdir=dataset_dir,
                    title_prefix=f'CD4+ T Cell Subtypes by {sample_col} - {dataset}',
                    filename_prefix=f'{sample_col}_CD4TSubtypePrevalence',
                    dataset_name=dataset,
                    exclude_specific=True
                )
            else:
                print(f"Skipping {sample_col} analysis - column not found or not enough unique values")

        print(f"=== Completed processing for dataset: {dataset} ===")

if __name__ == "__main__":
    main()
