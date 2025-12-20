import pandas as pd
import numpy as np
import umap
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import os
import math
import argparse
from matplotlib.patches import Rectangle
import time
from datetime import datetime

def is_marker_match(column, marker_spec):
    """
    Determine if a column matches the specified marker.
    
    Parameters:
    column (str): The column name from the dataset
    marker_spec (str): The marker specification from user
    
    Returns:
    bool: True if the column matches the marker specification
    """
    # Case 1: Exact match (full column name specified or specific pattern)
    if marker_spec in column:
        # If marker includes measurement type (e.g., _Int-mean_WC), do exact matching
        if '_Int-mean_' in marker_spec:
            return marker_spec in column
    
    # Case 2: General marker name (e.g., "CD3" without location)
    # We want to match any CD3 measurement regardless of location
    # But avoid false positives (e.g., "CD34" should not match "CD3")
    marker_base = marker_spec.split('_')[0] if '_' in marker_spec else marker_spec
    
    # Check if the marker base appears in the column as a complete marker name
    # This needs to handle cases like "CD3" vs "CD34" vs "FOXP3"
    col_parts = column.split('_')
    if len(col_parts) > 0:
        col_marker = col_parts[0]
        return col_marker == marker_base
        
    return False

# -------------------- MARKER SELECTION -------------------------
# Specify the markers you want to use for UMAP analysis here
# Leave as None to use all markers, or specify a list of markers
# 
# You can specify markers in several ways:
# 1. Just the marker name: "CD3" (will match any CD3 regardless of location)
# 2. Marker with location: "CD3_Int-mean_WC" (exact match)
# 3. Location suffix: "_Int-mean_WC" (to select all whole cell measurements)
#
# Example: SELECTED_MARKERS = ["CD3", "CD4_Int-mean_WC", "CD8_Int-mean_Mem"]
SELECTED_MARKERS = ["TCRd_Int-mean_Mem","CD86_Int-mean_WC","Ki67_Int-mean_Nuc","CD45RA_Int-mean_Mem",
"MERTK_Int-mean_WC","CD127_Int-mean_Mem","IgG4_Int-mean_WC","Foxp3_Int-mean_Nuc","cCasp3_Int-mean_WC",
"BDCA1_Int-mean_WC","CD90_Int-mean_WC","SLAMF7_Int-mean_WC","CD34_Int-mean_WC","CD56_Int-mean_Mem",
"Claudin1_Int-mean_WC","BDCA2_Int-mean_WC","PDPN_Int-mean_WC","TCF1_Int-mean_Nuc","CD163_Int-mean_WC",
"ICOS_Int-mean_Mem","CD11c_Int-mean_WC","MUC1_Int-mean_WC","CD141_Int-mean_WC","GZMB_Int-mean_WC",
"CD103_Int-mean_Mem","CD138_Int-mean_WC","CD68_Int-mean_WC","PD-1_Int-mean_Mem","CD20_Int-mean_Mem",
"GZMK_Int-mean_WC","CD69_Int-mean_WC","CLEC9A_Int-mean_WC",
"GPNMB_Int-mean_WC","CD27_Int-mean_Mem","CD8_Int-mean_Mem","HIF1a_Int-mean_WC","CD3_Int-mean_Mem",
"iNOS_Int-mean_WC","TIGIT_Int-mean_Mem","CD11b_Int-mean_WC","CD4_Int-mean_Mem",
"GZMA_Int-mean_WC","CD10_Int-mean_WC","MPO_Int-mean_WC","HLAII_Int-mean_WC","TOX_Int-mean_Nuc",
"aSMA_Int-mean_WC","T-bet_Int-mean_Nuc","CD16_Int-mean_WC"]  # Set to None to use all markers

# Define visualization options
USE_COLOR_PANEL_LEGEND = True  # True for color rectangles, False for dots
LEGEND_MARKER_SCALE = 3.0  # Scale factor for legend markers/dots

# -------------------- MARKER EXPRESSION VISUALIZATION OPTIONS -------------------------
# Colorbar range for marker expression (set to None for auto-scaling per marker)
MARKER_EXPRESSION_VMIN = 0      # Minimum value for colorbar
MARKER_EXPRESSION_VMAX = 100    # Maximum value for colorbar

# Colormap for marker expression - choose from:
# 'plasma' - purple to pink to yellow (excellent for high contrast)
# 'magma' - black to purple to pink to yellow (high contrast, dark background)
# 'viridis' - dark blue to green to yellow (perceptually uniform)
# 'inferno' - black to red to yellow (high contrast)
# 'hot' - black to red to yellow to white (classic heat map)
# 'YlOrRd' - yellow to orange to red (intuitive heat colors)
MARKER_EXPRESSION_COLORMAP = 'plasma'  # Recommended for best visibility in small panels
# --------------------------------------------------------------------------------------

# Set random seed for reproducibility
RANDOM_SEED = 42
CELLS_PER_DATASET = 10000
TOTAL_SAMPLE_SIZE = 30000

# Set your input and output paths here
INPUT_PATH = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/combined_SAM_scores_filtered_alldataset5_10woMonointerstitial.csv"
BASE_OUTPUT_DIR = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/umaps5_10woMonointerstitial"

# -------------------- NEW OPTIONS -------------------------
# Cell types to exclude from certain visualizations
EXCLUDE_CELL_TYPES = ["RBC", "TubuleBackground","unclassified"]

# Cell types/categories to highlight in specialized visualizations
# Set to None to disable this feature
HIGHLIGHT_CELL_TYPES = None
HIGHLIGHT_CELL_CATEGORIES = None #["CD4T", "CD8T", "OtherLymphoid", "Humoral", "Myeloid","Stromal"]

# Color for other/background cells in highlight visualizations
OTHER_CELLS_COLOR = [0.9, 0.9, 0.9]  # Light gray - same as 'Other datasets'

# New darker color for unclassified cells - changed from almost white to light gray
UNCLASSIFIED_COLOR = [0.9, 0.9, 0.9]  # Medium gray that's more visible
# ----------------------------------------------------------------

color_dict = {
    # T cell (Cool blue for CD4, Warm red for CD8, Purple for specialized)
    'NaiveCD4T': [100/255, 150/255, 255/255],          # Light blue
    'CD4T': [50/255, 100/255, 255/255],                # Medium blue
    'ICOS+Treg': [0/255, 50/255, 255/255],             # Deep blue
    'CD103+CD4Treg': [0/255, 0/255, 200/255],          # Navy blue
    'HLAII+CD27+Treg': [65/255, 105/255, 225/255],     # Royal blue
    'Tfh': [70/255, 130/255, 180/255],                 # Steel blue
    'TIGIT+Tfh': [30/255, 144/255, 255/255],           # Dodger blue
    'HLAII+GZMK+Tfh': [0/255, 191/255, 255/255],       # Deep sky blue
    'GZMA+GZMK+CD4CTL': [135/255, 206/255, 250/255],   # Light sky blue - modified from CD4CTL
    'GZMB+CD4CTL': [135/255, 206/255, 235/255],        # Sky blue - new 
    'CD4Tcm-Tem': [176/255, 196/255, 222/255],         # Light steel blue
    'NaiveCD8T': [255/255, 160/255, 160/255],          # Light red
    'CD8T': [255/255, 99/255, 71/255],                 # Tomato red
    'GZMA+GZMK+CD8CTL': [220/255, 20/255, 60/255],     # Crimson - modified from GZMA+CD8CTL
    'GZMB+CD8CTL': [233/255, 150/255, 122/255],        # Dark salmon - new
    'CD103+CD8Treg': [178/255, 34/255, 34/255],        # Fire brick red
    'CD8Trm': [139/255, 0/255, 0/255],                 # Dark red
    'CD103+CD8Trm': [165/255, 42/255, 42/255],         # Brown red
    'TIGIT+CD8T': [128/255, 0/255, 0/255],             # Maroon
    'CD8Tcm-Tem': [205/255, 92/255, 92/255],           # Indian red
    'γδT': [147/255, 112/255, 219/255],                # Medium purple
    'ILC': [138/255, 43/255, 226/255],                 # Blue violet
    'ImmatureNK': [148/255, 0/255, 211/255],           # Dark violet
    'MatureNK': [153/255, 50/255, 204/255],            # Dark orchid
    'NKT': [186/255, 85/255, 211/255],                 # Medium orchid
    
    # Humoral cell (Green gradient)
    'NaiveB': [144/255, 238/255, 144/255],             # Light green
    'TransitionalB': [124/255, 252/255, 0/255],        # Lawn green
    'B': [34/255, 139/255, 34/255],                    # Forest green
    'CD27+B': [0/255, 128/255, 0/255],                 # Green
    'DN2B': [0/255, 100/255, 0/255],                   # Dark green
    'AgeAsscB': [46/255, 139/255, 87/255],             # Sea green
    'Plasmablast': [60/255, 179/255, 113/255],         # Medium sea green
    'IgG4+Plasmablast': [32/255, 178/255, 170/255],    # Light sea green
    'Plasma': [143/255, 188/255, 143/255],             # Dark sea green
    'IgG4+Plasma': [152/255, 251/255, 152/255],        # Pale green
    
    # Myeloid cell (Yellow-Brown gradient)
    'M1Macrophage': [255/255, 193/255, 37/255],             # Golden yellow
    'MPO+Macrophage': [255/255, 185/255, 15/255],           # Dark golden yellow
    'M2Macrophage': [255/255, 165/255, 0/255],              # Orange
    'MerTK+M1Macrophage': [255/255, 140/255, 0/255],        # Dark orange
    'MerTK+M2Macrophage': [255/255, 127/255, 80/255],       # Coral
    'GPNMB+M1Macrophage': [255/255, 99/255, 71/255],        # Tomato
    'GPNMB+M2Macrophage': [255/255, 69/255, 0/255],         # Red-orange
    'cDC1': [244/255, 164/255, 96/255],                     # Sandy brown
    'cDC2': [218/255, 160/255, 33/255],                     # Golden yellow
    'pDC': [250/255, 237/255, 125/255],                     # Pale yellow
    'Neutrophil': [255/255, 228/255, 181/255],              # Moccasin
    
    # Stromal cell (Pink shades)
    'Myofibroblast-SmoothMuscle': [199/255, 21/255, 133/255],   # Medium violet red
    'CD90+Myofibroblast': [219/255, 112/255, 147/255],          # Pale violet red
    'MMT': [255/255, 20/255, 147/255],                          # Deep pink

    # Structural cell (Gray-range colors)
    'ProximalTubular': [105/255, 105/255, 105/255],           # Dim gray - changed from ProximalTubule
    'DistalTubular': [128/255, 128/255, 128/255],            # Gray
    'StressedTubular': [169/255, 169/255, 169/255],          # Dark gray
    'Endothelial': [190/255, 190/255, 190/255],              # Gray
    'LymphaticEndothelial': [211/255, 211/255, 211/255],     # Light gray
    'Podocyte': [220/255, 220/255, 220/255],                 # Gainsboro
    'PEC': [245/255, 245/255, 245/255],                      # White smoke

    # Other (Distinctive colors)
    'Apoptotic': [102/255, 51/255, 153/255],                 # Dark purple
    'RBC': [255/255, 0/255, 0/255],                          # Pure red
    'TubuleBackground': [224/255, 255/255, 255/255],         # Light cyan
        
    # Unclassified (Changed to be more visible - Medium gray)
    'unclassified': UNCLASSIFIED_COLOR                      # Changed to medium gray for better visibility
}

# Updated broad categories dictionary to match changes
broad_categories = {
    'unclassified': {
        'color': UNCLASSIFIED_COLOR,  # Changed to medium gray for visibility
        'cells': ['unclassified']
    },
    'CD4T': {
        'color': [50/255, 100/255, 255/255],  # Medium blue
        'cells': ['NaiveCD4T', 'CD4T', 'ICOS+Treg', 'CD103+CD4Treg', 'HLAII+CD27+Treg',
                 'Tfh', 'TIGIT+Tfh', 'HLAII+GZMK+Tfh', 'GZMA+GZMK+CD4CTL', 'GZMB+CD4CTL', 'CD4Tcm-Tem']
    },
    'CD8T': {
        'color': [220/255, 20/255, 60/255],  # Crimson
        'cells': ['NaiveCD8T', 'CD8T', 'GZMA+GZMK+CD8CTL', 'GZMB+CD8CTL', 'CD103+CD8Treg', 'CD8Trm',
                 'CD103+CD8Trm', 'TIGIT+CD8T', 'CD8Tcm-Tem']
    },
    'OtherLymphoid': {
        'color': [147/255, 112/255, 219/255],  # Medium purple
        'cells': ['γδT', 'ILC', 'ImmatureNK', 'MatureNK', 'NKT']
    },
    'Humoral': {
        'color': [34/255, 139/255, 34/255],  # Forest green
        'cells': ['NaiveB', 'TransitionalB','B', 'CD27+B', 'DN2B', 'AgeAsscB', 'Plasmablast', 'IgG4+Plasmablast','Plasma', 'IgG4+Plasma']
    },
    'Myeloid': {
        'color': [255/255, 165/255, 0/255],  # Orange
        'cells': ['M1Macrophage', 'MPO+Macrophage', 'M2Macrophage',
                 'MerTK+M1Macrophage', 'MerTK+M2Macrophage', 'GPNMB+M1Macrophage',
                 'GPNMB+M2Macrophage', 'cDC1', 'cDC2', 'pDC', 'Neutrophil']
    },
    'Stromal': {
        'color': [219/255, 112/255, 147/255],  # Pale violet red
        'cells': ['Myofibroblast-SmoothMuscle', 'CD90+Myofibroblast', 'MMT']
    },
    'Structural': {
        'color': [169/255, 169/255, 169/255],  # Dark gray
        'cells': ['ProximalTubular', 'DistalTubular', 'StressedTubular', 'Endothelial',
                 'LymphaticEndothelial', 'Podocyte', 'PEC']
    },
    'Other': {
        'color': [102/255, 51/255, 153/255],  # Dark purple
        'cells': ['Apoptotic', 'RBC', 'TubuleBackground']
    }
}
# Dataset color dictionary and annotation mapping
dataset_colors = {
    'IgG4': [0, 0, 1],       # Blue
    'AAV': [1, 0, 1],        # Magenta
    'NK2': [0, 1, 0],        # Green
    'other': [0.9, 0.9, 0.9]  # Light grey for other datasets
}

# Dataset annotation mapping
dataset_annotations = {
    'IgG4': 'IgG4RKD',
    'AAV': 'AAV',
    'NK2': 'KC'
}

def setup_output_directories(base_dir, timestamp=None):
    """
    Create a simplified directory structure for organizing output files.
    
    Parameters:
    base_dir (str): Base directory for all outputs
    timestamp (str, optional): Timestamp string to append to directory names
    
    Returns:
    dict: Dictionary containing paths for all output directories and files
    """
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define simplified directory structure
    dirs = {
        'base': base_dir,
        'markers': os.path.join(base_dir, "markers"),
        'by_dataset': os.path.join(base_dir, "by_dataset"),
        'filtered': os.path.join(base_dir, "filtered"),  # New directory for filtered visualizations
        'highlighted': os.path.join(base_dir, "highlighted")  # New directory for highlighted visualizations
    }
    
    # Create all directories
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    
    # Define output file paths as strings
    paths = {
        'cell_types': os.path.join(dirs['base'], "cell_types.tiff"),
        'cell_categories': os.path.join(dirs['base'], "cell_categories.tiff"),
        'datasets': os.path.join(dirs['base'], "datasets.tiff"),
        'combined_markers': os.path.join(dirs['markers'], "combined_markers.tiff"),
        'markers_dir': dirs['markers'],
        'by_dataset_dir': dirs['by_dataset'],
        'filtered_dir': dirs['filtered'],  # Path to filtered visualizations directory
        'highlighted_dir': dirs['highlighted'],  # Path to highlighted visualizations directory
        'log_file': os.path.join(dirs['base'], f"umap_analysis_log_{timestamp}.txt")
    }
    
    # Create a readme file with run information
    with open(os.path.join(dirs['base'], "README.txt"), 'w') as f:
        f.write(f"UMAP Visualization Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Input file: {INPUT_PATH}\n")
        f.write(f"Directory structure:\n")
        f.write(f"- cell_types.tiff: Main UMAP showing detailed cell types\n")
        f.write(f"- cell_categories.tiff: UMAP showing broad cell categories\n")
        f.write(f"- datasets.tiff: UMAP showing dataset distribution\n")
        f.write(f"- markers/: Directory containing individual marker expression UMAPs\n")
        f.write(f"  - combined_markers.tiff: Combined visualization of all markers\n")
        f.write(f"- by_dataset/: Directory containing dataset-specific UMAPs\n")
        f.write(f"- filtered/: Directory containing UMAPs with specific cell types excluded\n")
        f.write(f"- highlighted/: Directory containing UMAPs with specific cell types highlighted\n")
    
    return dirs, paths

def process_data(file_path, selected_markers=None, log_file=None):
    """
    Process the data for all UMAP visualizations.
    Sample 10,000 cells from each dataset (IgG4, AAV, NK2) for a total of 30,000 cells.
    
    Parameters:
    file_path (str): Path to the input CSV file
    selected_markers (list, optional): List of specific marker names to include in the analysis.
                                      If None, use all Int-mean_WC markers.
                                      Names are matched if they appear anywhere in the column name.
    log_file (str, optional): Path to log file for output
    """
    # Function to log messages to both console and file
    def log_message(message):
        print(message)
        if log_file:
            with open(log_file, 'a') as f:
                f.write(message + "\n")
    
    # Read the CSV file
    log_message("Reading CSV file...")
    df = pd.read_csv(file_path, low_memory=False)
    total_cells = len(df)
    log_message(f"Total cells in the dataset: {total_cells:,}")
    
    # Check which datasets are present
    available_datasets = df['Dataset'].unique()
    log_message(f"Available datasets: {', '.join(available_datasets)}")
    
    # Sample 10,000 cells from each dataset
    sampled_dfs = []
    for dataset in ['IgG4', 'AAV', 'NK2']:
        if dataset in available_datasets:
            dataset_df = df[df['Dataset'] == dataset]
            dataset_count = len(dataset_df)
            if dataset_count > CELLS_PER_DATASET:
                log_message(f"Sampling {CELLS_PER_DATASET:,} cells from {dataset} (total: {dataset_count:,})")
                dataset_df = dataset_df.sample(n=CELLS_PER_DATASET, random_state=RANDOM_SEED)
            else:
                log_message(f"Using all {dataset_count:,} cells from {dataset} (fewer than {CELLS_PER_DATASET:,})")
            sampled_dfs.append(dataset_df)
        else:
            log_message(f"Warning: Dataset '{dataset}' not found in input CSV")
    
    # Combine the sampled dataframes
    if not sampled_dfs:
        raise ValueError("No datasets (IgG4, AAV, NK2) found in input CSV")
    
    df_sampled = pd.concat(sampled_dfs)
    total_sampled = len(df_sampled)
    log_message(f"Total cells after balanced sampling: {total_sampled:,}")
    
    # Get all marker intensity columns (Int-mean_WC)
    all_marker_cols = [col for col in df_sampled.columns if 'Int-mean_WC' in col]
    log_message(f"Total available markers: {len(all_marker_cols)}")
    
    # Apply marker selection if specified
    if selected_markers is not None:
        # Filter marker columns that match the specified marker criteria
        marker_cols = []
        location_patterns = set()
        
        # First, identify any location-specific patterns (e.g., _Int-mean_WC)
        for marker in selected_markers:
            if marker.startswith('_') and '_Int-mean_' in marker:
                location_patterns.add(marker)
        
        # Process all columns to find matches
        for col in all_marker_cols:
            # Check if column matches a location pattern
            for pattern in location_patterns:
                if pattern in col:
                    marker_cols.append(col)
                    break
            
            # Check if column matches a specified marker
            for marker in selected_markers:
                if not marker.startswith('_') and is_marker_match(col, marker):
                    marker_cols.append(col)
                    break
        
        # Remove duplicates (in case a column matched multiple criteria)
        marker_cols = list(set(marker_cols))
        
        log_message(f"Using {len(marker_cols)} selected markers out of {len(all_marker_cols)} total markers")
        
        # Print the selected markers and those that weren't found
        log_message("Selected markers that were found:")
        found_markers = set()
        for col in marker_cols:
            # Extract marker name and location for display
            parts = col.split('_Int-mean_')
            if len(parts) == 2:
                marker_name = parts[0]
                location = parts[1]
                log_message(f"- {marker_name} ({location})")
            else:
                log_message(f"- {col}")
                
            # Track which marker terms were successfully matched
            for marker in selected_markers:
                if is_marker_match(col, marker) or (marker.startswith('_') and marker in col):
                    found_markers.add(marker)
        
        # Report any markers that weren't found
        not_found = set(selected_markers) - found_markers
        if not_found:
            log_message("\nWARNING: The following specified markers were not found in the data:")
            for marker in not_found:
                log_message(f"- {marker}")
            
        # Print the number of selected markers vs. found
        log_message(f"\n{len(found_markers)} out of {len(selected_markers)} specified markers were found in the data")
    else:
        marker_cols = all_marker_cols
        log_message(f"Using all {len(marker_cols)} available markers")
    
    # If no markers remain after filtering, raise an error
    if not marker_cols:
        raise ValueError("No markers match the selection criteria")
    
    # Check if MemCatName column exists and use it for cell types
    if 'MemCatName' in df_sampled.columns:
        log_message("Found 'MemCatName' column - using for cell type annotations")
        # Create a mapping of cell types to colors
        cell_types = df_sampled['MemCatName'].unique()
        cell_types = [ct for ct in cell_types if ct in color_dict]
        log_message(f"Found {len(cell_types)} unique cell types in 'MemCatName'")
    else:
        # Fallback to original method if MemCatName doesn't exist
        log_message("Warning: 'MemCatName' column not found, falling back to cell type columns")
        cell_types = [col for col in df_sampled.columns if col in color_dict.keys() and col != 'Unclassified']
        log_message(f"Number of cell type categories: {len(cell_types)}")
    
    # Extract the features for UMAP (marker intensities)
    X = df_sampled[marker_cols].values
    
    # Scale the features
    log_message("Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform UMAP
    log_message("Performing UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=RANDOM_SEED)
    embedding = reducer.fit_transform(X_scaled)
    
    return embedding, df_sampled, cell_types, marker_cols

def add_color_legend(ax, labels, colors, title=None, bbox_to_anchor=(1.05, 1), use_color_panel=True, fontsize=10):
    """
    Add a color panel legend or scatter point legend to the specified axis.
    
    Parameters:
    ax (Axes): The matplotlib axis to add the legend to
    labels (list): List of labels for each color
    colors (list): List of RGB colors corresponding to labels
    title (str, optional): Title for the legend
    bbox_to_anchor (tuple): Position for the legend
    use_color_panel (bool): If True, use color rectangles; if False, use scatter points
    fontsize (int): Font size for the legend
    """
    if use_color_panel:
        # Create a list of colored rectangles
        handles = [Rectangle((0, 0), 1, 1, color=color, linewidth=1, edgecolor='black') for color in colors]
        
        # Create the legend with rectangles
        legend = ax.legend(handles, labels, 
                         bbox_to_anchor=bbox_to_anchor,
                         loc='upper left',
                         fontsize=fontsize,
                         title=title,
                         title_fontsize=fontsize+2)
    else:
        # Create a series of colored scatter points for the legend
        # This uses the existing legend but with larger markers
        scatter_handles = []
        for color in colors:
            scatter = ax.scatter([], [], c=[color], s=100)
            scatter_handles.append(scatter)
        
        # Create the legend with scatter points
        legend = ax.legend(scatter_handles, labels, 
                         bbox_to_anchor=bbox_to_anchor,
                         loc='upper left',
                         fontsize=fontsize,
                         title=title,
                         title_fontsize=fontsize+2,
                         markerscale=1.5)  # Increase marker size in legend
    
    return legend

def plot_cell_types_umap(embedding, df, cell_types, output_path, legend_marker_scale=3.0, 
                       use_color_panel=True, dataset_filter=None, axis_limits=None,
                       exclude_cell_types=None, highlight_cell_types=None):
    """
    Generate detailed UMAP visualization using MemCatName for cell types, optionally filtered by dataset.
    
    Parameters:
    embedding (ndarray): UMAP coordinates
    df (DataFrame): DataFrame with cell type data
    cell_types (list): List of cell type names (either from MemCatName or cell type columns)
    output_path (str): Path to save the plot
    legend_marker_scale (float): Scale factor for legend markers
    use_color_panel (bool): Whether to use color panels or dots in legend
    dataset_filter (str, optional): Filter to a specific dataset
    axis_limits (tuple, optional): Tuple containing (x_min, x_max, y_min, y_max) for consistent axes
    exclude_cell_types (list, optional): List of cell types to exclude from visualization
    highlight_cell_types (list, optional): List of cell types to highlight (others will be shown in gray)
    """
    dataset_label = ""
    filter_label = ""
    highlight_label = ""
    
    # Filter by dataset if specified
    if dataset_filter:
        mask = df['Dataset'] == dataset_filter
        if not mask.any():
            print(f"Warning: No cells found for dataset {dataset_filter}")
            return
        
        # Create a view of the data with only this dataset
        filtered_df = df[mask].copy()
        filtered_embedding = embedding[mask]
        dataset_label = f" - {dataset_annotations.get(dataset_filter, dataset_filter)} Dataset"
    else:
        filtered_df = df
        filtered_embedding = embedding
    
    # Create filter label if excluding cell types
    if exclude_cell_types:
        filter_label = f" (Excluding {', '.join(exclude_cell_types)})"
    
    # Create highlight label if highlighting specific cell types
    if highlight_cell_types:
        highlight_label = f" (Highlighting {', '.join(highlight_cell_types)})"
    
    print(f"Generating cell types UMAP visualization{' for ' + dataset_filter if dataset_filter else ''}{filter_label}{highlight_label}...")
    fig, ax = plt.subplots(figsize=(14, 12))  # Make figure wider to accommodate larger legend
    
    # Create dictionaries to store scatter plots and their labels/colors
    scatter_plots = {}
    used_labels = set()
    
    # Check if we're using MemCatName column
    if 'MemCatName' in filtered_df.columns:
        # Include 'unclassified' as a valid cell type (lowercase/uppercase check)
        valid_cell_types = list(cell_types) # Make a copy of the list
        
        # For highlighting mode, first plot all cells as background
        if highlight_cell_types:
            # Plot all cells as background first
            plt.scatter(filtered_embedding[:, 0],
                      filtered_embedding[:, 1],
                      c=[OTHER_CELLS_COLOR],
                      alpha=0.5,
                      s=10,
                      label="Other cells")
            
            # Add "Other cells" to the legend
            scatter_plots['other_cells'] = plt.scatter([], [], c=[OTHER_CELLS_COLOR], alpha=0.7, s=10)
            used_labels.add('other_cells')
            
            # Now plot only the highlighted cell types
            for cell_type in highlight_cell_types:
                if cell_type in color_dict:
                    mask = filtered_df['MemCatName'] == cell_type
                    if mask.any():
                        scatter = plt.scatter(filtered_embedding[mask, 0],
                                            filtered_embedding[mask, 1],
                                            c=[color_dict[cell_type]],
                                            alpha=1.0,  # Higher alpha for highlighted cells
                                            s=15)  # Larger size for highlighted cells
                        scatter_plots[cell_type] = scatter
                        used_labels.add(cell_type)
        else:
            # Standard visualization mode (no highlighting)
            
            # Plot unclassified cells first (those with MemCatName='unclassified' or missing from recognized list)
            unclassified_mask = filtered_df['MemCatName'].str.lower() == 'unclassified'
            if unclassified_mask.any():
                scatter = plt.scatter(filtered_embedding[unclassified_mask, 0],
                                    filtered_embedding[unclassified_mask, 1],
                                    c=[color_dict.get('unclassified', UNCLASSIFIED_COLOR)],
                                    alpha=0.7,
                                    s=10)
                scatter_plots['unclassified'] = scatter
                used_labels.add('unclassified')
            
            # Also check for cells with no recognized cell type
            unknown_types_mask = ~filtered_df['MemCatName'].isin(valid_cell_types) & ~unclassified_mask
            if unknown_types_mask.any():
                scatter = plt.scatter(filtered_embedding[unknown_types_mask, 0],
                                    filtered_embedding[unknown_types_mask, 1],
                                    c=[color_dict.get('unclassified', UNCLASSIFIED_COLOR)],
                                    alpha=0.7,
                                    s=10)
                if 'unclassified' not in used_labels:
                    scatter_plots['unclassified'] = scatter
                    used_labels.add('unclassified')
            
            # Plot each cell type that exists in the data
            for cell_type in valid_cell_types:
                # Skip excluded cell types
                if exclude_cell_types and cell_type in exclude_cell_types:
                    continue
                    
                if cell_type in color_dict:
                    mask = filtered_df['MemCatName'] == cell_type
                    if mask.any():
                        scatter = plt.scatter(filtered_embedding[mask, 0],
                                            filtered_embedding[mask, 1],
                                            c=[color_dict[cell_type]],
                                            alpha=0.7,
                                            s=10)
                        scatter_plots[cell_type] = scatter
                        used_labels.add(cell_type)
    else:
        # Fallback to original method using cell type columns
        
        # For highlighting mode, first plot all cells as background
        if highlight_cell_types:
            # Plot all cells as background first
            plt.scatter(filtered_embedding[:, 0],
                      filtered_embedding[:, 1],
                      c=[OTHER_CELLS_COLOR],
                      alpha=0.5,
                      s=10)
            
            # Add "Other cells" to the legend
            scatter_plots['other_cells'] = plt.scatter([], [], c=[OTHER_CELLS_COLOR], alpha=0.7, s=10)
            used_labels.add('other_cells')
            
            # Now plot only the highlighted cell types
            for col in highlight_cell_types:
                if col in filtered_df.columns:
                    mask = filtered_df[col] > 0
                    if mask.any():
                        scatter = plt.scatter(filtered_embedding[mask, 0],
                                            filtered_embedding[mask, 1],
                                            c=[color_dict[col]],
                                            alpha=1.0,  # Higher alpha for highlighted cells
                                            s=15)  # Larger size for highlighted cells
                        scatter_plots[col] = scatter
                        used_labels.add(col)
        else:
            # Standard visualization (no highlighting)
            
            # First, identify unclassified cells
            # For exclusion mode, we need to modify which columns we consider
            if exclude_cell_types:
                included_types = [ct for ct in cell_types if ct not in exclude_cell_types]
            else:
                included_types = cell_types
                
            classified_mask = filtered_df[included_types].sum(axis=1) > 0
            unclassified_mask = ~classified_mask
            
            # Plot unclassified cells first
            if unclassified_mask.any():
                scatter = plt.scatter(filtered_embedding[unclassified_mask, 0],
                                    filtered_embedding[unclassified_mask, 1],
                                    c=[color_dict.get('unclassified', UNCLASSIFIED_COLOR)],
                                    alpha=0.7,
                                    s=10)
                scatter_plots['unclassified'] = scatter
                used_labels.add('unclassified')
            
            # Plot classified cells
            for col in cell_types:
                # Skip excluded cell types
                if exclude_cell_types and col in exclude_cell_types:
                    continue
                    
                mask = filtered_df[col] > 0
                if mask.any():
                    scatter = plt.scatter(filtered_embedding[mask, 0],
                                        filtered_embedding[mask, 1],
                                        c=[color_dict[col]],
                                        alpha=0.7,
                                        s=10)
                    scatter_plots[col] = scatter
                    used_labels.add(col)
    
    # Create ordered legend based on color_dict order
    legend_labels = []
    legend_colors = []
    legend_handles = []
    
    # If in highlight mode, add "Other cells" first
    if highlight_cell_types and 'other_cells' in used_labels:
        legend_labels.append('Other cells')
        legend_colors.append(OTHER_CELLS_COLOR)
        legend_handles.append(scatter_plots['other_cells'])
    
    # Then add 'unclassified' if present and not in highlight mode
    elif 'unclassified' in used_labels:
        legend_labels.append('unclassified')
        legend_colors.append(color_dict.get('unclassified', UNCLASSIFIED_COLOR))
        legend_handles.append(scatter_plots['unclassified'])
    
    # Then add cell types in the order they appear in color_dict (or highlight_cell_types if specified)
    if highlight_cell_types:
        # For highlight mode, use the exact order from highlight_cell_types
        for cell_type in highlight_cell_types:
            if cell_type in used_labels:
                legend_labels.append(cell_type)
                legend_colors.append(color_dict[cell_type])
                legend_handles.append(scatter_plots[cell_type])
    else:
        # For normal mode, follow color_dict order
        for cell_type in color_dict.keys():
            if cell_type in used_labels and cell_type != 'unclassified':
                legend_labels.append(cell_type)
                legend_colors.append(color_dict[cell_type])
                legend_handles.append(scatter_plots[cell_type])
    
    total_cells = len(filtered_df)
    plt.title(f'Cell Types UMAP Visualization{dataset_label}{filter_label}{highlight_label}\n'
             f'(Based on marker intensities, n={total_cells:,} cells)', fontsize=14)
    plt.xlabel('UMAP1', fontsize=12)
    plt.ylabel('UMAP2', fontsize=12)
    
    # Set consistent axes limits if provided
    if axis_limits:
        ax.set_xlim(axis_limits[0], axis_limits[1])
        ax.set_ylim(axis_limits[2], axis_limits[3])
    
    # Use custom ordered legend
    if use_color_panel:
        # Create a list of colored rectangles
        handles = [Rectangle((0, 0), 1, 1, color=color, linewidth=1, edgecolor='black') for color in legend_colors]
        legend = ax.legend(handles, legend_labels, 
                          bbox_to_anchor=(1.05, 1),
                          loc='upper left',
                          fontsize=10,
                          title_fontsize=12)
    else:
        # Use the scatter plot handles directly for the legend
        legend = ax.legend(legend_handles, legend_labels, 
                          bbox_to_anchor=(1.05, 1),
                          loc='upper left',
                          fontsize=10,
                          title_fontsize=12,
                          markerscale=legend_marker_scale)
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='tiff')
    plt.close()
    print(f"Cell types visualization saved to: {output_path}")
    
    return ax.get_xlim(), ax.get_ylim()

def plot_cell_categories_umap(embedding, df, cell_types, output_path, legend_marker_scale=3.0, 
                          use_color_panel=True, dataset_filter=None, axis_limits=None,
                          exclude_cell_types=None, highlight_categories=None):
    """
    Generate broad categories UMAP visualization, optionally filtered by dataset.
    
    Parameters:
    embedding (ndarray): UMAP coordinates
    df (DataFrame): DataFrame with cell type data
    cell_types (list): List of cell type names (either from MemCatName or cell type columns)
    output_path (str): Path to save the plot
    legend_marker_scale (float): Scale factor for legend markers
    use_color_panel (bool): Whether to use color panels or dots in legend
    dataset_filter (str, optional): Filter to a specific dataset
    axis_limits (tuple, optional): Tuple containing (x_min, x_max, y_min, y_max) for consistent axes
    exclude_cell_types (list, optional): List of cell types to exclude from visualization
    highlight_categories (list, optional): List of cell categories to highlight (others will be shown in gray)
    """
    dataset_label = ""
    filter_label = ""
    highlight_label = ""
    
    # Filter by dataset if specified
    if dataset_filter:
        mask = df['Dataset'] == dataset_filter
        if not mask.any():
            print(f"Warning: No cells found for dataset {dataset_filter}")
            return
        
        # Create a view of the data with only this dataset
        filtered_df = df[mask].copy()
        filtered_embedding = embedding[mask]
        dataset_label = f" - {dataset_annotations.get(dataset_filter, dataset_filter)} Dataset"
    else:
        filtered_df = df
        filtered_embedding = embedding
    
    # Create filter label if excluding cell types
    if exclude_cell_types:
        filter_label = f" (Excluding {', '.join(exclude_cell_types)})"
    
    # Create highlight label if highlighting specific categories
    if highlight_categories:
        highlight_label = f" (Highlighting {', '.join(highlight_categories)})"
    
    print(f"Generating cell categories UMAP visualization{' for ' + dataset_filter if dataset_filter else ''}{filter_label}{highlight_label}...")
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create dictionaries to store scatter plots and track which categories are used
    scatter_plots = {}
    used_categories = set()
    
    # Check if we're using MemCatName column
    if 'MemCatName' in filtered_df.columns:
        # Create a mapping from cell types to broad categories
        cell_to_category = {}
        for category, info in broad_categories.items():
            for cell in info['cells']:
                cell_to_category[cell] = category
        
        # For highlighting mode, first plot all cells as background
        if highlight_categories:
            # Plot all cells as background first
            plt.scatter(filtered_embedding[:, 0],
                      filtered_embedding[:, 1],
                      c=[OTHER_CELLS_COLOR],
                      alpha=0.5,
                      s=10)
            
            # Add "Other categories" to the legend
            scatter_plots['other_categories'] = plt.scatter([], [], c=[OTHER_CELLS_COLOR], alpha=0.7, s=10)
            used_categories.add('other_categories')
            
            # Now plot only the highlighted categories
            for category in highlight_categories:
                if category in broad_categories:
                    # Get cells in this category
                    info = broad_categories[category]
                    
                    # Check if any cell types need to be excluded
                    if exclude_cell_types:
                        category_cells = [cell for cell in cell_types if cell in info['cells'] and cell not in exclude_cell_types]
                    else:
                        category_cells = [cell for cell in cell_types if cell in info['cells']]
                    
                    if category_cells:
                        category_mask = filtered_df['MemCatName'].isin(category_cells)
                        if category_mask.any():
                            scatter = plt.scatter(filtered_embedding[category_mask, 0],
                                              filtered_embedding[category_mask, 1],
                                              c=[info['color']],
                                              alpha=1.0,  # Higher alpha for highlighted categories
                                              s=15)  # Larger size for highlighted categories
                            scatter_plots[category] = scatter
                            used_categories.add(category)
        else:
            # Standard visualization (no highlighting)
            
            # First handle unclassified cells - check for exact match in 'MemCatName'
            unclassified_mask = filtered_df['MemCatName'].str.lower() == 'unclassified'
            
            # Also check for cells not in any recognized category
            unknown_types_mask = ~filtered_df['MemCatName'].isin(cell_to_category.keys()) & ~unclassified_mask
            
            # Combine both masks for all unclassified cells
            combined_unclassified_mask = unclassified_mask | unknown_types_mask
            
            if combined_unclassified_mask.any():
                scatter = plt.scatter(filtered_embedding[combined_unclassified_mask, 0],
                                  filtered_embedding[combined_unclassified_mask, 1],
                                  c=[broad_categories['unclassified']['color']],
                                  alpha=0.7,
                                  s=10)
                scatter_plots['unclassified'] = scatter
                used_categories.add('unclassified')
            
            # Plot each broad category
            for category, info in broad_categories.items():
                if category != 'unclassified':
                    # Check if this category contains any excluded cell types
                    if exclude_cell_types:
                        # If any of the category's cells are in the excluded list, 
                        # we need to filter them out
                        category_cells = [cell for cell in cell_types if cell in info['cells'] and cell not in exclude_cell_types]
                    else:
                        category_cells = [cell for cell in cell_types if cell in info['cells']]
                    
                    if category_cells:
                        category_mask = filtered_df['MemCatName'].isin(category_cells)
                        if category_mask.any():
                            scatter = plt.scatter(filtered_embedding[category_mask, 0],
                                              filtered_embedding[category_mask, 1],
                                              c=[info['color']],
                                              alpha=0.7,
                                              s=10)
                            scatter_plots[category] = scatter
                            used_categories.add(category)
    else:
        # Fallback to original method using cell type columns
        
        # For highlighting mode, first plot all cells as background
        if highlight_categories:
            # Plot all cells as background first
            plt.scatter(filtered_embedding[:, 0],
                      filtered_embedding[:, 1],
                      c=[OTHER_CELLS_COLOR],
                      alpha=0.5,
                      s=10)
            
            # Add "Other categories" to the legend
            scatter_plots['other_categories'] = plt.scatter([], [], c=[OTHER_CELLS_COLOR], alpha=0.7, s=10)
            used_categories.add('other_categories')
            
            # Now plot only the highlighted categories
            for category in highlight_categories:
                if category in broad_categories:
                    info = broad_categories[category]
                    
                    # Filter out excluded cell types
                    if exclude_cell_types:
                        category_cols = [col for col in info['cells'] if col in cell_types and col not in exclude_cell_types]
                    else:
                        category_cols = [col for col in info['cells'] if col in cell_types]
                    
                    if category_cols:
                        category_mask = filtered_df[category_cols].sum(axis=1) > 0
                        if category_mask.any():
                            scatter = plt.scatter(filtered_embedding[category_mask, 0],
                                              filtered_embedding[category_mask, 1],
                                              c=[info['color']],
                                              alpha=1.0,  # Higher alpha for highlighted categories
                                              s=15)  # Larger size for highlighted categories
                            scatter_plots[category] = scatter
                            used_categories.add(category)
        else:
            # Standard visualization (no highlighting)
            # Create masks for broad categories
            for category, info in broad_categories.items():
                if category == 'unclassified':
                    # For exclusion mode, we need to modify which columns we consider
                    if exclude_cell_types:
                        included_types = [ct for ct in cell_types if ct not in exclude_cell_types]
                    else:
                        included_types = cell_types
                    
                    # Handle unclassified cells
                    classified_mask = filtered_df[included_types].sum(axis=1) > 0
                    category_mask = ~classified_mask
                else:
                    # Create mask for cells in this broad category
                    # Only include cells that are in both the category's list and the available cell types
                    # Also filter out excluded cell types
                    if exclude_cell_types:
                        category_cols = [col for col in info['cells'] if col in cell_types and col not in exclude_cell_types]
                    else:
                        category_cols = [col for col in info['cells'] if col in cell_types]
                    
                    if category_cols:
                        category_mask = filtered_df[category_cols].sum(axis=1) > 0
                    else:
                        category_mask = pd.Series(False, index=filtered_df.index)
                
                if category_mask.any():
                    scatter = plt.scatter(filtered_embedding[category_mask, 0],
                                      filtered_embedding[category_mask, 1],
                                      c=[info['color']],
                                      alpha=0.7,
                                      s=10)
                    scatter_plots[category] = scatter
                    used_categories.add(category)
    
    # Create ordered legend based on broad_categories order
    legend_labels = []
    legend_colors = []
    legend_handles = []
    
    # If in highlight mode, add "Other categories" first
    if highlight_categories and 'other_categories' in used_categories:
        legend_labels.append('Other categories')
        legend_colors.append(OTHER_CELLS_COLOR)
        legend_handles.append(scatter_plots['other_categories'])
    
    # Then add categories in specific order
    if highlight_categories:
        # For highlight mode, use the exact order from highlight_categories
        for category in highlight_categories:
            if category in used_categories:
                legend_labels.append(category)
                legend_colors.append(broad_categories[category]['color'])
                legend_handles.append(scatter_plots[category])
    else:
        # For normal mode, follow broad_categories order
        # Add 'unclassified' first if present
        if 'unclassified' in used_categories:
            legend_labels.append('unclassified')
            legend_colors.append(broad_categories['unclassified']['color'])
            legend_handles.append(scatter_plots['unclassified'])
        
        # Then add all other categories
        for category in broad_categories.keys():
            if category in used_categories and category != 'unclassified':
                legend_labels.append(category)
                legend_colors.append(broad_categories[category]['color'])
                legend_handles.append(scatter_plots[category])
    
    total_cells = len(filtered_df)
    plt.title(f'Cell Categories UMAP Visualization{dataset_label}{filter_label}{highlight_label}\n'
             f'(Based on marker intensities, n={total_cells:,} cells)', fontsize=14)
    plt.xlabel('UMAP1', fontsize=12)
    plt.ylabel('UMAP2', fontsize=12)
    
    # Set consistent axes limits if provided
    if axis_limits:
        ax.set_xlim(axis_limits[0], axis_limits[1])
        ax.set_ylim(axis_limits[2], axis_limits[3])
    
    # Use custom ordered legend
    if use_color_panel:
        # Create a list of colored rectangles
        handles = [Rectangle((0, 0), 1, 1, color=color, linewidth=1, edgecolor='black') for color in legend_colors]
        legend = ax.legend(handles, legend_labels, 
                          bbox_to_anchor=(1.05, 1),
                          loc='upper left',
                          fontsize=10,
                          title_fontsize=12)
    else:
        # Use the scatter plot handles directly for the legend
        legend = ax.legend(legend_handles, legend_labels, 
                          bbox_to_anchor=(1.05, 1),
                          loc='upper left',
                          fontsize=10,
                          title_fontsize=12,
                          markerscale=legend_marker_scale)
    
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='tiff')
    plt.close()
    print(f"Cell categories visualization saved to: {output_path}")

def plot_filtered_umaps(embedding, df, cell_types, output_dir, exclude_cell_types, axis_limits=None):
    """
    Generate UMAP visualizations with specific cell types excluded.
    
    Parameters:
    embedding (ndarray): UMAP coordinates
    df (DataFrame): DataFrame with cell type data
    cell_types (list): List of cell type names (either from MemCatName or cell type columns)
    output_dir (str): Directory to save the filtered plots
    exclude_cell_types (list): List of cell types to exclude from visualization
    axis_limits (tuple, optional): Tuple containing (x_min, x_max, y_min, y_max) for consistent axes
    """
    if not exclude_cell_types:
        print("No cell types specified for exclusion, skipping filtered UMAP generation.")
        return
    
    print(f"Generating filtered UMAPs excluding: {', '.join(exclude_cell_types)}...")
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate excluded cell types label for filenames
    excluded_str = "_".join([cell_type.replace(" ", "_") for cell_type in exclude_cell_types])
    
    # Cell types UMAP with excluded cell types
    cell_types_output_path = os.path.join(output_dir, f"cell_types_excluded_{excluded_str}.tiff")
    plot_cell_types_umap(
        embedding, df, cell_types, cell_types_output_path,
        legend_marker_scale=LEGEND_MARKER_SCALE,
        use_color_panel=USE_COLOR_PANEL_LEGEND,
        axis_limits=axis_limits,
        exclude_cell_types=exclude_cell_types
    )
    
    # Cell categories UMAP with excluded cell types
    categories_output_path = os.path.join(output_dir, f"cell_categories_excluded_{excluded_str}.tiff")
    plot_cell_categories_umap(
        embedding, df, cell_types, categories_output_path,
        legend_marker_scale=LEGEND_MARKER_SCALE,
        use_color_panel=USE_COLOR_PANEL_LEGEND,
        axis_limits=axis_limits,
        exclude_cell_types=exclude_cell_types
    )
    
    # Also create dataset-specific filtered UMAPs
    for dataset in ['IgG4', 'AAV', 'NK2']:
        if dataset in df['Dataset'].unique():
            # Cell types UMAP for this dataset with excluded cell types
            dataset_cell_types_path = os.path.join(output_dir, f"{dataset}_cell_types_excluded_{excluded_str}.tiff")
            plot_cell_types_umap(
                embedding, df, cell_types, dataset_cell_types_path,
                legend_marker_scale=LEGEND_MARKER_SCALE,
                use_color_panel=USE_COLOR_PANEL_LEGEND,
                dataset_filter=dataset,
                axis_limits=axis_limits,
                exclude_cell_types=exclude_cell_types
            )
            
            # Cell categories UMAP for this dataset with excluded cell types
            dataset_categories_path = os.path.join(output_dir, f"{dataset}_cell_categories_excluded_{excluded_str}.tiff")
            plot_cell_categories_umap(
                embedding, df, cell_types, dataset_categories_path,
                legend_marker_scale=LEGEND_MARKER_SCALE,
                use_color_panel=USE_COLOR_PANEL_LEGEND,
                dataset_filter=dataset,
                axis_limits=axis_limits,
                exclude_cell_types=exclude_cell_types
            )
    
    print(f"Completed generating filtered UMAPs in: {output_dir}")

def plot_highlighted_umaps(embedding, df, cell_types, output_dir, 
                        highlight_cell_types=None, highlight_categories=None, 
                        exclude_cell_types=None, axis_limits=None):
    """
    Generate UMAP visualizations with specific cell types or categories highlighted.
    
    Parameters:
    embedding (ndarray): UMAP coordinates
    df (DataFrame): DataFrame with cell type data
    cell_types (list): List of cell type names (either from MemCatName or cell type columns)
    output_dir (str): Directory to save the highlighted plots
    highlight_cell_types (list, optional): List of cell types to highlight
    highlight_categories (list, optional): List of cell categories to highlight
    exclude_cell_types (list, optional): List of cell types to exclude from visualization
    axis_limits (tuple, optional): Tuple containing (x_min, x_max, y_min, y_max) for consistent axes
    """
    if not highlight_cell_types and not highlight_categories:
        print("No cell types or categories specified for highlighting, skipping highlighted UMAP generation.")
        return
    
    print("Generating highlighted UMAPs...")
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate highlighted cell types UMAPs if specified
    if highlight_cell_types:
        highlighted_types_str = "_".join([cell_type.replace(" ", "_") for cell_type in highlight_cell_types])
        
        # Cell types UMAP with highlighted cell types
        cell_types_output_path = os.path.join(output_dir, f"cell_types_highlighted_{highlighted_types_str}.tiff")
        plot_cell_types_umap(
            embedding, df, cell_types, cell_types_output_path,
            legend_marker_scale=LEGEND_MARKER_SCALE,
            use_color_panel=USE_COLOR_PANEL_LEGEND,
            axis_limits=axis_limits,
            exclude_cell_types=exclude_cell_types,
            highlight_cell_types=highlight_cell_types
        )
        
        # Also create dataset-specific highlighted UMAPs
        for dataset in ['IgG4', 'AAV', 'NK2']:
            if dataset in df['Dataset'].unique():
                dataset_cell_types_path = os.path.join(output_dir, f"{dataset}_cell_types_highlighted_{highlighted_types_str}.tiff")
                plot_cell_types_umap(
                    embedding, df, cell_types, dataset_cell_types_path,
                    legend_marker_scale=LEGEND_MARKER_SCALE,
                    use_color_panel=USE_COLOR_PANEL_LEGEND,
                    dataset_filter=dataset,
                    axis_limits=axis_limits,
                    exclude_cell_types=exclude_cell_types,
                    highlight_cell_types=highlight_cell_types
                )
    
    # Generate highlighted categories UMAPs if specified
    if highlight_categories:
        highlighted_categories_str = "_".join([category.replace(" ", "_") for category in highlight_categories])
        
        # Cell categories UMAP with highlighted categories
        categories_output_path = os.path.join(output_dir, f"cell_categories_highlighted_{highlighted_categories_str}.tiff")
        plot_cell_categories_umap(
            embedding, df, cell_types, categories_output_path,
            legend_marker_scale=LEGEND_MARKER_SCALE,
            use_color_panel=USE_COLOR_PANEL_LEGEND,
            axis_limits=axis_limits,
            exclude_cell_types=exclude_cell_types,
            highlight_categories=highlight_categories
        )
        
        # Also create dataset-specific highlighted UMAPs
        for dataset in ['IgG4', 'AAV', 'NK2']:
            if dataset in df['Dataset'].unique():
                dataset_categories_path = os.path.join(output_dir, f"{dataset}_cell_categories_highlighted_{highlighted_categories_str}.tiff")
                plot_cell_categories_umap(
                    embedding, df, cell_types, dataset_categories_path,
                    legend_marker_scale=LEGEND_MARKER_SCALE,
                    use_color_panel=USE_COLOR_PANEL_LEGEND,
                    dataset_filter=dataset,
                    axis_limits=axis_limits,
                    exclude_cell_types=exclude_cell_types,
                    highlight_categories=highlight_categories
                )
    
    print(f"Completed generating highlighted UMAPs in: {output_dir}")

def plot_marker_expression_umaps(embedding, df, marker_cols, output_dir, axis_limits=None, exclude_cell_types=None):
    """
    Generate UMAP visualizations for each marker showing expression intensity
    with a standardized colorbar range and improved colormap for visibility.
    
    Parameters:
    embedding (ndarray): UMAP coordinates
    df (DataFrame): DataFrame with marker expression data
    marker_cols (list): List of marker column names
    output_dir (str): Directory to save the plots
    axis_limits (tuple, optional): Tuple containing (x_min, x_max, y_min, y_max) for consistent axes
    exclude_cell_types (list, optional): List of cell types to exclude from visualization
    """
    excluded_text = ""
    
    # Filter out excluded cell types if specified
    if exclude_cell_types:
        excluded_text = f" (Excluding {', '.join(exclude_cell_types)})"
        filter_label = f"_excluded_{'_'.join([ct.replace(' ', '_') for ct in exclude_cell_types])}"
        
        # Create a copy of the data for filtering
        filtered_df = df.copy()
        filtered_embedding = embedding.copy()
        
        # Check if we're using MemCatName column or cell type columns
        if 'MemCatName' in filtered_df.columns:
            # Filter using MemCatName
            exclude_mask = ~filtered_df['MemCatName'].isin(exclude_cell_types)
            filtered_df = filtered_df[exclude_mask].copy()
            filtered_embedding = filtered_embedding[exclude_mask]
            print(f"Excluded {(~exclude_mask).sum()} cells with cell types: {', '.join(exclude_cell_types)}")
        else:
            # Filter using cell type columns
            # Create a mask for cells that belong to any excluded cell type
            exclude_cols = [col for col in exclude_cell_types if col in filtered_df.columns]
            if exclude_cols:
                exclude_mask = filtered_df[exclude_cols].sum(axis=1) == 0
                filtered_df = filtered_df[exclude_mask].copy()
                filtered_embedding = filtered_embedding[exclude_mask]
                print(f"Excluded {(~exclude_mask).sum()} cells with cell types: {', '.join(exclude_cell_types)}")
    else:
        filtered_df = df
        filtered_embedding = embedding
        filter_label = ""
    
    print(f"Generating marker expression UMAP visualizations{excluded_text}...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # If filtering, create a subfolder for filtered marker plots
    if exclude_cell_types:
        filtered_output_dir = os.path.join(output_dir, f"filtered{filter_label}")
        os.makedirs(filtered_output_dir, exist_ok=True)
        # Use this as the output directory
        target_output_dir = filtered_output_dir
    else:
        target_output_dir = output_dir
    
    total_cells = len(filtered_df)
    # Plot each marker's expression
    for marker in marker_cols:
        # Get marker name and location for the title
        parts = marker.split('_Int-mean_')
        if len(parts) == 2:
            marker_name = parts[0]
            location = parts[1]
            title_name = f"{marker_name}"
        else:
            # Fallback for unexpected format
            marker_name = marker.replace('Int-mean_WC', '').strip('_ ')
            title_name = marker_name
        
        # Create figure
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        
        # Get marker expression values
        expression = filtered_df[marker].values
        
        # Create scatter plot with expression color using improved settings
        scatter = plt.scatter(filtered_embedding[:, 0], filtered_embedding[:, 1],
                            c=expression, 
                            cmap=MARKER_EXPRESSION_COLORMAP,
                            s=5, alpha=0.8,
                            vmin=MARKER_EXPRESSION_VMIN,
                            vmax=MARKER_EXPRESSION_VMAX)
        
        # Add colorbar with increased size
        cbar = plt.colorbar(scatter, shrink=0.75, pad=0.02)
        cbar.set_label(f'Expression Level', fontsize=12)
        cbar.ax.tick_params(labelsize=10)
        
        plt.title(f'UMAP - {title_name} Expression{excluded_text}\n' 
                f'(n={total_cells:,} cells)', fontsize=14)
        plt.xlabel('UMAP1', fontsize=12)
        plt.ylabel('UMAP2', fontsize=12)
        
        # Set consistent axes limits if provided
        if axis_limits:
            ax.set_xlim(axis_limits[0], axis_limits[1])
            ax.set_ylim(axis_limits[2], axis_limits[3])
        
        plt.tight_layout()
        
        # Save figure
        sanitized_marker_name = marker_name.replace(" ", "_").replace("/", "_")
        if 'location' in locals():
            sanitized_location = location.replace(" ", "_").replace("/", "_")
            output_path = os.path.join(target_output_dir, f'umap_{sanitized_marker_name}_{sanitized_location}.tiff')
        else:
            output_path = os.path.join(target_output_dir, f'umap_{sanitized_marker_name}.tiff')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='tiff')
        plt.close()
        
    print(f"Saved {len(marker_cols)} marker expression UMAP visualizations to: {target_output_dir}")


def plot_combined_marker_umaps(embedding, df, marker_cols, output_path, axis_limits=None, exclude_cell_types=None):
    """
    Generate a single figure containing all marker expression UMAPs in a grid layout,
    arranged alphabetically and excluding TAF* and DAPI markers with standardized colorbar range.
    
    Parameters:
    embedding (ndarray): UMAP coordinates
    df (DataFrame): DataFrame with marker expression data
    marker_cols (list): List of marker column names
    output_path (str): Path to save the plot
    axis_limits (tuple, optional): Tuple containing (x_min, x_max, y_min, y_max) for consistent axes
    exclude_cell_types (list, optional): List of cell types to exclude from visualization
    """
    excluded_text = ""
    
    # Filter out excluded cell types if specified
    if exclude_cell_types:
        excluded_text = f" (Excluding {', '.join(exclude_cell_types)})"
        filter_label = f"_excluded_{'_'.join([ct.replace(' ', '_') for ct in exclude_cell_types])}"
        
        # Create a copy of the data for filtering
        filtered_df = df.copy()
        filtered_embedding = embedding.copy()
        
        # Check if we're using MemCatName column or cell type columns
        if 'MemCatName' in filtered_df.columns:
            # Filter using MemCatName
            exclude_mask = ~filtered_df['MemCatName'].isin(exclude_cell_types)
            filtered_df = filtered_df[exclude_mask].copy()
            filtered_embedding = filtered_embedding[exclude_mask]
            print(f"Excluded {(~exclude_mask).sum()} cells with cell types: {', '.join(exclude_cell_types)}")
        else:
            # Filter using cell type columns
            # Create a mask for cells that belong to any excluded cell type
            exclude_cols = [col for col in exclude_cell_types if col in filtered_df.columns]
            if exclude_cols:
                exclude_mask = filtered_df[exclude_cols].sum(axis=1) == 0
                filtered_df = filtered_df[exclude_mask].copy()
                filtered_embedding = filtered_embedding[exclude_mask]
                print(f"Excluded {(~exclude_mask).sum()} cells with cell types: {', '.join(exclude_cell_types)}")
                
        # Modify output path to include filter label
        basename, ext = os.path.splitext(output_path)
        output_path = f"{basename}{filter_label}{ext}"
    else:
        filtered_df = df
        filtered_embedding = embedding
    
    print(f"Generating combined marker expression UMAP visualization{excluded_text}...")
    
    # Filter out TAF* and DAPI markers
    filtered_markers = [col for col in marker_cols if not (col.startswith('TAF') or 'DAPI' in col)]
    
    # Get marker names with locations and sort alphabetically by marker name
    marker_names_with_location = []
    for col in filtered_markers:
        parts = col.split('_Int-mean_')
        if len(parts) == 2:
            marker_name = parts[0]
            location = parts[1]
            marker_names_with_location.append((marker_name, location, col))
        else:
            # Fallback for unexpected format
            marker_name = col.replace('Int-mean_WC', '').strip('_ ')
            marker_names_with_location.append((marker_name, "", col))
    
    # Sort by marker name first, then by location
    sorted_markers_info = sorted(marker_names_with_location, key=lambda x: (x[0].lower(), x[1]))
    
    # Extract the sorted column names and display names
    sorted_markers = [info[2] for info in sorted_markers_info]
    sorted_display_names = [f"{info[0]} ({info[1]})" if info[1] else info[0] for info in sorted_markers_info]
    
    # Calculate grid dimensions based on number of markers
    num_markers = len(sorted_markers)
    grid_size = math.ceil(math.sqrt(num_markers))
    
    total_cells = len(filtered_df)
    # Create a figure with a grid layout
    fig = plt.figure(figsize=(5*grid_size, 5*grid_size))
    fig.suptitle(f'Combined Marker Expression UMAPs{excluded_text}\n(n={total_cells:,} cells)', 
                fontsize=24, y=0.98)
    
    # Create grid spec with spacing between subplots
    gs = gridspec.GridSpec(grid_size, grid_size, figure=fig, 
                         wspace=0.3, hspace=0.4)
    
    # Plot each marker in its own subplot in alphabetical order
    for i, (marker, display_name) in enumerate(zip(sorted_markers, sorted_display_names)):
        # Calculate grid position (left to right, top to bottom)
        row = i // grid_size
        col = i % grid_size
        
        # Create subplot
        ax = fig.add_subplot(gs[row, col])
        
        # Get marker expression values
        expression = filtered_df[marker].values
        
        # Create scatter plot with expression color using improved settings
        scatter = ax.scatter(filtered_embedding[:, 0], filtered_embedding[:, 1],
                           c=expression, 
                           cmap=MARKER_EXPRESSION_COLORMAP,
                           s=1.5, alpha=0.8,
                           vmin=MARKER_EXPRESSION_VMIN,
                           vmax=MARKER_EXPRESSION_VMAX)
        
        # Add colorbar with enhanced visibility
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label(f'Expression', fontsize=8)
        cbar.ax.tick_params(labelsize=7)
        
        # Set subplot title and labels
        ax.set_title(display_name, fontsize=10, fontweight='bold')
        ax.set_xlabel('UMAP1', fontsize=8)
        ax.set_ylabel('UMAP2', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
        
        # Set consistent axes limits if provided
        if axis_limits:
            ax.set_xlim(axis_limits[0], axis_limits[1])
            ax.set_ylim(axis_limits[2], axis_limits[3])
    
    # Adjust empty subplots if we have fewer markers than grid spaces
    for i in range(num_markers, grid_size * grid_size):
        row = i // grid_size
        col = i % grid_size
        ax = fig.add_subplot(gs[row, col])
        ax.axis('off')
    
    # Save the combined figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='tiff')
    plt.close()
    print(f"Combined marker expression visualization saved to: {output_path}")
    print(f"Sorted markers (alphabetical order): {', '.join(sorted_display_names[:10])}... (and {len(sorted_display_names)-10} more)")


def plot_dataset_umaps(embedding, df, output_dir, axis_limits=None, exclude_cell_types=None):
    """
    Create UMAPs showing dataset distributions with different colors:
    1. One UMAP with all datasets colored differently
    2. Individual UMAPs for each dataset (highlighted with others in grey)
    
    Parameters:
    embedding (ndarray): UMAP coordinates
    df (DataFrame): DataFrame with dataset information
    output_dir (str): Directory to save the plots
    axis_limits (tuple, optional): Tuple containing (x_min, x_max, y_min, y_max) for consistent axes
    exclude_cell_types (list, optional): List of cell types to exclude from visualization
    """
    # Get available datasets
    available_datasets = df['Dataset'].unique()
    
    # Create a copy of the data for filtering
    filtered_df = df.copy()
    filtered_embedding = embedding.copy()
    
    # Filter out excluded cell types if specified
    if exclude_cell_types:
        filter_label = f"_excluded_{'_'.join([ct.replace(' ', '_') for ct in exclude_cell_types])}"
        
        # Check if we're using MemCatName column or cell type columns
        if 'MemCatName' in filtered_df.columns:
            # Filter using MemCatName
            exclude_mask = ~filtered_df['MemCatName'].isin(exclude_cell_types)
            filtered_df = filtered_df[exclude_mask].copy()
            filtered_embedding = filtered_embedding[exclude_mask]
            print(f"Excluded {(~exclude_mask).sum()} cells with cell types: {', '.join(exclude_cell_types)}")
        else:
            # Filter using cell type columns
            # Create a mask for cells that belong to any excluded cell type
            exclude_cols = [col for col in exclude_cell_types if col in filtered_df.columns]
            if exclude_cols:
                exclude_mask = filtered_df[exclude_cols].sum(axis=1) == 0
                filtered_df = filtered_df[exclude_mask].copy()
                filtered_embedding = filtered_embedding[exclude_mask]
                print(f"Excluded {(~exclude_mask).sum()} cells with cell types: {', '.join(exclude_cell_types)}")
    else:
        filter_label = ""
    
    # Define output paths
    all_datasets_path = os.path.join(output_dir, f"datasets{filter_label}.tiff")
    
    # First UMAP: All datasets with different colors
    print(f"Generating UMAP visualization for all datasets{' (with exclusions)' if exclude_cell_types else ''}...")
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create lists to store labels and colors for the legend
    legend_labels = []
    legend_colors = []
    
    for dataset in available_datasets:
        if dataset in dataset_colors and dataset != 'other':
            mask = filtered_df['Dataset'] == dataset
            if mask.any():
                annotation = dataset_annotations.get(dataset, dataset)
                plt.scatter(filtered_embedding[mask, 0], 
                          filtered_embedding[mask, 1],
                          c=[dataset_colors[dataset]],
                          alpha=0.7,
                          s=10)
                legend_labels.append(f"{annotation} (n={mask.sum():,})")
                legend_colors.append(dataset_colors[dataset])
    
    total_cells = len(filtered_df)
    excluded_text = f" - Excluding {', '.join(exclude_cell_types)}" if exclude_cell_types else ""
    plt.title(f'UMAP Visualization of Different Datasets{excluded_text}\n'
             f'(n={total_cells:,} cells total)', fontsize=14)
    plt.xlabel('UMAP1', fontsize=12)
    plt.ylabel('UMAP2', fontsize=12)
    
    # Set consistent axes limits if provided
    if axis_limits:
        ax.set_xlim(axis_limits[0], axis_limits[1])
        ax.set_ylim(axis_limits[2], axis_limits[3])
    
    # Add custom legend with larger marker size or color panel
    add_color_legend(ax, legend_labels, legend_colors, 
                    bbox_to_anchor=(1.05, 1), 
                    use_color_panel=USE_COLOR_PANEL_LEGEND,
                    fontsize=10)
    
    plt.tight_layout()
    
    plt.savefig(all_datasets_path, dpi=300, bbox_inches='tight', format='tiff')
    plt.close()
    print(f"All datasets visualization saved to: {all_datasets_path}")
    
    # Individual dataset UMAPs - saved to the by_dataset directory
    for dataset in ['IgG4', 'AAV', 'NK2']:
        if dataset in available_datasets:
            print(f"Generating UMAP visualization for {dataset}{' (with exclusions)' if exclude_cell_types else ''}...")
            fig, ax = plt.subplots(figsize=(14, 12))
            
            # First plot all other datasets in light grey
            other_mask = filtered_df['Dataset'] != dataset
            if other_mask.any():
                plt.scatter(filtered_embedding[other_mask, 0],
                          filtered_embedding[other_mask, 1],
                          c=[dataset_colors['other']],
                          alpha=0.3,
                          s=5)  # Smaller dots for background
            
            # Then plot the dataset of interest with its color
            dataset_mask = filtered_df['Dataset'] == dataset
            if dataset_mask.any():
                annotation = dataset_annotations.get(dataset, dataset)
                plt.scatter(filtered_embedding[dataset_mask, 0],
                          filtered_embedding[dataset_mask, 1],
                          c=[dataset_colors[dataset]],
                          alpha=0.8,
                          s=15)  # Larger dots for the highlighted dataset
            
            plt.title(f'UMAP Visualization of {annotation} Dataset{excluded_text}\n'
                     f'(Highlighted among {total_cells:,} total cells)', fontsize=14)
            plt.xlabel('UMAP1', fontsize=12)
            plt.ylabel('UMAP2', fontsize=12)
            
            # Set consistent axes limits if provided
            if axis_limits:
                ax.set_xlim(axis_limits[0], axis_limits[1])
                ax.set_ylim(axis_limits[2], axis_limits[3])
            
            # Simple legend for this case (only two items)
            legend_labels = ["Other datasets", f"{annotation} (n={dataset_mask.sum():,})"]
            legend_colors = [dataset_colors['other'], dataset_colors[dataset]]
            
            add_color_legend(ax, legend_labels, legend_colors, 
                           bbox_to_anchor=(1.05, 1), 
                           use_color_panel=USE_COLOR_PANEL_LEGEND,
                           fontsize=10)
            
            plt.tight_layout()
            
            # Save to the by_dataset directory with appropriate filter label
            output_path = os.path.join(output_dir, f"dataset_{dataset}{filter_label}.tiff")
            plt.savefig(output_path, dpi=300, bbox_inches='tight', format='tiff')
            plt.close()
            print(f"{dataset} visualization saved to: {output_path}")
        else:
            print(f"Skipping visualization for {dataset} (not found in data)")

def generate_dataset_specific_umaps(embedding, df, cell_types, output_dir, axis_limits=None, exclude_cell_types=None):
    """
    Generate cell type and cell category UMAPs for each dataset and save in by_dataset directory.
    
    Parameters:
    embedding (ndarray): UMAP coordinates
    df (DataFrame): DataFrame with cell type data
    cell_types (list): List of cell type names (either from MemCatName or cell type columns)
    output_dir (str): Directory to save the plots
    axis_limits (tuple, optional): Tuple containing (x_min, x_max, y_min, y_max) for consistent axes
    exclude_cell_types (list, optional): List of cell types to exclude from visualization
    """
    # Get available datasets
    available_datasets = df['Dataset'].unique()
    
    # Create filter label for filenames if excluding cell types
    if exclude_cell_types:
        filter_label = f"_excluded_{'_'.join([ct.replace(' ', '_') for ct in exclude_cell_types])}"
    else:
        filter_label = ""
    
    for dataset in ['IgG4', 'AAV', 'NK2']:
        if dataset in available_datasets:
            # Create dataset-specific output paths
            cell_types_path = os.path.join(output_dir, f"{dataset}_cell_types{filter_label}.tiff")
            cell_categories_path = os.path.join(output_dir, f"{dataset}_cell_categories{filter_label}.tiff")
            
            # Generate the UMAPs for this dataset with consistent axis limits
            plot_cell_types_umap(embedding, df, cell_types, cell_types_path, 
                              legend_marker_scale=LEGEND_MARKER_SCALE,
                              use_color_panel=USE_COLOR_PANEL_LEGEND,
                              dataset_filter=dataset,
                              axis_limits=axis_limits,
                              exclude_cell_types=exclude_cell_types)
            
            plot_cell_categories_umap(embedding, df, cell_types, cell_categories_path,
                                   legend_marker_scale=LEGEND_MARKER_SCALE,
                                   use_color_panel=USE_COLOR_PANEL_LEGEND,
                                   dataset_filter=dataset,
                                   axis_limits=axis_limits,
                                   exclude_cell_types=exclude_cell_types)


def main():
    try:
        # Generate timestamp for this run
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Setup output directories 
        dirs, paths = setup_output_directories(BASE_OUTPUT_DIR, timestamp)
        
        print(f"Starting UMAP analysis with output to: {dirs['base']}")
        
        # Start log file
        with open(paths['log_file'], 'w') as f:
            f.write(f"UMAP Analysis Log - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Input file: {INPUT_PATH}\n")
            f.write(f"Output directory: {dirs['base']}\n")
            f.write(f"Marker expression colormap: {MARKER_EXPRESSION_COLORMAP}\n")
            f.write(f"Marker expression range: {MARKER_EXPRESSION_VMIN} - {MARKER_EXPRESSION_VMAX}\n\n")
            
            # Log the excluded cell types and highlighted cell types/categories settings
            if EXCLUDE_CELL_TYPES:
                f.write(f"Excluded cell types: {', '.join(EXCLUDE_CELL_TYPES)}\n")
            if HIGHLIGHT_CELL_TYPES:
                f.write(f"Highlighted cell types: {', '.join(HIGHLIGHT_CELL_TYPES)}\n")
            if HIGHLIGHT_CELL_CATEGORIES:
                f.write(f"Highlighted cell categories: {', '.join(HIGHLIGHT_CELL_CATEGORIES)}\n")
            f.write("\n")
        
        # Validate input file
        if not os.path.exists(INPUT_PATH):
            raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")
        
        # Process the data (sample cells from each dataset)
        embedding, df, cell_types, marker_cols = process_data(
            INPUT_PATH, 
            SELECTED_MARKERS,
            paths['log_file']
        )
        
        # Determine global axis limits for consistent scaling
        # First, get the min and max values for each dimension
        x_min, x_max = embedding[:, 0].min(), embedding[:, 0].max()
        y_min, y_max = embedding[:, 1].min(), embedding[:, 1].max()
        
        # Add a small buffer (5%) around the limits
        x_range = x_max - x_min
        y_range = y_max - y_min
        buffer = 0.05  # 5% buffer
        
        x_min -= buffer * x_range
        x_max += buffer * x_range
        y_min -= buffer * y_range
        y_max += buffer * y_range
        
        # Create consistent axis limits
        axis_limits = (x_min, x_max, y_min, y_max)
        
        # Log the axis limits
        with open(paths['log_file'], 'a') as f:
            f.write(f"\nUsing consistent axis limits for all UMAPs:\n")
            f.write(f"X-axis: [{x_min:.2f}, {x_max:.2f}]\n")
            f.write(f"Y-axis: [{y_min:.2f}, {y_max:.2f}]\n\n")
        
        # Make sure unclassified color is updated
        color_dict['unclassified'] = UNCLASSIFIED_COLOR
        broad_categories['unclassified']['color'] = UNCLASSIFIED_COLOR
        
        # Handle capitalization differences in data
        if 'Unclassified' in color_dict:
            color_dict['Unclassified'] = UNCLASSIFIED_COLOR
        if 'Unclassified' in broad_categories:
            broad_categories['Unclassified']['color'] = UNCLASSIFIED_COLOR
        
        # Count unclassified cells for logging
        if 'MemCatName' in df.columns:
            unclassified_count = df['MemCatName'].str.lower().isin(['unclassified']).sum()
            with open(paths['log_file'], 'a') as f:
                f.write(f"Found {unclassified_count} cells labeled as 'unclassified' in MemCatName column ({unclassified_count/len(df)*100:.2f}% of total)\n\n")
        
        # ==================================================================================
        # STANDARD VISUALIZATIONS (with and without exclusions)
        # ==================================================================================
        
        # Generate main UMAPs with consistent axis limits (standard version)
        plot_cell_types_umap(embedding, df, cell_types, paths['cell_types'], 
                          legend_marker_scale=LEGEND_MARKER_SCALE,
                          use_color_panel=USE_COLOR_PANEL_LEGEND,
                          axis_limits=axis_limits)
        
        plot_cell_categories_umap(embedding, df, cell_types, paths['cell_categories'],
                               legend_marker_scale=LEGEND_MARKER_SCALE,
                               use_color_panel=USE_COLOR_PANEL_LEGEND,
                               axis_limits=axis_limits)
        
        # Generate main UMAPs with cell type exclusions if specified
        if EXCLUDE_CELL_TYPES:
            filter_label = f"_excluded_{'_'.join([ct.replace(' ', '_') for ct in EXCLUDE_CELL_TYPES])}"
            excluded_cell_types_path = os.path.join(dirs['base'], f"cell_types{filter_label}.tiff")
            excluded_cell_categories_path = os.path.join(dirs['base'], f"cell_categories{filter_label}.tiff")
            
            plot_cell_types_umap(embedding, df, cell_types, excluded_cell_types_path, 
                              legend_marker_scale=LEGEND_MARKER_SCALE,
                              use_color_panel=USE_COLOR_PANEL_LEGEND,
                              axis_limits=axis_limits,
                              exclude_cell_types=EXCLUDE_CELL_TYPES)
            
            plot_cell_categories_umap(embedding, df, cell_types, excluded_cell_categories_path,
                                   legend_marker_scale=LEGEND_MARKER_SCALE,
                                   use_color_panel=USE_COLOR_PANEL_LEGEND,
                                   axis_limits=axis_limits,
                                   exclude_cell_types=EXCLUDE_CELL_TYPES)
        
        # ==================================================================================
        # MARKER EXPRESSION VISUALIZATIONS (with and without exclusions)
        # ==================================================================================
        
        # Generate marker UMAPs with consistent axis limits (standard version)
        plot_marker_expression_umaps(embedding, df, marker_cols, paths['markers_dir'], 
                                   axis_limits=axis_limits)
        
        plot_combined_marker_umaps(embedding, df, marker_cols, paths['combined_markers'], 
                                axis_limits=axis_limits)
        
        # Generate marker UMAPs with cell type exclusions if specified
        if EXCLUDE_CELL_TYPES:
            # Individual marker UMAPs with exclusions
            plot_marker_expression_umaps(embedding, df, marker_cols, paths['markers_dir'],
                                      axis_limits=axis_limits,
                                      exclude_cell_types=EXCLUDE_CELL_TYPES)
            
            # Combined marker UMAP with exclusions
            plot_combined_marker_umaps(embedding, df, marker_cols, paths['combined_markers'],
                                    axis_limits=axis_limits,
                                    exclude_cell_types=EXCLUDE_CELL_TYPES)
        
        # ==================================================================================
        # DATASET VISUALIZATIONS (with and without exclusions)
        # ==================================================================================
        
        # Generate dataset comparison UMAPs with consistent axis limits (standard version)
        plot_dataset_umaps(embedding, df, dirs['base'], axis_limits=axis_limits)
        
        # Generate dataset UMAPs with cell type exclusions if specified
        if EXCLUDE_CELL_TYPES:
            plot_dataset_umaps(embedding, df, dirs['base'], 
                            axis_limits=axis_limits,
                            exclude_cell_types=EXCLUDE_CELL_TYPES)
        
        # ==================================================================================
        # DATASET-SPECIFIC VISUALIZATIONS (with and without exclusions)
        # ==================================================================================
        
        # Generate dataset-specific UMAPs with consistent axis limits (standard version)
        generate_dataset_specific_umaps(embedding, df, cell_types, paths['by_dataset_dir'], 
                                     axis_limits=axis_limits)
        
        # Generate dataset-specific UMAPs with cell type exclusions if specified
        if EXCLUDE_CELL_TYPES:
            generate_dataset_specific_umaps(embedding, df, cell_types, paths['by_dataset_dir'],
                                         axis_limits=axis_limits,
                                         exclude_cell_types=EXCLUDE_CELL_TYPES)
        
        # ==================================================================================
        # SPECIALIZED VISUALIZATIONS (filtered and highlighted)
        # ==================================================================================
        
        # Generate filtered UMAPs (excluding specific cell types)
        if EXCLUDE_CELL_TYPES:
            plot_filtered_umaps(
                embedding, df, cell_types, paths['filtered_dir'],
                exclude_cell_types=EXCLUDE_CELL_TYPES,
                axis_limits=axis_limits
            )
        
        # Generate highlighted UMAPs for specific cell types and categories
        plot_highlighted_umaps(
            embedding, df, cell_types, paths['highlighted_dir'],
            highlight_cell_types=HIGHLIGHT_CELL_TYPES,
            highlight_categories=HIGHLIGHT_CELL_CATEGORIES,
            exclude_cell_types=EXCLUDE_CELL_TYPES,
            axis_limits=axis_limits
        )
        
        # ==================================================================================
        # STATISTICS AND LOGGING
        # ==================================================================================
        
        # Print statistics
        total_cells = len(df)
        
        # Get classification statistics based on MemCatName or cell type columns
        if 'MemCatName' in df.columns:
            # Check for both 'unclassified' and 'Unclassified'
            unclassified_mask = df['MemCatName'].str.lower() == 'unclassified'
            classified_mask = ~unclassified_mask
        else:
            classified_mask = df[cell_types].sum(axis=1) > 0
            
        unclassified_cells = (~classified_mask).sum()
        classified_cells = classified_mask.sum()
        
        # Log statistics
        with open(paths['log_file'], 'a') as f:
            f.write(f"\nClassification Statistics:\n")
            f.write(f"Total cells: {total_cells}\n")
            f.write(f"Classified cells: {classified_cells} ({classified_cells/total_cells*100:.1f}%)\n")
            f.write(f"Unclassified cells: {unclassified_cells} ({unclassified_cells/total_cells*100:.1f}%)\n")
            f.write(f"Total markers used for UMAP: {len(marker_cols)}\n\n")
            
            f.write("Markers used for UMAP analysis:\n")
            for col in marker_cols:
                # Extract marker name without 'Int-mean_WC'
                marker_name = col.replace('Int-mean_WC', '').strip('_ ')
                f.write(f"- {marker_name}\n")
            
            f.write(f"\nDataset Statistics:\n")
            for dataset in df['Dataset'].unique():
                dataset_count = (df['Dataset'] == dataset).sum()
                f.write(f"{dataset}: {dataset_count:,} cells ({dataset_count/total_cells*100:.1f}%)\n")
            
            # Log excluded cell type statistics if applicable
            if EXCLUDE_CELL_TYPES:
                f.write(f"\nExcluded Cell Types Statistics:\n")
                for cell_type in EXCLUDE_CELL_TYPES:
                    if 'MemCatName' in df.columns:
                        cell_count = (df['MemCatName'] == cell_type).sum()
                    else:
                        if cell_type in df.columns:
                            cell_count = (df[cell_type] > 0).sum()
                        else:
                            cell_count = 0
                    f.write(f"{cell_type}: {cell_count:,} cells ({cell_count/total_cells*100:.1f}%)\n")
        
        print(f"\nAnalysis completed successfully!")
        print(f"Results saved to: {dirs['base']}")
        print(f"Log file: {paths['log_file']}")
        print(f"Marker expression visualizations use colormap: '{MARKER_EXPRESSION_COLORMAP}' with range {MARKER_EXPRESSION_VMIN}-{MARKER_EXPRESSION_VMAX}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    # Run the main function directly
    exit_code = main()
    exit(exit_code)
