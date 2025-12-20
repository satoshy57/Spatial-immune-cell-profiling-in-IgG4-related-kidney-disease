# -*- coding: utf-8 -*-
"""
Script to classify cells based on their location in different structural masks:
- outside the tissue (cells not on the tissue mask)
- glomeruli
- tubules
- vessels
- interstitial (cells on the tissue but not in any specific structure)

This version applies a fixed 1/10 scaling factor to structural masks (glomeruli, tubules, vessels)
since these masks are downsized to 1/10 of the original tissue mask size while maintaining
the same aspect ratio.
"""
import os
import pandas as pd
import numpy as np
from tifffile import imread
from scipy.ndimage import center_of_mass
from scipy.spatial.distance import cdist

# Input CSV file path
input_csv = '/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/combined_SAM_scores_filtered_alldataset5_10woMono.csv'

# Output path for the classified CSV
output_dir = os.path.dirname(input_csv)
output_file = os.path.join(output_dir, 'combined_SAM_scores_filtered_alldataset5_10woMonostructure.csv')

# Load the CSV file
print(f"Loading input CSV file: {input_csv}")
df = pd.read_csv(input_csv, low_memory=False)  # Added low_memory=False to avoid DtypeWarning

# Initialize the structure column with 'interstitial' (default)
df['structure'] = 'interstitial'

# Process each unique Dataset and CompName combination
unique_datasets_comps = df.groupby(['Dataset', 'CompName']).size().reset_index().rename(columns={0: 'count'})
total_combinations = len(unique_datasets_comps)
print(f"Found {total_combinations} unique Dataset-CompName combinations to process")

# Track overall statistics
total_cells = 0
total_outside_tissue = 0
total_glomeruli = 0
total_tubules = 0
total_vessels = 0
total_interstitial = 0

# Fixed scaling factors for structural masks (known to be 1/10 the size of tissue masks)
STRUCTURAL_SCALE = 0.1  # This is the 1/10 ratio for downsized masks

for idx, row in unique_datasets_comps.iterrows():
    dataset = row['Dataset']
    comp_name = row['CompName']
    
    print(f"\nProcessing {idx+1}/{total_combinations}: {dataset} - {comp_name}")
    
    # Get sample and area from comp_name
    sample = comp_name.split('_')[0]
    area = comp_name.split('_')[1]
    
    # Define paths to the mask files based on dataset
    # Different path patterns for NK2 vs. AAV/IgG4 datasets
    if dataset == 'NK2':
        # Path pattern for NK2 dataset
        glomeruli_mask_path = f'/project/mclark/SCAMPI_datasets/{dataset}/structural_segmentation/glomeruli/masks/{comp_name}.tif'
        tubules_mask_path = f'/project/mclark/SCAMPI_datasets/{dataset}/structural_segmentation/tubules/masks/{comp_name}.tif'
        vessels_mask_path = f'/project/mclark/SCAMPI_datasets/{dataset}/structural_segmentation/vessels/{comp_name}.tif'
    else:
        # Path pattern for AAV and IgG4 datasets
        glomeruli_mask_path = f'/project/mclark/SCAMPI_datasets/{dataset}/structural_segmentation/glomeruli/masks/{comp_name}.tif_rgb_cp_masks.tif'
        tubules_mask_path = f'/project/mclark/SCAMPI_datasets/{dataset}/structural_segmentation/tubules/masks/{comp_name}_structure_overlay_ds10_16bit_cleaned_cp_masks.tif'
        vessels_mask_path = f'/project/mclark/SCAMPI_datasets/{dataset}/structural_segmentation/vessels/masks/{comp_name}_cp_masks.tif'
    
    # Define paths to the tissue mask files
    tissue_mask_path1 = f'/project/mclark/SCAMPI_datasets/{dataset}/tissue_composite_masks/DAPI_UV_1_{dataset}_{comp_name}.tif'
    tissue_mask_path2 = f'/project/mclark/SCAMPI_datasets/{dataset}/tissue_composite_masks/DAPI_UV_2_{dataset}_{comp_name}.tif'
    
    # Report the paths being checked
    print(f"  Checking structural mask paths:")
    print(f"  - Glomeruli: {glomeruli_mask_path}")
    print(f"  - Tubules:   {tubules_mask_path}")
    print(f"  - Vessels:   {vessels_mask_path}")
    print(f"  - Tissue 1:  {tissue_mask_path1}")
    print(f"  - Tissue 2:  {tissue_mask_path2}")
    
    # Define path to whole cell segmentation mask
    # Try both UV_1 and UV_2 variations since it's not clear which one is needed
    whole_cell_mask_path1 = f'/project/mclark/SCAMPI_datasets/{dataset}/wholeCell_segmentations_WS/{sample}/{area}/DAPI_UV_1_{dataset}_{comp_name}_seg.npy'
    whole_cell_mask_path2 = f'/project/mclark/SCAMPI_datasets/{dataset}/wholeCell_segmentations_WS/{sample}/{area}/DAPI_UV_2_{dataset}_{comp_name}_seg.npy'
    
    # Load masks if they exist
    glomeruli_mask = None
    tubules_mask = None
    vessels_mask = None
    tissue_mask = None
    whole_cell_mask = None
    using_whole_cell = False
    
    # Try to load tissue mask
    if os.path.exists(tissue_mask_path1):
        try:
            tissue_mask = imread(tissue_mask_path1)
            print(f"  Loaded tissue mask 1 with shape {tissue_mask.shape}")
            # Print stats about the mask
            non_zero = np.sum(tissue_mask > 0)
            print(f"  Tissue mask has {non_zero} non-zero pixels ({non_zero/tissue_mask.size*100:.2f}% of total)")
        except Exception as e:
            print(f"  Error loading tissue mask 1: {e}")
    else:
        print(f"  Tissue mask 1 not found at: {tissue_mask_path1}")
    
    if tissue_mask is None and os.path.exists(tissue_mask_path2):
        try:
            tissue_mask = imread(tissue_mask_path2)
            print(f"  Loaded tissue mask 2 with shape {tissue_mask.shape}")
            # Print stats about the mask
            non_zero = np.sum(tissue_mask > 0)
            print(f"  Tissue mask has {non_zero} non-zero pixels ({non_zero/tissue_mask.size*100:.2f}% of total)")
        except Exception as e:
            print(f"  Error loading tissue mask 2: {e}")
    else:
        if tissue_mask is None:
            print(f"  Tissue mask 2 not found at: {tissue_mask_path2}")
    
    # Try to load whole cell segmentation mask
    if os.path.exists(whole_cell_mask_path1):
        try:
            whole_cell_data = np.load(whole_cell_mask_path1, allow_pickle=True)
            print(f"  Loaded whole cell mask 1, type: {type(whole_cell_data)}, shape: {whole_cell_data.shape if hasattr(whole_cell_data, 'shape') else 'No shape'}")
            
            # Enhanced inspection of the loaded data to help debugging
            if hasattr(whole_cell_data, 'shape') and len(whole_cell_data.shape) == 0:
                # The data is a 0-dimensional array, possibly containing an object
                print(f"  Detected 0-dimensional array, inspecting content type: {type(whole_cell_data.item()) if hasattr(whole_cell_data, 'item') else 'unknown'}")
                
                # Try to extract a 2D array if it's stored in a container
                try:
                    # If it's a dictionary with a key that contains the mask
                    if hasattr(whole_cell_data, 'item') and isinstance(whole_cell_data.item(), dict):
                        dict_data = whole_cell_data.item()
                        print(f"  Dict keys: {list(dict_data.keys())}")
                        
                        # We now know the dictionary structure for these files
                        # The actual mask data is in the 'masks' key
                        if 'masks' in dict_data:
                            mask_data = dict_data['masks']
                            if isinstance(mask_data, np.ndarray):
                                # Check if it's already a valid 2D array
                                if len(mask_data.shape) == 2:
                                    whole_cell_mask = mask_data
                                    using_whole_cell = True
                                    print(f"  Found 2D whole cell mask in 'masks' key, shape: {whole_cell_mask.shape}")
                                # Check if it's a 3D array with first dimension of 1
                                elif len(mask_data.shape) == 3 and mask_data.shape[0] == 1:
                                    whole_cell_mask = mask_data[0]  # Extract the 2D slice
                                    using_whole_cell = True
                                    print(f"  Extracted 2D whole cell mask from 3D array, shape: {whole_cell_mask.shape}")
                                else:
                                    print(f"  'masks' key contains array with unsupported shape: {mask_data.shape}")
                            else:
                                print(f"  'masks' key exists but doesn't contain an ndarray: {type(mask_data)}")
                    
                    # If it's a list or tuple that might contain the mask as an item
                    elif hasattr(whole_cell_data, 'item') and isinstance(whole_cell_data.item(), (list, tuple)):
                        container = whole_cell_data.item()
                        print(f"  Container has {len(container)} items")
                        
                        # Look for the first valid 2D array in the container
                        for i, item in enumerate(container):
                            if isinstance(item, np.ndarray) and len(item.shape) == 2:
                                whole_cell_mask = item
                                using_whole_cell = True
                                print(f"  Found valid 2D array at index {i}, shape: {whole_cell_mask.shape}")
                                break
                except Exception as nested_e:
                    print(f"  Error while trying to extract 2D array from container: {nested_e}")
            
            # Standard check if it's already a valid 2D array
            elif hasattr(whole_cell_data, 'shape') and len(whole_cell_data.shape) == 2:
                whole_cell_mask = whole_cell_data
                using_whole_cell = True
                print(f"  Whole cell mask 1 is valid with shape {whole_cell_mask.shape}")
            else:
                print(f"  Whole cell mask 1 is not a valid 2D array, skipping")
        except Exception as e:
            print(f"  Error loading whole cell mask 1: {e}")
    
    if not using_whole_cell and os.path.exists(whole_cell_mask_path2):
        try:
            whole_cell_data = np.load(whole_cell_mask_path2, allow_pickle=True)
            print(f"  Loaded whole cell mask 2, type: {type(whole_cell_data)}, shape: {whole_cell_data.shape if hasattr(whole_cell_data, 'shape') else 'No shape'}")
            
            # Enhanced inspection of the loaded data to help debugging
            if hasattr(whole_cell_data, 'shape') and len(whole_cell_data.shape) == 0:
                # The data is a 0-dimensional array, possibly containing an object
                print(f"  Detected 0-dimensional array, inspecting content type: {type(whole_cell_data.item()) if hasattr(whole_cell_data, 'item') else 'unknown'}")
                
                # Try to extract a 2D array if it's stored in a container
                try:
                    # If it's a dictionary with a key that contains the mask
                    if hasattr(whole_cell_data, 'item') and isinstance(whole_cell_data.item(), dict):
                        dict_data = whole_cell_data.item()
                        print(f"  Dict keys: {list(dict_data.keys())}")
                        
                        # We now know the dictionary structure for these files
                        # The actual mask data is in the 'masks' key
                        if 'masks' in dict_data:
                            mask_data = dict_data['masks']
                            if isinstance(mask_data, np.ndarray):
                                # Check if it's already a valid 2D array
                                if len(mask_data.shape) == 2:
                                    whole_cell_mask = mask_data
                                    using_whole_cell = True
                                    print(f"  Found 2D whole cell mask in 'masks' key, shape: {whole_cell_mask.shape}")
                                # Check if it's a 3D array with first dimension of 1
                                elif len(mask_data.shape) == 3 and mask_data.shape[0] == 1:
                                    whole_cell_mask = mask_data[0]  # Extract the 2D slice
                                    using_whole_cell = True
                                    print(f"  Extracted 2D whole cell mask from 3D array, shape: {whole_cell_mask.shape}")
                                else:
                                    print(f"  'masks' key contains array with unsupported shape: {mask_data.shape}")
                            else:
                                print(f"  'masks' key exists but doesn't contain an ndarray: {type(mask_data)}")
                    
                    # If it's a list or tuple that might contain the mask as an item
                    elif hasattr(whole_cell_data, 'item') and isinstance(whole_cell_data.item(), (list, tuple)):
                        container = whole_cell_data.item()
                        print(f"  Container has {len(container)} items")
                        
                        # Look for the first valid 2D array in the container
                        for i, item in enumerate(container):
                            if isinstance(item, np.ndarray) and len(item.shape) == 2:
                                whole_cell_mask = item
                                using_whole_cell = True
                                print(f"  Found valid 2D array at index {i}, shape: {whole_cell_mask.shape}")
                                break
                except Exception as nested_e:
                    print(f"  Error while trying to extract 2D array from container: {nested_e}")
            
            # Standard check if it's already a valid 2D array
            elif hasattr(whole_cell_data, 'shape') and len(whole_cell_data.shape) == 2:
                whole_cell_mask = whole_cell_data
                using_whole_cell = True
                print(f"  Whole cell mask 2 is valid with shape {whole_cell_mask.shape}")
            else:
                print(f"  Whole cell mask 2 is not a valid 2D array, skipping")
        except Exception as e:
            print(f"  Error loading whole cell mask 2: {e}")
    
    # Try to load glomeruli mask
    if os.path.exists(glomeruli_mask_path):
        try:
            glomeruli_mask = imread(glomeruli_mask_path)
            print(f"  Loaded glomeruli mask with shape {glomeruli_mask.shape}")
            # Print stats about the mask
            non_zero = np.sum(glomeruli_mask > 0)
            print(f"  Glomeruli mask has {non_zero} non-zero pixels ({non_zero/glomeruli_mask.size*100:.2f}% of total)")
        except Exception as e:
            print(f"  Error loading glomeruli mask: {e}")
    else:
        print(f"  Glomeruli mask not found at: {glomeruli_mask_path}")
    
    # Try to load tubules mask
    if os.path.exists(tubules_mask_path):
        try:
            tubules_mask = imread(tubules_mask_path)
            print(f"  Loaded tubules mask with shape {tubules_mask.shape}")
            # Print stats about the mask
            non_zero = np.sum(tubules_mask > 0)
            print(f"  Tubules mask has {non_zero} non-zero pixels ({non_zero/tubules_mask.size*100:.2f}% of total)")
        except Exception as e:
            print(f"  Error loading tubules mask: {e}")
    else:
        print(f"  Tubules mask not found at: {tubules_mask_path}")
    
    # Try to load vessels mask
    if os.path.exists(vessels_mask_path):
        try:
            vessels_mask = imread(vessels_mask_path)
            print(f"  Loaded vessels mask with shape {vessels_mask.shape}")
            # Print stats about the mask
            non_zero = np.sum(vessels_mask > 0)
            print(f"  Vessels mask has {non_zero} non-zero pixels ({non_zero/vessels_mask.size*100:.2f}% of total)")
        except Exception as e:
            print(f"  Error loading vessels mask: {e}")
    else:
        print(f"  Vessels mask not found at: {vessels_mask_path}")
    
    # Get subset of data for current Dataset and CompName
    subset = df[(df['Dataset'] == dataset) & (df['CompName'] == comp_name)]
    subset_size = len(subset)
    total_cells += subset_size
    print(f"  Found {subset_size} cells to classify for this component")
    
    # Skip if no cells found
    if subset_size == 0:
        print(f"  Skipping - no cells found")
        continue
    
    # We can continue even without tissue mask, but will warn
    if tissue_mask is None:
        print(f"  Warning: Tissue mask not found. Unable to classify cells as 'outside the tissue'.")
    
    # Get cell coordinate ranges
    x_min, x_max = subset['CellCentroidCol'].min(), subset['CellCentroidCol'].max()
    y_min, y_max = subset['CellCentroidRow'].min(), subset['CellCentroidRow'].max()
    print(f"  Cell coordinate ranges: X: {x_min:.1f} to {x_max:.1f}, Y: {y_min:.1f} to {y_max:.1f}")
    
    # Check if we have at least one mask to use
    if tissue_mask is None and glomeruli_mask is None and tubules_mask is None and vessels_mask is None:
        print(f"  Could not find any masks - skipping")
        continue
    
    # Initialize counters for each structure
    outside_tissue_count = 0
    glomeruli_count = 0
    tubules_count = 0
    vessels_count = 0
    interstitial_count = 0
    
    # Using centroid-based classification with fixed scaling
    print("  Using cell centroid coordinates with fixed scaling for classification")
    print(f"  Applying FIXED scaling factor of {STRUCTURAL_SCALE} (1/10) to structural masks")
    
    # Get dimensions for each type of mask for bounds checking
    tissue_height = tissue_width = 0
    glomeruli_height = glomeruli_width = 0
    tubules_height = tubules_width = 0
    vessels_height = vessels_width = 0
    
    if tissue_mask is not None:
        tissue_height, tissue_width = tissue_mask.shape
        print(f"  Tissue mask dimensions: {tissue_width}x{tissue_height}")
    
    if glomeruli_mask is not None:
        glomeruli_height, glomeruli_width = glomeruli_mask.shape
        print(f"  Glomeruli mask dimensions: {glomeruli_width}x{glomeruli_height}")
    
    if tubules_mask is not None:
        tubules_height, tubules_width = tubules_mask.shape
        print(f"  Tubules mask dimensions: {tubules_width}x{tubules_height}")
    
    if vessels_mask is not None:
        vessels_height, vessels_width = vessels_mask.shape
        print(f"  Vessels mask dimensions: {vessels_width}x{vessels_height}")
    
    # Classification hierarchy: 
    # 1. First check if outside tissue
    # 2. Then check if in glomeruli
    # 3. Then check if in tubules
    # 4. Then check if in vessels
    # 5. Otherwise, it's interstitial
    for i, cell_idx in enumerate(subset.index):
        if i % 5000 == 0 and i > 0:
            print(f"  Processed {i}/{subset_size} cells")
            
        # Get original cell coordinates
        x = df.loc[cell_idx, 'CellCentroidCol']
        y = df.loc[cell_idx, 'CellCentroidRow']
        
        # First check if the cell is outside the tissue
        if tissue_mask is not None:
            # No scaling needed for tissue mask - it's at original size
            # Convert to integer coordinates
            x_tissue_int = int(round(x))
            y_tissue_int = int(round(y))
            
            # Make sure coordinates are within bounds
            if x_tissue_int < 0 or x_tissue_int >= tissue_width or y_tissue_int < 0 or y_tissue_int >= tissue_height:
                # If out of bounds, consider outside tissue
                df.loc[cell_idx, 'structure'] = 'outside the tissue'
                outside_tissue_count += 1
                continue
            
            if tissue_mask[y_tissue_int, x_tissue_int] == 0:
                df.loc[cell_idx, 'structure'] = 'outside the tissue'
                outside_tissue_count += 1
                continue
        
        # Check if the cell is in a glomerulus
        if glomeruli_mask is not None:
            # Apply fixed 1/10 scaling for glomeruli mask
            x_glom = x * STRUCTURAL_SCALE
            y_glom = y * STRUCTURAL_SCALE
            
            # Convert to integer coordinates
            x_glom_int = int(round(x_glom))
            y_glom_int = int(round(y_glom))
            
            # Make sure coordinates are within bounds
            if 0 <= x_glom_int < glomeruli_width and 0 <= y_glom_int < glomeruli_height:
                if glomeruli_mask[y_glom_int, x_glom_int] > 0:
                    df.loc[cell_idx, 'structure'] = 'glomeruli'
                    glomeruli_count += 1
                    continue
        
        # Check if the cell is in a tubule
        if tubules_mask is not None:
            # Apply fixed 1/10 scaling for tubules mask
            x_tubule = x * STRUCTURAL_SCALE
            y_tubule = y * STRUCTURAL_SCALE
            
            # Convert to integer coordinates
            x_tubule_int = int(round(x_tubule))
            y_tubule_int = int(round(y_tubule))
            
            # Make sure coordinates are within bounds
            if 0 <= x_tubule_int < tubules_width and 0 <= y_tubule_int < tubules_height:
                if tubules_mask[y_tubule_int, x_tubule_int] > 0:
                    df.loc[cell_idx, 'structure'] = 'tubules'
                    tubules_count += 1
                    continue
                
        # Check if the cell is in a vessel
        if vessels_mask is not None:
            # Apply fixed 1/10 scaling for vessels mask
            x_vessel = x * STRUCTURAL_SCALE
            y_vessel = y * STRUCTURAL_SCALE
            
            # Convert to integer coordinates
            x_vessel_int = int(round(x_vessel))
            y_vessel_int = int(round(y_vessel))
            
            # Make sure coordinates are within bounds
            if 0 <= x_vessel_int < vessels_width and 0 <= y_vessel_int < vessels_height:
                if vessels_mask[y_vessel_int, x_vessel_int] > 0:
                    df.loc[cell_idx, 'structure'] = 'vessels'
                    vessels_count += 1
                    continue
        
        # If not in any specific structure but on tissue, it's interstitial
        df.loc[cell_idx, 'structure'] = 'interstitial'
        interstitial_count += 1
    
    # Update overall counts
    total_outside_tissue += outside_tissue_count
    total_glomeruli += glomeruli_count
    total_tubules += tubules_count
    total_vessels += vessels_count
    total_interstitial += interstitial_count
    
    # Print structure counts for this component
    print(f"  Structure distribution for {comp_name}:")
    print(f"    outside the tissue: {outside_tissue_count} cells ({outside_tissue_count/subset_size*100:.2f}%)")
    print(f"    glomeruli: {glomeruli_count} cells ({glomeruli_count/subset_size*100:.2f}%)")
    print(f"    tubules: {tubules_count} cells ({tubules_count/subset_size*100:.2f}%)")
    print(f"    vessels: {vessels_count} cells ({vessels_count/subset_size*100:.2f}%)")
    print(f"    interstitial: {interstitial_count} cells ({interstitial_count/subset_size*100:.2f}%)")

# Save the updated DataFrame
print(f"\nSaving classified data to: {output_file}")
df.to_csv(output_file, index=False)

# Print final summary
print("\nClassification complete!")
print("\nOverall structure distribution:")
print(f"  outside the tissue: {total_outside_tissue} cells ({total_outside_tissue/total_cells*100:.2f}%)")
print(f"  glomeruli: {total_glomeruli} cells ({total_glomeruli/total_cells*100:.2f}%)")
print(f"  tubules: {total_tubules} cells ({total_tubules/total_cells*100:.2f}%)")
print(f"  vessels: {total_vessels} cells ({total_vessels/total_cells*100:.2f}%)")
print(f"  interstitial: {total_interstitial} cells ({total_interstitial/total_cells*100:.2f}%)")
