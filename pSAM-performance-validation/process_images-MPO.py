import pandas as pd
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
import re
import csv

Image.MAX_IMAGE_PIXELS = None  # Set to None to disable the limit

def cut_sample(cellid, img, pred_data):
    cellsegs = pred_data
    mask = (cellsegs == cellid).astype(np.uint8)
    cellid = int(cellid)
    rows, cols = np.where(cellsegs == cellid)
    row_min = np.min(rows)
    row_max = np.max(rows)
    col_min = np.min(cols)
    col_max = np.max(cols)
    row_center = (row_min + row_max) / 2
    col_center = (col_min + col_max) / 2
    width = cellsegs.shape[0]
    height = cellsegs.shape[1]
    row_1 = row_center - 75
    row_2 = row_center + 75
    col_1 = col_center - 75
    col_2 = col_center + 75
    if row_1 < 0:
        row_1 = 0   
    if row_2 > width:
        row_2 = width               
    if col_1 < 0:
        col_1 = 0   
    if col_2 > height:
        col_2 = height
    row_1 = int(row_1)
    row_2 = int(row_2)
    col_1 = int(col_1)
    col_2 = int(col_2)
    cellimg_crop = img[row_1:row_2, col_1:col_2]
    return cellimg_crop

def get_file_with_pattern_variants(base_path, patterns):
    """Try multiple file patterns and return the first one that exists."""
    for pattern in patterns:
        if os.path.exists(pattern):
            return pattern
    # If no file is found, return None
    print(f"Warning: None of the pattern variants exist for {base_path}")
    return None

def main():
    # Read the list of cells to process
    with open("/project/mclark/For/Bayard/Ground_truthing/IGG4/to_process-MPO.txt", "r") as file:
        to_process_lines = [line.strip() for line in file]
    
    # Read the list of already labeled cells
    labeled_cells = []
    if os.path.exists("/project/mclark/For/Bayard/Ground_truthing/IGG4/labeled-MPO.txt"):
        with open("/project/mclark/For/Bayard/Ground_truthing/IGG4/labeled-MPO.txt", "r") as file:
            labeled_cells = [line.strip() for line in file]
    
    # Read the list of hard-skipped cells
    hard_skipped_cells = []
    if os.path.exists("/project/mclark/For/Bayard/Ground_truthing/IGG4/hard_skip-MPO.txt"):
        with open("/project/mclark/For/Bayard/Ground_truthing/IGG4/hard_skip-MPO.txt", "r") as file:
            hard_skipped_cells = [line.strip() for line in file]
    
    # Combine all cells to skip
    cells_to_skip = set(labeled_cells + hard_skipped_cells)
    
    # Process only cells that are not in the skip list
    for file in to_process_lines:
        if file in cells_to_skip:
            print(f"Skipping already processed cell: {file}")
            continue
            
        if os.path.exists(file):
            continue
            
        match = re.search(r'/([^/_]+_[^/_]+)_\d+_r_(\d+)\.0_c_(\d+)\.0\.npy$', file)
        c_name = match.group(1) 
        r = int(match.group(2))
        c = int(match.group(3))
        sample, area = c_name.split('_')
        
        # Define base paths for images
        fps_base = [
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/CD3_Atto550_12_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/CD4_AF488_6_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/CD8_AF488_7_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/TCRd_Cy5_4_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/CD20_AF488_8_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/CD138_AF488_4_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/SLAMF7_Atto550_1_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/CD11c_AF488_1_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/BDCA1_Cy5_1_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/BDCA2_AF488_2_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/aSMA_AF488_15_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/CD16_Atto550_7_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/HLAII_Cy5_11_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/CD68_Cy5_13_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/CD163_AF488_13_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/MERTK_Atto550_13_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/CD56_Cy5_6_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/Foxp3_Atto550_4_IgG4_{sample}_{area}.tif',
            # DAPI will be handled separately
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/CD34_AF488_11_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/MPO_Atto550_19_IgG4_{sample}_{area}.tif'
        ]
        
        # Handle DAPI file with both patterns
        dapi_patterns = [
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/DAPI_UV_1_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/DAPI_UV_2_IgG4_{sample}_{area}.tif'
        ]
        dapi_file = get_file_with_pattern_variants("DAPI", dapi_patterns)
        
        # If no DAPI file found, skip this sample
        if dapi_file is None:
            print(f"No DAPI file found for {sample}_{area}. Skipping.")
            continue
            
        # Insert DAPI file in the correct position (position 18)
        fps = fps_base[:18] + [dapi_file] + fps_base[18:]
        
        # Try both segmentation file patterns
        seg_patterns = [
            f'/project/mclark/SCAMPI_datasets/IgG4/wholeCell_segmentations_WS/{sample}/{area}/DAPI_UV_1_IgG4_{sample}_{area}_seg.npy',
            f'/project/mclark/SCAMPI_datasets/IgG4/wholeCell_segmentations_WS/{sample}/{area}/DAPI_UV_2_IgG4_{sample}_{area}_seg.npy'
        ]
        
        if os.path.exists(seg_patterns[0]):
            seg_file_path = seg_patterns[0]
        elif os.path.exists(seg_patterns[1]):
            seg_file_path = seg_patterns[1]
        else:
            print(f"Neither segmentation file found for {sample}_{area}. Skipping.")
            continue
        
        try:
            pred_data = np.load(seg_file_path, allow_pickle=True).item()
            pred_data = pred_data['masks']
            pred_data = np.squeeze(pred_data)
            
            if int(r) >= pred_data.shape[0] or int(c) >= pred_data.shape[1]:
                print(f"Cell centroid falls outside of image. Check: {sample} {area}")
                continue
                
            value = pred_data[int(r), int(c)]
            output_file_path = f'/project/mclark/For/Bayard/Ground_truthing/IGG4/cut_samples-MPO/{sample}_{area}_{value}_r_{r}_c_{c}.npy'
            
            if os.path.exists(output_file_path):
                continue
            
            # Check if all input files exist
            missing_files = [fp for fp in fps if not os.path.exists(fp)]
            if missing_files:
                print(f"Missing input files for {sample}_{area}: {missing_files}")
                continue
                
            # Open all images
            opened_lst = []
            for file_path in fps:
                try:
                    img = np.array(Image.open(file_path))
                    opened_lst.append(img)
                except FileNotFoundError:
                    print(f"Error: Could not open {file_path}")
                    raise
            
            images = [cut_sample(value, img, pred_data) for img in opened_lst]
            mask = cut_sample(value, pred_data, pred_data)
            images = images + [mask]
            stacked_array = np.stack(images, axis=0)  
            
            np.save(output_file_path, stacked_array)
            
            txt_file_path = '/project/mclark/For/Bayard/Ground_truthing/IGG4/processed-MPO.txt'
            with open(txt_file_path, 'a') as file:
                file.write(output_file_path + '\n')
                print("WRITE!", flush=True)
        except Exception as e:
            print(f"Error processing {sample}_{area}: {str(e)}")
            continue

if __name__ == '__main__':
    main()
