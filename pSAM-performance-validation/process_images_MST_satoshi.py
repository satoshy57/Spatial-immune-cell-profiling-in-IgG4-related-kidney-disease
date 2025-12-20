import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import csv
import re
Image.MAX_IMAGE_PIXELS = None

def cut_sample(cellid, img, pred_data):
    cellsegs = pred_data
    cellid = int(cellid)
    rows, cols = np.where(cellsegs == cellid)
    
    # Check if cell ID exists in the image
    if len(rows) == 0 or len(cols) == 0:
        print(f"Cell ID {cellid} not found in segmentation mask", flush=True)
        return None
        
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

def main(): 
    with open("/project/mclark/For/Bayard/Ground_truthing/IGG4/to_process-TUBULES.txt", "r") as file:
        lines = [line.strip() for line in file]
    
    for file in lines:
        # Skip if file already exists
        if os.path.exists(file):
            print(f"File {file} already exists. Skipping...", flush=True)
            continue
            
        match = re.search(r'/([^/_]+_[^/_]+)_\d+_r_(\d+)\.0_c_(\d+)\.0\.npy$', file)
        if not match:
            print(f"Could not parse filename: {file}", flush=True)
            continue
            
        c_name = match.group(1) 
        r = int(match.group(2))
        c = int(match.group(3))
        sample, area = c_name.split('_')
        
        fps = [
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites-OLD/{sample}/{area}/CD138_AF488_4_IgG4_{sample}_{area}.tif', # 
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites-OLD/{sample}/{area}/HLAII_Cy5_11_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites-OLD/{sample}/{area}/DAPI_UV_1_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites-OLD/{sample}/{area}/CD10_AF488_10_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites-OLD/{sample}/{area}/Claudin1_Cy5_10_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites-OLD/{sample}/{area}/MUC1_AF750_10_IgG4_{sample}_{area}.tif',
        ]   
        
        # Check if all image files exist
        all_files_exist = True
        for fp in fps:
            if not os.path.exists(fp):
                print(f"Image file does not exist: {fp}", flush=True)
                all_files_exist = False
                
        if not all_files_exist:
            print(f"Skipping due to missing image files for {sample}_{area}", flush=True)
            continue
            
        file_path = f'/project/mclark/SCAMPI_datasets/IgG4/wholeCell_segmentations_WS/{sample}/{area}/DAPI_UV_1_IgG4_{sample}_{area}_seg.npy'
        
        if not os.path.exists(file_path):
            print(f"Segmentation file does not exist: {file_path}", flush=True)
            continue
            
        try:
            pred_data = np.load(file_path, allow_pickle=True).item()
            pred_data = pred_data['masks']
            pred_data = np.squeeze(pred_data)
            
            # Check if coordinates are within bounds
            if r >= pred_data.shape[0] or c >= pred_data.shape[1]:
                print(f"Coordinates out of bounds: r={r}, c={c} for shape {pred_data.shape}", flush=True)
                print(f"Skipping {file}", flush=True)
                continue
                
            value = pred_data[int(r), int(c)]
            
            # Ensure the value is not 0 (background)
            if value == 0:
                print(f"Value at coordinates r={r}, c={c} is 0 (background). Skipping {file}", flush=True)
                continue
                
            output_file_path = f'/project/mclark/For/Bayard/Ground_truthing/IGG4/cut_samples-TUBULES/{sample}_{area}_{value}_r_{r}_c_{c}.npy'
            
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            
            # Process each image
            opened_lst = []
            for img_path in fps:
                try:
                    img = np.array(Image.open(img_path))
                    opened_lst.append(img)
                except Exception as e:
                    print(f"Error opening image {img_path}: {e}", flush=True)
                    break
                    
            if len(opened_lst) != len(fps):
                print(f"Failed to open all images for {sample}_{area}", flush=True)
                continue
                
            images = []
            for img in opened_lst:
                cropped = cut_sample(value, img, pred_data)
                if cropped is None:
                    break
                images.append(cropped)
                
            if len(images) != len(opened_lst):
                print(f"Failed to crop all images for cell {value} in {sample}_{area}", flush=True)
                continue
                
            # Process mask
            mask = cut_sample(value, pred_data, pred_data)
            if mask is None:
                print(f"Failed to create mask for cell {value} in {sample}_{area}", flush=True)
                continue
                
            images.append(mask)
            
            # Stack and save
            try:
                stacked_array = np.stack(images, axis=0)  
                np.save(output_file_path, stacked_array)
                
                # Record successful processing
                txt_file_path = '/project/mclark/For/Bayard/Ground_truthing/IGG4/processed-TUBULES.txt'
                with open(txt_file_path, 'a') as file:
                    file.write(output_file_path + '\n')
                    print(f"Successfully processed and saved: {output_file_path}", flush=True)
            except Exception as e:
                print(f"Error saving stacked array: {e}", flush=True)
                
        except Exception as e:
            print(f"Error processing {file}: {e}", flush=True)
    
if __name__ == '__main__':
    main()
