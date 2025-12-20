import pandas as pd
import numpy as np
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os
import re


Image.MAX_IMAGE_PIXELS = None  # Set to None to disable the limit



def cut_sample(cellid,img,pred_data):
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

import csv


def main():
    with open("/project/mclark/For/Bayard/Ground_truthing/IGG4/to_process.txt", "r") as file:
        lines = [line.strip() for line in file]
    for file in lines:
        if os.path.exists(file):
            continue
        match = re.search(r'/([^/_]+_[^/_]+)_\d+_r_(\d+)\.0_c_(\d+)\.0\.npy$', file)
        c_name = match.group(1) 
        r = int(match.group(2))
        c = int(match.group(3))
        sample, area = c_name.split('_')
      #  print(sample,flush=True)
      #  print(area,flush=True)
      #  print(r,c,flush=True)
      #  print(file,flush=True)
      #  raise
        fps = [
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/CD3_Atto550_12_IgG4_{sample}_{area}.tif', #
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/CD4_AF488_6_IgG4_{sample}_{area}.tif', #
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/CD8_AF488_7_IgG4_{sample}_{area}.tif', #
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/TCRd_Cy5_4_IgG4_{sample}_{area}.tif', # NOTE
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/CD20_AF488_8_IgG4_{sample}_{area}.tif', #
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/CD138_AF488_4_IgG4_{sample}_{area}.tif', # 
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/SLAMF7_Atto550_1_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/CD11c_AF488_1_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/BDCA1_Cy5_1_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/BDCA2_AF488_2_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/aSMA_AF488_15_IgG4_{sample}_{area}.tif', # NOTE
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/CD16_Atto550_7_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/HLAII_Cy5_11_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/CD68_Cy5_13_IgG4_{sample}_{area}.tif', #
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/CD163_AF488_13_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/MERTK_Atto550_13_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/CD56_Cy5_6_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/Foxp3_Atto550_4_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/DAPI_UV_1_IgG4_{sample}_{area}.tif',
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/CD34_AF488_11_IgG4_{sample}_{area}.tif', # NOTE
            f'/project/mclark/SCAMPI_datasets/IgG4/Normalized_composites/{sample}/{area}/T-bet_Atto550_5_IgG4_{sample}_{area}.tif'
        ]   

        


        file_path = f'/project/mclark/SCAMPI_datasets/IgG4/wholeCell_segmentations_WS/{sample}/{area}/DAPI_UV_1_IgG4_{sample}_{area}_seg.npy'
        pred_data = np.load(file_path, allow_pickle=True).item()
        pred_data = pred_data['masks']
        pred_data = np.squeeze(pred_data)
        if int(r) >= pred_data.shape[0] or int(c) >= pred_data.shape[1]:
            continue
        value = pred_data[int(r), int(c)]
        ffpp = f'/project/mclark/For/Bayard/Ground_truthing/IGG4/cut_samples/{sample}_{area}_{value}_r_{r}_c_{c}.npy'
        if os.path.exists(ffpp):
            continue
        opened_lst = [np.array(Image.open(file_path)) for file_path in fps]
        images = [cut_sample(value, img,pred_data) for img in opened_lst]
        mask = cut_sample(value,pred_data,pred_data)
        images = images + [mask]
        stacked_array = np.stack(images, axis=0)  
        file_path = f'/project/mclark/For/Bayard/Ground_truthing/IGG4/cut_samples/{sample}_{area}_{value}_r_{r}_c_{c}.npy'
        np.save(file_path, stacked_array)
        txt_file_path = '/project/mclark/For/Bayard/Ground_truthing/IGG4/processed.txt'
        with open(txt_file_path, 'a') as file:
            file.write(file_path + '\n')
            print("WRITE!",flush=True)
            
     

if __name__ == '__main__':
    main()
