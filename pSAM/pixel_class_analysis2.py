#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:14:23 2023
Modified for memory efficiency

@author: durkeems
"""
import os, argparse
import numpy as np
import pandas as pd
import pickle as pkl
from tifffile import imread, imwrite
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sn
import matplotlib
import gc  # For explicit garbage collection

matplotlib.use('Agg')  # Use a non-interactive backend

def log_memory_usage(message):
    """Monitor memory usage for debugging"""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        print(f"{message}: {memory_info.rss / (1024 * 1024 * 1024):.2f} GB")
    except ImportError:
        print(f"{message}: Memory monitoring requires psutil package")

def rgb_to_hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def make_color_MIP(mdir, mddir, tseg_dir, cats, mcats, ncats, cdict, comp):
    """Create color MIPs with reduced memory usage"""
    print(f"Creating color MIP for {comp}")
    
    # Process categories in groups to reduce memory footprint
    process_color_map(mdir, mddir, tseg_dir, cats, cdict, comp, "")
    process_color_map(mdir, mddir, tseg_dir, mcats, cdict, comp, "mem")
    process_color_map(mdir, mddir, tseg_dir, ncats, cdict, comp, "nuc")
    
    gc.collect()  # Force garbage collection

def process_color_map(mdir, mddir, tseg_dir, cats, cdict, comp, prefix):
    """Process a single color map type (normal, membrane, or nucleus)"""
    # Get tissue mask
    tims = os.listdir(tseg_dir)
    tim = [x for x in tims if comp in x]
    if not tim:
        print(f"Warning: No tissue mask found for {comp}")
        return
        
    timg = imread(os.path.join(tseg_dir, tim[0]))
    timg[timg > 0] = 1
    
    # File paths
    class_file = os.path.join(mdir, f"{comp}_{prefix}MaxClass.tif" if prefix else f"{comp}_maxClass.tif")
    score_file = os.path.join(mdir, f"{comp}_{prefix}MaxScore.tif" if prefix else f"{comp}_maxScore.tif")
    out_file = os.path.join(mddir, f"{comp}_{prefix}ColorMIP.tif" if prefix else f"{comp}_colorMIP.tif")
    scaled_out_file = os.path.join(mddir, f"{comp}_{prefix}ColorMIP-Scaled.tif" if prefix else f"{comp}_colorMIP-Scaled.tif")
    
    # Check if files exist
    if not (os.path.exists(class_file) and os.path.exists(score_file)):
        print(f"Warning: Input files not found for {comp} {prefix}")
        return
    
    # Load class and score images
    catimg = imread(class_file)
    scoreimg = imread(score_file)
    
    # Apply tissue mask and adjust class indices
    catimg = (catimg + 1) * timg
    scoreimg = scoreimg * timg
    
    # Get image dimensions
    r, c = catimg.shape
    
    # Create output RGB images
    rgb_disp = np.zeros([r, c, 3], dtype=np.uint8)
    rgb_scaled = np.zeros([r, c, 3], dtype=np.uint8)
    
    # Add background to categories
    label1 = ['Bkgd']
    categories = label1 + cats
    
    # Process one category at a time to reduce memory usage
    for i, cat in enumerate(categories):
        if cat == 'Bkgd':
            continue
            
        # Find pixels belonging to this category
        category_mask = (catimg == i)
        if np.any(category_mask):
            # Get coordinates for this category
            y_coords, x_coords = np.where(category_mask)
            
            # Set colors for this category
            for channel in range(3):
                rgb_disp[y_coords, x_coords, channel] = cdict[cat][channel]
        
        # Release memory
        del category_mask
        if 'y_coords' in locals():
            del y_coords, x_coords
            
    # Now calculate the scaled version (same as original code)
    # We do this after assigning all RGB values to ensure identical results
    rgb_scaled = rgb_disp.copy()
    rgb_scaled[:,:,0] = np.uint8(np.float32(rgb_scaled[:,:,0]) * (scoreimg/255))
    rgb_scaled[:,:,1] = np.uint8(np.float32(rgb_scaled[:,:,1]) * (scoreimg/255))
    rgb_scaled[:,:,2] = np.uint8(np.float32(rgb_scaled[:,:,2]) * (scoreimg/255))
            
    # Save output images
    imwrite(out_file, rgb_disp)
    imwrite(scaled_out_file, rgb_scaled)
    
    # Release memory
    del catimg, scoreimg, rgb_disp, rgb_scaled
    gc.collect()

def process_category_chunk(sam_dir, sample, area, categories, r, c, timg=None):
    """Process a chunk of categories to find max scores and classes"""
    chunk_stack = np.zeros((r, c, len(categories)), dtype=np.uint8)
    
    for j, cat in enumerate(tqdm(categories, desc=f"Processing {len(categories)} categories")):
        # Find image for this category
        imname = [x for x in os.listdir(os.path.join(sam_dir, sample, area)) if x.split('_')[2] == cat]
        if not imname:
            continue
            
        # Load image and apply tissue mask if provided
        img = imread(os.path.join(sam_dir, sample, area, imname[0]))
        if timg is not None:
            img = img * timg
            
        # Add to stack
        chunk_stack[:, :, j] = img
        
        # Release memory
        del img
    
    # Find max score and class for this chunk
    chunk_maxscore = np.max(chunk_stack, axis=2)
    chunk_maxclass = np.argmax(chunk_stack, axis=2)
    
    # Release memory
    del chunk_stack
    gc.collect()
    
    return chunk_maxclass, chunk_maxscore

def get_class_map_MIPs_single_batch(batch):
    """Process a single batch with memory efficiency improvements"""
    sample = batch['Sample']
    area = batch['Area']
    sam_dir = batch['SAMDir']
    colordict = batch['ColorDict']
    mip_dir = batch['MIPSaveDir']
    mip_display_dir = batch['MIPDisplayDir']
    tissueseg_dir = batch['TissueSegDir']
    
    print(f"Processing {sample}_{area}")
    
    # Get list of images and extract categories
    ims = os.listdir(os.path.join(sam_dir, sample, area))
    cats = sorted(list(set([x.split('_')[2] for x in ims])))
    nuccats = sorted([x for x in cats if 'Nuc' in x])
    memcats = sorted([x for x in cats if 'Nuc' not in x])
    
    # Get tissue mask if available
    tims = os.listdir(tissueseg_dir)
    tim = [x for x in tims if sample in x and area in x]
    if tim:
        timg = imread(os.path.join(tissueseg_dir, tim[0]))
        timg = (timg > 0).astype(np.uint8)
    else:
        timg = None
    
    # Get image dimensions from first image
    first_image = os.path.join(sam_dir, sample, area, ims[0])
    tmp = imread(first_image)
    r, c = tmp.shape
    del tmp
    gc.collect()
    
    # Process each category set separately with chunking
    chunk_size = 5  # Process 5 categories at a time to reduce memory usage
    
    # 1. Process all categories
    process_category_set(sam_dir, sample, area, cats, mip_dir, timg, r, c, chunk_size, "")
    
    # 2. Process membrane categories
    if memcats:
        process_category_set(sam_dir, sample, area, memcats, mip_dir, timg, r, c, chunk_size, "mem")
    
    # 3. Process nucleus categories
    if nuccats:
        process_category_set(sam_dir, sample, area, nuccats, mip_dir, timg, r, c, chunk_size, "nuc")
    
    # Create color MIP if this is the display component
    if (sample + '_' + area) == batch['DispComp']:
        make_color_MIP(mip_dir, mip_display_dir, tissueseg_dir, cats, memcats, nuccats, colordict, batch['DispComp'])
    
    gc.collect()  # Force garbage collection

def process_category_set(sam_dir, sample, area, categories, mip_dir, timg, r, c, chunk_size, prefix):
    """Process a set of categories in chunks to reduce memory usage"""
    print(f"Processing {len(categories)} {prefix} categories for {sample}_{area}")
    
    # Initialize output arrays
    maxclass = np.zeros((r, c), dtype=np.uint8)
    maxscore = np.zeros((r, c), dtype=np.uint8)
    
    # Process in chunks to reduce memory usage
    for chunk_start in range(0, len(categories), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(categories))
        chunk_categories = categories[chunk_start:chunk_end]
        
        # Process this chunk
        chunk_maxclass, chunk_maxscore = process_category_chunk(
            sam_dir, sample, area, chunk_categories, r, c, timg
        )
        
        # Update global max score and class where the new score is higher
        update_mask = chunk_maxscore > maxscore
        maxscore[update_mask] = chunk_maxscore[update_mask]
        maxclass[update_mask] = chunk_maxclass[update_mask] + chunk_start  # Adjust class index
        
        # Release memory
        del chunk_maxclass, chunk_maxscore, update_mask
        gc.collect()
    
    # Save results
    file_prefix = f"{sample}_{area}_"
    if prefix:
        file_prefix += f"{prefix}MaxClass.tif" 
    else:
        file_prefix += "maxClass.tif"
    
    imwrite(os.path.join(mip_dir, file_prefix), maxclass.astype(np.uint8))
    
    score_prefix = file_prefix.replace("MaxClass.tif", "MaxScore.tif").replace("maxClass.tif", "maxScore.tif")
    imwrite(os.path.join(mip_dir, score_prefix), maxscore.astype(np.uint8))
    
    # Save category order
    order_prefix = file_prefix.replace("MaxClass.tif", "CatOrder.pkl").replace("maxClass.tif", "catOrder.pkl")
    if prefix:
        order_prefix = order_prefix.replace("CatOrder.pkl", f"{prefix[:-3]}CatOrder.pkl")
    with open(os.path.join(mip_dir, order_prefix), 'wb') as f:
        pkl.dump(categories, f)
    
    # Release memory
    del maxclass, maxscore
    gc.collect()

def get_class_map_MIPs(batches):
    """Process batches with memory efficiency improvements"""
    for batch in batches:
        get_class_map_MIPs_single_batch(batch)

def process_sample_summary(mip_dir, tissueseg_dir, sample, area, area_dir, colordict, dispname_dict):
    """Process summary statistics for a single sample/area"""
    comp = f"{sample}_{area}"
    px_dicts = []
    
    # Get category list
    orderfile = f"{comp}_catOrder.pkl"
    
    if not os.path.exists(os.path.join(mip_dir, orderfile)):
        print(f"Order file not found for {comp}")
        return None
    
    print(f"Processing summary for {sample}_{area}")
    
    # Load categories
    p1 = ['Bkgd']
    with open(os.path.join(mip_dir, orderfile), 'rb') as f:
        p2 = pkl.load(f)
    p = p1 + p2
    
    # Load tissue mask
    tims = os.listdir(tissueseg_dir)
    tim = [x for x in tims if sample in x and area in x]
    if tim:
        timg = imread(os.path.join(tissueseg_dir, tim[0]))
        timg[timg > 0] = 1
    else:
        print(f"No tissue mask found for {sample}_{area}")
        return None
    
    # Load category and score images
    catimg_path = os.path.join(mip_dir, f"{comp}_maxClass.tif")
    scoreimg_path = os.path.join(mip_dir, f"{comp}_maxScore.tif")
    
    if not (os.path.exists(catimg_path) and os.path.exists(scoreimg_path)):
        print(f"Class or score image not found for {comp}")
        return None
    
    # Load images
    catimg = imread(catimg_path)
    scoreimg = imread(scoreimg_path)
    
    # Apply tissue mask
    scoreimg = scoreimg * timg
    catimg = (catimg + 1) * timg
    
    # Calculate statistics for each category
    cats, cts = np.unique(catimg, return_counts=True)
    
    for i, cat in enumerate(p):
        pdict = {'Sample': sample, 'Area': area, 'CompName': comp, 'Cat': cat}
        if i in cats:
            idx = list(cats).index(i)
            pdict.update({'MaxCounts': cts[idx]})
            # Calculate mean score only where category is present
            if cts[idx] > 0:
                pdict.update({'MeanScoreWhereMax': np.mean(scoreimg[catimg == i])})
            else:
                pdict.update({'MeanScoreWhereMax': 0})
        else:
            pdict.update({'MaxCounts': 0})
            pdict.update({'MeanScoreWhereMax': 0})
        px_dicts.append(pdict)
    
    # Create summary dataframe
    new_df = pd.DataFrame(px_dicts)
    
    # Release memory
    del catimg, scoreimg, timg
    if 'cats' in locals():
        del cats, cts
    gc.collect()
    
    # Create plots if we have data
    if not new_df.empty:
        create_plots(new_df, comp, area_dir, colordict, dispname_dict)
    
    return new_df

def create_plots(df, comp, area_dir, colordict, dispname_dict):
    """Create plots for a single component"""
    # Filter data for this component and exclude background
    df2 = df[(df['CompName'] == comp) & (df['Cat'] != 'Bkgd')].copy()
    
    if df2.empty:
        print(f"No non-background data for {comp}")
        return
    
    # Get unique categories and set up colors
    catcolors = df2['Cat'].unique()
    try:
        colors = [rgb_to_hex(colordict[x][0], colordict[x][1], colordict[x][2]) for x in catcolors]
        sn.set_palette(sn.color_palette(colors))
        dispnames = [dispname_dict.get(x, x) for x in catcolors]
    except KeyError as e:
        print(f"Warning: Color dictionary missing key {e}. Using default colors.")
        sn.set_palette("husl", n_colors=len(catcolors))
        dispnames = [str(x) for x in catcolors]
    
    # Calculate class prevalence
    df2.loc[:, 'MeanScoreWhereMax'] = df2['MeanScoreWhereMax'] / 255
    df2.loc[:, 'Class Prevalence'] = df2['MaxCounts'] / df2['MaxCounts'].sum()
    
    # Create scatter plot
    scatter_path = os.path.join(area_dir, f"{comp}_MaxClass-by-MaxScore.png")
    print(f"Creating scatter plot: {scatter_path}")
    plt.figure(figsize=(20, 12))
    sn.scatterplot(data=df2, x='Class Prevalence', y='MeanScoreWhereMax',
                   hue='Cat', hue_order=catcolors, sizes=[250]*len(catcolors))
    plt.xscale('log')
    plt.yscale('log')
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.xlabel('Pixel Class Prevalence', fontsize=8)
    plt.ylabel('Mean Class Score', fontsize=8)
    plt.legend(loc=2, bbox_to_anchor=(1.05, 1.0))
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=600)
    plt.close()
    
    # Create bar plots
    bar_paths = [
        os.path.join(area_dir, f"{comp}_Class-by-Max_log.png"),
        os.path.join(area_dir, f"{comp}_Class-by-Max.png"),
        os.path.join(area_dir, f"{comp}_Class-by-MaxScore.png")
    ]
    
    for bar_path in bar_paths:
        print(f"Creating bar plot: {bar_path}")
        plt.figure(figsize=(8, 3.5))
        sn.barplot(data=df2, x='Cat', y='Class Prevalence',
                  hue='Cat', dodge=False, hue_order=catcolors, order=catcolors)
        plt.xticks(ticks=np.arange(0, len(catcolors)), labels=dispnames, rotation=90, fontsize=6)
        plt.ylabel('Pixel Class Prevalence', fontsize=8)
        plt.xlabel('')
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.2, top=0.8)
        plt.savefig(bar_path, dpi=600)
        plt.close()
    
    # Release memory
    del df2
    gc.collect()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir", type=str, default='/project/mclark/SCAMPI_datasets', help="")
    parser.add_argument("--dataset", type=str, default='AAV', help="")
    parser.add_argument("--tissuesegs", type=str, default='tissue_composite_masks', help="")
    parser.add_argument("--sam", type=str, default='spectral_angle_mapping/class_maps', help="")
    parser.add_argument("--sample", type=str, default='223253423242', help="")
    parser.add_argument("--area", type=str, default='Area3', help="")
    parser.add_argument("--pxsz", type=float, default=0.1507, help="")
    parser.add_argument("--save", type=str, default='SAM_analysis', help="")
    parser.add_argument("--max_workers", type=int, default=4, help="Maximum workers for parallel processing")
    parser.add_argument("--chunk_size", type=int, default=5, help="Number of categories to process at once")

    args, unparsed = parser.parse_known_args()
    
    tissueseg_dir = os.path.join(args.rootdir, args.dataset, args.tissuesegs)
    sam_dir = os.path.join(args.rootdir, args.dataset, args.sam)
    save_dir = os.path.join(args.rootdir, args.dataset, args.save)
    mip_dir = os.path.join(save_dir, 'MIPs')
    mip_display_dir = os.path.join(save_dir, 'MIP_display')
    
    # Create directories if they don't exist
    for directory in [save_dir, mip_dir, mip_display_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    # Get sample list (filtered as in original code)
    samples = os.listdir(sam_dir)
    samples = [x for x in samples if 'S15' not in x]
    samples = [x for x in samples if len(x.split('.')) == 1]
    
    # Filter samples based on arguments
    if args.sample != 'all_samples':
        samples = [x for x in samples if x == args.sample]
        if not samples:
            print(f"Warning: Sample {args.sample} not found")
            return
    
    # Set up color dictionary based on dataset (same as original)
    if args.dataset in ['IgG4', 'AAV', 'NK2']:
        colordict = {
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
'Interstitialfibrosis': [255, 105, 180],        # Hot pink
'Storiformfibrosis': [255, 182, 193],           # Light pink

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

# Nucleus class (Highly distinct colors across spectrum)
'GenericNucleus': [25, 25, 112],                # Midnight blue
'Foxp3+Nucleus': [255, 0, 255],                 # Magenta
'Foxp3+TCF1+Nucleus': [218, 112, 214],          # Orchid
'Foxp3+TOX+Nucleus': [238, 130, 238],           # Violet
'Foxp3+TCF1+TOX+Nucleus': [221, 160, 221],      # Plum
'Tbet+Nucleus': [0, 0, 205],                    # Medium blue
'Tbet+TCF1+Nucleus': [100, 149, 237],           # Cornflower blue
'Tbet+TOX+Nucleus': [30, 144, 255],             # Dodger blue
'Tbet+TCF1+TOX+Nucleus': [135, 206, 250],       # Light sky blue
'Ki67+Nucleus': [220, 20, 60],                  # Crimson
'TCF1+Nucleus': [0, 206, 209],                  # Dark turquoise
'Ki67+TCF1+Nucleus': [255, 20, 147],            # Deep pink 
'TCF1+TOX+Nucleus': [255, 140, 0],              # Dark orange
'Ki67+TCF1+TOX+Nucleus': [255, 127, 80],        # Coral
'TOX+Nucleus': [50, 205, 50],                   # Lime green
'ApoptoticNucleus': [189, 183, 107],            # Dark khaki
'NucBackground': [230, 230, 250]                # Lavender
        }

        disp_names = [
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
    'Myofibroblast-SmoothMuscle', 'CD90+Myofibroblast', 'MMT', 'Interstitialfibrosis', 'Storiformfibrosis',
    'ProximalTubular', 'DistalTubular', 'StressedTubular',
    'Endothelial', 'LymphaticEndothelial', 'Podocyte', 'PEC',
    
    # Other
    'Apoptotic', 'RBC', 'TubuleBackground',
    
    # Nucleus
    'GenericNucleus', 'Foxp3+Nucleus', 'Foxp3+TCF1+Nucleus', 'Foxp3+TOX+Nucleus', 
    'Foxp3+TCF1+TOX+Nucleus', 'Tbet+Nucleus', 'Tbet+TCF1+Nucleus', 'Tbet+TOX+Nucleus', 
    'Tbet+TCF1+TOX+Nucleus', 'Ki67+Nucleus', 'TCF1+Nucleus', 'Ki67+TCF1+Nucleus', 
    'TCF1+TOX+Nucleus', 'Ki67+TCF1+TOX+Nucleus', 'TOX+Nucleus', 'ApoptoticNucleus', 'NucBackground'
        ]

        dispname_dict = {key: disp_names[i] for i, key in enumerate(colordict.keys())}
        
    else:
        colordict = {
            'B': [28, 25, 227], 
            'CD4T': [237, 14, 29], 
            'Endothelial cell': [186, 174, 141], 
            'NKT': [245, 210, 157],
            'Plasma': [9, 237, 188],
            'Unclassified': [168, 167, 165],
        }

        disp_names = [
            'B cell', 'CD4T cell', 'Endothelial cell', 'NKT cell', 'Plasma cell', 'Unclassified'
        ]

        dispname_dict = {key: disp_names[i] for i, key in enumerate(colordict.keys())}

    # Set up batches
    batches = []
    for sample in samples:
        areas = os.listdir(os.path.join(sam_dir, sample))
        for area in areas:
            if args.area != 'all_areas' and args.area != area:
                continue
                
            # Handle BC_CODEX dataset
            if args.dataset == 'BC_CODEX':
                color_list = list(colordict.values())
                kys = [x.split('_')[2] for x in os.listdir(os.path.join(sam_dir, sample, area))]
                colordict = {ky: color_list[i] for i, ky in enumerate(kys)}
                dispname_dict = {ky: ky for ky in kys}
            
            # Set display component
            if args.dataset == 'PSC_CRC':
                dcomp = 'S16-14100F1_Area1'
            else:
                dcomp = sample + '_' + area
                
            batch_dict = {
                'Sample': sample,
                'Area': area,
                'SAMDir': sam_dir,
                'ColorDict': colordict,
                'MIPSaveDir': mip_dir,
                'MIPDisplayDir': mip_display_dir,
                'TissueSegDir': tissueseg_dir,
                'DispComp': dcomp
            }
            batches.append(batch_dict)
    
    # Determine number of workers based on available resources
    num_workers = min(args.max_workers, len(batches))
    print(f"Processing {len(batches)} batches using {num_workers} workers")
    
    # Process in smaller batches to avoid memory issues
    if len(batches) <= num_workers:
        # Process one batch per worker
        batches = [[x] for x in batches]
    else:
        # Group batches more efficiently
        batchsize = int(np.ceil(len(batches) / num_workers))
        grouped_batches = []
        for i in range(0, len(batches), batchsize):
            batch_group = batches[i:i+batchsize]
            grouped_batches.append(batch_group)
        batches = grouped_batches
    
    # Process batches in parallel with controlled memory usage
    with mp.Pool(num_workers) as pool:
        pool.map(get_class_map_MIPs, batches)
        # Explicitly clean up to help with memory management
        pool.close()
        pool.join()
    
    # Process summary in a memory-efficient way
    summary_file_path = os.path.join(save_dir, 'MIP_pixel_summary.csv')
    if os.path.exists(summary_file_path):
        # Instead of loading the entire file, we'll append to it incrementally
        print(f"Found existing summary file: {summary_file_path}")
    else:
        # Initialize a file with proper headers
        initial_df = pd.DataFrame(columns=['Sample', 'Area', 'CompName', 'Cat', 'MaxCounts', 'MeanScoreWhereMax'])
        initial_df.to_csv(summary_file_path, index=False)
        print(f"Created new summary file: {summary_file_path}")
    
    # Process one sample/area at a time to manage memory
    for sample in samples:
        sample_dir = os.path.join(save_dir, sample)
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        
        # Get areas for this sample
        areas = os.listdir(os.path.join(sam_dir, sample))
        if args.area != 'all_areas':
            areas = [x for x in areas if x == args.area]
            if not areas:
                print(f"Warning: Area {args.area} not found in sample {sample}")
                continue
        
        for area in areas:
            # Make sure area directory exists
            area_dir = os.path.join(sample_dir, area)
            if not os.path.exists(area_dir):
                os.makedirs(area_dir)
            
            print(f"Processing summary for sample {sample}, area {area}")
            
            # Process this sample/area and get summary data
            new_df = process_sample_summary(
                mip_dir, tissueseg_dir, sample, area, area_dir, colordict, dispname_dict
            )
            
            # Append new data to summary file if we have any
            if new_df is not None and not new_df.empty:
                # Read only the header from the existing file
                header = pd.read_csv(summary_file_path, nrows=0)
                
                # Check if header matches our new data
                if set(header.columns) != set(new_df.columns) and not header.empty:
                    print(f"Warning: Column mismatch between existing file and new data")
                    # Make sure columns match
                    new_df = new_df[header.columns]
                
                # Append to file without loading the entire thing into memory
                new_df.to_csv(summary_file_path, mode='a', header=False, index=False)
                print(f"Appended {len(new_df)} rows to summary file")
            
            # Force garbage collection
            if 'new_df' in locals():
                del new_df
            gc.collect()

if __name__ == '__main__':
    try:
        # Try to install psutil for memory monitoring if not already installed
        try:
            import psutil
        except ImportError:
            import subprocess
            print("Installing psutil for memory monitoring...")
            subprocess.check_call(['pip', 'install', 'psutil'])
            import psutil
        
        # Log initial memory state
        process = psutil.Process(os.getpid())
        print(f"Initial memory usage: {process.memory_info().rss / (1024 * 1024 * 1024):.2f} GB")
        
        # Run the main function
        main()
        
        # Log final memory state
        print(f"Final memory usage: {process.memory_info().rss / (1024 * 1024 * 1024):.2f} GB")
    except ImportError:
        # If psutil installation fails, just run the main function
        print("Running without memory monitoring")
        main()
