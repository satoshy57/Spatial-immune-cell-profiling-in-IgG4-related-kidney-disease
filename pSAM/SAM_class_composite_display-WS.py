import os
import argparse
import numpy as np
import pandas as pd
from tifffile import imread,imwrite
import multiprocessing as mp
from tqdm import tqdm
from skimage.measure import block_reduce
import matplotlib.pyplot as plt

def hex_to_rgb(hex):
  return tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))

def map_to_composite(batches):
    for batch in batches:
        cellPredDir = batch['cellPredDir']
        nucPredDir = batch['nucPredDir']
        sample = batch['Sample']
        area = batch['Area']
        PredName = batch['PredName']
        if 'S15' in PredName:
            continue

        mem_maskName = batch['MemMaskName']
        nuc_maskName = batch['NucMaskName']
        memscore_maskName = mem_maskName.replace('.tif','_score.tif')
        nucscore_maskName = nuc_maskName.replace('.tif','_score.tif')

        saveDir = batch['SaveDir']
        
        # Check if temporary files already exist - skip if they do
        if (os.path.exists(os.path.join(saveDir, mem_maskName)) and
            os.path.exists(os.path.join(saveDir, nuc_maskName)) and
            os.path.exists(os.path.join(saveDir, memscore_maskName)) and
            os.path.exists(os.path.join(saveDir, nucscore_maskName))):
            print(f"Temporary files for {sample}_{area} (batch {mem_maskName}) already exist. Skipping...")
            continue
            
        if not os.path.exists(os.path.join(cellPredDir,sample,area)):
            print('Whole cell segmentations have not yet been completed for this sample. Displaying only nucleus segmentations...')
            continue
        
        catdf = batch['CatDF']
        memcolor_dict = batch['MemColorDict']
        nuccolor_dict = batch['NucColorDict']
        minR,maxR,minC,maxC = batch['Bbox']
        
        mem_pred_comp = np.load(os.path.join(cellPredDir,sample,area,PredName),allow_pickle=True).item()
        mem_pred_comp = mem_pred_comp['masks']
        if len(mem_pred_comp.shape) == 3:
            mem_pred_comp = mem_pred_comp[0]  # Remove the first dimension if it exists
            
        nuc_pred_comp = np.load(os.path.join(nucPredDir,sample,area,PredName),allow_pickle=True).item()
        nuc_pred_comp = nuc_pred_comp['masks']
        if len(nuc_pred_comp.shape) == 3:
            nuc_pred_comp = nuc_pred_comp[0]  # Remove the first dimension if it exists

        R = maxR-minR
        C = maxC-minC
        
        mem_pred_comp = mem_pred_comp[minR:maxR,minC:maxC]
        nuc_pred_comp = nuc_pred_comp[minR:maxR,minC:maxC]
        
        # Initialize masks with correct dimensions
        mem_mask = np.zeros((maxR-minR, maxC-minC, 3), dtype=np.float32)
        mem_score_mask = np.zeros((maxR-minR, maxC-minC, 3), dtype=np.float32)
        nuc_mask = np.zeros((maxR-minR, maxC-minC, 3), dtype=np.float32)
        nuc_score_mask = np.zeros((maxR-minR, maxC-minC, 3), dtype=np.float32)

        cells = list(catdf['CellID'])
        for cell in tqdm(cells):
            cmask = np.where(mem_pred_comp==cell,1,0)
            nmask = np.where(nuc_pred_comp==cell,1,0)
            
            # Ensure cmask and nmask are 2D
            if len(cmask.shape) > 2:
                cmask = cmask.reshape(cmask.shape[-2:])
            if len(nmask.shape) > 2:
                nmask = nmask.reshape(nmask.shape[-2:])
                
            celldf = catdf[catdf['CellID']==cell]
            
            memcellcats = list(celldf['MemCatName'])
            nuccellcats = list(celldf['NucCatName'])
            
            memcatscore = list(celldf['MemCatScore'])
            nuccatscore = list(celldf['NucCatScore'])
            
            if not memcellcats or not nuccellcats:  # Skip if no categories found
                continue
                
            cellcat_mem = memcellcats[0]
            cellcolor_mem = memcolor_dict[cellcat_mem]
            cellcat_nuc = nuccellcats[0] 
            cellcolor_nuc = nuccolor_dict[cellcat_nuc]
            
            # Add broadcasting dimensions for proper multiplication
            cmask = cmask[:,:,np.newaxis]
            nmask = nmask[:,:,np.newaxis]
            
            # Update masks
            for i in range(3):
                mem_mask[:,:,i] += cmask[:,:,0] * cellcolor_mem[i]
                nuc_mask[:,:,i] += nmask[:,:,0] * cellcolor_nuc[i]
                mem_score_mask[:,:,i] += cmask[:,:,0] * cellcolor_mem[i] * memcatscore[0]
                nuc_score_mask[:,:,i] += nmask[:,:,0] * cellcolor_nuc[i] * nuccatscore[0]

        # Convert to uint8 before saving
        mem_mask = np.clip(mem_mask, 0, 255).astype(np.uint8)
        nuc_mask = np.clip(nuc_mask, 0, 255).astype(np.uint8)
        mem_score_mask = np.clip(mem_score_mask, 0, 255).astype(np.uint8)
        nuc_score_mask = np.clip(nuc_score_mask, 0, 255).astype(np.uint8)
        
        # Save the masks
        imwrite(os.path.join(saveDir,mem_maskName),mem_mask)
        imwrite(os.path.join(saveDir,nuc_maskName),nuc_mask)
        imwrite(os.path.join(saveDir,memscore_maskName),mem_score_mask)
        imwrite(os.path.join(saveDir,nucscore_maskName),nuc_score_mask)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir',type=str,default='/project/mclark/SCAMPI_datasets',help='')
    parser.add_argument('--dataset',type=str,default='IgG4',help='')
    parser.add_argument('--nuc_pred_dir',type=str,default='nucleus_segmentations_WS',help='')
    parser.add_argument('--cell_pred_dir',type=str,default='wholeCell_segmentations_WS',help='')
    parser.add_argument('--sample',type=str,default='24052882340623',help='')
    parser.add_argument('--area',type=str,default='Area1',help='')
    parser.add_argument('--catcsv',type=str,default='classify_by_dilation-sameTH/with-MPI',help='')
    parser.add_argument('--save_dir',type=str,default='Composite_predictions-withSAMclass',help='')
    parser.add_argument('--max_cpu',type=int,default=100,help='')

    args,unparsed = parser.parse_known_args()

    cellpredDir = os.path.join(args.rootdir,args.dataset,args.cell_pred_dir)
    nucpredDir = os.path.join(args.rootdir,args.dataset,args.nuc_pred_dir)
    csvDir = os.path.join(args.rootdir,args.dataset,args.catcsv)
    saveDir = os.path.join(args.rootdir,args.dataset,args.save_dir)
    
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    if not os.path.exists(os.path.join(saveDir,'downsampled')):
        os.makedirs(os.path.join(saveDir,'downsampled'))
    
    csvname = os.listdir(csvDir)
    csvname = [x for x in csvname if x.endswith('filtered5_10woMono.csv')]
    if len(csvname)>1:
        if (args.sample=='all_samples') and (args.area=='all_areas'):
            csvname=[x for x in csvname if 'combined' in x]
        else:
            csvname = [x for x in csvname if args.sample in x]
            csvname = [x for x in csvname if args.area in x]
    catcsv = pd.read_csv(os.path.join(csvDir,csvname[0]), low_memory=False)
    catcsv = catcsv.drop(columns = [x for x in catcsv.columns if 'Unnamed' in x])
    catcsv = catcsv.drop_duplicates()
    catcsv.to_csv(os.path.join(csvDir,csvname[0]))
    catcsv['MemCatScore']=catcsv['MemCatScore']/catcsv['MemCatScore'].max()
    catcsv['NucCatScore']=catcsv['NucCatScore']/catcsv['NucCatScore'].max()

    memcats = np.unique(catcsv['MemCat'])
    nuccats = np.unique(catcsv['NucCat'])
    memnames = np.unique(catcsv['MemCatName'])
    nucnames = list(np.unique(catcsv['NucCatName']))
    
    memcolor_dict = {}
    memname_dict = {}
    
    if args.dataset=='PSC_CRC':
        #PSC panel
        memcolordict={'APC':[155,5,255],'ActivatedCD8T':[10,242,14],'B':[28,25,227],
                   'CytotoxicT':[6,140,8],'HelperT':[237,14,29],'IL17-HelperT':[166,3,3],
                   'IL17CD45':[247,7,235],'IgG-Plasma':[9,237,188],'OtherT':[246,250,5],
                   'OtherCD45':[123,7,116],'Plasma':[10,166,132],'Plasmablast':[45,148,196],
                   'unclassified':[168,167,165], 'structure':[133,113,60]}
        mem_names = ['APC','GZB+ CD8T cell','B cell','CD8T cell','CD4T cell','IL17+ CD4T cell','IL17+ Other Immune cell','IgG+ Plasma cell',
                    'Other T cell','Other Immune cell','Plasma cell','Plasmablast','Unclassified','Structural cell']
        memname_dict = {ky:mem_names[i] for i,ky in enumerate(list(memcolordict.keys()))}
    else:
        #kidney panel
        memcolordict={
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
    
                 # Unclassified (Very bright white-gray)
                'unclassified': [245, 245, 245]            # Almost white gray
}
        
        mem_names = [

 # T cells
    'NaiveCD4T', 'CD4T', 'ICOS+Treg', 'CD103+CD4Treg', 'HLAII+CD27+Treg',
    'Tfh', 'TIGIT+Tfh', 'HLAII+GZMK+Tfh', 'GZMA+GZMK+CD4CTL', 'GZMB+CD4CTL', 'CD4Tcm-Tem',
    'NaiveCD8T', 'CD8T', 'GZMA+GZMK+CD8CTL', 'GZMB+CD8CTL', 'CD103+CD8Treg', 'CD8Trm',
    'CD103+CD8Trm', 'TIGIT+CD8T', 'CD8Tcm-Tem', 'γδT', 'ILC', 'ImmatureNK', 'MatureNK', 'NKT', 
    
    # Humoral cells
    'NaiveB', 'TransitionalB', 'B', 'CD27+B', 'DN2B', 'AgeAsscB',
    'Plasmablast', 'IgG4+Plasmablast', 'Plasma', 'IgG4+Plasma',
    
    # Myeloid cells
    'M1Macrophage', 'MPO+Macrophage',
    'M2Macrophage', 'MerTK+M1Macrophage', 'MerTK+M2Macrophage',
    'GPNMB+M1Macrophage', 'GPNMB+M2Macrophage', 'cDC1', 'cDC2', 'pDC',
    'Neutrophil',
    
    # Stromal and structural cells
    'Myofibroblast-SmoothMuscle', 'CD90+Myofibroblast', 'MMT', 
    'ProximalTubular', 'DistalTubular', 'StressedTubular',
    'Endothelial', 'LymphaticEndothelial', 'Podocyte', 'PEC',
    
    # Other
    'Apoptotic', 'RBC', 'TubuleBackground','unclassified']
        memname_dict = {ky:mem_names[i] for i,ky in enumerate(list(memcolordict.keys()))}
    if args.dataset == 'BC_CODEX':
        colors = list(memcolordict.values())
        memcolordict = {ky:colors[i] for i,ky in enumerate(memnames)}
        memname_dict = {ky:ky for ky in memnames}

    disp_memnames=[]
    memcolor_dict = {}
    # Use dictionary order directly from memcolordict
    for mname, color in memcolordict.items():
        # Find corresponding category from memnames mapping
        matching_cats = [cat for cat in memcats if memnames[np.where(memcats == cat)[0][0]] == mname]
        if matching_cats:
            memcat = matching_cats[0]
            memcolor_dict[memcat] = color
            disp_memnames.append(memname_dict[mname])

    CL = np.array(list(memcolor_dict.values()))
    CL = CL[:,:,np.newaxis]
    CL = np.moveaxis(CL,-1,1)
    fig,ax = plt.subplots(1,1,dpi=600,figsize=(3.5,5))
    plt.imshow(CL)
    plt.yticks(np.arange(0,len(disp_memnames)))
    plt.xticks(np.arange(0,0))
    ax.set_yticklabels(disp_memnames,rotation=0,fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(saveDir,'membrane_colorbar.png'))
    plt.close()
    
    nuccolor_dict={}
    nucname_dict={}
    #selected nucleus colors
    if args.dataset=='PSC_CRC':
        nuccolordict = {'Nucleus':[200,200,200],
                     'RegT-Nuc':[135,16,88],'pSTAT3-Nuc':[252,152,3]}
        nuc_names = ['Generic Nucleus','FoxP3+ Nucleus','pSTAT3+ Nucleus']
        nucname_dict = {ky:nuc_names[i] for i,ky in enumerate(list(nuccolordict.keys()))}
    else:
        nuccolordict = {  # Nucleus class (Highly distinct colors across spectrum)
    'GenericNucleus': [25, 25, 112],                # Midnight blue
    'Foxp3+Nucleus': [255, 0, 255],                 # Magenta
    'Foxp3+TCF1+Nucleus': [218, 112, 214],          # Orchid - new
    'Foxp3+TOX+Nucleus': [238, 130, 238],           # Violet - new
    'Foxp3+TCF1+TOX+Nucleus': [221, 160, 221],      # Plum - new
    'Tbet+Nucleus': [0, 0, 205],                    # Medium blue
    'Tbet+TCF1+Nucleus': [100, 149, 237],           # Cornflower blue - new
    'Tbet+TOX+Nucleus': [30, 144, 255],             # Dodger blue - new
    'Tbet+TCF1+TOX+Nucleus': [135, 206, 250],       # Light sky blue - new
    'Ki67+Nucleus': [220, 20, 60],                  # Crimson
    'TCF1+Nucleus': [0, 206, 209],                  # Dark turquoise
    'Ki67+TCF1+Nucleus': [255, 20, 147],            # Deep pink 
    'TCF1+TOX+Nucleus': [255, 140, 0],              # Dark orange
    'Ki67+TCF1+TOX+Nucleus': [255, 127, 80],        # Coral - new
    'TOX+Nucleus': [50, 205, 50],                   # Lime green
    'ApoptoticNucleus': [189, 183, 107],            # Dark khaki - renamed from ApoptoticNuc
    'NucBackground': [230, 230, 250]                # Lavender
}
        nuc_names = ['GenericNucleus', 'Foxp3+Nucleus', 'Foxp3+TCF1+Nucleus', 'Foxp3+TOX+Nucleus',     'Foxp3+TCF1+TOX+Nucleus', 'Tbet+Nucleus', 'Tbet+TCF1+Nucleus', 'Tbet+TOX+Nucleus',     'Tbet+TCF1+TOX+Nucleus', 'Ki67+Nucleus', 'TCF1+Nucleus', 'Ki67+TCF1+Nucleus',    'TCF1+TOX+Nucleus', 'Ki67+TCF1+TOX+Nucleus', 'TOX+Nucleus', 'ApoptoticNucleus', 'NucBackground']
        nucname_dict = {ky:nuc_names[i] for i,ky in enumerate(list(nuccolordict.keys()))}
    if args.dataset == 'BC_CODEX':
        colors = list(nuccolordict.values())
        nuccolordict = {ky:colors[i] for i,ky in enumerate(nucnames)}
        nucname_dict = {ky:ky for ky in nucnames}

    disp_nucnames = []
    nuccolor_dict = {}
    # Use dictionary order directly from nuccolordict
    for nname, color in nuccolordict.items():
        # Find corresponding category from nucnames mapping
        matching_cats = [cat for cat in nuccats if nucnames[np.where(nuccats == cat)[0][0]] == nname]
        if matching_cats:
            nuccat = matching_cats[0]
            nuccolor_dict[nuccat] = color
            disp_nucnames.append(nucname_dict[nname])

    CL = np.array(list(nuccolor_dict.values()))
    CL = CL[:,:,np.newaxis]
    CL = np.moveaxis(CL,-1,1)
    fig,ax = plt.subplots(1,1,dpi=600,figsize=(3.5,1.5))
    plt.imshow(CL)
    plt.yticks(np.arange(0,len(disp_nucnames)))
    plt.xticks(np.arange(0,0))
    ax.set_yticklabels(disp_nucnames,rotation=0,fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(saveDir,'nucleus_colorbar.png'))
    plt.close()
    
    disp_names=[]
    color_dict={}
    # Add membrane colors first, maintaining dictionary order
    for mname, color in memcolordict.items():
        matching_cats = [cat for cat in memcats if memnames[np.where(memcats == cat)[0][0]] == mname]
        if matching_cats:
            memcat = matching_cats[0]
            color_dict[memcat] = color
            disp_names.append(memname_dict[mname])

    # Add nucleus colors, maintaining dictionary order
    for nname, color in nuccolordict.items():
        matching_cats = [cat for cat in nuccats if nucnames[np.where(nuccats == cat)[0][0]] == nname]
        if matching_cats:
            nuccat = matching_cats[0]
            color_dict[nuccat + memcat + 1] = color
            disp_names.append(nucname_dict[nname])

    CL = np.array(list(color_dict.values()))
    CL = CL[:,:,np.newaxis]
    CL = np.moveaxis(CL,-1,1)
    fig,ax = plt.subplots(1,1,dpi=600,figsize=(3.5,5))
    plt.imshow(CL)
    plt.yticks(np.arange(0,len(disp_names)))
    plt.xticks(np.arange(0,0))
    ax.set_yticklabels(disp_names,rotation=0,fontsize=6)
    plt.tight_layout()
    plt.savefig(os.path.join(saveDir,'full_colorbar.png'))
    plt.close()
    
    samples = os.listdir(cellpredDir)
    compdicts = []
    locdict = {}
    for sample in samples:
        if args.sample!='all_samples':
            if sample not in args.sample:
                continue
        if 'S15' in sample:
            continue
        areas = os.listdir(os.path.join(cellpredDir,sample))
        for area in areas:
            if args.area!='all_areas':
                if area not in args.area:
                    continue
            comp=os.listdir(os.path.join(cellpredDir,sample,area))
            comp = comp[0]
            segs = np.load(os.path.join(cellpredDir,sample,area,comp),allow_pickle=True).item()
            print("Type of loaded data:", type(segs))
            print("Keys in loaded data:", segs.keys())
            segs = segs['masks']
            print("Shape of masks:", np.shape(segs))
            cells = np.unique(segs)
            cells = np.delete(cells,0)
            
            segs = segs[0]
            r,c = np.shape(segs)
            locdict.update({'Size':[r,c]})
            catdf = catcsv[catcsv['CompName']==sample+'_'+area]
            batchsize = int(np.ceil(len(cells)/100))
            for i in tqdm(range(100)):
                cellids = cells[i*batchsize:(i+1)*batchsize]
                tmp = np.where((segs>=np.min(cellids)) & (segs<=np.max(cellids)),1,0)
                cellidx = np.where(tmp==1)
                del tmp
                minR = np.min(cellidx[0])
                minC = np.min(cellidx[1])
                maxR = np.max(cellidx[0])
                maxC = np.max(cellidx[1])
                subdf = catdf[catdf['CellID'].isin(cellids)]
                mem_maskName = sample+'_'+area+'_'+str(i)+'_membrane-classes.tif'
                nuc_maskName = sample+'_'+area+'_'+str(i)+'_nucleus-classes.tif'
                locdict.update({str(i):[minR,maxR,minC,maxC]})
                compdicts.append({'cellPredDir':cellpredDir,'nucPredDir':nucpredDir,'SaveDir':saveDir,
                                  'Sample':sample,'Area':area,'PredName':comp,'MemMaskName':mem_maskName,
                                  'NucMaskName':nuc_maskName,'CatDF':subdf,'MemColorDict':memcolordict,
                                  'NucColorDict':nuccolordict,'Bbox':[minR,maxR,minC,maxC]})
                
    if len(compdicts) > args.max_cpu:
        num_workers = args.max_cpu
    else:
        num_workers = len(compdicts)

    batchsize = int(np.ceil(len(compdicts)/num_workers))
    batches = []
    for i in range(num_workers):
        batches.append(compdicts[i*batchsize:(i+1)*batchsize])
    
    pool = mp.Pool(num_workers)
    pool.map(map_to_composite,batches)
    pool.close()
    
    # Create a dictionary to store masks for each sample and area
    sample_area_masks = {}
    
    for compdict in compdicts:
        sample = compdict['Sample']
        area = compdict['Area']
        key = (sample, area)
        
        # Check if the final files already exist for this sample+area
        mem_maskName = f"{sample}_{area}_membrane-classes.tif"
        nuc_maskName = f"{sample}_{area}_nucleus-classes.tif"
        memscore_maskName = f"{sample}_{area}_membraneScore-classes.tif"
        nucscore_maskName = f"{sample}_{area}_nucleusScore-classes.tif"
        
        if (os.path.exists(os.path.join(saveDir, mem_maskName)) and
            os.path.exists(os.path.join(saveDir, nuc_maskName)) and
            os.path.exists(os.path.join(saveDir, memscore_maskName)) and
            os.path.exists(os.path.join(saveDir, nucscore_maskName)) and
            os.path.exists(os.path.join(saveDir, 'downsampled', mem_maskName)) and
            os.path.exists(os.path.join(saveDir, 'downsampled', nuc_maskName)) and
            os.path.exists(os.path.join(saveDir, 'downsampled', memscore_maskName)) and
            os.path.exists(os.path.join(saveDir, 'downsampled', nucscore_maskName))):
            print(f"Final output files for {sample}_{area} already exist. Skipping...")
            continue
        
        if key not in sample_area_masks:
            # Get the size for this sample+area from the first relevant compdict
            segs = np.load(os.path.join(cellpredDir,sample,area,compdict['PredName']),allow_pickle=True).item()
            segs = segs['masks']
            r, c = np.shape(segs[0])  # Assuming masks has shape [1, r, c]
            
            # Initialize arrays for this sample+area
            sample_area_masks[key] = {
                'mem_full': np.zeros([r,c,3]),
                'nuc_full': np.zeros([r,c,3]),
                'memscore_full': np.zeros([r,c,3]),
                'nucscore_full': np.zeros([r,c,3])
            }
        
        try:
            # Check if the temporary files exist
            if not (os.path.exists(os.path.join(compdict['SaveDir'],compdict['MemMaskName'])) and
                    os.path.exists(os.path.join(compdict['SaveDir'],compdict['NucMaskName'])) and
                    os.path.exists(os.path.join(compdict['SaveDir'],compdict['MemMaskName'].replace('.tif','_score.tif'))) and
                    os.path.exists(os.path.join(compdict['SaveDir'],compdict['NucMaskName'].replace('.tif','_score.tif')))):
                print(f"Temporary files for {sample}_{area} (batch {compdict['MemMaskName']}) not found. Skipping...")
                continue
                
            mempatch = imread(os.path.join(compdict['SaveDir'],compdict['MemMaskName']))
            nucpatch = imread(os.path.join(compdict['SaveDir'],compdict['NucMaskName']))
            memscorepatch = imread(os.path.join(compdict['SaveDir'],compdict['MemMaskName'].replace('.tif','_score.tif')))
            nucscorepatch = imread(os.path.join(compdict['SaveDir'],compdict['NucMaskName'].replace('.tif','_score.tif')))
            
            rmin,rmax,cmin,cmax = compdict['Bbox']
            
            # Ensure patches match the size of the target region
            target_height = rmax - rmin
            target_width = cmax - cmin
            
            if mempatch.shape[:2] != (target_height, target_width):
                mempatch = mempatch[:target_height, :target_width]
                nucpatch = nucpatch[:target_height, :target_width]
                memscorepatch = memscorepatch[:target_height, :target_width]
                nucscorepatch = nucscorepatch[:target_height, :target_width]
            
            # Add patches to the appropriate sample+area masks
            sample_area_masks[key]['mem_full'][rmin:rmax,cmin:cmax] += mempatch
            sample_area_masks[key]['nuc_full'][rmin:rmax,cmin:cmax] += nucpatch
            sample_area_masks[key]['memscore_full'][rmin:rmax,cmin:cmax] += memscorepatch
            sample_area_masks[key]['nucscore_full'][rmin:rmax,cmin:cmax] += nucscorepatch
            
        except Exception as e:
            print(f"Error processing {compdict['MemMaskName']}: {str(e)}")
            continue
    
    # Save separate files for each sample+area combination
    for (sample, area), masks in sample_area_masks.items():
        # Define filenames for this sample+area
        mem_maskName = f"{sample}_{area}_membrane-classes.tif"
        nuc_maskName = f"{sample}_{area}_nucleus-classes.tif"
        memscore_maskName = f"{sample}_{area}_membraneScore-classes.tif"
        nucscore_maskName = f"{sample}_{area}_nucleusScore-classes.tif"
        
        # Check if final files already exist
        if (os.path.exists(os.path.join(saveDir, mem_maskName)) and
            os.path.exists(os.path.join(saveDir, nuc_maskName)) and
            os.path.exists(os.path.join(saveDir, memscore_maskName)) and
            os.path.exists(os.path.join(saveDir, nucscore_maskName)) and
            os.path.exists(os.path.join(saveDir, 'downsampled', mem_maskName)) and
            os.path.exists(os.path.join(saveDir, 'downsampled', nuc_maskName)) and
            os.path.exists(os.path.join(saveDir, 'downsampled', memscore_maskName)) and
            os.path.exists(os.path.join(saveDir, 'downsampled', nucscore_maskName))):
            print(f"Final output files for {sample}_{area} already exist. Skipping final merge...")
            continue
        
        # Create full resolution files
        imwrite(os.path.join(saveDir, mem_maskName), masks['mem_full'].astype(np.uint8))
        imwrite(os.path.join(saveDir, nuc_maskName), masks['nuc_full'].astype(np.uint8))
        imwrite(os.path.join(saveDir, memscore_maskName), masks['memscore_full'].astype(np.uint8))
        imwrite(os.path.join(saveDir, nucscore_maskName), masks['nucscore_full'].astype(np.uint8))
        
        # Create downsampled versions
        small_mem_mask = block_reduce(masks['mem_full'], block_size=(4,4,1), func=np.max)
        small_nuc_mask = block_reduce(masks['nuc_full'], block_size=(4,4,1), func=np.max)
        small_memscore_mask = block_reduce(masks['memscore_full'], block_size=(4,4,1), func=np.max)
        small_nucscore_mask = block_reduce(masks['nucscore_full'], block_size=(4,4,1), func=np.max)
        
        imwrite(os.path.join(saveDir, 'downsampled', mem_maskName), small_mem_mask.astype(np.uint8))
        imwrite(os.path.join(saveDir, 'downsampled', nuc_maskName), small_nuc_mask.astype(np.uint8))
        imwrite(os.path.join(saveDir, 'downsampled', memscore_maskName), small_memscore_mask.astype(np.uint8))
        imwrite(os.path.join(saveDir, 'downsampled', nucscore_maskName), small_nucscore_mask.astype(np.uint8))
    
    # Only clean up temporary files if they're not needed anymore
    # Get list of all sample/area combinations that have complete final files
    completed_samples = []
    for sample in samples:
        if args.sample != 'all_samples' and sample not in args.sample:
            continue
        if 'S15' in sample:
            continue
        areas = os.listdir(os.path.join(cellpredDir, sample))
        for area in areas:
            if args.area != 'all_areas' and area not in args.area:
                continue
                
            # Check if this sample/area has complete final output
            mem_maskName = f"{sample}_{area}_membrane-classes.tif"
            nuc_maskName = f"{sample}_{area}_nucleus-classes.tif"
            memscore_maskName = f"{sample}_{area}_membraneScore-classes.tif"
            nucscore_maskName = f"{sample}_{area}_nucleusScore-classes.tif"
            
            if (os.path.exists(os.path.join(saveDir, mem_maskName)) and
                os.path.exists(os.path.join(saveDir, nuc_maskName)) and
                os.path.exists(os.path.join(saveDir, memscore_maskName)) and
                os.path.exists(os.path.join(saveDir, nucscore_maskName)) and
                os.path.exists(os.path.join(saveDir, 'downsampled', mem_maskName)) and
                os.path.exists(os.path.join(saveDir, 'downsampled', nuc_maskName)) and
                os.path.exists(os.path.join(saveDir, 'downsampled', memscore_maskName)) and
                os.path.exists(os.path.join(saveDir, 'downsampled', nucscore_maskName))):
                completed_samples.append((sample, area))
    
    # Only remove temporary files for completed samples
    rmfiles = []
    for compdict in compdicts:
        if (compdict['Sample'], compdict['Area']) in completed_samples:
            rmfiles.append(compdict['MemMaskName'])
            rmfiles.append(compdict['NucMaskName'])
            rmfiles.append(compdict['MemMaskName'].replace('.tif','_score.tif'))
            rmfiles.append(compdict['NucMaskName'].replace('.tif','_score.tif'))
    
    # Remove temp files for completed samples
    for rmfile in rmfiles:
        try:
            if os.path.exists(os.path.join(saveDir, rmfile)):
                os.remove(os.path.join(saveDir, rmfile))
        except Exception as e:
            print(f"Error removing temporary file {rmfile}: {str(e)}")

if __name__=="__main__":
    main()
