import os,argparse
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_multiotsu
from tifffile import imread
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp

def get_cell_class(batches):
    for batch in batches:
        save_dir = batch['SaveDir']
        sam_dir = batch['SAMDir']
        sample = batch['Sample']
        area = batch['Area']
        cellids = batch['CellIDs']
        batchnum = batch['BatchNum']
        nucseg_dir = batch['NucSegDir']
        cellseg_dir = batch['CellSegDir']
        tseg_dir = batch['TissueSegDir']
        cat_dict_nuc = batch['NucCatDict']
        cat_dict_mem = batch['MemCatDict']

        # Check if output file already exists
        output_file = os.path.join(save_dir, sample, area, f'{sample}_{area}_{batchnum}_SAM-CellCats.csv')
        if os.path.exists(output_file):
            print(f'Skipping {sample} {area} {batchnum} - output file already exists')
            continue

        print('Getting segmentations for', sample, area, batchnum, '...')
        
        try:
            # Load nuclear segmentations
            nucsegname = os.listdir(os.path.join(nucseg_dir, sample, area))
            nucsegs_data = np.load(os.path.join(nucseg_dir, sample, area, nucsegname[0]), allow_pickle=True).item()
            nucsegs = nucsegs_data['masks'].squeeze()  # Remove any singleton dimensions
            
            # Load cell segmentations
            cellsegname = os.listdir(os.path.join(cellseg_dir, sample, area))
            cellsegs_data = np.load(os.path.join(cellseg_dir, sample, area, cellsegname[0]), allow_pickle=True).item()
            cellsegs = cellsegs_data['masks'].squeeze()  # Remove any singleton dimensions
            
            # Verify shapes
            if len(nucsegs.shape) != 2 or len(cellsegs.shape) != 2:
                print(f"Error: Invalid segmentation shapes - nucsegs: {nucsegs.shape}, cellsegs: {cellsegs.shape}")
                continue
                
            # Load tissue segmentations
            tsegname = [x for x in os.listdir(tseg_dir) if (sample in x) and (area in x)]
            if not tsegname:
                print(f"Warning: No tissue segmentation found for {sample} {area}")
                continue
            tsegs = imread(os.path.join(tseg_dir, tsegname[0]))

            # Calculate membrane segmentations
            memsegs = cellsegs - nucsegs
            del cellsegs  # Free memory
            
            r, c = nucsegs.shape
            memmask = np.zeros((r, c), dtype=bool)
            valid_cells = (memsegs >= np.min(cellids)) & (memsegs <= np.max(cellids))
            memmask[valid_cells] = True

            # Find ROI boundaries
            cellidx = np.nonzero(memmask)
            if not cellidx[0].size:
                print(f"Warning: No cells found in mask for {sample} {area}")
                continue
                
            minR, maxR = np.min(cellidx[0]), np.max(cellidx[0])
            minC, maxC = np.min(cellidx[1]), np.max(cellidx[1])
            R = maxR - minR
            C = maxC - minC

            # Crop segmentations to ROI
            memsegs = memsegs[minR:maxR, minC:maxC]
            nucsegs = nucsegs[minR:maxR, minC:maxC]

            memcats = list(cat_dict_mem.values())
            nuccats = list(cat_dict_nuc.values())

            print('Stacking class maps for', sample, area, batchnum, '...')
            
            # Initialize SAM arrays
            memSAM = np.zeros((len(memcats), R, C))
            nucSAM = np.zeros((len(nuccats), R, C))

            # Process membrane categories
            for idx, cat in enumerate(memcats):
                sam_path = os.path.join(sam_dir, sample, area, f'{sample}_{area}_{cat}_similarity.tif')
                if not os.path.exists(sam_path):
                    print(f"Warning: Missing SAM file: {sam_path}")
                    continue
                tmp = imread(sam_path)
                th = 0
                tmp = np.where(tmp > th, tmp, 0)
                memSAM[idx] = tmp[minR:maxR, minC:maxC]

            # Process nuclear categories
            for idx, cat in enumerate(nuccats):
                sam_path = os.path.join(sam_dir, sample, area, f'{sample}_{area}_{cat}_similarity.tif')
                if not os.path.exists(sam_path):
                    print(f"Warning: Missing SAM file: {sam_path}")
                    continue
                tmp = imread(sam_path)
                th = 128
                tmp = np.where(tmp > th, tmp, 0)
                nucSAM[idx] = tmp[minR:maxR, minC:maxC]

            print(f'{batchnum} SAM maps stacked')

            # Process cells
            df = pd.DataFrame()
            print(f'\nCalculating cell scores for {sample} {area}...')

            for cellid in tqdm(cellids):
                try:
                    cellidx = np.where(memsegs == cellid)
                    if not cellidx[0].size:
                        print(f"Warning: Cell {cellid} not found in membrane segmentation")
                        continue
                        
                    rmin, rmax = np.min(cellidx[0]), np.max(cellidx[0])
                    cmin, cmax = np.min(cellidx[1]), np.max(cellidx[1])
                    centroid = (np.mean(cellidx[0]) + minR, np.mean(cellidx[1]) + minC)

                    # Process membrane patch
                    mempatch = memsegs[rmin:rmax, cmin:cmax]
                    mempatch = (mempatch == cellid).astype(float)
                    mempatch = distance_transform_edt(mempatch)
                    if np.max(mempatch) > 0:
                        mempatch /= np.max(mempatch)
                    mempatch = mempatch[np.newaxis, :, :]

                    # Process nuclear patch
                    nucpatch = nucsegs[rmin:rmax, cmin:cmax]
                    nucpatch = (nucpatch == cellid).astype(float)
                    nucpatch = distance_transform_edt(nucpatch)
                    if np.max(nucpatch) > 0:
                        nucpatch /= np.max(nucpatch)
                    nucpatch = nucpatch[np.newaxis, :, :]

                    # Extract and process SAM patches
                    memSAMpatch = memSAM[:, rmin:rmax, cmin:cmax]
                    nucSAMpatch = nucSAM[:, rmin:rmax, cmin:cmax]

                    memSAMpatch = mempatch * memSAMpatch
                    nucSAMpatch = nucpatch * nucSAMpatch

                    # Calculate scores
                    mem_scores = np.mean(memSAMpatch, axis=(1, 2))
                    nuc_scores = np.mean(nucSAMpatch, axis=(1, 2))

                    # Create cell dictionary
                    cell_dict = {
                        'Sample': sample,
                        'Area': area,
                        'CompName': f'{sample}_{area}',
                        'CellID': cellid,
                        'CellCentroidRow': centroid[0],
                        'CellCentroidCol': centroid[1]
                    }
                    
                    # Add scores to dictionary
                    for cat, score in zip(memcats, mem_scores):
                        cell_dict[cat] = score
                    for cat, score in zip(nuccats, nuc_scores):
                        cell_dict[cat] = score

                    # Append to dataframe
                    df = pd.concat([df, pd.DataFrame([cell_dict])], ignore_index=True)
                    
                except Exception as e:
                    print(f"Error processing cell {cellid}: {str(e)}")
                    continue

            # Save results
            save_path = os.path.join(save_dir, sample, area, f'{sample}_{area}_{batchnum}_SAM-CellCats.csv')
            df.to_csv(save_path, index=False)
            print(f'{sample} {area} {batchnum} SAM csv saved.')
            
        except Exception as e:
            print(f"Error processing batch for {sample} {area} {batchnum}: {str(e)}")
            continue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rootdir",type=str,default='/project/mclark/SCAMPI_datasets',help="")
    parser.add_argument("--dataset",type=str,default='IgG4',help="")
    parser.add_argument("--nucsegs",type=str,default='nucleus_segmentations_WS',help="")
    parser.add_argument("--cellsegs",type=str,default='wholeCell_segmentations_WS',help="")
    parser.add_argument("--tissuesegs",type=str,default='tissue_composite_masks',help="")
    parser.add_argument("--sam",type=str,default='spectral_angle_mapping/class_maps',help="")
    parser.add_argument("--sample",type=str,default='24052882340623',help="")
    parser.add_argument("--area",type=str,default='Area1',help="")
    parser.add_argument("--pxsz",type=float,default=0.1507,help="")
    parser.add_argument("--save",type=str,default='classify_by_dilation-sameTH',help="")

    args, unparsed = parser.parse_known_args()
    
    nucseg_dir=os.path.join(args.rootdir,args.dataset,args.nucsegs)
    cellseg_dir=os.path.join(args.rootdir,args.dataset,args.cellsegs)
    tseg_dir=os.path.join(args.rootdir,args.dataset,args.tissuesegs)
    sam_dir=os.path.join(args.rootdir,args.dataset,args.sam)
    save_dir=os.path.join(args.rootdir,args.dataset,args.save)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    reference_spectra = pd.read_csv(os.path.join(args.rootdir,args.dataset,'reference_spectra.csv'))
    cols = reference_spectra.columns
    cols = [x for x in cols if 'Unnamed' not in x]
    cols = [x for x in cols if 'PSTAT' not in x]
    cols = [x for x in cols if x!='RBCs']
 
    cat_dict_mem={}
    cat_dict_nuc={}
    for i,col in enumerate(cols):
        if 'Nuc' in col:
            cat_dict_nuc.update({i:col})
        elif col=='blank':
            continue
        elif 'arker' in col:
            continue
        else:
            cat_dict_mem.update({i:col})
    
    cellbatches = []
        
    samples = os.listdir(sam_dir)
    samples = [x for x in samples if len(x.split('.'))==1]
    samples = [x for x in samples if 'S15' not in x]
    
    for sample in samples:
        if args.sample!='all_samples':
            if args.sample!=sample:
                continue
        if not os.path.exists(os.path.join(save_dir,sample)):
            os.makedirs(os.path.join(save_dir,sample))
        areas = os.listdir(os.path.join(sam_dir,sample))
        for area in areas:
            if args.area!='all_areas':
                if args.area!=area:
                    continue
            if not os.path.exists(os.path.join(save_dir,sample,area)):
                os.makedirs(os.path.join(save_dir,sample,area))
            if os.path.exists(os.path.join(save_dir,sample,area,'_'.join([sample,area,'SAM-CellCats.csv']))):
                continue
            segname = os.listdir(os.path.join(nucseg_dir,sample,area))
            segs = np.load(os.path.join(nucseg_dir,sample,area,segname[0]),allow_pickle=True).item()
            cells = np.unique(segs['masks'])
            del segs
            cells = np.delete(cells,0)
            cbatch = 50
            cellbatchsize = int(np.ceil(len(cells)/cbatch))
            batches=[]
            for i in range(cbatch):
                cellbatch = cells[i*cellbatchsize:(i+1)*cellbatchsize]
                batchnum = len(batches)
                batches.append({'SaveDir':save_dir,'SAMDir':sam_dir,'Sample':sample,
                                'Area':area,'NucSegDir':nucseg_dir,'CellSegDir':cellseg_dir,
                                'TissueSegDir':tseg_dir,'CellIDs':cellbatch,'BatchNum':batchnum,
                                'NucCatDict':cat_dict_nuc,'MemCatDict':cat_dict_mem})
            cellbatches.extend(batches)
    cellbatches = [x for x in cellbatches if len(x['CellIDs'])>0]
    print(len(cellbatches))
    num_workers=100
    if len(cellbatches)<=num_workers:
        num_workers = len(cellbatches)
        batches = [[x] for x in cellbatches]
        del cellbatches
    else:
        batchsize = int(np.ceil(len(cellbatches)/num_workers))
        batches=[]
        for i in range(num_workers):
            batches.append(cellbatches[i*batchsize:(i+1)*batchsize])
        del cellbatches
    print('')
    
    print(num_workers)
    pool=mp.Pool(num_workers)
    pool.map(get_cell_class,batches)
    pool.close()
    

    master_df = pd.DataFrame()
    clist = []
    for sample in samples:
        if sample=='S15-13202E1':
            continue
        if args.sample!='all_samples':
            if args.sample!=sample:
                continue
        areas = os.listdir(os.path.join(save_dir,sample))
        for area in areas:
            if (sample=='S20-16349D1') and (area=='Area2'):
                continue
            if (sample=='S16-14100F1') and (area=='Area3'):
                continue
            if args.area!='all_areas':
                if args.area!=area:
                    continue
            compdf = pd.DataFrame()
            cname = '_'.join([sample,area,'SAM-scores.csv'])
            fs = os.listdir(os.path.join(save_dir,sample,area))
            fs = [x for x in fs if x!=cname]
            fs = [x for x in fs if 'cellMPI' not in x]
            clist.append(sample+'_'+area)
            for f in fs:
                cdf = pd.read_csv(os.path.join(save_dir,sample,area,f))
                compdf = pd.concat([compdf,cdf],ignore_index=True)
            compdf.to_csv(os.path.join(save_dir,sample,area,cname))
            [os.remove(os.path.join(save_dir,sample,area,x)) for x in fs]
            if args.sample=='all_samples' and args.area=='all_areas':
                master_df=pd.concat([master_df,compdf],ignore_index=True)
    if len(clist) > 1:
        master_df.to_csv(os.path.join(save_dir,'combined_SAM-scores.csv'))       

    
if __name__=="__main__":
    main()
