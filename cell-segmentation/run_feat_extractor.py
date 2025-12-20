import os
import warnings
warnings.simplefilter(action='ignore')
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from tifffile import imread
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
import multiprocessing as mp
from skimage.measure import label, regionprops_table
from config import config

def calculate_cell_feats(batch):
    predDir = batch['predDir']
    imDir = batch['ImDir']
    sample = batch['Sample']
    area = batch['Area']
    saveDir = batch['saveDir']
    batchnum = batch['BatchNum']
    cellids = batch['CellIDs']
    
    if len(cellids) < 1:
        return
        
    newname = '_'.join([sample, area, 'cellMPI', str(batchnum) + '.csv'])
    if os.path.exists(os.path.join(saveDir, sample, area, newname)):
        print("Skip", flush=True)
        return

    # Initialize DataFrame with same structure as original
    data = {
        'Sample': [sample] * len(cellids),
        'Area': [area] * len(cellids),
        'CompName': [f"{sample}_{area}"] * len(cellids),
        'CellID': cellids
    }
    save_df = pd.DataFrame(data)

    # Load segmentation masks
    predName = os.listdir(os.path.join(predDir, sample, area))
    if len(predName) < 1 or not predName[0].endswith('.npy'):
        print('Cell segmentations not done for', sample, area, '. Continuing...')
        return

    try:
        cellsegs_f = np.load(os.path.join(predDir, sample, area, predName[0]), allow_pickle=True)
        if not isinstance(cellsegs_f, dict):
            cellsegs_f = cellsegs_f.item()
    except Exception as e:
        print(f"Error loading segmentation for {sample} {area}: {str(e)}")
        return

    # Load masks
    cellsegs = np.squeeze(cellsegs_f['masks'], axis=0)
    nucsegs = np.squeeze(cellsegs_f['NucMasks'], axis=0)
    memsegs = cellsegs - nucsegs

    # Get ROI bounds for this batch
    cellmask = np.where((cellsegs >= np.min(cellids)) & (cellsegs <= np.max(cellids)), 1, 0)
    cellidx = np.where(cellmask == 1)
    if len(cellidx[0]) == 0:
        return

    minR = np.min(cellidx[0])
    minC = np.min(cellidx[1])
    maxR = np.max(cellidx[0])
    maxC = np.max(cellidx[1])

    if (maxR-minR <= 0) or (maxC-minC <= 0):
        return

    # Crop masks to ROI
    cellsegs = cellsegs[minR:maxR, minC:maxC]
    nucsegs = nucsegs[minR:maxR, minC:maxC]
    memsegs = memsegs[minR:maxR, minC:maxC]

    # Calculate cell coordinates and shape features first - these are marker-independent
    cell_properties = []
    batch_size = 100  # Adjust based on available memory
    
    for i in range(0, len(cellids), batch_size):
        batch_cells = cellids[i:i + batch_size]
        
        for cell in batch_cells:
            cellidx2 = np.where(cellsegs == cell)
            if len(cellidx2[0]) == 0:
                cell_properties.append({
                    'area': 0,
                    'equivalent_diameter': 0,
                    'eccentricity': 0,
                    'centroid-0': 0,
                    'centroid-1': 0
                })
                continue

            rmin = np.min(cellidx2[0])
            rmax = np.max(cellidx2[0])
            cmin = np.min(cellidx2[1])
            cmax = np.max(cellidx2[1])

            cell_patch = cellsegs[rmin:rmax, cmin:cmax]
            cell_mask = (cell_patch == cell).astype(float)

            # Calculate shape features
            props = regionprops_table(
                label(cell_mask),
                properties=['area', 'equivalent_diameter', 'eccentricity', 'centroid']
            )
            
            # Store properties with adjusted coordinates
            cell_properties.append({
                'area': props['area'][0],
                'equivalent_diameter': props['equivalent_diameter'][0],
                'eccentricity': props['eccentricity'][0],
                'centroid-0': props['centroid-0'][0] + minR + rmin,
                'centroid-1': props['centroid-1'][0] + minC + cmin
            })

    # Add cell-specific properties to DataFrame
    save_df['GlobalMaskCentroidRow'] = [p['centroid-0'] for p in cell_properties]
    save_df['GlobalMaskCentroidCol'] = [p['centroid-1'] for p in cell_properties]
    save_df['WC_Area'] = [p['area'] for p in cell_properties]
    save_df['WC_Diameter'] = [p['equivalent_diameter'] for p in cell_properties]
    save_df['WC_eccentricity'] = [p['eccentricity'] for p in cell_properties]

    # Process images one at a time to save memory
    ims = os.listdir(os.path.join(imDir, sample, area))
    for im in ims:
        stain = im.split('_')[0]
        img = imread(os.path.join(imDir, sample, area, im))
        imgr, imgc = np.shape(img)
        if (imgr == 0) or (imgc == 0):
            continue
            
        img = img[minR:maxR, minC:maxC]
        
        mpi_c_vec = []
        mpi_n_vec = []
        mpi_m_vec = []

        # Process cells in smaller batches
        for i in range(0, len(cellids), batch_size):
            batch_cells = cellids[i:i + batch_size]
            
            for cell in batch_cells:
                cellidx2 = np.where(cellsegs == cell)
                if len(cellidx2[0]) == 0:
                    mpi_c_vec.append(0)
                    mpi_n_vec.append(0)
                    mpi_m_vec.append(0)
                    continue

                rmin = np.min(cellidx2[0])
                rmax = np.max(cellidx2[0])
                cmin = np.min(cellidx2[1])
                cmax = np.max(cellidx2[1])

                # Extract patches
                cell_patch = cellsegs[rmin:rmax, cmin:cmax]
                nuc_patch = nucsegs[rmin:rmax, cmin:cmax]
                mem_patch = memsegs[rmin:rmax, cmin:cmax]
                img_patch = img[rmin:rmax, cmin:cmax]

                # Create masks
                cell_mask = (cell_patch == cell).astype(float)
                nuc_mask = (nuc_patch == cell).astype(float)
                mem_mask = (mem_patch == cell).astype(float)

                # Calculate intensities
                try:
                    mpi_c = np.mean(img_patch * cell_mask)
                    mpi_n = np.mean(img_patch * nuc_mask)
                    mpi_m = np.mean(img_patch * mem_mask)
                except:
                    mpi_c = 0
                    mpi_n = 0
                    mpi_m = 0

                mpi_c_vec.append(mpi_c)
                mpi_n_vec.append(mpi_n)
                mpi_m_vec.append(mpi_m)

        # Add marker-specific measurements to DataFrame
        save_df[stain + '_Int-mean_WC'] = mpi_c_vec
        save_df[stain + '_Int-mean_Nuc'] = mpi_n_vec
        save_df[stain + '_Int-mean_Mem'] = mpi_m_vec

        # Clear memory
        del img, mpi_c_vec, mpi_n_vec, mpi_m_vec

    # Save results
    save_df.to_csv(os.path.join(saveDir, sample, area, newname))
    print("SAVE DIRECTORY FOR CSV, SUCCESSFUL: ", os.path.join(saveDir, sample, area, newname), flush=True)



def main():
    cfg = config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir',type=str,default=cfg.rootdir,help='')
    parser.add_argument('--dataset',type=str,default=cfg.dataset,help='')
    parser.add_argument('--pred_dir',type=str,default='wholeCell_segmentations_WS',help='')
    parser.add_argument('--im_dir',type=str,default='Normalized_composites',help='')
    parser.add_argument('--csv_dir',type=str,default='cell_MPI',help='')
    parser.add_argument('--sample',type=str,default=cfg.sample,help='')
    parser.add_argument('--area',type=str,default=cfg.area,help='')
    parser.add_argument('--patch_pad',type=float,default=1,help='in microns')
    parser.add_argument('--pxsize',type=float,default=0.1507,help='in microns')
    parser.add_argument('--cpu_cap',type=int,default=200,help='')

    args,unparsed = parser.parse_known_args()

    PredDir = os.path.join(args.rootdir,args.dataset,args.pred_dir)
    csvDir = os.path.join(args.rootdir,args.dataset,args.csv_dir)
    imDir = os.path.join(args.rootdir,args.dataset,args.im_dir)

    if not os.path.exists(csvDir):
        os.makedirs(csvDir)
    
    samples = os.listdir(PredDir)
    for sample in samples:
        if sample == 'S17-17393A2':
            continue
        if not args.sample=='all_samples':
            if args.sample!=sample:
                continue
        if not os.path.exists(os.path.join(csvDir,sample)):
            os.makedirs(os.path.join(csvDir,sample))
        areas = os.listdir(os.path.join(PredDir,sample))
        for area in areas:
            if not args.area=='all_areas':
                if args.area!=area:
                    continue
            if not os.path.exists(os.path.join(csvDir,sample,area)):
                os.makedirs(os.path.join(csvDir,sample,area))
            master_name = sample+'_'+area+'_cellMPI.csv'
            if os.path.exists(os.path.join(csvDir,sample,area,master_name)):
                continue
            cellbatches=[]
            segs = os.listdir(os.path.join(PredDir,sample,area))
            if len(segs)==0:
                continue
            try:
                cellsegs = np.load(os.path.join(PredDir,sample,area,segs[0]),allow_pickle=True).item()
            except:
                cellsegs = np.load(os.path.join(PredDir,sample,area,segs[0]),allow_pickle=True)
            cells = np.unique(cellsegs['masks'])
            del cellsegs
            cells = np.delete(cells,0)
            batchsize = int(np.ceil(len(cells)/args.cpu_cap))
            for i in range(args.cpu_cap):
                cellBatch = cells[i*batchsize:(i+1)*batchsize]
                batchdict = {'predDir':PredDir,'ImDir':imDir,'Sample':sample,
                             'saveDir':csvDir,'CellIDs':cellBatch,'Area':area,
                             'BatchNum':i}
                cellbatches.append(batchdict)
                
            num_workers = args.cpu_cap   
            pool = mp.Pool(num_workers)
            pool.map(calculate_cell_feats,cellbatches)
            pool.close()
    
            csvs = os.listdir(os.path.join(csvDir,sample,area))
            csvs = [x for x in csvs if 'SAM-scores' not in x]
            master_name = sample+'_'+area+'_cellMPI.csv'
            master_df = pd.DataFrame()
            csvs = [x for x in csvs if x!=master_name]
            for csv in csvs:
                df = pd.read_csv(os.path.join(csvDir,sample,area,csv))
                master_df = pd.concat([master_df,df],ignore_index=True)
            keepcols = [x for x in master_df.columns if 'Unnamed' not in x]
            master_df = master_df[keepcols]
            master_df.to_csv(os.path.join(csvDir,sample,area,master_name))
            print(master_name,'saved')
            [os.remove(os.path.join(csvDir,sample,area,x)) for x in csvs if x!=master_name]

    if (args.sample=='all_samples') and (args.area=='all_areas'):
        combined_master_df = pd.DataFrame()
        for sample in samples:
            areas = os.listdir(os.path.join(csvDir,sample))
            for area in areas:
                csvs = os.listdir(os.path.join(csvDir,sample,area))
                csvs = [x for x in csvs if x.endswith('cellMPI.csv')]
                if len(csvs)>1:
                    print('Too many cell csvs in',sample,area,'. Check files.')
                    continue
                elif len(csvs)==0:
                    print('Skipping sample',sample,area)
                    continue
                df = pd.read_csv(os.path.join(csvDir,sample,area,csvs[0]))
                combined_master_df = pd.concat([combined_master_df,df],ignore_index=True)
        combined_master_df.to_csv(os.path.join(csvDir,'combined_cellMPI.csv'))

if __name__ == '__main__':
    main()
