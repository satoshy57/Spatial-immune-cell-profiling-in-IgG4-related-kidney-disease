
import os
import argparse
import numpy as np
from random import shuffle
from tifffile import imread
from scipy.ndimage import distance_transform_edt
from skimage.segmentation import watershed
import multiprocessing as mp
from config import config


def dilate_nuclei(batch):
  #  print()
    rdir = batch['ReadDir']
    sdir = batch['SaveDir']
    imgs = batch['Images']
    dil_f = batch['Dilation']
    tseg_path = batch['TSegPath']
    
   # raise
    print(imgs,flush=True)
    for img in imgs:
        if len(img)==1:
            print("A",flush=True)
            img = img[0]
            if os.path.exists(img.replace(rdir,sdir)):
                return
            try:
                cellmasks = np.load(img,allow_pickle=True).item()
                cellmasks = cellmasks['masks']
            except:
                cellmasks = np.load(img,allow_pickle=True)
                cellmasks = cellmasks['masks']
            binary_cell_masks = cellmasks > 0 # my edit 
            print("C",flush=True)
            dist_inv = distance_transform_edt(~binary_cell_masks) # TODO: make this more memory efficent 
            dil = (dist_inv < dil_f) 
            whole_cells = watershed(dist_inv,cellmasks,mask=dil,watershed_line=False)
            imdict = {}
            imdict['masks']=whole_cells
            imdict['ImName']=img
            imdict['NucMasks']=cellmasks
            
            print(img.replace(rdir,sdir),'saved',flush=True)
            np.save(img.replace(rdir,sdir),imdict)
            for key in  ['masks', 'NucMasks', 'ImName']:
                if key in imdict:
                    print(f"Key '{key}' is present.",flush=True)
                else:
                    print(f"Key '{key}' is missing.",flush=True)
            print("XXxxx",flush=True)
        else:
            print("B",flush=True)
            tsegs = imread(tseg_path)
            r,c = np.shape(tsegs)
            del tsegs
            
            M=0
            for im in img:
                print(im,flush=True)
                if os.path.exists(im.replace(rdir,sdir)):
                    continue
                try:
                    cellmasks = np.load(im,allow_pickle=True).item()
                    rmin,rmax,cmin,cmax = cellmasks['bbox']
                    cellmasks = cellmasks['masks']
                except:
                    cellmasks = np.load(im,allow_pickle=True)
                    rmin,rmax,cmin,cmax = cellmasks['bbox']
                    cellmasks = cellmasks['masks']
                binary_cell_masks = cellmasks > 0
                print("D",flush=True)

              #  binary_cell_masks = np.where(cellmasks>0,True,False)
                dist_inv = distance_transform_edt(~binary_cell_masks)
                dil = (dist_inv < dil_f) 
                whole_cells = watershed(dist_inv,cellmasks,mask=dil,watershed_line=False)
                whole_cells = np.where(whole_cells>0,whole_cells+M,0)
                cellmasks = np.where(cellmasks>0,cellmasks+M,0)
                M+=np.max(whole_cells,axis=None)
                imdict = {}
                imdict['masks']=whole_cells.astype(np.uint16)
                imdict['ImName']=im
                imdict['NucMasks']=cellmasks.astype(np.uint16)
               # np.save(im.replace(rdir,sdir),imdict)
                print(im,'saved',flush=True)
                print("XX",flush=True)


def main():
    cfg = config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir',type=str,default='/project/mclark/SCAMPI_datasets',help='')
    parser.add_argument('--dataset',type=str,default=cfg.dataset,help='')
    parser.add_argument('--nucleus_dir',type=str,default='nucleus_segmentations_WS',help='')
    parser.add_argument('--cell_dir',type=str,default='wholeCell_segmentations_WS',help='')
    parser.add_argument('--tseg_dir',type=str,default='tissue_composite_masks',help='')
    parser.add_argument('--sample',type=str,default=cfg.sample,help='')
    parser.add_argument('--area',type=str,default=cfg.area,help='')
    parser.add_argument('--pxsz',type=float,default=0.1507,help='in microns')
    parser.add_argument('--pad',type=int,default=1,help='in microns')
    parser.add_argument('--max_cpu',type=int,default=100,help='')

    args,unparsed = parser.parse_known_args()

    readdir = os.path.join(args.rootdir,args.dataset,args.nucleus_dir)
    writedir = os.path.join(args.rootdir,args.dataset,args.cell_dir)
    tsegdir = os.path.join(args.rootdir,args.dataset,args.tseg_dir)
    df = int(args.pad/args.pxsz)

    imList = []
    samples = os.listdir(readdir)
    tsegs = os.listdir(tsegdir)
    tsegpaths = []
    for sample in samples:
        if args.sample!='all_samples':
            if sample!=args.sample:
                continue
        if not os.path.exists(os.path.join(writedir,sample)):
            os.makedirs(os.path.join(writedir,sample))
        areas = os.listdir(os.path.join(readdir,sample))
        for area in areas:
            if args.area!='all_areas':
                if area!=args.area:
                    continue
            if not os.path.exists(os.path.join(writedir,sample,area)):
                os.makedirs(os.path.join(writedir,sample,area))
            ims = os.listdir(os.path.join(readdir,sample,area))
            ims = [os.path.join(readdir,sample,area,x) for x in ims]
            imList.append(ims)
            tsegpath = [os.path.join(tsegdir,x) for x in tsegs if (sample in x) and (area in x)] 
            tsegpaths.append(tsegpath[0])
   
    if len(imList) < args.max_cpu:
        num_workers = len(imList)
    else:
        num_workers = args.max_cpu

    batchsize = int(np.ceil(len(imList)/num_workers))
    batches = []
    for i in range(num_workers):
        batches.append({'Images':imList[i*batchsize:(i+1)*batchsize],'SaveDir':writedir,'ReadDir':readdir,'Dilation':df,'TSegPath':tsegpaths[i*batchsize:(i+1)*batchsize]})
    
    print(len(batches))
    print(batches,flush=True)
    pool = mp.Pool(num_workers)
    pool.map(dilate_nuclei,batches)
    pool.close()

if __name__=="__main__":
    main()
