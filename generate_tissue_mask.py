import os
import argparse
import numpy as np
from tifffile import imread,imwrite
from scipy.ndimage import gaussian_filter,binary_fill_holes
import multiprocessing as mp
from random import shuffle
from config import config

def segment_tissue(batch):
    sdir = batch['WriteDir']
    samples = batch['Images']
    for sample in samples:
        print('***--',sample,'---***')
        img = imread(sample)
        r,c = np.shape(img)
        sname = sample.split('/')[-1]
        img = img-np.mean(img)
        img = 1/(1+np.exp(-img))
        gf_img = gaussian_filter(img,sigma=75)
        M = np.max(gf_img)
        th_img = np.where(gf_img > 0.25*M,1,0)
        th_img = binary_fill_holes(th_img).astype(int)
        th_img = th_img*255
        tr,tc = np.shape(th_img)
        th_img = np.pad(th_img,((0,r-tr),(0,c-tc)),mode='constant')
        imwrite(os.path.join(sdir,sname),th_img.astype(np.uint8))

def main():
    cfg = config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir',type=str,default=cfg.rootdir,help='')
    parser.add_argument('--dataset',type=str,default=cfg.dataset,help='')
    parser.add_argument('--read',type=str,default='Ashlar/Output_channels',help='')
    parser.add_argument('--write',type=str,default='tissue_composite_masks',help='')
    parser.add_argument('--sample',type=str,default=cfg.sample,help='')
    parser.add_argument('--area',type=str,default=cfg.area,help='')
    parser.add_argument('--dapi_cycle',type=str,default=cfg.DAPI_CYCLE,help='')
    parser.add_argument('--cpu_cap',type=int,default=mp.cpu_count(),help='')

    args,unparsed = parser.parse_known_args()

    readdir = os.path.join(args.rootdir,args.dataset,args.read)
    savedir = os.path.join(args.rootdir,args.dataset,args.write)

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    samples = os.listdir(readdir)
    dapi_comps=[]
    for sample in samples:
        if args.sample!='all_samples':
            if sample!=args.sample:
                continue
        areas = os.listdir(os.path.join(readdir,sample))
        for area in areas:
            if args.area!='all_areas':
                if args.area!=area:
                    continue
            ims = os.listdir(os.path.join(readdir,sample,area))
            dims = [x for x in ims if 'DAPI' in x]
            dims = [x for x in dims if x.split('_')[2]==args.dapi_cycle]
            dimpaths = [os.path.join(readdir,sample,area,x) for x in dims]
            dapi_comps.extend(dimpaths)
        
    num_workers = mp.cpu_count()
    if num_workers > args.cpu_cap:
        num_workers = args.cpu_cap
    if len(dapi_comps) < num_workers:
        num_workers = len(dapi_comps)
    batchsize = int(np.ceil(len(dapi_comps)/num_workers))
    batches = []
    for i in range(num_workers):
        dbatch = dapi_comps[i*batchsize:(i+1)*batchsize]
        batches.append({'WriteDir':savedir,'Images':dbatch})
    batches = [x for x in batches if len(x) > 0]

    pool = mp.Pool(num_workers)
    pool.map(segment_tissue,batches)
    pool.close()


if __name__=='__main__':
    main()
