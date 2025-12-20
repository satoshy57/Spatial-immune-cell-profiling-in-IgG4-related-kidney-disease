import os
import argparse
import numpy as np
from tifffile import imread,imwrite
import multiprocessing as mp
from config import config

def rename_blanks(batch):
    for b in batch:
        rdir = b['ReadDir']
        sdir = b['SaveDir']
        sample = b['Sample']
        area = b['Area']

        ims = os.listdir(os.path.join(rdir,sample,area))
        blank_ims = [x for x in ims if 'blank' in x]

        for blank_im in blank_ims:
            chan = blank_im.split('_')[1][-3:]
            newname = blank_im.replace('blank','TAF'+chan)
            img = imread(os.path.join(rdir,sample,area,blank_im))
            img = (255/65535)*img
            img = np.uint8(img)
            imwrite(os.path.join(sdir,sample,area,newname),img.astype(np.uint8))

def main():
    cfg = config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir',type=str,default=cfg.rootdir,help='')
    parser.add_argument('--dataset',type=str,default=cfg.dataset,help='')
    parser.add_argument('--read',type=str,default='Ashlar/Output_channels',help='')
    parser.add_argument('--write',type=str,default='Normalized_composites',help='')
    parser.add_argument('--sample',type=str,default=cfg.sample,help='')
    parser.add_argument('--area',type=str,default=cfg.area,help='')
    parser.add_argument('--num_cpus',type=int,default=mp.cpu_count(),help='')

    args,unparsed = parser.parse_known_args()

    readdir = os.path.join(args.rootdir,args.dataset,args.read)
    savedir = os.path.join(args.rootdir,args.dataset,args.write)

    samples = os.listdir(readdir)
    subdicts = []
    for sample in samples:
        if args.sample!='all_samples':
            if sample not in args.sample:
                continue
        areas = os.listdir(os.path.join(readdir,sample))
        for area in areas:
            if args.area!='all_areas':
                if area!=args.area:
                    continue
            subdicts.append({'ReadDir':readdir,'SaveDir':savedir,'Sample':sample,'Area':area})

    if len(subdicts) > args.num_cpus:
        num_workers = args.num_cpus
    else:
        num_workers = len(subdicts)

    batchsize = int(np.ceil(len(subdicts)/num_workers))
    batches=[]
    for i in range(num_workers):
        sbatch = subdicts[i*batchsize:(i+1)*batchsize]
        batches.append(sbatch)

    batches = [x for x in batches if len(x) > 0]
    if len(batches) < num_workers:
        num_workers = len(batches)

    pool = mp.Pool(num_workers)
    pool.map(rename_blanks,batches)
    pool.close()
    
    
if __name__=='__main__':
    main()
