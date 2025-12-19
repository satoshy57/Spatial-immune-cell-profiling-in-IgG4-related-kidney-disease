import os
import argparse
import numpy as np
from tifffile import imread,imwrite
from skimage.filters import gaussian
from skimage.exposure import equalize_adapthist
import pickle as pkl
import multiprocessing as mp
from config import config

def dapi_correction(batch):
    for b in batch:
        imdir = b['ImageReadDir']
        segdir = b['SegReadDir']
        compwrite = b['CompSaveDir']
        samp = b['Sample']
        A = b['Area']
        imname = b['ImName']
        
        marker,_,_,tissue,_,_ = imname.split('_')
        img = imread(os.path.join(imdir,samp,A,imname))
        seg = imread(os.path.join(segdir,imname))
        
        ir,ic = np.shape(img)
        sr,sc = np.shape(seg)
        if (ir!=sr) or (ic!=sc):
            print('Composite and segmentation different sizes. Aborting ',imname)
            continue
        
        # sub out background
        imbkgd = np.where(seg==0,img,-1)
        imbkgd_vals=imbkgd.flatten()
        imbkdg_vals=imbkgd_vals[imbkgd_vals>=0]
        img2 = img - np.mean(imbkgd_vals)
        
        # bright spot fix
        gimg = gaussian(img,sigma=100)
        gbkgd = np.where(seg==0,gimg,-1)
        gbkgdvals = gbkgd.flatten()
        gbkgdvals = gbkgdvals[gbkgdvals>=0]
        gimg = gimg/np.mean(gbkgdvals)
        img3 = np.max(gimg)*img2/gimg
            
        # DAPI noise reduction and scaling
        img3 = np.where(img3<0,0,img3)
        img4 = gaussian(img3,sigma=1)
        i4bkgd = np.where(seg==0,img4,-1)
        i4bkgdvals = i4bkgd.flatten()
        i4bkgdvals = i4bkgdvals[i4bkgdvals>=0]
        img4 = img4-np.mean(i4bkgdvals)
        img4 = np.where(img4<0,0,img4)
        img4 = np.uint16(65535*img4/np.max(img4))

        # histogram truncation based on image size
        img_vals = img4.flatten()
        tiss_vals = np.where(seg>0,1,0)
        pospx = np.sum(tiss_vals)
        vals,cts = np.unique(img_vals,return_counts=True)
        ctmin = int(pospx*0.000005)
        fixed_vals = [x for i,x in enumerate(vals) if cts[i]>ctmin]
        M = np.max(fixed_vals)
        img4 = np.where(img4>M,M,img4)
        img5 = 255*(img4/M)
        img5 = np.where(img5>255,255,img5)
        imwrite(os.path.join(compwrite,imname),img5.astype(np.uint8))
        
            
def main():
    cfg = config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir',type=str,default=cfg.rootdir,help='')
    parser.add_argument('--dataset',type=str,default=cfg.dataset,help='')
    parser.add_argument('--comp_read',type=str,default='Normalized_composites',help='')
    parser.add_argument('--comp_write',type=str,default='Corrected_DAPI_composites',help='')
    parser.add_argument('--seg_read',type=str,default='tissue_composite_masks',help='')
    parser.add_argument('--sample',type=str,default=cfg.sample,help='')
    parser.add_argument('--area',type=str,default=cfg.area,help='')
    parser.add_argument('--dapi_cycle',type=str,default=cfg.DAPI_CYCLE,help='')
    parser.add_argument('--cpu_cap',type=int,default=mp.cpu_count(),help='')

    args,unparsed = parser.parse_known_args()

    im_readdir = os.path.join(args.rootdir,args.dataset,args.comp_read)
    comp_savedir = os.path.join(args.rootdir,args.dataset,args.comp_write)
    seg_readdir = os.path.join(args.rootdir,args.dataset,args.seg_read)

    if not os.path.exists(comp_savedir):
        os.makedirs(comp_savedir)

    dapi_ims = os.listdir(seg_readdir)
    dapi_ims.sort()
    imdicts = []
    for im in dapi_ims:
        print(im)
        if im.startswith('.'):
            continue
        marker,FL,cycle,tissue,sample,area = im.split('_')
        if cycle!=args.dapi_cycle:
            continue
        if args.sample!='all_samples':
            if sample not in args.sample:
                continue
        area = area.split('.')[0]
        if args.area!='all_areas':
            if area!=args.area:
                continue
        imdict = {'ImageReadDir':im_readdir,'SegReadDir':seg_readdir,'CompSaveDir':comp_savedir,'Sample':sample,'Area':area,'ImName':im}
        imdicts.append(imdict)
    num_workers = mp.cpu_count()
    if num_workers > args.cpu_cap:
        num_workers = args.cpu_cap

    if len(imdicts) < num_workers:
        num_workers = len(imdicts)

    batchsize = int(np.ceil(len(imdicts)/num_workers))
    batches = []
    for i in range(num_workers):
        ibatch = imdicts[i*batchsize:(i+1)*batchsize]
        batches.append(ibatch)
    
    pool = mp.Pool(num_workers)
    pool.map(dapi_correction,batches)
    pool.close()

if __name__=='__main__':
    main()
