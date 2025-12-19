import os
import argparse
import numpy as np
from shutil import copyfile
from tifffile import imread,imwrite
import multiprocessing as mp
import pandas as pd
from skimage.filters import threshold_otsu,threshold_multiotsu,gaussian
import matplotlib.pyplot as plt
from config import config

def process_channels(batch):
    for channel in batch:
        rdir = channel['ReadDir']
        sdir = channel['WriteDir']
        cdir = channel['CSVDir']
        channel_im = channel['MarkerName']
        bkgd_im = channel['BackgroundName']
        N = channel['Norm']
        QC = channel['QC']
        dcycle = channel['DAPI_cycle']
        t_mask = channel['TissueMask']
        
        marker,FL,cycle,tissue,sample,area = channel_im.split('_')
        area = area.split('.')[0] 
        if QC==True:
            df = pd.DataFrame()
            chdict = dict()
            chdict['ImName']=channel_im
            chdict['Sample']=sample
            chdict['Marker']=channel_im.split('_')[0]
            chdict['Wavelength']=channel_im.split('_')[1]
        if not os.path.exists(os.path.join(sdir,sample)):
            try:
                os.makedirs(os.path.join(sdir,sample))
            except:
                print('Folder generated asynchronously. Continuing...')
        if not os.path.exists(os.path.join(sdir,sample,area)):
            try:
                os.makedirs(os.path.join(sdir,sample,area))
            except:
                print('Folder generated asynchronously. Continuing...')
        
        if os.path.exists(os.path.join(sdir,sample,area,channel_im)):
            continue
        
        # if dapi, just copy to dest
        if len(bkgd_im)==0:
            copyfile(os.path.join(rdir,sample,area,channel_im),os.path.join(sdir,sample,area,channel_im))
        else:
            img = imread(os.path.join(rdir,sample,area,channel_im))
            t_img = imread(t_mask)
            r,c = np.shape(img)
            try:
                bkgd_img = imread(os.path.join(rdir,sample,area,bkgd_im))
            except:
                print('issue with background image ',bkgd_im)
                continue
     
            img = img.astype(float)
            bkgd_img = bkgd_img.astype(float)
            new_ch = img-bkgd_img
            new_ch2 = np.where(new_ch<0,0,new_ch)
            del bkgd_img

            img_vals = new_ch2[t_img>0]
            img_vals.sort()
            m_idx = int(0.75*len(img_vals))
            M_idx = int(0.99*len(img_vals))
            m = img_vals[m_idx]
            M = img_vals[M_idx]
             
            if M < (10*m):
                M=(10*m)
            
            scaled_ch = 255*(new_ch2-m)/(M-m)
            scaled_ch = np.clip(scaled_ch,0,255)
            scaled_ch = gaussian(scaled_ch,sigma=1)
            scaled_ch = np.where(t_img>0,scaled_ch,0)
            
            imwrite(os.path.join(sdir,sample,area,channel_im),scaled_ch.astype(np.uint8))
            print(channel_im,'saved')  
            
            if QC==True:
                svals = np.unique(new_ch2)
                if len(svals)>1:
                    th = threshold_otsu(new_ch2)
                else:
                    th = np.min(svals)
                del svals
                
                tissue_mask = imread(t_mask)
                tissue_ind = np.where(tissue_mask!=0)
                bkgd_ind = np.where(tissue_mask==0)
                signal_ind = np.where(new_ch2 > th)
                non_signal_ind = np.where(new_ch2 <= th)
                tissue_only_ind = np.where((tissue_mask!=0) & (new_ch2<=th))

                raw_tissue_vals = img[tissue_ind]
                sub_tissue_vals = new_ch2[tissue_ind]
                
                raw_tissueonly_vals = img[tissue_only_ind]
                sub_tissueonly_vals = new_ch2[tissue_only_ind]
                
                raw_bkgd_vals = img[bkgd_ind]
                sub_bkgd_vals = new_ch2[bkgd_ind]
                
                raw_signal_vals = img[signal_ind]
                sub_signal_vals = new_ch2[signal_ind]

                raw_nonsignal_vals = img[non_signal_ind]
                sub_nonsignal_vals = new_ch2[non_signal_ind]

                chdict['RawMin']=[np.min(img)]
                chdict['RawMax']=[np.max(img)]
                chdict['RawMean']=[np.mean(img)]
                chdict['RawMedian']=[np.median(img)]
                chdict['RawStd']=[np.std(img)]
                chdict['RawVar']=[np.var(img)]
                chdict['RawStainBR']=[np.mean(raw_signal_vals)/np.mean(raw_bkgd_vals)]
                chdict['RawStainTissueRatio']=[np.mean(raw_signal_vals)/np.mean(raw_tissue_vals)]
                chdict['RawStainNonStainRatio']=[np.mean(raw_signal_vals)/np.mean(raw_nonsignal_vals)]
                chdict['RawTissueBR']=[np.mean(raw_tissue_vals)/np.mean(raw_bkgd_vals)]
                chdict['RawTissueOnlyBR']=[np.mean(raw_tissueonly_vals)/np.mean(raw_bkgd_vals)]
                chdict['RawSNR']=[np.mean(raw_signal_vals)/np.std(raw_bkgd_vals)]
                chdict['RawSNR2']=[np.mean(raw_signal_vals)/np.std(raw_nonsignal_vals)]
                chdict['SubMin']=[np.min(new_ch)]
                chdict['SubMax']=[np.max(new_ch)]
                chdict['SubMean']=[np.mean(new_ch)]
                chdict['SubMedian']=[np.median(new_ch)]
                chdict['SubStd']=[np.std(new_ch)]
                chdict['SubVar']=[np.var(new_ch)]
                chdict['SubStainBR']=[np.mean(sub_signal_vals)/np.mean(sub_bkgd_vals)]
                chdict['SubStainTissueRatio']=[np.mean(sub_signal_vals)/np.mean(sub_tissue_vals)]
                chdict['SubStainNonStainRatio']=[np.mean(sub_signal_vals)/np.mean(sub_nonsignal_vals)]
                chdict['SubTissueBR']=[np.mean(sub_tissue_vals)/np.mean(sub_bkgd_vals)]
                chdict['SubTissueOnlyBR']=[np.mean(sub_tissueonly_vals)/np.mean(sub_bkgd_vals)]
                chdict['SubSNR']=[np.mean(sub_signal_vals)/np.std(sub_bkgd_vals)]
                chdict['SubSNR2']=[np.mean(sub_signal_vals)/np.std(sub_nonsignal_vals)]

                if N==True:
                    norm_tissue_vals = scaled_ch[tissue_ind]
                    norm_tissueonly_vals = scaled_ch[tissue_only_ind]
                    norm_bkgd_vals = scaled_ch[bkgd_ind]
                    norm_signal_vals = scaled_ch[signal_ind]
                    norm_nonsignal_vals = scaled_ch[non_signal_ind]
                    chdict['NormMin']=[np.min(scaled_ch)]
                    chdict['NormMax']=[np.max(scaled_ch)]
                    chdict['NormMean']=[np.mean(scaled_ch)]
                    chdict['NormMedian']=[np.median(scaled_ch)]
                    chdict['NormStd']=[np.std(scaled_ch)]
                    chdict['NormVar']=[np.var(scaled_ch)]
                    chdict['NormStainBR']=[np.mean(norm_signal_vals)/np.mean(norm_bkgd_vals)]
                    chdict['NormStainTissueRatio']=[np.mean(norm_signal_vals)/np.mean(norm_tissue_vals)]
                    chdict['NormStainNonStainRatio']=[np.mean(norm_signal_vals)/np.mean(norm_nonsignal_vals)]
                    chdict['NormTissueBR']=[np.mean(norm_tissue_vals)/np.mean(norm_bkgd_vals)]
                    chdict['NormTissueOnlyBR']=[np.mean(norm_tissueonly_vals)/np.mean(norm_bkgd_vals)]
                    chdict['NormSNR']=[np.mean(norm_signal_vals)/np.std(norm_bkgd_vals)]
                    chdict['NormSNR2']=[np.mean(norm_signal_vals)/np.std(norm_nonsignal_vals)]

                cdf = pd.DataFrame.from_dict(chdict)
                df = pd.concat([df,cdf],ignore_index=True)
                csvname = channel_im.replace('.tif','.csv')
                df.to_csv(os.path.join(cdir,csvname))

def main():
    cfg = config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir',type=str,default=cfg.rootdir,help='')
    parser.add_argument('--dataset',type=str,default=cfg.dataset,help='')
    parser.add_argument('--read',type=str,default='Ashlar/Output_channels',help='')
    parser.add_argument('--tissue_mask',type=str,default='tissue_composite_masks',help='')
    parser.add_argument('--write',type=str,default='Normalized_composites',help='')
    parser.add_argument('--sample',type=str,default=cfg.sample,help='')
    parser.add_argument('--area',type=str,default=cfg.area,help='')
    parser.add_argument('--csvs',type=str,default='Composite_specs',help='')
    parser.add_argument('--taf_cycle',type=str,default=cfg.TAF_CYCLE,help='')
    parser.add_argument('--dapi_cycle',type=str,default=cfg.DAPI_CYCLE,help='')
    parser.add_argument('--num_cpus',type=int,default=50,help='')
    parser.add_argument('--normalize',type=bool,default=cfg.NORM,help='')
    parser.add_argument('--keep_QC',type=bool,default=cfg.KEEP_QC,help='')

    args,unparsed = parser.parse_known_args()

    readdir = os.path.join(args.rootdir,args.dataset,args.read)
    maskdir = os.path.join(args.rootdir,args.dataset,args.tissue_mask)
    savedir = os.path.join(args.rootdir,args.dataset,args.write)
    csvdir = os.path.join(args.rootdir,args.dataset,args.csvs)

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    if args.normalize==True:
        N=True
    else:
        N=False

    if args.keep_QC==True:
        QC=True
    else:
        QC=False
    if QC==True:
        if not os.path.exists(csvdir):
            os.makedirs(csvdir)
        if not os.path.exists(maskdir):
            print('Tissue mask not computed yet. Exiting...')
            return
    samples = os.listdir(readdir)
    channel_ims = []
    for sample in samples:
        if args.sample!='all_samples':
            if sample not in args.sample:
                continue
        areas = os.listdir(os.path.join(readdir,sample))
        for area in areas:
            if args.area!='all_areas':
                if area!=args.area:
                    continue
            ims = os.listdir(os.path.join(readdir,sample,area))
            channel_ims.extend(ims)
    channel_ims.sort()
    channel_ims = [x for x in channel_ims if x.endswith('.tif')]
    marker_ims = [x for x in channel_ims if 'blank' not in x]
    marker_ims = [x for x in channel_ims if 'TAF' not in x]
    bkgd_ims = [x for x in channel_ims if ('TAF' in x) or ('blank' in x)]
    tmasks = os.listdir(maskdir)
    subdicts = []
    for marker_im in marker_ims:
        marker,FL,cycle,tissue,sample,area = marker_im.split('_')
        if marker=='blank':
            continue
        if args.sample!='all_samples':
            if sample not in args.sample:
                continue

        area = area.split('.')[0]
        if args.area!='all_areas':
            if area!=args.area:
                continue
        
        if os.path.exists(os.path.join(savedir,sample,area,marker_im)):
            continue
        
        bkgd = [x for x in bkgd_ims if sample in x]
        bkgd = [x for x in bkgd if area in x]
        bkgd = [x for x in bkgd if FL in x]
        bkgd = [x for x in bkgd if '_'+args.taf_cycle+'_' in x]
        tmask = [x for x in tmasks if sample in x]
        tmask = [x for x in tmask if area in x]
        if len(tmask)==0:
            print('No tissue mask for:',sample,area,'. Continuing...')
            continue
        if len(bkgd)==1:
            subdict = {'ReadDir':readdir,'WriteDir':savedir,'CSVDir':csvdir,'MarkerName':marker_im,'BackgroundName':bkgd[0],'Norm':N,'QC':QC,'DAPI_cycle':args.dapi_cycle,'TissueMask':os.path.join(maskdir,tmask[0])}
        elif len(bkgd)==0:
            subdict = {'ReadDir':readdir,'WriteDir':savedir,'CSVDir':csvdir,'MarkerName':marker_im,'BackgroundName':[],'Norm':N,'QC':QC,'DAPI_cycle':args.dapi_cycle,'TissueMask':os.path.join(maskdir,tmask[0])}
        else:
            print('Error with background selection. Check channel image names')
        subdicts.append(subdict)
    num_workers = args.num_cpus

    print(len(subdicts))
    if len(subdicts)<num_workers:
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
    pool.map(process_channels,batches)
    pool.close()
    
    if QC==True:
        dfs = os.listdir(csvdir)
        ddf = pd.DataFrame()
        for df in dfs:
            sdf = pd.read_csv(os.path.join(csvdir,df))
            ddf = pd.concat([ddf,sdf],ignore_index=True)
        ddf.to_csv(os.path.join(csvdir,args.dataset+'_specs.csv'))
        for df in dfs:
            os.remove(os.path.join(csvdir,df))
    
if __name__=='__main__':
    main()
