# read in DIC and cycle 0 for each location
# drop in DIC to channel 0
# move DIC to Ashlar/DIC_tif during development (delete later)

# fix S20 samples that had DIC directory issue

import os
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
from tifffile import imread,imwrite
from config import config

def add_chans(batch):
    sareas = batch['Images']
    csvdir = batch['CSVDir']
    DIC = batch['DIC']
    addDAPI = batch['DAPI']
    
    csvfiles = os.listdir(csvdir)
    csvfiles = [x for x in csvfiles if x.endswith('_markers.csv')]

    if len(csvfiles) > 1:
        print('Multiple csv files detected. Please check files')
        return

    markerdf = pd.read_csv(os.path.join(csvdir,csvfiles[0]))
    cols = markerdf.columns
    if 'marker' not in cols:
        print('Edit column headers in marker csv to include "marker"')
        return
    markers = list(markerdf['marker'])
    if DIC==True:
        if 'empty' not in markers:
            print('No location to place DIC in stack')
            print('If empty location exists, please edit csv to include "empty" in that location.')
            return

        dicspots = markerdf.index[markerdf['marker']=='empty']
        dicloc = dicspots[0]
        dic_place_cycle = markerdf.loc[dicloc]['cycle']
        dic_place_channel = markerdf.loc[dicloc]['channel']-(dic_place_cycle*5)
        dic_cycle = str(dic_place_cycle) 
        markerdf.at[dicloc,'marker']='DIC'
        try:
            markerdf.to_csv(os.path.join(csvdir,csvfiles[0]))
        except:
            print('********')
            print('PERMISSIONS ERROR WITH SAVING EDITED CSV')
            print('********')
    
    for sarea in sareas:
        print('***---',sarea,'---***')
        fields = os.listdir(sarea)
        fields = [x for x in fields if not x.endswith('.xml')]
        for field in fields:
            ims = os.listdir(os.path.join(sarea,field))
            if DIC==True:
                dic = [x for x in ims if 'DIC' in x]
                if len(dic)==0:
                    print('No DIC for',sarea)
                    continue
            if addDAPI==True:
                dapi = [x for x in ims if 'DAPI' in x]
                # Place high power dapi in first marker- hard coded for now
                c1 = [x for x in ims if '_1_' in x]
                for cim in c1:
                    try: 
                        Dimg = imread(os.path.join(sarea,field,cim))
                        if len(np.shape(Dimg))<3:
                            print('Error with image size in ',sarea,field,cim)
                            continue
                        Fd = cim.split('_')[-1]
                        da = [x for x in dapi if Fd in x]
                        if len(da)==1:
                            Da = da[0]
                        elif len(da)==0:
                            print(Fd,x)
                            continue
                        else:
                            print('Found multiple DAPI protocols for',sarea)
                            continue
                        try:
                            daimg = imread(os.path.join(sarea,field,Da))
                            Dimg[-1,:,:]=daimg
                            imwrite(os.path.join(sarea,field,cim),Dimg)
                            os.remove(os.path.join(sarea,field,Da))
                        except:
                            print('Error with loading',Da)
                    except:
                        print('Error with loading',cim)        
                for dapiim in dapi:
                    if os.path.exists(os.path.join(sarea,field,dapiim)):
                        os.remove(os.path.join(sarea,field,dapiim))
            
            if DIC==True:        
                c0 = [x for x in ims if '_'+dic_cycle+'_' in x]
                for c in c0:
                    try:
                        img = imread(os.path.join(sarea,field,c))
                        if len(np.shape(img))<3:
                            print('Error with image size in ',sarea,field,c)
                            continue
                        F = c.split('_')[-1]
                        d = [x for x in dic if F in x]
                        if len(d)==1:
                            D = d[0]
                        elif len(d)==0:
                            print(F)
                            continue
                        else:
                            D = d[0]
                            print('Found multiple DIC for',sarea)
                            continue
                        try:
                            dimg = imread(os.path.join(sarea,field,D))
                            img[dic_place_channel,:,:]=dimg
                            imwrite(os.path.join(sarea,field,c),img)
                            os.remove(os.path.join(sarea,field,D))
                        except:
                            print('Error with loading',D)
                    except:
                        print('Error with loading',c) 
                for dicim in dic:
                    if os.path.exists(os.path.join(sarea,field,dicim)):
                        os.remove(os.path.join(sarea,field,dicim))
            

def main():
    cfg = config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir',type=str,default=cfg.rootdir,help='')
    parser.add_argument('--dataset',type=str,default=cfg.dataset,help='')
    parser.add_argument('--read_dir',type=str,default='Ashlar/Input',help='')
    parser.add_argument('--csv_dir',type=str,default='raw_data_from_microscope',help='')
    parser.add_argument('--sample',type=str,default=cfg.sample,help='')
    parser.add_argument('--area',type=str,default=cfg.area,help='')
    parser.add_argument('--add_DIC',type=bool,default=cfg.ADD_DIC,help='')
    parser.add_argument('--add_DAPI',type=bool,default=cfg.ADD_DAPI,help='')
    parser.add_argument('--cpu_cap',type=int,default=100,help='')
    
    args,unparsed=parser.parse_known_args()

    readdir = os.path.join(args.rootdir,args.dataset,args.read_dir)
    csvdir = os.path.join(args.rootdir,args.dataset,args.csv_dir)

    if args.add_DIC==True:
        dic=True
    else:
        dic=False

    if args.add_DAPI==True:
        addDAPI=True
    else:
        addDAPI=False

    if (dic==False) and (addDAPI)==False:
        return
    
    samples = os.listdir(readdir)
    if args.sample!='all_samples':
        samples = [x for x in samples if args.sample in x]
    sampleareas = []
    for sample in samples:
        areas = os.listdir(os.path.join(readdir,sample))
        if args.area!='all_areas':
            areas = [x for x in areas if args.area in x]
        sa = [os.path.join(readdir,sample,x) for x in areas]
        sampleareas.extend(sa)
    
    num_workers = mp.cpu_count()
    if num_workers > args.cpu_cap:
        num_workers = args.cpu_cap

    if len(sampleareas) < num_workers:
        num_workers = len(sampleareas)

    batchsize = int(np.ceil(len(sampleareas)/num_workers))
    batches=[]
    for i in range(num_workers):
        batch = sampleareas[i*batchsize:(i+1)*batchsize]
        batches.append({'Images':batch,'CSVDir':csvdir,'DIC':dic,'DAPI':addDAPI})

    pool = mp.Pool(num_workers)
    pool.map(add_chans,batches)
    pool.close()

if __name__=="__main__":
    main()

