import os
import argparse
import numpy as np
import pandas as pd
from config import config

def main():
    cfg = config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir',type=str,default=cfg.rootdir,help='SCAMPI data directory')
    parser.add_argument('--dataset',type=str,default=cfg.dataset,help='current dataset')
    parser.add_argument('--dset',type=str,default=cfg.dset,help='current dataset')
    parser.add_argument('--read_dir',type=str,default='Normalized_composites',help='tiff files for input into ashlar')
    parser.add_argument('--csv_dir',type=str,default='raw_data_from_microscope',help='')
    parser.add_argument('--sample',type=str,default=cfg.sample,help='tiff files for input into ashlar')
    parser.add_argument('--area',type=str,default=cfg.area,help='tiff files for input into ashlar')
    parser.add_argument('--channel_comp_dir',type=str,default='Ashlar/Output_channels/',help='directory to save aligned channel composites')

    args,unparsed = parser.parse_known_args()
    readdir = os.path.join(args.rootdir,args.dataset,args.read_dir)
    chandir = os.path.join(args.rootdir,args.dataset,args.channel_comp_dir)
    csvdir = os.path.join(args.rootdir,args.dataset,args.csv_dir)
    
    marker_df = pd.read_csv(os.path.join(csvdir,args.dset+'_markers.csv'))
    markers = marker_df['marker']
    markers = list(np.unique(markers))
    markers = [x for x in markers if x!='empty']
    markers = [x for x in markers if x!='blank']

    chan_df = marker_df[marker_df['marker']!='empty']
    chan_df2 = chan_df[chan_df['marker']!='blank']
    chan_df3 = chan_df2[chan_df2['marker']!='DAPI']
    chans,_ = chan_df3.shape
    chans+=1 #add in one dapi
    samples = os.listdir(readdir)

    with open(os.path.join(args.rootdir,args.dataset,'composite_QC.txt'),'w') as f:
        f.write('***Checking Normalized_composites folder****\n')
        f.write('\n')
        print('***Checking Normalized_composites folder****')
        print('')
        for sample in samples:
            if args.sample!='all_samples':
                if sample!=args.sample:
                    continue
            areas = os.listdir(os.path.join(readdir,sample))
            for area in areas:
                if args.area!='all_areas':
                    if area!=args.area:
                        continue
                ims = os.listdir(os.path.join(readdir,sample,area))
                if len(ims) < chans:
                    print('Channels missing from',sample,area,'. Checking which ones...')
                    f.write('Channels missing from '+sample+'_'+area+'. Checking which ones...')
                    chmarkers = [x.split('_')[0] for x in ims]
                    missingMarkers = list(set(markers)-set(chmarkers))
                    print('Markers missing are: ')
                    f.write('Markers missing are: \n')
                    [print(x) for x in missingMarkers]
                    [f.write(x+'\n') for x in missingMarkers]
                    f.write('\n')
    
        f.write('***Checking Ashlar/Output_channels folder****\n')
        f.write('\n')
        print('***Checking Ashlar/Output_channels folder****')
        print('')
        for sample in samples:
            if args.sample!='all_samples':
                if sample!=args.sample:
                    continue
            for area in areas:
                if args.area!='all_areas':
                    if area!=args.area:
                        continue
                aims = os.listdir(os.path.join(chandir,sample,area))
                blanks = [x for x in aims if 'blank' in x]
                print(len(blanks),'TAF images saved.')
                f.write(str(len(blanks))+' TAF images saved.\n\n')
                if len(aims) < chans:
                    print('Channels missing from',sample,area,'. Checking which ones...')
                    f.write('Channels missing from '+sample+'_'+area+'. Checking which ones...')
                    chmarkers = [x.split('_')[0] for x in aims]
                    missingMarkers = list(set(markers)-set(chmarkers))
                    print('Markers missing are: ')
                    f.write('Markers missing are: \n')
                    [print(x) for x in missingMarkers]
                    [f.write(x+'\n') for x in missingMarkers]
                f.write('\n')
        print('***Composite QC done. Address any issues by running individual sample+area through this shell script.')
        f.write('***Composite QC done. Address any issues by running individual sample+area through this shell script.')
        f.close()


if __name__=="__main__":
    main()
