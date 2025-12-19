import os
import subprocess
import shlex
import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
import xml.etree.ElementTree as ET
from tifffile import imread,imwrite
from config import config

class Marker:
    def __init__(self,chNum,cycNum,target,color):
        self.channel = chNum
        self.cycle = cycNum
        self.target = target
        self.color = color

def break_stack_into_chans(stackdir,sname,chandir,readdir,dcycle):
    dset,sample,area = sname.split('_')
    area = area.split('.')[0]
    if not os.path.exists(os.path.join(chandir,sample)):
        os.makedirs(os.path.join(chandir,sample))

    if not os.path.exists(os.path.join(chandir,sample,area)):
        os.makedirs(os.path.join(chandir,sample,area))

    mdir = readdir.split('/Ashlar/')[0]+'/raw_data_from_microscope'
    print(mdir)
    mfiles = os.listdir(mdir)
    mfiles = [x for x in mfiles if x.endswith('.csv')]
    for mfile in mfiles:
        if 'DIC' in mfile:
            mfiles = [mfile]

    if len(mfiles)>1:
        mfiles.sort()
        print('Multiple csv files found for markers. Using',mfiles[0])

    mdf = pd.read_csv(os.path.join(mdir,mfiles[0]))
    # parse csv
    chans = list(mdf['channel'])
    cycles = list(mdf['cycle'])
    targets = list(mdf['marker'])
    colors = list(mdf['color'])

    check_targets = [x for x in targets if x!='empty']
    check_targets = [x for x in check_targets if x!='DAPI']
    check_targets.append('DAPI')

    saved_channels = os.listdir(os.path.join(chandir,sample,area))
    saved_channels = [x.split('_')[0] for x in saved_channels]
    if len(saved_channels)==len(check_targets):
        return
    
    # make list of marker objects
    all_markers = [Marker(chNum=x,cycNum=y,target=z,color=k) for (x,y,z,k) in zip(chans,cycles,targets,colors)]
    # load in stack
    imstack = imread(os.path.join(stackdir,sname))
    print('***---Breaking',sname,'into channels---***')
    
    ch,r,c = np.shape(imstack)
    
    if ch != len(chans):
        print('Error with stitching ',sname)
    elif (r < 100) or (c < 100):
        print('Error with stitching ',sname)

    for i in range(ch):
        thismarker = [x for x in all_markers if x.channel==i]
        new_name = str(thismarker[0].target)+'_'+str(thismarker[0].color)+'_'+str(thismarker[0].cycle)+'_'+sname
        if os.path.exists(os.path.join(chandir,sample,area,new_name)):
            continue
        if len(thismarker)==0:
            continue
        if thismarker[0].target=='empty':
            continue
        if (thismarker[0].target=='DAPI') and (str(thismarker[0].cycle)!=dcycle):
            continue
        #if (thismarker[0].target=='blank') and (thismarker[0].cycle!=0):
        #    continue
        ch_comp = imstack[i,:,:]
        imwrite(os.path.join(chandir,sample,area,new_name),ch_comp)

def run_ashlar(batch):
    rdir = batch['ReadDir']
    stdir = batch['StackDir']
    chdir = batch['ChannelDir']
    samples = batch['Samples']
    pxsize = batch['PixelSize']
    dapicycle = batch['DAPI_cycle']
    layout = batch['Layout']
    direction = batch['Direction']
    overlap = batch['Overlap']
    maxshift = batch['MaxShift']
    sigma = batch['Sigma']
    align_channel = batch['AlignChannel']
    dset = batch['Dset']
    dataset = rdir.split('/')[-3]
    
    
    for sample in samples:
        print('***---',sample,'---***')
        area = sample.split('/')[1]
        sample = sample.split('/')[0]
        if len(sample.split('_'))>1:
            samplename=sample.split('_')[-1]
        else:
            samplename=sample
        savename = '_'.join([dset,samplename,area])+'.tif'
        print(stdir,savename)
        if os.path.exists(os.path.join(stdir,savename)):
            print('Stack already generated. Moving to channel breakdown...')
            break_stack_into_chans(stdir,savename,chdir,rdir,dapicycle)
            continue

        fields = os.listdir(os.path.join(rdir,sample,area))
        xmlFiles = [x for x in fields if x.endswith('.xml')]
        fields = [x for x in fields if not x.endswith('.xml')]
        fields.sort()
        xmlFiles.sort()
        if len(xmlFiles)==0:
            print(sample,area,'NO XML FILE')
            continue

        for i,field in enumerate(fields):
            tifdir = os.path.join(rdir,sample,area,field)
            tiles = os.listdir(tifdir)
            tnames = [x for x in tiles if x.endswith('.tif')]
            tname = tnames[0]
            try:
                tissue,sampname,samparea,cnum,F = tname.split('_')
            except:
                print('Duplicates not removed during conversion; ignoring them')
                continue
            tree = ET.parse(os.path.join(rdir,sample,area,xmlFiles[i]))
            root = tree.getroot()
            dims = [x.attrib for x in root.iter('dimensions')]
            dims = dims[0]
            width = dims['stack_columns']
            height = dims['stack_rows']
            fileList = os.listdir(tifdir)
            fileList = [x for x in fileList if x.endswith('.tif')]
            cycleList = [x.split('_')[3] for x in fileList]
            cycleList = np.unique(cycleList)
            cycleList = [int(x) for x in cycleList]
            cycleList.sort()
            cycleList = [str(x) for x in cycleList]
            fileseries_arg = ''
            for cyclenum in cycleList:
                fileseries_arg=fileseries_arg+'fileseries|'+tifdir+'|pattern='+'_'.join([tissue,sampname,samparea,str(cyclenum),'F{series}.tif|width='])+str(width)+'|height='+str(height)+'|pixel_size='+str(pxsize)+'|overlap='+str(overlap)+'|layout='+layout+'|direction='+direction+' '
            cmd = 'ashlar '+fileseries_arg+' --flip-y --pyramid --maximum-shift '+str(maxshift)+' --filter-sigma '+str(sigma)+' -c '+str(align_channel)+' -o '+stdir+savename
            subprocess.call(shlex.split(cmd))
            break_stack_into_chans(stdir,savename,chdir,rdir,dapicycle)


def main():
    cfg = config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir',type=str,default=cfg.rootdir,help='SCAMPI data directory')
    parser.add_argument('--dataset',type=str,default=cfg.dataset,help='current dataset')
    parser.add_argument('--dset',type=str,default=cfg.dset,help='current dataset')
    parser.add_argument('--read_dir',type=str,default='Ashlar/Input',help='tiff files for input into ashlar')
    parser.add_argument('--sample',type=str,default=cfg.sample,help='sample(s) to run')
    parser.add_argument('--area',type=str,default=cfg.area,help='region(s) to run')
    parser.add_argument('--dapi_cycle',type=str,default=cfg.DAPI_CYCLE,help='cycle for nucleus segmentation')
    parser.add_argument('--pxsz',type=str,default=cfg.pxsz,help='pixel size')
    parser.add_argument('--comp_stack_dir',type=str,default='Ashlar/Output_stacks/',help='directory to save stitched, aligned composite stack with all channels')
    parser.add_argument('--channel_comp_dir',type=str,default='Ashlar/Output_channels/',help='directory to save aligned channel composites')
    parser.add_argument('--layout',type=str,default=cfg.LAYOUT,help='directory to save aligned channel composites')
    parser.add_argument('--direction',type=str,default=cfg.DIRECTION,help='directory to save aligned channel composites')
    parser.add_argument('--overlap',type=str,default=cfg.OVERLAP,help='directory to save aligned channel composites')
    parser.add_argument('--maxshift',type=str,default=cfg.MAXSHIFT,help='directory to save aligned channel composites')
    parser.add_argument('--sigma',type=str,default=cfg.SIGMA,help='directory to save aligned channel composites')
    parser.add_argument('--align_channel',type=str,default=cfg.ALIGN_CHANNEL,help='directory to save aligned channel composites')
    parser.add_argument('--cpu_cap',type=int,default=mp.cpu_count(),help='max number of cpus to use')

    args,unparsed = parser.parse_known_args()
    readdir = os.path.join(args.rootdir,args.dataset,args.read_dir)
    stackdir = os.path.join(args.rootdir,args.dataset,args.comp_stack_dir)
    chandir = os.path.join(args.rootdir,args.dataset,args.channel_comp_dir)
    print(readdir)
    if not os.path.exists(stackdir):
        os.makedirs(stackdir)
    if not os.path.exists(chandir):
        os.makedirs(chandir)

    samples = os.listdir(readdir)
    if args.sample!='all_samples':
        samples = [x for x in samples if args.sample in x]
    sample_areas = []
    for sample in samples:
        sareas = os.listdir(os.path.join(readdir,sample))
        if args.area!='all_areas':
            sareas = [x for x in sareas if args.area in x]
        sample_areas.extend([sample+'/'+x for x in sareas])

    num_workers = mp.cpu_count()
    if num_workers > args.cpu_cap:
        num_workers = args.cpu_cap
    if len(sample_areas) < num_workers:
        num_workers = len(sample_areas)
    
    batchsize = int(np.ceil(len(sample_areas)/num_workers))
    batches = []
    for i in range(num_workers):
        sampbatch = sample_areas[i*batchsize:(i+1)*batchsize]
        batches.append({'ReadDir':readdir,'StackDir':stackdir,'ChannelDir':chandir,'Samples':sampbatch,'DAPI_cycle':args.dapi_cycle,'PixelSize':str(args.pxsz),'Layout':args.layout,'Direction':args.direction,'Overlap':args.overlap,'MaxShift':args.maxshift,'Sigma':args.sigma,'AlignChannel':args.align_channel,'Dset':args.dset})
    
    pool = mp.Pool(num_workers)
    pool.map(run_ashlar,batches)
    pool.close()
    

if __name__=="__main__":
    main()
