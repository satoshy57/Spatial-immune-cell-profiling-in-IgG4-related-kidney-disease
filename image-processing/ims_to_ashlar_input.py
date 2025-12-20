#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 10:56:23 2022

@author: juntingai

Edited on Wed Oct 19 11:38:35 2022

@author2: durkeems
"""

import os
import argparse
from tifffile import TiffWriter
import numpy as np
import h5py
from skimage.util import img_as_uint
from shutil import copyfile,move
from tifffile import imwrite
import skimage.io as io
import pandas as pd
from skimage.transform import pyramid_reduce
import multiprocessing as mp
from config import config

def get_bad_frame_index(first_time_point):
    # go back through the frames looking for the zeros and find the index of the first zero-frame
    if first_time_point.shape[0]==1:
        return 1
    first_bad_frame_index = first_time_point.shape[0]-1
    for i_z in range(first_time_point.shape[0])[::-1]:
        if not first_time_point[i_z].any():
            first_bad_frame_index = i_z
    return first_bad_frame_index

def get_h5_file_infor(h5_dataset):
    # get a list of all of the resolution options
    resolution_levels = list(h5_dataset)

    #resolution_levels.sort(key=lambda x: int(x.split(' ')[-1]))

    # get a list of the availabe time points
    time_points = list(h5_dataset[resolution_levels[0]])
    #time_points.sort(key=lambda x: int (x.split('')[-1]))
    n_time_points = len(time_points)

    # get a list of the channels
    channels = list(h5_dataset[resolution_levels[0]][time_points[0]])
    #channels.sort(key=lambda x: int(x.split('')[-1]))
    n_channels = len(channels)

    # get the number of z levels
    n_z_levels = h5_dataset[resolution_levels[0]][time_points[0]][channels[0]]['Data'].shape[0]
    z_levels = list(range(n_z_levels))

    # get the plane dimensions
    n_rows,n_cols = h5_dataset[resolution_levels[0]][time_points[0]][channels[0]]['Data'].shape[1:]

    return resolution_levels,time_points,n_time_points,channels,n_channels,n_z_levels,z_levels,n_rows,n_cols

# convert the ims file to a tif with no downsampling
def convert_to_tif_projection(imsPath,tifPath,DAPI_last):
    # get marker csv
    csvdir = imsPath.split('Images')[0]
    csvs = os.listdir(csvdir)
    csvs = [x for x in csvs if x.endswith('_markers.csv')]
    if len(csvs) > 1:
        print('MULTIPLE CSVS FOUND. PLEASE CHECK')
    df = pd.read_csv(os.path.join(csvdir,csvs[0]))
    
    try:
        read_file = h5py.File(imsPath,'r')
        base_data = read_file['DataSet']
        f_name = tifPath.split('/')[-1].split('.')[0]

        # assume a multicolor z stack in time
        resolution_levels, \
        time_points,n_time_points, \
        channels,n_channels, \
        n_z_levels,z_levels, \
        n_rows,n_cols= get_h5_file_infor(base_data)

        # get the index of the bad frame start
        bad_index_start = get_bad_frame_index(np.array(base_data[resolution_levels[0]][time_points[0]][channels[0]]['Data']))

        banner_text = 'File Breakdown'
        print(banner_text)
        print('_'*len(banner_text))
        print ('Channels: %d' % n_channels)
        print('Time points: %d' % n_time_points)
        print('Z levels: %d' % (bad_index_start))
        print('Native (rows, cols): (%d, %d)' % (n_rows, n_cols))
        print ('_'*len(banner_text))
        
        with TiffWriter(tifPath,imagej=True) as out_tif:
            mmap_fname = f_name+'.mmap'
            output_stack = np.memmap(mmap_fname,dtype=np.uint16,shape=(n_time_points,bad_index_start,n_channels,n_rows,n_cols), mode='w+')

            for i_t, t in enumerate(time_points):
                #print('%s/%d'%(t, n_time_points-1))
                for i_z, z_lvl in enumerate(z_levels[:bad_index_start]):
                    #print('%s/%d Z %d/%d' %(t,n_time_points-1,i_z+1,bad_index_start))
                    for i_channel, channel in enumerate(channels):
                        output_stack[i_t,i_z,i_channel] = img_as_uint(np.array(base_data[resolution_levels[0]][time_points[i_t]][channels[i_channel]]['Data'][i_z]))
              
            # save the file
            out_tif.write(output_stack)
            # delete the stack
            del output_stack
            os.remove(mmap_fname)     
    except:
        print(imsPath,'corrupted. Saving blank ROI in place')
        tifname = tifPath.split('/')[-1]
        tissue,sampleid,area,cyclenum,protocol,tilenum=tifname.split('_')
        tifPath=tifPath.replace('_'+protocol+'_','_')
        with TiffWriter(tifPath,imagej=True) as out_tif:
            tmp = np.zeros([5,2048,2048],dtype=np.uint16)
            out_tif.write(tmp)
            del tmp

    tifname = tifPath.split('/')[-1]
    try:
        tissue,sampleid,area,cyclenum,protocol,tilenum=tifname.split('_')
    except:
        print(tifname)
        return
    # For DAPI and DIC protocols, don't add changes
    if protocol=='DIC':
        return
    elif protocol=='DAPI':
        return

    # project Z stacks and add empty channels   
    IM = io.imread(tifPath) # read image
    oldtifPath = tifPath
    tifPath=tifPath.replace('_'+protocol+'_','_')
    ### If shape of IM has length of 3 --> ch,r,c ###
    ### If shape of IM has length of 4 --> z,r,c,ch ##
    # Create MIP if z stack detected
    if len(IM.shape)==4:
        IM2 = np.max(IM,axis=0)
        IM = np.moveaxis(IM2,-1,0)
        print(tifname,'MIP generated. New size: ',IM.shape)

    # Pull empty channel locations from csv
    if np.shape(IM)[0]==5:
        if DAPI_last==True:
            ch4 = IM[4,:,:]
            ch3 = IM[3,:,:]
            ch2 = IM[2,:,:]
            ch1 = IM[1,:,:]
            ch0 = IM[0,:,:]
        else:
            ch4 = IM[3,:,:]
            ch3 = IM[4,:,:]
            ch2 = IM[2,:,:]
            ch1 = IM[1,:,:]
            ch0 = IM[0,:,:]
        new_img=np.stack([ch0,ch1,ch2,ch3,ch4],axis=0)
        imwrite(tifPath, new_img)
        print(tifPath.split('/')[-1],'DAPI location corrected')
    elif np.shape(IM)[0]<5:
        cyclenum = int(cyclenum)
        cdf = df[df['cycle']==cyclenum]
        if 'empty' in list(cdf['marker']):
            blankloc = list(cdf[cdf['marker']=='empty']['channel'])
            blankloc = [x-(cyclenum*5)-1 for x in blankloc]
            blank_img = np.zeros([np.shape(IM)[1],np.shape(IM)[2]],dtype = 'uint16')
            chans = np.arange(np.shape(IM)[0])
            newchans = np.arange(np.shape(IM)[0])
            for bl in blankloc:
                newchans = np.where(newchans>=bl,newchans+1,newchans)
            imstack = [blank_img]*5
            for i,x in enumerate(newchans):
                imstack[x] = IM[i,:,:]
            new_img = np.stack(imstack,axis=0)
            imwrite(tifPath, new_img)
            print(tifPath.split('/')[-1],' blank channel added. New shape: ',np.shape(new_img))
    else:
        imwrite(tifPath,IM)
    os.remove(oldtifPath)
    print(tifname,' SAVED')         
    print('')
                
def convert_to_tif_and_save(rdir,sname,ashdir,cdict,DAPI_last):
    # make area and field folders in ashlar directory
    sarea = sname.split('/')[1]
    sname = sname.split('/')[0]
    cycles = os.listdir(os.path.join(rdir,sname))
    cycles = [x for x in cycles if os.path.isdir(os.path.join(rdir,sname,x))]
    cycles = [x for x in cycles if sarea in x]
    dropdirs = ['metadata','.DS_Store']
    cycles = list(set(cycles)-set(dropdirs))
    # loop through cycles --> map date and time to cycle number
    for cycle in cycles:
        # get info from file name
        if len(cycle.split('_'))!=8:
            print('Cycle name error',sname,cycle)
            continue
        # join date and time strings for dict key
        tissue,sampleid,date,protocol,area,time,_,fieldnum=cycle.split('_')
        if not os.path.exists(os.path.join(ashdir,sname)):
            os.makedirs(os.path.join(ashdir,sname))
        if not os.path.exists(os.path.join(ashdir,sname,area)):
            os.makedirs(os.path.join(ashdir,sname,area))
        if not os.path.exists(os.path.join(ashdir,sname,area,'Field'+fieldnum)):
            os.makedirs(os.path.join(ashdir,sname,area,'Field'+fieldnum))
        if protocol=='DAPI':
            cyclenum='998'
        elif protocol=='DIC':
            cyclenum='999'
        else:
            dt = ''.join(date.split('-'))+''.join(time.split('.'))
            cyclenum = cdict[area][fieldnum][dt]
        # get list of imaris files
        imsTiles = os.listdir(os.path.join(rdir,sname,cycle))
        xmlName = [x for x in imsTiles if x.endswith('.xml')]
        imsTiles = [x for x in imsTiles if x.endswith('.ims')]
        imsTiles.sort()
        if len(xmlName)>0:
            newXmlName = xmlName[0].replace(date+'_'+protocol+'_'+area+'_'+time,area+'_'+cyclenum+'_Field'+fieldnum)
            xmlSource = os.path.join(rdir,sname,cycle,xmlName[0])
            xmlDest = os.path.join(ashdir,sname,area,newXmlName)
            if os.path.exists(xmlSource):
                if not os.path.exists(xmlDest):
                    copyfile(xmlSource,xmlDest)
        else:
            print('No xml file for ',rdir,sname,cycle)
        for tile in imsTiles:
            # full path to imaris tile
            ImsPath = os.path.join(rdir,sname,cycle,tile)
            # new name for tiff file with cycle in lieu of date and time
            tifName = tile.replace(date+'_'+protocol+'_'+area+'_'+time,area+'_'+cyclenum+'_'+protocol)
            tifName = tifName.replace('.ims','.tif')
            # full save path
            TifPath = os.path.join(ashdir,sname,area,'Field'+fieldnum,tifName)
            # convert
            if os.path.exists(TifPath):
                continue
            else:
                newtifpath = TifPath.replace('_'+protocol+'_','_')
                if os.path.exists(newtifpath):
                    print(newtifpath,' already exists. Continuing without overwriting...')
                    continue
                else:
                    convert_to_tif_projection(ImsPath,TifPath,DAPI_last)

def make_cycle_dict(rdir,sname):
    # get cycles and organize
    sarea = sname.split('/')[1]
    sname = sname.split('/')[0]
    fldrs = os.listdir(os.path.join(rdir,sname))
    fldrs = [x for x in fldrs if sarea in os.path.join(rdir,sname,x)]
    # DIC and metadata folders have different structure
    dropdirs = ['metadata','.DS_Store']
    fldrs = list(set(fldrs)-set(dropdirs))
    
    # collect name info and area list
    name_dicts = []
    areas = []
    for fldr in fldrs:
        tissue,sampleid,date,protocol,area,time,_,fieldnum=fldr.split('_')
        name_dicts.append({'Area':area,'Field':fieldnum,'Date':date,'Time':time,'Protocol':protocol})
        areas.append(area)

    # store unique areas for selecting cycles
    areas = np.unique(areas)
    cycle_dict=dict()
    for a in areas:
        # pull unique fields in each area (all 1 for now)
        field_cycles = [x for x in name_dicts if x['Area']==a]
        afields = [x['Field'] for x in field_cycles]
        afields = np.unique(afields)
        # add to area-field list for downstream directory structure
        fdict=dict()
        for f in afields:
            # all cycles per area and field combo
            cycles = [x for x in field_cycles if x['Field']==f]
            datetimes = []
            for cycle in cycles:
                if cycle['Protocol']=='DAPI':
                    continue
                elif cycle['Protocol']=='DIC':
                    continue
                else:
                    # make giant date-time integer to sort cycles
                    d = ''.join(cycle['Date'].split('-'))
                    t = ''.join(cycle['Time'].split('.'))
                    dt = int(d+t)
                    datetimes.append(dt)
            # sort date-times --> indices are cycle number
            datetimes.sort()
            dtdict=dict()
            for i,datetime in enumerate(datetimes):
                # update date-time dictionary
                cdict = {str(datetime):str(i)}
                dtdict.update(cdict)
            # update field with all date-times
            fdict.update({f:dtdict})
        # update area with all fields
        cycle_dict.update({a:fdict})
    return cycle_dict
            
def batch_convert(samplebatch):
    readdir = samplebatch['ReadDir']
    ashlarinput = samplebatch['AshlarDir']
    samples = samplebatch['Samples']
    DAPI_last = samplebatch['DAPILast']

    for sample in samples:
        # create dictionary to map dates and times to cycle number
        # also keeps track of area and field number
        cycledict = make_cycle_dict(readdir,sample)
        
        print('***---',sample,'---***')
        # convert images to tiff files and savy to ashlar input directory
        convert_to_tif_and_save(readdir,sample,ashlarinput,cycledict,DAPI_last)
       
def main():
    cfg = config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir',type=str,default=cfg.rootdir,help='SCAMPI data directory') 
    parser.add_argument('--dataset',type=str,default=cfg.dataset,help='current dataset')
    parser.add_argument('--imsdir',type=str,default='raw_data_from_microscope/Images',help='directory with imaris sample folders')
    parser.add_argument('--ashinput',type=str,default='Ashlar/Input/',help='directory to save tiff files to')
    parser.add_argument('--sample',type=str,default=cfg.sample,help='directory to save tiff files to')
    parser.add_argument('--area',type=str,default=cfg.area,help='directory to save tiff files to')
    parser.add_argument('--dapilast',type=bool,default=cfg.DAPI_LAST,help='use tag if dapi channel is last in image stack')
    parser.add_argument('--cpu_cap',type=int,default=mp.cpu_count(),help='max number of CPUs to use')
    
    # read in user inputs
    args,unparsed = parser.parse_known_args()
    readdir = os.path.join(args.rootdir,args.dataset,args.imsdir)
    ashlarinput = os.path.join(args.rootdir,args.dataset,args.ashinput)
    
    if args.dapilast==True:
        DAPI_last=True
    else:
        DAPI_last=False

    # make directory to save tifs to
    if not os.path.exists(ashlarinput):
        os.makedirs(ashlarinput)
    # get sample names and organize
    samples = os.listdir(readdir)
    samples = [x for x in samples if os.path.isdir(os.path.join(readdir,x))]
    if args.sample!='all_samples':
        samples = [x for x in samples if args.sample in x]
    sample_areas = []
    for sample in samples:
        sareas = os.listdir(os.path.join(readdir,sample))
        dropareas = ['metadata','.DS_Store']
        sareas = list(set(sareas)-set(dropareas))
        sareas = [x.split('_')[4] for x in sareas]
        sareas = np.unique(sareas)
        if args.area!='all_areas':
            sareas = [x for x in sareas if args.area in x]
        sample_areas.extend([sample+'/'+x for x in sareas])
    sample_areas.sort()
    print(len(sample_areas),'batches to run')
    
    # loop through samples and convert
    # set up batches for multiprocessing
    num_workers = mp.cpu_count()
    if num_workers > args.cpu_cap:
        num_workers=args.cpu_cap
    #sample_areas = sample_areas[:2]
    if len(sample_areas) <= num_workers:
        num_workers=len(sample_areas)
        batches = [{'ReadDir':readdir,'AshlarDir':ashlarinput,'Samples':[x],'DAPILast':DAPI_last} for x in sample_areas]
    else:
        batchsize = np.ceil(len(sample_areas)/num_workers)
        batches = []
        for i in range(num_workers):
            sampbatch = samples[i*batchsize:(i+1)*batchsize]
            batches.append({'ReadDir':readdir,'AshlarDir':ashlarinput,'Samples':sampbatch,'DAPILast':DAPI_last})
    
    # send to cpus
    pool = mp.Pool(num_workers)
    pool.map(batch_convert,batches)
    pool.close()
    
    
if __name__ =="__main__":
    main()
