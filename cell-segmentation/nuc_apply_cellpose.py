import os
import argparse
import numpy as np
from cellpose import utils,io,models
import tifffile as tiff
from multiprocessing.pool import ThreadPool
#from mpi4py import MPI
import torch
import inspect
from config import config
import glob



signature = inspect.signature(models.Cellpose)


'''
NOTE
# select one of the following for model types 

CellPose-FT_Lupus_Nephritis
CellPose-FT_PSC_CRC
CellPose-FT_Renal_Allograft
CellPose-FT_SCAMPI_TEST_SAMPLE
'''

cellpose_model = 'CellPose-FT_Lupus_Nephritis'

custom_model = True # toggle between fine-tuned models and pretrained ones


def process_specific(dapi_path,directory):

    base_name = os.path.basename(dapi_path)
    file_name, _ = os.path.splitext(base_name)


    savedir = directory + '/' + file_name + '.npy'
    if os.path.exists(savedir):
            print("WS already saved at:",savedir, flush=True)
            return

    dapi = tiff.imread(dapi_path)
    model_path = f"/project/mclark/For/Bayard/cell_instance_segmentation/Cellpose/CustomCPModels/{cellpose_model}"
    
    model = models.CellposeModel(gpu=True, pretrained_model=model_path) # NOTE: this works 
    print("MODEL IS LOADED",flush=True)
    print("GPU?: ", torch.cuda.is_available(),flush=True)
    channels = [0,0]
    
    if custom_model: # this is for fine-tuned models
         masks, flows, styles = model.eval([dapi],batch_size=1, diameter=None, channels=channels) 
    else: # note this is for pretrained models instead of fine-tuned ones
        masks, flows, styles, diams = model.eval([dapi],batch_size=1, diameter=None, channels=channels)

    if not os.path.exists(directory):
        os.makedirs(directory)



    print("WS saved at:",savedir, flush=True)
    np.save(savedir, {'masks': np.array(masks).astype(np.uint16)}, allow_pickle=True)    



def main():
    cfg = config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir',type=str,default=cfg.rootdir,help='')
    parser.add_argument('--dataset',type=str,default=cfg.dataset,help='')
    parser.add_argument('--comp_read',type=str,default='Corrected_DAPI_composites',help='')
    parser.add_argument('--comp_write',type=str,default='nucleus_segmentations_WS',help='')
    parser.add_argument('--sample',type=str,default=cfg.sample,help='')
    parser.add_argument('--area',type=str,default=cfg.area,help='')
    parser.add_argument('--dapi_cycle',type=str,default=cfg.DAPI_CYCLE,help='')
    args = parser.parse_args()

    directory_path = args.rootdir + '/' + args.dataset + '/' + args.comp_read + '/'
    files = glob.glob(os.path.join(directory_path, '*.tif'))
    sp = args.rootdir + '/' + args.dataset + '/' + args.comp_write + '/'
    for f in files:
        if args.sample == 'all_samples':
            if args.area == 'all_areas':
                process_specific(f,sp)
            elif args.area in f:
                process_specific(f,sp)
                          
        elif args.sample in f:
            if args.area == 'all_areas':
                process_specific(f,sp)
            elif args.area in f:
               # print(f)
                process_specific(f,sp)



if __name__ =="__main__":
    main()
        
    
