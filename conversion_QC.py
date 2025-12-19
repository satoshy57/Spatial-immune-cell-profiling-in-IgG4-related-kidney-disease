import os
import argparse
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from tifffile import imread
from config import config

def main():
    cfg = config()
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir',type=str,default=cfg.rootdir,help='')
    parser.add_argument('--dataset',type=str,default=cfg.dataset,help='')
    parser.add_argument('--readdir',type=str,default='Ashlar/Input',help='')
    parser.add_argument('--savedir',type=str,default='Ashlar',help='')
    parser.add_argument('--csvdir',type=str,default='raw_data_from_microscope',help='')
    parser.add_argument('--sample',type=str,default=cfg.sample,help='')
    parser.add_argument('--area',type=str,default=cfg.area,help='')
    parser.add_argument('--check_size',type=bool,default=False,help='')

    args,unparsed = parser.parse_known_args()

    input_dir = os.path.join(args.rootdir,args.dataset,args.readdir)
    csv_dir = os.path.join(args.rootdir,args.dataset,args.csvdir)
    save_dir = os.path.join(args.rootdir,args.dataset,args.savedir)

    if args.check_size==True:
        check_sizes=True
    else:
        check_sizes=False

    csvs = os.listdir(csv_dir)
    csvs = [x for x in csvs if x.endswith('_markers.csv')]

    df = pd.read_csv(os.path.join(csv_dir,csvs[0]))

    cycles = list(df['cycle'])
    cycles = np.unique(cycles)

    samples = os.listdir(input_dir)
    if args.sample!='all_samples':
        samples = [x for x in samples if args.sample in x]
    for sample in samples:
        areas = os.listdir(os.path.join(input_dir,sample))
        if args.area!='all_areas':
            areas = [x for x in areas if args.area in x]
        for area in areas:
            with open(os.path.join(save_dir,sample+'_'+area+'_tile-conversion-QC.txt'),'w') as f:
                f.write('***--- Check to make sure all tiles were converted for '+args.dataset+' '+args.sample+' '+args.area+' ---***\n')
                fields = os.listdir(os.path.join(input_dir,sample,area))
                xmls = [x for x in fields if x.endswith('.xml')]
                tree = ET.parse(os.path.join(input_dir,sample,area,xmls[0]))
                root = tree.getroot()
                dims = [x.attrib for x in root.iter('dimensions')]
                dims = dims[0]
                W = dims['stack_columns']
                H = dims['stack_rows']
                num_fields = len(cycles)*int(W)*int(H)
                fields = [x for x in fields if not x.endswith('.xml')]
                for field in fields:
                    imgs = os.listdir(os.path.join(input_dir,sample,area,field))
                    f.write('****'+sample+'/'+area+'****')
                    print('****',sample,'/',area,'****')
                    if len(imgs) == num_fields:
                        f.write('Contragulations! All tiles have been successfully converted for '+sample+'/'+area+'.\n')
                        print('Contragulations! All tiles have been successfully converted for '+sample+'/'+area+'.\n')
                    elif len(imgs) > num_fields:
                        f.write('Hmmm you somehow have converted more images than you started with. Please check your dataset.\n')
                        print('Hmmm you somehow have converted more images than you started with. Please check your dataset.\n')
                        print(len(imgs),num_fields)
                    else:
                        f.write('Unfortunately a few of your images have hit an error. You are missing '+str(num_fields-len(imgs))+' images.\n')
                        print('Unfortunately a few of your images have hit an error. You are missing '+str(num_fields-len(imgs))+' images.\n')
                    
                    f.write('\n')
                    f.write('>>>> Checking tile names for '+sample+'/'+area+'...\n')
                    check_imgnames = [x for x in imgs if len(x.split('_'))!=5]
                    if len(check_imgnames)>0:
                        f.write('Some names have incorrect format:\n')
                        print('Some names have incorrect format:\n')
                        [f.write(x+'\n') for x in check_imgnames]
                    else:
                        f.write('All file names have correct format for stitching and alignment!\n')
                        print('All file names have correct format for stitching and alignment!\n')
                    
                    if check_sizes==True:
                        f.write('>>>> Checking tile sizes for '+sample+'/'+area+'...\n')
                        print('Now checking tile sizes. Please see',args.rootdir,'/',args.dataset,'for output.')
                        for img in imgs:
                            im = imread(os.path.join(input_dir,sample,area,field,img))
                            ch,r,c = np.shape(im)
                            if ch!=5:
                                f.write(img+' has incorrect number of channels\n')
                            elif r!=2048:
                                f.write(img+' had incorrect number of rows\n')
                            elif c!=2048:
                                f.write(img+' had incorrect number of columns\n')
                    
                    f.write('\n')
                    f.write('Conversion quality checks are complete. Please address any errors to continue.\n')
                    f.write('\n')
            f.close()


if __name__=='__main__':
    main()
