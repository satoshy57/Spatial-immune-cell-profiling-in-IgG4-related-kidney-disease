#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 17:13:05 2023

@author: durkeems
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from neuroCombat import neuroCombat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootdir',type=str,default='/project/mclark/SCAMPI_datasets',help='')
    parser.add_argument('--dataset',type=str,default='NK2',help='')
    parser.add_argument('--csvdir',type=str,default='combined_cell_feature_csv',help='')
    parser.add_argument('--savedir',type=str,default='Batch_Correction-withCosMx_CENTS',help='')

    args,unparsed = parser.parse_known_args()

    rdir = os.path.join(args.rootdir,args.dataset,args.csvdir)
    sdir = os.path.join(args.rootdir,args.dataset,args.csvdir,args.savedir)

    if not os.path.exists(sdir):
        os.makedirs(sdir)
    if not os.path.exists(os.path.join(sdir,'Feature_histograms')):
        os.makedirs(os.path.join(sdir,'Feature_histograms'))
        
    featFile = os.listdir(rdir)
    featFile = [x for x in featFile if 'combined_output_with_cents.csv' in x]
    if len(featFile)==0:
        print("None found!")
        return
    print(featFile[0])
    df = pd.read_csv(os.path.join(rdir,featFile[0]))

    #sample QC
    df = df[df['CompName']!='S20-16349D1_Area2'] # PSC-streak artifact
    df = df[df['CompName']!='S15-13202E1_Area1'] # PSC-cycle misalignment
    print(df.shape)
    df['CompName'] = df['CompName'].astype(str)
    df = df.dropna(subset=['CompName'])
    df = df[df['CompName'] != 'nan']
    print(df.shape)

    print(np.unique(df['CompName']))
    print(len(np.unique(df['CompName'])))
     

    # set up features and columns
    cols = df.columns
    cells,feats = df.shape

    feat_cols = [x for x in cols if ('-' in x) or ('_' in x)]
    feat_df = df[feat_cols]
    feat_df=feat_df.fillna(0)

    stainBatch = df['StainBatch']
    imageBatch = df['ImagingBatch']
    compNames = df['CompName']
    centroidRows = df['GlobalMaskCentroidRow']
    centroidCols = df['GlobalMaskCentroidCol']
    cent_areas = df['Area']
    cent_cells = df['CellID']

    imbatches = list(imageBatch)
    uimb = list(np.unique(imbatches))
    imbatches = [uimb.index(x) for x in imbatches]

    batch_col='StainBatch'
    covars = {'StainBatch':stainBatch,'ImagingBatch':imageBatch}
    covars = pd.DataFrame(covars)
    categorical_cols=[]
    print("Now loading nerucombat",flush=True)
    data_combat_all = neuroCombat(dat=feat_df.T,covars=covars,batch_col=batch_col,categorical_cols=categorical_cols)['data']
    dc_all_2 = np.transpose(data_combat_all)
    dcAll_df = pd.DataFrame(dc_all_2,columns=feat_cols)
    dcAll_df['StainBatch']=stainBatch

    feat_df['StainBatch']=stainBatch
    for col in feat_cols:
        dcAll_df.groupby('StainBatch')[col].plot.hist(bins=100,alpha=0.3,density=True)
        plt.legend()
        plt.title('Corrected '+col)
        plt.xlabel('Feature Value')
        plt.savefig(os.path.join(sdir,'Feature_histograms',col+'_BatchCorrected.png'))
        #plt.show()
        plt.close()
        feat_df.groupby('StainBatch')[col].plot.hist(bins=100,alpha=0.3,density=True)
        plt.legend()
        plt.title('Raw Feature Values '+col)
        plt.xlabel('Feature Value')
        plt.savefig(os.path.join(sdir,'Feature_histograms',col+'_RawValues.png'))
        #plt.show()
        plt.close()
    dcAll_df['ImagingBatch']=np.array(imageBatch)
    dcAll_df['CompName']=np.array(compNames)
    dcAll_df['GlobalMaskCentroidRow']=np.array(centroidRows)
    dcAll_df['GlobalMaskCentroidCol']=np.array(centroidCols)
    dcAll_df['Area']=np.array(cent_areas)
    dcAll_df['CellID']=np.array(cent_cells)



    print(os.path.join(sdir,'BatchCorrected-AllFeats_cell_features.csv'))
    dcAll_df.to_csv(os.path.join(sdir,'BatchCorrected-AllFeats_cell_features.csv'))

if __name__=="__main__":
    main()
