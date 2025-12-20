#!/bin/bash
#SBATCH --job-name=process_MPO.images
#SBATCH --output=process_MPO.images.out
#SBATCH --error=process_MPO.images.err
#SBATCH --account=pi-mclark
#SBATCH --partition=mclark-hm
#SBATCH --qos=mclark
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mail-type=ALL  # Email notification options: ALL, BEGIN, END, FAIL, ALL, NONE
#SBATCH --mail-user=satoshihara@rcc.uchicago.edu  # Replace jdoe with your CNET and be sure to include "@rcc"



module load python/anaconda-2023.09 

source activate /project/mclark/For/Bayards_Environments/.conda/envs/visualizer_env

#python process_images_MST.py
#python process_images_MST_satoshi.py
python process_images-MPO.py
