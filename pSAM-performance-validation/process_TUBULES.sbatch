#!/bin/bash
#SBATCH --job-name=process_TUBULES.images
#SBATCH --output=process_TUBULES.images.out
#SBATCH --error=process_TUBULES.images.err
#SBATCH --account=pi-mclark
#SBATCH --partition=amd
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mail-type=ALL  # Email notification options: ALL, BEGIN, END, FAIL, ALL, NONE
#SBATCH --mail-user=satoshihara@rcc.uchicago.edu  # Replace jdoe with your CNET and be sure to include "@rcc"



module load python/anaconda-2023.09 

source activate /project/mclark/For/Bayards_Environments/.conda/envs/visualizer_env

python process_images_MST.py
#python process_images_MST_satoshi.py
#python process_images-MPO.py
