#!/bin/bash
#SBATCH --job-name=dilate
#SBATCH --output=dilate.out
#SBATCH --error=dilate.err
#SBATCH --account=pi-durkeems
#SBATCH --partition=caslake
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=7
#SBATCH --exclusive
#SBATCH --mail-type=ALL  # Email notification options: ALL, BEGIN, END, FAIL, ALL, NONE
#SBATCH --mail-user=satoshihara@rcc.uchicago.edu  # Replace jdoe with your CNET and be sure to include "@rcc"

module load python/anaconda-2023.09 

source activate /project/mclark/For/Bayards_Environments/.conda/envs/visualizer_env

python dilate_nuclear_segmentations-wholeSlide.py
