#!/bin/bash
#SBATCH --job-name=batchcorrection_NK2
#SBATCH --output=nc.out
#SBATCH --error=nc.err
#SBATCH --account=pi-mclark
#SBATCH --partition=amd-hm
#SBATCH --time=20:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mail-type=ALL  # Email notification options: ALL, BEGIN, END, FAIL, ALL, NONE
#SBATCH --mail-user=satoshihara@rcc.uchicago.edu  # Replace jdoe with your CNET and be sure to include "@rcc"


module load python/anaconda-2023.09 

source activate /project/mclark/For/Bayards_Environments/.conda/envs/visualizer_env

python combat_batch_correction.py
