#!/bin/bash
#SBATCH --job-name=nuc_cellpose1
#SBATCH --output=nuc_cellpose.out
#SBATCH --error=nuc_cellpose.err
#SBATCH --account=pi-mclark
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --constraint=a100
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mail-type=ALL  # Email notification options: ALL, BEGIN, END, FAIL, ALL, NONE
#SBATCH --mail-user=satoshihara@rcc.uchicago.edu  # Replace jdoe with your CNET and be sure to include "@rcc"

module load python/anaconda-2023.09 

source activate /project/mclark/For/Bayards_Environments/.conda/envs/cellpose2

python nuc_apply_cellpose.py
