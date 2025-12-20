#!/bin/bash
#SBATCH --job-name=feats
#SBATCH --output=feats.out
#SBATCH --error=feats.err
#SBATCH --account=pi-mclark
#SBATCH --partition=mclark-hm
#SBATCH --qos=mclark
#SBATXH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mail-type=ALL  # Email notification options: ALL, BEGIN, END, FAIL, ALL, NONE
#SBATCH --mail-user=satoshihara@rcc.uchicago.edu  # Replace jdoe with your CNET and be sure to include "@rcc"

module load python/anaconda-2023.09 

source activate /project/mclark/For/Bayards_Environments/.conda/envs/visualizer_env

python run_feat_extractor.py
