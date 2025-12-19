#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --output=preprocess.out
#SBATCH --error=preprocess.err
#SBATCH --account=pi-mclark
#SBATCH --partition=mclark-hm
#SBATCH --time=2:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --exclusive
#SBATCH --mail-type=ALL  # Email notification options: ALL, BEGIN, END, FAIL, ALL, NONE
#SBATCH --mail-user=satoshihara@rcc.uchicago.edu  # Replace jdoe with your CNET and be sure to include "@rcc"

module load python

source activate ashlar

python generate_tissue_mask.py

python background_subtraction_and_normalization_2.py

python no_bkgd_sub_option.py

python copy_and_rename_blanks.py

python dapi_composite_correction.py

python composite_QC.py
