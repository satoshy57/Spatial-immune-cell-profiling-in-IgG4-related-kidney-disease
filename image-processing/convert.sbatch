#!/bin/bash
#SBATCH --job-name=convert_preprocess
#SBATCH --output=convert_tiles.out
#SBATCH --error=convert_tiles.err
#SBATCH --account=pi-mclark
#SBATCH --partition=amd
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mail-type=ALL  # Email notification options: ALL, BEGIN, END, FAIL, ALL, NONE
#SBATCH --mail-user=satoshihara@rcc.uchicago.edu  # Replace jdoe with your CNET and be sure to include "@rcc"

module load python/anaconda-2023.09

source activate ashlar

python ims_to_ashlar_input.py

python add_special_protocols_to_ashlar_input.py

python conversion_QC.py
