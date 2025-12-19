#!/bin/bash
#SBATCH --job-name=ashlar_preprocessing
#SBATCH --output=ashlar.out
#SBATCH --error=ashlar.err
#SBATCH --account=pi-mclark
#SBATCH --partition=bigmem
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --mem=1500GB
#SBATCH --mail-type=ALL  # Email notification options: ALL, BEGIN, END, FAIL, ALL, NONE
#SBATCH --mail-user=satoshihara@rcc.uchicago.edu  # Replace jdoe with your CNET and be sure to include "@rcc"

module load python/anaconda-2023.09

source activate ashlar

python run_ashlar.py


