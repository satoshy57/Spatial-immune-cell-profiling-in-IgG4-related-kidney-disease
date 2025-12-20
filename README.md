# Spatial immune profiling in IgG4-related kidney disease (IgG4-RKD) and AAV
This repository contains the computational pipeline and processed datasets for the study:
"Spatial immune cells profiling reveals disease-specific immune microenvironments in IgG4-related kidney disease" (Hara S, et al. 2026).

## Project Overview
This study utilizes CODEX (Co-detection by Indexing) highly multiplexed imaging to characterize the in situ immune landscape of IgG4-RKD compared to ANCA-associated vasculitis (AAV). We employed pSAM (pseudo-spectral angle mapping) for high-granularity cell classification and spatial neighborhood analysis to identify disease-specific immune niches.

Key Features
pSAM Classification: Accurate cell type annotation based on membrane and nuclear protein expression profiles.

Structural Segmentation: Integration of Omnipose and U-Net for kidney compartment identification.

Neighborhood Analysis: Detection of cellular neighborhoods (CNs) using DBSCAN and Louvain clustering.

## Repository Structure
├── code/
│   ├── 01_pSAM_classification.ipynb    # Pixel- and cell-level immune cell classification
│   ├── 02_kidney_segmentation.py      # Pipeline for kidney compartment segmentation
│   ├── 03_neighborhood_analysis.ipynb # DBSCAN/Louvain clustering for CN detection
│   └── 04_statistical_analysis.R      # Between-group comparisons and correlation plots
├── data/
│   ├── processed_cell_data.csv        # Anonymized cell-level marker intensities and metadata
│   └── spatial_coordinates.csv        # X-Y coordinates of cells for neighborhood analysis
├── results/
│   ├── figures/                       # Representative output plots (UMAP, CN maps)
│   └── tables/                        # Summary of cell densities and proportions
└── requirements.txt                   # List of required Python/R libraries

## Getting Started
Prerequisites
Python 3.9+

Recommended environment: Anaconda or venv

## Installation
### Clone the repository
git clone https://github.com/satoshy57/Spatial-immune-cell-profiling-in-IgG4-related-kidney-disease-and-ANCA-associated-vasculitis.git
cd Spatial-immune-cell-profiling-in-IgG4-related-kidney-disease-and-ANCA-associated-vasculitis

### Install dependencies
pip install -r requirements.txt

## Usage
Cell Classification: Use 01_pSAM_classification.ipynb to process normalized CODEX marker intensities. This script implements the pseudo-spectral angle mapping algorithm to assign cells to 59 membrane and 16 nuclear classes.

Spatial Analysis: Use 03_neighborhood_analysis.ipynb to identify immune niches. This script calculates cell densities and identifies unique Cellular Neighborhoods (CNs) such as CN2 (Trm-rich) and CN13 (GPNMB+ macrophage-rich).

## Data Availability
Processed Data: Available in the data/ folder of this repository.

Raw Imaging Data: CODEX QPTIFF files and full datasets are deposited at Zenodo [Insert DOI Link here].

## Contact
Satoshi Hara, MD, PhD: satoshihara@staff.kanazawa-u.ac.jp
Department of Nephrology and Rheumatology, Kanazawa University

Marcus R. Clark, MD: mclark@bsd.uchicago.edu
Section of Rheumatology, The University of Chicago

## Citation
If you use this code or findings in your research, please cite:
Hara, S., et al. (2026). Spatial immune cells profiling reveals disease-specific immune microenvironments in IgG4-related kidney disease. Kidney International. [DOI: XXXX/XXXX]
