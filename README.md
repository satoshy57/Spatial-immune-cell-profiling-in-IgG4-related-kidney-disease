# Spatial immune profiling in IgG4-related kidney disease

This repository contains Python scripts and processed data for the analysis presented in the paper:  
**"Spatial immune cells profiling reveals disease-specific immune microenvironments in IgG4-related kidney disease"** (Hara S, et al. 2026)

---

## Repository Structure
- `code/`: Python scripts for pSAM classification and neighborhood analysis.
- `data/`: Processed cell-level data (CSV files). 
- `results/`: Output figures and summary tables.
- `requirements.txt`: List of required Python libraries.

##  Getting Started

### Prerequisites
To run the scripts, you need Python 3.9+ and the following libraries:
- pandas
- scanpy
- squidpy (for spatial analysis)

### Installation
Clone this repository and install dependencies:
```bash
git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
cd your-repo-name
pip install -r requirements.txt
````

## Data Availability
The primary CODEX imaging data and cell-level datasets are available at:

[Zenodo DOI Link] (if applicable)

## Usage
pSAM Classification: Run code/01_pSAM_classification.ipynb to classify immune cells.

Neighborhood Analysis: Run code/02_neighborhood_analysis.py to identify CNs.

## Contact
Satoshi Hara: satoshihara@staff.kanazawa-u.ac.jp
Marcus R. Clark: mclark@bsd.uchicago.edu

##Citation
If you use this code or data, please cite:
Hara, S., et al. (2026). Spatial immune cells profiling reveals disease-specific immune environment in IgG4-related kidney disease.
