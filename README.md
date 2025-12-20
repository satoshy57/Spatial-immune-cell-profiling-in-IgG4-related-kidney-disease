# Spatial immune profiling in IgG4-related kidney disease (IgG4-RKD) and AAV
This repository contains the computational pipeline and processed datasets for the study:

"Spatial immune cells profiling reveals disease-specific immune microenvironments in IgG4-related kidney disease" (Hara S, et al. 2026).

## Project Overview
This study utilizes CODEX (Co-detection by Indexing) highly multiplexed imaging to characterize the in situ immune landscape of IgG4-RKD compared to ANCA-associated vasculitis (AAV). We employed pSAM (pseudo-spectral angle mapping) for high-granularity cell classification and spatial neighborhood analysis to identify disease-specific immune niches.

Key Features
pSAM Classification: Accurate cell type annotation based on membrane and nuclear protein expression profiles.

Structural Segmentation: Integration of Omnipose and U-Net for kidney compartment identification.

Neighborhood Analysis: Detection of cellular neighborhoods (CNs) using DBSCAN and Louvain clustering.

## Installation
environment.yaml --contains all the packages and versions used. Use this to create a virtual environment with the appropriate packages

## Code descriptions
### image-processing
1. convert.sbatch

1-1. ims_to_ashlar_input.py --raw IMS files are converted to ASHLAR input files

1-2. add_special_protocols_to_ashlar_input.py

1-3. conversion_QC.py --quality control for the conversion

2. preprocess.sbatch

2-1. generate_tissue_mask.py --generate tissue mask (whole kidney tissue area)

2-2. background_subtraction_and_normalization_2.py --perform normalization

2-3. no_bkgd_sub_option.py

2-4. copy_and_rename_blanks.py

2-5. dapi_composite_correction.py --create DAPI composite

2-6. composite_QC.py --quality control for the composition


3. ashlar.sbatch

3-1. run_ashlar.py --stitch all tiles to create whole-slide image of each channel


### cell-segmentation
1. nuc_cellpose.sbatch

1-1. nuc_apply_cellpose.py --nuclear segmentation using fine-tuned Cellpose 2

2. dilate.sbatch

2-1. dilate_nuclear_segmentations-wholeSlide.py --perform nuclear dilation to generate whole cell segmentation mask


3. feats.sbatch
3-1. run_feat_extractor.py --extract the features of location of cell centroid, mean pixel intensities of CODEX marker panel


4. batch_compactor.py  --


5. norm.sbatch

5-1. combat_batch_correction.py --normalization of mean pixel intensities of CODEX markers to reduce batch effect

### pSAM
SAM_analysis_plots-general2.py

SAM_class_composite_display-WS.py

add_datasetcolumn.py

add_patientcolumn.py

get_cell_MPI_csv.py

match_SAM_to_nuc-WS-fast.py

merge_class_and_MPI_csvs.py

pixel_class_analysis2.py

reference_spectra.csv

spectral_angle_mapping.py

### pSAM-performance-validation
process_MPO.sbatch

process_TUBULES.sbatch

process_images-MPO.py

process_images.py

process_images_MST_satoshi.py

pull_cells_MPI.ipynb

write_labels-MPO.ipynb

write_labels-TUBULES.ipynb

write_labels.ipynb

igg4_gui.py

igg4_gui-MPO_satoshi.py

igg4_gui-TUBULES_satoshi.py

cell_annotation_analysis.py

usage_cell_annotation_analysis.py

### five-level-hierarchical-classification
add_CellCatName.py

add_CellCatName2.py

add_CellCatName3.py

add_MemCatName2_column.py

add_MemCatName3_column.py

### frequency-analysis
CD4CTL_CellCatName3_stackbar.py

CD4T_CellCatName2_stackbar.py

CD4Treg_CellCatName3_stackbar.py

CD8CTL_CellCatName3_stackbar.py

CD8T_CellCatName2_stackbar.py

CD8Treg_CellCatName3_stackbar.py

Humoral_CellCatName2_stackbar.py

Myeloid_CellCatName2_stackbar.py

OtherLymphoid_CellCatName2_stackbar.py

Stromal_CellCatName2_stackbar.py

Tfh_CellCatName3_stackbar.py

UMAP_CD4T.py

UMAP_CD8T.py

UMAP_Humoral.py

UMAP_Myeloid.py

UMAP_OtherLymphoid.py

UMAP_Stromal.py

UMAP_pSAM.py

multipledataset_mpi_class_matrix.py

proportions_densities_of_cellstates_CD4T.py

proportions_densities_of_cellstates_CD8T.py

### correlation-analysis
Spearmann's_correlation_matrix_for_CellCatName2_tissuemask.py

circos_plot_cellcatname3_tissuemask.py


### spatial-analysis
dbscan2-Louvain_cellcatname3_tissuemask.py

structure-segmentation-analysis


## Contact
Satoshi Hara, MD, PhD: satoshihara@staff.kanazawa-u.ac.jp

Marcus R. Clark, MD: mclark@bsd.uchicago.edu
