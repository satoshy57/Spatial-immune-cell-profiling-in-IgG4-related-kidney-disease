# Spatial immune profiling in IgG4-related kidney disease (IgG4-RKD) and AAV
This repository contains the computational pipeline and processed datasets for the study:

"Spatial immune cells profiling reveals disease-specific immune microenvironments in IgG4-related kidney disease" (Hara S, et al. 2026).

## Project Overview
This study utilizes CODEX (Co-detection by Indexing) highly multiplexed imaging to characterize the in situ immune landscape of IgG4-RKD compared to ANCA-associated vasculitis (AAV). We employed pSAM (pseudo-spectral angle mapping) for high-granularity cell classification and spatial neighborhood analysis to identify disease-specific immune niches.


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


4. batch_compactor.py  --integrate the data of features from each tissue into one file at dataset level 


5. norm.sbatch

5-1. combat_batch_correction.py --normalization of mean pixel intensities of CODEX markers to reduce batch effect

### pSAM
before start: reference_spectra.csv --definition of reference spectra of membrane and nucleus classes based on CODEX marker expressions. Make sure to put this file in the directory of the dataset. Also, make sure to completely align the marker names in between Normalized_composites and reference_spectra.csv. 

1. spectral_angle_mapping.py --generate class maps

2. pixel_class_analysis2.py --calculate similarity score and generate pSAM maximum intensity projection

3. match_SAM_to_nuc-WS-fast.py --assign pSAM class to segmented nuclei

4. get_cell_MPI_csv.py --extract mean pixel intensity (MPI) on each cell (can be skipped this step when you have already gotten it)

5. merge_class_and_MPI_csvs.py --Calculate MPIs for each classified cell

6. add_datasetcolumn.py --add dataset column

7. add_patientcolumn.py --add patient column

8. SAM_analysis_plots-general2.py --classify membrane and nucleus classes on each segmented cell and plot class information

9. SAM_class_composite_display-WS.py --generate cell class visualizations


### pSAM-performance-validation
1. pull_cells_MPI.ipynb --use MPI thresholding as a guide for pulling which cells to label

2a. ground truthing immune cells using panel with T-bet

2a-1. process_images.py --preprocess ids/ strings to images for efficient ground truthing

2a-2. igg4_gui.py --label images using GUI

2a-3. write_labels.ipynb --convert manual_ground_truth.csv to labels_manual_ground_truth.csv

2b. ground truthing immune cells using panel with MPO

2b-1. process_images-MPO.py --preprocess ids/ strings to images for efficient ground truthing

2b-2. igg4_gui-MPO_satoshi.py --label images using GUI

2b-3. write_labels-MPO.ipynb --convert manual_ground_truth.csv to labels_manual_ground_truth.csv

2c. ground truthing tubular cells using panel for tubules

2c-1. process_images-MST_satoshi.py --preprocess ids/ strings to images for efficient ground truthing

2c-2. igg4_gui-TUBULES_satoshi.py --label images using GUI

2c-3. write_labels-TUBULES.ipynb --convert manual_ground_truth.csv to labels_manual_ground_truth.csv

3. Comparing pSAM classification with ground-truthed cells

3-1. run 'usage_cell_annotation_analysis.py'. Make sure to put the script 'cell_annotation_analysis.py' in the same directory.

### five-level-hierarchical-classification
1. add_CellCatName.py --add the column 'CellCatName'(Level(Lv1) cell classification)

2. add_CellCatName2.py --add the column 'CellCatName2'(Lv3 cell classification)

3. add_CellCatName3.py --add the column 'CellCatName3'(Lv2 cell classification)

4. add_MemCatName2_column.py --add the column 'MemCatName2'(Lv4 cell classification)

5. add_MemCatName3_column.py --add the column 'MemCatName3'(Lv5 cell classification)

### structure-segmentation-analysis
cellclass_proportion_analysis_per_structure.alldataset.py

extract_structure_rows.py

inflammation_masks_highqualitytiff.py

structure_classification2.py

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
