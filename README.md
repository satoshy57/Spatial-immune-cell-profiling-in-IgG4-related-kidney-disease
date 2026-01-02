# Spatial immune profiling in IgG4-related kidney disease (IgG4-RKD)

This repository contains the computational pipeline and processed datasets for the study:

"Spatial immune cells profiling reveals disease-specific immune microenvironments in IgG4-related kidney disease" (Hara S, et al. 2026).

## Table of contents

- [Project overview](#project-overview)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
  - [image-processing](#image-processing)
  - [cell-segmentation](#cell-segmentation)
  - [pSAM (pixel spectral angle mapping)](#psam-pixel-spectral-angle-mapping)
  - [pSAM-performance-validation](#psam-performance-validation)
  - [five-level-hierarchical-classification](#five-level-hierarchical-classification)
  - [structure-segmentation-analysis](#structure-segmentation-analysis)
  - [frequency-analysis](#frequency-analysis)
  - [correlation-analysis](#correlation-analysis)
  - [spatial-analysis](#spatial-analysis)
- [Contact](#contact)


## Project overview

We use CODEX (Co-detection by Indexing) highly multiplexed imaging combined with pSAM (pseudo-spectral angle mapping) and downstream spatial analyses to characterize disease-specific immune microenvironments in IgG4-RKD vs ANCA-associated vasculitis (AAV).

## Quick start

- Create the conda env from:

	- `environment.yaml`

- Recommended: open the repo in VS Code and follow the directory-level READMEs/scripts under each folder (links below).

## Repository layout (files are clickable)

Below are the main folders and key scripts. Filenames are shown inline with `code` style and linked to their relative paths for quick access.

### Image-processing
- [convert.sbatch](image-processing/convert.sbatch)
	- [ims_to_ashlar_input.py](image-processing/ims_to_ashlar_input.py) — convert raw IMS files to ASHLAR input
	- [add_special_protocols_to_ashlar_input.py](image-processing/add_special_protocols_to_ashlar_input.py)
	- [conversion_QC.py](image-processing/conversion_QC.py) — conversion QC

- [preprocess.sbatch](image-processing/preprocess.sbatch)
	- [generate_tissue_mask.py](image-processing/generate_tissue_mask.py) — generate tissue mask (whole kidney)
	- [background_subtraction_and_normalization_2.py](image-processing/background_subtraction_and_normalization_2.py) — normalization
	- [no_bkgd_sub_option.py](image-processing/no_bkgd_sub_option.py)
	- [copy_and_rename_blanks.py](image-processing/copy_and_rename_blanks.py)
	- [dapi_composite_correction.py](image-processing/dapi_composite_correction.py) — DAPI composite
	- [composite_QC.py](image-processing/composite_QC.py) — composition QC

- [ashlar.sbatch](image-processing/ashlar.sbatch)
	- [run_ashlar.py](image-processing/run_ashlar.py) — stitch tiles to whole-slide images

### Cell-segmentation
- [nuc_cellpose.sbatch](cell-segmentation/nuc_cellpose.sbatch)
	- [nuc_apply_cellpose.py](cell-segmentation/nuc_apply_cellpose.py) — nuclear segmentation (Cellpose v2)

- [dilate.sbatch](cell-segmentation/dilate.sbatch)
	- [dilate_nuclear_segmentations-wholeSlide.py](cell-segmentation/dilate_nuclear_segmentations-wholeSlide.py) — nuclear dilation to whole-cell masks

- [feats.sbatch](cell-segmentation/feats.sbatch)
	- [run_feat_extractor.py](cell-segmentation/run_feat_extractor.py) — extract centroid and mean pixel intensities (MPI)

- [batch_compactor.py](cell-segmentation/batch_compactor.py) — merge features across tissues
- [norm.sbatch](cell-segmentation/norm.sbatch)
	- [combat_batch_correction.py](cell-segmentation/combat_batch_correction.py) — batch normalization of MPIs

### pSAM (pixel spectral angle mapping)

Before running pSAM, place `reference_spectra.csv` in the dataset directory and ensure marker names match between `Normalized_composites` and `reference_spectra.csv`.

- [spectral_angle_mapping.py](pSAM/spectral_angle_mapping.py) — generate class maps
- [pixel_class_analysis2.py](pSAM/pixel_class_analysis2.py) — similarity scores, pSAM projection
- [match_SAM_to_nuc-WS-fast.py](pSAM/match_SAM_to_nuc-WS-fast.py) — assign pSAM classes to nuclei
- [get_cell_MPI_csv.py](pSAM/get_cell_MPI_csv.py) — extract cell MPIs (optional)
- [merge_class_and_MPI_csvs.py](pSAM/merge_class_and_MPI_csvs.py) — combine class & MPI per cell
- [add_datasetcolumn.py](pSAM/add_datasetcolumn.py)
- [add_patientcolumn.py](pSAM/add_patientcolumn.py)
- [SAM_analysis_plots-general2.py](pSAM/SAM_analysis_plots-general2.py) — classify and plot
- [SAM_class_composite_display-WS.py](pSAM/SAM_class_composite_display-WS.py) — class visualizations

### pSAM-performance-validation
- [pull_cells_MPI.ipynb](pSAM-performance-validation/pull_cells_MPI.ipynb)
- Ground-truth pipelines (T-bet, MPO, tubule panels): see scripts under `pSAM-performance-validation/` such as `process_images.py`, `igg4_gui.py`, and `write_labels.ipynb` variants.
- To compare pSAM vs ground truth: run `usage_cell_annotation_analysis.py` alongside `cell_annotation_analysis.py` in the same directory.

### Five-level-hierarchical-classification
- [add_CellCatName.py](five-level-hierarchical-classification/add_CellCatName.py)
- [add_CellCatName2.py](five-level-hierarchical-classification/add_CellCatName2.py)
- [add_CellCatName3.py](five-level-hierarchical-classification/add_CellCatName3.py)
- [add_MemCatName2_column.py](five-level-hierarchical-classification/add_MemCatName2_column.py)
- [add_MemCatName3_column.py](five-level-hierarchical-classification/add_MemCatName3_column.py)

### Structure-segmentation-analysis
- [structure_classification2.py](structure-segmentation-analysis/structure_classification2.py) — map cells to kidney compartments using masks
- [extract_structure_rows.py](structure-segmentation-analysis/extract_structure_rows.py)
- [cellclass_proportion_analysis_per_structure.alldataset.py](structure-segmentation-analysis/cellclass_proportion_analysis_per_structure.alldataset.py)
- [inflammation_masks_highqualitytiff.py](structure-segmentation-analysis/inflammation_masks_highqualitytiff.py)

### Frequency-analysis
- UMAPs and heatmaps:
	- [UMAP_pSAM.py](frequency-analysis/UMAP_pSAM.py)
	- [multipledataset_mpi_class_matrix.py](frequency-analysis/multipledataset_mpi_class_matrix.py)
	- [UMAP_CD4T.py](frequency-analysis/UMAP_CD4T.py)
	- [UMAP_CD8T.py](frequency-analysis/UMAP_CD8T.py)
	- [UMAP_Humoral.py](frequency-analysis/UMAP_Humoral.py)
	- [UMAP_Myeloid.py](frequency-analysis/UMAP_Myeloid.py)
	- [UMAP_OtherLymphoid.py](frequency-analysis/UMAP_OtherLymphoid.py)
	- [UMAP_Stromal.py](frequency-analysis/UMAP_Stromal.py)

- Proportion stackbars and density analyses: look under `frequency-analysis/` for scripts like `CD4T_CellCatName2_stackbar.py`, `proportions_densities_of_cellstates_CD4T.py`, etc. Files are linked in the folder.

### correlation-analysis
- [Spearmann's_correlation_matrix_for_CellCatName2_tissuemask.py](correlation-analysis/Spearmann's_correlation_matrix_for_CellCatName2_tissuemask.py)
- [circos_plot_cellcatname3_tissuemask.py](correlation-analysis/circos_plot_cellcatname3_tissuemask.py)

### spatial-analysis
- [dbscan2-Louvain_cellcatname3_tissuemask.py](spatial-analysis/dbscan2-Louvain_cellcatname3_tissuemask.py) — DBSCAN + Louvain to detect cell neighborhoods (CNs)

## Contact

- Satoshi Hara, MD, PhD — <satoshihara@staff.kanazawa-u.ac.jp>
- Marcus R. Clark, MD — <mclark@bsd.uchicago.edu>


