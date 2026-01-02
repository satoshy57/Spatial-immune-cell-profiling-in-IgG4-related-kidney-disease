import pandas as pd
import numpy as np
import os
import argparse
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
from itertools import combinations
warnings.filterwarnings('ignore')

def calculate_tissue_area(dataset, comp_name, base_path):
    """
    Calculate the tissue area from tissue mask
    """
    original_dataset = dataset
    if dataset == "IgG4-RKD":
        original_dataset = "IgG4"
    elif dataset == "KC":
        original_dataset = "NK2"
    elif dataset == "AAV":
        original_dataset = "AAV"
    
    tissue_mask_path1 = f'{base_path}/{original_dataset}/tissue_composite_masks/DAPI_UV_1_{original_dataset}_{comp_name}.tif'
    tissue_mask_path2 = f'{base_path}/{original_dataset}/tissue_composite_masks/DAPI_UV_2_{original_dataset}_{comp_name}.tif'
    tissue_mask_path1_alt = f'{base_path}/{original_dataset}/tissue_composite_masks/DAPI_UV_1_{original_dataset}_{comp_name}.tiff'
    tissue_mask_path2_alt = f'{base_path}/{original_dataset}/tissue_composite_masks/DAPI_UV_2_{original_dataset}_{comp_name}.tiff'
    
    tissue_mask_path = None
    for path in [tissue_mask_path1, tissue_mask_path2, tissue_mask_path1_alt, tissue_mask_path2_alt]:
        if os.path.exists(path):
            tissue_mask_path = path
            break
    
    if not tissue_mask_path:
        print(f"No tissue mask found for {comp_name}")
        return None
    
    try:
        tissue_mask = Image.open(tissue_mask_path)
        tissue_array = np.array(tissue_mask) > 0
        tissue_area = np.sum(tissue_array)
        print(f"Tissue area for {comp_name}: {tissue_area} pixels")
        return tissue_area
    except Exception as e:
        print(f"Error loading tissue mask {tissue_mask_path}: {str(e)}")
        return None

def calculate_inflammation_area(dataset, comp_name, base_path):
    """
    Calculate the inflammation area from inflammation masks
    """
    inflammation_base_path = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/inflammation_masks_output"
    
    original_dataset = dataset
    if dataset == "IgG4-RKD":
        original_dataset = "IgG4"
    elif dataset == "KC":
        original_dataset = "NK2"
    elif dataset == "AAV":
        original_dataset = "AAV"
    
    possible_dirs = [
        f"{original_dataset}_{comp_name}",
        f"{dataset}_{comp_name}",
        comp_name,
        f"{dataset}_{comp_name}".replace("-", ""),
        f"{dataset.replace('-', '')}_{comp_name}",
        f"{original_dataset.replace('-', '')}_{comp_name}"
    ]
    
    inflammation_mask_path = None
    for dir_name in possible_dirs:
        test_path = f"{inflammation_base_path}/{dir_name}/inflammation_mask.tiff"
        if os.path.exists(test_path):
            inflammation_mask_path = test_path
            break
    
    if not inflammation_mask_path:
        print(f"No inflammation mask found for {comp_name}. Tried directories: {possible_dirs}")
        return None
    
    try:
        inflammation_mask = Image.open(inflammation_mask_path)
        inflammation_array = np.array(inflammation_mask)
        inflammation_area = np.sum(inflammation_array == 255)
        print(f"Inflammation area for {comp_name}: {inflammation_area} pixels")
        print(f"Using inflammation mask: {inflammation_mask_path}")
        return inflammation_area
    except Exception as e:
        print(f"Error loading inflammation mask {inflammation_mask_path}: {str(e)}")
        return None

def get_cell_counts(df, comp_name, category_column='MemCatName3', exclude_categories=None):
    """
    Get cell counts for each cell type in a specific component/image
    """
    comp_df = df[df['CompName'] == comp_name]
    
    if category_column not in comp_df.columns:
        print(f"Error: {category_column} column not found for component {comp_name}")
        return {}
    
    if exclude_categories:
        comp_df = comp_df[~comp_df[category_column].isin(exclude_categories)]
    
    cell_counts = comp_df[category_column].value_counts().to_dict()
    return cell_counts

def calculate_densities(cell_counts, area, pixels_per_mm2):
    """
    Calculate cell densities (cells per mm² of area)
    """
    area_mm2 = area / pixels_per_mm2
    densities = {}
    for cell_type, count in cell_counts.items():
        densities[cell_type] = count / area_mm2 if area_mm2 > 0 else 0
    return densities

def calculate_cell_proportions(cell_counts):
    """
    Calculate proportions of each cell type relative to total cell count
    """
    total_cells = sum(cell_counts.values())
    proportions = {}
    for cell_type, count in cell_counts.items():
        proportions[cell_type] = count / total_cells if total_cells > 0 else 0
    return proportions

def aggregate_by_patient(df, patient_column, comp_to_dataset, pixels_per_mm2, base_path, exclude_categories=None, output_dir=None):
    """
    Aggregate data by patient instead of by component
    Now calculates densities per both inflammation area AND tissue area for each cell type
    """
    print(f"Aggregating data by {patient_column}...")
    
    if patient_column not in df.columns:
        print(f"Error: {patient_column} column not found in the dataframe")
        return None, None, None, None
    
    patient_data = []
    component_data = []
    
    unique_patients = df[patient_column].unique()
    print(f"Found {len(unique_patients)} unique patients/samples")
    
    Image.MAX_IMAGE_PIXELS = 1100000000
    
    for patient_id in unique_patients:
        print(f"Processing patient/sample: {patient_id}")
        
        patient_df = df[df[patient_column] == patient_id]
        patient_components = patient_df['CompName'].unique()
        
        if len(patient_components) == 0:
            print(f"No components found for patient {patient_id}, skipping")
            continue
        
        sample_comps_with_dataset = [c for c in patient_components if c in comp_to_dataset]
        if not sample_comps_with_dataset:
            print(f"No dataset information for patient {patient_id}, skipping")
            continue
        
        dataset = comp_to_dataset[sample_comps_with_dataset[0]]
        
        total_cell_counts = {}
        total_inflammation_area = 0
        total_tissue_area = 0
        valid_component_count = 0
        
        for comp_name in patient_components:
            if comp_name not in comp_to_dataset:
                print(f"Skipping component {comp_name} - dataset not determined")
                continue
            
            if comp_to_dataset[comp_name] != dataset:
                print(f"Warning: Component {comp_name} has different dataset ({comp_to_dataset[comp_name]}) than expected ({dataset})")
                continue
            
            try:
                inflammation_area = calculate_inflammation_area(dataset, comp_name, base_path)
                tissue_area = calculate_tissue_area(dataset, comp_name, base_path)
                
                if inflammation_area is None or inflammation_area == 0:
                    print(f"Invalid inflammation area for {comp_name}, skipping")
                    continue
                
                if tissue_area is None or tissue_area == 0:
                    print(f"Invalid tissue area for {comp_name}, skipping")
                    continue
                
                cell_counts = get_cell_counts(df, comp_name, category_column='MemCatName3', exclude_categories=exclude_categories)
                
                if not cell_counts:
                    print(f"No valid cell counts for {comp_name}, skipping")
                    continue
                
                total_inflammatory_cells = sum(cell_counts.values())
                inflammation_ratio = inflammation_area / tissue_area if tissue_area > 0 else 0
                density_inflammation_area = total_inflammatory_cells / (inflammation_area / pixels_per_mm2) if inflammation_area > 0 else 0
                density_tissue_area = total_inflammatory_cells / (tissue_area / pixels_per_mm2) if tissue_area > 0 else 0
                
                component_record = {
                    'PatientID': patient_id,
                    'ComponentName': comp_name,
                    'Dataset': dataset,
                    'InflammationArea': inflammation_area,
                    'TissueArea': tissue_area,
                    'InflammationRatio': inflammation_ratio,
                    'TotalInflammatoryCells': total_inflammatory_cells,
                    'DensityPerInflammationArea': density_inflammation_area,
                    'DensityPerTissueArea': density_tissue_area
                }
                component_data.append(component_record)
                
                for cell_type, count in cell_counts.items():
                    if cell_type in total_cell_counts:
                        total_cell_counts[cell_type] += count
                    else:
                        total_cell_counts[cell_type] = count
                
                total_inflammation_area += inflammation_area
                total_tissue_area += tissue_area
                valid_component_count += 1
                
            except Exception as e:
                print(f"Error processing component {comp_name} for patient {patient_id}: {str(e)}")
        
        if valid_component_count == 0 or total_inflammation_area == 0:
            print(f"No valid components for patient {patient_id}, skipping")
            continue
        
        # Calculate densities per inflammation area for each cell type
        patient_densities_inflammation = calculate_densities(total_cell_counts, total_inflammation_area, pixels_per_mm2)
        # Calculate densities per tissue area for each cell type
        patient_densities_tissue = calculate_densities(total_cell_counts, total_tissue_area, pixels_per_mm2)
        # Calculate proportions
        patient_proportions = calculate_cell_proportions(total_cell_counts)
        
        patient_inflammation_ratio = total_inflammation_area / total_tissue_area if total_tissue_area > 0 else 0
        patient_total_inflammatory_cells = sum(total_cell_counts.values())
        patient_density_tissue_total = patient_total_inflammatory_cells / (total_tissue_area / pixels_per_mm2) if total_tissue_area > 0 else 0
        patient_density_inflammation_total = patient_total_inflammatory_cells / (total_inflammation_area / pixels_per_mm2) if total_inflammation_area > 0 else 0
        
        patient_record = {
            'PatientID': patient_id,
            'Dataset': dataset,
            'ComponentCount': valid_component_count,
            'TotalInflammationArea': total_inflammation_area,
            'TotalInflammationAreaMM2': total_inflammation_area / pixels_per_mm2,
            'TotalTissueArea': total_tissue_area,
            'TotalTissueAreaMM2': total_tissue_area / pixels_per_mm2,
            'InflammationRatio': patient_inflammation_ratio,
            'TotalInflammatoryCells': patient_total_inflammatory_cells,
            'DensityPerTissueArea': patient_density_tissue_total,
            'DensityPerInflammationArea': patient_density_inflammation_total
        }
        
        for cell_type in total_cell_counts.keys():
            patient_record[f"{cell_type}_density_inflammation"] = patient_densities_inflammation[cell_type]
            patient_record[f"{cell_type}_density_tissue"] = patient_densities_tissue[cell_type]
            patient_record[f"{cell_type}_proportion"] = patient_proportions[cell_type]
            patient_record[f"{cell_type}_count"] = total_cell_counts[cell_type]
        
        patient_data.append(patient_record)
    
    if not patient_data:
        print("No patients were successfully processed")
        return None, None, None, None
    
    patient_df = pd.DataFrame(patient_data)
    component_df = pd.DataFrame(component_data)
    
    # Create density_inflammation_df (densities per inflammation area)
    density_inflammation_cols = [col for col in patient_df.columns if col.endswith('_density_inflammation')]
    density_inflammation_df = pd.DataFrame()
    density_inflammation_df['PatientID'] = patient_df['PatientID']
    density_inflammation_df['Dataset'] = patient_df['Dataset']
    for col in density_inflammation_cols:
        cell_type = col.replace('_density_inflammation', '')
        density_inflammation_df[cell_type] = patient_df[col]
    
    # Create density_tissue_df (densities per tissue area)
    density_tissue_cols = [col for col in patient_df.columns if col.endswith('_density_tissue')]
    density_tissue_df = pd.DataFrame()
    density_tissue_df['PatientID'] = patient_df['PatientID']
    density_tissue_df['Dataset'] = patient_df['Dataset']
    for col in density_tissue_cols:
        cell_type = col.replace('_density_tissue', '')
        density_tissue_df[cell_type] = patient_df[col]
    
    # Create proportion_df
    proportion_cols = [col for col in patient_df.columns if col.endswith('_proportion')]
    proportion_df = pd.DataFrame()
    proportion_df['PatientID'] = patient_df['PatientID']
    proportion_df['Dataset'] = patient_df['Dataset']
    for col in proportion_cols:
        cell_type = col.replace('_proportion', '')
        proportion_df[cell_type] = patient_df[col]
    
    if output_dir:
        patient_df.to_csv(os.path.join(output_dir, "patient_level_combined_data.csv"), index=False)
        component_df.to_csv(os.path.join(output_dir, "component_level_inflammation_data.csv"), index=False)
        density_inflammation_df.to_csv(os.path.join(output_dir, "patient_cell_density_inflammation_data.csv"), index=False)
        density_tissue_df.to_csv(os.path.join(output_dir, "patient_cell_density_tissue_data.csv"), index=False)
    
    return density_inflammation_df, density_tissue_df, proportion_df, component_df

def plot_inflammation_ratio_vs_density(component_df, save_path=None):
    """
    Create scatter plots of inflammation ratio vs density
    """
    if component_df is None or component_df.empty:
        print("No component data available for inflammation ratio plots")
        return
    
    color_palette = {
        "IgG4-RKD": "dodgerblue",
        "AAV": "hotpink",
        "KC": "limegreen"
    }
    
    plt.figure(figsize=(10, 8))
    
    for dataset in component_df['Dataset'].unique():
        dataset_data = component_df[component_df['Dataset'] == dataset]
        plt.scatter(dataset_data['InflammationRatio'], 
                   dataset_data['DensityPerTissueArea'],
                   c=color_palette.get(dataset, 'gray'),
                   label=dataset,
                   alpha=0.6,
                   s=80)
    
    plt.xlabel('Inflammation Area Ratio (Inflammation / Tissue)', fontsize=12)
    plt.ylabel('Density (Inflammatory Cells / mm² Tissue Area)', fontsize=12)
    plt.title('Inflammation Ratio vs Cell Density (per Tissue Area)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, "inflammation_ratio_vs_density_tissue.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    plt.figure(figsize=(10, 8))
    
    for dataset in component_df['Dataset'].unique():
        dataset_data = component_df[component_df['Dataset'] == dataset]
        plt.scatter(dataset_data['InflammationRatio'], 
                   dataset_data['DensityPerInflammationArea'],
                   c=color_palette.get(dataset, 'gray'),
                   label=dataset,
                   alpha=0.6,
                   s=80)
    
    plt.xlabel('Inflammation Area Ratio (Inflammation / Tissue)', fontsize=12)
    plt.ylabel('Density (Inflammatory Cells / mm² Inflammation Area)', fontsize=12)
    plt.title('Inflammation Ratio vs Cell Density (per Inflammation Area)', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, "inflammation_ratio_vs_density_inflammation.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def compare_tissue_density_across_datasets(patient_df, save_path=None):
    """
    Compare inflammatory cell density per tissue area across datasets
    """
    if patient_df is None or patient_df.empty:
        print("No patient data available for tissue density comparison")
        return
    
    full_patient_df = pd.read_csv(os.path.join(save_path, "patient_level_combined_data.csv"))
    
    if 'DensityPerTissueArea' not in full_patient_df.columns:
        print("DensityPerTissueArea column not found")
        return
    
    tissue_density_df = full_patient_df[['PatientID', 'Dataset', 'DensityPerTissueArea']].copy()
    
    datasets = tissue_density_df['Dataset'].unique()
    pairs = [
        ('IgG4-RKD', 'KC'),
        ('AAV', 'KC'),
        ('IgG4-RKD', 'AAV')
    ]
    
    print("\n" + "="*50)
    print("TISSUE DENSITY COMPARISONS")
    print("="*50)
    
    results = []
    for d1, d2 in pairs:
        if d1 in datasets and d2 in datasets:
            group1 = tissue_density_df[tissue_density_df['Dataset'] == d1]['DensityPerTissueArea'].dropna()
            group2 = tissue_density_df[tissue_density_df['Dataset'] == d2]['DensityPerTissueArea'].dropna()
            
            if len(group1) > 0 and len(group2) > 0:
                stat, pval = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                mean1 = group1.mean()
                mean2 = group2.mean()
                fold_change = mean1 / mean2 if mean2 != 0 else float('inf')
                
                results.append({
                    'Comparison': f"{d1} vs {d2}",
                    f'{d1} Mean': mean1,
                    f'{d2} Mean': mean2,
                    'Fold Change': fold_change,
                    'P-value': pval,
                    'Significant': pval < 0.05
                })
                
                print(f"\n{d1} vs {d2}:")
                print(f"  {d1} mean: {mean1:.2f} cells/mm²")
                print(f"  {d2} mean: {mean2:.2f} cells/mm²")
                print(f"  Fold change: {fold_change:.2f}")
                print(f"  P-value: {pval:.4f}")
                print(f"  Significant: {'Yes' if pval < 0.05 else 'No'}")
    
    if save_path and results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(save_path, "tissue_density_comparison_summary.csv"), index=False)
    
    plt.figure(figsize=(10, 6))
    
    color_palette = {
        "IgG4-RKD": "dodgerblue",
        "AAV": "hotpink",
        "KC": "limegreen"
    }
    
    sns.violinplot(x='Dataset', y='DensityPerTissueArea', data=tissue_density_df, palette=color_palette)
    sns.stripplot(x='Dataset', y='DensityPerTissueArea', data=tissue_density_df, color='black', alpha=0.4)
    
    plt.xlabel('Dataset', fontsize=12)
    plt.ylabel('Density (Inflammatory Cells / mm² Tissue Area)', fontsize=12)
    plt.title('Distribution of Inflammatory Cell Density per Tissue Area by Dataset', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, "tissue_density_violin_plot.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()

def perform_pairwise_comparisons(density_data, dataset_column='Dataset'):
    """
    Perform pairwise statistical comparisons between datasets with Benjamini-Hochberg correction
    """
    pairwise_results = {}
    
    if density_data is None or density_data.empty:
        print("No data available for pairwise statistical comparison.")
        return pairwise_results
    
    non_cell_columns = [dataset_column, 'PatientID']
    cell_types = [col for col in density_data.columns 
                 if col not in non_cell_columns and pd.api.types.is_numeric_dtype(density_data[col])]
    
    datasets = density_data[dataset_column].unique()
    
    pairs = [
        ('IgG4-RKD', 'KC'),
        ('AAV', 'KC'),
        ('IgG4-RKD', 'AAV')
    ]
    
    valid_pairs = [(d1, d2) for d1, d2 in pairs if d1 in datasets and d2 in datasets]
    
    for pair in valid_pairs:
        pairwise_results[pair] = {}
    
    for d1, d2 in valid_pairs:
        p_values_dict = {}
        valid_tests = []
        
        for cell_type in cell_types:
            group1 = density_data[density_data[dataset_column] == d1][cell_type].dropna()
            group2 = density_data[density_data[dataset_column] == d2][cell_type].dropna()
            
            if len(group1) > 0 and len(group2) > 0:
                try:
                    stat, pval = stats.mannwhitneyu(group1, group2, alternative='two-sided')
                    
                    mean1 = group1.mean()
                    mean2 = group2.mean()
                    
                    if mean2 == 0:
                        if mean1 == 0:
                            fold_change = 1.0
                        else:
                            fold_change = float('inf')
                    else:
                        fold_change = mean1 / mean2
                    
                    p_values_dict[cell_type] = {
                        'statistic': stat,
                        'p_value': pval,
                        'mean_1': mean1,
                        'mean_2': mean2,
                        'fold_change': fold_change
                    }
                    valid_tests.append(cell_type)
                    
                except Exception as e:
                    pairwise_results[(d1, d2)][cell_type] = {
                        'error': str(e)
                    }
            else:
                pairwise_results[(d1, d2)][cell_type] = {
                    'error': f"Insufficient data for one or both groups ({len(group1)}, {len(group2)})"
                }
        
        if valid_tests:
            p_values = [p_values_dict[ct]['p_value'] for ct in valid_tests]
            
            p_values_sorted = np.sort(p_values)
            ranks = np.arange(1, len(p_values_sorted) + 1)
            
            fdr_threshold = 0.05
            critical_values = ranks / len(p_values) * fdr_threshold
            
            significant_threshold = 0
            for i, p in enumerate(p_values_sorted):
                if p <= critical_values[i]:
                    significant_threshold = p
            
            for ct in valid_tests:
                pval = p_values_dict[ct]['p_value']
                pairwise_results[(d1, d2)][ct] = p_values_dict[ct]
                
                pairwise_results[(d1, d2)][ct]['significant'] = pval <= significant_threshold
                
                idx = np.where(p_values_sorted == pval)[0][0]
                rank = idx + 1
                adjusted_p = min(pval * len(p_values) / rank, 1.0)
                pairwise_results[(d1, d2)][ct]['adjusted_p_value'] = adjusted_p
        
        print(f"Completed pairwise comparison for {d1} vs {d2} with Benjamini-Hochberg correction")
    
    return pairwise_results

def perform_within_dataset_comparisons(data_df, dataset_column='Dataset'):
    """
    Perform pairwise statistical comparisons between cell types within each dataset
    """
    within_dataset_results = {}
    
    if data_df is None or data_df.empty:
        print("No data available for within-dataset statistical comparison.")
        return within_dataset_results
    
    non_cell_columns = [dataset_column, 'PatientID']
    cell_types = [col for col in data_df.columns 
                 if col not in non_cell_columns and pd.api.types.is_numeric_dtype(data_df[col])]
    
    datasets = data_df[dataset_column].unique()
    
    for dataset in datasets:
        print(f"Performing within-dataset comparisons for {dataset}")
        within_dataset_results[dataset] = {}
        
        dataset_data = data_df[data_df[dataset_column] == dataset]
        
        if len(dataset_data) < 3:
            print(f"Insufficient data for dataset {dataset} (n={len(dataset_data)})")
            continue
        
        cell_type_pairs = list(combinations(cell_types, 2))
        
        p_values_dict = {}
        valid_tests = []
        
        for cell_type1, cell_type2 in cell_type_pairs:
            paired_data = dataset_data[[cell_type1, cell_type2]].dropna()
            
            if len(paired_data) < 3:
                within_dataset_results[dataset][(cell_type1, cell_type2)] = {
                    'error': f"Insufficient paired data (n={len(paired_data)})"
                }
                continue
            
            group1 = paired_data[cell_type1]
            group2 = paired_data[cell_type2]
            
            try:
                stat, pval = stats.wilcoxon(group1, group2, alternative='two-sided')
                
                mean1 = group1.mean()
                mean2 = group2.mean()
                
                if mean2 == 0:
                    if mean1 == 0:
                        fold_change = 1.0
                    else:
                        fold_change = float('inf')
                else:
                    fold_change = mean1 / mean2
                
                p_values_dict[(cell_type1, cell_type2)] = {
                    'statistic': stat,
                    'p_value': pval,
                    'mean_1': mean1,
                    'mean_2': mean2,
                    'fold_change': fold_change,
                    'n_pairs': len(paired_data)
                }
                valid_tests.append((cell_type1, cell_type2))
                
            except Exception as e:
                within_dataset_results[dataset][(cell_type1, cell_type2)] = {
                    'error': str(e)
                }
        
        if valid_tests:
            p_values = [p_values_dict[pair]['p_value'] for pair in valid_tests]
            
            p_values_sorted = np.sort(p_values)
            ranks = np.arange(1, len(p_values_sorted) + 1)
            
            fdr_threshold = 0.05
            critical_values = ranks / len(p_values) * fdr_threshold
            
            significant_threshold = 0
            for i, p in enumerate(p_values_sorted):
                if p <= critical_values[i]:
                    significant_threshold = p
            
            for pair in valid_tests:
                pval = p_values_dict[pair]['p_value']
                within_dataset_results[dataset][pair] = p_values_dict[pair]
                
                within_dataset_results[dataset][pair]['significant'] = pval <= significant_threshold
                
                idx = np.where(p_values_sorted == pval)[0][0]
                rank = idx + 1
                adjusted_p = min(pval * len(p_values) / rank, 1.0)
                within_dataset_results[dataset][pair]['adjusted_p_value'] = adjusted_p
        
        print(f"Completed within-dataset comparison for {dataset}")
    
    return within_dataset_results

def create_within_dataset_summary_report(within_dataset_results, data_df, save_path=None, analysis_type="density_inflammation"):
    """
    Create a comprehensive summary report of within-dataset comparisons
    """
    if not within_dataset_results or data_df is None or data_df.empty:
        print("Insufficient data to generate within-dataset summary report.")
        return
    
    report_rows = []
    
    for dataset, results in within_dataset_results.items():
        dataset_data = data_df[data_df['Dataset'] == dataset]
        
        non_cell_columns = ['Dataset', 'PatientID']
        cell_types = [col for col in dataset_data.columns 
                     if col not in non_cell_columns and pd.api.types.is_numeric_dtype(dataset_data[col])]
        
        mean_values = {}
        for cell_type in cell_types:
            mean_values[cell_type] = dataset_data[cell_type].mean()
        
        for pair, result in results.items():
            if isinstance(pair, tuple) and len(pair) == 2 and isinstance(result, dict):
                if 'p_value' in result:
                    cell_type1, cell_type2 = pair
                    
                    row = {
                        'Dataset': dataset,
                        'Cell Type 1': cell_type1,
                        'Cell Type 2': cell_type2,
                        'Cell Type 1 Mean': mean_values.get(cell_type1, np.nan),
                        'Cell Type 2 Mean': mean_values.get(cell_type2, np.nan),
                        'Fold Change': result['fold_change'],
                        'P-value': result['p_value'],
                        'Adjusted P-value': result['adjusted_p_value'],
                        'Significant': result['significant'],
                        'N Pairs': result.get('n_pairs', np.nan)
                    }
                    
                    report_rows.append(row)
    
    report_df = pd.DataFrame(report_rows)
    
    if report_df.empty:
        print("No valid comparisons found for summary report.")
        return None
    
    report_df = report_df.sort_values(['Dataset', 'Adjusted P-value'])
    
    if save_path:
        csv_path = os.path.join(save_path, f"within_dataset_{analysis_type}_comparison_summary.csv")
        report_df.to_csv(csv_path, index=False)
        print(f"Within-dataset summary report saved to {csv_path}")
    
    return report_df

def plot_within_dataset_violin_plots(data_df, save_path=None, analysis_type="density_inflammation"):
    """
    Create violin plots comparing different cell types within each dataset
    """
    if data_df is None or data_df.empty:
        print("No data available for creating within-dataset violin plots.")
        return
    
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
    
    df = data_df.copy()
    
    dataset_col = 'Dataset'
    id_columns = [dataset_col, 'PatientID']
    cell_type_cols = [col for col in df.columns if col not in id_columns and pd.api.types.is_numeric_dtype(df[col])]
    
    non_zero_cell_types = [col for col in cell_type_cols if df[col].max() > 0]
    
    if not non_zero_cell_types:
        print("No cell types with non-zero values found.")
        return
    
    datasets = sorted(df[dataset_col].unique())
    
    n_cell_types = len(non_zero_cell_types)
    colors = plt.cm.Set3(np.linspace(0, 1, n_cell_types))
    cell_type_colors = dict(zip(non_zero_cell_types, colors))
    
    for dataset in datasets:
        dataset_data = df[df[dataset_col] == dataset]
        
        if len(dataset_data) < 3:
            print(f"Insufficient data for dataset {dataset} (n={len(dataset_data)})")
            continue
        
        plot_data = dataset_data.melt(
            id_vars=['PatientID'], 
            value_vars=non_zero_cell_types,
            var_name='Cell Type', 
            value_name='Value'
        )
        
        plot_data = plot_data[plot_data['Value'] > 0]
        
        if plot_data.empty:
            print(f"No non-zero values for dataset {dataset}")
            continue
        
        plt.figure(figsize=(max(12, len(non_zero_cell_types) * 0.8), 8))
        
        ax = sns.violinplot(
            x='Cell Type', 
            y='Value', 
            data=plot_data,
            palette=[cell_type_colors[ct] for ct in non_zero_cell_types if ct in plot_data['Cell Type'].values]
        )
        
        sns.stripplot(
            x='Cell Type', 
            y='Value', 
            data=plot_data,
            color='black', 
            alpha=0.6, 
            size=4
        )
        
        plt.xlabel('Cell Type', fontsize=12)
        
        if analysis_type == "proportion":
            measure_type = "Proportion"
        elif analysis_type == "density_tissue":
            measure_type = "Density (cells/mm² tissue area)"
        else:  # density_inflammation
            measure_type = "Density (cells/mm² inflammation area)"
        
        plt.ylabel(f'{measure_type}', fontsize=12)
        
        if analysis_type == "proportion":
            title_measure = "Proportions"
        elif analysis_type == "density_tissue":
            title_measure = "Densities (tissue area)"
        else:
            title_measure = "Densities (inflammation area)"
        
        plt.title(f'Distribution of Cell {title_measure} within {dataset} Dataset', fontsize=14)
        
        if "density" in analysis_type:
            plt.ylim(bottom=0)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, f"within_dataset_{analysis_type}_violin_{dataset}.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close()
    
    plt.figure(figsize=(max(16, len(non_zero_cell_types) * len(datasets) * 0.6), 10))
    
    plot_data_all = df.melt(
        id_vars=[dataset_col, 'PatientID'], 
        value_vars=non_zero_cell_types,
        var_name='Cell Type', 
        value_name='Value'
    )
    
    plot_data_all = plot_data_all[plot_data_all['Value'] > 0]
    
    if not plot_data_all.empty:
        ax = sns.violinplot(
            x='Cell Type', 
            y='Value', 
            hue=dataset_col,
            data=plot_data_all,
            palette={
                "IgG4-RKD": "lightblue",
                "AAV": "pink", 
                "KC": "lightgreen"
            },
            split=False
        )
        
        plt.xlabel('Cell Type', fontsize=12)
        plt.ylabel(f'{measure_type}', fontsize=12)
        
        plt.title(f'Comparison of Cell {title_measure} across Datasets and Cell Types', fontsize=14)
        
        if "density" in analysis_type:
            plt.ylim(bottom=0)
        
        plt.xticks(rotation=45, ha='right')
        
        plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(os.path.join(save_path, f"within_dataset_{analysis_type}_violin_combined.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close()
    
    try:
        if not plot_data_all.empty:
            g = sns.FacetGrid(
                plot_data_all, 
                col=dataset_col, 
                height=6, 
                aspect=1.2,
                sharey=True
            )
            
            g.map_dataframe(
                sns.violinplot, 
                x='Cell Type', 
                y='Value',
                palette=cell_type_colors
            )
            
            g.map_dataframe(
                sns.stripplot, 
                x='Cell Type', 
                y='Value',
                color='black', 
                alpha=0.4, 
                size=3
            )
            
            g.set_axis_labels('Cell Type', f'{measure_type}')
            g.set_titles(col_template='{col_name} Dataset')
            
            if "density" in analysis_type:
                for ax in g.axes.flat:
                    ax.set_ylim(bottom=0)
            
            for ax in g.axes.flat:
                ax.tick_params(axis='x', rotation=45)
            
            g.tight_layout()
            
            if save_path:
                g.savefig(os.path.join(save_path, f"within_dataset_{analysis_type}_violin_facet.png"), 
                        dpi=300, bbox_inches='tight')
                plt.close()
        else:
            print("No categories with non-zero values found for facet grid")
        
    except Exception as e:
        print(f"Error creating facet grid: {str(e)}")
    
    return df

def create_summary_report(pairwise_results, density_df, save_path=None, analysis_type="density_inflammation"):
    """
    Create a comprehensive summary report of all pairwise comparisons
    """
    if not pairwise_results or density_df is None or density_df.empty:
        print("Insufficient data to generate summary report.")
        return
    
    report_rows = []
    
    cell_types = set()
    for pair_results in pairwise_results.values():
        cell_types.update(pair_results.keys())
    
    cell_types = sorted(list(cell_types))
    datasets = sorted(density_df['Dataset'].unique())
    
    mean_values = {}
    for dataset in datasets:
        dataset_data = density_df[density_df['Dataset'] == dataset]
        mean_values[dataset] = {}
        for cell_type in cell_types:
            if cell_type in dataset_data.columns:
                mean_values[dataset][cell_type] = dataset_data[cell_type].mean()
    
    for cell_type in cell_types:
        row = {'Cell Type': cell_type}
        
        for dataset in datasets:
            if dataset in mean_values and cell_type in mean_values[dataset]:
                row[f"{dataset} Mean"] = mean_values[dataset][cell_type]
        
        for pair, results in pairwise_results.items():
            d1, d2 = pair
            pair_name = f"{d1} vs {d2}"
            
            if cell_type in results and 'p_value' in results[cell_type]:
                row[f"{pair_name} p-value"] = results[cell_type]['p_value']
                row[f"{pair_name} fold change"] = results[cell_type]['fold_change']
                row[f"{pair_name} significant"] = results[cell_type]['significant']
        
        report_rows.append(row)
    
    report_df = pd.DataFrame(report_rows)
    
    if save_path:
        csv_path = os.path.join(save_path, f"{analysis_type}_comparison_summary.csv")
        report_df.to_csv(csv_path, index=False)
        print(f"Summary report saved to {csv_path}")
    
    return report_df

def plot_pairwise_volcano_plots(pairwise_results, save_path=None, analysis_type="density_inflammation"):
    """
    Create volcano plots of pairwise statistical comparison results using BH-adjusted p-values
    """
    if not pairwise_results:
        print("No pairwise results to visualize.")
        return
    
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for pair, results in pairwise_results.items():
        if not results:
            print(f"No results for pair {pair}")
            continue
        
        d1, d2 = pair
        pair_name = f"{d1}_vs_{d2}"
        
        cell_types = []
        p_values = []
        adjusted_p_values = []
        fold_changes = []
        
        for cell_type, result in results.items():
            if 'p_value' in result and 'adjusted_p_value' in result and 'fold_change' in result:
                cell_types.append(cell_type)
                p_values.append(result['p_value'])
                adjusted_p_values.append(result['adjusted_p_value'])
                fold_changes.append(result['fold_change'])
        
        if not cell_types:
            print(f"No valid data for pair {pair}")
            continue
        
        plot_df = pd.DataFrame({
            'Cell Type': cell_types,
            'P-Value': p_values,
            'Adjusted P-Value': adjusted_p_values,
            'Fold Change': fold_changes,
            '-log10(P)': -np.log10(p_values),
            '-log10(Adjusted P)': -np.log10(adjusted_p_values),
            'log2(FC)': np.log2([fc if fc != float('inf') else 1000 for fc in fold_changes])
        })
        
        plot_df = plot_df.sort_values('Adjusted P-Value')
        
        for use_adjusted in [False, True]:
            plt.figure(figsize=(12, 10))
            
            p_col = 'Adjusted P-Value' if use_adjusted else 'P-Value'
            log_p_col = '-log10(Adjusted P)' if use_adjusted else '-log10(P)'
            
            colors = []
            for _, row in plot_df.iterrows():
                if row[p_col] < 0.05:
                    if row['Fold Change'] > 1:
                        colors.append('red')
                    else:
                        colors.append('blue')
                else:
                    colors.append('gray')
            
            plt.scatter(plot_df['log2(FC)'], plot_df[log_p_col], c=colors, alpha=0.7, s=80)
            
            plt.axhline(y=-np.log10(0.05), color='black', linestyle='--', alpha=0.5)
            
            plt.axvline(x=np.log2(2), color='black', linestyle='--', alpha=0.5)
            plt.axvline(x=-np.log2(2), color='black', linestyle='--', alpha=0.5)
            
            for _, row in plot_df[plot_df[p_col] < 0.05].iterrows():
                plt.annotate(row['Cell Type'], 
                           xy=(row['log2(FC)'], row[log_p_col]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=10, ha='left', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
            
            plt.xlabel('log2(Fold Change)', fontsize=12)
            plt.ylabel(f'-log10({p_col})', fontsize=12)
            
            if analysis_type == "proportion":
                measure_type = "Proportions"
            elif analysis_type == "density_tissue":
                measure_type = "Densities (tissue area)"
            else:
                measure_type = "Densities (inflammation area)"
            
            p_type = "BH-Adjusted" if use_adjusted else "Raw"
            plt.title(f'Volcano Plot of Patient-Level Cell {measure_type}: {d1} vs {d2}\n({p_type} P-values)', 
                      fontsize=14)
            
            plt.grid(True, alpha=0.3)
            
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label=f'Higher in {d1}'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label=f'Higher in {d2}'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, label='Not significant')
            ]
            plt.legend(handles=legend_elements, loc='best', fontsize=10)
            
            plt.tight_layout()
            
            if save_path:
                p_value_type = "adjusted" if use_adjusted else "raw"
                plt.savefig(os.path.join(save_path, f"volcano_plot_{analysis_type}_{pair_name}_{p_value_type}.png"), 
                            dpi=300, bbox_inches='tight')
                plt.close()

def plot_integrated_heatmap(pairwise_results, save_path=None, analysis_type="density_inflammation"):
    """
    Create integrated heatmap of pairwise statistical comparison results
    """
    if not pairwise_results:
        print("No pairwise results to visualize.")
        return
    
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
    
    pairs = [
        ('IgG4-RKD', 'KC'),
        ('AAV', 'KC'),
        ('IgG4-RKD', 'AAV')
    ]
    
    available_pairs = [pair for pair in pairs if pair in pairwise_results and pairwise_results[pair]]
    
    if not available_pairs:
        print("No valid comparison pairs found in results.")
        return
    
    all_cell_types = set()
    for pair in available_pairs:
        all_cell_types.update(pairwise_results[pair].keys())
    
    all_cell_types = sorted(list(all_cell_types))
    n_cell_types = len(all_cell_types)
    n_pairs = len(available_pairs)
    
    for use_adjusted in [False, True]:
        p_key = 'adjusted_p_value' if use_adjusted else 'p_value'
        p_type = "BH-Adjusted" if use_adjusted else "Raw"
        
        fold_changes = np.full((n_cell_types, n_pairs), np.nan)
        log10_p_values = np.full((n_cell_types, n_pairs), np.nan)
        significance = np.full((n_cell_types, n_pairs), False)
        
        for j, pair in enumerate(available_pairs):
            for i, cell_type in enumerate(all_cell_types):
                if (cell_type in pairwise_results[pair] and 
                    isinstance(pairwise_results[pair][cell_type], dict) and
                    'fold_change' in pairwise_results[pair][cell_type]):
                    
                    fc = pairwise_results[pair][cell_type]['fold_change']
                    if fc == float('inf'):
                        fc = 1000
                    elif fc == 0:
                        fc = 0.001
                    
                    fold_changes[i, j] = fc
                    
                    if p_key in pairwise_results[pair][cell_type]:
                        p_val = pairwise_results[pair][cell_type][p_key]
                        log10_p_values[i, j] = -np.log10(max(p_val, 1e-10))
                        significance[i, j] = p_val < 0.05
        
        log2_fc = np.log2(fold_changes)
        
        log2_fc[~np.isfinite(log2_fc)] = np.nan
        
        masked_log2_fc = np.ma.array(log2_fc, mask=np.isnan(log2_fc))
        masked_log10_p = np.ma.array(log10_p_values, mask=np.isnan(log10_p_values))
        
        plt.figure(figsize=(16, max(6, min(n_cell_types * 0.5, 20))))
        
        gs = plt.GridSpec(1, 2, width_ratios=[3, 1])
        
        ax1 = plt.subplot(gs[0])
        
        cmap_fc = plt.cm.RdBu_r
        cmap_fc.set_bad('lightgray')
        
        valid_fc = masked_log2_fc.compressed()
        if len(valid_fc) > 0:
            max_abs_log2_fc = max(2, np.max(np.abs(valid_fc)))
        else:
            max_abs_log2_fc = 2
        norm_fc = plt.Normalize(-max_abs_log2_fc, max_abs_log2_fc)
        
        im1 = ax1.imshow(masked_log2_fc, cmap=cmap_fc, norm=norm_fc, aspect='equal')
        
        ax1.set_yticks(np.arange(n_cell_types))
        ax1.set_yticklabels(all_cell_types)
        
        ax1.set_xticks(np.arange(n_pairs))
        pair_labels = [f"{d1} vs {d2}" for d1, d2 in available_pairs]
        ax1.set_xticklabels(pair_labels, rotation=45, ha='right')
        
        ax1.set_xticks(np.arange(-0.5, n_pairs, 1), minor=True)
        ax1.set_yticks(np.arange(-0.5, n_cell_types, 1), minor=True)
        ax1.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        ax1.tick_params(which='minor', size=0)
        
        cbar1 = plt.colorbar(im1, ax=ax1)
        cbar1.set_label('log2(Fold Change)')
        
        # IMPROVED ASTERISK PLOTTING WITH DYNAMIC COLORING
        for i in range(n_cell_types):
            for j in range(n_pairs):
                if significance[i, j]:
                    # Determine text color based on background darkness
                    if not np.isnan(log2_fc[i, j]):
                        normalized_value = (log2_fc[i, j] + max_abs_log2_fc) / (2 * max_abs_log2_fc)
                        # Black asterisks for middle 50% (-50% to +50%), white for outer extremes
                        if normalized_value < 0.25 or normalized_value > 0.75:
                            text_color = 'white'
                        else:
                            text_color = 'black'
                    else:
                        text_color = 'black'
                    ax1.text(j, i, '*', ha='center', va='center', color=text_color, fontsize=40, fontweight='bold')
        
        ax1.set_title('Fold Change (log2)')
        
        ax2 = plt.subplot(gs[1])
        
        cmap_p = plt.cm.YlOrRd
        cmap_p.set_bad('lightgray')
        
        valid_p = masked_log10_p.compressed()
        if len(valid_p) > 0:
            max_log10_p = max(2, np.max(valid_p))
        else:
            max_log10_p = 2
        norm_p = plt.Normalize(0, max_log10_p)
        
        im2 = ax2.imshow(masked_log10_p, cmap=cmap_p, norm=norm_p, aspect='equal')
        
        ax2.set_yticks([])
        
        ax2.set_xticks([0])
        ax2.set_xticklabels([f'{p_type} P-value'], rotation=45, ha='right')
        
        ax2.set_xticks(np.arange(-0.5, 1, 1), minor=True)
        ax2.set_yticks(np.arange(-0.5, n_cell_types, 1), minor=True)
        ax2.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        ax2.tick_params(which='minor', size=0)
        
        cbar2 = plt.colorbar(im2, ax=ax2)
        cbar2.set_label(f'-log10({p_type} P-value)')
        
        significance_level = -np.log10(0.05)
        cbar2.ax.axhline(y=significance_level, color='k', linestyle='--')
        cbar2.ax.text(0.5, significance_level, 'p=0.05', va='bottom', ha='center', transform=cbar2.ax.get_yaxis_transform())
        
        ax2.set_title('Statistical Significance')
        
        if analysis_type == "proportion":
            measure_type = "Cell Proportions"
        elif analysis_type == "density_tissue":
            measure_type = "Cell Densities (tissue area)"
        else:
            measure_type = "Cell Densities (inflammation area)"
        
        plt.suptitle(f'Patient-Level {measure_type} Differences Across Disease Groups ({p_type} P-values)', fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            p_value_type = "adjusted" if use_adjusted else "raw"
            plt.savefig(os.path.join(save_path, f"patient_{analysis_type}_heatmap_{p_value_type}.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close()

def plot_stacked_bar_by_dataset(data_df, save_path=None, analysis_type="density_inflammation"):
    """
    Create stacked bar charts showing the proportions of cell categories by dataset
    """
    if data_df is None or data_df.empty:
        print("No data available for creating stacked bar charts.")
        return
    
    df = data_df.copy()
    
    dataset_col = 'Dataset'
    id_columns = [dataset_col, 'PatientID']
    cell_type_cols = [col for col in df.columns if col not in id_columns and pd.api.types.is_numeric_dtype(df[col])]
    
    dataset_means = df.groupby(dataset_col)[cell_type_cols].mean().reset_index()
    
    for idx, row in dataset_means.iterrows():
        total_value = sum(row[col] for col in cell_type_cols)
        if total_value > 0:
            for col in cell_type_cols:
                dataset_means.loc[idx, f"{col}_prop"] = row[col] / total_value * 100
        else:
            for col in cell_type_cols:
                dataset_means.loc[idx, f"{col}_prop"] = 0
    
    prop_cols = [col for col in dataset_means.columns if col.endswith('_prop')]
    
    plt.figure(figsize=(14, 8))
    
    colors = plt.cm.tab10.colors
    
    bottom = np.zeros(len(dataset_means))
    for i, col in enumerate(prop_cols):
        category = col.replace('_prop', '')
        values = dataset_means[col].values
        plt.bar(dataset_means[dataset_col], values, bottom=bottom, 
                label=category, color=colors[i % len(colors)])
        bottom += values
    
    plt.xlabel('Dataset', fontsize=14)
    plt.ylabel('Proportion (%)', fontsize=14)
    
    if analysis_type == "proportion":
        measure_type = "Proportions"
    elif analysis_type == "density_tissue":
        measure_type = "Densities (tissue area)"
    else:
        measure_type = "Densities (inflammation area)"
    
    plt.title(f'Patient-Level Cell {measure_type} by Dataset', fontsize=16)
    
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(os.path.join(save_path, f"patient_{analysis_type}_stacked_bar.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    if save_path:
        dataset_means.to_csv(os.path.join(save_path, f"patient_{analysis_type}_proportions_by_dataset.csv"), index=False)
    
    return dataset_means

def plot_grouped_violinplots(data_df, save_path=None, analysis_type="density_inflammation"):
    """
    Create grouped violin plots for each cell type across datasets
    """
    if data_df is None or data_df.empty:
        print("No data available for creating violin plots.")
        return
    
    df = data_df.copy()
    
    dataset_col = 'Dataset'
    id_columns = [dataset_col, 'PatientID']
    cell_type_cols = [col for col in df.columns if col not in id_columns and pd.api.types.is_numeric_dtype(df[col])]
    
    custom_palette = {
        "IgG4-RKD": "lightblue",
        "AAV": "pink",
        "KC": "lightgreen"
    }
    
    for cell_type in cell_type_cols:
        plt.figure(figsize=(10, 6))
        
        if df[cell_type].max() > 0:
            sns.violinplot(x=dataset_col, y=cell_type, data=df, palette=custom_palette)
            sns.stripplot(x=dataset_col, y=cell_type, data=df, color='black', alpha=0.4)
            
            plt.xlabel('Dataset', fontsize=12)
            
            if analysis_type == "proportion":
                measure_type = "Proportion"
            elif analysis_type == "density_tissue":
                measure_type = "Density (cells/mm² tissue area)"
            else:
                measure_type = "Density (cells/mm² inflammation area)"
            
            plt.ylabel(f'{cell_type} {measure_type}', fontsize=12)
            
            if analysis_type == "proportion":
                title_measure = "Proportion"
            elif analysis_type == "density_tissue":
                title_measure = "Density (tissue area)"
            else:
                title_measure = "Density (inflammation area)"
            
            plt.title(f'Distribution of {cell_type} {title_measure} by Dataset', fontsize=14)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(os.path.join(save_path, f"patient_{analysis_type}_violinplot_{cell_type}.png"), 
                            dpi=300, bbox_inches='tight')
                plt.close()
        else:
            print(f"Skipping violin plot for {cell_type} - all values are zero")
            plt.close()
    
    try:
        non_zero_cats = [cat for cat in cell_type_cols if df[cat].max() > 0]
        
        if non_zero_cats:
            plot_data = df.melt(id_vars=[dataset_col, 'PatientID'], 
                              value_vars=non_zero_cats,
                              var_name='Cell Type', value_name='Value')
            
            g = sns.FacetGrid(plot_data, col='Cell Type', col_wrap=3, 
                            height=4, aspect=1.2, sharey=False)
            
            g.map_dataframe(sns.violinplot, x=dataset_col, y='Value', palette=custom_palette)
            g.map_dataframe(sns.stripplot, x=dataset_col, y='Value', color='black', alpha=0.4)
            
            if analysis_type == "proportion":
                measure_type = "Proportion"
            elif analysis_type == "density_tissue":
                measure_type = "Density (cells/mm² tissue area)"
            else:
                measure_type = "Density (cells/mm² inflammation area)"
            
            g.set_axis_labels('Dataset', f'{measure_type}')
            g.set_titles(col_template='{col_name}')
            
            g.tight_layout()
            
            if save_path:
                g.savefig(os.path.join(save_path, f"patient_{analysis_type}_all_violinplots.png"), 
                        dpi=300, bbox_inches='tight')
                plt.close()
        else:
            print("No categories with non-zero values found for facet grid")
        
    except Exception as e:
        print(f"Error creating facet grid: {str(e)}")
    
    return df

def plot_within_dataset_heatmaps(within_dataset_results, save_path=None, analysis_type="density_inflammation"):
    """
    Create heatmaps showing within-dataset cell type comparisons
    """
    if not within_dataset_results:
        print("No within-dataset results to visualize.")
        return
    
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for use_adjusted in [False, True]:
        p_key = 'adjusted_p_value' if use_adjusted else 'p_value'
        p_type = "BH-Adjusted" if use_adjusted else "Raw"
        
        datasets = list(within_dataset_results.keys())
        n_datasets = len(datasets)
        
        if n_datasets == 0:
            continue
        
        fig_width = min(6 * n_datasets, 20)
        fig_height = max(8, min(12, 4 * n_datasets))
        
        fig, axes = plt.subplots(1, n_datasets, figsize=(fig_width, fig_height))
        if n_datasets == 1:
            axes = [axes]
        
        for idx, dataset in enumerate(datasets):
            results = within_dataset_results[dataset]
            
            cell_types = set()
            for pair in results.keys():
                if isinstance(pair, tuple) and len(pair) == 2:
                    cell_types.update(pair)
            
            cell_types = sorted(list(cell_types))
            n_cell_types = len(cell_types)
            
            if n_cell_types < 2:
                axes[idx].text(0.5, 0.5, f'Insufficient data\nfor {dataset}', 
                              ha='center', va='center', transform=axes[idx].transAxes)
                axes[idx].set_title(f'{dataset}')
                continue
            
            fold_changes = np.full((n_cell_types, n_cell_types), np.nan)
            log10_p_values = np.full((n_cell_types, n_cell_types), np.nan)
            significance = np.full((n_cell_types, n_cell_types), False)
            
            for pair, result in results.items():
                if isinstance(pair, tuple) and len(pair) == 2 and isinstance(result, dict):
                    if 'fold_change' in result and p_key in result:
                        cell_type1, cell_type2 = pair
                        
                        try:
                            i = cell_types.index(cell_type1)
                            j = cell_types.index(cell_type2)
                            
                            fc = result['fold_change']
                            if fc == float('inf'):
                                fc = 1000
                            elif fc == 0:
                                fc = 0.001
                            
                            fold_changes[i, j] = fc
                            fold_changes[j, i] = 1/fc if fc != 0 else 1000
                            
                            p_val = result[p_key]
                            log10_p_val = -np.log10(max(p_val, 1e-10))
                            log10_p_values[i, j] = log10_p_val
                            log10_p_values[j, i] = log10_p_val
                            
                            sig = result['significant']
                            significance[i, j] = sig
                            significance[j, i] = sig
                            
                        except ValueError:
                            continue
            
            np.fill_diagonal(fold_changes, 1)
            np.fill_diagonal(log10_p_values, 0)
            np.fill_diagonal(significance, False)
            
            log2_fc = np.log2(fold_changes)
            log2_fc[~np.isfinite(log2_fc)] = np.nan
            
            masked_log2_fc = np.ma.array(log2_fc, mask=np.isnan(log2_fc))
            
            cmap_fc = plt.cm.RdBu_r
            cmap_fc.set_bad('lightgray')
            
            valid_fc = masked_log2_fc.compressed()
            if len(valid_fc) > 0:
                max_abs_log2_fc = max(2, np.max(np.abs(valid_fc)))
            else:
                max_abs_log2_fc = 2
            norm_fc = plt.Normalize(-max_abs_log2_fc, max_abs_log2_fc)
            
            im = axes[idx].imshow(masked_log2_fc, cmap=cmap_fc, norm=norm_fc, aspect='equal')
            
            axes[idx].set_xticks(np.arange(n_cell_types))
            axes[idx].set_yticks(np.arange(n_cell_types))
            axes[idx].set_xticklabels(cell_types, rotation=45, ha='right')
            axes[idx].set_yticklabels(cell_types)
            
            axes[idx].set_xticks(np.arange(-0.5, n_cell_types, 1), minor=True)
            axes[idx].set_yticks(np.arange(-0.5, n_cell_types, 1), minor=True)
            axes[idx].grid(which='minor', color='black', linestyle='-', linewidth=0.5)
            axes[idx].tick_params(which='minor', size=0)
            
            # IMPROVED ASTERISK PLOTTING WITH DYNAMIC COLORING
            for i in range(n_cell_types):
                for j in range(n_cell_types):
                    if significance[i, j] and i != j:
                        # Determine text color based on background darkness
                        if not np.isnan(log2_fc[i, j]):
                            normalized_value = (log2_fc[i, j] + max_abs_log2_fc) / (2 * max_abs_log2_fc)
                            # Black asterisks for middle 50% (-50% to +50%), white for outer extremes
                            if normalized_value < 0.25 or normalized_value > 0.75:
                                text_color = 'white'
                            else:
                                text_color = 'black'
                        else:
                            text_color = 'black'
                        axes[idx].text(j, i, '*', ha='center', va='center', 
                                     color=text_color, fontsize=40, fontweight='bold')
            
            axes[idx].set_title(f'{dataset}')
            
            if idx == n_datasets - 1:
                cbar = plt.colorbar(im, ax=axes[idx])
                cbar.set_label('log2(Fold Change)')
        
        if analysis_type == "proportion":
            measure_type = "Cell Proportions"
        elif analysis_type == "density_tissue":
            measure_type = "Cell Densities (tissue area)"
        else:
            measure_type = "Cell Densities (inflammation area)"
        
        plt.suptitle(f'Within-Dataset {measure_type} Comparisons ({p_type} P-values)', fontsize=16)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            p_value_type = "adjusted" if use_adjusted else "raw"
            plt.savefig(os.path.join(save_path, f"within_dataset_{analysis_type}_heatmap_{p_value_type}.png"), 
                        dpi=300, bbox_inches='tight')
            plt.close()

def plot_within_dataset_volcano_plots(within_dataset_results, save_path=None, analysis_type="density_inflammation"):
    """
    Create volcano plots for within-dataset cell type comparisons
    """
    if not within_dataset_results:
        print("No within-dataset results to visualize.")
        return
    
    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for dataset, results in within_dataset_results.items():
        if not results:
            print(f"No results for dataset {dataset}")
            continue
        
        for use_adjusted in [False, True]:
            p_key = 'adjusted_p_value' if use_adjusted else 'p_value'
            p_type = "BH-Adjusted" if use_adjusted else "Raw"
            
            comparison_pairs = []
            p_values = []
            adjusted_p_values = []
            fold_changes = []
            
            for pair, result in results.items():
                if isinstance(pair, tuple) and len(pair) == 2 and isinstance(result, dict):
                    if 'p_value' in result and 'adjusted_p_value' in result and 'fold_change' in result:
                        comparison_pairs.append(f"{pair[0]} vs {pair[1]}")
                        p_values.append(result['p_value'])
                        adjusted_p_values.append(result['adjusted_p_value'])
                        fold_changes.append(result['fold_change'])
            
            if not comparison_pairs:
                print(f"No valid data for dataset {dataset}")
                continue
            
            plot_df = pd.DataFrame({
                'Comparison': comparison_pairs,
                'P-Value': p_values,
                'Adjusted P-Value': adjusted_p_values,
                'Fold Change': fold_changes,
                '-log10(P)': -np.log10(p_values),
                '-log10(Adjusted P)': -np.log10(adjusted_p_values),
                'log2(FC)': np.log2([fc if fc != float('inf') else 1000 for fc in fold_changes])
            })
            
            p_col = 'Adjusted P-Value' if use_adjusted else 'P-Value'
            log_p_col = '-log10(Adjusted P)' if use_adjusted else '-log10(P)'
            plot_df = plot_df.sort_values(p_col)
            
            plt.figure(figsize=(12, 8))
            
            colors = []
            for _, row in plot_df.iterrows():
                if row[p_col] < 0.05:
                    if row['Fold Change'] > 1:
                        colors.append('red')
                    else:
                        colors.append('blue')
                else:
                    colors.append('gray')
            
            plt.scatter(plot_df['log2(FC)'], plot_df[log_p_col], c=colors, alpha=0.7, s=80)
            
            plt.axhline(y=-np.log10(0.05), color='black', linestyle='--', alpha=0.5)
            
            plt.axvline(x=np.log2(2), color='black', linestyle='--', alpha=0.5)
            plt.axvline(x=-np.log2(2), color='black', linestyle='--', alpha=0.5)
            
            for _, row in plot_df[plot_df[p_col] < 0.05].iterrows():
                plt.annotate(row['Comparison'], 
                           xy=(row['log2(FC)'], row[log_p_col]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, ha='left', va='bottom',
                           bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3))
            
            plt.xlabel('log2(Fold Change)', fontsize=12)
            plt.ylabel(f'-log10({p_col})', fontsize=12)
            
            if analysis_type == "proportion":
                measure_type = "Proportions"
            elif analysis_type == "density_tissue":
                measure_type = "Densities (tissue area)"
            else:
                measure_type = "Densities (inflammation area)"
            
            plt.title(f'Within-Dataset Cell {measure_type} Comparisons: {dataset}\n({p_type} P-values)', 
                      fontsize=14)
            
            plt.grid(True, alpha=0.3)
            
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, 
                       label='Higher in first cell type'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, 
                       label='Higher in second cell type'),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=10, 
                       label='Not significant')
            ]
            plt.legend(handles=legend_elements, loc='best', fontsize=10)
            
            plt.tight_layout()
            
            if save_path:
                p_value_type = "adjusted" if use_adjusted else "raw"
                plt.savefig(os.path.join(save_path, f"within_dataset_volcano_{analysis_type}_{dataset}_{p_value_type}.png"), 
                            dpi=300, bbox_inches='tight')
                plt.close()

def main():
    """Main function to run the analysis pipeline with inflammation area, tissue area, and proportion comparisons"""
    data_file = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/processed_combined_SAM_scores_filtered_alldataset5_10woMonointerstitial_memcatname3cellcatname23.csv"
    base_path = "/project/mclark/SCAMPI_datasets"
    output_dir = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/MemCat3_inflammation_density_proportion_analysis_results2"
    
    pixels_per_mm2 = 44033550
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Reading data from {data_file}")
    
    df = pd.read_csv(data_file, low_memory=False)
    
    if 'MemCatName3' not in df.columns:
        print("Error: MemCatName3 column not found in the data file")
        return
    
    exclude_categories = ['Structural', 'Others']
    print(f"Excluding categories: {exclude_categories}")
    
    patient_column = None
    for possible_column in ['PatientID', 'Patient', 'Sample', 'SampleID', 'Subject', 'SubjectID']:
        if possible_column in df.columns:
            patient_column = possible_column
            print(f"Found patient/sample identifier column: {patient_column}")
            break
    
    if not patient_column:
        print("No patient/sample identifier column found. Attempting to extract from component names...")
        
        if 'CompName' in df.columns:
            df['PatientID'] = df['CompName'].apply(lambda x: x.split('_')[0] if '_' in str(x) else x)
            patient_column = 'PatientID'
            print(f"Created patient ID column by extracting from component names")
        else:
            print("Error: Cannot determine patient/sample identifiers. Component name column not found.")
            return
    
    comp_to_dataset = {}
    for comp_name in df['CompName'].unique():
        if 'Dataset' in df.columns:
            original_dataset = df[df['CompName'] == comp_name]['Dataset'].iloc[0]
            if original_dataset == "IgG4":
                dataset = "IgG4-RKD"
            elif original_dataset == "AAV":
                dataset = "AAV"
            elif original_dataset == "NK2":
                dataset = "KC"
            else:
                continue
        else:
            if "IgG4" in comp_name:
                dataset = "IgG4-RKD"
            elif "AAV" in comp_name:
                dataset = "AAV"
            elif "NK2" in comp_name:
                dataset = "KC"
            else:
                continue
        
        comp_to_dataset[comp_name] = dataset
    
    # This now returns 4 dataframes
    density_inflammation_df, density_tissue_df, proportion_df, component_df = aggregate_by_patient(
        df, patient_column, comp_to_dataset, pixels_per_mm2, base_path, exclude_categories, output_dir=output_dir
    )
    
    if density_inflammation_df is not None and density_tissue_df is not None and proportion_df is not None:
        print(f"Patient-level density (inflammation) dataframe shape: {density_inflammation_df.shape}")
        print(f"Patient-level density (tissue) dataframe shape: {density_tissue_df.shape}")
        print(f"Patient-level proportion dataframe shape: {proportion_df.shape}")
        print(f"Component-level dataframe shape: {component_df.shape}")
        
        print("\n" + "="*50)
        print("CREATING INFLAMMATION RATIO SCATTER PLOTS")
        print("="*50)
        plot_inflammation_ratio_vs_density(component_df, save_path=output_dir)
        
        print("\n" + "="*50)
        print("COMPARING TOTAL TISSUE DENSITY ACROSS DATASETS")
        print("="*50)
        compare_tissue_density_across_datasets(density_inflammation_df, save_path=output_dir)
        
        # ========== INFLAMMATION AREA DENSITY ANALYSIS ==========
        print("\n" + "="*70)
        print("PERFORMING CROSS-DATASET COMPARISONS - INFLAMMATION AREA DENSITIES")
        print("="*70)
        
        density_inflammation_pairwise_results = perform_pairwise_comparisons(density_inflammation_df)
        
        plot_pairwise_volcano_plots(density_inflammation_pairwise_results, save_path=output_dir, analysis_type="density_inflammation")
        plot_integrated_heatmap(density_inflammation_pairwise_results, save_path=output_dir, analysis_type="density_inflammation")
        plot_stacked_bar_by_dataset(density_inflammation_df, save_path=output_dir, analysis_type="density_inflammation")
        plot_grouped_violinplots(density_inflammation_df, save_path=output_dir, analysis_type="density_inflammation")
        
        density_inflammation_summary = create_summary_report(density_inflammation_pairwise_results, density_inflammation_df, 
                                                             save_path=output_dir, analysis_type="density_inflammation")
        
        print("\n" + "="*70)
        print("PERFORMING WITHIN-DATASET COMPARISONS - INFLAMMATION AREA DENSITIES")
        print("="*70)
        
        density_inflammation_within_results = perform_within_dataset_comparisons(density_inflammation_df)
        
        plot_within_dataset_heatmaps(density_inflammation_within_results, save_path=output_dir, analysis_type="density_inflammation")
        plot_within_dataset_volcano_plots(density_inflammation_within_results, save_path=output_dir, analysis_type="density_inflammation")
        plot_within_dataset_violin_plots(density_inflammation_df, save_path=output_dir, analysis_type="density_inflammation")
        
        density_inflammation_within_summary = create_within_dataset_summary_report(density_inflammation_within_results, 
                                                                                   density_inflammation_df, 
                                                                                   save_path=output_dir, 
                                                                                   analysis_type="density_inflammation")
        
        # ========== TISSUE AREA DENSITY ANALYSIS ==========
        print("\n" + "="*70)
        print("PERFORMING CROSS-DATASET COMPARISONS - TISSUE AREA DENSITIES")
        print("="*70)
        
        density_tissue_pairwise_results = perform_pairwise_comparisons(density_tissue_df)
        
        plot_pairwise_volcano_plots(density_tissue_pairwise_results, save_path=output_dir, analysis_type="density_tissue")
        plot_integrated_heatmap(density_tissue_pairwise_results, save_path=output_dir, analysis_type="density_tissue")
        plot_stacked_bar_by_dataset(density_tissue_df, save_path=output_dir, analysis_type="density_tissue")
        plot_grouped_violinplots(density_tissue_df, save_path=output_dir, analysis_type="density_tissue")
        
        density_tissue_summary = create_summary_report(density_tissue_pairwise_results, density_tissue_df, 
                                                       save_path=output_dir, analysis_type="density_tissue")
        
        print("\n" + "="*70)
        print("PERFORMING WITHIN-DATASET COMPARISONS - TISSUE AREA DENSITIES")
        print("="*70)
        
        density_tissue_within_results = perform_within_dataset_comparisons(density_tissue_df)
        
        plot_within_dataset_heatmaps(density_tissue_within_results, save_path=output_dir, analysis_type="density_tissue")
        plot_within_dataset_volcano_plots(density_tissue_within_results, save_path=output_dir, analysis_type="density_tissue")
        plot_within_dataset_violin_plots(density_tissue_df, save_path=output_dir, analysis_type="density_tissue")
        
        density_tissue_within_summary = create_within_dataset_summary_report(density_tissue_within_results, 
                                                                             density_tissue_df, 
                                                                             save_path=output_dir, 
                                                                             analysis_type="density_tissue")
        
        # ========== PROPORTION ANALYSIS ==========
        print("\n" + "="*70)
        print("PERFORMING CROSS-DATASET COMPARISONS - PROPORTIONS")
        print("="*70)
        
        proportion_pairwise_results = perform_pairwise_comparisons(proportion_df)
        
        plot_pairwise_volcano_plots(proportion_pairwise_results, save_path=output_dir, analysis_type="proportion")
        plot_integrated_heatmap(proportion_pairwise_results, save_path=output_dir, analysis_type="proportion")
        plot_stacked_bar_by_dataset(proportion_df, save_path=output_dir, analysis_type="proportion")
        plot_grouped_violinplots(proportion_df, save_path=output_dir, analysis_type="proportion")
        
        proportion_summary = create_summary_report(proportion_pairwise_results, proportion_df, 
                                                   save_path=output_dir, analysis_type="proportion")
        
        print("\n" + "="*70)
        print("PERFORMING WITHIN-DATASET COMPARISONS - PROPORTIONS")
        print("="*70)
        
        proportion_within_results = perform_within_dataset_comparisons(proportion_df)
        
        plot_within_dataset_heatmaps(proportion_within_results, save_path=output_dir, analysis_type="proportion")
        plot_within_dataset_volcano_plots(proportion_within_results, save_path=output_dir, analysis_type="proportion")
        plot_within_dataset_violin_plots(proportion_df, save_path=output_dir, analysis_type="proportion")
        
        proportion_within_summary = create_within_dataset_summary_report(proportion_within_results, 
                                                                         proportion_df, 
                                                                         save_path=output_dir, 
                                                                         analysis_type="proportion")
        
        # Save final dataframes
        density_inflammation_df.to_csv(os.path.join(output_dir, "patient_cell_density_inflammation_data.csv"), index=False)
        density_tissue_df.to_csv(os.path.join(output_dir, "patient_cell_density_tissue_data.csv"), index=False)
        proportion_df.to_csv(os.path.join(output_dir, "patient_cell_proportion_data.csv"), index=False)
        
        print(f"\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print(f"Results saved to {output_dir}")
        print("\nGenerated files include:")
        print("\n1. SCATTER PLOTS:")
        print("   - Inflammation ratio vs density (tissue area)")
        print("   - Inflammation ratio vs density (inflammation area)")
        print("\n2. TOTAL DENSITY COMPARISON:")
        print("   - Tissue density comparison summary CSV")
        print("   - Tissue density violin plot")
        print("\n3. INFLAMMATION AREA DENSITY ANALYSIS:")
        print("   - Cross-dataset volcano plots (raw and BH-adjusted p-values)")
        print("   - Cross-dataset heatmaps (raw and BH-adjusted p-values)")
        print("   - Cross-dataset violin plots and stacked bar charts")
        print("   - Cross-dataset summary report CSV")
        print("   - Within-dataset volcano plots (raw and BH-adjusted p-values)")
        print("   - Within-dataset heatmaps (raw and BH-adjusted p-values)")
        print("   - Within-dataset violin plots")
        print("   - Within-dataset summary report CSV")
        print("\n4. TISSUE AREA DENSITY ANALYSIS:")
        print("   - Cross-dataset volcano plots (raw and BH-adjusted p-values)")
        print("   - Cross-dataset heatmaps (raw and BH-adjusted p-values)")
        print("   - Cross-dataset violin plots and stacked bar charts")
        print("   - Cross-dataset summary report CSV")
        print("   - Within-dataset volcano plots (raw and BH-adjusted p-values)")
        print("   - Within-dataset heatmaps (raw and BH-adjusted p-values)")
        print("   - Within-dataset violin plots")
        print("   - Within-dataset summary report CSV")
        print("\n5. PROPORTION ANALYSIS:")
        print("   - Cross-dataset volcano plots (raw and BH-adjusted p-values)")
        print("   - Cross-dataset heatmaps (raw and BH-adjusted p-values)")
        print("   - Cross-dataset violin plots and stacked bar charts")
        print("   - Cross-dataset summary report CSV")
        print("   - Within-dataset volcano plots (raw and BH-adjusted p-values)")
        print("   - Within-dataset heatmaps (raw and BH-adjusted p-values)")
        print("   - Within-dataset violin plots")
        print("   - Within-dataset summary report CSV")
        print("\n6. DATA FILES:")
        print("   - Patient-level combined data CSV")
        print("   - Component-level inflammation data CSV")
        print("   - Patient-level density (inflammation area) data CSV")
        print("   - Patient-level density (tissue area) data CSV")
        print("   - Patient-level proportion data CSV")
        
        return {
            'density_inflammation_df': density_inflammation_df,
            'density_tissue_df': density_tissue_df,
            'proportion_df': proportion_df,
            'component_df': component_df,
            'cross_dataset_results': {
                'inflammation': density_inflammation_pairwise_results,
                'tissue': density_tissue_pairwise_results,
                'proportion': proportion_pairwise_results
            },
            'within_dataset_results': {
                'inflammation': density_inflammation_within_results,
                'tissue': density_tissue_within_results,
                'proportion': proportion_within_results
            }
        }
    else:
        print("Patient-level aggregation failed. Check data structure and path.")
        return None


if __name__ == "__main__":
    main()
