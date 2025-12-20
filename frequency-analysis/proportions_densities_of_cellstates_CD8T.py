import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from collections import defaultdict
from scipy import stats
from scipy.stats import chi2_contingency, kruskal
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Define the specific order of CD8T cell types
CELL_TYPE_ORDER = [
    'NaiveCD8T', 'OtherCD8T', 'TIGIT+CD8T', 'CD8Trm', 'CD103+CD8Trm', 
    'GZMA+GZMK+CD8CTL', 'GZMB+CD8CTL', 'CD8Tcm-Tem', 'CD8Treg', 'TIGIT+CD8Treg', 
    'CD103+CD8Treg', 'CD69+CD8Treg', 'CD8Trm-Treg', 'GZMA+GZMK+CD8Treg', 'GZMB+CD8Treg', 
    'CD8+iTreg', 'Tc1', 'TIGIT+Tc1', 'Tc1-EM', 'GZMA+GZMK+Tc1-EM', 
    'GZMB+Tc1-EM', 'CD103+Tc1-like', 'Tbet+CD8Trm', 'Tbet+CD103+CD8Trm', 
    'MKI+CD8T', 'MKI+GZMA+GZMK+CD8CTL', 'MKI+GZMB+CD8CTL', 'Apoptotic'
]

# Define the specific order of states (for x-axis: left to right)
STATE_ORDER = ['Other', 'Stem-like', 'ProgenitorExhausted', 'Exhausted']

# Define the specific order of datasets (for y-axis: top to bottom)
DATASET_ORDER = ['IgG4-RKD', 'AAV', 'KC']

# Dataset name mapping from original to display names
DATASET_MAPPING = {
    'IgG4': 'IgG4-RKD',
    'AAV': 'AAV', 
    'NK2': 'KC'
}

# Define custom color palette for states
STATE_COLORS = {
    'Other': '#E41A1C',
    'Stem-like': '#377EB8', 
    'ProgenitorExhausted': '#4DAF4A',
    'Exhausted': '#984EA3'
}

def map_dataset_names(df):
    """
    Map original dataset names to display names
    """
    df_mapped = df.copy()
    df_mapped['Dataset_Display'] = df_mapped['Dataset'].map(DATASET_MAPPING)
    return df_mapped

def get_state_colors(states):
    """
    Get colors for the given states in the correct order
    """
    return [STATE_COLORS.get(state, '#999999') for state in states]

def calculate_tissue_area(dataset, comp_name, base_path="/project/mclark/SCAMPI_datasets"):
    """
    Calculate the tissue area from tissue mask
    Args:
        dataset: Dataset name (original name like IgG4, AAV, NK2)
        comp_name: Component name
        base_path: Base path to the datasets
    Returns:
        tissue_area: Number of pixels in the tissue area
    """
    # Map display names back to original dataset names
    original_dataset = dataset
    if dataset == "IgG4-RKD":
        original_dataset = "IgG4"
    elif dataset == "KC":
        original_dataset = "NK2"
    elif dataset == "AAV":
        original_dataset = "AAV"
    
    # Try different possible tissue mask paths
    tissue_mask_path1 = f'{base_path}/{original_dataset}/tissue_composite_masks/DAPI_UV_1_{original_dataset}_{comp_name}.tif'
    tissue_mask_path2 = f'{base_path}/{original_dataset}/tissue_composite_masks/DAPI_UV_2_{original_dataset}_{comp_name}.tif'
    tissue_mask_path1_alt = f'{base_path}/{original_dataset}/tissue_composite_masks/DAPI_UV_1_{original_dataset}_{comp_name}.tiff'
    tissue_mask_path2_alt = f'{base_path}/{original_dataset}/tissue_composite_masks/DAPI_UV_2_{original_dataset}_{comp_name}.tiff'
    
    # Increase PIL's maximum image size limit
    Image.MAX_IMAGE_PIXELS = 1100000000
    
    tissue_mask_path = None
    for path in [tissue_mask_path1, tissue_mask_path2, tissue_mask_path1_alt, tissue_mask_path2_alt]:
        if os.path.exists(path):
            tissue_mask_path = path
            break
    
    if not tissue_mask_path:
        print(f"No tissue mask found for {comp_name}. Tried paths: {[tissue_mask_path1, tissue_mask_path2, tissue_mask_path1_alt, tissue_mask_path2_alt]}")
        return None
    
    try:
        # Load tissue mask
        tissue_mask = Image.open(tissue_mask_path)
        tissue_array = np.array(tissue_mask) > 0
        tissue_area = np.sum(tissue_array)
        
        print(f"Tissue area for {comp_name}: {tissue_area} pixels")
        print(f"Using tissue mask: {tissue_mask_path}")
        
        return tissue_area
        
    except Exception as e:
        print(f"Error loading tissue mask {tissue_mask_path}: {str(e)}")
        return None

def calculate_inflammation_area(dataset, comp_name):
    """
    Calculate the inflammation area from inflammation masks
    Args:
        dataset: Dataset name (original name like IgG4, AAV, NK2)
        comp_name: Component name
    Returns:
        inflammation_area: Number of pixels in the inflammation area
    """
    # The inflammation masks are in a fixed location
    inflammation_base_path = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/inflammation_masks_output"
    
    # Try different possible directory name constructions
    possible_dirs = [
        f"{dataset}_{comp_name}",
        comp_name,
        f"{dataset}_{comp_name}".replace("-", ""),
        f"{dataset.replace('-', '')}_{comp_name}",
    ]
    
    # Increase PIL's maximum image size limit
    Image.MAX_IMAGE_PIXELS = 1100000000
    
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
        # Load inflammation mask
        inflammation_mask = Image.open(inflammation_mask_path)
        inflammation_array = np.array(inflammation_mask)
        
        # Count pixels with value 255 (inflammation area)
        inflammation_area = np.sum(inflammation_array == 255)
        
        print(f"Inflammation area for {comp_name}: {inflammation_area} pixels")
        print(f"Using inflammation mask: {inflammation_mask_path}")
        
        return inflammation_area
        
    except Exception as e:
        print(f"Error loading inflammation mask {inflammation_mask_path}: {str(e)}")
        return None

def calculate_cell_densities_by_state(df, output_dir=".", exclude_cell_types=None):
    """
    Calculate cell densities for each state across datasets using inflammation area at PATIENT LEVEL
    """
    if exclude_cell_types is None:
        exclude_cell_types = []
    
    print("Calculating patient-level cell densities by state using inflammation area...")
    
    if 'Patient' not in df.columns:
        print("Warning: 'Patient' column not found. Cannot perform patient-level density analysis.")
        return None
    
    pixels_per_mm2 = 44033550
    
    if exclude_cell_types:
        df = df[~df['CellCatName3'].isin(exclude_cell_types)].copy()
    
    component_data = []
    
    for dataset in df['Dataset'].unique():
        dataset_df = df[df['Dataset'] == dataset]
        components = dataset_df['CompName'].unique()
        
        for comp_name in components:
            comp_df = dataset_df[dataset_df['CompName'] == comp_name]
            
            if len(comp_df['Patient'].unique()) != 1:
                print(f"Warning: Multiple patients found for component {comp_name}, skipping")
                continue
            
            patient_id = comp_df['Patient'].iloc[0]
            inflammation_area = calculate_inflammation_area(dataset, comp_name)
            
            if inflammation_area is None or inflammation_area == 0:
                print(f"Skipping {comp_name} - no valid inflammation area")
                continue
            
            area_mm2 = inflammation_area / pixels_per_mm2
            state_counts = comp_df['State'].value_counts()
            
            for state in STATE_ORDER:
                count = state_counts.get(state, 0)
                component_data.append({
                    'Dataset': dataset,
                    'Dataset_Display': DATASET_MAPPING.get(dataset, dataset),
                    'Patient': patient_id,
                    'CompName': comp_name,
                    'State': state,
                    'Cell_Count': count,
                    'Inflammation_Area_MM2': area_mm2
                })
    
    if not component_data:
        print("No valid component data found for density calculation")
        return None
    
    component_df = pd.DataFrame(component_data)
    
    patient_aggregated = component_df.groupby(['Dataset', 'Dataset_Display', 'Patient', 'State']).agg({
        'Cell_Count': 'sum',
        'Inflammation_Area_MM2': 'sum'
    }).reset_index()
    
    patient_aggregated['Density'] = patient_aggregated['Cell_Count'] / patient_aggregated['Inflammation_Area_MM2']
    patient_aggregated['Density'] = patient_aggregated['Density'].fillna(0)
    
    density_summary = patient_aggregated.groupby(['Dataset', 'Dataset_Display', 'State'])['Density'].agg([
        'count', 'mean', 'std', 'median', 'min', 'max'
    ]).reset_index()
    density_summary.columns = ['Dataset', 'Dataset_Display', 'State', 'N_Patients', 'Mean_Density', 'Std_Density', 
                              'Median_Density', 'Min_Density', 'Max_Density']
    density_summary = density_summary.fillna(0)
    
    try:
        patient_path = os.path.join(output_dir, 'cd8t_patient_level_densities_by_state.csv')
        patient_aggregated.to_csv(patient_path, index=False)
        print(f"Patient-level densities saved to: {patient_path}")
        
        summary_path = os.path.join(output_dir, 'cd8t_state_densities_summary_by_dataset.csv')
        density_summary.to_csv(summary_path, index=False)
        print(f"Density summary statistics saved to: {summary_path}")
    except Exception as e:
        print(f"Warning: Could not save density data: {e}")
    
    print(f"Patient-level density calculation completed.")
    print(f"Patient-level data shape: {patient_aggregated.shape}")
    print(f"Summary data shape: {density_summary.shape}")
    
    return density_summary, patient_aggregated

def calculate_cell_densities_by_state_tissue(df, output_dir=".", exclude_cell_types=None, base_path="/project/mclark/SCAMPI_datasets"):
    """
    Calculate cell densities for each state across datasets using TISSUE area at PATIENT LEVEL
    """
    if exclude_cell_types is None:
        exclude_cell_types = []
    
    print("Calculating patient-level cell densities by state using TISSUE area...")
    
    if 'Patient' not in df.columns:
        print("Warning: 'Patient' column not found. Cannot perform patient-level density analysis.")
        return None
    
    pixels_per_mm2 = 44033550
    
    if exclude_cell_types:
        df = df[~df['CellCatName3'].isin(exclude_cell_types)].copy()
    
    component_data = []
    
    for dataset in df['Dataset'].unique():
        dataset_df = df[df['Dataset'] == dataset]
        components = dataset_df['CompName'].unique()
        
        for comp_name in components:
            comp_df = dataset_df[dataset_df['CompName'] == comp_name]
            
            if len(comp_df['Patient'].unique()) != 1:
                print(f"Warning: Multiple patients found for component {comp_name}, skipping")
                continue
            
            patient_id = comp_df['Patient'].iloc[0]
            tissue_area = calculate_tissue_area(dataset, comp_name, base_path)
            
            if tissue_area is None or tissue_area == 0:
                print(f"Skipping {comp_name} - no valid tissue area")
                continue
            
            area_mm2 = tissue_area / pixels_per_mm2
            state_counts = comp_df['State'].value_counts()
            
            for state in STATE_ORDER:
                count = state_counts.get(state, 0)
                component_data.append({
                    'Dataset': dataset,
                    'Dataset_Display': DATASET_MAPPING.get(dataset, dataset),
                    'Patient': patient_id,
                    'CompName': comp_name,
                    'State': state,
                    'Cell_Count': count,
                    'Tissue_Area_MM2': area_mm2
                })
    
    if not component_data:
        print("No valid component data found for tissue density calculation")
        return None
    
    component_df = pd.DataFrame(component_data)
    
    patient_aggregated = component_df.groupby(['Dataset', 'Dataset_Display', 'Patient', 'State']).agg({
        'Cell_Count': 'sum',
        'Tissue_Area_MM2': 'sum'
    }).reset_index()
    
    patient_aggregated['Density'] = patient_aggregated['Cell_Count'] / patient_aggregated['Tissue_Area_MM2']
    patient_aggregated['Density'] = patient_aggregated['Density'].fillna(0)
    
    density_summary = patient_aggregated.groupby(['Dataset', 'Dataset_Display', 'State'])['Density'].agg([
        'count', 'mean', 'std', 'median', 'min', 'max'
    ]).reset_index()
    density_summary.columns = ['Dataset', 'Dataset_Display', 'State', 'N_Patients', 'Mean_Density', 'Std_Density', 
                              'Median_Density', 'Min_Density', 'Max_Density']
    density_summary = density_summary.fillna(0)
    
    try:
        patient_path = os.path.join(output_dir, 'cd8t_patient_level_densities_by_state_tissue.csv')
        patient_aggregated.to_csv(patient_path, index=False)
        print(f"Patient-level tissue densities saved to: {patient_path}")
        
        summary_path = os.path.join(output_dir, 'cd8t_state_densities_summary_by_dataset_tissue.csv')
        density_summary.to_csv(summary_path, index=False)
        print(f"Tissue density summary statistics saved to: {summary_path}")
    except Exception as e:
        print(f"Warning: Could not save tissue density data: {e}")
    
    print(f"Patient-level tissue density calculation completed.")
    
    return density_summary, patient_aggregated

def create_cell_density_heatmap(density_summary, patient_densities=None, output_dir=".", exclude_cell_types=None):
    """
    Create heatmap comparing cell densities of each state across datasets (using patient-level data)
    """
    if density_summary is None or density_summary.empty:
        print("No density data available for heatmap creation")
        return
    
    print("Creating patient-level cell density heatmap...")
    
    pivot_density = density_summary.pivot(index='Dataset_Display', columns='State', values='Mean_Density')
    pivot_density = pivot_density.reindex(columns=STATE_ORDER, fill_value=0)
    pivot_density = pivot_density.reindex(DATASET_ORDER, fill_value=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(pivot_density, 
                annot=False, 
                fmt='.2f', 
                cmap='viridis',
                cbar_kws={'label': 'Mean Density (cells/mm² inflammation area)'},
                linewidths=0.5,
                linecolor='white',
                square=True,
                ax=ax)
    
    plt.title('CD8T Cell State Densities Across Datasets (Patient-Level)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Cell State', fontsize=14)
    plt.ylabel('Dataset', fontsize=14)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12, rotation=0)
    
    plt.tight_layout()
    
    try:
        output_path = os.path.join(output_dir, 'cd8t_state_density_heatmap_patient_level.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved patient-level cell density heatmap: {output_path}")
    except Exception as e:
        print(f"Warning: Could not save density heatmap: {e}")
    
    plt.close()
    
    print("\nPatient-Level Cell Density Summary (cells/mm² inflammation area):")
    print("=" * 70)
    for state in STATE_ORDER:
        state_data = density_summary[density_summary['State'] == state]
        print(f"\n{state}:")
        for _, row in state_data.iterrows():
            print(f"  {row['Dataset_Display']}: {row['Mean_Density']:.2f} ± {row['Std_Density']:.2f} "
                  f"(median: {row['Median_Density']:.2f}, n={row['N_Patients']} patients)")
    
    if patient_densities is not None and not patient_densities.empty:
        fig2, ax2 = plt.subplots(figsize=(14, 8))
        
        plot_data = patient_densities.copy()
        plot_data['State'] = pd.Categorical(plot_data['State'], categories=STATE_ORDER, ordered=True)
        
        sns.violinplot(data=plot_data, x='State', y='Density', hue='Dataset_Display', 
                      order=STATE_ORDER, hue_order=DATASET_ORDER, ax=ax2)
        
        ax2.set_title('Distribution of Patient-Level Cell State Densities', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Cell State', fontsize=14)
        ax2.set_ylabel('Density (cells/mm² inflammation area)', fontsize=14)
        ax2.tick_params(axis='both', labelsize=12)
        ax2.legend(title='Dataset', fontsize=12)
        
        plt.tight_layout()
        
        try:
            output_path2 = os.path.join(output_dir, 'cd8t_patient_density_distributions.png')
            plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved patient-level density distributions: {output_path2}")
        except Exception as e:
            print(f"Warning: Could not save density distributions plot: {e}")
        
        plt.close()
        
        print(f"\nSTATISTICAL ANALYSIS OF PATIENT-LEVEL DENSITIES:")
        print("=" * 60)
        
        for state in STATE_ORDER:
            state_data = plot_data[plot_data['State'] == state]
            if len(state_data) < 3:
                continue
                
            datasets_in_state = state_data['Dataset_Display'].unique()
            if len(datasets_in_state) < 2:
                continue
                
            print(f"\n{state} State:")
            
            groups = [state_data[state_data['Dataset_Display'] == dataset]['Density'].values 
                     for dataset in datasets_in_state]
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) >= 2:
                try:
                    stat, p_val = kruskal(*groups)
                    print(f"  Kruskal-Wallis test: H = {stat:.4f}, p = {p_val:.4f}")
                    if p_val < 0.05:
                        print(f"  Result: Significant differences between datasets (p < 0.05)")
                    else:
                        print(f"  Result: No significant differences between datasets (p >= 0.05)")
                except Exception as e:
                    print(f"  Error in statistical test: {e}")
            
            for dataset in datasets_in_state:
                n_patients = len(state_data[state_data['Dataset_Display'] == dataset])
                mean_density = state_data[state_data['Dataset_Display'] == dataset]['Density'].mean()
                print(f"    {dataset}: n={n_patients} patients, mean={mean_density:.2f}")
        
        return density_summary, patient_densities
    
    return density_summary

def create_cell_density_heatmap_tissue(density_summary, patient_densities=None, output_dir=".", exclude_cell_types=None):
    """
    Create heatmap comparing cell densities of each state across datasets (using TISSUE area at patient-level)
    """
    if density_summary is None or density_summary.empty:
        print("No tissue density data available for heatmap creation")
        return
    
    print("Creating patient-level cell density heatmap using TISSUE area...")
    
    pivot_density = density_summary.pivot(index='Dataset_Display', columns='State', values='Mean_Density')
    pivot_density = pivot_density.reindex(columns=STATE_ORDER, fill_value=0)
    pivot_density = pivot_density.reindex(DATASET_ORDER, fill_value=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.heatmap(pivot_density, 
                annot=False, 
                fmt='.2f', 
                cmap='viridis',
                cbar_kws={'label': 'Mean Density (cells/mm² tissue area)'},
                linewidths=0.5,
                linecolor='white',
                square=True,
                ax=ax)
    
    plt.title('CD8T Cell State Densities Across Datasets (Patient-Level, Tissue Area)', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Cell State', fontsize=14)
    plt.ylabel('Dataset', fontsize=14)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12, rotation=0)
    
    plt.tight_layout()
    
    try:
        output_path = os.path.join(output_dir, 'cd8t_state_density_heatmap_patient_level_tissue.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved patient-level cell density heatmap (tissue area): {output_path}")
    except Exception as e:
        print(f"Warning: Could not save tissue density heatmap: {e}")
    
    plt.close()
    
    print("\nPatient-Level Cell Density Summary (cells/mm² tissue area):")
    print("=" * 70)
    for state in STATE_ORDER:
        state_data = density_summary[density_summary['State'] == state]
        print(f"\n{state}:")
        for _, row in state_data.iterrows():
            print(f"  {row['Dataset_Display']}: {row['Mean_Density']:.2f} ± {row['Std_Density']:.2f} "
                  f"(median: {row['Median_Density']:.2f}, n={row['N_Patients']} patients)")
    
    if patient_densities is not None and not patient_densities.empty:
        fig2, ax2 = plt.subplots(figsize=(14, 8))
        
        plot_data = patient_densities.copy()
        plot_data['State'] = pd.Categorical(plot_data['State'], categories=STATE_ORDER, ordered=True)
        
        sns.violinplot(data=plot_data, x='State', y='Density', hue='Dataset_Display', 
                      order=STATE_ORDER, hue_order=DATASET_ORDER, ax=ax2)
        
        ax2.set_title('Distribution of Patient-Level Cell State Densities (Tissue Area)', fontsize=16, fontweight='bold')
        ax2.set_xlabel('Cell State', fontsize=14)
        ax2.set_ylabel('Density (cells/mm² tissue area)', fontsize=14)
        ax2.tick_params(axis='both', labelsize=12)
        ax2.legend(title='Dataset', fontsize=12)
        
        plt.tight_layout()
        
        try:
            output_path2 = os.path.join(output_dir, 'cd8t_patient_density_distributions_tissue.png')
            plt.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved patient-level tissue density distributions: {output_path2}")
        except Exception as e:
            print(f"Warning: Could not save tissue density distributions plot: {e}")
        
        plt.close()
        
        print(f"\nSTATISTICAL ANALYSIS OF PATIENT-LEVEL TISSUE DENSITIES:")
        print("=" * 60)
        
        for state in STATE_ORDER:
            state_data = plot_data[plot_data['State'] == state]
            if len(state_data) < 3:
                continue
                
            datasets_in_state = state_data['Dataset_Display'].unique()
            if len(datasets_in_state) < 2:
                continue
                
            print(f"\n{state} State:")
            
            groups = [state_data[state_data['Dataset_Display'] == dataset]['Density'].values 
                     for dataset in datasets_in_state]
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) >= 2:
                try:
                    stat, p_val = kruskal(*groups)
                    print(f"  Kruskal-Wallis test: H = {stat:.4f}, p = {p_val:.4f}")
                    if p_val < 0.05:
                        print(f"  Result: Significant differences between datasets (p < 0.05)")
                    else:
                        print(f"  Result: No significant differences between datasets (p >= 0.05)")
                except Exception as e:
                    print(f"  Error in statistical test: {e}")
            
            for dataset in datasets_in_state:
                n_patients = len(state_data[state_data['Dataset_Display'] == dataset])
                mean_density = state_data[state_data['Dataset_Display'] == dataset]['Density'].mean()
                print(f"    {dataset}: n={n_patients} patients, mean={mean_density:.2f}")
        
        return density_summary, patient_densities
    
    return density_summary

def load_and_process_data(file_path):
    """
    Load the data and perform initial processing
    """
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def filter_cd8t_states(df, exclude_cell_types=None):
    """
    Filter data for the specific CD8T cell states of interest, including 'Other'
    """
    if exclude_cell_types is None:
        exclude_cell_types = []
    
    if exclude_cell_types:
        df = df[~df['CellCatName3'].isin(exclude_cell_types)].copy()
        print(f"Excluded cell types: {exclude_cell_types}")
    
    df_copy = df.copy()
    df_copy['State'] = 'Other'
    
    print(f"\nDEBUG: Sample CellCatName values:")
    sample_cellcatnames = df_copy['CellCatName'].value_counts().head(20)
    print(sample_cellcatnames)
    
    stem_patterns = ['Stem-like', 'stem-like', 'Stemlike', 'stemlike']
    for pattern in stem_patterns:
        mask = df_copy['CellCatName'].str.contains(pattern, case=False, na=False, regex=False)
        count_matched = mask.sum()
        if count_matched > 0:
            print(f"Found {count_matched} cells matching Stem-like pattern '{pattern}'")
            df_copy.loc[mask, 'State'] = 'Stem-like'
    
    progenitor_patterns = ['ProgenitorExhausted', 'progenitorexhausted', 'Progenitor-Exhausted', 'progenitor-exhausted']
    for pattern in progenitor_patterns:
        mask = df_copy['CellCatName'].str.contains(pattern, case=False, na=False, regex=False)
        count_matched = mask.sum()
        if count_matched > 0:
            print(f"Found {count_matched} cells matching ProgenitorExhausted pattern '{pattern}'")
            df_copy.loc[mask, 'State'] = 'ProgenitorExhausted'
    
    exhausted_patterns = ['Exhausted', 'exhausted']
    for pattern in exhausted_patterns:
        mask = (df_copy['CellCatName'].str.contains(pattern, case=False, na=False, regex=False)) & (df_copy['State'] == 'Other')
        count_matched = mask.sum()
        if count_matched > 0:
            print(f"Found {count_matched} cells matching Exhausted pattern '{pattern}' (excluding ProgenitorExhausted)")
            df_copy.loc[mask, 'State'] = 'Exhausted'
    
    print(f"\nFiltered data shape: {df_copy.shape}")
    print(f"Unique datasets: {df_copy['Dataset'].value_counts()}")
    print(f"Unique CD8T cell types: {df_copy['CellCatName3'].nunique()}")
    print(f"States found: {df_copy['State'].value_counts()}")
    
    print(f"\nDEBUG: CellCatName patterns by assigned state:")
    for state in STATE_ORDER:
        if state in df_copy['State'].values:
            state_cellcatnames = df_copy[df_copy['State'] == state]['CellCatName'].value_counts().head(10)
            print(f"\n{state} state examples:")
            print(state_cellcatnames)
    
    return df_copy

def sort_cell_types(cell_types):
    """
    Sort cell types according to the predefined order
    """
    order_map = {cell_type: i for i, cell_type in enumerate(CELL_TYPE_ORDER)}
    sorted_types = sorted(cell_types, key=lambda x: order_map.get(x, len(CELL_TYPE_ORDER)))
    return sorted_types

def sort_states_for_plotting(states):
    """
    Sort states according to the predefined order for consistent plotting
    """
    order_map = {state: i for i, state in enumerate(STATE_ORDER)}
    sorted_states = sorted(states, key=lambda x: order_map.get(x, len(STATE_ORDER)))
    return sorted_states

def calculate_proportions_by_dataset(df):
    """
    Calculate proportions of each state per CD8T cell type, separated by dataset and patient
    """
    df = map_dataset_names(df)
    
    if 'Patient' not in df.columns:
        print("Warning: 'Patient' column not found. Calculating proportions at dataset level.")
        counts = df.groupby(['Dataset', 'Dataset_Display', 'CellCatName3', 'State']).size().reset_index(name='Count')
        totals = df.groupby(['Dataset', 'Dataset_Display', 'CellCatName3']).size().reset_index(name='Total')
        proportions = counts.merge(totals, on=['Dataset', 'Dataset_Display', 'CellCatName3'])
        proportions['Proportion'] = proportions['Count'] / proportions['Total']
        proportions['Percentage'] = proportions['Proportion'] * 100
        return proportions, counts, totals
    
    print("Calculating proportions at patient level...")
    
    patient_counts = df.groupby(['Dataset', 'Dataset_Display', 'Patient', 'CellCatName3', 'State']).size().reset_index(name='Count')
    patient_totals = df.groupby(['Dataset', 'Dataset_Display', 'Patient', 'CellCatName3']).size().reset_index(name='Total')
    patient_proportions = patient_counts.merge(patient_totals, on=['Dataset', 'Dataset_Display', 'Patient', 'CellCatName3'])
    patient_proportions['Proportion'] = patient_proportions['Count'] / patient_proportions['Total']
    patient_proportions['Percentage'] = patient_proportions['Proportion'] * 100
    
    print(f"Patient-level data shape: {patient_proportions.shape}")
    print(f"Number of patients per dataset:")
    for dataset in df['Dataset_Display'].unique():
        n_patients = df[df['Dataset_Display'] == dataset]['Patient'].nunique()
        print(f"  {dataset}: {n_patients} patients")
    
    dataset_counts = patient_counts.groupby(['Dataset', 'Dataset_Display', 'CellCatName3', 'State'])['Count'].sum().reset_index()
    dataset_totals = patient_totals.groupby(['Dataset', 'Dataset_Display', 'CellCatName3'])['Total'].sum().reset_index()
    
    dataset_summary = dataset_counts.merge(dataset_totals, on=['Dataset', 'Dataset_Display', 'CellCatName3'])
    dataset_summary['Proportion'] = dataset_summary['Count'] / dataset_summary['Total']
    dataset_summary['Percentage'] = dataset_summary['Proportion'] * 100
    
    patient_stats = patient_proportions.groupby(['Dataset', 'Dataset_Display', 'CellCatName3', 'State'])['Percentage'].agg([
        'mean', 'std', 'count'
    ]).reset_index()
    patient_stats.columns = ['Dataset', 'Dataset_Display', 'CellCatName3', 'State', 'Patient_Mean_Percentage', 'Patient_Std_Percentage', 'N_patients']
    
    dataset_summary = dataset_summary.merge(patient_stats, on=['Dataset', 'Dataset_Display', 'CellCatName3', 'State'], how='left')
    
    print(f"Dataset summary shape: {dataset_summary.shape}")
    
    return dataset_summary, patient_counts, patient_totals, patient_proportions

def create_dataset_specific_visualizations(proportions, output_dir=".", exclude_cell_types=None):
    """
    Create separate visualizations for each dataset - one figure per plot type
    """
    if exclude_cell_types is None:
        exclude_cell_types = []
    
    datasets = proportions['Dataset_Display'].unique()
    
    for dataset in datasets:
        dataset_data = proportions[proportions['Dataset_Display'] == dataset]
        
        if exclude_cell_types:
            dataset_data = dataset_data[~dataset_data['CellCatName3'].isin(exclude_cell_types)]
        
        cell_types_in_data = dataset_data['CellCatName3'].unique()
        sorted_cell_types = sort_cell_types(cell_types_in_data)
        
        states_in_data = dataset_data['State'].unique()
        sorted_states = sort_states_for_plotting(states_in_data)
        
        pivot_props = dataset_data.pivot(index='CellCatName3', columns='State', values='Percentage').fillna(0)
        pivot_props = pivot_props.reindex(sorted_cell_types, fill_value=0)
        pivot_props = pivot_props.reindex(columns=sorted_states, fill_value=0)
        
        pivot_counts = dataset_data.pivot(index='CellCatName3', columns='State', values='Count').fillna(0)
        pivot_counts = pivot_counts.reindex(sorted_cell_types, fill_value=0)
        pivot_counts = pivot_counts.reindex(columns=sorted_states, fill_value=0)
        
        colors = get_state_colors(sorted_states)
        
        # Stacked bar chart
        fig1 = plt.figure(figsize=(16, 10))
        pivot_props.plot(kind='bar', stacked=True, ax=plt.gca(), color=colors)
        plt.title(f'CD8T Cell State Proportions (Stacked) - Dataset: {dataset}', fontsize=16, fontweight='bold')
        plt.xlabel('CD8T Cell Type', fontsize=0)
        plt.ylabel('Percentage (%)', fontsize=14)
        plt.xticks(rotation=90, ha='right', fontsize=18)
        plt.legend(title='Cell State', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        plt.tight_layout()
        try:
            original_dataset = dataset_data['Dataset'].iloc[0]
            output_path1 = os.path.join(output_dir, f'cd8t_{original_dataset}_stacked_proportions.png')
            fig1.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved stacked proportions plot for {dataset}: {output_path1}")
        except Exception as e:
            print(f"Warning: Could not save stacked plot for {dataset}: {e}")
        plt.close()
        
        # Grouped bar chart
        fig2 = plt.figure(figsize=(16, 10))
        pivot_props.plot(kind='bar', ax=plt.gca(), color=colors)
        plt.title(f'CD8T Cell State Proportions (Grouped) - Dataset: {dataset}', fontsize=16, fontweight='bold')
        plt.xlabel('CD8T Cell Type', fontsize=14)
        plt.ylabel('Percentage (%)', fontsize=14)
        plt.xticks(rotation=90, ha='right', fontsize=12)
        plt.legend(title='Cell State', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        plt.tight_layout()
        try:
            original_dataset = dataset_data['Dataset'].iloc[0]
            output_path2 = os.path.join(output_dir, f'cd8t_{original_dataset}_grouped_proportions.png')
            fig2.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved grouped proportions plot for {dataset}: {output_path2}")
        except Exception as e:
            print(f"Warning: Could not save grouped plot for {dataset}: {e}")
        plt.close()
        
        # Heatmap
        fig3 = plt.figure(figsize=(16, 8))
        sns.heatmap(pivot_props.T, 
                    annot=False, 
                    fmt='.1f', 
                    cmap='viridis', 
                    cbar_kws={'label': 'Percentage (%)'}, 
                    linewidths=0.5,
                    linecolor='white',
                    square=True,
                    annot_kws={'size': 10})
        plt.title(f'CD8T Cell State Heatmap - Dataset: {dataset}', fontsize=16, fontweight='bold')
        plt.xlabel('CD8T Cell Type', fontsize=14)
        plt.ylabel('Cell State', fontsize=14)
        plt.xticks(rotation=90, ha='right', fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        try:
            original_dataset = dataset_data['Dataset'].iloc[0]
            output_path3 = os.path.join(output_dir, f'cd8t_{original_dataset}_heatmap.png')
            fig3.savefig(output_path3, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved heatmap for {dataset}: {output_path3}")
        except Exception as e:
            print(f"Warning: Could not save heatmap for {dataset}: {e}")
        plt.close()
        
        # Absolute counts
        fig4 = plt.figure(figsize=(16, 10))
        pivot_counts.plot(kind='bar', stacked=True, ax=plt.gca(), color=colors)
        plt.title(f'CD8T Cell State Counts - Dataset: {dataset}', fontsize=16, fontweight='bold')
        plt.xlabel('CD8T Cell Type', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.xticks(rotation=90, ha='right', fontsize=12)
        plt.legend(title='Cell State', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
        plt.tight_layout()
        try:
            original_dataset = dataset_data['Dataset'].iloc[0]
            output_path4 = os.path.join(output_dir, f'cd8t_{original_dataset}_counts.png')
            fig4.savefig(output_path4, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved counts plot for {dataset}: {output_path4}")
        except Exception as e:
            print(f"Warning: Could not save counts plot for {dataset}: {e}")
        plt.close()
        
        # Pie chart
        fig5 = plt.figure(figsize=(10, 8))
        state_totals = dataset_data.groupby('State')['Count'].sum()
        state_totals = state_totals.reindex([s for s in STATE_ORDER if s in state_totals.index])
        if len(state_totals) > 0:
            pie_colors = get_state_colors(state_totals.index)
            plt.pie(state_totals.values, labels=state_totals.index, autopct='%1.1f%%', 
                    colors=pie_colors, startangle=90, textprops={'fontsize': 14})
            plt.title(f'Overall CD8T Cell State Distribution - Dataset: {dataset}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        try:
            original_dataset = dataset_data['Dataset'].iloc[0]
            output_path5 = os.path.join(output_dir, f'cd8t_{original_dataset}_distribution.png')
            fig5.savefig(output_path5, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved distribution plot for {dataset}: {output_path5}")
        except Exception as e:
            print(f"Warning: Could not save distribution plot for {dataset}: {e}")
        plt.close()
        
        # Box plot
        fig6 = plt.figure(figsize=(12, 8))
        melted_props = pivot_props.reset_index().melt(id_vars='CellCatName3', 
                                                      var_name='State', 
                                                      value_name='Percentage')
        if not melted_props.empty:
            melted_props['State'] = pd.Categorical(melted_props['State'], categories=sorted_states, ordered=True)
            sns.boxplot(data=melted_props, x='State', y='Percentage', order=sorted_states, palette=STATE_COLORS)
            plt.title(f'Distribution of State Proportions - Dataset: {dataset}', fontsize=16, fontweight='bold')
            plt.xlabel('Cell State', fontsize=14)
            plt.ylabel('Percentage (%)', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
        plt.tight_layout()
        try:
            original_dataset = dataset_data['Dataset'].iloc[0]
            output_path6 = os.path.join(output_dir, f'cd8t_{original_dataset}_boxplot.png')
            fig6.savefig(output_path6, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved boxplot for {dataset}: {output_path6}")
        except Exception as e:
            print(f"Warning: Could not save boxplot for {dataset}: {e}")
        plt.close()

def create_comparative_visualizations(proportions, output_dir=".", exclude_cell_types=None):
    """
    Create separate visualization files comparing all datasets
    """
    if exclude_cell_types is None:
        exclude_cell_types = []
    
    if exclude_cell_types:
        proportions = proportions[~proportions['CellCatName3'].isin(exclude_cell_types)]
    
    states_in_data = proportions['State'].unique()
    sorted_states = sort_states_for_plotting(states_in_data)
    colors = get_state_colors(sorted_states)
    
    # Average proportions by dataset
    fig1 = plt.figure(figsize=(12, 8))
    dataset_state_summary = proportions.groupby(['Dataset_Display', 'State'])['Percentage'].mean().reset_index()
    pivot_dataset_state = dataset_state_summary.pivot(index='Dataset_Display', columns='State', values='Percentage').fillna(0)
    pivot_dataset_state = pivot_dataset_state.reindex(columns=sorted_states, fill_value=0)
    pivot_dataset_state = pivot_dataset_state.reindex(DATASET_ORDER, fill_value=0)
    pivot_dataset_state.plot(kind='bar', ax=plt.gca(), color=colors)
    plt.title('Average State Proportions by Dataset', fontsize=16, fontweight='bold')
    plt.xlabel('Dataset', fontsize=14)
    plt.ylabel('Average Percentage (%)', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.legend(title='Cell State', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    try:
        output_path1 = os.path.join(output_dir, 'cd8t_comparison_average_proportions.png')
        fig1.savefig(output_path1, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved average proportions comparison: {output_path1}")
    except Exception as e:
        print(f"Warning: Could not save average proportions plot: {e}")
    plt.close()
    
    # Comparison heatmap
    fig2 = plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_dataset_state, 
                annot=False, 
                fmt='.1f', 
                cmap='viridis', 
                cbar_kws={'label': 'Average Percentage (%)'}, 
                linewidths=0.5,
                linecolor='white',
                square=True,
                annot_kws={'size': 12})
    plt.title('Dataset Comparison Heatmap - Proportions', fontsize=16, fontweight='bold')
    plt.xlabel('Cell State', fontsize=14)
    plt.ylabel('Dataset', fontsize=14)
    plt.xticks(fontsize=12, rotation=45)
    plt.yticks(fontsize=12, rotation=0)
    plt.tight_layout()
    try:
        output_path2 = os.path.join(output_dir, 'cd8t_comparison_proportions_heatmap.png')
        fig2.savefig(output_path2, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved comparison proportions heatmap: {output_path2}")
    except Exception as e:
        print(f"Warning: Could not save heatmap: {e}")
    plt.close()
    
    # Individual state comparisons
    for i, state in enumerate(sorted_states):
        if state not in proportions['State'].values:
            continue
        fig = plt.figure(figsize=(10, 6))
        state_data = proportions[proportions['State'] == state]
        sns.boxplot(data=state_data, x='Dataset_Display', y='Percentage', 
                   order=DATASET_ORDER, color=STATE_COLORS.get(state, '#999999'))
        plt.title(f'{state} State Across Datasets', fontsize=16, fontweight='bold')
        plt.xlabel('Dataset', fontsize=14)
        plt.ylabel('Percentage (%)', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        try:
            output_path = os.path.join(output_dir, f'cd8t_comparison_{state.lower()}_boxplot.png')
            fig.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Saved {state} comparison boxplot: {output_path}")
        except Exception as e:
            print(f"Warning: Could not save {state} boxplot: {e}")
        plt.close()
    
    # Cell type diversity
    fig6 = plt.figure(figsize=(10, 6))
    cell_type_counts = proportions.groupby('Dataset_Display')['CellCatName3'].nunique()
    cell_type_counts = cell_type_counts.reindex(DATASET_ORDER, fill_value=0)
    cell_type_counts.plot(kind='bar', ax=plt.gca())
    plt.title('Number of CD8T Cell Types per Dataset', fontsize=16, fontweight='bold')
    plt.xlabel('Dataset', fontsize=14)
    plt.ylabel('Number of Cell Types', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()
    try:
        output_path6 = os.path.join(output_dir, 'cd8t_comparison_cell_type_diversity.png')
        fig6.savefig(output_path6, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved cell type diversity comparison: {output_path6}")
    except Exception as e:
        print(f"Warning: Could not save diversity plot: {e}")
    plt.close()
    
    # Sample sizes
    fig7 = plt.figure(figsize=(10, 6))
    sample_sizes = proportions.groupby('Dataset_Display')['Count'].sum()
    sample_sizes = sample_sizes.reindex(DATASET_ORDER, fill_value=0)
    sample_sizes.plot(kind='bar', ax=plt.gca())
    plt.title('Total Sample Sizes by Dataset', fontsize=16, fontweight='bold')
    plt.xlabel('Dataset', fontsize=14)
    plt.ylabel('Total Count', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.tight_layout()
    try:
        output_path7 = os.path.join(output_dir, 'cd8t_comparison_sample_sizes.png')
        fig7.savefig(output_path7, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved sample sizes comparison: {output_path7}")
    except Exception as e:
        print(f"Warning: Could not save sample sizes plot: {e}")
    plt.close()
    
    # State distribution line plot
    fig8 = plt.figure(figsize=(12, 8))
    for dataset in DATASET_ORDER:
        if dataset in proportions['Dataset_Display'].values:
            dataset_data = proportions[proportions['Dataset_Display'] == dataset]
            state_dist = dataset_data.groupby('State')['Count'].sum()
            state_dist = state_dist.reindex([s for s in STATE_ORDER if s in state_dist.index], fill_value=0)
            state_dist_pct = (state_dist / state_dist.sum()) * 100
            plt.plot(state_dist_pct.index, state_dist_pct.values, marker='o', label=dataset, 
                    linewidth=3, markersize=8)
    plt.title('State Distribution by Dataset', fontsize=16, fontweight='bold')
    plt.xlabel('Cell State', fontsize=14)
    plt.ylabel('Percentage of Total (%)', fontsize=14)
    plt.legend(fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    try:
        output_path8 = os.path.join(output_dir, 'cd8t_comparison_state_distribution.png')
        fig8.savefig(output_path8, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved state distribution comparison: {output_path8}")
    except Exception as e:
        print(f"Warning: Could not save state distribution plot: {e}")
    plt.close()
    
    # Violin plot
    fig9 = plt.figure(figsize=(12, 8))
    melted_props = proportions.melt(id_vars=['Dataset_Display', 'CellCatName3'], 
                                   value_vars=['Percentage'], 
                                   var_name='Metric', value_name='Value')
    melted_props = melted_props.merge(proportions[['Dataset_Display', 'CellCatName3', 'State']], 
                                     on=['Dataset_Display', 'CellCatName3'])
    melted_props['State'] = pd.Categorical(melted_props['State'], categories=sorted_states, ordered=True)
    sns.violinplot(data=melted_props, x='State', y='Value', hue='Dataset_Display', 
                  order=sorted_states, hue_order=DATASET_ORDER)
    plt.title('Distribution of State Proportions', fontsize=16, fontweight='bold')
    plt.xlabel('Cell State', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.xticks(fontsize=12)
    plt.legend(title='Dataset', fontsize=12)
    plt.tight_layout()
    try:
        output_path9 = os.path.join(output_dir, 'cd8t_comparison_violin_plot.png')
        fig9.savefig(output_path9, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved violin plot comparison: {output_path9}")
    except Exception as e:
        print(f"Warning: Could not save violin plot: {e}")
    plt.close()
    
    # Correlation matrix
    fig10 = plt.figure(figsize=(8, 6))
    avg_props = proportions.groupby(['Dataset_Display', 'CellCatName3', 'State'])['Percentage'].mean().reset_index()
    pivot_for_corr = avg_props.pivot_table(index=['CellCatName3', 'State'], 
                                           columns='Dataset_Display', 
                                           values='Percentage').fillna(0)
    corr_matrix = pivot_for_corr.corr()
    corr_matrix = corr_matrix.reindex(index=DATASET_ORDER, columns=DATASET_ORDER, fill_value=0)
    sns.heatmap(corr_matrix, 
                annot=False, 
                fmt='.2f', 
                cmap='viridis', 
                center=0,
                cbar_kws={'label': 'Correlation'}, 
                linewidths=0.5,
                linecolor='white',
                square=True,
                annot_kws={'size': 12})
    plt.title('Dataset Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    try:
        output_path10 = os.path.join(output_dir, 'cd8t_comparison_correlation_matrix.png')
        fig10.savefig(output_path10, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved correlation matrix: {output_path10}")
    except Exception as e:
        print(f"Warning: Could not save correlation matrix: {e}")
    plt.close()
    
    # Variable cell types
    fig11 = plt.figure(figsize=(14, 8))
    cell_type_variance = proportions.groupby('CellCatName3')['Percentage'].var().sort_values(ascending=False).head(10)
    cell_type_variance.plot(kind='bar', ax=plt.gca())
    plt.title('Most Variable Cell Types (Top 10)', fontsize=16, fontweight='bold')
    plt.xlabel('Cell Type', fontsize=14)
    plt.ylabel('Variance in Percentage', fontsize=14)
    plt.xticks(rotation=90, fontsize=10)
    plt.tight_layout()
    try:
        output_path11 = os.path.join(output_dir, 'cd8t_comparison_variable_cell_types.png')
        fig11.savefig(output_path11, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved variable cell types analysis: {output_path11}")
    except Exception as e:
        print(f"Warning: Could not save variable cell types plot: {e}")
    plt.close()
    
    # State composition
    fig12 = plt.figure(figsize=(10, 6))
    dataset_state_counts = proportions.groupby(['Dataset_Display', 'State'])['Count'].sum().reset_index()
    pivot_state_counts = dataset_state_counts.pivot(index='Dataset_Display', columns='State', values='Count').fillna(0)
    pivot_state_counts = pivot_state_counts.reindex(columns=sorted_states, fill_value=0)
    pivot_state_counts = pivot_state_counts.reindex(DATASET_ORDER, fill_value=0)
    pivot_state_pct = pivot_state_counts.div(pivot_state_counts.sum(axis=1), axis=0) * 100
    pivot_state_pct.plot(kind='bar', stacked=True, ax=plt.gca(), color=colors)
    plt.title('State Composition by Dataset (%)', fontsize=16, fontweight='bold')
    plt.xlabel('Dataset', fontsize=14)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.xticks(rotation=45, fontsize=12)
    plt.legend(title='Cell State', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    try:
        output_path12 = os.path.join(output_dir, 'cd8t_comparison_state_composition.png')
        fig12.savefig(output_path12, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved state composition comparison: {output_path12}")
    except Exception as e:
        print(f"Warning: Could not save state composition plot: {e}")
    plt.close()

def perform_statistical_analysis(proportions, patient_proportions, density_summary=None, patient_densities=None, output_dir="."):
    """
    Perform statistical tests to compare BOTH proportions AND densities among datasets using patient-level data
    """
    output_text = []
    
    def print_and_save(text):
        print(text)
        output_text.append(text)
    
    print_and_save("\n" + "="*80)
    print_and_save("COMPREHENSIVE STATISTICAL ANALYSIS - COMPARING DATASETS (PATIENT-LEVEL)")
    print_and_save("INCLUDES BOTH PROPORTIONS AND DENSITIES")
    print_and_save("="*80)
    
    if patient_proportions is not None:
        proportion_analysis_data = patient_proportions
        print_and_save("Using PATIENT-LEVEL PROPORTION data for statistical analysis (recommended)")
        
        patient_counts_by_dataset = proportion_analysis_data.groupby('Dataset_Display')['Patient'].nunique()
        print_and_save(f"\nPatient counts by dataset (proportions):")
        for dataset, count in patient_counts_by_dataset.items():
            print_and_save(f"  {dataset}: {count} patients")
            
    else:
        proportion_analysis_data = proportions
        print_and_save("Using DATASET-LEVEL PROPORTION data for statistical analysis (fallback)")
    
    if patient_densities is not None:
        density_analysis_data = patient_densities
        print_and_save("Using PATIENT-LEVEL DENSITY data for statistical analysis")
        
        patient_counts_density = density_analysis_data.groupby('Dataset_Display')['Patient'].nunique()
        print_and_save(f"\nPatient counts by dataset (densities):")
        for dataset, count in patient_counts_density.items():
            print_and_save(f"  {dataset}: {count} patients")
    else:
        density_analysis_data = None
        print_and_save("No patient-level density data available")
    
    datasets = proportion_analysis_data['Dataset_Display'].unique()
    states = proportion_analysis_data['State'].unique()
    
    # PART 1: PROPORTION ANALYSIS
    print_and_save("\n" + "="*60)
    print_and_save("PART 1: PROPORTION ANALYSIS")
    print_and_save("="*60)
    
    print_and_save("\n1. CHI-SQUARE TEST FOR DATASET-STATE ASSOCIATION (PROPORTIONS)")
    print_and_save("-" * 60)
    
    if patient_proportions is not None:
        contingency_data = patient_proportions.groupby(['Dataset_Display', 'State'])['Count'].sum().reset_index()
        contingency_table = contingency_data.pivot(index='Dataset_Display', columns='State', values='Count').fillna(0)
    else:
        contingency_table = pd.crosstab(proportions['Dataset_Display'], proportions['State'], 
                                       values=proportions['Count'], aggfunc='sum').fillna(0)
    
    print_and_save("Contingency Table (Total Counts - Proportions):")
    print_and_save(str(contingency_table))
    
    try:
        contingency_path = os.path.join(output_dir, 'cd8t_contingency_table_proportions.csv')
        contingency_table.to_csv(contingency_path)
        print_and_save(f"Contingency table (proportions) saved to: {contingency_path}")
    except Exception as e:
        print_and_save(f"Warning: Could not save contingency table: {e}")
    
    try:
        chi2, p_value, dof, expected = chi2_contingency(contingency_table)
        print_and_save(f"\nChi-square statistic: {chi2:.4f}")
        print_and_save(f"p-value: {p_value:.4e}")
        print_and_save(f"Degrees of freedom: {dof}")
        
        alpha = 0.05
        if p_value < alpha:
            print_and_save(f"Result: SIGNIFICANT association between dataset and cell state (p < {alpha})")
        else:
            print_and_save(f"Result: NO significant association between dataset and cell state (p >= {alpha})")
    except Exception as e:
        print_and_save(f"Error in chi-square test: {e}")
    
    print_and_save(f"\n2. KRUSKAL-WALLIS TESTS FOR PROPORTIONS (Non-parametric ANOVA) - PATIENT-LEVEL")
    print_and_save("-" * 60)
    
    kw_results_prop = []
    pairwise_results_prop = []
    
    for state in states:
        print_and_save(f"\nState: {state} (Proportions)")
        state_data = proportion_analysis_data[proportion_analysis_data['State'] == state]
        
        datasets_with_data = state_data['Dataset_Display'].unique()
        if len(datasets_with_data) < 2:
            print_and_save("  Not enough datasets for comparison")
            kw_results_prop.append({
                'State': state,
                'KW_Statistic': 'NA',
                'KW_p_value': 'NA',
                'Result': 'Insufficient datasets'
            })
            continue
        
        if patient_proportions is not None:
            groups = []
            group_labels = []
            for dataset in datasets_with_data:
                dataset_state_data = state_data[state_data['Dataset_Display'] == dataset]
                patient_percentages = dataset_state_data['Percentage'].values
                if len(patient_percentages) > 0:
                    groups.append(patient_percentages)
                    group_labels.append(dataset)
                    print_and_save(f"  {dataset}: {len(patient_percentages)} observations from {dataset_state_data['Patient'].nunique()} patients")
        else:
            groups = [state_data[state_data['Dataset_Display'] == dataset]['Percentage'].values 
                     for dataset in datasets_with_data 
                     if len(state_data[state_data['Dataset_Display'] == dataset]) > 0]
            group_labels = [d for d in datasets_with_data if len(state_data[state_data['Dataset_Display'] == d]) > 0]
        
        if len(groups) < 2 or any(len(group) == 0 for group in groups):
            print_and_save("  Insufficient data for statistical test")
            kw_results_prop.append({
                'State': state,
                'KW_Statistic': 'NA',
                'KW_p_value': 'NA',
                'Result': 'Insufficient data'
            })
            continue
            
        try:
            stat, p_val = kruskal(*groups)
            print_and_save(f"  Kruskal-Wallis statistic: {stat:.4f}")
            print_and_save(f"  p-value: {p_val:.4e}")
            
            if p_val < 0.05:
                result_text = "SIGNIFICANT differences"
                print_and_save(f"  Result: SIGNIFICANT differences between datasets (p < 0.05)")
                
                from itertools import combinations
                print_and_save("  Pairwise comparisons:")
                for i, (ds1, ds2) in enumerate(combinations(group_labels, 2)):
                    idx1 = group_labels.index(ds1)
                    idx2 = group_labels.index(ds2)
                    data1 = groups[idx1]
                    data2 = groups[idx2]
                    
                    if len(data1) > 0 and len(data2) > 0:
                        try:
                            u_stat, u_p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                            print_and_save(f"    {ds1} vs {ds2}: p = {u_p:.4f}")
                            pairwise_results_prop.append({
                                'State': state,
                                'Dataset1': ds1,
                                'Dataset2': ds2,
                                'Mann_Whitney_p': u_p,
                                'N1': len(data1),
                                'N2': len(data2)
                            })
                        except Exception:
                            print_and_save(f"    {ds1} vs {ds2}: Unable to compute")
                            pairwise_results_prop.append({
                                'State': state,
                                'Dataset1': ds1,
                                'Dataset2': ds2,
                                'Mann_Whitney_p': 'Unable to compute',
                                'N1': len(data1),
                                'N2': len(data2)
                            })
            else:
                result_text = "NO significant differences"
                print_and_save(f"  Result: NO significant differences between datasets (p >= 0.05)")
            
            kw_results_prop.append({
                'State': state,
                'KW_Statistic': stat,
                'KW_p_value': p_val,
                'Result': result_text,
                'N_total': sum(len(g) for g in groups)
            })
            
        except Exception as e:
            print_and_save(f"  Error in Kruskal-Wallis test: {e}")
            kw_results_prop.append({
                'State': state,
                'KW_Statistic': 'Error',
                'KW_p_value': 'Error',
                'Result': f'Error: {e}',
                'N_total': 'NA'
            })
    
    # PART 2: DENSITY ANALYSIS
    if density_analysis_data is not None:
        print_and_save("\n" + "="*60)
        print_and_save("PART 2: DENSITY ANALYSIS")
        print_and_save("="*60)
        
        print_and_save("\n3. CHI-SQUARE TEST FOR DATASET-STATE ASSOCIATION (DENSITIES)")
        print_and_save("-" * 60)
        
        density_contingency_data = density_analysis_data.groupby(['Dataset_Display', 'State'])['Cell_Count'].sum().reset_index()
        density_contingency_table = density_contingency_data.pivot(index='Dataset_Display', columns='State', values='Cell_Count').fillna(0)
        
        print_and_save("Contingency Table (Total Cell Counts for Density Analysis):")
        print_and_save(str(density_contingency_table))
        
        try:
            density_contingency_path = os.path.join(output_dir, 'cd8t_contingency_table_densities.csv')
            density_contingency_table.to_csv(density_contingency_path)
            print_and_save(f"Contingency table (densities) saved to: {density_contingency_path}")
        except Exception as e:
            print_and_save(f"Warning: Could not save density contingency table: {e}")
        
        try:
            chi2_dens, p_value_dens, dof_dens, expected_dens = chi2_contingency(density_contingency_table)
            print_and_save(f"\nChi-square statistic (densities): {chi2_dens:.4f}")
            print_and_save(f"p-value (densities): {p_value_dens:.4e}")
            print_and_save(f"Degrees of freedom (densities): {dof_dens}")
            
            alpha = 0.05
            if p_value_dens < alpha:
                print_and_save(f"Result: SIGNIFICANT association between dataset and cell state densities (p < {alpha})")
            else:
                print_and_save(f"Result: NO significant association between dataset and cell state densities (p >= {alpha})")
        except Exception as e:
            print_and_save(f"Error in chi-square test for densities: {e}")
        
        print_and_save(f"\n4. KRUSKAL-WALLIS TESTS FOR DENSITIES (Non-parametric ANOVA) - PATIENT-LEVEL")
        print_and_save("-" * 60)
        
        kw_results_dens = []
        pairwise_results_dens = []
        
        for state in states:
            if state not in density_analysis_data['State'].values:
                continue
                
            print_and_save(f"\nState: {state} (Densities)")
            state_data = density_analysis_data[density_analysis_data['State'] == state]
            
            datasets_with_data = state_data['Dataset_Display'].unique()
            if len(datasets_with_data) < 2:
                print_and_save("  Not enough datasets for comparison")
                kw_results_dens.append({
                    'State': state,
                    'KW_Statistic': 'NA',
                    'KW_p_value': 'NA',
                    'Result': 'Insufficient datasets'
                })
                continue
            
            groups = []
            group_labels = []
            for dataset in datasets_with_data:
                dataset_state_data = state_data[state_data['Dataset_Display'] == dataset]
                patient_densities_vals = dataset_state_data['Density'].values
                if len(patient_densities_vals) > 0:
                    groups.append(patient_densities_vals)
                    group_labels.append(dataset)
                    print_and_save(f"  {dataset}: {len(patient_densities_vals)} observations from {dataset_state_data['Patient'].nunique()} patients")
            
            if len(groups) < 2 or any(len(group) == 0 for group in groups):
                print_and_save("  Insufficient data for statistical test")
                kw_results_dens.append({
                    'State': state,
                    'KW_Statistic': 'NA',
                    'KW_p_value': 'NA',
                    'Result': 'Insufficient data'
                })
                continue
                
            try:
                stat, p_val = kruskal(*groups)
                print_and_save(f"  Kruskal-Wallis statistic: {stat:.4f}")
                print_and_save(f"  p-value: {p_val:.4e}")
                
                if p_val < 0.05:
                    result_text = "SIGNIFICANT differences"
                    print_and_save(f"  Result: SIGNIFICANT differences between datasets (p < 0.05)")
                    
                    from itertools import combinations
                    print_and_save("  Pairwise comparisons:")
                    for i, (ds1, ds2) in enumerate(combinations(group_labels, 2)):
                        idx1 = group_labels.index(ds1)
                        idx2 = group_labels.index(ds2)
                        data1 = groups[idx1]
                        data2 = groups[idx2]
                        
                        if len(data1) > 0 and len(data2) > 0:
                            try:
                                u_stat, u_p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                                print_and_save(f"    {ds1} vs {ds2}: p = {u_p:.4f}")
                                pairwise_results_dens.append({
                                    'State': state,
                                    'Dataset1': ds1,
                                    'Dataset2': ds2,
                                    'Mann_Whitney_p': u_p,
                                    'N1': len(data1),
                                    'N2': len(data2)
                                })
                            except Exception:
                                print_and_save(f"    {ds1} vs {ds2}: Unable to compute")
                                pairwise_results_dens.append({
                                    'State': state,
                                    'Dataset1': ds1,
                                    'Dataset2': ds2,
                                    'Mann_Whitney_p': 'Unable to compute',
                                    'N1': len(data1),
                                    'N2': len(data2)
                                })
                else:
                    result_text = "NO significant differences"
                    print_and_save(f"  Result: NO significant differences between datasets (p >= 0.05)")
                
                kw_results_dens.append({
                    'State': state,
                    'KW_Statistic': stat,
                    'KW_p_value': p_val,
                    'Result': result_text,
                    'N_total': sum(len(g) for g in groups)
                })
                
            except Exception as e:
                print_and_save(f"  Error in Kruskal-Wallis test: {e}")
                kw_results_dens.append({
                    'State': state,
                    'KW_Statistic': 'Error',
                    'KW_p_value': 'Error',
                    'Result': f'Error: {e}',
                    'N_total': 'NA'
                })
    
    # 5. Summary statistics
    print_and_save(f"\n5. DESCRIPTIVE STATISTICS BY DATASET (PATIENT-LEVEL)")
    print_and_save("-" * 60)
    
    print_and_save(f"\nPROPORTIONS:")
    summary_stats_prop = proportion_analysis_data.groupby(['Dataset_Display', 'State'])['Percentage'].agg([
        'count', 'mean', 'std', 'median', 'min', 'max'
    ]).round(3)
    print_and_save(str(summary_stats_prop))
    
    if density_analysis_data is not None:
        print_and_save(f"\nDENSITIES:")
        summary_stats_dens = density_analysis_data.groupby(['Dataset_Display', 'State'])['Density'].agg([
            'count', 'mean', 'std', 'median', 'min', 'max'
        ]).round(3)
        print_and_save(str(summary_stats_dens))
    
    # Save all results
    try:
        kw_df_prop = pd.DataFrame(kw_results_prop)
        kw_path_prop = os.path.join(output_dir, 'cd8t_kruskal_wallis_results_proportions.csv')
        kw_df_prop.to_csv(kw_path_prop, index=False)
        print_and_save(f"Kruskal-Wallis results (proportions) saved to: {kw_path_prop}")
        
        if pairwise_results_prop:
            pw_df_prop = pd.DataFrame(pairwise_results_prop)
            pw_path_prop = os.path.join(output_dir, 'cd8t_pairwise_comparisons_proportions.csv')
            pw_df_prop.to_csv(pw_path_prop, index=False)
            print_and_save(f"Pairwise comparisons (proportions) saved to: {pw_path_prop}")
        
        stats_path_prop = os.path.join(output_dir, 'cd8t_descriptive_statistics_proportions.csv')
        summary_stats_prop.to_csv(stats_path_prop)
        print_and_save(f"Descriptive statistics (proportions) saved to: {stats_path_prop}")
        
        if density_analysis_data is not None:
            kw_df_dens = pd.DataFrame(kw_results_dens)
            kw_path_dens = os.path.join(output_dir, 'cd8t_kruskal_wallis_results_densities.csv')
            kw_df_dens.to_csv(kw_path_dens, index=False)
            print_and_save(f"Kruskal-Wallis results (densities) saved to: {kw_path_dens}")
            
            if pairwise_results_dens:
                pw_df_dens = pd.DataFrame(pairwise_results_dens)
                pw_path_dens = os.path.join(output_dir, 'cd8t_pairwise_comparisons_densities.csv')
                pw_df_dens.to_csv(pw_path_dens, index=False)
                print_and_save(f"Pairwise comparisons (densities) saved to: {pw_path_dens}")
            
            stats_path_dens = os.path.join(output_dir, 'cd8t_descriptive_statistics_densities.csv')
            summary_stats_dens.to_csv(stats_path_dens)
            print_and_save(f"Descriptive statistics (densities) saved to: {stats_path_dens}")
        
    except Exception as e:
        print_and_save(f"Warning: Could not save some results: {e}")
    
    # 6. Effect size analysis
    print_and_save(f"\n6. EFFECT SIZE ANALYSIS (Cohen's d) - PATIENT-LEVEL")
    print_and_save("-" * 60)
    
    def cohens_d(x, y):
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2
        pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / dof)
        return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std != 0 else 0
    
    from itertools import combinations
    
    effect_size_results_prop = []
    effect_size_results_dens = []
    
    print_and_save(f"\nPROPORTIONS EFFECT SIZES:")
    for state in states:
        print_and_save(f"\nState: {state} (Proportions)")
        state_data = proportion_analysis_data[proportion_analysis_data['State'] == state]
        
        datasets_with_data = state_data['Dataset_Display'].unique()
        for ds1, ds2 in combinations(datasets_with_data, 2):
            data1 = state_data[state_data['Dataset_Display'] == ds1]['Percentage'].values
            data2 = state_data[state_data['Dataset_Display'] == ds2]['Percentage'].values
            
            if len(data1) > 1 and len(data2) > 1:
                d = cohens_d(data1, data2)
                
                if abs(d) < 0.2:
                    interpretation = "negligible"
                elif abs(d) < 0.5:
                    interpretation = "small"
                elif abs(d) < 0.8:
                    interpretation = "medium"
                else:
                    interpretation = "large"
                
                print_and_save(f"  {ds1} vs {ds2}: d = {d:.3f} ({interpretation} effect)")
                
                if patient_proportions is not None:
                    n_patients_1 = state_data[state_data['Dataset_Display'] == ds1]['Patient'].nunique()
                    n_patients_2 = state_data[state_data['Dataset_Display'] == ds2]['Patient'].nunique()
                else:
                    n_patients_1 = 'NA'
                    n_patients_2 = 'NA'
                
                effect_size_results_prop.append({
                    'State': state,
                    'Dataset1': ds1,
                    'Dataset2': ds2,
                    'Cohens_d': d,
                    'Effect_Size': interpretation,
                    'Mean1': np.mean(data1),
                    'Mean2': np.mean(data2),
                    'N_observations_1': len(data1),
                    'N_observations_2': len(data2),
                    'N_patients_1': n_patients_1,
                    'N_patients_2': n_patients_2
                })
    
    if density_analysis_data is not None:
        print_and_save(f"\nDENSITY EFFECT SIZES:")
        for state in states:
            if state not in density_analysis_data['State'].values:
                continue
                
            print_and_save(f"\nState: {state} (Densities)")
            state_data = density_analysis_data[density_analysis_data['State'] == state]
            
            datasets_with_data = state_data['Dataset_Display'].unique()
            for ds1, ds2 in combinations(datasets_with_data, 2):
                data1 = state_data[state_data['Dataset_Display'] == ds1]['Density'].values
                data2 = state_data[state_data['Dataset_Display'] == ds2]['Density'].values
                
                if len(data1) > 1 and len(data2) > 1:
                    d = cohens_d(data1, data2)
                    
                    if abs(d) < 0.2:
                        interpretation = "negligible"
                    elif abs(d) < 0.5:
                        interpretation = "small"
                    elif abs(d) < 0.8:
                        interpretation = "medium"
                    else:
                        interpretation = "large"
                    
                    print_and_save(f"  {ds1} vs {ds2}: d = {d:.3f} ({interpretation} effect)")
                    
                    n_patients_1 = state_data[state_data['Dataset_Display'] == ds1]['Patient'].nunique()
                    n_patients_2 = state_data[state_data['Dataset_Display'] == ds2]['Patient'].nunique()
                    
                    effect_size_results_dens.append({
                        'State': state,
                        'Dataset1': ds1,
                        'Dataset2': ds2,
                        'Cohens_d': d,
                        'Effect_Size': interpretation,
                        'Mean1': np.mean(data1),
                        'Mean2': np.mean(data2),
                        'N_observations_1': len(data1),
                        'N_observations_2': len(data2),
                        'N_patients_1': n_patients_1,
                        'N_patients_2': n_patients_2
                    })
    
    try:
        effect_df_prop = pd.DataFrame(effect_size_results_prop)
        effect_path_prop = os.path.join(output_dir, 'cd8t_effect_sizes_proportions.csv')
        effect_df_prop.to_csv(effect_path_prop, index=False)
        print_and_save(f"Effect size analysis (proportions) saved to: {effect_path_prop}")
        
        if density_analysis_data is not None:
            effect_df_dens = pd.DataFrame(effect_size_results_dens)
            effect_path_dens = os.path.join(output_dir, 'cd8t_effect_sizes_densities.csv')
            effect_df_dens.to_csv(effect_path_dens, index=False)
            print_and_save(f"Effect size analysis (densities) saved to: {effect_path_dens}")
    except Exception as e:
        print_and_save(f"Warning: Could not save effect size results: {e}")
    
    # Save patient-level data
    if patient_proportions is not None:
        try:
            patient_path = os.path.join(output_dir, 'cd8t_patient_level_proportions.csv')
            patient_proportions.to_csv(patient_path, index=False)
            print_and_save(f"Patient-level proportions saved to: {patient_path}")
        except Exception as e:
            print_and_save(f"Warning: Could not save patient-level proportion data: {e}")
    
    if patient_densities is not None:
        try:
            patient_dens_path = os.path.join(output_dir, 'cd8t_patient_level_densities.csv')
            patient_densities.to_csv(patient_dens_path, index=False)
            print_and_save(f"Patient-level densities saved to: {patient_dens_path}")
        except Exception as e:
            print_and_save(f"Warning: Could not save patient-level density data: {e}")
    
    # Save report
    try:
        report_path = os.path.join(output_dir, 'cd8t_comprehensive_statistical_analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write('\n'.join(output_text))
        print_and_save(f"\nComplete comprehensive statistical analysis report saved to: {report_path}")
    except Exception as e:
        print_and_save(f"Warning: Could not save statistical analysis report: {e}")

def generate_comprehensive_summary(proportions, output_dir=".", exclude_cell_types=None):
    """
    Generate comprehensive summary including dataset comparisons
    """
    if exclude_cell_types is None:
        exclude_cell_types = []
    
    print("\n" + "="*80)
    print("COMPREHENSIVE SUMMARY REPORT")
    print("="*80)
    
    datasets = proportions['Dataset_Display'].unique()
    states = proportions['State'].unique()
    
    print(f"\n1. DATASET OVERVIEW")
    print("-" * 40)
    for dataset in datasets:
        dataset_data = proportions[proportions['Dataset_Display'] == dataset]
        total_cells = dataset_data['Count'].sum()
        unique_cell_types = dataset_data['CellCatName3'].nunique()
        print(f"{dataset}: {total_cells} cells, {unique_cell_types} cell types")
    
    print(f"\n2. STATE DISTRIBUTION BY DATASET")
    print("-" * 40)
    for dataset in datasets:
        print(f"\n{dataset}:")
        dataset_data = proportions[proportions['Dataset_Display'] == dataset]
        state_summary = dataset_data.groupby('State').agg({
            'Count': 'sum',
            'Percentage': 'mean'
        }).round(2)
        state_summary = state_summary.reindex([s for s in STATE_ORDER if s in state_summary.index])
        for state in state_summary.index:
            count = state_summary.loc[state, 'Count']
            pct = state_summary.loc[state, 'Percentage']
            print(f"  {state}: {count} cells (avg {pct}% per cell type)")
    
    print(f"\n3. CELL TYPE DIVERSITY")
    print("-" * 40)
    cell_type_analysis = proportions.groupby(['Dataset_Display', 'CellCatName3']).size().reset_index(name='Observations')
    for dataset in datasets:
        dataset_cell_types = cell_type_analysis[cell_type_analysis['Dataset_Display'] == dataset]
        print(f"{dataset}: {len(dataset_cell_types)} unique cell types observed")
    
    if exclude_cell_types:
        print(f"\n4. EXCLUDED CELL TYPES")
        print("-" * 40)
        print(f"Excluded from analysis: {', '.join(exclude_cell_types)}")
    
    try:
        detailed_path = os.path.join(output_dir, 'cd8t_detailed_analysis_by_dataset.csv')
        proportions.to_csv(detailed_path, index=False)
        
        summary_by_dataset = proportions.groupby(['Dataset_Display', 'State']).agg({
            'Count': 'sum',
            'Percentage': ['mean', 'std', 'min', 'max']
        }).round(3)
        summary_path = os.path.join(output_dir, 'cd8t_summary_by_dataset.csv')
        summary_by_dataset.to_csv(summary_path)
        
        comparison_summary = proportions.groupby(['Dataset_Display', 'CellCatName3', 'State'])['Percentage'].mean().reset_index()
        comparison_pivot = comparison_summary.pivot_table(
            index=['CellCatName3', 'State'], 
            columns='Dataset_Display', 
            values='Percentage'
        ).fillna(0)
        
        sorted_index = []
        for cell_type in sort_cell_types(comparison_pivot.index.get_level_values(0).unique()):
            for state in [s for s in STATE_ORDER if s in states]:
                if (cell_type, state) in comparison_pivot.index:
                    sorted_index.append((cell_type, state))
        
        comparison_pivot = comparison_pivot.reindex(sorted_index, fill_value=0)
        comparison_path = os.path.join(output_dir, 'cd8t_cross_dataset_comparison.csv')
        comparison_pivot.to_csv(comparison_path)
        
        print(f"\nFiles saved in {output_dir}:")
        print(f"- {os.path.basename(detailed_path)}")
        print(f"- {os.path.basename(summary_path)}")
        print(f"- {os.path.basename(comparison_path)}")
        
    except Exception as e:
        print(f"Warning: Could not save some files: {e}")

def main(file_path, output_directory="./results", exclude_cell_types=None):
    """
    Main analysis function with dataset-specific analysis and patient-level statistics
    
    Parameters:
    -----------
    file_path : str
        Path to the input data file
    output_directory : str
        Directory to save output files
    exclude_cell_types : list, optional
        List of cell types to exclude from analysis
        
    Updated features:
    - Compares BOTH cell proportions AND densities using statistical tests
    - Calculates densities using BOTH inflammation area AND tissue area
    - Proper axis arrangement in heatmaps: x-axis = states, y-axis = datasets
    - Dataset name mapping: IgG4 -> IgG4-RKD, NK2 -> KC, AAV -> AAV
    """
    if exclude_cell_types is None:
        exclude_cell_types = []
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created output directory: {output_directory}")
    
    print("Starting Enhanced CD8T Cell State Analysis with Dataset Comparison...")
    print("="*70)
    print(f"Output directory: {os.path.abspath(output_directory)}")
    if exclude_cell_types:
        print(f"Excluded cell types: {exclude_cell_types}")
    print("Features:")
    print("- Patient-level statistical analysis")
    print("- Cell density analysis using BOTH inflammation area AND tissue area")
    print("- Proportions AND densities statistical comparison")
    print("- Proper heatmap axis arrangement")
    print("- Dataset name mapping (IgG4->IgG4-RKD, NK2->KC)")
    print("="*70)
    
    # Load data
    df = load_and_process_data(file_path)
    if df is None:
        return
    
    # Check required columns
    required_cols = ['CellCatName3', 'CellCatName', 'Dataset', 'CompName']
    optional_cols = ['Patient']
    
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Required columns {required_cols} not found in data")
        print(f"Available columns: {list(df.columns)}")
        return
        
    missing_optional = [col for col in optional_cols if col not in df.columns]
    if missing_optional:
        print(f"Note: Optional columns {missing_optional} not found - will use dataset-level analysis")
    else:
        print(f"Patient column found - will perform patient-level statistical analysis")
    
    # Validate datasets
    expected_datasets = ['IgG4', 'AAV', 'NK2']
    actual_datasets = df['Dataset'].unique()
    missing_datasets = set(expected_datasets) - set(actual_datasets)
    if missing_datasets:
        print(f"Warning: Expected datasets {missing_datasets} not found")
        print(f"Found datasets: {list(actual_datasets)}")
    
    print(f"Dataset mapping will be applied:")
    for orig, display in DATASET_MAPPING.items():
        if orig in actual_datasets:
            print(f"  {orig} -> {display}")
    
    # Filter for CD8T states
    filtered_df = filter_cd8t_states(df, exclude_cell_types)
    if filtered_df.empty:
        print("No data found after filtering")
        return
    
    # Calculate proportions
    result = calculate_proportions_by_dataset(filtered_df)
    
    if len(result) == 3:
        proportions, counts, totals = result
        patient_proportions = None
        print("Using dataset-level proportions for analysis")
    else:
        proportions, counts, totals, patient_proportions = result
        print("Using patient-level proportions for statistical analysis")
    
    # Calculate cell densities using INFLAMMATION area
    print("\n" + "="*70)
    print("CALCULATING PATIENT-LEVEL CELL DENSITIES USING INFLAMMATION AREA")
    print("="*70)
    
    density_result = calculate_cell_densities_by_state(filtered_df, output_directory, exclude_cell_types)
    
    density_summary = None
    patient_densities = None
    if density_result is not None:
        if isinstance(density_result, tuple):
            density_summary, patient_densities = density_result
            create_cell_density_heatmap(density_summary, patient_densities, output_directory, exclude_cell_types)
        else:
            create_cell_density_heatmap(density_result, None, output_directory, exclude_cell_types)
            density_summary = density_result
    
    # Calculate cell densities using TISSUE area
    print("\n" + "="*70)
    print("CALCULATING PATIENT-LEVEL CELL DENSITIES USING TISSUE AREA")
    print("="*70)
    
    tissue_density_result = calculate_cell_densities_by_state_tissue(filtered_df, output_directory, exclude_cell_types)
    
    tissue_density_summary = None
    tissue_patient_densities = None
    if tissue_density_result is not None:
        if isinstance(tissue_density_result, tuple):
            tissue_density_summary, tissue_patient_densities = tissue_density_result
            create_cell_density_heatmap_tissue(tissue_density_summary, tissue_patient_densities, output_directory, exclude_cell_types)
        else:
            create_cell_density_heatmap_tissue(tissue_density_result, None, output_directory, exclude_cell_types)
            tissue_density_summary = tissue_density_result
    
    # Create dataset-specific visualizations
    print("\nGenerating dataset-specific visualizations...")
    create_dataset_specific_visualizations(proportions, output_directory, exclude_cell_types)
    
    # Create comparative visualizations
    print("\nGenerating cross-dataset comparison visualizations...")
    create_comparative_visualizations(proportions, output_directory, exclude_cell_types)
    
    # Perform statistical analysis
    print("\nPerforming comprehensive statistical analysis...")
    perform_statistical_analysis(proportions, patient_proportions, density_summary, patient_densities, output_directory)
    
    # Generate summary
    generate_comprehensive_summary(proportions, output_directory, exclude_cell_types)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"Key improvements in this version:")
    print(f"1. Statistical tests compare BOTH proportions AND densities")
    print(f"2. Cell densities calculated using BOTH inflammation area AND tissue area")
    print(f"3. Heatmaps have proper axis arrangement:")
    print(f"   - X-axis: {' -> '.join(STATE_ORDER)}")
    print(f"   - Y-axis: {' -> '.join(DATASET_ORDER)}")
    print(f"4. Comprehensive patient-level statistical analysis")
    print(f"5. Separate result files for proportions, inflammation densities, and tissue densities")
    print(f"6. Dataset name mapping applied consistently")
    print(f"\nGenerated output files:")
    print(f"- cd8t_patient_level_densities_by_state.csv")
    print(f"- cd8t_patient_level_densities_by_state_tissue.csv")
    print(f"- cd8t_state_densities_summary_by_dataset.csv")
    print(f"- cd8t_state_densities_summary_by_dataset_tissue.csv")
    print(f"- cd8t_state_density_heatmap_patient_level.png")
    print(f"- cd8t_state_density_heatmap_patient_level_tissue.png")
    print(f"- cd8t_patient_density_distributions.png")
    print(f"- cd8t_patient_density_distributions_tissue.png")
    print(f"- cd8t_comprehensive_statistical_analysis_report.txt")
    print(f"- Plus all dataset-specific visualizations and comparison plots")
    print(f"\nCheck the output directory for all generated files and plots.")

# Example usage
if __name__ == "__main__":
    file_path = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/processed_combined_SAM_scores_filtered_alldataset5_10woMonointerstitial_memcatname3cellcatname23_CD8T.csv"
    
    output_directory = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/cd8t_states_analysis2"
    
    exclude_cell_types = [
        'NaiveCD8T', 'MKI+CD8T', 'MKI+GZMA+GZMK+CD8CTL', 'MKI+GZMB+CD8CTL', 'Apoptotic'
    ]
    
    main(file_path, output_directory, exclude_cell_types)
