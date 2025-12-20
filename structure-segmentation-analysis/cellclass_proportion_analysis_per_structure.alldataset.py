import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the custom cell type order
CELL_TYPE_ORDER = [
    'NaiveCD4T', 'CD4T', 'ICOS+Treg', 'CD103+CD4Treg', 'HLAII+CD27+Treg',
    'Tfh', 'TIGIT+Tfh', 'HLAII+GZMK+Tfh', 'GZMA+GZMK+CD4CTL', 'GZMB+CD4CTL','CD4Tcm-Tem',
    'NaiveCD8T', 'CD8T', 'GZMA+GZMK+CD8CTL', 'GZMB+CD8CTL','CD103+CD8Treg', 'CD8Trm',
    'CD103+CD8Trm', 'TIGIT+CD8T', 'CD8Tcm-Tem', 'γδT', 'ILC', 'ImmatureNK', 'MatureNK', 'NKT', 
    'NaiveB', 'TransitionalB','B', 'CD27+B', 'DN2B', 'AgeAsscB',
    'Plasmablast', 'IgG4+Plasmablast', 'Plasma', 'IgG4+Plasma',
    'M1Macrophage', 'MPO+Macrophage',
    'M2Macrophage', 'MerTK+M1Macrophage', 'MerTK+M2Macrophage',
    'GPNMB+M1Macrophage', 'GPNMB+M2Macrophage', 'cDC1', 'cDC2', 'pDC',
    'Neutrophil',
    'Myofibroblast-SmoothMuscle', 'CD90+Myofibroblast', 'MMT',
    'ProximalTubular', 'DistalTubular', 'StressedTubular',
    'Endothelial', 'LymphaticEndothelial', 'Podocyte', 'PEC', 'Apoptotic','RBC','TubuleBackground','unclassified'
]

def analyze_cell_proportions(file_path, output_dir='cell_proportion_results', min_cell_threshold=0):
    """
    Analyzes cell type proportions across different structures and datasets.
    Creates heatmaps and tables to visualize the results with consistent color scaling.
    
    Args:
        file_path: Path to the CSV file containing the data
        output_dir: Directory where results will be saved
        min_cell_threshold: Minimum number of cells required to include a cell type (0 = include all)
    """
    # Fix the DtypeWarning by specifying low_memory=False
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, low_memory=False)
    
    # Calculate cell proportions
    print("Calculating cell type proportions...")
    cell_counts = df.groupby(['Dataset', 'structure', 'MemCatName']).size().reset_index(name='count')
    structure_totals = df.groupby(['Dataset', 'structure']).size().reset_index(name='total')
    proportions = pd.merge(cell_counts, structure_totals, on=['Dataset', 'structure'])
    proportions['proportion'] = proportions['count'] / proportions['total']
    
    # Calculate global min and max for consistent color scaling
    global_min = proportions['proportion'].min()
    global_max = proportions['proportion'].max()
    print(f"Global proportion range: {global_min:.3f} to {global_max:.3f}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results to CSV
    detailed_csv = os.path.join(output_dir, 'cell_proportions_detailed.csv')
    print(f"Saving detailed proportions to {detailed_csv}...")
    proportions.to_csv(detailed_csv, index=False)
    
    # Create dataset-specific directories and process each dataset
    datasets = df['Dataset'].unique()
    print(f"Processing datasets: {', '.join(datasets)}")
    
    for dataset in datasets:
        print(f"\nProcessing {dataset} dataset...")
        
        # Create dataset-specific directory
        dataset_dir = os.path.join(output_dir, f'dataset_{dataset}')
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Filter data for this dataset
        dataset_data = proportions[proportions['Dataset'] == dataset]
        
        # Create a pivot table
        pivot_table = dataset_data.pivot_table(
            values='proportion',
            index='structure',
            columns='MemCatName',
            fill_value=0
        )
        
        # Use exact ordering for cell proportions based on the CELL_TYPE_ORDER list
        # Get all cell types present in this dataset
        dataset_cell_types = pivot_table.columns.tolist()
        
        # Create a column ordering that strictly follows CELL_TYPE_ORDER
        # First, include cell types from our predefined order if they exist in the data
        ordered_cols = [cell for cell in CELL_TYPE_ORDER if cell in dataset_cell_types]
        
        # Then add any additional cell types not in our predefined order
        ordered_cols.extend([cell for cell in dataset_cell_types if cell not in CELL_TYPE_ORDER])
        
        # Apply the ordering to the pivot table
        pivot_table = pivot_table[ordered_cols]
        
        # Save the pivot table to CSV in dataset directory
        dataset_csv = os.path.join(dataset_dir, f'cell_proportions_{dataset}.csv')
        pivot_table.to_csv(dataset_csv)
        
        # Create and save the heatmap with consistent color scaling
        # Calculate figure size with better proportions for readability
        fig_width = max(12, min(30, len(ordered_cols) * 1.5))
        fig_height = max(8, len(pivot_table) * 1.5)
        plt.figure(figsize=(fig_width, fig_height))
        
        # Use consistent color scaling across all datasets
        sns.heatmap(pivot_table, annot=False, fmt='.1%', cmap='viridis', linewidths=.8,
                   annot_kws={'size': 12, 'weight': 'bold'},
                   cbar_kws={'shrink': 0.8, 'label': 'Proportion'},
                   vmin=global_min, vmax=global_max)  # Consistent color scaling
        plt.title(f'Cell Type Proportions in {dataset} Dataset', fontsize=16)
        plt.xlabel('Cell Type', fontsize=14)
        plt.ylabel('Structure', fontsize=14)
        plt.xticks(rotation=90, fontsize=12)
        plt.yticks(fontsize=22)
        plt.tight_layout()
        
        dataset_png = os.path.join(dataset_dir, f'cell_proportions_{dataset}.png')
        plt.savefig(dataset_png, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Created heatmap for {dataset}: {dataset_png}")
        
        # Create dataset-specific summary table with top 5 cell types for each structure
        print(f"Creating summary table for {dataset}...")
        summary_rows = []
        
        for structure in df[df['Dataset'] == dataset]['structure'].unique():
            cells = proportions[(proportions['Dataset'] == dataset) & 
                               (proportions['structure'] == structure)]
            
            if not cells.empty:
                # Get top 5 cell types
                top5 = cells.nlargest(5, 'proportion')
                
                for i, (_, row) in enumerate(top5.iterrows(), 1):
                    summary_rows.append({
                        'Dataset': dataset,
                        'Structure': structure,
                        'Rank': i,
                        'Cell Type': row['MemCatName'],
                        'Proportion': row['proportion'],
                        'Cell Count': row['count'],
                        'Total Cells': row['total']
                    })
        
        # Save dataset-specific summary table
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_csv = os.path.join(dataset_dir, f'cell_proportions_summary_{dataset}.csv')
            summary_df.to_csv(summary_csv, index=False)
            print(f"Saved {dataset} summary table to {summary_csv}")
    
    # Create overall summary table with all datasets
    print("\nCreating overall summary table of top cell types...")
    all_summary_rows = []
    
    for dataset in datasets:
        for structure in df[df['Dataset'] == dataset]['structure'].unique():
            cells = proportions[(proportions['Dataset'] == dataset) & 
                               (proportions['structure'] == structure)]
            
            if not cells.empty:
                # Get top 5 cell types
                top5 = cells.nlargest(5, 'proportion')
                
                for i, (_, row) in enumerate(top5.iterrows(), 1):
                    all_summary_rows.append({
                        'Dataset': dataset,
                        'Structure': structure,
                        'Rank': i,
                        'Cell Type': row['MemCatName'],
                        'Proportion': row['proportion'],
                        'Cell Count': row['count'],
                        'Total Cells': row['total']
                    })
    
    # Create and save the overall summary table
    if all_summary_rows:
        summary_df = pd.DataFrame(all_summary_rows)
        summary_csv = os.path.join(output_dir, 'cell_proportions_summary_all_datasets.csv')
        summary_df.to_csv(summary_csv, index=False)
        print(f"Saved overall summary table to {summary_csv}")
    
    # Create combined heatmap for all datasets with consistent color scaling
    print("Creating combined heatmap for all datasets...")
    combined_pivot = proportions.pivot_table(
        values='proportion',
        index=['Dataset', 'structure'],
        columns='MemCatName',
        fill_value=0
    )
    
    # Apply strict ordering from CELL_TYPE_ORDER
    # Get all cell types present in the data
    all_cell_types = combined_pivot.columns.tolist()
    
    # Create ordering: first those in CELL_TYPE_ORDER, then any others
    ordered_columns = [cell for cell in CELL_TYPE_ORDER if cell in all_cell_types]
    ordered_columns.extend([cell for cell in all_cell_types if cell not in CELL_TYPE_ORDER])
    
    # Apply the ordering
    combined_pivot = combined_pivot[ordered_columns]
    
    # Save the combined table
    combined_csv = os.path.join(output_dir, 'combined_cell_proportions.csv')
    combined_pivot.to_csv(combined_csv)
    
    # Create the combined heatmap with consistent color scaling
    # Calculate better figure dimensions for readability
    fig_width = max(16, min(36, len(ordered_columns) * 1.2))
    fig_height = max(10, len(combined_pivot) * 1.0)
    plt.figure(figsize=(fig_width, fig_height))
    
    # Use consistent color scaling for combined heatmap
    sns.heatmap(combined_pivot, annot=False, fmt='.1%', cmap='viridis', linewidths=.8,
               annot_kws={'size': 11, 'weight': 'bold'},
               cbar_kws={'shrink': 0.8, 'label': 'Proportion'},
               vmin=global_min, vmax=global_max)  # Consistent color scaling
    plt.title('Cell Type Proportions by Dataset and Structure', fontsize=16)
    plt.xlabel('Cell Type', fontsize=14)
    plt.ylabel('Dataset - Structure', fontsize=14)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=22)
    plt.tight_layout()
    
    combined_png = os.path.join(output_dir, 'combined_cell_proportions.png')
    plt.savefig(combined_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created combined heatmap: {combined_png}")
    
    print(f"\nCell proportion analysis complete! Results saved to {output_dir} directory")
    print("Dataset-specific results are in subdirectories: dataset_IgG4, dataset_AAV, dataset_NK2")


def analyze_structure_proportions(file_path, output_dir, min_cell_threshold=0):
    """
    Analyzes structure proportions for each cell type across different datasets.
    Creates heatmaps and tables to visualize the results with consistent color scaling.
    
    Args:
        file_path: Path to the CSV file containing the data
        output_dir: Directory where results will be saved
        min_cell_threshold: Minimum number of cells required to include a cell type (0 = include all)
    """
    # Fix the DtypeWarning by specifying low_memory=False
    print(f"Loading data for structure proportion analysis...")
    df = pd.read_csv(file_path, low_memory=False)
    
    # Calculate structure proportions for each cell type
    print("Calculating structure proportions for each cell type...")
    cell_structure_counts = df.groupby(['Dataset', 'MemCatName', 'structure']).size().reset_index(name='count')
    cell_totals = df.groupby(['Dataset', 'MemCatName']).size().reset_index(name='total')
    structure_proportions = pd.merge(cell_structure_counts, cell_totals, on=['Dataset', 'MemCatName'])
    structure_proportions['proportion'] = structure_proportions['count'] / structure_proportions['total']
    
    # Calculate global min and max for consistent color scaling
    global_min = structure_proportions['proportion'].min()
    global_max = structure_proportions['proportion'].max()
    print(f"Global structure proportion range: {global_min:.3f} to {global_max:.3f}")
    
    # Create structure proportion directory
    structure_dir = os.path.join(output_dir, 'structure_proportions')
    os.makedirs(structure_dir, exist_ok=True)
    
    # Save detailed results to CSV
    detailed_csv = os.path.join(structure_dir, 'structure_proportions_detailed.csv')
    print(f"Saving detailed structure proportions to {detailed_csv}...")
    structure_proportions.to_csv(detailed_csv, index=False)
    
    # Process each dataset separately
    datasets = df['Dataset'].unique()
    
    for dataset in datasets:
        print(f"\nProcessing structure proportions for {dataset} dataset...")
        
        # Create dataset-specific directory within structure_proportions
        dataset_structure_dir = os.path.join(structure_dir, f'dataset_{dataset}')
        os.makedirs(dataset_structure_dir, exist_ok=True)
        
        # Filter data for this dataset
        dataset_data = structure_proportions[structure_proportions['Dataset'] == dataset]
        
        # Create a pivot table
        pivot_table = dataset_data.pivot_table(
            values='proportion',
            index='MemCatName',
            columns='structure',
            fill_value=0
        )
        
        # Get the ordered index that matches CELL_TYPE_ORDER
        # Create ordering: first those in CELL_TYPE_ORDER, then any others
        available_cells = pivot_table.index.tolist()
        ordered_index = []
        
        # First add cells from our predefined order if they're in the data
        for cell in CELL_TYPE_ORDER:
            if cell in available_cells:
                ordered_index.append(cell)
        
        # Then add any cells not in our order
        for cell in available_cells:
            if cell not in CELL_TYPE_ORDER:
                ordered_index.append(cell)
        
        # Apply the strict ordering to create a new DataFrame with ordered rows
        if ordered_index:
            # Create new DataFrame with explicit row ordering and populated with zeros
            ordered_pivot = pd.DataFrame(0, index=ordered_index, columns=pivot_table.columns, dtype=float)
            
            # Fill with data from original pivot table
            for idx in ordered_index:
                if idx in pivot_table.index:
                    # Use column-by-column copying to avoid shape mismatch
                    for col in pivot_table.columns:
                        ordered_pivot.loc[idx, col] = pivot_table.loc[idx, col]
            
            # Use this ordered pivot for further processing
            pivot_table = ordered_pivot
            
            # Save the pivot table to CSV in dataset-specific directory
            dataset_csv = os.path.join(dataset_structure_dir, f'structure_proportions_{dataset}.csv')
            pivot_table.to_csv(dataset_csv)
            
            # Create and save the heatmap with consistent color scaling
            # Better figure sizing for readability
            fig_width = max(10, len(pivot_table.columns) * 2.0)
            fig_height = max(8, min(30, len(pivot_table) * 1.0))
            plt.figure(figsize=(fig_width, fig_height))
            
            # Enhanced visual styling with consistent color scaling
            sns.heatmap(pivot_table, annot=False, fmt='.1%', cmap='viridis', linewidths=.8,
                       annot_kws={'size': 12, 'weight': 'bold'},
                       cbar_kws={'shrink': 0.8, 'label': 'Proportion'},
                       vmin=global_min, vmax=global_max)  # Consistent color scaling
            plt.title(f'Structure Distribution for Cell Types in {dataset} Dataset', fontsize=16)
            plt.ylabel('Cell Type', fontsize=14)
            plt.xlabel('Structure', fontsize=14)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=22)
            plt.tight_layout()
            
            dataset_png = os.path.join(dataset_structure_dir, f'structure_proportions_{dataset}.png')
            plt.savefig(dataset_png, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Created structure proportion heatmap for {dataset}: {dataset_png}")
    
    # Create summary table showing where each cell type is most commonly found
    print("\nCreating summary table of cell type locations...")
    location_summary = []
    
    for dataset in datasets:
        dataset_cells = cell_totals[cell_totals['Dataset'] == dataset]
        
        for _, cell_row in dataset_cells.iterrows():
            cell_type = cell_row['MemCatName']
            cell_count = cell_row['total']
            
            # Get all structure proportions for this cell type
            structures = structure_proportions[
                (structure_proportions['Dataset'] == dataset) & 
                (structure_proportions['MemCatName'] == cell_type)
            ].sort_values('proportion', ascending=False)
            
            # Top structure (most common location)
            if not structures.empty:
                top_structure = structures.iloc[0]
                location_summary.append({
                    'Dataset': dataset,
                    'Cell Type': cell_type,
                    'Total Count': cell_count,
                    'Primary Structure': top_structure['structure'],
                    'Structure Proportion': top_structure['proportion'],
                    'Structure Count': top_structure['count']
                })
    
    # Create and save the location summary
    if location_summary:
        summary_df = pd.DataFrame(location_summary)
        summary_csv = os.path.join(structure_dir, 'cell_location_summary_all_datasets.csv')
        summary_df.to_csv(summary_csv, index=False)
        print(f"Saved cell location summary to {summary_csv}")
        
        # Also create dataset-specific location summaries
        for dataset in datasets:
            dataset_summary = summary_df[summary_df['Dataset'] == dataset]
            if not dataset_summary.empty:
                dataset_summary_csv = os.path.join(structure_dir, f'dataset_{dataset}', f'cell_location_summary_{dataset}.csv')
                dataset_summary.to_csv(dataset_summary_csv, index=False)
                print(f"Saved {dataset} location summary to {dataset_summary_csv}")
    
    # Create combined structure proportion heatmap for all datasets with consistent color scaling
    print("Creating combined structure proportion heatmap for all datasets...")
    
    # Get all structures and all cell types across all datasets
    all_datasets = df['Dataset'].unique()
    all_mems = df['MemCatName'].unique()
    all_structures = df['structure'].unique()
    
    # Create a combined structure proportion DataFrame with proper ordering
    # First build a properly ordered multi-index
    multi_idx = []
    for dataset in all_datasets:
        # Add cell types in our preferred order first
        for cell in CELL_TYPE_ORDER:
            if cell in all_mems and (dataset, cell) in cell_totals.set_index(['Dataset', 'MemCatName']).index:
                multi_idx.append((dataset, cell))
                
        # Then add any remaining cell types
        for cell in all_mems:
            if cell not in CELL_TYPE_ORDER and (dataset, cell) in cell_totals.set_index(['Dataset', 'MemCatName']).index:
                multi_idx.append((dataset, cell))
    
    # Create the multi-index
    multi_idx = pd.MultiIndex.from_tuples(multi_idx, names=['Dataset', 'MemCatName'])
    
    # Initialize a DataFrame with zeros - this is safer than NaNs
    combined_pivot = pd.DataFrame(0, index=multi_idx, columns=all_structures, dtype=float)
    
    # Fill in the data from our proportions
    for _, row in structure_proportions.iterrows():
        dataset = row['Dataset']
        mem_cat = row['MemCatName']
        structure = row['structure']
        proportion = row['proportion']
        
        if (dataset, mem_cat) in combined_pivot.index and structure in combined_pivot.columns:
            combined_pivot.loc[(dataset, mem_cat), structure] = proportion
    
    # Save the combined table
    combined_csv = os.path.join(structure_dir, 'combined_structure_proportions.csv')
    combined_pivot.to_csv(combined_csv)
    
    # Create the combined heatmap with consistent color scaling
    # Calculate appropriate figure dimensions
    row_count = len(combined_pivot)
    num_cols = combined_pivot.columns.size
    
    # More generous sizing to ensure annotations are readable
    fig_width = max(12, num_cols * 2.0)
    fig_height = max(12, min(40, row_count * 0.8))
    
    plt.figure(figsize=(fig_width, fig_height))
        
    # Enhanced visual styling with consistent color scaling
    sns.heatmap(combined_pivot, annot=False, fmt='.1%', cmap='viridis', linewidths=.8,
               annot_kws={'size': 11, 'weight': 'bold'},
               cbar_kws={'shrink': 0.8, 'label': 'Proportion'},
               vmin=global_min, vmax=global_max)  # Consistent color scaling
    plt.title('Structure Distribution by Cell Type and Dataset', fontsize=16)
    plt.ylabel('Dataset - Cell Type', fontsize=14)
    plt.xlabel('Structure', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=22)
    plt.tight_layout()
    
    combined_png = os.path.join(structure_dir, 'combined_structure_proportions.png')
    plt.savefig(combined_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Created combined structure proportion heatmap: {combined_png}")
    
    print(f"\nStructure proportion analysis complete! Results saved to {structure_dir} directory")
    print("Dataset-specific results are in subdirectories: dataset_IgG4, dataset_AAV, dataset_NK2")


if __name__ == "__main__":
    # Set your input file path and output directory here
    input_path = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/combined_SAM_scores_filtered_alldataset5_10woMonostructure.csv"
    output_dir = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/cell_proportion_per_structure5_10woMono"
    
    # Run both analyses with no minimum threshold to include all cell types
    analyze_cell_proportions(input_path, output_dir, min_cell_threshold=0)
    analyze_structure_proportions(input_path, output_dir, min_cell_threshold=0)
