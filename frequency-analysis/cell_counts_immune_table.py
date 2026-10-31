import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_kidney_cells(csv_path, output_dir=True):
    """
    Analyzes kidney cell data to count the number of cells by different groupings.
    
    Parameters:
    csv_path (str): Path to the CSV file containing kidney cell data
    output_dir (str): Directory to save output files (if None, uses current directory)
    """
    # Set up output directory
    if output_dir is None:
        output_dir = "kidney_cell_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully read CSV file with {len(df)} total cells.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Display basic info about the dataframe
    print("\nDataFrame columns:", df.columns.tolist())
    print("\nSample data (first 5 rows):")
    print(df.head())
    
    # Count cells by Dataset
    dataset_counts = df['Dataset'].value_counts().reset_index()
    dataset_counts.columns = ['Dataset', 'Cell Count']
    dataset_counts = dataset_counts.sort_values('Cell Count', ascending=False)
    
    # Count cells by Patient
    patient_counts = df['Patient'].value_counts().reset_index()
    patient_counts.columns = ['Patient', 'Cell Count']
    patient_counts = patient_counts.sort_values('Cell Count', ascending=False)
    
    # Count cells by CompName
    compname_counts = df['CompName'].value_counts().reset_index()
    compname_counts.columns = ['CompName', 'Cell Count']
    compname_counts = compname_counts.sort_values('Cell Count', ascending=False)
    
    # Display results
    print("\n--- Cell Counts by Dataset ---")
    print(dataset_counts.to_string(index=False))
    print(f"\nTotal Datasets: {len(dataset_counts)}")
    
    print("\n--- Cell Counts by Patient ---")
    print(patient_counts.to_string(index=False))
    print(f"\nTotal Patients: {len(patient_counts)}")
    
    print("\n--- Cell Counts by CompName ---")
    if len(compname_counts) > 20:
        print(compname_counts.head(20).to_string(index=False))
        print(f"... and {len(compname_counts) - 20} more CompNames")
    else:
        print(compname_counts.to_string(index=False))
    print(f"\nTotal CompNames: {len(compname_counts)}")
    
    # Save results to CSV files
    dataset_csv_path = os.path.join(output_dir, 'cell_counts_by_dataset.csv')
    patient_csv_path = os.path.join(output_dir, 'cell_counts_by_patient.csv')
    compname_csv_path = os.path.join(output_dir, 'cell_counts_by_compname.csv')
    
    dataset_counts.to_csv(dataset_csv_path, index=False)
    patient_counts.to_csv(patient_csv_path, index=False)
    compname_counts.to_csv(compname_csv_path, index=False)
    
    print(f"\nResults have been saved to CSV files in {output_dir}")
    
    # Analyze immune cells
    analyze_immune_cells(df, output_dir)
    
    # Create visualizations
    create_visualizations(dataset_counts, patient_counts, compname_counts, output_dir)

def analyze_immune_cells(df, output_dir):
    """
    Analyzes immune vs non-immune cell counts per dataset.
    
    Parameters:
    df (DataFrame): The kidney cell dataframe
    output_dir (str): Directory to save output files
    """
    # Define non-immune cell types
    non_immune_cells = [
        'Myofibroblast-SmoothMuscle', 'CD90+Myofibroblast', 'MMT',
        'Interstitialfibrosis', 'Storiformfibrosis', 'ProximalTubular',
        'DistalTubular', 'StressedTubular', 'Endothelial',
        'LymphaticEndothelial', 'Podocyte', 'PEC', 'Apoptotic',
        'RBC', 'TubuleBackground','unclassified'
    ]
    
    # Add immune cell classification
    df['Cell_Type'] = df['MemCatName'].apply(
        lambda x: 'Non-Immune' if x in non_immune_cells else 'Immune'
    )
    
    # Count immune vs non-immune cells per dataset
    immune_counts = df.groupby(['Dataset', 'Cell_Type']).size().unstack(fill_value=0)
    immune_counts['Total'] = immune_counts.sum(axis=1)
    immune_counts['Immune_Percentage'] = (immune_counts['Immune'] / immune_counts['Total'] * 100).round(2)
    immune_counts['Non-Immune_Percentage'] = (immune_counts['Non-Immune'] / immune_counts['Total'] * 100).round(2)
    
    # Reset index for better display
    immune_counts_display = immune_counts.reset_index()
    
    print("\n--- Immune vs Non-Immune Cell Counts by Dataset ---")
    print(immune_counts_display.to_string(index=False))
    
    # Save to CSV
    immune_csv_path = os.path.join(output_dir, 'immune_cell_counts_by_dataset.csv')
    immune_counts_display.to_csv(immune_csv_path, index=False)
    print(f"\nImmune cell counts saved to {immune_csv_path}")
    
    # Create detailed MemCatName counts and percentages per dataset
    create_memcatname_table(df, output_dir)

def create_memcatname_table(df, output_dir):
    """
    Creates a detailed table of cell counts and percentages for each MemCatName per dataset.
    
    Parameters:
    df (DataFrame): The kidney cell dataframe
    output_dir (str): Directory to save output files
    """
    # Count cells by Dataset and MemCatName
    memcat_by_dataset = df.groupby(['Dataset', 'MemCatName']).size().reset_index(name='Count')
    
    # Calculate total cells per dataset
    dataset_totals = df.groupby('Dataset').size().reset_index(name='Total')
    
    # Merge to calculate percentages
    memcat_by_dataset = memcat_by_dataset.merge(dataset_totals, on='Dataset')
    memcat_by_dataset['Percentage'] = (memcat_by_dataset['Count'] / memcat_by_dataset['Total'] * 100).round(2)
    
    # Select and order columns
    memcat_by_dataset = memcat_by_dataset[['Dataset', 'MemCatName', 'Count', 'Percentage', 'Total']]
    memcat_by_dataset = memcat_by_dataset.sort_values(['Dataset', 'Count'], ascending=[True, False])
    
    print("\n--- Cell Counts and Percentages by MemCatName per Dataset ---")
    print(f"(Showing first 50 rows of {len(memcat_by_dataset)} total rows)")
    print(memcat_by_dataset.head(50).to_string(index=False))
    
    # Save to CSV
    memcat_csv_path = os.path.join(output_dir, 'memcatname_counts_by_dataset.csv')
    memcat_by_dataset.to_csv(memcat_csv_path, index=False)
    print(f"\nMemCatName counts and percentages saved to {memcat_csv_path}")
    
    # Create a pivot table for easier viewing (wide format)
    pivot_counts = df.groupby(['Dataset', 'MemCatName']).size().unstack(fill_value=0)
    pivot_counts['Total'] = pivot_counts.sum(axis=1)
    
    # Save pivot table
    pivot_csv_path = os.path.join(output_dir, 'memcatname_counts_pivot.csv')
    pivot_counts.to_csv(pivot_csv_path)
    print(f"MemCatName counts pivot table saved to {pivot_csv_path}")
    
    # Create percentage pivot table
    pivot_percentages = pivot_counts.div(pivot_counts['Total'], axis=0) * 100
    pivot_percentages = pivot_percentages.round(2)
    
    pivot_pct_csv_path = os.path.join(output_dir, 'memcatname_percentages_pivot.csv')
    pivot_percentages.to_csv(pivot_pct_csv_path)
    print(f"MemCatName percentages pivot table saved to {pivot_pct_csv_path}")

def create_visualizations(dataset_counts, patient_counts, compname_counts, output_dir):
    """
    Creates bar charts for cell counts by different groupings.
    
    Parameters:
    dataset_counts (DataFrame): Cell counts by Dataset
    patient_counts (DataFrame): Cell counts by Patient
    compname_counts (DataFrame): Cell counts by CompName
    output_dir (str): Directory to save visualization files
    """
    # Create directory for plots
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot cell counts by Dataset
    plt.figure(figsize=(12, 6))
    plt.bar(dataset_counts['Dataset'], dataset_counts['Cell Count'])
    plt.title('Cell Counts by Dataset')
    plt.xlabel('Dataset')
    plt.ylabel('Cell Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'cell_counts_by_dataset.png'))
    plt.close()
    
    # Plot cell counts by Patient (top 20 if more than 20)
    plt.figure(figsize=(14, 7))
    if len(patient_counts) > 20:
        patient_plot_data = patient_counts.head(20)
    else:
        patient_plot_data = patient_counts
    
    plt.bar(patient_plot_data['Patient'], patient_plot_data['Cell Count'])
    plt.title('Cell Counts by Patient' + (' (Top 20)' if len(patient_counts) > 20 else ''))
    plt.xlabel('Patient')
    plt.ylabel('Cell Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'cell_counts_by_patient.png'))
    plt.close()
    
    # Plot cell counts by CompName (top 20 if more than 20)
    plt.figure(figsize=(16, 8))
    if len(compname_counts) > 20:
        compname_plot_data = compname_counts.head(20)
    else:
        compname_plot_data = compname_counts
    
    plt.bar(compname_plot_data['CompName'], compname_plot_data['Cell Count'])
    plt.title('Cell Counts by CompName' + (' (Top 20)' if len(compname_counts) > 20 else ''))
    plt.xlabel('CompName')
    plt.ylabel('Cell Count')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'cell_counts_by_compname.png'))
    plt.close()
    
    print(f"Visualizations saved to {plots_dir}")

if __name__ == "__main__":
    # Replace with the actual path to your CSV file
    csv_file_path = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/combined_SAM_scores_filtered_alldataset5_10woMonostructure.csv"
    
    # Specify output directory (optional)
    output_directory = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/immune_cell_counts_total_revised"
    
    analyze_kidney_cells(csv_file_path, output_directory)
