import os

import matplotlib
matplotlib.use("Agg")  # safe on headless/cluster nodes
import matplotlib.pyplot as plt
import pandas as pd

# Category "levels" to summarise
LEVEL_COLUMNS = ['MemCatName', 'MemCatName2', 'MemCatName3',
                 'CellCatName', 'CellCatName2', 'CellCatName3']

# (output folder, label for titles, filename suffix, MemCatName3 values to exclude)
SUBSETS = [
    ("counts_per_dataset", "", "", []),
    ("counts_per_dataset_no_others", "No Others", "_no_others", ["Others"]),
    ("counts_per_dataset_no_others_no_structural", "No Others/Structural",
     "_no_others_no_structural", ["Others", "Structural"]),
]


def analyze_kidney_cells(csv_path, output_dir=None):
    """
    Counts kidney cells overall, per dataset, and per patient (within each dataset)
    at every category level, for the full data and the two filtered subsets.

    Parameters:
    csv_path (str): Path to the CSV file containing kidney cell data
    output_dir (str): Directory to save output files (if None, uses 'kidney_cell_analysis')
    """
    if output_dir is None:
        output_dir = "kidney_cell_analysis"
    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(csv_path)
        print(f"Successfully read CSV file with {len(df)} total cells.")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    print("\nDataFrame columns:", df.columns.tolist())
    print("\nSample data (first 5 rows):")
    print(df.head())

    # ---- Overall counts -------------------------------------------------
    dataset_counts = _count(df, ['Dataset'])
    # Patient IDs may repeat across datasets, so count per Dataset + Patient
    patient_counts = _count(df, ['Dataset', 'Patient'])
    compname_counts = _count(df, ['CompName'])

    for title, table in [("Dataset", dataset_counts),
                         ("Dataset x Patient", patient_counts),
                         ("CompName", compname_counts)]:
        print(f"\n--- Cell Counts by {title} ---")
        print(table.to_string(index=False))
        print(f"\nTotal rows: {len(table)}")

    dataset_counts.to_csv(os.path.join(output_dir, 'cell_counts_by_dataset.csv'), index=False)
    patient_counts.to_csv(os.path.join(output_dir, 'cell_counts_by_patient.csv'), index=False)
    compname_counts.to_csv(os.path.join(output_dir, 'cell_counts_by_compname.csv'), index=False)
    print(f"\nOverall results saved to {output_dir}")

    create_visualizations(dataset_counts, patient_counts, compname_counts, output_dir)

    # ---- Per-level counts (all cells, then filtered subsets) ------------
    for folder, label, suffix, exclude in SUBSETS:
        analyze_levels(df, output_dir, folder, label, suffix, exclude)


def _count(df, group_cols):
    """Cell count per group, sorted by group then descending count."""
    out = df.groupby(group_cols).size().reset_index(name='Cell Count')
    sort_cols = group_cols[:-1] + ['Cell Count']
    ascending = [True] * (len(group_cols) - 1) + [False]
    return out.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def _add_percentage(counts, group_cols):
    """Add percentage of cells within each group (e.g. within each dataset or patient)."""
    totals = counts.groupby(group_cols)['Cell Count'].transform('sum')
    counts['Percentage'] = (counts['Cell Count'] / totals * 100).round(2)
    return counts


def analyze_levels(df, output_dir, folder, label, suffix, exclude):
    """
    Produces per-dataset and per-patient tables for every level in LEVEL_COLUMNS.

    Outputs in <output_dir>/<folder>/:
      cell_counts_<level>_per_dataset<suffix>.csv           (long: Dataset, level)
      per_patient/cell_counts_<level>_per_patient<suffix>.csv        (long: Dataset, Patient, level)
      per_patient/cell_counts_<level>_per_patient_wide<suffix>.csv   (rows = patients, cols = categories, counts + Total)
      per_patient/cell_counts_<level>_per_patient_wide_pct<suffix>.csv (same, as % of each patient's cells)
      plots/cell_counts_<level>_per_dataset<suffix>.png

    Parameters:
    df (DataFrame): The full dataframe with kidney cell data
    output_dir (str): Base output directory
    folder (str): Subfolder name for this subset
    label (str): Text appended to plot titles (empty for the full data)
    suffix (str): Suffix for filenames
    exclude (list): MemCatName3 values to remove before counting
    """
    header = f"CELL COUNTS PER DATASET AND PER PATIENT{' (' + label + ')' if label else ''}"
    print("\n" + "=" * 80)
    print(header)
    print("=" * 80)

    if exclude:
        if 'MemCatName3' not in df.columns:
            print("\nWarning: Column 'MemCatName3' not found. Skipping filtered analysis...")
            return
        df_use = df[~df['MemCatName3'].isin(exclude)].copy()
        removed = len(df) - len(df_use)
        print(f"\nOriginal total cells: {len(df)}")
        print(f"Cells removed ({', '.join(exclude)} in MemCatName3): {removed}")
        print(f"Remaining cells: {len(df_use)}")
        print(f"Percentage retained: {len(df_use) / len(df) * 100:.2f}%")
        if df_use.empty:
            print("\nNo data remaining after filtering. Skipping analysis.")
            return
    else:
        df_use = df

    subset_dir = os.path.join(output_dir, folder)
    patient_dir = os.path.join(subset_dir, 'per_patient')
    os.makedirs(patient_dir, exist_ok=True)

    for col in LEVEL_COLUMNS:
        if col not in df_use.columns:
            print(f"\nWarning: Column '{col}' not found in dataframe. Skipping...")
            continue

        print(f"\n{'=' * 80}\nAnalyzing: {col}{' (' + label + ')' if label else ''}\n{'=' * 80}")
        col_l = col.lower()

        # ---- per dataset (long) ----
        by_dataset = _count(df_use, ['Dataset', col])
        by_dataset = _add_percentage(by_dataset, ['Dataset'])
        path = os.path.join(subset_dir, f'cell_counts_{col_l}_per_dataset{suffix}.csv')
        by_dataset.to_csv(path, index=False)

        for dataset in sorted(by_dataset['Dataset'].unique()):
            sub = by_dataset[by_dataset['Dataset'] == dataset]
            print(f"\n--- {col} counts for Dataset: {dataset} ---")
            print(sub.to_string(index=False))
            print(f"Total unique {col} in {dataset}: {len(sub)}")
            print(f"Total cells in {dataset} for this category: {sub['Cell Count'].sum()}")

        # ---- per patient (long) ----
        by_patient = _count(df_use, ['Dataset', 'Patient', col])
        by_patient = _add_percentage(by_patient, ['Dataset', 'Patient'])
        by_patient.to_csv(
            os.path.join(patient_dir, f'cell_counts_{col_l}_per_patient{suffix}.csv'), index=False)

        # ---- per patient (wide: patients x categories) ----
        wide = by_patient.pivot_table(index=['Dataset', 'Patient'], columns=col,
                                      values='Cell Count', aggfunc='sum', fill_value=0)
        wide['Total'] = wide.sum(axis=1)
        wide.to_csv(os.path.join(patient_dir, f'cell_counts_{col_l}_per_patient_wide{suffix}.csv'))

        wide_pct = wide.drop(columns='Total').div(wide['Total'], axis=0).mul(100).round(2)
        wide_pct.to_csv(
            os.path.join(patient_dir, f'cell_counts_{col_l}_per_patient_wide_pct{suffix}.csv'))

        print(f"\nPer-patient table for {col}: {wide.shape[0]} patients x "
              f"{wide.shape[1] - 1} categories")
        print(wide.to_string())

        create_per_dataset_visualization(by_dataset, col, subset_dir, label, suffix)
        print(f"\nSaved tables for {col} to {subset_dir}")

    print(f"\n{'=' * 80}\nAll outputs for this subset saved to: {subset_dir}\n{'=' * 80}")


def create_per_dataset_visualization(data, column_name, output_dir, label="", suffix=""):
    """
    Bar charts of cell counts for one level, one panel per dataset.

    Parameters:
    data (DataFrame): Columns ['Dataset', column_name, 'Cell Count', ...]
    column_name (str): Category column being visualised
    output_dir (str): Directory in which a 'plots' folder is created
    label (str): Text appended to the panel titles
    suffix (str): Suffix for the output filename
    """
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    datasets = sorted(data['Dataset'].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(8 * len(datasets), 10), squeeze=False)

    for ax, dataset in zip(axes[0], datasets):
        sub = data[data['Dataset'] == dataset]
        ax.bar(range(len(sub)), sub['Cell Count'])
        ax.set_title(f'{column_name} - {dataset}' + (f' ({label})' if label else ''))
        ax.set_xlabel(column_name)
        ax.set_ylabel('Cell Count')
        ax.set_xticks(range(len(sub)))
        ax.set_xticklabels(sub[column_name], rotation=90, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'cell_counts_{column_name.lower()}_per_dataset{suffix}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def create_visualizations(dataset_counts, patient_counts, compname_counts, output_dir):
    """Bar charts for overall cell counts by Dataset, Patient, and CompName."""
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.bar(dataset_counts['Dataset'], dataset_counts['Cell Count'])
    plt.title('Cell Counts by Dataset')
    plt.xlabel('Dataset')
    plt.ylabel('Cell Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'cell_counts_by_dataset.png'))
    plt.close()

    # Patient labels prefixed with dataset so identical IDs in different datasets stay distinct
    labels = patient_counts['Dataset'].astype(str) + ' | ' + patient_counts['Patient'].astype(str)
    plt.figure(figsize=(14, 7))
    plt.bar(labels, patient_counts['Cell Count'])
    plt.title('Cell Counts by Patient')
    plt.xlabel('Dataset | Patient')
    plt.ylabel('Cell Count')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'cell_counts_by_patient.png'))
    plt.close()

    plt.figure(figsize=(16, 8))
    plt.bar(compname_counts['CompName'], compname_counts['Cell Count'])
    plt.title('Cell Counts by CompName')
    plt.xlabel('CompName')
    plt.ylabel('Cell Count')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'cell_counts_by_compname.png'))
    plt.close()

    print(f"Visualizations saved to {plots_dir}")


if __name__ == "__main__":
    csv_file_path = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/processed_combined_SAM_scores_filtered_alldataset5_10woMonointerstitial_memcatname23cellcatname23.csv"
    output_directory = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/cellcounts_alllevels_perpatient"

    analyze_kidney_cells(csv_file_path, output_directory)
