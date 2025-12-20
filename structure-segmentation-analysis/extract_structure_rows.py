import pandas as pd

# Configuration - Modify these variables as needed
input_csv = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/combined_SAM_scores_filtered_alldataset5_10woMonostructure.csv"  # Change this to your input CSV file path
output_csv = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/combined_SAM_scores_filtered_alldataset5_10woMonoglomeruli.csv"  # Change this to your desired output file path
structures_to_extract = ['glomeruli']  # Choose from 'glomeruli', 'interstitial', 'tubules', 'vessels','outside the tissue'

# Read the CSV file
print(f"Reading {input_csv}...")
df = pd.read_csv(input_csv)

# Validate that 'structure' column exists
if 'structure' not in df.columns:
    print(f"Error: 'structure' column not found in {input_csv}")
    print(f"Available columns: {', '.join(df.columns)}")
else:
    # Filter rows based on structure values
    print(f"Filtering for structures: {', '.join(structures_to_extract)}")
    filtered_df = df[df['structure'].isin(structures_to_extract)]
    
    # Check if we have any matching rows
    if filtered_df.empty:
        print("No matching rows found!")
        print(f"Unique values in structure column: {df['structure'].unique()}")
    else:
        # Write the filtered data to a new CSV file
        print(f"Writing {len(filtered_df)} rows to {output_csv}")
        filtered_df.to_csv(output_csv, index=False)
        print("Done!")
