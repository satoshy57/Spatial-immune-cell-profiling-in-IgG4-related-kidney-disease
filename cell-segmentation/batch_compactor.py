import os
import pandas as pd

csv_p = '/project/mclark/SCAMPI_datasets/CODEX_img_information.csv'
df = pd.read_csv(csv_p)

def concat_csvs_in_directory(root_dir, output_file):
    # List to hold each DataFrame from CSVs
    dataframes = []
    
    # Traverse through directory and subdirectories
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            # Check if file is a CSV
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                # Read CSV and append to list
                df = pd.read_csv(file_path)
                dataframes.append(df)
    
    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    dd = '/project/mclark/SCAMPI_datasets/CODEX_img_information.csv'
    codex = pd.read_csv(dd)
    combined_df = combined_df.rename(columns={'Sample': 'Coverslip'})
    combined_df['Coverslip'] = combined_df['Coverslip'].astype(str)
    codex['Coverslip'] = codex['Coverslip'].astype(str)
    combined_df = combined_df.merge(codex[['Coverslip', 'Area', 'Staining']], on=['Coverslip', 'Area'], how='left')
    combined_df = combined_df.rename(columns={'Coverslip': 'ImagingBatch'})
    combined_df = combined_df.rename(columns={'Staining': 'StainBatch'})
    



    # Save to the specified output file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Save the file
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved as {output_file}")

# Example usage
concat_csvs_in_directory('/project/mclark/SCAMPI_datasets/AAV/cell_MPI', '/project/mclark/SCAMPI_datasets/AAV/combined_cell_feature_csv/combined_output_with_cents.csv')
