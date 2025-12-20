import pandas as pd

def add_memcatname3_column(input_csv, output_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Create the mapping dictionary from MemCatName to MemCatName3
    mapping = {
        'NaiveCD4T': 'CD4T',
        'CD4T': 'CD4T',
        'ICOS+Treg': 'CD4T',
        'CD103+CD4Treg': 'CD4T',
        'HLAII+CD27+Treg': 'CD4T',
        'Tfh': 'CD4T',
        'TIGIT+Tfh': 'CD4T',
        'HLAII+GZMK+Tfh': 'CD4T',
        'GZMA+GZMK+CD4CTL': 'CD4T',
        'GZMB+CD4CTL': 'CD4T',
        'CD4Tcm-Tem': 'CD4T',
        'NaiveCD8T': 'CD8T',
        'CD8T': 'CD8T',
        'GZMA+GZMK+CD8CTL': 'CD8T',
        'GZMB+CD8CTL': 'CD8T',
        'CD103+CD8Treg': 'CD8T',
        'CD8Trm': 'CD8T',
        'CD103+CD8Trm': 'CD8T',
        'TIGIT+CD8T': 'CD8T',
        'CD8Tcm-Tem': 'CD8T',
        'γδT': 'OtherLymphoid',
        'NaiveB': 'Humoral',
        'TransitionalB': 'Humoral',
        'B': 'Humoral',
        'CD27+B': 'Humoral',
        'DN2B': 'Humoral',
        'AgeAsscB': 'Humoral',
        'Plasmablast': 'Humoral',
        'IgG4+Plasmablast': 'Humoral',
        'Plasma': 'Humoral',
        'IgG4+Plasma': 'Humoral',
        'M1Macrophage': 'Myeloid',
        'MPO+Macrophage': 'Myeloid',
        'M2Macrophage': 'Myeloid',
        'MerTK+M1Macrophage': 'Myeloid',
        'MerTK+M2Macrophage': 'Myeloid',
        'GPNMB+M1Macrophage': 'Myeloid',
        'GPNMB+M2Macrophage': 'Myeloid',  # Fixed "Maro" to "Macro"
        'cDC1': 'Myeloid',
        'cDC2': 'Myeloid',
        'pDC': 'Myeloid',
        'ILC': 'OtherLymphoid',
        'ImmatureNK': 'OtherLymphoid',  # Removed comma
        'MatureNK': 'OtherLymphoid',
        'NKT': 'OtherLymphoid',
        'Neutrophil': 'Myeloid',
        'Myofibroblast-SmoothMuscle': 'Stromal',
        'CD90+Myofibroblast': 'Stromal',
        'MMT': 'Stromal',
        'ProximalTubular': 'Structural',  # Fixed spacing issue
        'DistalTubular': 'Structural',
        'StressedTubular': 'Structural',
        'Endothelial': 'Structural',
        'LymphaticEndothelial': 'Structural',
        'Podocyte': 'Structural',
        'PEC': 'Structural',
        'Apoptotic': 'Others',
        'RBC': 'Others',
        'TubuleBackground': 'Others',
        'unclassified': 'Others'
    }
    
    # Add the MemCatName3 column based on the mapping
    df['MemCatName3'] = df['MemCatName'].map(mapping)
    
    # Handle any unmapped values (if any)
    if df['MemCatName3'].isna().any():
        print("Warning: Some MemCatName values could not be mapped:")
        unmapped = df[df['MemCatName3'].isna()]['MemCatName'].unique()
        print(unmapped)
        # Fill unmapped values with original values
        df['MemCatName3'] = df['MemCatName3'].fillna(df['MemCatName'])
    
    # Save the result to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Saved output to {output_csv}")

# Example usage
if __name__ == "__main__":
    input_file = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/processed_combined_SAM_scores_filtered_alldataset5_10woMonointerstitial.csv"  # Replace with your input file path
    output_file = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/processed_combined_SAM_scores_filtered_alldataset5_10woMonointerstitial_memcatname3.csv"  # Replace with your desired output file path
    add_memcatname3_column(input_file, output_file)
