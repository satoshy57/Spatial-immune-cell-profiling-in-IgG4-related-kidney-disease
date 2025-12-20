import pandas as pd

def add_memcatname2_column(input_csv, output_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Create the mapping dictionary from MemCatName to MemCatName2
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
        'γδT': 'γδT',
        'NaiveB': 'B',
        'TransitionalB': 'B',
        'B': 'B',
        'CD27+B': 'B',
        'DN2B': 'B',
        'AgeAsscB': 'B',
        'Plasmablast': 'Plasma',
        'IgG4+Plasmablast': 'Plasma',
        'Plasma': 'Plasma',
        'IgG4+Plasma': 'Plasma',
        'M1Macrophage': 'Macrophages',
        'MPO+Macrophage': 'Macrophages',
        'M2Macrophage': 'Macrophages',
        'MerTK+M1Macrophage': 'Macrophages',
        'MerTK+M2Macrophage': 'Macrophages',
        'GPNMB+M1Macrophage': 'Macrophages',
        'GPNMB+M2Macrophage': 'Macrophages',  # Fixed "Maro" to "Macro"
        'cDC1': 'OthercDCs',
        'cDC2': 'cDC2',
        'pDC': 'pDC',
        'ILC': 'OtherLymph',
        'ImmatureNK': 'NK',  # Removed comma
        'MatureNK': 'NK',
        'NKT': 'NKT',
        'Neutrophil': 'Neutrophils',
        'Myofibroblast-SmoothMuscle': 'SMC/MyoF',
        'CD90+Myofibroblast': 'SMC/MyoF',
        'MMT': 'SMC/MyoF',
        'ProximalTubular': 'Tubules',  # Fixed spacing issue
        'DistalTubular': 'Tubules',
        'StressedTubular': 'Tubules',
        'Endothelial': 'Endothelial',
        'LymphaticEndothelial': 'Lymphatic',
        'Podocyte': 'Podocytes',
        'PEC': 'PEC',
        'Apoptotic': 'Apoptotic',
        'RBC': 'RBC',
        'TubuleBackground': 'Tubules',
        'unclassified': 'unclassified'
    }
    
    # Add the MemCatName2 column based on the mapping
    df['MemCatName2'] = df['MemCatName'].map(mapping)
    
    # Handle any unmapped values (if any)
    if df['MemCatName2'].isna().any():
        print("Warning: Some MemCatName values could not be mapped:")
        unmapped = df[df['MemCatName2'].isna()]['MemCatName'].unique()
        print(unmapped)
        # Fill unmapped values with original values
        df['MemCatName2'] = df['MemCatName2'].fillna(df['MemCatName'])
    
    # Save the result to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Saved output to {output_csv}")

# Example usage
if __name__ == "__main__":
    input_file = "/project/mclark/SCAMPI_datasets/IgG4/classify_by_dilation-sameTH/with-MPI/combined_SAM_scores_filtered10_10woMono.csv"  # Replace with your input file path
    output_file = "/project/mclark/SCAMPI_datasets/IgG4/classify_by_dilation-sameTH/with-MPI/combined_SAM_scores_filtered10_10woMono_memcatname2.csv"  # Replace with your desired output file path
    add_memcatname2_column(input_file, output_file)
