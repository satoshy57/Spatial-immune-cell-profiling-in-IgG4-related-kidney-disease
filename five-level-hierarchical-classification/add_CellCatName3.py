import pandas as pd
import re

def add_cellcatname3_column(input_csv, output_csv):
    # Read the CSV file
    df = pd.read_csv(input_csv)
    
    # Create the mapping dictionary from CellCatName to CellCatName3
    mapping = {
        # NaiveCD4T category
        "NaiveCD4T": "NaiveCD4T",
        "Apoptotic": "Apoptotic",
        
        # CD4T related categories
        "OtherCD4T": "OtherCD4T",
        "CD4Treg": "CD4Treg",
        "Stem-likeCD4Treg": "CD4Treg",
        "ProgenitorExhaustedCD4Treg": "CD4Treg",
        "ExhaustedCD4Treg": "CD4Treg",
        "Th1": "Th1",
        "Stem-likeTh1": "Th1",
        "ProgenitorExhaustedTh1": "Th1",
        "ExhaustedTh1": "Th1",
        "ProliferatingTfh-like": "MKI+CD4T",
        "Stem-likeCD4T": "OtherCD4T",
        "ProgenitorExhaustedCD4T": "OtherCD4T",
        "ExhaustedCD4T": "OtherCD4T",
        
        # CD4Tcm-Tem related categories
        "CD4Tcm-Tem": "CD4Tcm-Tem",
        "CD4+iTreg": "CD4+iTreg",
        "Stem-likeCD4+iTreg": "CD4+iTreg",
        "ProgenitorExhaustedCD4+iTreg": "CD4+iTreg",
        "ExhaustedCD4+iTreg": "CD4+iTreg",
        "Th1-EM": "Th1-EM",
        "Stem-likeTh1-EM": "Th1-EM",
        "ProgenitorExhaustedTh1-EM": "Th1-EM",
        "ExhaustedTh1-EM": "Th1-EM",
        "Stem-likeCD4Tcm-Tem": "CD4Tcm-Tem",
        "ProgenitorExhaustedCD4Tcm-Tem": "CD4Tcm-Tem",
        "ExhaustedCD4Tcm-Tem": "CD4Tcm-Tem",
        
        # ICOS+Treg related categories
        "CD4Trm": "CD4Trm",
        "ICOS+Treg": "ICOS+Treg",
        "Stem-likeICOS+Treg": "ICOS+Treg",
        "ProgenitorExhaustedICOS+Treg": "ICOS+Treg",
        "ExhaustedICOS+Treg": "ICOS+Treg",
        "Th1-RM": "Th1-RM",
        "Stem-likeTh1-RM": "Th1-RM",
        "ProgenitorExhaustedTh1-RM": "Th1-RM",
        "ExhaustedTh1-RM": "Th1-RM",
        "Stem-likeCD4Trm": "CD4Trm",
        "ProgenitorExhaustedCD4Trm": "CD4Trm",
        "ExhaustedCD4Trm": "CD4Trm",
        
        # CD103+CD4Treg related categories
        "CD103+CD4T": "CD103+CD4T",
        "CD103+CD4Treg": "CD103+CD4Treg",
        "Stem-likeCD103+CD4Treg": "CD103+CD4Treg",
        "ProgenitorExhaustedCD103+CD4+Treg": "CD103+CD4Treg",
        "ExhaustedCD103+CD4+Treg": "CD103+CD4Treg",
        "ProliferatingCD103+CD4T": "MKI+CD103+CD4T",
        "Stem-likeCD103+CD4T": "CD103+CD4T",
        "ProgenitorExhaustedCD103+CD4T": "CD103+CD4T",
        "ExhaustedCD103+CD4T": "CD103+CD4T",
        
        # HLAII+CD27+Treg related categories
        "HLAII+CD27+CD4T": "HLAII+CD27+CD4T",
        "HLAII+CD27+Treg": "HLAII+CD27+Treg",
        "Stem-likeHLAII+CD27+Treg": "HLAII+CD27+Treg",
        "ProgenitorExhaustedHLAII+CD27+Treg": "HLAII+CD27+Treg",
        "ExhaustedHLAII+CD27+Treg": "HLAII+CD27+Treg",
        "HLAII+CD27+Th1-like": "HLAII+CD27+Th1-like",
        "Stem-likeHLAII+CD27+Th1-like": "HLAII+CD27+Th1-like",
        "ProgenitorExhaustedHLAII+CD27+Th1-like": "HLAII+CD27+Th1-like",
        "ExhaustedHLAII+CD27+Th1-like": "HLAII+CD27+Th1-like",
        "ProliferatingHLAII+CD27+CD4T": "MKI+HLAII+CD27+CD4T",
        "Stem-likeHLAII+CD27+CD4T": "HLAII+CD27+CD4T",
        "ProgenitorExhaustedHLAII+CD27+CD4T": "HLAII+CD27+CD4T",
        "ExhaustedHLAII+CD27+CD4T": "HLAII+CD27+CD4T",
        
        # Tfh related categories
        "Tfh": "Tfh",
        "Tfr": "Tfr",
        "Stem-likeTfr": "Tfr",
        "ProgenitorExhaustedTfr": "Tfr",
        "ExhaustedTfr": "Tfr",
        "Tfh1": "Tfh1",
        "Stem-likeTfh1": "Tfh1",
        "ProgenitorExhaustedTfh1": "Tfh1",
        "ExhaustedTfh1": "Tfh1",
        "MKI+Tfh": "MKI+Tfh",
        "Stem-likeTfh": "Tfh",
        "ProgenitorExhaustedTfh": "Tfh",
        "ExhaustedTfh": "Tfh",
        
        # TIGIT+Tfh related categories
        "TIGIT+Tfh": "TIGIT+Tfh",
        "TIGIT+Tfr": "TIGIT+Tfr",
        "Stem-likeTIGIT+Tfr": "TIGIT+Tfr",
        "ProgenitorExhaustedTIGIT+Tfr": "TIGIT+Tfr",
        "ExhaustedTIGIT+Tfr": "TIGIT+Tfr",
        "TIGIT+Tfh1": "TIGIT+Tfh1",
        "Stem-likeTIGIT+Tfh1": "TIGIT+Tfh1",
        "ProgenitorExhaustedTIGIT+Tfh1": "TIGIT+Tfh1",
        "ExhaustedTIGIT+Tfh1": "TIGIT+Tfh1",
        "MKI+TIGIT+Tfh": "MKI+TIGIT+Tfh",
        "Stem-likeTIGIT+Tfh": "TIGIT+Tfh",
        "ProgenitorExhaustedTIGIT+Tfh": "TIGIT+Tfh",
        "ExhaustedTIGIT+Tfh": "TIGIT+Tfh",
        
        # HLAII+GZMK+Tfh related categories
        "HLAII+GZMK+Tfh": "HLAII+GZMK+Tfh",
        "HLAII+GZMK+Tfr": "HLAII+GZMK+Tfr",
        "Stem-likeHLAII+GZMK+Tfr": "HLAII+GZMK+Tfr",
        "ProgenitorExhaustedHLAII+GZMK+Tfr": "HLAII+GZMK+Tfr",
        "ExhaustedHLAII+GZMK+Tfr": "HLAII+GZMK+Tfr",
        "HLAII+GZMK+Tfh1": "HLAII+GZMK+Tfh1",
        "Stem-likeHLAII+GZMK+Tfh1": "HLAII+GZMK+Tfh1",
        "ProgenitorExhaustedHLAII+GZMK+Tfh1": "HLAII+GZMK+Tfh1",
        "ExhaustedHLAII+GZMK+Tfh1": "HLAII+GZMK+Tfh1",
        "MKI+HLAII+GZMK+Tfh": "MKI+HLAII+GZMK+Tfh",
        "Stem-likeHLAII+GZMK+Tfh": "HLAII+GZMK+Tfh",
        "ProgenitorExhaustedHLAII+GZMK+Tfh": "HLAII+GZMK+Tfh",
        "ExhaustedHLAII+GZMK+Tfh": "HLAII+GZMK+Tfh",
        
        # GZMA+GZMK+CD4CTL related categories
        "GZMA+GZMK+CD4CTL": "GZMA+GZMK+CD4CTL",
        "GZMA+GZMK+CD4Treg": "GZMA+GZMK+CD4Treg",
        "Stem-likeGZMA+GZMK+CD4Treg": "GZMA+GZMK+CD4Treg",
        "ProgenitorExhaustedGZMA+GZMK+CD4Treg": "GZMA+GZMK+CD4Treg",
        "ExhaustedGZMA+GZMK+CD4Treg": "GZMA+GZMK+CD4Treg",
        "GZMA+GZMK+Th1-EM": "GZMA+GZMK+Th1-EM",
        "Stem-likeGZMA+GZMK+Th1-EM": "GZMA+GZMK+Th1-EM",
        "ProgenitorExhaustedGZMA+GZMK+Th1-EM": "GZMA+GZMK+Th1-EM",
        "ExhaustedGZMA+GZMK+Th1-EM": "GZMA+GZMK+Th1-EM",
        "MKI+GZMA+GZMK+CD4CTL": "MKI+GZMA+GZMK+CD4CTL",
        "Stem-likeGZMA+GZMK+CD4CTL": "GZMA+GZMK+CD4CTL",
        "ProgenitorExhaustedGZMA+GZMK+CD4CTL": "GZMA+GZMK+CD4CTL",
        "ExhaustedGZMA+GZMK+CD4CTL": "GZMA+GZMK+CD4CTL",
        
        # GZMB+CD4CTL related categories
        "GZMB+CD4CTL": "GZMB+CD4CTL",
        "GZMB+CD4Treg": "GZMB+CD4Treg",
        "Stem-likeGZMB+CD4Treg": "GZMB+CD4Treg",
        "ProgenitorExhaustedGZMB+CD4Treg": "GZMB+CD4Treg",
        "ExhaustedGZMB+CD4Treg": "GZMB+CD4Treg",
        "GZMB+Th1-EM": "GZMB+Th1-EM",
        "Stem-likeGZMB+Th1-EM": "GZMB+Th1-EM",
        "ProgenitorExhaustedGZMB+Th1-EM": "GZMB+Th1-EM",
        "ExhaustedGZMB+Th1-EM": "GZMB+Th1-EM",
        "MKI+GZMB+CD4CTL": "MKI+GZMB+CD4CTL",
        "Stem-likeGZMB+CD4CTL": "GZMB+CD4CTL",
        "ProgenitorExhaustedGZMB+CD4CTL": "GZMB+CD4CTL",
        "ExhaustedGZMB+CD4CTL": "GZMB+CD4CTL",
        
        # NaiveCD8T related categories
        "NaiveCD8T": "NaiveCD8T",
        
        # CD8T related categories
        "otherCD8T": "otherCD8T",
        "CD8Treg": "CD8Treg",
        "Stem-likeCD8Treg": "CD8Treg",
        "ProgenitorExhaustedCD8Treg": "CD8Treg",
        "ExhaustedCD8Treg": "CD8Treg",
        "Tc1": "Tc1",
        "Stem-likeTc1": "Tc1",
        "ProgenitorExhaustedTc1": "Tc1",
        "ExhaustedTc1": "Tc1",
        "MKI+CD8T": "MKI+CD8T",
        "Stem-likeCD8T": "otherCD8T",
        "ExhaustedCD8T": "otherCD8T",
        "ProgenitorExhaustedCD8T": "otherCD8T",
        
        # GZMA+GZMK+CD8CTL related categories
        "GZMA+GZMK+CD8CTL": "GZMA+GZMK+CD8CTL",
        "GZMA+GZMK+CD8Treg": "GZMA+GZMK+CD8Treg",
        "Stem-likeGZMA+GZMK+CD8Treg": "GZMA+GZMK+CD8Treg",
        "ProgenitorExhaustedGZMA+GZMK+CD8Treg": "GZMA+GZMK+CD8Treg",
        "ExhaustedGZMA+GZMK+CD8Treg": "GZMA+GZMK+CD8Treg",
        "GZMA+GZMK+Tc1-EM": "GZMA+GZMK+Tc1-EM",
        "Stem-likeGZMA+GZMK+Tc1-EM": "GZMA+GZMK+Tc1-EM",
        "ProgenitorExhaustedGZMA+GZMK+Tc1-EM": "GZMA+GZMK+Tc1-EM",
        "ExhaustedGZMA+GZMK+Tc1-EM": "GZMA+GZMK+Tc1-EM",
        "MKI+GZMA+GZMK+CD8CTL": "MKI+CD8T",
        "Stem-likeGZMA+GZMK+CD8CTL": "GZMA+GZMK+CD8CTL",
        "ProgenitorExhaustedGZMA+GZMK+CD8CTL": "GZMA+GZMK+CD8CTL",
        "ExhaustedGZMA+GZMK+CD8CTL": "GZMA+GZMK+CD8CTL",
        
        # GZMB+CD8CTL related categories
        "GZMB+CD8CTL": "GZMB+CD8CTL",
        "GZMB+CD8Treg": "GZMB+CD8Treg",
        "Stem-likeGZMB+CD8Treg": "GZMB+CD8Treg",
        "ProgenitorExhaustedGZMB+CD8Treg": "GZMB+CD8Treg",
        "ExhaustedGZMB+CD8Treg": "GZMB+CD8Treg",
        "GZMB+Tc1-EM": "GZMB+Tc1-EM",
        "Stem-likeGZMB+Tc1-EM": "GZMB+Tc1-EM",
        "ProgenitorExhaustedGZMB+Tc1-EM": "GZMB+Tc1-EM",
        "ExhaustedGZMB+Tc1-EM": "GZMB+Tc1-EM",
        "MKI+GZMB+CD8CTL": "MKI+CD8T",
        "Stem-likeGZMB+CD8CTL": "GZMB+CD8CTL",
        "ProgenitorExhaustedGZMB+CD8CTL": "GZMB+CD8CTL",
        "ExhaustedGZMB+CD8CTL": "GZMB+CD8CTL",
        
        # CD103+CD8Treg related categories
        "CD103+CD8Treg": "CD103+CD8Treg",
        "Stem-likeCD103+CD8Treg": "CD103+CD8Treg",
        "ProgenitorExhaustedCD103+CD8Treg": "CD103+CD8Treg",
        "ExhaustedCD103+CD8Treg": "CD103+CD8Treg",
        "CD103+Tc1-like": "CD103+Tc1-like",
        "Stem-likeCD103+Tc1-like": "CD103+Tc1-like",
        "ProgenitorExhaustedCD103+Tc1-like": "CD103+Tc1-like",
        "ExhaustedCD103+Tc1-like": "CD103+Tc1-like",
        "ProliferatingCD103+CD8Treg": "CD103+CD8Treg",
        
        # CD8Trm related categories
        "CD8Trm": "CD8Trm",
        "CD69+CD8Treg": "CD69+CD8Treg",
        "Stem-likeCD69+CD8Treg": "CD69+CD8Treg",
        "ProgenitorExhaustedCD69+CD8Treg": "CD69+CD8Treg",
        "ExhaustedCD69+CD8Treg": "CD69+CD8Treg",
        "Tbet+CD8Trm": "Tbet+CD8Trm",
        "Stem-likeTbet+CD8Trm": "Tbet+CD8Trm",
        "ProgenitorExhaustedTbet+CD8Trm": "Tbet+CD8Trm",
        "ExhaustedTbet+CD8Trm": "Tbet+CD8Trm",
        "ProliferatingCD8Trm": "MKI+CD8T",
        "Stem-likeCD8Trm": "CD8Trm",
        "ProgenitorExhaustedCD8Trm": "CD8Trm",
        "ExhaustedCD8Trm": "CD8Trm",
        
        # CD103+CD8Trm related categories
        "CD103+CD8Trm": "CD103+CD8Trm",
        "CD8Trm-Treg": "CD8Trm-Treg",
        "Stem-likeCD8Trm-Treg": "CD8Trm-Treg",
        "ProgenitorExhaustedCD8Trm-Treg": "CD8Trm-Treg",
        "ExhaustedCD8Trm-Treg": "CD8Trm-Treg",
        "Tbet+CD103+CD8Trm": "Tbet+CD103+CD8Trm",
        "Stem-likeTbet+CD103+CD8Trm": "Tbet+CD103+CD8Trm",
        "ProgenitorExhaustedTbet+CD103+CD8Trm": "Tbet+CD103+CD8Trm",
        "ExhaustedTbet+CD103+CD8Trm": "Tbet+CD103+CD8Trm",
        "ProliferatingCD103+CD8Trm": "MKI+CD8T",
        "Stem-likeCD103+CD8Trm": "CD103+CD8Trm",
        "ProgenitorExhaustedCD103+CD8Trm": "CD103+CD8Trm",
        "ExhaustedCD103+CD8Trm": "CD103+CD8Trm",
        
        # TIGIT+CD8T related categories
        "TIGIT+CD8T": "TIGIT+CD8T",
        "TIGIT+CD8Treg": "TIGIT+CD8Treg",
        "Stem-likeTIGIT+CD8Treg": "TIGIT+CD8Treg",
        "ProgenitorExhaustedTIGIT+CD8Treg": "TIGIT+CD8Treg",
        "ExhaustedTIGIT+CD8Treg": "TIGIT+CD8Treg",
        "TIGIT+Tc1": "TIGIT+Tc1",
        "Stem-likeTIGIT+Tc1": "TIGIT+Tc1",
        "ProgenitorExhaustedTIGIT+Tc1": "TIGIT+Tc1",
        "ExhaustedTIGIT+Tc1": "TIGIT+Tc1",
        "MKI+TIGIT+CD8T": "MKI+CD8T",
        "Stem-likeTIGIT+CD8T": "TIGIT+CD8T",
        "ProgenitorExhaustedTIGIT+CD8T": "TIGIT+CD8T",
        "ExhaustedTIGIT+CD8T": "TIGIT+CD8T",
        
        # CD8Tcm-Tem related categories
        "CD8Tcm-Tem": "CD8Tcm-Tem",
        "CD8+iTreg": "CD8+iTreg",
        "Stem-likeCD8+iTreg": "CD8+iTreg",
        "ProgenitorExhaustedCD8+iTreg": "CD8+iTreg",
        "ExhaustedCD8+iTreg": "CD8+iTreg",
        "Tc1-EM": "Tc1-EM",
        "Stem-likeTc1-EM": "Tc1-EM",
        "ProgenitorExhaustedTc1-EM": "Tc1-EM",
        "ExhaustedTc1-EM": "Tc1-EM",
        "ProliferatingCD8Tcm-Tem": "MKI+CD8T",
        "Stem-likeCD8Tcm-Tem": "CD8Tcm-Tem",
        "ProgenitorExhaustedCD8cm-Tem": "CD8Tcm-Tem",
        "ExhaustedCD8Tcm-Tem": "CD8Tcm-Tem",
        
        # γδT related categories
        "γδT": "γδT",
        "γδTreg": "γδTreg",
        "Stem-likeγδTreg": "γδTreg",
        "ProgenitorExhaustedγδTreg": "γδTreg",
        "ExhaustedγδTreg": "γδTreg",
        "γδT1": "γδT1",
        "Stem-likeγδT1": "γδT1",
        "ProgenitorExhaustedγδT1": "γδT1",
        "ExhaustedγδT1": "γδT1",
        "ProliferatingγδT": "MKI+γδT",
        "Stem-likeγδT": "γδT",
        "ProgenitorExhaustedγδT": "γδT",
        "ExhaustedγδT": "γδT",
        
        # Other cell types
        "ILC": "ILC",
        "ImmatureNK": "ImmatureNK",
        "MatureNK": "MatureNK",
        "NKT": "NKT",
        "NaiveB": "NaiveB",
        "DN3B": "DN3B",
        "ProliferatingB": "MKI+B",
        "TransitionalB": "TransitionalB",
        "otherB": "otherB",
        "Breg": "Breg",
        "GCB": "GCB",
        "Tbet+MemoryB": "Tbet+MemoryB",
        "MemoryB": "MemoryB",
        "CD11c+B": "CD11c+B",
        "DN2B": "DN2B",
        "CD11c+CD27+B": "CD11c+CD27+B",
        "AgeAsscB": "AgeAsscB",
        "Plasmablast": "Plasmablast",
        "IgG4+Plasmablast": "IgG4+Plasmablast",
        "Plasma": "Plasma",
        "IgG4+Plasma": "IgG4+Plasma",
        "M1Macrophage": "M1Macrophage",
        "MPO+Macrophage": "MPO+Macrophage",
        "M2Macrophage": "M2Macrophage",
        "MerTK+M1Macrophage": "MerTK+M1Macrophage",
        "MerTK+M2Macrophage": "MerTK+M2Macrophage",
        "GPNMB+M1Macrophage": "GPNMB+M1Macrophage",
        "GPNMB+M2Macrophage": "GPNMB+M2Macrophage",
        "cDC1": "cDC1",
        "cDC2": "cDC2",
        "pDC": "pDC",
        "Neutrophil": "Neutrophil",
        "Myofibroblast-SmoothMuscle": "Myofibroblast-SmoothMuscle",
        "CD90+Myofibroblast": "CD90+Myofibroblast",
        "MMT": "MMT",
        "ProximalTubular": "ProximalTubular",
        "DistalTubular": "DistalTubular",
        "StressedTubular": "StressedTubular",
        "Endothelial": "Endothelial",
        "LymphaticEndothelial": "LymphaticEndothelial",
        "Podocyte": "Podocyte",
        "PEC": "PEC",
        "RBC": "RBC",
        "TubuleBackground": "TubuleBackground",
        "unclassified": "unclassified"
    }
    
    # Add the CellCatName3 column based on the mapping
    df['CellCatName3'] = df['CellCatName'].map(mapping)
    
    # Handle any unmapped values (if any)
    if df['CellCatName3'].isna().any():
        print("Warning: Some CellCatName values could not be mapped:")
        unmapped = df[df['CellCatName3'].isna()]['CellCatName'].unique()
        print(unmapped)
        # Fill unmapped values with original values
        df['CellCatName3'] = df['CellCatName3'].fillna(df['CellCatName'])
    
    # Save the result to a new CSV file
    df.to_csv(output_csv, index=False)
    print(f"Saved output to {output_csv}")

# Example usage
if __name__ == "__main__":
    input_file = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/processed_combined_SAM_scores_filtered_alldataset5_10woMonointerstitial_memcatname3cellcatname2.csv"  # Replace with your input file path
    output_file = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/processed_combined_SAM_scores_filtered_alldataset5_10woMonointerstitial_memcatname3cellcatname23.csv"  # Replace with your desired output file path
    add_cellcatname3_column(input_file, output_file)
