import pandas as pd

# Mapping of relationships between Sample, Dataset, and CompName
dataset_mapping = {
    "24052882340623_Area2": "AAV",
    "24052882340623_Area3": "AAV",
    "234852332928_Area1": "AAV",
    "234852332928_Area2": "AAV",
    "234852332928_Area3": "AAV",
    "23210152340499_Area1": "AAV",
    "23210152340499_Area2": "AAV",
    "23210152340499_Area3": "AAV",
    "223253423242_Area1": "AAV",
    "223253423242_Area2": "AAV",
    "223253423242_Area3": "AAV",
    "223253423242_Area4": "AAV",
    "222883222327572229647_Area1": "AAV",
    "222883222327572229647_Area2": "AAV",
    "222883222327572229647_Area3": "AAV",
"115405_Area1":	"IgG4",
"115405_Area2": "IgG4",
"115405_Area3": "IgG4",
"2129474_Area1": "IgG4",
"2129474_Area2": "IgG4",
"2129474_Area3": "IgG4",
"2122622117581_Area1": "IgG4",
"2122622117581_Area2": "IgG4",
"21170502133810_Area1": "IgG4",
"21170502133810_Area2": "IgG4",
"21170502133810_Area3": "IgG4",
"24052882340623_Area1": "IgG4",
"6306263478_Area1": "NK2",
"6306263478_Area2": "NK2",
"6306263478_Area3": "NK2",
"6306263478_Area4": "NK2",
"6306263478_Area5": "NK2",
"243386766395_Area1": "NK2",
"243386766395_Area2": "NK2",
"243386766395_Area3": "NK2",
"243386766395_Area4": "NK2",
"243386766395_Area5": "NK2",
"243386766395_Area6": "NK2"}

# Load the CSV data into a pandas DataFrame
df = pd.read_csv('/project/mclark/SCAMPI_datasets/NK2/classify_by_dilation-sameTH/with-MPI/combined_SAM-scores_with-MPI.csv', low_memory=False)

# Create a function to extract the Patient from the CompName
def get_dataset(compname):
    return dataset_mapping.get(compname, None)

# Add the Dataset column based on the CompName
df['Dataset'] = df['CompName'].apply(get_dataset)

# Reorder columns so that Dataset is left of Sample
cols = list(df.columns)
sample_index = cols.index('Sample')
cols.insert(sample_index - 1, cols.pop(cols.index('Dataset')))

df = df[cols]

# Save the updated DataFrame to a new CSV file
df.to_csv('/project/mclark/SCAMPI_datasets/NK2/classify_by_dilation-sameTH/with-MPI/combined_SAM-scores_with-MPI2.csv', index=False)

print("Dataset column added and CSV updated successfully!")

