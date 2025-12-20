import pandas as pd

# Mapping of relationships between Sample, Patient, and CompName
patient_mapping = {
    "24052882340623_Area2": "2340623",
    "24052882340623_Area3": "2340623",
    "234852332928_Area1": "23485",
    "234852332928_Area2": "23485",
    "234852332928_Area3": "2332928",
    "23210152340499_Area1": "2321015",
    "23210152340499_Area2": "2321015",
    "23210152340499_Area3": "2340499",
    "223253423242_Area1": "2232534",
    "223253423242_Area2": "2232534",
    "223253423242_Area3": "23242",
    "223253423242_Area4": "23242",
    "222883222327572229647_Area1": "2228832",
    "222883222327572229647_Area2": "2232757",
    "222883222327572229647_Area3": "2229647",
"115405_Area1":	"115405",
"115405_Area2": "115405",
"115405_Area3": "115405",
"2129474_Area1": "2129474",
"2129474_Area2": "2129474",
"2129474_Area3": "2129474",
"2122622117581_Area1": "212262",
"2122622117581_Area2": "2117581",
"21170502133810_Area1": "2117050",
"21170502133810_Area2": "2133810",
"21170502133810_Area3": "2133810",
"24052882340623_Area1": "245288",
"6306263478_Area1": "63062",
"6306263478_Area2": "63062",
"6306263478_Area3": "63062",
"6306263478_Area4": "63478",
"6306263478_Area5": "63478",
"243386766395_Area1": "2433867",
"243386766395_Area2": "2433867",
"243386766395_Area3": "2433867",
"243386766395_Area4": "2433867",
"243386766395_Area5": "66395",
"243386766395_Area6": "66395"}

# Load the CSV data into a pandas DataFrame
df = pd.read_csv('/project/mclark/SCAMPI_datasets/NK2/classify_by_dilation-sameTH/with-MPI/combined_SAM-scores_with-MPI2.csv', low_memory=False)

# Create a function to extract the Patient from the CompName
def get_patient(compname):
    return patient_mapping.get(compname, None)

# Add the Patient column based on the CompName
df['Patient'] = df['CompName'].apply(get_patient)

# Reorder columns so that Patient is between Sample and CompName
cols = list(df.columns)
sample_index = cols.index('Sample')
compname_index = cols.index('CompName')
cols.insert(sample_index + 1, cols.pop(cols.index('Patient')))

df = df[cols]

# Save the updated DataFrame to a new CSV file
df.to_csv('/project/mclark/SCAMPI_datasets/NK2/classify_by_dilation-sameTH/with-MPI/combined_SAM-scores_with-MPI3.csv', index=False)

print("Patient column added and CSV updated successfully!")

