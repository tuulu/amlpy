import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Define the folder containing the sample_table.txt files
folder_path = "C:/Users/tuulu/Desktop/Python_project/data/E-GEOD-14924" # Tuulu: You have to change this path to the path of the folder containing the sample_table.txt files
id_path = "C:/Users/tuulu/Desktop/Python_project/data/E-GEOD-14924/E-GEOD-14924.sdrf.txt" # And this too, it tells you what sample belongs to which patient

# Get a list of all sample_table.txt files in the folder
file_list = glob(os.path.join(folder_path, "*sample_table.txt"))

# Checking all files
print(f"Number of files: {len(file_list)}")
for i, file in enumerate(file_list):
    print(f"File {i}: Shape {pd.read_csv(file, sep='\t').shape}")

# Open the id_path file
id_df = pd.read_csv(id_path, sep="\t")

#Removing (space)1 from Source Name
print(id_df["Source Name"])
id_df["Source Name"] = id_df["Source Name"].str.replace(" 1", "")
print(id_df.head())

# Create the first df to start with
df = pd.read_csv(file_list[0], sep="\t")
filename = os.path.basename(file_list[0])
df["Sample Nr."] = filename.replace("_sample_table.txt", "")
df["Filename"] = filename
sample_no = df["Sample Nr."].iloc[0]
mask = id_df["Source Name"] == sample_no
df["Disease State"] = id_df.loc[mask, "Characteristics[disease state]"].values[0]
df["Cell Type"] = id_df.loc[mask, "Factor Value[cell type]"].values[0]

# Process the rest of the files and concatenate
for file in file_list[1:]:
    temp_df = pd.read_csv(file, sep="\t")
    filename = os.path.basename(file)
    temp_df["Sample Nr."] = filename.replace("_sample_table.txt", "")
    temp_df["Filename"] = filename
    sample_no = temp_df["Sample Nr."].iloc[0]
    mask = id_df["Source Name"] == sample_no
    temp_df["Disease State"] = id_df.loc[mask, "Characteristics[disease state]"].values[0]
    temp_df["Cell Type"] = id_df.loc[mask, "Factor Value[cell type]"].values[0]
    df = pd.concat([df, temp_df], ignore_index=True)

# Merging Disease State and Cell Type into one column (normal to healthy, acute myeloid leukemia to AML, CD4 T cell to CD4, CD8 T cell to CD8)
df["Disease State"] = df["Disease State"].str.replace("normal", "healthy")
df["Disease State"] = df["Disease State"].str.replace("acute myeloid leukemia", "AML")
df["Cell Type"] = df["Cell Type"].str.replace("CD4 T cell", "CD4")
df["Cell Type"] = df["Cell Type"].str.replace("CD8 T cell", "CD8")
df["Patient"] =  df["Sample Nr."] + "_" + df["Disease State"] + "_" + df["Cell Type"]
df = df.drop('Disease State', axis=1)
df = df.drop('Cell Type', axis=1)
df = df.drop('Sample Nr.', axis=1)

# Removing AFFX control probes
df = df[~df['ID_REF'].str.contains('AFFX')]

# Removing genes wuth ABSENT calls
df = df[df['ABS_CALL'] != 'A']

# Optional: Remove genes with more than 0.05 P-Value
df = df[df['DETECTION P-VALUE'] < 0.05]

# Removing ABS_CALL, DETECTION P-VALUE and Filename columns
df = df.drop('ABS_CALL', axis=1)
df = df.drop('DETECTION P-VALUE', axis=1)
df = df.drop('Filename', axis=1)

# Checking structure of df
print(df.head())
print(df.shape)

df.to_csv("cleaned_data.csv")

# Create a column for each Patient with their own values and ID_REF as rows (this is the matrix)
combined_df = df.pivot_table(index='ID_REF', columns='Patient', values='VALUE', aggfunc='first')


# Reset index to make ID_REF a column again
combined_df = combined_df.reset_index()


# Set 'ID_REF' as the index (genes as rows, samples as columns)
combined_df.set_index("ID_REF", inplace=True)

# Checking matrix
print(combined_df.head())
print(combined_df.columns)
print(combined_df.shape)

# Creating a new dataframe that stores the sample info (for plotting etc.)
sample_ids = combined_df.columns.tolist()
sample_info = pd.DataFrame({
    'Sample': patient_ids,
    'Patient_Group': [f"{id.split('_')[1]}_{id.split('_')[2]}" for id in sample_ids]
})

# Convert to numeric
combined_df = combined_df.apply(pd.to_numeric, errors="coerce")

# Count NA values in each column
na_counts = combined_df.isna().sum()
print(na_counts.mean())

# Removing rows with only NA values (WE CAN IMPUTE LATER, FOR NOW WE WANT TO KEEP AS MUCH DATA AS POSSIBLE)
combined_df = combined_df[combined_df.notna().any(axis=1)]


# Checking if the dataset was already normalized
print(combined_df.describe())
# Print median of all columns
print("Overall median value:", combined_df.median().median())

plt.figure(figsize=(8,5))
sns.histplot(df["VALUE"], bins=50, kde=True)
plt.xlabel("Expression Value")
plt.ylabel("Frequency")
plt.title("Distribution of Expression Values")
plt.show()

# Optional: Normalize the data

combined_df = np.log2(combined_df + 1)  # for DGE
combined_df_zscore = combined_df.apply(zscore, axis=1)  # for machine learning



# Save the normalized count matrix
combined_df.to_csv("normalized_count_matrix.csv")

# Print summary
print(combined_df.head())


