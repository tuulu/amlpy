# This script processes the RNA microarray data from both datasets and prepares them for further analysis (DGE, GSEA, prediction)

#------------------------------------------------------------------------------------------------
# Call libraries
import os
import re
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer

#------------------------------------------------------------------------------------------------

# Only overlapping genes are relevant for further analysis
def define_overlap(df1, df2):
    """Selects the overlapping genes between the AML and healthy datasets"""
   
    # Turning all Gene Symbols to strings as a precaution
    df1["Gene Symbol"] = df1["Gene Symbol"].astype(str)
    df2["Gene Symbol"] = df2["Gene Symbol"].astype(str)

    # Finding the intersection of the two sets of gene symbols
    overlap = np.intersect1d(df1["Gene Symbol"], df2["Gene Symbol"])
    df1 = df1[df1["Gene Symbol"].isin(overlap)]
    df2 = df2[df2["Gene Symbol"].isin(overlap)]
    print(f"df1 shape: {df1.shape}")
    print(f"df2 shape: {df2.shape}")

    # Merging duplicate genes by averaging the values
    df1_merged = df1.groupby('Gene Symbol', as_index=False).mean()
    df2_merged = df2.groupby('Gene Symbol', as_index=False).mean()
    print("Merged data shape:")
    print(f"df1 merged data shape: {df1_merged.shape}")
    print(f"df2 merged data shape: {df2_merged.shape}")

    #Reorder the rows in df2_merged to match the order of df1_merged
    df2_merged = df2_merged.reindex(df1_merged.index)

    return df1_merged, df2_merged

#------------------------------------------------------------------------------------------------

def load_blood_samples():
    """Loads the blood samples from the GTEx dataset using the .soft file"""
    # Load the .soft file
    with open("data/GSE45878_family.soft", "r") as f:
        data = f.read()

    # Split into individual samples
    samples = data.split("^SAMPLE")[1:]

    # Creates a dictionary to store only blood samples
    blood_samples = {}

    for sample in samples:
        # Get sample accession
        id_match = re.search(r"Sample_geo_accession = (GSM\d+)", sample)
        sample_id = id_match.group(1) if id_match else None

        # Search for tissue source in "source_name_ch1" or any characteristics
        if "blood -" in sample.lower():
            blood_samples[sample_id] = sample.strip()
    
    # Output count or list of IDs
    print(f"Number of blood samples found: {len(blood_samples)}")
    
    sample_list = list(blood_samples.keys())

    return sample_list

#------------------------------------------------------------------------------------------------

# This function does some AFFX data-specific cleaning
def affx_clean(df):
    """Cleans irrelevant probes from the dataframe"""
    # Removing AFFX control probes
    df = df[~df["ID_REF"].str.contains('AFFX')]

    # Removing genes with ABSENT calls
    df = df[df["ABS_CALL"] != 'A']

    # Remove genes with more than 0.05 P-Value 
    df = df[df["DETECTION P-VALUE"] < 0.05]

    # Removing ABS_CALL, DETECTION P-VALUE and Filename columns
    df = df.drop("ABS_CALL", axis=1)
    df = df.drop("DETECTION P-VALUE", axis=1)
    df = df.drop("Filename", axis=1)
    
    return df

#------------------------------------------------------------------------------------------------

def normalize_if_needed(df, gene_column = 0, threshold = 50, z_score = False):
    """Checks if expression values in a dataframe are log2-transformed.
    If not, applies log2(x + 1e-6) transformation and Z-score normalization."""
    # Drop the gene column to prevent errors
    expr_df = df.drop(df.columns[gene_column], axis=1)
    
    # Check if the 95th percentile exceeds threshold
    high_value = np.percentile(expr_df.values.astype(float), 95)
    
    if high_value > threshold:
        print("Data seems to be not log2-transformed. Applying log2(x + 1e-6).")
        expr_df = np.log2(expr_df.astype(float) + 1e-6)
    else:
        print("Data seems to be already log2-transformed. Nothing to do.")

    # Z-score normalization (if True)
    if z_score is True:
        expr_df = (expr_df - expr_df.mean(axis=1).values.reshape(-1, 1)) / expr_df.std(axis=1).values.reshape(-1, 1)
    
    # Reinsert gene names
    df[df.columns[gene_column]] = df[df.columns[gene_column]]
    df.update(expr_df)
    
    return df

#------------------------------------------------------------------------------------------------

def impute_data(df, n_neighbors = 3):
    """Perform KNN imputation on the data"""
    imputer = KNNImputer(n_neighbors=n_neighbors)

    # Skip if the data is already imputed / complete
    if df.isnull().sum().sum() == 0:
        print("Data is already complete. Skipping imputation.")
        return df
    
    df = pd.DataFrame(
        imputer.fit_transform(df), 
        columns = df.columns, 
        index = df.index)
    
    return df

def verify_imputation(df, df_imputed):
    """Verify imputation by comparing correlations"""
    print("Original Correlations:")
    print(df.iloc[:,1:].corr().head())
    print("\nImputed Correlations:")
    print(df_imputed.iloc[:,1:].corr().head())
    
#------------------------------------------------------------------------------------------------

def histo_data(df1, df2, output_dir, hour_timestamp):
    """Creates basic histogram plots both datasets."""
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Histogram of AML expression values
    sns.histplot(data = df1.iloc[:, 1:].values.flatten(), bins = 50, ax = ax1)
    ax1.set_title('AML Dataset Expression')
    ax1.set_xlabel('Relative Expression Value')
    ax1.set_ylabel('Count')
    
    # Plot 2: Histogram of Healthy expression values
    sns.histplot(data = df2.iloc[:, 1:].values.flatten(), bins = 50, ax = ax2)
    ax2.set_title('Healthy Dataset Expression')
    ax2.set_xlabel('Relative Expression Value')
    ax2.set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figures/histo_plot_{hour_timestamp}.png", bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

#------------------------------------------------------------------------------------------------
