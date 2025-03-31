# This file contains helper functions for the main scripts

#------------------------------------------------------------------------------------------------
# Call libraries
import pandas as pd
from Ensembl_converter import EnsemblConverter

#------------------------------------------------------------------------------------------------
# This function converts the Ensembl gene IDs to gene symbols
# GSE45878 (GTEx) dataset utilizes a custom CDF file, which is not available online anymore
# Therefore, we use the the package Ensembl_converter to map the ESNG IDs to the gene symbols
# Alternatively, we could download the relevant data from the Ensembl Biomart and manually map them (https://www.ensembl.org/biomart/martview/)
# Doing this would be more time-efficient, due to the networking constraints of the Ensembl_converter package
def convert_ensg_to_symbol(df, file_path = "data/GSE45878_series_matrix.txt"):
    """Converts the Ensembl gene IDs to gene symbols"""
    # Skip if the ID_REF column contains no _at i.e., it is already annotated
    if not df.iloc[:, 0].str.contains("_at").any() or df.columns[0] == "Gene Symbol":
        return df
    else:
        df = pd.read_csv(file_path, header = 0, skiprows = 61, delimiter = '\t') # Skip the header and the rows with the metadata

    # Initialize the converter
    converter = EnsemblConverter(use_progress_bar = True)

    #Define the column containing the ENSG IDs
    ensembl_ids = df["ID_REF"]

    # Remove _at from the ENSG IDs
    ensembl_ids = ensembl_ids.str.replace("_at", "")

    #Convert the ENSG IDs to gene symbols
    result = converter.convert_ids(ensembl_ids)

    # Add the gene symbols to the dataframe
    df["Gene Symbol"] = result["Symbol"]

    # Move the gene symbols to the first column
    df = df[["Gene Symbol"] + [col for col in df.columns if col != "Gene Symbol"]]

    # Drop the ID_REF column
    df = df.drop(columns=["ID_REF"])

    # Return the dataframe with the gene symbols
    return df

#------------------------------------------------------------------------------------------------
# This function annotates the dataframe with the gene symbols
# GSE68833 (TCGA) dataset has an associated annotation file, which can be use
# However, the annotation file itself must be processed first
def annotate_df(df, file_path = "data/GSE68833_series_matrix.txt", annotation_file = "data/GSE68833_annotation.txt"):
    """Annotates the gene symbols"""
    # Skip if the ID_REF column contains no _at i.e., it is already annotated
    if not df.iloc[:, 0].str.contains("_at").any() or df.columns[0] == "Gene Symbol":
        return df
    else:
        df = pd.read_csv(file_path, header = 0, skiprows = 74, delimiter = '\t') # Skip the header and the rows with the metadata
    
    # Read the annotation file
    annotation_df = pd.read_csv(annotation_file, sep = "\t", comment = '#')

    # If the GeneSymbols column has multiple values separated by "|" choose the first one
    annotation_df["Gene Symbol"] = annotation_df["GeneSymbols"].str.split("|").str[0]
    # Drop the GeneSymbols column
    annotation_df = annotation_df.drop(columns=["GeneSymbols"])
    # Keep only ProbeName and Gene Symbol
    annotation_df = annotation_df[["ProbeName", "Gene Symbol"]]
    # Rename the ProbeName column to ID_REF
    annotation_df = annotation_df.rename(columns = {"ProbeName": "ID_REF"})

    # Get the overlap of the two dataframes on the "ID_REF" column
    overlap = pd.merge(df, annotation_file, on = "ID_REF", how = "inner")

    # Drop ID_REF and move Gene Symbol to the first column
    overlap = overlap.drop(columns=["ID_REF"])
    overlap = overlap[["Gene Symbol"] + [col for col in overlap.columns if col != "Gene Symbol"]]

    # Return the dataframe with the gene symbols
    return overlap

#------------------------------------------------------------------------------------------------
