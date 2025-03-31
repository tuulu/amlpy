# This script conducts differential gene expression (DGE) analysis on both datasets
# This prepares them for GSEA analysis and prediction

#------------------------------------------------------------------------------------------------
# Call libraries
import os
import numpy as np 
import pandas as pd
import scanpy as sc
import gseapy as gp
import matplotlib.pyplot as plt
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats 
from gseapy import gseaplot
from gseapy import barplot, dotplot 

#------------------------------------------------------------------------------------------------
def create_metadata_df(df, condition):
    """Creates a metadata dataframe from column (Sample) names."""
    # Get sample names (all columns except first)
    sample_names = df.columns[1:]
    
    # Create metadata DataFrame
    
    if condition == 'AML':
        metadata = pd.DataFrame({
        'Sample': sample_names,
        'Condition': ['AML'] * len(sample_names)
    })
    else:
        metadata = pd.DataFrame({
        'Sample': sample_names,
        'Condition': ['Healthy'] * len(sample_names)
    })
    
    # Set Sample as index
    metadata = metadata.set_index('Sample')
    
    return metadata

#------------------------------------------------------------------------------------------------
def combine_data(df1, df2, round = False):
    """Combines the AML and healthy dataframes and returns the combined counts"""
    combined_counts = pd.concat([df1, df2], axis=1)
    # Keep only the first Gene Symbol column and drop the second Gene Symbol column
    combined_counts = combined_counts.loc[:, ~combined_counts.columns.duplicated(keep='first')]
    # Turn the first column into the index (to prevent the gene symbols from being included in the counts)
    combined_counts.set_index(combined_counts.columns[0], inplace=True)
    # Round up counts to the nearest integer (for pydeseq2)
    if round is True:
        combined_counts = combined_counts.round(0)
    
    return combined_counts

#------------------------------------------------------------------------------------------------
def prepare_dge(counts_df, metadata_df):
    """Prepares data for DGE analysis."""
    # Transpose counts and convert to int
    counts_n = counts_df.T
    
    # Initialize DeseqDataSet
    dds = DeseqDataSet(counts = counts_n, metadata = metadata_df, design_factors = 'Condition')
    # Observations
    dds.obs
    # Gene matrix
    dds.X 
    # Run DESeq2
    dds.deseq2()
    
    return dds

#------------------------------------------------------------------------------------------------
def run_dge(dds, contrast = ('Condition','AML','Healthy')):
    """ Runs differential gene expression (DGE) analysis."""
    # Run statistical analysis
    stat_res = DeseqStats(dds, contrast = contrast)
    stat_res.summary()
    
    # Get results
    res = stat_res.results_df
    
    # Filter for significant genes
    significant = res[(res.padj < 0.05) & (abs(res.log2FoldChange) > 0.5)]
    
    return significant

#------------------------------------------------------------------------------------------------
def pca_plot(dds, output_dir, hour_timestamp):
    """Creates a PCA plot of the DGE data."""
    sc.tl.pca(dds, n_comps=5) 
    sc.pl.pca(dds, 
          color='Condition', 
          size=300,
          show=False)  # Don't show immediately

    # Adjust the plot to be more compact
    plt.tight_layout()
    #Save the plot
    plt.savefig(f"{output_dir}/figures/PCA_plot_{hour_timestamp}.png", bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

#------------------------------------------------------------------------------------------------
def run_gsea(combined_data_gsea, conditions, output_dir, hour_timestamp):
    """Runs gene set enrichment analysis (GSEA) using the GSEAPY package."""
    # Running GSEAPY using 'GO_Biological_Process_2018' dataset
    print("\n Running GSEAPY...")
    gsea_results = gp.gsea(
        data = combined_data_gsea,
        cls = conditions,
        gene_sets = "GO_Biological_Process_2018",             
        permutation_type = "gene_set",
        min_size = 5, 
        max_size = 1000,                
        method = "signal_to_noise")

    # gsea object with relevant statistics 
    gsea_results.res2d
    print(gsea_results.res2d.head())  
    # Creating a plot, NES= normalized enrichment score 
    ax = dotplot(gsea_results.res2d,column = "NES",title = 'KEGG_2021_Human',cmap = 'viridis', size = 5,figsize = (4,5), cutoff = 1) 
    ax.figure.savefig(f"{output_dir}/figures/GSEA_dotplot_{hour_timestamp}.png", bbox_inches = 'tight', dpi = 300)

    return gsea_results