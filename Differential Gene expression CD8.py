# Importing the required libraries
import pandas as pd
import os
import numpy as np 
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats 

# Setting the working directory 

# Importing the imputed csv file 
counts_CD8=pd.read_csv("output/CD8 Lymphocytes", index_col=0)
print(counts_CD8.head())
counts_CD8.info
#Transposing the data frame
counts_CD8_n=counts_CD8.T
counts_CD8_n = counts_CD8_n.astype(int)
print(counts_CD8_n) 

# Creating metadata
metadata=pd.DataFrame(zip(counts_CD8_n.index, ['AML','AML','AML','AML','AML','AML','AML','AML','AML','AML','healthy','healthy','healthy','healthy','healthy','healthy','healthy','healthy','healthy','healthy','healthy']), columns= ['Sample','Condition'])
metadata=metadata.set_index('Sample')
print(metadata) 

# Initializing a dds object
dds = DeseqDataSet(counts=counts_CD8_n, metadata=metadata, design_factors='Condition') 
# Observations
dds.obs
# Gene matrix
dds.X 

# Running the deseq2() 
dds.deseq2() 
dds # now we can conduct the differential gene expression 

# Differential Gene Expression
stat_res=DeseqStats(dds, contrast=('Condition','AML','healthy'))
stat_res.summary()  # Conducting the statistical tests to determine differentially expressed data 

# Getting the Dataframe with the results 
res= stat_res.results_df
res # As output we get the statistics on gene expression 

# Finding the significantly expressed based on p values and log2FC values 
# log 2(Expression in condition A/Expression in condition B) is the metrics of Wald test 

signs= res[(res.padj < 0.05) & abs(res.log2FoldChange > 0.5)]
print(signs) 

# Counts corresponding to the differencially sexpressed genes were determined 
sign_CD8 = counts_CD8.merge(signs, left_index=True, right_index=True)
sign_CD8.drop(columns=['baseMean', 'log2FoldChange', 'lfcSE', 'stat', 'pvalue', 'padj'], inplace=True)
print(sign_CD8.head())
sign_CD8.info

# Upon determining significantly expressed genes we assigned ID_REF to Gene Symbol
annotation = pd.read_csv("data/HG-U133_Plus_2.na36.annot.csv", sep=",", skiprows=25)
print(annotation.head())
print(annotation.columns) # This dataframe includes annotation data 

# Merging dataframes 
merged_CD8 = sign_CD8.merge(annotation[['Probe Set ID', 'Gene Symbol']], left_index=True, right_on='Probe Set ID', how='left')
merged_CD8.set_index('Gene Symbol', inplace=True) # Now, set the 'Gene Symbol' as the index
annotated_CD8=merged_CD8.drop('Probe Set ID', axis=1)
print(annotated_CD8.head())

# Storing differentially expressed genes as csv file 
annotated_CD8.to_csv("output/DiffGenExpres CD8")

# Conducting principal component analysis (PCA)=dimension reduction on the data 
import scanpy as sc 
sc.tl.pca(dds, n_comps=5) 
# PCA plot 
sc.pl.pca(dds, color='Condition', size=200) 

# GSEA
#Gen Set Enrichment Analysis: determining the biological processes the differencially expressed genes are engaged in 
import gseapy as gp
from gseapy import gseaplot
from gseapy import barplot, dotplot 
# Running the GSEAPY  using 'GO_Biological_Process_2018' dataset
gsea_results = gp.gsea(
    data=annotated_CD8,
    cls=['AML','AML','AML','AML','AML','AML','AML','AML','AML','AML','healthy','healthy','healthy','healthy','healthy','healthy','healthy','healthy','healthy','healthy','healthy'],            
    gene_sets='GO_Biological_Process_2018',             
    permutation_type='gene_set',
    min_size=15, 
    max_size=200,                
    method='signal_to_noise'     
)
# gsea object with relevant statistics 
gsea_results.res2d
print(gsea_results.res2d.head())  
# Creating a plot, NES= normalized enrichment score 
ax = dotplot(gsea_results.res2d,column="NES",title='KEGG_2021_Human',cmap='viridis', size=5,figsize=(4,5), cutoff=1) 
ax.figure.savefig('CD8_dotplot.png', bbox_inches='tight', dpi=300) 


