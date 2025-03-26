# Importing the required libraries
import pandas as pd
import os
import numpy as np 
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats 


# Importing the imputed csv file 
counts_CD4=pd.read_csv("output/CD4 Lymphocytes", index_col=0)
print(counts_CD4.head())

#Transposing the data frame
counts_CD4_n=counts_CD4.T
counts_CD4_n = counts_CD4_n.astype(int)
print(counts_CD4_n) 

# Creating metadata
metadata=pd.DataFrame(zip(counts_CD4_n.index, ['AML','AML','AML','AML','AML','AML','AML','AML','AML','AML','healthy','healthy','healthy','healthy','healthy','healthy','healthy','healthy','healthy','healthy']), columns= ['Sample','Condition'])
metadata=metadata.set_index('Sample')
print(metadata) 

# Initializing a dds object
dds = DeseqDataSet(counts=counts_CD4_n, metadata=metadata, design_factors='Condition') 
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
sign_CD4 = counts_CD4.merge(signs, left_index=True, right_index=True)
sign_CD4.drop(columns=['baseMean', 'log2FoldChange', 'lfcSE', 'stat', 'pvalue', 'padj'], inplace=True)
print(sign_CD4.head())
sign_CD4.info

# Upon determining significantly expressed genes we assigned ID_REF to Gene Symbol
annotation = pd.read_csv("data/HG-U133_Plus_2.na36.annot.csv", sep=",", skiprows=25)
print(annotation.head())
print(annotation.columns) # This dataframe includes annotation data 

# Merging dataframes 
merged_CD4 = sign_CD4.merge(annotation[['Probe Set ID', 'Gene Symbol']], left_index=True, right_on='Probe Set ID', how='left')
merged_CD4.set_index('Gene Symbol', inplace=True) # Now, set the 'Gene Symbol' as the index
annotated_CD4=merged_CD4.drop('Probe Set ID', axis=1)
print(annotated_CD4.head())

# Storing differentially expressed genes as csv file 
annotated_CD4.to_csv("output/DiffGenExpres CD4")

# Conducting principal component analysis (PCA)=dimension reduction on the data 
import scanpy as sc 
sc.tl.pca(dds, n_comps=5) 
# PCA plot 
sc.pl.pca(dds, color='Condition', size=200) 

# GSEA
#Gen Set Enrichment Analysis: determining the biological processes the differencially expressed genes are engaged in 
import gseapy as gp
from gseapy import gseaplot

#Conducting GSEA analysis on DEGs 
signs # this dataframe depicts significant DEGs with adjuscent statistics

#Filtering out unnecessary columns
ranking = signs['stat'].dropna().sort_values(ascending=False)
print(ranking.head()) 
probe_ids=list(ranking.index) # Here we store the probe Gen_IDs as a list 

