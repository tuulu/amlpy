# Call libraries
import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Define the folder containing the sample_table.txt files
folder_path = "data/E-GEOD-14924" # Tuulu: You have to change this path to the path of the folder containing the sample_table.txt files
id_path = "data/E-GEOD-14924/E-GEOD-14924.sdrf.txt" # And this too, it tells you what sample belongs to which patient

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

print(df)

# Removing AFFX control probes
df = df[~df['ID_REF'].str.contains('AFFX')]

# Removing genes wuth ABSENT calls
# df = df[df['ABS_CALL'] != 'A']
# As the p values refer to the absent, marginal genes while comparing with the internal control this code becomnes largely uncessary 

# Remove genes with more than 0.05 P-Value 
df = df[df['DETECTION P-VALUE'] < 0.05]  
# The p values here correspond to A(absent), M(marginal) and P(present) genes 
# Low p values correspond to present genes 
# Whereas high p values correspond to present genes 

# Removing ABS_CALL, DETECTION P-VALUE and Filename columns
df = df.drop('ABS_CALL', axis=1)
df = df.drop('DETECTION P-VALUE', axis=1)
df = df.drop('Filename', axis=1)

# Checking structure of df
print(df.head())
print(df.shape)

# For DGE we split the data to CD4 and CD8 subtypes
state_1='CD4'
state_2='CD8'
CD4_df = df[df['Patient'].str.contains(state_1, case=False, na=False)]
CD8_df= df[df['Patient'].str.contains(state_2, case=False, na=False)]
# Assessing the head and the shapes of the dataframes 
#CD4
print(CD4_df.head())
print(CD4_df.shape)
# CD8
print(CD8_df.head())
print(CD8_df.shape) 
# building a reshaped matrix with genes as rows and conditions as colums

# Pivot the DataFrame so that genes are rows and samples are columns 
CD4_pivot = CD4_df.pivot(index='ID_REF', columns='Patient', values='VALUE')
print(CD4_pivot.head())

CD8_pivot = CD8_df.pivot(index='ID_REF', columns='Patient', values='VALUE')
print(CD8_pivot.head())

# Determining the total number of NaN  values
na_counts_CD4 = CD4_pivot.isna().sum().sum()
na_counts_CD8 = CD8_pivot.isna().sum().sum()
print(na_counts_CD4)
print(na_counts_CD8)

# I suggest to throw genes which contain more than 2 NaN values per column 
# Genes having 2 or 1  NaN value can be modified via imputation 
CD4_clean = CD4_pivot.dropna(thresh=CD4_pivot.shape[1]-2)
CD8_clean = CD8_pivot.dropna(thresh=CD8_pivot.shape[1]-2)

# Now we can assess the number of NaN values again
na_clean_CD4 = CD4_clean.isna().sum().sum()
na_clean_CD8 = CD8_clean.isna().sum().sum()
print(na_clean_CD4)
print(na_clean_CD8)
# The total NA number has dropped significantly 

# Checking if the data was normalized before imputation
# Print the median across each column 
print(CD4_clean.median())
print(CD8_clean.median()) 
# Print median of all columns 
print("Overall median value CD4:", CD4_clean.median().median()) 
print("Overall median value CD8:", CD8_clean.median().median()) 
# As a result we see that the difference between means is not great, the data was already normalized

# IMPUTATION
### Now used the k value of 3, need to solve the problem of adjesnment later 

# imputing the necessary libraries 
from sklearn.impute import KNNImputer

# Initializing the imputer
imputer=KNNImputer(n_neighbors=3)

# Imputing the values to the dataframes
CD4_imput=pd.DataFrame(imputer.fit_transform(CD4_clean), columns=CD4_clean.columns, index=CD4_clean.index)
CD8_imput=pd.DataFrame(imputer.fit_transform(CD8_clean), columns=CD8_clean.columns, index=CD8_clean.index) 

# Printing the dataframes and analyzing the values 
print(CD4_imput.head())
print(CD8_imput.head()) 

# In order to prove that the imputation did not change the data 
print(CD4_clean.corr().head())
print(CD4_imput.corr().head())

print(CD8_clean.corr().head()) 
print(CD8_imput.corr().head())  

# Z-SCORE NORMALIZATION = making the mean value 0 and standart variance 1 
#from sklearn.preprocessing import StandardScaler 
# Z-score Normalization (standardization)
#scaler = StandardScaler()
# For CD4 Dataframe 
#z_score_data_CD4= pd.DataFrame(scaler.fit_transform(annotated_CD4.T), index=annotated_CD4.columns, columns=annotated_CD4.index).T  
#print(z_score_data_CD4.head()) 

# For CD8 Dataframe
#z_score_data_CD8= pd.DataFrame(scaler.fit_transform(annotated_CD8.T), index=annotated_CD8.columns, columns=annotated_CD8.index).T  
#print(z_score_data_CD8.head()) 

# Storing the values as csv files
CD4_imput.to_csv("output/CD4 Lymphocytes")
CD8_imput.to_csv("output/CD8 Lymphocytes")