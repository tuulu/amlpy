#------------------------------------------------------------------------------------------------
# Welcome to amlpy!
# This is the main script for the project
# It will be used to run almost all of the subscripts and save the results
import intro
intro.intro()

#------------------------------------------------------------------------------------------------
# Install required packages (requirements.txt file
# pip install -r requirements.txt -> Run to, you know...run
# python ./src/data_import.py -> Run beforehand to download the data

# START OF PIPELINE
print("Starting amlpy...")

#------------------------------------------------------------------------------------------------
# Call libraries
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import DataLoader

# Import scripts
# import src.data_import -> run by itself as a script
import src.helper_functions
import src.data_process
import src.DGE
import src.random_forest
from src.MLP import AffyDataset, AMLClassifier, train_model, evaluate_model

print("\n Library import DONE")

#------------------------------------------------------------------------------------------------
# We will use the following timestamps for the output files
# Smaller timestamp
def hour_timestamp(custom = None):
  if custom:
    return datetime.now().strftime(custom)
  else:
    return datetime.now().strftime("%H_%M_%S")

# Larger timestamp
day_timestamp = datetime.now().strftime("%Y_%m_%d")

# Define the output directory (year, month, and day)
output_dir = f"output/{day_timestamp}"
os.makedirs(output_dir, exist_ok = True)
os.makedirs(f"{output_dir}/figures", exist_ok = True)

print(f"Started at {hour_timestamp("%H:%M:%S")}.")
time_start = datetime.now()

#------------------------------------------------------------------------------------------------
# We import the data from the GSE68833 and GSE45878 datasets
# Define the path to the data directory
folder_path = "./data"
file_list = glob(os.path.join(folder_path, "*series_matrix.txt"))

print("\n Data import DONE")

# Check all files
print(f"Number of files: {len(file_list)}")
for i, file in enumerate(file_list):
    print(f"File {i}: Shape {pd.read_csv(file, sep='\t').shape}")

# Reading the data
aml_data = pd.read_csv("data/GSE68833_series_matrix.txt") 
healthy_data = pd.read_csv("data/GSE45878_series_matrix.txt")

print("Data read DONE")

# Check the data
print(aml_data.head())
print(healthy_data.head())

#------------------------------------------------------------------------------------------------
# We annotate the datasets with gene symbols
# The annotation files must be processed first
# However, the datasets are already pre-annotated but this is how it goes:
print("\n Annotating the data...")
aml_data = src.helper_functions.annotate_df(aml_data)
healthy_data = src.helper_functions.convert_ensg_to_symbol(healthy_data)
print("Data annotation DONE")

# Afterwards, we take the overlapping genes from both datasets and discard the rest
print("\n Identifying overlapping genes...")
aml_merged, healthy_merged = src.data_process.define_overlap(aml_data, healthy_data)
aml = aml_merged
healthy = healthy_merged
print("Overlapping genes DONE")

# For the healthy dataset, we keep only the blood samples since AML is a blood cancer
print("\n Isolating blood samples from the healthy dataset...")
blood_samples = src.data_process.load_blood_samples()
healthy = healthy[['Gene Symbol'] + blood_samples]
print("Blood samples DONE")

# In case there are any missing values, we impute them using KNNImputer
print("\n Imputing missing values...")
aml_imputed = src.data_process.impute_data(aml)
healthy_imputed = src.data_process.impute_data(healthy)
print("Imputation DONE")

# And check if the imputation was successful (not needed right now)
# src.data_process.verify_imputation(aml, aml_imputed)
# src.data_process.verify_imputation(healthy, healthy_imputed)

# Overwrite the original dataframes with the imputed ones
aml = aml_imputed
healthy = healthy_imputed

# Check the data
print(f"\n Final AML data shape: {aml.shape}")
print(f"Final healthy data shape: {healthy.shape}")

#Save the resulting dataframes
print("\n Saving the cleaned dataframes...")
aml.to_csv(f"{output_dir}/AML_data_{hour_timestamp()}.csv", index=False)
healthy.to_csv(f"{output_dir}/healthy_data_{hour_timestamp()}.csv", index=False)

#------------------------------------------------------------------------------------------------
# Since the two datasets are from different platforms, we need to normalize them accordingly
# First, we check if the data is already normalized, then perform log2 transformation (and z-score) if needed
# We plot the data to check if it is normalized
print("\n Plotting the distribution of the data...")
src.data_process.histo_data(aml, healthy, output_dir = output_dir, hour_timestamp = hour_timestamp(), normalize = False)

# We then normalize the data if it needs to be normalized
print("\n Normalizing the data...")
aml = src.data_process.normalize_if_needed(aml)
print(aml.head())
healthy = src.data_process.normalize_if_needed(healthy)
print(healthy.head())
print("Normalization DONE")

# Do some summary stats (combined mean, median, std, min, max across all samples)
print("\n Generating some summary stats...")
print(aml.describe())
print(healthy.describe())

# Plot both datasets again to check if normalization was successful
print("\n Plotting the distribution of the data again post-normalization...")
src.data_process.histo_data(aml, healthy, output_dir = output_dir, hour_timestamp = hour_timestamp(), normalize = True)

#------------------------------------------------------------------------------------------------

# DISABLE / COMMENT OUT FOR FULL PIPELINE: Using a smaller subset of the data for faster processing
# print("\n Using a smaller subset of the data for faster processing (first 2000 genes)...")
# aml = aml.iloc[:2000,:]
# healthy = healthy.iloc[:2000,:]
# print(aml.shape)
# print(healthy.shape)

#------------------------------------------------------------------------------------------------
# Differential gene expression (DGE) analysis
print("\n Performing differential gene expression analysis...")
# Creating metadata
meta_aml = src.DGE.create_metadata_df(aml, 'AML')
meta_healthy = src.DGE.create_metadata_df(healthy, 'Healthy')

# Creating the combined data and metadata
combined_data = src.DGE.combine_data(aml, healthy, round = True)
combined_meta = pd.concat([meta_aml, meta_healthy], axis=0)
print(f"Combined data shape: {combined_data.shape}")
print(f"Combined metadata shape: {combined_meta.shape}")

# Defining the dds object and significant genes
dds_obj = src.DGE.prepare_dge(combined_data, combined_meta)
signs = src.DGE.run_dge(dds_obj)

# Turn signs.index into a df (for determining the overlapping genes between DGE and the expression data)
signs_df = pd.DataFrame(signs.index)
print(f"signs_df shape: {signs_df.shape}")

# Determine the overlapping genes between DGE and the expression data
aml_merged, signs_merged = src.data_process.define_overlap(aml, signs_df)
aml = aml_merged
signs = signs_merged
aml_merged, healthy_merged = src.data_process.define_overlap(aml, healthy)
aml = aml_merged
healthy = healthy_merged

# Assessing the validity of the data
print(f"AML DGE shape: {aml.shape}")
print(f"Healthy DGE shape: {healthy.shape}")
print(f"The total number of differentially expressed genes: {len(signs)}")

#Save the resulting DGE dataframes
print("\n Saving the cleaned dataframes...")
aml.to_csv(f"{output_dir}/AML_DGE_data_{hour_timestamp()}.csv", index=False)
healthy.to_csv(f"{output_dir}/healthy_DGE_data_{hour_timestamp()}.csv", index=False)

# Create a PCA plot of the two datasets
src.DGE.pca_plot(dds_obj, output_dir = output_dir, hour_timestamp = hour_timestamp())

print("\n DGE DONE")

#------------------------------------------------------------------------------------------------
# Gene set enrichment analysis (GSEA)
print("\n Performing gene set enrichment analysis...")

# Combine data for GSEA
combined_data_gsea = src.DGE.combine_data(aml, healthy)
print(combined_data_gsea.head())

# Create a list of conditions for GSEA using the metadata
conditions = combined_meta["Condition"].tolist()

# Run GSEA and save the results (plot is also saved)
gsea_res = src.DGE.run_gsea(combined_data_gsea, conditions, output_dir = output_dir, hour_timestamp = hour_timestamp())

print("\n GSEA DONE")

#------------------------------------------------------------------------------------------------
# AI / Machine learning
# Normalize the data again (this time with Z-score normalization for machine learning)
# Remove just one column and save the removed column in a separate variable (for later use)
aml = src.data_process.normalize_if_needed(aml, z_score = True)
healthy = src.data_process.normalize_if_needed(healthy, z_score = True)

# Combine the data once again (no rounding to integer this time)
combined_data_ml = src.DGE.combine_data(aml, healthy, round = False)
conditions_ml = combined_meta["Condition"].tolist()

#------------------------------------------------------------------------------------------------
# Random Forest prediction
print("\n Performing Random Forest prediction...")
# Mapping the conditions to 1 (AML) and 0 (Healthy)
label_map = {"AML": 1, "Healthy": 0}

# Removing the data from tuple and setting the data and target
X_tp, target_tp = src.random_forest.prepare_rf(combined_data_ml, conditions_ml, label_map)
X = X_tp # This is the data
target = target_tp # This is the target / condition data
print("X")
print(X.head())
print(target.head())

# Training the Random Forest model
rf, X_train_pca, X_test_pca, y_train_resampled, y_test = src.random_forest.train_rf(X, target, test_size = 0.2, random_state = 42)

# Evaluating accuracy of the model
rf_train_accuracy, rf_test_accuracy = src.random_forest.evaluate_rf(rf, X_train_pca, X_test_pca, y_train_resampled, y_test, output_dir = output_dir, hour_timestamp = hour_timestamp(), optimize = False)

# Training the Random Forest model with optimization (can take a looong time)
rf, X_train_pca, X_test_pca, y_train_resampled, y_test = src.random_forest.train_rf(X, target, test_size = 0.2, random_state = 42, optimize = True)

# Evaluating accuracy of the model
rf_train_accuracy, rf_test_accuracy = src.random_forest.evaluate_rf(rf, X_train_pca, X_test_pca, y_train_resampled, y_test, output_dir = output_dir, hour_timestamp = hour_timestamp(), optimize = True)

print("\n Random Forest prediction DONE")

#------------------------------------------------------------------------------------------------
#Multilayer Perceptron (MLP) prediction
print("\n Performing MLP prediction...")

# Pop the two last columns from each df (using them for inference later)
# The index is set to the original index to keep the sample names
aml_removed_1 = pd.DataFrame(aml.pop(aml.columns[-1]), index=aml.index)
aml_removed_2 = pd.DataFrame(aml.pop(aml.columns[-1]), index=aml.index)
healthy_removed_1 = pd.DataFrame(healthy.pop(healthy.columns[-1]), index=healthy.index)
healthy_removed_2 = pd.DataFrame(healthy.pop(healthy.columns[-1]), index=healthy.index)

# They're transposed to be used as tensors later
aml_removed_1 = aml_removed_1.T
aml_removed_2 = aml_removed_2.T
healthy_removed_1 = healthy_removed_1.T
healthy_removed_2 = healthy_removed_2.T

print(f"aml_removed_1: {aml_removed_1.head()}")

# Load your data
aml_df = aml.set_index('Gene Symbol')
healthy_df = healthy.set_index('Gene Symbol')

# Transpose so each row is a sample
aml_df = aml_df.T
healthy_df = healthy_df.T

# Assign labels
aml_labels = [1] * len(aml_df)
healthy_labels = [0] * len(healthy_df)

# Combine data and labels
full_df = pd.concat([aml_df, healthy_df], axis=0)
full_labels = aml_labels + healthy_labels

# 80:20 data split
X_train, X_val, y_train, y_val = train_test_split(
    full_df, full_labels, test_size = 0.2, stratify = full_labels, random_state = 42
)

# Create datasets and loaders
train_dataset = AffyDataset(X_train, y_train)
val_dataset = AffyDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True)
val_loader = DataLoader(val_dataset, batch_size = 32)

# Initialize model
input_dim = full_df.shape[1]    # number of genes
model = AMLClassifier(input_dim)

# Training
trained_model = train_model(model, train_loader, val_loader, epochs=50)

# Evaluating the model
mlp_test_accuracy = evaluate_model(trained_model, val_loader, output_dir = output_dir, hour_timestamp = hour_timestamp())

# Saving the model (not really needed here, but just in case)
torch.save(trained_model, f"{output_dir}/MLP_model_{hour_timestamp()}.pth")

# Inference (predicting new samples)
trained_model.eval()

# Convert dfs to tensors and run inference (no_grad() is used to disable gradient calculation and save time)
with torch.no_grad():
    # Convert dfs to tensors
    aml_removed_1_tensor = torch.tensor(aml_removed_1.values, dtype=torch.float32)
    aml_removed_2_tensor = torch.tensor(aml_removed_2.values, dtype=torch.float32)
    healthy_removed_1_tensor = torch.tensor(healthy_removed_1.values, dtype=torch.float32)
    healthy_removed_2_tensor = torch.tensor(healthy_removed_2.values, dtype=torch.float32)

    # Run inference
    aml_removed_1_pred = trained_model(aml_removed_1_tensor)
    aml_removed_2_pred = trained_model(aml_removed_2_tensor)
    healthy_removed_1_pred = trained_model(healthy_removed_1_tensor)
    healthy_removed_2_pred = trained_model(healthy_removed_2_tensor)

# Print the predictions
print("\nPredictions for removed samples:")
print(f"AML sample 1 prediction: {aml_removed_1_pred.item():.4f}")
print(f"AML sample 2 prediction: {aml_removed_2_pred.item():.4f}")
print(f"Healthy sample 1 prediction: {healthy_removed_1_pred.item():.4f}")
print(f"Healthy sample 2 prediction: {healthy_removed_2_pred.item():.4f}")

#Save the predictions
predictions = pd.DataFrame({
    "AML sample 1": [aml_removed_1_pred.item()],
    "AML sample 2": [aml_removed_2_pred.item()],
    "Healthy sample 1": [healthy_removed_1_pred.item()],
    "Healthy sample 2": [healthy_removed_2_pred.item()]
}, index=['Probability of AML%'])

predictions.to_csv(f"{output_dir}/inference_results_{hour_timestamp()}.csv", index=True)

print("\n MLP prediction DONE")
#------------------------------------------------------------------------------------------------
# Comparing the results of the MLP and Random Forest models
print("\n Comparing the results of the MLP and Random Forest models...")

# Simple comparison of the test accuracies
print(f"Random Forest test accuracy: {rf_test_accuracy:.2f}")
print(f"MLP test accuracy: {mlp_test_accuracy:.2f}")

print("\n Simply put...")
if mlp_test_accuracy > rf_test_accuracy:
    print("MLP performs better on the test set.")
elif mlp_test_accuracy < rf_test_accuracy:
    print("Random Forest performs better on the test set.")
else:
    print("Both models have the same accuracy on the test set.")

print("\n Comparison DONE")

#------------------------------------------------------------------------------------------------
# END OF PIPELINE
print("\n amlpy DONE")
print(f"Finished at {hour_timestamp("%H:%M:%S")}.")
time_end = datetime.now()
time_diff = time_end - time_start
print(f"Total runtime (approx.): {int(round(time_diff.total_seconds()/60, 0))} minute(s)")
print(f"All output saved in directory: {output_dir}.")


#------------------------------------------------------------------------------------------------
# Cheers! Thanks for taking the time to run the pipeline!