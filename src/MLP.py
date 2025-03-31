# This file contains the code for the multi-layer perceptron (MLP) model
# We will use it for classification of the AML and healthy samples

#------------------------------------------------------------------------------------------------
# Call libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, classification_report,
    roc_curve, precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay
)

#------------------------------------------------------------------------------------------------
# Create the dataset class
class AffyDataset(Dataset):
    def __init__(self, expression_df, labels_list):
        self.X = torch.tensor(expression_df.values, dtype = torch.float32)
        self.y = torch.tensor(labels_list, dtype = torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Defining the neural network
class AMLClassifier(nn.Module):
    def __init__(self, input_dim):
        super(AMLClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        return torch.sigmoid(self.fc3(x))

# Training function
def train_model(model, train_loader, val_loader, epochs = 50, lr = 1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    return model

# Prediction and evaluation
def evaluate_model(model, val_loader, output_dir, hour_timestamp):
    model.eval()
    y_true, y_pred_prob = [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch).squeeze()
            y_true.extend(y_batch.numpy())
            y_pred_prob.extend(outputs.numpy())
    
    y_pred = [1 if p >= 0.5 else 0 for p in y_pred_prob]

    print("\nEvaluation Metrics")
    print(f"AUC-ROC: {roc_auc_score(y_true, y_pred_prob):.4f}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names = ["Healthy", "AML"]))

    # Plot the metrics
    plot_metrics(y_true, y_pred_prob, y_pred, output_dir, hour_timestamp)

    return accuracy_score(y_true, y_pred)

#------------------------------------------------------------------------------------------------

# Plotting utilities
def plot_metrics(y_true, y_pred_prob, y_pred, output_dir, hour_timestamp):
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    cm = confusion_matrix(y_true, y_pred)

    # ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_pred_prob):.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("MLP ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/figures/ROC_curve_{hour_timestamp}.png", bbox_inches = 'tight', dpi = 300)
    plt.show()

    # Precision-Recall Curve
    plt.figure()
    plt.plot(recall, precision)
    plt.title("MLP Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.savefig(f"{output_dir}/figures/Precision_Recall_Curve_{hour_timestamp}.png", bbox_inches = 'tight', dpi = 300)
    plt.show()

    # Confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["Healthy", "AML"])
    disp.plot(cmap=plt.cm.Blues, values_format="d")
    plt.title("MLP Confusion Matrix")
    plt.savefig(f"{output_dir}/figures/Confusion_Matrix_{hour_timestamp}.png", bbox_inches = 'tight', dpi = 300)
    plt.show()

    # Histogram of predicted probabilities
    plt.figure()
    plt.hist(y_pred_prob, bins = 20, alpha = 0.7)
    plt.title("MLP Predicted Probabilities")
    plt.xlabel("Probability of AML")
    plt.ylabel("Number of Samples")
    plt.grid(True)
    plt.savefig(f"{output_dir}/figures/MLP_Predicted_Probabilities_{hour_timestamp}.png", bbox_inches = 'tight', dpi = 300)
    plt.show()

#------------------------------------------------------------------------------------------------
