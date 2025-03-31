# This file contains the code for the random forest model
# We will use it for classification of the AML and healthy samples

#------------------------------------------------------------------------------------------------
# Call libraries
import os as os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve, precision_recall_curve, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import precision_score,recall_score 
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

#------------------------------------------------------------------------------------------------
def prepare_rf(df, labels, label_map):
    """Prepares the data for the random forest model"""
    # Checking if the df is already transposed (genes as columns i.e., we dont want any "GSM..." column names)
    if df.columns.str.contains("GSM").any():
        df.columns = labels
        df = df.T

    # Setting the labels
    target = pd.Series([label_map[label] for label in labels], name = "State")

    return df, target

#------------------------------------------------------------------------------------------------
def train_rf(X, y, test_size = 0.2, random_state = 42, optimize = False):
    """Trains a Random Forest model with SMOTE and PCA. """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = random_state)
    
    # Apply SMOTE
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Apply PCA
    pca = PCA(n_components = 2)
    X_train_pca = pca.fit_transform(X_train_resampled)
    X_test_pca = pca.transform(X_test)
    
    if optimize:
        # Use optimized parameters if optimize is True
        rf = optimize_rf_parameters(X_train_pca, y_train_resampled)
    else:
        # Use default parameters
        rf = RandomForestClassifier(
            n_estimators = 500,
            max_depth = 2,
            bootstrap = True,
            min_samples_leaf = 1,
            random_state = random_state,
            max_features = 3
        )
        rf.fit(X_train_pca, y_train_resampled)
    
    return rf, X_train_pca, X_test_pca, y_train_resampled, y_test

#------------------------------------------------------------------------------------------------
def evaluate_rf(rf, X_train_pca, X_test_pca, y_train, y_test, output_dir, hour_timestamp):
    """Evaluates the model performance and creates visualizations."""
    # Make predictions
    training_predictions = rf.predict(X_train_pca)
    test_predictions = rf.predict(X_test_pca)
    test_probabilities = rf.predict_proba(X_test_pca)[:, 1]
    
    # Calculate accuracies
    train_accuracy = accuracy_score(y_train, training_predictions)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    print(f"Random Forest Training Accuracy: {train_accuracy}")
    print(f"Random Forest Test Accuracy: {test_accuracy}")
    
    # Calculate additional metrics
    print("\nEvaluation Metrics")
    print(f"AUC-ROC: {roc_auc_score(y_test, test_probabilities):.4f}")
    print(f"Precision: {precision_score(y_test, test_predictions):.4f}")
    print(f"Recall: {recall_score(y_test, test_predictions):.4f}")
    print(f"F1 Score: {f1_score(y_test, test_predictions):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_predictions, target_names=["Healthy", "AML"]))
    
    # Create visualizations if output directory is provided
    if output_dir is not None and hour_timestamp is not None:
        plot_rf_metrics(y_test, test_probabilities, test_predictions, output_dir, hour_timestamp)
    
    return train_accuracy, test_accuracy

def plot_rf_metrics(y_true, y_pred_prob, y_pred, output_dir, hour_timestamp):
    """Creates and saves visualization plots for Random Forest results."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    cm = confusion_matrix(y_true, y_pred)

    # ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_pred_prob):.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title("Random Forest ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_dir}/figures/RF_ROC_curve_{hour_timestamp}.png", bbox_inches = 'tight', dpi = 300)
    plt.show()

    # Precision-Recall Curve
    plt.figure()
    plt.plot(recall, precision)
    plt.title("Random Forest Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.savefig(f"{output_dir}/figures/RF_Precision_Recall_Curve_{hour_timestamp}.png", bbox_inches = 'tight', dpi = 300)
    plt.show()

    # Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ["Healthy", "AML"])
    disp.plot(cmap = 'Blues', values_format = "d")
    plt.title("Random Forest Confusion Matrix")
    plt.savefig(f"{output_dir}/figures/RF_Confusion_Matrix_{hour_timestamp}.png", bbox_inches = 'tight', dpi = 300)
    plt.show()

    # Histogram of predicted probabilities
    plt.figure()
    plt.hist(y_pred_prob, bins=20, alpha=0.7)
    plt.title("Random Forest Predicted Probabilities")
    plt.xlabel("Probability of AML")
    plt.ylabel("Number of Samples")
    plt.grid(True)
    plt.savefig(f"{output_dir}/figures/RF_Predicted_Probabilities_{hour_timestamp}.png", bbox_inches = 'tight', dpi = 300)
    plt.show()

#------------------------------------------------------------------------------------------------
# This function is entirely optional and not required for the pipeline
# It is used to optimize the Random Forest parameters using GridSearchCV
# It takes a bit to run :)
def optimize_rf_parameters(X_train_pca, y_train_resampled):
    """Optimizes Random Forest parameters using GridSearchCV."""
    # Define parameter grid focusing on key parameters
    param_grid = {
        'n_estimators': [100, 300, 500],    # Number of trees
        'max_depth': [2, 3, 4, 5, None],    # Tree depth
        'max_features': [2, 3, 4, 'sqrt'],  # Features to consider for splits
        'min_samples_leaf': [1, 2]}  # Minimum samples at leaf nodes
    
    # Initialize base model
    base_rf = RandomForestClassifier(
        bootstrap=True, # Keep bootstrap as True
        random_state = 42)
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator = base_rf,
        param_grid = param_grid,
        cv = 5, # 5-fold cross-validation
        n_jobs = -1,    # Use all available cores
        scoring = 'accuracy',
        verbose = 1)
    
    # Fit GridSearchCV
    grid_search.fit(X_train_pca, y_train_resampled)
    
    # Print best parameters and score
    print("\nBest parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    print("Final score:")
    return grid_search.best_estimator_
