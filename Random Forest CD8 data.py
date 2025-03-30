# Importing the necessary libraries
import pandas as pd
import os as os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score,recall_score 
from sklearn.decomposition import PCA
# Setting the working directory
os.chdir("C:/Users/User/OneDrive/Dokumente/Python_For_Life_Sciences/Coding for group project") 
# Reading csv and manipulating with row names 
diff_CD8=pd.read_csv('./DiffGenExpres CD8')
diff_CD8.set_index('Gene Symbol', inplace=True) 
diff_CD8.head()
# Filtering out the samples which could not mapped to a gen symbol 
diff_CD8_filtered = diff_CD8[diff_CD8.index != '---']
diff_CD8_filtered.shape
diff_CD8_filtered.head()
diff_CD8_filtered.columns
# In order to simplify the training the column names were assigned as 'AML' or'healthy'
column_names=['AML','AML','AML','AML','AML','AML','AML','AML','AML','AML','healthy','healthy','healthy','healthy','healthy','healthy','healthy','healthy','healthy','healthy','healthy']
diff_CD8_filtered.columns = column_names
print(diff_CD8_filtered.head())
# Proving if there are NA values in the dataset
diff_CD8_filtered.isna().sum() # 0 NA values across the rows 

# Setting the labels
labels=['AML','AML','AML','AML','AML','AML','AML','AML','AML','AML','healthy','healthy','healthy','healthy','healthy','healthy','healthy','healthy','healthy','healthy','healthy']
# Preparing labels 1 for AML and 0 for healthy
label_map = {'AML': 1, 'healthy': 0}
target = pd.Series([label_map[label] for label in labels], name="State")
# Transposing to have samples as rows 
X = diff_CD8_filtered.T 

# Random Forest Model 
# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, target, test_size=0.2, random_state=42) 
y_train.hist() 
# As the train data was unblalanced SMOTE was implemented for oversampling a minority class
# !pip install imbalanced-learn
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
y_train_resampled.hist() 

# Applying Principal Component Analysis=PCA to reduce the dimensionality of the data
# Reducing to 2 components
pca = PCA(n_components=2) 
# Fit and transform the training data; PCA learns the 2 most important components 
X_train_pca = pca.fit_transform(X_train_resampled) 
# Transform the test data best on 2 most essential components 
X_test_pca = pca.transform(X_test)      
# Initialize and fit the Random Forest model
rf = RandomForestClassifier(n_estimators=500, max_depth=2,bootstrap=True,min_samples_leaf= 1,random_state=42,max_features= 3)
rf.fit(X_train_pca, y_train_resampled)

# Assessing accuracy on training and test data 
training_predictions=rf.predict(X_train_pca)
test_predictions = rf.predict(X_test_pca) 

# Calculate accuracy on both training and test data
train_accuracy = accuracy_score(y_train_resampled, training_predictions)
test_accuracy = accuracy_score(y_test, test_predictions)
# Print the accuracies
print(f"Training Accuracy: {train_accuracy}")
print(f"Test Accuracy: {test_accuracy}") 



# To determine the best fitting parameteres the GridSearch was conducted 
# The Grid Search parameters
#gr_space = {
#    'max_depth': [2,3,4,5],
#    'n_estimators': [100, 200, 300, 400, 500],
#    'max_features': [3,5,7,9],
#    'min_samples_leaf': [1, 2, 4]
#}
# Specifying the GridSearch algorithm 
#grid = GridSearchCV(rf, gr_space, cv = 3, scoring='accuracy', verbose = 3)
#model_grid = grid.fit(X_train_resampled, y_train_resampled)

#print('Best hyperparameters are '+str(model_grid.best_params_))
#print('Best score is: ' + str(model_grid.best_score_)) 

# Best hyperparameters are {'max_depth': 2, 'max_features': 3, 'min_samples_leaf': 1, 'n_estimators': 100}
#rf_optimized=RandomForestClassifier( max_depth=2, max_features= 3, min_samples_leaf= 1, n_estimators=100) 
#rf_optimized.fit(X_train_resampled,y_train_resampled)
#y_pred=rf_optimized.predict(X_test)
# Scoring the results
#print(f'Accuracy score is:{accuracy_score(y_test, y_pred)}')
#print(f'Recall score is:{recall_score(y_test, y_pred)}')
#print(f'Precision score is:{precision_score(y_test, y_pred)}') 
