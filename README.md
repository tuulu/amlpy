### **amlpy**

amlpy is a small-scale tool designed to analyze Affymetrix microarray data for detecting acute myeloid leukemia (AML). Utilizing several AI/ML methods, amlpy can assess mRNA microarray samples to determine the likelihood of them being cancerous.

## 🚀 **Features**

amlpy offers several features designed to facilitate effective analysis and interpretation of microarray data:

Data Preprocessing: Automates the cleaning and normalization of Affymetrix microarray data to prepare it for analysis.

Machine Learning Model Training & Evaluation: Implements robust training procedures with cross-validation to optimize and evaluate predictive models.

Cancer Prediction Generation: Delivers predictions on the probability of AML presence in microarray samples

## 🛠️ **Installation**
To get started with amlpy, follow these steps to set up the environment on your local machine:

1) Clone the Repository: Clone the amlpy repository to your local machine using the following command:
```bash
git clone https://github.com/tuulu/amlpy.git

cd amlpy
```
2) Install dependencies: Install all necessary Python packages listed in the requirements.txt file:
```bash
pip install -r requirements.txt
```
## 🧪 **Usage**

1) Please ensure you have the data downloaded first by running .src/data_import.py as a standalone script 
```bash
python src/data_import.py
```
2) Once the datasets are installed, start the pipeline by running .
```bash
python main.py
```


