# Codveda Machine Learning Tasks

This folder contains complete, reproducible solutions for the completed tasks from the provided task list.

## Setup

Run the following once in a terminal:

```powershell
python -m pip install -r requirements.txt
```

Then run each task from its own folder:

```powershell
python task1_preprocessing.py
python task2_linear_regression.py
python task3_knn_classifier.py
python task4_logistic_regression.py
python task5_decision_tree.py
```

Each script creates an `outputs` folder containing the generated dataset, metrics, reports, and/or charts. The solutions use deterministic random seeds so the results can be reproduced.

## Tasks included

1. **Data preprocessing:** missing-value imputation, categorical encoding, numerical standardization, and an 80/20 train/test split.
2. **Simple linear regression:** predicts the diabetes disease-progression target using the BMI feature; reports R-squared, MSE, and the model coefficient.
3. **KNN classifier:** classifies Iris species; compares K values 1 through 15 and reports accuracy, precision, recall, and a confusion matrix.
4. **Logistic regression:** predicts customer churn and reports accuracy, precision, recall, ROC-AUC, odds ratios, and a ROC curve.
5. **Decision tree classification:** classifies Iris species, compares an unpruned tree with a cost-complexity-pruned tree, visualizes the final tree, and reports accuracy and macro F1-score.

## Note on data

The two original ZIP archives were not available in the accessible Downloads/Desktop folders. To keep the submission executable, Task 1 creates a clearly documented raw demo dataset with the required missing and categorical values, Tasks 2, 3, and 5 use stable datasets bundled with scikit-learn, and Task 4 creates a documented synthetic customer-churn dataset with missing values for preprocessing.
