# Level 2 - Task 1: Logistic Regression for Binary Classification

This task implements a complete binary-classification workflow using logistic regression to predict customer churn.

## Assignment objectives covered

- Load and preprocess a binary-classification dataset.
- Handle missing numerical and categorical values.
- One-hot encode categorical features and standardize numerical features.
- Train a logistic regression model with scikit-learn.
- Interpret model coefficients and odds ratios.
- Evaluate the model using accuracy, precision, recall, and ROC-AUC.
- Visualize the ROC curve and confusion matrix.

## Dataset

`raw_churn_data.csv` contains 600 documented synthetic customer records. The dataset includes tenure, monthly charges, contract type, internet service, support plan, city, and a binary `churned` target. Missing values are intentionally included so the preprocessing pipeline can demonstrate imputation.

## How to run

From this folder:

```powershell
python task4_logistic_regression.py
```

Dependencies are listed in the repository-level `requirements.txt`.

## Main outputs

Running the script creates or refreshes the files in `outputs/`:

- `evaluation_metrics.csv` - accuracy, precision, recall, and ROC-AUC.
- `coefficient_and_odds_ratio.csv` - fitted coefficients and `exp(coefficient)` odds ratios.
- `test_predictions.csv` - actual labels, predicted labels, and churn probabilities.
- `logistic_regression_report.txt` - model setup, metrics, interpretation, and classification report.
- `roc_curve.png` - ROC curve with AUC.
- `confusion_matrix.png` - confusion matrix for the test set.

## Reproducible result

With `random_state=42`, the committed run reports approximately:

- Accuracy: **0.6917**
- Precision: **0.4130**
- Recall: **0.6552**
- ROC-AUC: **0.7378**

The model uses `class_weight="balanced"` because churn is the minority class in the generated dataset. Positive coefficients increase the log-odds of churn, while negative coefficients decrease them. An odds ratio above 1 indicates increased churn odds and a value below 1 indicates decreased churn odds.
