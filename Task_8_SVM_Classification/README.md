# Level 3 - Task 2: Support Vector Machine Classification

This task compares linear and RBF Support Vector Machine classifiers using the Wisconsin Diagnostic Breast Cancer dataset bundled with scikit-learn.

## What the script does

- Loads 569 labeled records with 30 numerical features.
- Treats malignant diagnosis as the positive class.
- Uses a stratified 80/20 training and test split.
- Standardizes all input features inside leakage-safe pipelines.
- Tunes the linear and RBF kernels independently with 5-fold `GridSearchCV`.
- Compares accuracy, malignant-class precision, recall, F1-score, and ROC-AUC.
- Creates ROC curves and confusion matrices for both kernels.
- Uses a training-fitted two-component PCA projection to visualize decision boundaries.
- Saves tuning results, predictions, metrics, plots, and a written interpretation.

The 2D boundary plots are explanatory visualization models trained only on the two PCA components. The reported evaluation metrics come from the full 30-feature pipelines.

## Run

From the repository root:

```powershell
python .\Task_8_SVM_Classification\task8_svm_classification.py
```

## Generated files

- `breast_cancer_data.csv` - dataset used by the task.
- `outputs/kernel_comparison.csv` - held-out metrics and selected parameters.
- `outputs/grid_search_results.csv` - cross-validation results for both kernels.
- `outputs/test_predictions.csv` - actual and predicted labels plus decision scores.
- `outputs/classification_reports.txt` - per-class reports for both kernels.
- `outputs/roc_curves.png` - ROC comparison.
- `outputs/confusion_matrices.png` - confusion matrices for both kernels.
- `outputs/decision_boundaries.png` - linear and RBF boundaries in PCA space.
- `outputs/summary.txt` - method, results, and interpretation.
