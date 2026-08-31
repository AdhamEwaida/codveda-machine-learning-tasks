# Level 3 - Task 1: Random Forest Classifier

This task builds and tunes a Random Forest classifier using the Wisconsin Diagnostic Breast Cancer dataset bundled with scikit-learn.

## What the script does

- Loads 569 labeled records with 30 numerical features.
- Uses a stratified 80/20 training and test split.
- Tunes tree count, maximum depth, minimum split size, and minimum leaf size with 5-fold `GridSearchCV`.
- Uses F1-score for the malignant class as the tuning objective.
- Evaluates the selected model with accuracy, precision, recall, F1-score, and ROC-AUC.
- Reports independent 5-fold cross-validation metrics on the training set.
- Saves a confusion matrix and a ranked feature-importance chart.
- Writes all metrics, predictions, tuning results, and feature importances to reproducible output files.

The malignant diagnosis is treated as the positive class because detecting cancer cases is the important classification objective.

## Run

From the repository root:

```powershell
python .\Task_7_Random_Forest\task7_random_forest.py
```

## Generated files

- `breast_cancer_data.csv` - the source dataset used by the task.
- `outputs/evaluation_metrics.csv` - final held-out test metrics.
- `outputs/classification_report.txt` - per-class precision, recall, and F1-score.
- `outputs/cross_validation_scores.csv` - metrics for each validation fold.
- `outputs/grid_search_results.csv` - every tested hyperparameter combination.
- `outputs/test_predictions.csv` - true labels, predicted labels, and probabilities.
- `outputs/feature_importances.csv` - all features ranked by importance.
- `outputs/feature_importance.png` - top 15 feature importances.
- `outputs/confusion_matrix.png` - held-out test confusion matrix.
- `outputs/summary.txt` - dataset, best parameters, CV results, and test results.
