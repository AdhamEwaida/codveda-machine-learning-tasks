# Task 5 - Decision Trees for Classification

This folder contains the solution for **Level 2 (Intermediate) - Task 2: Decision Trees for Classification** from the Codveda Machine Learning task list.

## Objective

Build a decision tree classifier for a categorical outcome, visualize the tree, prune it to reduce overfitting, and evaluate it with classification metrics.

## Dataset

The script uses the **Iris dataset** bundled with scikit-learn. It contains 150 labeled flower samples across three classes: setosa, versicolor, and virginica.

## What the script does

1. Loads the Iris dataset and saves a copy to `outputs/iris_dataset.csv`.
2. Creates a stratified 80/20 train/test split.
3. Trains an unpruned Decision Tree baseline.
4. Uses cost-complexity pruning (`ccp_alpha`) with 5-fold cross-validation on the training set.
5. Chooses the best pruning strength by macro F1-score and trains the final pruned tree.
6. Evaluates both unpruned and pruned models using Accuracy and Macro F1-score.
7. Visualizes the pruned tree, pruning curve, and confusion matrix.

## Run

From this folder:

```powershell
python task5_decision_tree.py
```

Or, after installing the repository requirements:

```powershell
python -m pip install -r ../requirements.txt
python task5_decision_tree.py
```

## Generated outputs

The script creates an `outputs` folder containing:

- `iris_dataset.csv`
- `model_metrics.csv`
- `pruning_cv_results.csv`
- `classification_report.txt`
- `test_predictions.csv`
- `summary.txt`
- `decision_tree_pruned.svg`
- `pruning_curve.svg`
- `confusion_matrix.svg`

The run is reproducible with `random_state=42`.
