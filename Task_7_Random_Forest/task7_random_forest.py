"""Level 3 - Task 1: tuned Random Forest classification and feature importance."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.copy()
    # scikit-learn stores malignant as 0. Remap it to 1 so cancer is positive.
    y = (dataset.target == 0).astype(int).rename("malignant")
    exported_data = X.copy()
    exported_data["diagnosis"] = y.map({0: "benign", 1: "malignant"})
    exported_data.to_csv(BASE_DIR / "breast_cancer_data.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    cross_validation = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    parameter_grid = {
        "n_estimators": [100, 200, 400],
        "max_depth": [None, 8, 16],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    search = GridSearchCV(
        estimator=RandomForestClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        param_grid=parameter_grid,
        scoring="f1",
        cv=cross_validation,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    predicted_class = best_model.predict(X_test)
    malignant_probability = best_model.predict_proba(X_test)[:, 1]
    test_metrics = {
        "accuracy": accuracy_score(y_test, predicted_class),
        "precision_malignant": precision_score(y_test, predicted_class),
        "recall_malignant": recall_score(y_test, predicted_class),
        "f1_malignant": f1_score(y_test, predicted_class),
        "roc_auc": roc_auc_score(y_test, malignant_probability),
    }
    pd.DataFrame([test_metrics]).to_csv(
        OUTPUT_DIR / "evaluation_metrics.csv", index=False
    )

    validation_results = cross_validate(
        best_model,
        X_train,
        y_train,
        cv=cross_validation,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
        },
        n_jobs=-1,
    )
    validation_table = pd.DataFrame(
        {
            "fold": range(1, cross_validation.get_n_splits() + 1),
            "accuracy": validation_results["test_accuracy"],
            "precision_malignant": validation_results["test_precision"],
            "recall_malignant": validation_results["test_recall"],
            "f1_malignant": validation_results["test_f1"],
        }
    )
    validation_table.to_csv(
        OUTPUT_DIR / "cross_validation_scores.csv", index=False
    )

    grid_columns = [
        "rank_test_score",
        "mean_test_score",
        "std_test_score",
        "param_n_estimators",
        "param_max_depth",
        "param_min_samples_split",
        "param_min_samples_leaf",
    ]
    grid_results = pd.DataFrame(search.cv_results_)[grid_columns].sort_values(
        ["rank_test_score", "mean_test_score"], ascending=[True, False]
    )
    grid_results.to_csv(OUTPUT_DIR / "grid_search_results.csv", index=False)

    predictions = pd.DataFrame(
        {
            "row_index": X_test.index,
            "actual_diagnosis": y_test.map({0: "benign", 1: "malignant"}),
            "predicted_diagnosis": pd.Series(
                predicted_class, index=y_test.index
            ).map({0: "benign", 1: "malignant"}),
            "malignant_probability": malignant_probability,
        }
    ).reset_index(drop=True)
    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    importances = pd.DataFrame(
        {"feature": X.columns, "importance": best_model.feature_importances_}
    ).sort_values("importance", ascending=False, ignore_index=True)
    importances.to_csv(OUTPUT_DIR / "feature_importances.csv", index=False)

    top_features = importances.head(15).sort_values("importance")
    plt.figure(figsize=(9, 7))
    plt.barh(top_features["feature"], top_features["importance"], color="teal")
    plt.xlabel("Gini importance")
    plt.ylabel("Feature")
    plt.title("Top 15 Random Forest Feature Importances")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=160)
    plt.close()

    matrix = confusion_matrix(y_test, predicted_class, labels=[0, 1])
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
    )
    plt.xlabel("Predicted diagnosis")
    plt.ylabel("Actual diagnosis")
    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    report = classification_report(
        y_test,
        predicted_class,
        labels=[0, 1],
        target_names=["Benign", "Malignant"],
        digits=4,
    )
    (OUTPUT_DIR / "classification_report.txt").write_text(
        report, encoding="utf-8"
    )

    cv_means = validation_table.drop(columns="fold").mean()
    best_parameters = ", ".join(
        f"{name}={value}" for name, value in search.best_params_.items()
    )
    summary = f"""LEVEL 3 - TASK 1: RANDOM FOREST CLASSIFIER

Dataset: Wisconsin Diagnostic Breast Cancer dataset
Records: {len(X)}
Features: {X.shape[1]}
Positive class: malignant diagnosis
Split: stratified 80% training / 20% testing (random_state={RANDOM_STATE})
Hyperparameter tuning: 5-fold GridSearchCV optimized for malignant-class F1
Best parameters: {best_parameters}
Best grid-search F1: {search.best_score_:.4f}

Five-fold cross-validation means on training data
- Accuracy: {cv_means['accuracy']:.4f}
- Precision (malignant): {cv_means['precision_malignant']:.4f}
- Recall (malignant): {cv_means['recall_malignant']:.4f}
- F1 (malignant): {cv_means['f1_malignant']:.4f}

Held-out test metrics
- Accuracy: {test_metrics['accuracy']:.4f}
- Precision (malignant): {test_metrics['precision_malignant']:.4f}
- Recall (malignant): {test_metrics['recall_malignant']:.4f}
- F1 (malignant): {test_metrics['f1_malignant']:.4f}
- ROC-AUC: {test_metrics['roc_auc']:.4f}

Most important features
{importances.head(10).to_string(index=False)}
"""
    (OUTPUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
