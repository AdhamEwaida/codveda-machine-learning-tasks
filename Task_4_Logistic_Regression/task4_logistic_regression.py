"""Level 2 - Task 1: Logistic Regression for binary customer-churn prediction."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def create_churn_dataset(path: Path) -> pd.DataFrame:
    """Create documented raw churn data when the supplied archive is unavailable."""
    rng = np.random.default_rng(RANDOM_STATE)
    n_rows = 600
    tenure = rng.integers(1, 73, n_rows)
    monthly_charges = np.round(rng.normal(73, 22, n_rows).clip(20, 145), 2)
    contract = rng.choice(["month-to-month", "one-year", "two-year"], n_rows, p=[0.54, 0.25, 0.21])
    internet_service = rng.choice(["fiber", "dsl", "none"], n_rows, p=[0.47, 0.42, 0.11])
    support_plan = rng.choice(["yes", "no"], n_rows, p=[0.42, 0.58])
    city = rng.choice(["Hebron", "Ramallah", "Nablus", "Bethlehem"], n_rows)
    # This creates a realistic probabilistic binary target for an educational churn task.
    log_odds = (
        -1.15
        + 1.20 * (contract == "month-to-month")
        - 0.85 * (contract == "two-year")
        + 0.65 * (internet_service == "fiber")
        - 0.62 * (support_plan == "yes")
        - 0.024 * tenure
        + 0.014 * (monthly_charges - 70)
    )
    churn_probability = 1 / (1 + np.exp(-log_odds))
    data = pd.DataFrame(
        {
            "customer_id": range(2001, 2001 + n_rows),
            "tenure_months": tenure.astype(float),
            "monthly_charges": monthly_charges,
            "contract": contract,
            "internet_service": internet_service,
            "support_plan": support_plan,
            "city": city,
            "churned": rng.binomial(1, churn_probability),
        }
    )
    data.loc[rng.choice(n_rows, 20, replace=False), "monthly_charges"] = np.nan
    data.loc[rng.choice(n_rows, 12, replace=False), "city"] = None
    data.to_csv(path, index=False)
    return data


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    data_path = BASE_DIR / "raw_churn_data.csv"
    data = pd.read_csv(data_path) if data_path.exists() else create_churn_dataset(data_path)

    X = data.drop(columns=["customer_id", "churned"])
    y = data["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    numeric_features = ["tenure_months", "monthly_charges"]
    categorical_features = ["contract", "internet_service", "support_plan", "city"]
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )
    model = Pipeline(
        [
            ("preprocessing", preprocessing),
            # Balancing classes prioritizes finding likely churners in the minority class.
            ("logistic_regression", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]
    )
    model.fit(X_train, y_train)
    predicted_class = model.predict(X_test)
    predicted_probability = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, predicted_class),
        "precision": precision_score(y_test, predicted_class, zero_division=0),
        "recall": recall_score(y_test, predicted_class, zero_division=0),
        "roc_auc": roc_auc_score(y_test, predicted_probability),
    }
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "evaluation_metrics.csv", index=False)
    predictions = X_test.copy().reset_index(drop=True)
    predictions["actual_churned"] = y_test.reset_index(drop=True)
    predictions["predicted_churned"] = predicted_class
    predictions["churn_probability"] = predicted_probability
    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    feature_names = model.named_steps["preprocessing"].get_feature_names_out()
    coefficients = model.named_steps["logistic_regression"].coef_[0]
    coefficient_table = pd.DataFrame(
        {"feature": feature_names, "coefficient": coefficients, "odds_ratio": np.exp(coefficients)}
    ).sort_values("coefficient", key=np.abs, ascending=False)
    coefficient_table.to_csv(OUTPUT_DIR / "coefficient_and_odds_ratio.csv", index=False)

    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, predicted_probability)
    plt.figure(figsize=(7, 5))
    plt.plot(false_positive_rate, true_positive_rate, color="navy", linewidth=2, label=f"Logistic Regression (AUC = {metrics['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random classifier")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve - Customer Churn Prediction")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=160)
    plt.close()

    matrix = confusion_matrix(y_test, predicted_class)
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(matrix, annot=True, fmt="d", cbar=False, cmap="Blues", xticklabels=["Stayed", "Churned"], yticklabels=["Stayed", "Churned"])
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title("Confusion Matrix - Logistic Regression")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    report = f"""LEVEL 2 - TASK 1: LOGISTIC REGRESSION FOR BINARY CLASSIFICATION

Objective: predict whether a customer will churn (1) or stay (0).
Dataset: documented synthetic customer-churn data; 600 records.
Preprocessing: median/mode imputation, one-hot encoding, and numeric standardization.
Class handling: class_weight='balanced' to improve detection of the minority churn class.
Split: stratified 80% training / 20% testing (random_state={RANDOM_STATE}).

Test metrics
- Accuracy: {metrics['accuracy']:.4f}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- ROC-AUC: {metrics['roc_auc']:.4f}

Coefficient interpretation
Positive coefficients increase the log-odds of churn; negative coefficients decrease it.
The odds_ratio column in coefficient_and_odds_ratio.csv is exp(coefficient): values above
1 increase the odds of churn, while values below 1 decrease them.

Classification report
{classification_report(y_test, predicted_class, target_names=['Stayed', 'Churned'], zero_division=0)}
"""
    (OUTPUT_DIR / "logistic_regression_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print("Task 4 complete. Outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
