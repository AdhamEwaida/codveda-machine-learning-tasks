"""Level 1 - Task 1: prepare raw customer data for machine learning."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def create_raw_dataset(path: Path) -> pd.DataFrame:
    """Create a small raw dataset with missing and categorical values for this task."""
    rng = np.random.default_rng(RANDOM_STATE)
    rows = 120
    data = pd.DataFrame(
        {
            "customer_id": range(1001, 1001 + rows),
            "age": rng.integers(18, 70, size=rows).astype(float),
            "monthly_income": rng.normal(3500, 1100, size=rows).round(2),
            "account_type": rng.choice(["basic", "premium", "business"], size=rows),
            "city": rng.choice(["Hebron", "Ramallah", "Nablus"], size=rows),
            "churned": rng.choice([0, 1], size=rows, p=[0.78, 0.22]),
        }
    )
    # Deliberately introduce raw-data quality issues addressed by this solution.
    data.loc[rng.choice(rows, 10, replace=False), "age"] = np.nan
    data.loc[rng.choice(rows, 8, replace=False), "monthly_income"] = np.nan
    data.loc[rng.choice(rows, 6, replace=False), "city"] = None
    data.to_csv(path, index=False)
    return data


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    raw_path = BASE_DIR / "raw_customer_data.csv"
    raw_data = pd.read_csv(raw_path) if raw_path.exists() else create_raw_dataset(raw_path)

    # customer_id is an identifier, so it should not be used as a predictive feature.
    features = raw_data.drop(columns=["customer_id", "churned"])
    target = raw_data["churned"]
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.20, random_state=RANDOM_STATE, stratify=target
    )

    numeric_features = ["age", "monthly_income"]
    categorical_features = ["account_type", "city"]
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )

    # Fit only on training data, which prevents information leakage from the test set.
    X_train_ready = preprocessor.fit_transform(X_train)
    X_test_ready = preprocessor.transform(X_test)
    column_names = preprocessor.get_feature_names_out()
    train_ready = pd.DataFrame(X_train_ready, columns=column_names)
    test_ready = pd.DataFrame(X_test_ready, columns=column_names)
    train_ready["churned"] = y_train.reset_index(drop=True)
    test_ready["churned"] = y_test.reset_index(drop=True)
    train_ready.to_csv(OUTPUT_DIR / "train_preprocessed.csv", index=False)
    test_ready.to_csv(OUTPUT_DIR / "test_preprocessed.csv", index=False)

    summary = [
        "LEVEL 1 - TASK 1: DATA PREPROCESSING",
        f"Raw dataset shape: {raw_data.shape}",
        "Missing values before preprocessing:",
        raw_data.isna().sum().to_string(),
        f"\nTrain set shape: {train_ready.shape}",
        f"Test set shape: {test_ready.shape}",
        "\nOperations completed:",
        "- Numeric missing values filled with the training-set median.",
        "- Categorical missing values filled with the training-set mode.",
        "- Categorical variables encoded using one-hot encoding.",
        "- Numeric variables standardized with StandardScaler.",
        "- Data split into stratified 80% training and 20% testing sets.",
    ]
    (OUTPUT_DIR / "preprocessing_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("Task 1 complete. Outputs saved to:", OUTPUT_DIR)
    print("Training rows:", len(train_ready), "| Testing rows:", len(test_ready))


if __name__ == "__main__":
    main()
