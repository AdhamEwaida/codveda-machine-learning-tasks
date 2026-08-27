"""Level 1 - Task 2: a simple linear regression model for a continuous target."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    diabetes = load_diabetes(as_frame=True)
    # One predictor (BMI) is intentionally used so this is simple linear regression.
    data = diabetes.frame[["bmi", "target"]]
    data.to_csv(BASE_DIR / "diabetes_bmi_data.csv", index=False)
    X = data[["bmi"]]
    y = data["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    results = pd.DataFrame(
        {"bmi": X_test["bmi"].to_numpy(), "actual_target": y_test.to_numpy(), "predicted_target": predictions}
    ).sort_values("bmi")
    results.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    plt.figure(figsize=(8, 5))
    plt.scatter(X_test["bmi"], y_test, alpha=0.75, label="Actual test values")
    plt.plot(results["bmi"], results["predicted_target"], color="crimson", linewidth=2, label="Regression line")
    plt.xlabel("BMI feature (standardized)")
    plt.ylabel("Disease progression target")
    plt.title("Simple Linear Regression: BMI vs. Disease Progression")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "regression_plot.png", dpi=160)
    plt.close()

    intercept = model.intercept_
    coefficient = model.coef_[0]
    report = f"""LEVEL 1 - TASK 2: SIMPLE LINEAR REGRESSION

Dataset: scikit-learn diabetes dataset (442 observations)
Predictor: BMI feature (standardized)
Target: quantitative disease-progression measure
Training / test split: 80% / 20% (random_state={RANDOM_STATE})

Model equation:
predicted_target = {intercept:.4f} + ({coefficient:.4f} * BMI)

Interpretation:
For every one-unit increase in the standardized BMI feature, the predicted
disease-progression target increases by {coefficient:.4f} units on average.

Test-set R-squared: {r2:.4f}
Test-set mean squared error (MSE): {mse:.4f}
"""
    (OUTPUT_DIR / "linear_regression_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print("Task 2 complete. Outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
