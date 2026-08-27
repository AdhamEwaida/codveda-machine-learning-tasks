"""Level 1 - Task 3: compare K values for an Iris KNN classifier."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    iris = load_iris(as_frame=True)
    X = iris.data
    y = iris.target
    dataset = X.copy()
    dataset["species"] = y.map(dict(enumerate(iris.target_names)))
    dataset.to_csv(BASE_DIR / "iris_data.csv", index=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    comparison_rows = []
    models = {}
    for k in range(1, 16):
        pipeline = Pipeline([("scaler", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=k))])
        pipeline.fit(X_train, y_train)
        prediction = pipeline.predict(X_test)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, prediction, average="weighted", zero_division=0
        )
        comparison_rows.append(
            {"k": k, "accuracy": accuracy_score(y_test, prediction), "precision_weighted": precision,
             "recall_weighted": recall, "f1_weighted": f1}
        )
        models[k] = pipeline

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(OUTPUT_DIR / "k_comparison.csv", index=False)
    best_k = int(comparison.loc[comparison["accuracy"].idxmax(), "k"])
    best_model = models[best_k]
    best_prediction = best_model.predict(X_test)

    plt.figure(figsize=(8, 5))
    plt.plot(comparison["k"], comparison["accuracy"], marker="o", color="navy")
    plt.xticks(range(1, 16))
    plt.xlabel("Number of neighbors (K)")
    plt.ylabel("Test accuracy")
    plt.title("KNN Performance for Different K Values")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "k_accuracy_comparison.png", dpi=160)
    plt.close()

    matrix = confusion_matrix(y_test, best_prediction)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.xlabel("Predicted species")
    plt.ylabel("Actual species")
    plt.title(f"Confusion Matrix - Best KNN Model (K={best_k})")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    report = classification_report(y_test, best_prediction, target_names=iris.target_names, zero_division=0)
    report_text = f"LEVEL 1 - TASK 3: K-NEAREST NEIGHBORS CLASSIFIER\n\nBest K: {best_k}\n\nClassification report:\n{report}"
    (OUTPUT_DIR / "knn_classification_report.txt").write_text(report_text, encoding="utf-8")
    print(report_text)
    print("Task 3 complete. Outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
