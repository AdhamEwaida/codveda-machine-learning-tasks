from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

RANDOM_STATE = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    iris = load_iris(as_frame=True)
    X = iris.data.copy()
    y = iris.target.copy()

    dataset = X.copy()
    dataset["target"] = y
    dataset["species"] = y.map(dict(enumerate(iris.target_names)))
    dataset.to_csv(OUTPUT_DIR / "iris_dataset.csv", index=False)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    unpruned_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    unpruned_tree.fit(X_train, y_train)

    pruning_path = unpruned_tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = np.unique(pruning_path.ccp_alphas)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    pruning_rows = []
    for alpha in ccp_alphas:
        candidate = DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            ccp_alpha=float(alpha),
        )
        cv_scores = cross_val_score(candidate, X_train, y_train, cv=cv, scoring="f1_macro")
        pruning_rows.append(
            {
                "ccp_alpha": float(alpha),
                "mean_cv_f1_macro": float(cv_scores.mean()),
                "std_cv_f1_macro": float(cv_scores.std()),
            }
        )

    pruning_results = pd.DataFrame(pruning_rows)
    best_score = pruning_results["mean_cv_f1_macro"].max()
    best_alpha = float(
        pruning_results.loc[
            np.isclose(pruning_results["mean_cv_f1_macro"], best_score), "ccp_alpha"
        ].max()
    )

    pruned_tree = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        ccp_alpha=best_alpha,
    )
    pruned_tree.fit(X_train, y_train)

    unpruned_pred = unpruned_tree.predict(X_test)
    pruned_pred = pruned_tree.predict(X_test)

    metrics = pd.DataFrame(
        [
            {
                "model": "unpruned",
                "ccp_alpha": 0.0,
                "accuracy": accuracy_score(y_test, unpruned_pred),
                "f1_macro": f1_score(y_test, unpruned_pred, average="macro"),
                "tree_depth": unpruned_tree.get_depth(),
                "node_count": unpruned_tree.tree_.node_count,
            },
            {
                "model": "pruned",
                "ccp_alpha": best_alpha,
                "accuracy": accuracy_score(y_test, pruned_pred),
                "f1_macro": f1_score(y_test, pruned_pred, average="macro"),
                "tree_depth": pruned_tree.get_depth(),
                "node_count": pruned_tree.tree_.node_count,
            },
        ]
    )
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    pruning_results.to_csv(OUTPUT_DIR / "pruning_cv_results.csv", index=False)

    report = classification_report(
        y_test,
        pruned_pred,
        target_names=iris.target_names,
        digits=4,
    )
    (OUTPUT_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    predictions = X_test.reset_index(drop=True).copy()
    predictions["actual_species"] = y_test.reset_index(drop=True).map(dict(enumerate(iris.target_names)))
    predictions["predicted_species"] = pd.Series(pruned_pred).map(dict(enumerate(iris.target_names)))
    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    plt.figure(figsize=(16, 9))
    plot_tree(
        pruned_tree,
        feature_names=X.columns,
        class_names=list(iris.target_names),
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
    )
    plt.title("Pruned Decision Tree - Iris Classification")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "decision_tree_pruned.svg", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(
        pruning_results["ccp_alpha"],
        pruning_results["mean_cv_f1_macro"],
        marker="o",
    )
    plt.xlabel("ccp_alpha")
    plt.ylabel("Mean 5-fold CV Macro F1")
    plt.title("Cost-Complexity Pruning Selection")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pruning_curve.svg", bbox_inches="tight")
    plt.close()

    cm = confusion_matrix(y_test, pruned_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Pruned Decision Tree - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.svg", bbox_inches="tight")
    plt.close()

    pruned_metrics = metrics.loc[metrics["model"] == "pruned"].iloc[0]
    summary = (
        "LEVEL 2 - TASK 2: DECISION TREES FOR CLASSIFICATION\n\n"
        "Dataset: scikit-learn Iris dataset (150 labeled samples, 3 classes).\n"
        "Split: stratified 80% training / 20% testing (random_state=42).\n"
        "Pruning: cost-complexity pruning (ccp_alpha) selected with 5-fold CV on the training set.\n\n"
        f"Selected ccp_alpha: {best_alpha:.6f}\n"
        f"Pruned tree depth: {int(pruned_metrics['tree_depth'])}\n"
        f"Pruned tree node count: {int(pruned_metrics['node_count'])}\n"
        f"Test accuracy: {pruned_metrics['accuracy']:.4f}\n"
        f"Test macro F1-score: {pruned_metrics['f1_macro']:.4f}\n\n"
        "Classification report\n"
        f"{report}"
    )
    (OUTPUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")

    print(summary)
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
