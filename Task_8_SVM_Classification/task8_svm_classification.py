"""Level 3 - Task 2: compare linear and RBF SVM classifiers."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"


def build_search(kernel: str, cross_validation: StratifiedKFold) -> GridSearchCV:
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("svc", SVC(kernel=kernel, class_weight="balanced")),
        ]
    )
    parameter_grid: dict[str, list[object]] = {
        "svc__C": [0.01, 0.1, 1, 10, 100]
    }
    if kernel == "rbf":
        parameter_grid["svc__gamma"] = ["scale", 0.001, 0.01, 0.1, 1]
    return GridSearchCV(
        pipeline,
        parameter_grid,
        scoring="roc_auc",
        cv=cross_validation,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
    )


def evaluate_model(
    name: str,
    search: GridSearchCV,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    prediction = search.predict(X_test)
    decision_score = search.decision_function(X_test)
    row: dict[str, object] = {
        "kernel": name,
        "accuracy": accuracy_score(y_test, prediction),
        "precision_malignant": precision_score(y_test, prediction),
        "recall_malignant": recall_score(y_test, prediction),
        "f1_malignant": f1_score(y_test, prediction),
        "roc_auc": roc_auc_score(y_test, decision_score),
        "best_cv_roc_auc": search.best_score_,
        "best_C": search.best_params_["svc__C"],
        "best_gamma": search.best_params_.get("svc__gamma", "not applicable"),
    }
    return row, prediction, decision_score


def save_decision_boundaries(
    searches: dict[str, GridSearchCV],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_train_2d = pca.fit_transform(X_train_scaled)

    x_margin = 0.8
    y_margin = 0.8
    x_min, x_max = X_train_2d[:, 0].min() - x_margin, X_train_2d[:, 0].max() + x_margin
    y_min, y_max = X_train_2d[:, 1].min() - y_margin, X_train_2d[:, 1].max() + y_margin
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 350), np.linspace(y_min, y_max, 350)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    background = ListedColormap(["#d8eef8", "#f9d7d7"])
    point_colors = ListedColormap(["#1479b8", "#c62828"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    for axis, (name, search) in zip(axes, searches.items()):
        parameters = search.best_params_
        visual_model = SVC(
            kernel=name,
            C=parameters["svc__C"],
            gamma=parameters.get("svc__gamma", "scale"),
            class_weight="balanced",
        )
        visual_model.fit(X_train_2d, y_train)
        regions = visual_model.predict(grid).reshape(xx.shape)
        scores = visual_model.decision_function(grid).reshape(xx.shape)
        axis.contourf(xx, yy, regions, alpha=0.65, cmap=background)
        axis.contour(xx, yy, scores, levels=[0], colors="black", linewidths=1.2)
        axis.scatter(
            X_train_2d[:, 0],
            X_train_2d[:, 1],
            c=y_train,
            cmap=point_colors,
            edgecolor="white",
            linewidth=0.35,
            s=28,
            alpha=0.85,
        )
        axis.set_title(f"{name.upper()} kernel")
        axis.set_xlabel("Principal component 1")
    axes[0].set_ylabel("Principal component 2")
    fig.suptitle("SVM Decision Boundaries in Training-Fitted PCA Space")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "decision_boundaries.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_breast_cancer(as_frame=True)
    X = dataset.data.copy()
    # scikit-learn encodes malignant as 0; remap it to the positive class (1).
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

    searches = {
        "linear": build_search("linear", cross_validation),
        "rbf": build_search("rbf", cross_validation),
    }
    comparison_rows = []
    predictions: dict[str, np.ndarray] = {}
    decision_scores: dict[str, np.ndarray] = {}
    for name, search in searches.items():
        search.fit(X_train, y_train)
        row, prediction, score = evaluate_model(name, search, X_test, y_test)
        comparison_rows.append(row)
        predictions[name] = prediction
        decision_scores[name] = score

    comparison = pd.DataFrame(comparison_rows).sort_values(
        "roc_auc", ascending=False, ignore_index=True
    )
    comparison.to_csv(OUTPUT_DIR / "kernel_comparison.csv", index=False)

    grid_tables = []
    for name, search in searches.items():
        table = pd.DataFrame(search.cv_results_)[
            [
                "rank_test_score",
                "mean_test_score",
                "std_test_score",
                "param_svc__C",
                "param_svc__gamma",
            ]
            if name == "rbf"
            else [
                "rank_test_score",
                "mean_test_score",
                "std_test_score",
                "param_svc__C",
            ]
        ].copy()
        table.insert(0, "kernel", name)
        if "param_svc__gamma" not in table:
            table["param_svc__gamma"] = "not applicable"
        grid_tables.append(table)
    pd.concat(grid_tables, ignore_index=True).sort_values(
        ["kernel", "rank_test_score"]
    ).to_csv(OUTPUT_DIR / "grid_search_results.csv", index=False)

    prediction_table = pd.DataFrame(
        {
            "row_index": X_test.index,
            "actual_diagnosis": y_test.map({0: "benign", 1: "malignant"}),
        }
    ).reset_index(drop=True)
    for name in searches:
        prediction_table[f"{name}_prediction"] = pd.Series(
            predictions[name]
        ).map({0: "benign", 1: "malignant"})
        prediction_table[f"{name}_decision_score"] = decision_scores[name]
    prediction_table.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    reports = []
    for name in searches:
        report = classification_report(
            y_test,
            predictions[name],
            labels=[0, 1],
            target_names=["Benign", "Malignant"],
            digits=4,
        )
        reports.append(f"{name.upper()} KERNEL\n{'=' * 40}\n{report}")
    (OUTPUT_DIR / "classification_reports.txt").write_text(
        "\n\n".join(reports), encoding="utf-8"
    )

    plt.figure(figsize=(8, 6))
    for name in searches:
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_test, decision_scores[name]
        )
        auc = roc_auc_score(y_test, decision_scores[name])
        plt.plot(
            false_positive_rate,
            true_positive_rate,
            linewidth=2,
            label=f"{name.upper()} (AUC = {auc:.4f})",
        )
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random classifier")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("SVM Kernel ROC Comparison")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roc_curves.png", dpi=160)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, name in zip(axes, searches):
        matrix = confusion_matrix(y_test, predictions[name], labels=[0, 1])
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Benign", "Malignant"],
            yticklabels=["Benign", "Malignant"],
            ax=axis,
        )
        axis.set_title(f"{name.upper()} kernel")
        axis.set_xlabel("Predicted diagnosis")
        axis.set_ylabel("Actual diagnosis")
    fig.suptitle("SVM Confusion Matrices")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=160)
    plt.close(fig)

    save_decision_boundaries(searches, X_train, y_train)

    best = comparison.iloc[0]
    lines = [
        "LEVEL 3 - TASK 2: SUPPORT VECTOR MACHINE CLASSIFICATION",
        "",
        "Dataset: Wisconsin Diagnostic Breast Cancer dataset",
        f"Records: {len(X)}",
        f"Features: {X.shape[1]}",
        "Positive class: malignant diagnosis",
        f"Split: stratified 80% training / 20% testing (random_state={RANDOM_STATE})",
        "Preprocessing: StandardScaler fitted inside each cross-validation fold",
        "Tuning: separate 5-fold GridSearchCV searches optimized for ROC-AUC",
        "",
        "Kernel comparison on held-out test data",
        comparison.to_string(index=False),
        "",
        f"Best held-out ROC-AUC: {best['kernel'].upper()} kernel ({best['roc_auc']:.4f})",
        "",
        "Interpretation:",
        "- The linear kernel tests a straight separating boundary.",
        "- The RBF kernel can model nonlinear boundaries.",
        "- Full 30-feature models produce the reported metrics.",
        "- Separate PCA-based 2D models are used only to visualize the boundary shapes.",
    ]
    summary = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
