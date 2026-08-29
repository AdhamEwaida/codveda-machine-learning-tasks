from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_customer_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    segments = [
        (35, 25, 75),
        (40, 80, 65),
        (85, 25, 60),
        (90, 80, 70),
    ]
    frames = []
    for income, spending, size in segments:
        frames.append(pd.DataFrame({
            "annual_income_k": rng.normal(income, 8, size),
            "spending_score": rng.normal(spending, 9, size),
        }))
    data = pd.concat(frames, ignore_index=True)
    data["annual_income_k"] = data["annual_income_k"].clip(15, 130).round(2)
    data["spending_score"] = data["spending_score"].clip(1, 100).round(2)
    return data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def choose_elbow(k_values: list[int], inertias: list[float]) -> int:
    x = np.array(k_values, dtype=float)
    y = np.array(inertias, dtype=float)
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    start = np.array([x_norm[0], y_norm[0]])
    end = np.array([x_norm[-1], y_norm[-1]])
    line = end - start
    distances = []
    for xi, yi in zip(x_norm, y_norm):
        point = np.array([xi, yi])
        offset = point - start
        cross_2d = line[0] * offset[1] - line[1] * offset[0]
        distance = abs(cross_2d) / np.linalg.norm(line)
        distances.append(distance)
    return int(k_values[int(np.argmax(distances))])


def main() -> None:
    data = build_customer_dataset()
    raw_path = Path(__file__).resolve().parent / "customer_segmentation_data.csv"
    data.to_csv(raw_path, index=False)
    features = ["annual_income_k", "spending_score"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[features])

    k_values = list(range(1, 11))
    inertias = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        model.fit(X_scaled)
        inertias.append(float(model.inertia_))

    optimal_k = choose_elbow(k_values, inertias)
    final_model = KMeans(n_clusters=optimal_k, random_state=RANDOM_STATE, n_init=20)
    labels = final_model.fit_predict(X_scaled)
    silhouette = silhouette_score(X_scaled, labels)

    clustered = data.copy()
    clustered["cluster"] = labels
    clustered.to_csv(OUTPUT_DIR / "clustered_customers.csv", index=False)

    elbow_results = pd.DataFrame({"k": k_values, "inertia": inertias})
    elbow_results.to_csv(OUTPUT_DIR / "elbow_results.csv", index=False)

    centers_scaled = final_model.cluster_centers_
    centers = scaler.inverse_transform(centers_scaled)
    centers_df = pd.DataFrame(centers, columns=features)
    centers_df.index.name = "cluster"
    centers_df.to_csv(OUTPUT_DIR / "cluster_centers.csv")
    summary_rows = clustered.groupby("cluster")[features].agg(["count", "mean"])
    summary_rows.to_csv(OUTPUT_DIR / "cluster_summary.csv")

    plt.figure(figsize=(8, 5))
    plt.plot(k_values, inertias, marker="o")
    plt.axvline(optimal_k, linestyle="--", label=f"Selected K = {optimal_k}")
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for K-Means")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "elbow_curve.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 6))
    for cluster_id in sorted(clustered["cluster"].unique()):
        subset = clustered[clustered["cluster"] == cluster_id]
        plt.scatter(
            subset["annual_income_k"],
            subset["spending_score"],
            label=f"Cluster {cluster_id}",
        )
    plt.scatter(centers[:, 0], centers[:, 1], marker="X", s=220, label="Centroids")
    plt.xlabel("Annual Income (thousands)")
    plt.ylabel("Spending Score")
    plt.title("Customer Segments from K-Means")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "clusters_2d.png", dpi=160)
    plt.close()

    interpretations = []
    income_median = centers_df["annual_income_k"].median()
    spending_median = centers_df["spending_score"].median()
    for cluster_id, row in centers_df.iterrows():
        income_level = "high-income" if row["annual_income_k"] >= income_median else "lower-income"
        spending_level = "high-spending" if row["spending_score"] >= spending_median else "lower-spending"
        interpretations.append(f"Cluster {cluster_id}: {income_level}, {spending_level} customers.")

    summary = (
        "LEVEL 2 - TASK 3: K-MEANS CLUSTERING\n\n"
        f"Dataset: synthetic customer segmentation data with {len(data)} records.\n"
        "Features: annual income and spending score.\n"
        "Preprocessing: StandardScaler applied before clustering.\n"
        "K selection: elbow method over K=1..10.\n"
        f"Selected K: {optimal_k}\n"
        f"Silhouette score: {silhouette:.4f}\n\n"
        "Cluster interpretation\n- " + "\n- ".join(interpretations) + "\n"
    )
    (OUTPUT_DIR / "summary.txt").write_text(summary, encoding="utf-8")

    print(summary)
    print("Cluster centers:")
    print(centers_df.round(2))
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
