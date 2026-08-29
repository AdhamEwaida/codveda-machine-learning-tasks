# Level 2 - Task 3: K-Means Clustering

This task implements K-Means clustering for customer segmentation using Python, pandas, scikit-learn, and matplotlib.

## What the script does

- Builds a reproducible unlabeled customer-segmentation dataset.
- Uses annual income and spending score as clustering features.
- Standardizes the numerical features with `StandardScaler`.
- Runs K-Means for K values from 1 to 10.
- Uses the elbow method to select the number of clusters.
- Fits the final K-Means model.
- Calculates the silhouette score as an additional quality check.
- Visualizes the elbow curve and the final 2D customer clusters.
- Saves cluster centers, clustered rows, and cluster summaries.

## Run

From the repository root:

```powershell
python .\Task_6_KMeans_Clustering\task6_kmeans_clustering.py
```

The task is deterministic because it uses a fixed random seed.
## Generated files

Running the script creates:

- `customer_segmentation_data.csv` - the raw unlabeled dataset.
- `outputs/elbow_results.csv` - inertia for each tested K value.
- `outputs/clustered_customers.csv` - every customer with its assigned cluster.
- `outputs/cluster_centers.csv` - final cluster centroids in the original feature scale.
- `outputs/cluster_summary.csv` - count and mean values for each cluster.
- `outputs/elbow_curve.png` - elbow-method visualization.
- `outputs/clusters_2d.png` - 2D cluster visualization with centroids.
- `outputs/summary.txt` - selected K, silhouette score, and interpretation.

## Requirement coverage

This implementation covers the Codveda Level 2 K-Means objectives: scaling, K-Means clustering, elbow-method selection of K, 2D visualization, and interpretation of the clustering results.
