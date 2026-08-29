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
