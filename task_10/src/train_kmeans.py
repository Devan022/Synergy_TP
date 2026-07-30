import json
import os
from pyexpat import model
import pandas as pd
import numpy as np
from plots import plot_clusters
from data_utils import (
    load_dataset,
    clean_dataset
)
from metrics import silhouette_score
from kmeans import KMeans


def train_kmeans(csv_path):

    # Load dataset
    df = load_dataset(csv_path)
    df = clean_dataset(df)

    features = [
        "T",
        "RH",
        "AH"
    ]

    data = df[features].dropna()

    X = data.values

    # Normalize
    mean = X.mean(axis=0)
    std = X.std(axis=0)

    std[std == 0] = 1

    X = (X - mean) / std

    # Train KMeans
    model = KMeans(
        k=3,
        max_iterations=100
    )

    model.fit(X)

    counts = model.cluster_counts()
    inertia = model.inertia(X)
    silhouette = silhouette_score(X, model.labels)
    print("\nKMeans Results")
    print("----------------------")
    print("Cluster Counts:")
    print(counts)

    print("\nInertia:")
    print(inertia)
    print("\nSilhouette Score:")
    print(silhouette)
    # Create output folder
    os.makedirs("../output", exist_ok=True)

    # Save metrics
    metrics = {
        "Inertia": float(inertia),
        "Silhouette Score": float(silhouette),
        "Cluster Counts": {
            str(int(k)): int(v)
            for k, v in counts.items()
        }
    }

    with open("../output/clustering_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Save cluster assignments
    results = pd.DataFrame({
        "Cluster": model.labels
    })

    results.to_csv(
        "../output/clustering_assignments.csv",
        index=False
    )

    print("\nClustering results saved successfully!")
    plot_clusters(
    X,
    model.labels
    )

    print("Clustering plot saved successfully!")
    return model


if __name__ == "__main__":
    train_kmeans("../data/AirQualityUCI.csv")