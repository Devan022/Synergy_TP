import numpy as np


class KMeans:

    def __init__(self, k=3, max_iterations=100):

        self.k = k
        self.max_iterations = max_iterations

        self.centroids = None
        self.labels = None

    def fit(self, X):

        np.random.seed(42)

        random_indices = np.random.choice(
            len(X),
            self.k,
            replace=False
        )

        self.centroids = X[random_indices]

        for _ in range(self.max_iterations):

            labels = []

            for point in X:

                distances = np.linalg.norm(
                    point - self.centroids,
                    axis=1
                )

                labels.append(np.argmin(distances))

            labels = np.array(labels)

            new_centroids = []

            for cluster in range(self.k):

                cluster_points = X[labels == cluster]

                if len(cluster_points) == 0:
                    new_centroids.append(self.centroids[cluster])
                else:
                    new_centroids.append(
                        cluster_points.mean(axis=0)
                    )

            new_centroids = np.array(new_centroids)

            if np.allclose(
                self.centroids,
                new_centroids
            ):
                break

            self.centroids = new_centroids

        self.labels = labels

    def predict(self, X):

        predictions = []

        for point in X:

            distances = np.linalg.norm(
                point - self.centroids,
                axis=1
            )

            predictions.append(np.argmin(distances))

        return np.array(predictions)
    def inertia(self, X):
        """
        Calculate the total within-cluster sum of squares.
        """

        total = 0

        for i, point in enumerate(X):

            centroid = self.centroids[self.labels[i]]

            total += np.sum((point - centroid) ** 2)

        return total


    def cluster_counts(self):
        """
        Return the number of points in each cluster.
        """

        unique, counts = np.unique(self.labels, return_counts=True)

        return dict(zip(unique, counts))