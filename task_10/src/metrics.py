import numpy as np


# ----------------------------
# Regression Metrics
# ----------------------------

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))


def r2_score(y_true, y_pred):

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - (ss_res / ss_tot)


# ----------------------------
# Classification Metrics
# ----------------------------

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def confusion_matrix(y_true, y_pred):

    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return np.array([[tn, fp],
                     [fn, tp]])


def precision(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    tp = cm[1, 1]
    fp = cm[0, 1]

    if tp + fp == 0:
        return 0

    return tp / (tp + fp)


def recall(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    tp = cm[1, 1]
    fn = cm[1, 0]

    if tp + fn == 0:
        return 0

    return tp / (tp + fn)


def f1_score(y_true, y_pred):

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)

    if p + r == 0:
        return 0

    return 2 * p * r / (p + r)



def silhouette_score(X, labels):

    n = len(X)
    unique_clusters = np.unique(labels)

    scores = []

    for i in range(n):

        same_cluster = X[labels == labels[i]]

        if len(same_cluster) <= 1:
            scores.append(0)
            continue

        # a(i)
        a = np.mean(
            np.linalg.norm(
                same_cluster - X[i],
                axis=1
            )
        )

        # b(i)
        b = float("inf")

        for cluster in unique_clusters:

            if cluster == labels[i]:
                continue

            other_cluster = X[labels == cluster]

            distance = np.mean(
                np.linalg.norm(
                    other_cluster - X[i],
                    axis=1
                )
            )

            b = min(b, distance)

        score = (b - a) / max(a, b)

        scores.append(score)

    return np.mean(scores)