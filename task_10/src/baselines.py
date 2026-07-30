import numpy as np


class RegressionBaseline:
    """
    Always predicts the mean of the training target.
    """

    def __init__(self):
        self.mean_value = None

    def fit(self, y_train):
        self.mean_value = np.mean(y_train)

    def predict(self, X):
        return np.full(len(X), self.mean_value)


class ClassificationBaseline:
    """
    Always predicts the majority class.
    """

    def __init__(self):
        self.majority_class = None

    def fit(self, y_train):

        values, counts = np.unique(y_train, return_counts=True)
        self.majority_class = values[np.argmax(counts)]

    def predict(self, X):
        return np.full(len(X), self.majority_class)