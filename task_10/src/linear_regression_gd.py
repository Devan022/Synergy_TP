import numpy as np


class LinearRegressionGD:

    def __init__(self, learning_rate=0.001, iterations=3000):

        self.learning_rate = learning_rate
        self.iterations = iterations

        self.weights = None
        self.bias = 0

        self.loss_history = []

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):

            y_pred = np.dot(X, self.weights) + self.bias

            error = y_pred - y

            dw = (2 / n_samples) * np.dot(X.T, error)
            db = (2 / n_samples) * np.sum(error)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            loss = np.mean(error ** 2)
            self.loss_history.append(loss)

    def predict(self, X):

        return np.dot(X, self.weights) + self.bias