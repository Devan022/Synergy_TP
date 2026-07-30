import numpy as np


class LogisticRegressionGD:

    def __init__(self, learning_rate=0.01, iterations=3000):

        self.learning_rate = learning_rate
        self.iterations = iterations

        self.weights = None
        self.bias = 0

        self.loss_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):

            linear = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            epsilon = 1e-15
            predictions = np.clip(predictions, epsilon, 1 - epsilon)

            loss = -np.mean(
                y * np.log(predictions)
                + (1 - y) * np.log(1 - predictions)
            )

            self.loss_history.append(loss)

    def predict_probability(self, X):

        linear = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear)

    def predict(self, X):

        probabilities = self.predict_probability(X)
        return (probabilities >= 0.5).astype(int)