import numpy as np
import pandas as pd

class LogisticRegressionScratch:
    """
    Logistic Regression implemented from scratch
    using Gradient Descent.
    """

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = 0
    def sigmoid(self, z):
        """
        Compute the sigmoid activation.
        """
        return 1 / (1 + np.exp(-z))
    def fit(self, X, y):
        """
        Train the Logistic Regression model
        using Gradient Descent.
        """

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):

            linear_model = np.dot(X, self.weights) + self.bias

            predictions = self.sigmoid(linear_model)

            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict_probability(self, X):
        """
        Return the probability of belonging
        to the positive class.
        """

        linear_model = np.dot(X, self.weights) + self.bias

        return self.sigmoid(linear_model)


    def predict(self, X):
        """
        Predict binary class labels (0 or 1).
        """

        probabilities = self.predict_probability(X)

        return np.where(probabilities >= 0.5, 1, 0)
    def accuracy(self, y_true, y_pred):
        """
        Calculate classification accuracy.
        """

        return np.mean(y_true == y_pred)
if __name__ == "__main__":

    X = np.array([
        [1],
        [2],
        [3],
        [4]
    ])

    y = np.array([0, 0, 1, 1])

    model = LogisticRegressionScratch(
        learning_rate=0.1,
        iterations=1000
    )

    model.fit(X, y)

    predictions = model.predict(X)

    print("Predictions:", predictions)
    print("Accuracy:", model.accuracy(y, predictions))