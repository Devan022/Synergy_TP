import numpy as np
import pandas as pd
class LinearRegressionScratch:
    """
    Linear Regression implemented from scratch
    using Gradient Descent.
    """

    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = 0
    def fit(self, X, y):
        """
        Train the Linear Regression model
        using Gradient Descent.
        """

        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):

            y_predicted = np.dot(X, self.weights) + self.bias

            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights = self.weights - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db
    def predict(self, X):
        """
        Predict target values for the given input.
        """

        return np.dot(X, self.weights) + self.bias

    def mean_squared_error(self, y_true, y_pred):
        """
        Calculate Mean Squared Error (MSE).
        """
        return np.mean((y_true - y_pred) ** 2)


    def root_mean_squared_error(self, y_true, y_pred):
        """
        Calculate Root Mean Squared Error (RMSE).
        """
        mse = self.mean_squared_error(y_true, y_pred)
        return np.sqrt(mse)


    def r2_score(self, y_true, y_pred):
        """
        Calculate the R² (coefficient of determination).
        """
        ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
        ss_residual = np.sum((y_true - y_pred) ** 2)

        if ss_total == 0:
            return 0

        return 1 - (ss_residual / ss_total)


if __name__ == "__main__":

    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])

    model = LinearRegressionScratch(
        learning_rate=0.01,
        iterations=1000
    )

    model.fit(X, y)

    predictions = model.predict(X)

    print("MSE :", model.mean_squared_error(y, predictions))
    print("RMSE:", model.root_mean_squared_error(y, predictions))
    print("R²  :", model.r2_score(y, predictions))