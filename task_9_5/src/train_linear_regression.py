import pandas as pd
import numpy as np

from linear_regression import LinearRegressionScratch

# Load dataset
df = pd.read_csv(
    "task_9_5/data/oil_sales_assignment_dataset.csv"
)

# Select features
X = df[[
    "year",
    "month",
    "volume_sales",
    "average_price"
]].values.astype(float)

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Target
y = df["value_sales"].values.astype(float)

# Train model
model = LinearRegressionScratch(
    learning_rate=0.001,
    iterations=3000
)

model.fit(X, y)

# Predict
predictions = model.predict(X)
results = pd.DataFrame({
    "Actual": y,
    "Predicted": predictions
})

results.to_csv(
    "task_9_5/output/linear_regression_predictions.csv",
    index=False
)

print("Predictions saved successfully!")

# Evaluate
print("MSE :", model.mean_squared_error(y, predictions))
print("RMSE:", model.root_mean_squared_error(y, predictions))
print("R²  :", model.r2_score(y, predictions))