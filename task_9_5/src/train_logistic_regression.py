import pandas as pd
import numpy as np

from logistic_regression import LogisticRegressionScratch

# Load dataset
df = pd.read_csv(
    "task_9_5/data/heart_disease_risk_2026.csv"
)

# Select numeric features
feature_columns = [
    "age",
    "resting_bp_systolic",
    "resting_bp_diastolic",
    "cholesterol_total",
    "hdl",
    "ldl",
    "triglycerides",
    "fasting_blood_sugar",
    "hba1c",
    "bmi",
    "resting_heart_rate",
    "max_heart_rate_achieved",
    "st_depression",
    "alcohol_units_per_week",
    "exercise_minutes_per_week",
    "sleep_hours",
    "stress_score",
    "daily_steps",
    "diet_quality_score"
]

X = df[feature_columns].values.astype(float)

# Normalize features
X = (X - X.mean(axis=0)) / X.std(axis=0)

# Target
y = df["has_heart_disease"].values.astype(int)

# Train model
model = LogisticRegressionScratch(
    learning_rate=0.01,
    iterations=5000
)

model.fit(X, y)

# Predict
predictions = model.predict(X)

# Accuracy
accuracy = model.accuracy(y, predictions)

print("Accuracy:", accuracy)

# Save predictions
results = pd.DataFrame({
    "Actual": y,
    "Predicted": predictions
})

results.to_csv(
    "task_9_5/output/logistic_regression_predictions.csv",
    index=False
)

print("Predictions saved successfully!")