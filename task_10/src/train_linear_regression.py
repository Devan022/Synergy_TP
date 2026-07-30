import numpy as np
import json
import os
import pandas as pd

from data_utils import (
    load_dataset,
    clean_dataset,
    train_val_test_split,
    normalize_features
)
from plots import (
    plot_loss,
    plot_actual_vs_predicted
)
from linear_regression_gd import LinearRegressionGD
from metrics import mae, mse, rmse, r2_score


def train_linear_regression(csv_path):

    # Load and clean dataset
    df = load_dataset(csv_path)
    df = clean_dataset(df)

    # Features and target
    features = [
        "PT08.S1(CO)",
        "PT08.S2(NMHC)",
        "PT08.S3(NOx)",
        "PT08.S4(NO2)",
        "PT08.S5(O3)",
        "T",
        "RH",
        "AH"
    ]

    target = "CO(GT)"

    # Remove missing values
    data = df[features + [target]].dropna()

    X = data[features].values
    y = data[target].values

    # Split dataset
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X,
        y
    )

    # Normalize
    X_train, X_val, X_test = normalize_features(
        X_train,
        X_val,
        X_test
    )

    # Train model
    model = LinearRegressionGD(
        learning_rate=0.001,
        iterations=3000
    )

    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Calculate metrics
    mae_value = mae(y_test, predictions)
    mse_value = mse(y_test, predictions)
    rmse_value = rmse(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Print metrics
    print("\nRegression Results")
    print("-------------------")
    print("MAE :", mae_value)
    print("MSE :", mse_value)
    print("RMSE:", rmse_value)
    print("R²  :", r2)

    # Create output folder
    os.makedirs("../output", exist_ok=True)

    # Save metrics
    metrics = {
        "MAE": float(mae_value),
        "MSE": float(mse_value),
        "RMSE": float(rmse_value),
        "R2": float(r2)
    }

    with open("../output/regression_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Save predictions
    results = pd.DataFrame({
        "Actual": y_test,
        "Predicted": predictions
    })

    results.to_csv(
        "../output/regression_predictions.csv",
        index=False
    )

    print("\nRegression results saved successfully!")
    # Generate plots
    plot_loss(
        model.loss_history,
        "regression_loss_curve.png",
        "Linear Regression Loss"
    )

    plot_actual_vs_predicted(
        y_test,
        predictions
    )

    print("Regression plots saved successfully!")
    return model


if __name__ == "__main__":
    train_linear_regression("../data/AirQualityUCI.csv")