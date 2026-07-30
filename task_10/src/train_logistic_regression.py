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

from logistic_regression_gd import LogisticRegressionGD

from metrics import (
    accuracy,
    precision,
    recall,
    f1_score,
    confusion_matrix
)
from plots import (
    plot_loss,
    plot_confusion_matrix
)

def train_logistic_regression(csv_path):

    # Load dataset
    df = load_dataset(csv_path)
    df = clean_dataset(df)

    # Features
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

    data = df[features + [target]].dropna()

    # Binary classification label
    median = data[target].median()
    data["Pollution"] = (data[target] > median).astype(int)

    X = data[features].values
    y = data["Pollution"].values

    # Split
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
    model = LogisticRegressionGD(
        learning_rate=0.01,
        iterations=3000
    )

    model.fit(X_train, y_train)

    # Predict
    predictions = model.predict(X_test)

    # Metrics
    acc = accuracy(y_test, predictions)
    prec = precision(y_test, predictions)
    rec = recall(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    print("\nLogistic Regression Results")
    print("---------------------------")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1 Score :", f1)
    print("\nConfusion Matrix")
    print(cm)

    # Create output folder
    os.makedirs("../output", exist_ok=True)

    # Save metrics
    metrics = {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1 Score": float(f1),
        "Confusion Matrix": cm.tolist()
    }

    with open("../output/classification_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    # Save predictions
    results = pd.DataFrame({
        "Actual": y_test,
        "Predicted": predictions
    })

    results.to_csv(
        "../output/classification_predictions.csv",
        index=False
    )

    print("\nClassification results saved successfully!")
    # Generate plots
    plot_loss(
        model.loss_history,
        "classification_loss_curve.png",
        "Logistic Regression Loss"
    )

    plot_confusion_matrix(cm)

    print("Classification plots saved successfully!")
    return model


if __name__ == "__main__":
    train_logistic_regression("../data/AirQualityUCI.csv")