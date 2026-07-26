import pandas as pd
import numpy as np

def create_engineered_features(df):

    """
    Create additional engineered features for the ML-ready dataset.
    """

    engineered_df = df.copy()

    # Square of input value
    engineered_df["input_squared"] = (
        engineered_df["input_value"] ** 2
    )

    # Cube of input value
    engineered_df["input_cubed"] = (
        engineered_df["input_value"] ** 3
    )

    # Signal-to-input ratio
    engineered_df["signal_input_ratio"] = np.where(
        engineered_df["input_value"] != 0,
        engineered_df["signal"] / engineered_df["input_value"],
        np.nan
    )

    # Interaction between input and temperature
    engineered_df["input_temperature_interaction"] = (
        engineered_df["input_value"]
        * engineered_df["temperature_c"]
    )

    return engineered_df
def create_ml_ready_dataset(df):
    """
    Prepare the dataset for machine learning by
    encoding categorical variables and handling
    missing values.
    """

    ml_df = df.copy()

    # One-hot encode the domain column
    ml_df = pd.get_dummies(
        ml_df,
        columns=["domain"],
        drop_first=False
    )

    # Fill missing numeric values with the column mean
    numeric_cols = ml_df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        ml_df[col] = ml_df[col].fillna(
            ml_df[col].mean()
        )

    return ml_df
def save_engineered_features(df, output_path):
    """
    Save the engineered features dataset.
    """
    df.to_csv(output_path, index=False)


def save_ml_ready_dataset(df, output_path):
    """
    Save the ML-ready dataset.
    """
    df.to_csv(output_path, index=False)


if __name__ == "__main__":

    df = pd.read_csv(
        "task_9/data/calibration_measurements.csv"
    )

    engineered_df = create_engineered_features(df)

    save_engineered_features(
        engineered_df,
        "task_9/output/engineered_features.csv"
    )

    ml_ready_df = create_ml_ready_dataset(
        engineered_df
    )

    save_ml_ready_dataset(
        ml_ready_df,
        "task_9/output/ml_ready_dataset.csv"
    )

    print("Feature engineering completed successfully!")