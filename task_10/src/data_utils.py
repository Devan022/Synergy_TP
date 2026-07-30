import pandas as pd
import numpy as np


def load_dataset(file_path):
    """
    Load and perform initial cleaning of AirQualityUCI dataset.
    """

    df = pd.read_csv(file_path, sep=';')

    # Remove completely empty columns
    df = df.drop(columns=["Unnamed: 15", "Unnamed: 16"])

    # Remove completely empty rows
    df = df.dropna(how="all")

    return df
def clean_dataset(df):
    """
    Convert string columns to numeric and handle missing values.
    """

    string_columns = [
        "CO(GT)",
        "C6H6(GT)",
        "T",
        "RH",
        "AH"
    ]

    for column in string_columns:
        df[column] = (
            df[column]
            .astype(str)
            .str.replace(",", ".", regex=False)
        )

        df[column] = pd.to_numeric(df[column], errors="coerce")

    # Replace -200 (missing sensor values) with NaN
    df = df.replace(-200, np.nan)

    return df

def dataset_info(df):

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst Five Rows:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nData Types:")
    print(df.dtypes)

def train_val_test_split(X, y, train_size=0.7, val_size=0.15, random_state=42):
    """
    Split data into train, validation, and test sets.
    """

    np.random.seed(random_state)

    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X = X[indices]
    y = y[indices]

    train_end = int(len(X) * train_size)
    val_end = train_end + int(len(X) * val_size)

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val = X[train_end:val_end]
    y_val = y[train_end:val_end]

    X_test = X[val_end:]
    y_test = y[val_end:]

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test
    )

def normalize_features(X_train, X_val, X_test):
    """
    Normalize features using training set statistics.
    """

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    std[std == 0] = 1

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_val, X_test