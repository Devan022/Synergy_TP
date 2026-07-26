import pandas as pd
import numpy as np
from scipy.stats import t

def load_data(file_path: str):
    """
    Load the calibration dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)

def calculate_confidence_interval(mean: float, std: float, n: int):
    """
    Calculate the 95% confidence interval using the t-distribution.
    """

    if n < 2 or pd.isna(std):
        return (np.nan, np.nan)

    t_value = t.ppf(0.975, df=n - 1)
    margin = t_value * (std / np.sqrt(n))

    lower = mean - margin
    upper = mean + margin

    return (lower, upper)
def assign_stability_flag(coefficient_of_variation: float) -> str:
    """
    Assign a stability label based on coefficient of variation.
    """

    if pd.isna(coefficient_of_variation):
        return "unreliable"

    if coefficient_of_variation <= 0.05:
        return "stable"
    elif coefficient_of_variation <= 0.15:
        return "moderate"
    else:
        return "unstable"
def calculate_replicate_statistics(df):
    """
    Calculate replicate-level statistics for each measurement group.
    """

    grouping_columns = [
        "domain",
        "condition",
        "input_type",
        "input_value",
        "input_unit",
        "signal_unit"
    ]

    grouped = df.groupby(grouping_columns)

    summary_rows = []
    for group_name, group in grouped:

        signal = group["signal"].dropna()

        n = len(signal)

        mean = signal.mean()
        median = signal.median()
        minimum = signal.min()
        maximum = signal.max()

        if n >= 2:
            variance = signal.var(ddof=1)
            std = signal.std(ddof=1)
            standard_error = std / np.sqrt(n)

            ci_lower, ci_upper = calculate_confidence_interval(
                mean,
                std,
                n
                )

            coefficient_of_variation = (
                std / mean
                if mean != 0
                else np.nan
                )

        else:
            variance = np.nan
            std = np.nan
            standard_error = np.nan
            ci_lower = np.nan
            ci_upper = np.nan
            coefficient_of_variation = np.nan
        summary_rows.append({
            "domain": group_name[0],
            "condition": group_name[1],
            "input_type": group_name[2],
            "input_value": group_name[3],
            "input_unit": group_name[4],
            "signal_unit": group_name[5],

            "replicate_count": n,
            "mean_signal": mean,
            "median_signal": median,
            "variance_signal": variance,
            "standard_deviation_signal": std,
            "standard_error_signal": standard_error,
            "confidence_interval_lower": ci_lower,
            "confidence_interval_upper": ci_upper,
            "coefficient_of_variation": coefficient_of_variation,
            "minimum_signal": minimum,
            "maximum_signal": maximum,
            "stability_flag": assign_stability_flag(
                coefficient_of_variation
            )
        })

    return pd.DataFrame(summary_rows)
def save_replicate_summary(summary_df, output_path: str) -> None:
    """
    Save the replicate summary DataFrame to a CSV file.

    Args:
        summary_df: DataFrame containing replicate statistics.
        output_path: Path where the CSV should be saved.
    """
    summary_df.to_csv(output_path, index=False)

if __name__ == "__main__":

    df = load_data("task_9/data/calibration_measurements.csv")
    
    summary = calculate_replicate_statistics(df)

    save_replicate_summary(
        summary,
        "task_9/output/replicate_summary.csv"
    )

    print("\nReplicate summary saved successfully!")