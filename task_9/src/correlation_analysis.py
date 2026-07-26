import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

def calculate_correlations(df):
    """
    Calculate Pearson and Spearman correlations
    for the required domain relationships.
    """

    correlation_results = []

    relationships = [
        ("Biochem", "input_value", "signal"),
        ("Electronics", "input_value", "signal"),
        ("Electronics", "temperature_c", "signal"),
        ("Mechanical", "input_value", "signal"),
        ("Mechanical", "input_value", "stress_mpa")
    ]

    for domain, x_col, y_col in relationships:

        subset = df[df["domain"] == domain][[x_col, y_col]].dropna()

        if len(subset) < 2:
            continue

        pearson_corr, _ = pearsonr(
            subset[x_col],
            subset[y_col]
        )

        spearman_corr, _ = spearmanr(
            subset[x_col],
            subset[y_col]
        )

        correlation_results.append({
            "domain": domain,
            "x_variable": x_col,
            "y_variable": y_col,
            "pearson_correlation": pearson_corr,
            "spearman_correlation": spearman_corr,
            "sample_count": len(subset)
        })

    return pd.DataFrame(correlation_results)

def fit_calibration_line(df):
    """
    Fit a simple linear calibration line
    using NumPy polyfit.
    """

    calibration_results = []

    relationships = [
        ("Biochem", "input_value", "signal"),
        ("Electronics", "input_value", "signal"),
        ("Electronics", "temperature_c", "signal"),
        ("Mechanical", "input_value", "signal"),
        ("Mechanical", "input_value", "stress_mpa")
    ]

    for domain, x_col, y_col in relationships:

        subset = df[df["domain"] == domain][[x_col, y_col]].dropna()

        if len(subset) < 2:
            continue

        slope, intercept = np.polyfit(
            subset[x_col],
            subset[y_col],
            1
        )

        calibration_results.append({
            "domain": domain,
            "x_variable": x_col,
            "y_variable": y_col,
            "slope": slope,
            "intercept": intercept
        })

    return pd.DataFrame(calibration_results)
def calculate_fit_metrics(df):
    """
    Calculate R², MAE, and RMSE
    for each calibration relationship.
    """

    metric_results = []

    relationships = [
        ("Biochem", "input_value", "signal"),
        ("Electronics", "input_value", "signal"),
        ("Electronics", "temperature_c", "signal"),
        ("Mechanical", "input_value", "signal"),
        ("Mechanical", "input_value", "stress_mpa")
    ]

    for domain, x_col, y_col in relationships:

        subset = df[df["domain"] == domain][[x_col, y_col]].dropna()

        if len(subset) < 2:
            continue

        x = subset[x_col].values
        y = subset[y_col].values

        slope, intercept = np.polyfit(x, y, 1)

        predictions = slope * x + intercept

        residuals = y - predictions

        mae = np.mean(np.abs(residuals))

        rmse = np.sqrt(np.mean(residuals ** 2))

        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        if ss_tot == 0:
            r2 = np.nan
        else:
            r2 = 1 - (ss_res / ss_tot)

        metric_results.append({
            "domain": domain,
            "x_variable": x_col,
            "y_variable": y_col,
            "R_squared": r2,
            "MAE": mae,
            "RMSE": rmse
        })

    return pd.DataFrame(metric_results)

def plot_signal_input_scatter(df, output_path: str) -> None:
    """
    Plot raw signal versus input value for all domains.
    """

    plt.figure(figsize=(8, 6))

    domains = df["domain"].unique()

    for domain in domains:
        subset = df[df["domain"] == domain]

        plt.scatter(
            subset["input_value"],
            subset["signal"],
            label=domain
        )

    plt.xlabel("Input Value")
    plt.ylabel("Signal")
    plt.title("Signal vs Input Value")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
def plot_calibration_curve(df, domain, output_path):
    """
    Plot calibration curve (signal vs input value)
    with fitted regression line for a given domain.
    """

    subset = df[df["domain"] == domain][["input_value", "signal"]].dropna()

    if len(subset) < 2:
        return

    x = subset["input_value"].values
    y = subset["signal"].values

    slope, intercept = np.polyfit(x, y, 1)

    y_pred = slope * x + intercept

    plt.figure(figsize=(8, 6))

    plt.scatter(x, y, label="Observed Data")
    plt.plot(x, y_pred, label="Calibration Line")

    plt.xlabel("Input Value")
    plt.ylabel("Signal")
    plt.title(f"{domain} Calibration Curve")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_correlation_summary(df, output_path):
    """
    Save the correlation analysis results to a CSV file.
    """
    df.to_csv(output_path, index=False)


def save_calibration_summary(df, output_path):
    """
    Save the calibration analysis results to a CSV file.
    """
    df.to_csv(output_path, index=False)

if __name__ == "__main__":

    df = pd.read_csv(
        "task_9/data/calibration_measurements.csv"
    )

    # Generate correlation summary
    correlation_df = calculate_correlations(df)
    save_correlation_summary(
        correlation_df,
        "task_9/output/correlation_summary.csv"
    )

    # Generate calibration summary
    calibration_df = fit_calibration_line(df)

    metrics_df = calculate_fit_metrics(df)

    calibration_df = calibration_df.merge(
        metrics_df,
        on=["domain", "x_variable", "y_variable"]
    )

    save_calibration_summary(
        calibration_df,
        "task_9/output/calibration_summary.csv"
    )

    # Generate plots
    plot_signal_input_scatter(
        df,
        "task_9/output/correlation_signal_input.png"
    )

    for domain in df["domain"].unique():
        plot_calibration_curve(
            df,
            domain,
            f"task_9/output/{domain.lower()}_calibration_curve.png"
        )

    print("Correlation analysis completed successfully!")