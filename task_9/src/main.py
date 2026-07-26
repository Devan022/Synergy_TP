import pandas as pd

from replicate_statistics import (
    load_data,
    calculate_replicate_statistics,
    save_replicate_summary
)

from correlation_analysis import (
    calculate_correlations,
    fit_calibration_line,
    calculate_fit_metrics,
    save_correlation_summary,
    save_calibration_summary,
    plot_signal_input_scatter,
    plot_calibration_curve
)

from feature_engineering import (
    create_engineered_features,
    create_ml_ready_dataset,
    save_engineered_features,
    save_ml_ready_dataset
)

def main():

    data_path = "task_9/data/calibration_measurements.csv"

    df = load_data(data_path)

    # Replicate statistics
    replicate_df = calculate_replicate_statistics(df)

    save_replicate_summary(
        replicate_df,
        "task_9/output/replicate_summary.csv"
    )

    # Correlation summary
    correlation_df = calculate_correlations(df)

    save_correlation_summary(
        correlation_df,
        "task_9/output/correlation_summary.csv"
    )

    # Calibration summary
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

    # Plots
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

    # Feature engineering
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

    print("Task 9 completed successfully!")

if __name__ == "__main__":
    main()