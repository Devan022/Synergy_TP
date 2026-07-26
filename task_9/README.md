# Task 9 - Calibration Statistics, Correlation Analysis and Feature Engineering

## Objective

This project performs calibration statistics, correlation analysis, visualization, and feature engineering on calibration measurement data to prepare an ML-ready dataset.

---

## Folder Structure

```
task_9/
├── README.md
├── data/
├── output/
└── src/
```

---

## Source Files

- replicate_statistics.py
- correlation_analysis.py
- feature_engineering.py
- main.py

---

## Outputs

### CSV Files

- replicate_summary.csv
- correlation_summary.csv
- calibration_summary.csv
- engineered_features.csv
- ml_ready_dataset.csv

### Plots

- correlation_signal_input.png
- biochem_calibration_curve.png
- electronics_calibration_curve.png
- mechanical_calibration_curve.png

---

## Libraries Used

- pandas
- numpy
- matplotlib
- scipy

---

## Execution

Run the complete pipeline:

```bash
python3 task_9/src/main.py
```

---

## Result

The project successfully:

- computed replicate statistics
- performed correlation analysis
- generated calibration metrics
- created visualization plots
- engineered new features
- prepared an ML-ready dataset