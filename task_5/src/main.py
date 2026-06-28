import os
import sys
import pandas as pd

from visualize import (
    plot_average_score_by_domain,
    plot_attendance_distribution,
    plot_score_vs_attendance,
)


def main():

    if len(sys.argv) != 2:
        print("Usage:")
        print("python task_5/src/main.py task_4/output/cleaned_students.csv")
        sys.exit(1)

    input_csv = sys.argv[1]

    if not os.path.exists(input_csv):
        print(f"Error: {input_csv} not found.")
        sys.exit(1)

    df = pd.read_csv(input_csv)

    output_dir = "task_5/output"
    os.makedirs(output_dir, exist_ok=True)

    plot_average_score_by_domain(df, output_dir)
    plot_attendance_distribution(df, output_dir)
    plot_score_vs_attendance(df, output_dir)

    print("Task 5 completed successfully.")
    print("Generated:")
    print("- task_5/output/average_score_by_domain.png")
    print("- task_5/output/attendance_distribution.png")
    print("- task_5/output/score_vs_attendance.png")


if __name__ == "__main__":
    main()