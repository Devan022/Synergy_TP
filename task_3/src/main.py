import os
import sys

from manual_parser import (
    read_csv_manual,
    convert_types,
    calculate_summary,
    write_json as write_manual_json,
)

from pandas_parser import (
    read_csv_pandas,
    calculate_summary_pandas,
    write_json as write_pandas_json,
)


def write_comparison_report(manual_summary, pandas_summary, output_path):
    output_dir = os.path.dirname(output_path)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    match = manual_summary == pandas_summary

    with open(output_path, "w", encoding="utf-8") as file:

        file.write("# Manual Parser vs Pandas Comparison\n\n")

        if match:
            file.write("Both summaries match exactly.\n\n")
        else:
            file.write("The summaries are different.\n\n")

        file.write("## Manual Summary\n\n")
        file.write(str(manual_summary))
        file.write("\n\n")

        file.write("## Pandas Summary\n\n")
        file.write(str(pandas_summary))


def main():

    if len(sys.argv) != 2:
        print("Usage:")
        print("python task_3/src/main.py task_3/data/submissions.csv")
        sys.exit(1)

    input_csv = sys.argv[1]
    rows = read_csv_manual(input_csv)
    rows = convert_types(rows)
    manual_summary = calculate_summary(rows)

    write_manual_json(
        manual_summary,
        "task_3/output/manual_summary.json"
    )
    df = read_csv_pandas(input_csv)
    pandas_summary = calculate_summary_pandas(df)

    write_pandas_json(
        pandas_summary,
        "task_3/output/pandas_summary.json"
    )
    write_comparison_report(
        manual_summary,
        pandas_summary,
        "task_3/output/comparison_report.md"
    )

    print("Task 3 completed successfully.")
    print("Generated:")
    print("- task_3/output/manual_summary.json")
    print("- task_3/output/pandas_summary.json")
    print("- task_3/output/comparison_report.md")


if __name__ == "__main__":
    main()