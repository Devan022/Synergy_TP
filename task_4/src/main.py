import os
import sys

from clean_data import load_data, clean_data
from validate_data import validate_data


def save_cleaned_data(df, output_path):
    """
    Saves the cleaned DataFrame as CSV.
    """

    output_dir = os.path.dirname(output_path)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_path, index=False)


def write_report(cleaning_report, validation_report, output_path):
    """
    Writes the cleaning and validation report.
    """

    output_dir = os.path.dirname(output_path)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:

        file.write("# Data Cleaning Report\n\n")

        file.write("## Cleaning Summary\n\n")

        for key, value in cleaning_report.items():
            file.write(f"- {key}: {value}\n")

        file.write("\n")

        file.write("## Validation Summary\n\n")

        for key, value in validation_report.items():
            file.write(f"- {key}: {value}\n")


def main():

    if len(sys.argv) != 2:
        print("Usage:")
        print("python task_4/src/main.py task_4/data/messy_students.csv")
        sys.exit(1)

    input_csv = sys.argv[1]

    df = load_data(input_csv)

    cleaned_df, cleaning_report = clean_data(df)

    validation_report = validate_data(cleaned_df)

    save_cleaned_data(
        cleaned_df,
        "task_4/output/cleaned_students.csv"
    )

    write_report(
        cleaning_report,
        validation_report,
        "task_4/output/cleaning_report.md"
    )

    print("Task 4 completed successfully.")
    print("Generated:")
    print("- task_4/output/cleaned_students.csv")
    print("- task_4/output/cleaning_report.md")


if __name__ == "__main__":
    main()