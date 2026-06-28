import json
import os
def read_csv_manual(file_path: str) -> list[dict]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    rows = []

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    if not lines:
        raise ValueError("CSV file is empty.")

    header = lines[0].strip().split(",")

    for line in lines[1:]:

        line = line.strip()

        if line == "":
            continue

        values = line.split(",")

        if len(values) != len(header):
            print(f"Skipping malformed row: {line}")
            continue

        row = dict(zip(header, values))
        rows.append(row)

    return rows
def convert_types(rows: list[dict]) -> list[dict]:
    for row in rows:

        try:
            row["score"] = int(row["score"])
        except ValueError:
            row["score"] = 0

        row["submitted"] = row["submitted"].strip().lower()

        if row["submitted"] not in ["yes", "no"]:
            row["submitted"] = "no"

    return rows
def calculate_summary(rows: list[dict]) -> dict:

    total_students = len(rows)

    submitted_students = [
        row for row in rows
        if row["submitted"] == "yes"
    ]

    missing_students = [
        row["name"] for row in rows
        if row["submitted"] == "no"
    ]

    submitted_count = len(submitted_students)
    missing_count = len(missing_students)

    average_score = 0.0
    if submitted_students:
        average_score = round(
            sum(student["score"] for student in submitted_students)
            / submitted_count,
            2
        )

    highest_scorer = None
    lowest_scorer = None

    if submitted_students:
        highest_scorer = max(
            submitted_students,
            key=lambda x: x["score"]
        )["name"]

        lowest_scorer = min(
            submitted_students,
            key=lambda x: x["score"]
        )["name"]

    domain_scores = {}

    for student in submitted_students:

        domain = student["domain"]

        if domain not in domain_scores:
            domain_scores[domain] = []

        domain_scores[domain].append(student["score"])

    domain_average = {}

    for domain, scores in domain_scores.items():
        domain_average[domain] = round(
            sum(scores) / len(scores),
            2
        )

    below_five = [
        student["name"]
        for student in submitted_students
        if student["score"] < 5
    ]

    return {
        "total_students": total_students,
        "submitted_count": submitted_count,
        "missing_count": missing_count,
        "average_score": average_score,
        "highest_scorer": highest_scorer,
        "lowest_scorer_among_submitted": lowest_scorer,
        "domain_wise_average": domain_average,
        "missing_submissions": missing_students,
        "students_below_5": below_five
    }
def write_json(data: dict, output_path: str) -> None:

    output_dir = os.path.dirname(output_path)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)