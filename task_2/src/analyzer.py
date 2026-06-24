import csv
import json
import os
from typing import List, Dict, Any


def read_submissions(file_path: str) -> List[Dict[str, Any]]:
  
    submissions = []

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Input file not found: {file_path}")

    with open(file_path, mode="r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames is None:
            raise ValueError("CSV file is empty or invalid.")

        for row in reader:
            try:
                row["score"] = int(row["score"])
            except ValueError:
                raise ValueError(f"Invalid score value for student {row.get('name', 'Unknown')}")

            row["submitted"] = row["submitted"].strip().lower()
            submissions.append(row)

    if not submissions:
        raise ValueError("CSV file contains no student records.")

    return submissions


def get_submitted_students(submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
   
    return [student for student in submissions if student["submitted"] == "yes"]


def calculate_average_score(submissions: List[Dict[str, Any]]) -> float:
    
    submitted_students = get_submitted_students(submissions)

    if not submitted_students:
        return 0.0

    total_score = sum(student["score"] for student in submitted_students)
    return round(total_score / len(submitted_students), 2)


def get_domain_wise_average(submissions: List[Dict[str, Any]]) -> Dict[str, float]:
    
    domain_scores: Dict[str, List[int]] = {}

    for student in submissions:
        if student["submitted"] == "yes":
            domain = student["domain"]
            if domain not in domain_scores:
                domain_scores[domain] = []
            domain_scores[domain].append(student["score"])

    domain_average = {}
    for domain, scores in domain_scores.items():
        domain_average[domain] = round(sum(scores) / len(scores), 2)

    return domain_average


def get_missing_submissions(submissions: List[Dict[str, Any]]) -> List[str]:
    return [student["name"] for student in submissions if student["submitted"] == "no"]


def write_summary(output_path: str, summary_data: Dict[str, Any]) -> None:
   
    output_dir = os.path.dirname(output_path)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, mode="w", encoding="utf-8") as file:
        json.dump(summary_data, file, indent=4)