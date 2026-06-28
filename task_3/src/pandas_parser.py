import json
import os
import pandas as pd
def read_csv_pandas(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)
def calculate_summary_pandas(df) -> dict:
    df["score"] = pd.to_numeric(df["score"], errors="coerce").fillna(0).astype(int)
    df["submitted"] = df["submitted"].str.strip().str.lower()

    submitted_df = df[df["submitted"] == "yes"]
    missing_df = df[df["submitted"] == "no"]

    summary = {
        "total_students": len(df),
        "submitted_count": len(submitted_df),
        "missing_count": len(missing_df),
        "average_score": round(submitted_df["score"].mean(), 2),
        "highest_scorer": submitted_df.loc[
            submitted_df["score"].idxmax(), "name"
        ] if not submitted_df.empty else None,
        "lowest_scorer_among_submitted": submitted_df.loc[
            submitted_df["score"].idxmin(), "name"
        ] if not submitted_df.empty else None,
        "domain_wise_average": submitted_df.groupby("domain")["score"].mean().round(2).to_dict(),
        "missing_submissions": missing_df["name"].tolist(),
        "students_below_5": submitted_df[submitted_df["score"] < 5]["name"].tolist()
    }

    return summary
def write_json(data: dict, output_path: str) -> None:
    output_dir = os.path.dirname(output_path)

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)