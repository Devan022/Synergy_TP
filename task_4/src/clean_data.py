import os
import pandas as pd
def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)
def clean_data(df: pd.DataFrame):
    report = {}
    before = len(df)
    df = df.drop_duplicates()
    report["duplicates_removed"] = before - len(df)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["attendance"] = pd.to_numeric(df["attendance"], errors="coerce")
    missing_names = df["name"].isna().sum()
    df["name"] = df["name"].fillna("Unknown")
    report["missing_names_filled"] = int(missing_names)
    missing_scores = df["score"].isna().sum()

    score_mean = df["score"].mean()
    df["score"] = df["score"].fillna(score_mean)

    report["missing_scores_filled"] = int(missing_scores)
    invalid_scores = ((df["score"] < 0) | (df["score"] > 100)).sum()

    df.loc[df["score"] < 0, "score"] = 0
    df.loc[df["score"] > 100, "score"] = 100

    report["invalid_scores_corrected"] = int(invalid_scores)
    missing_attendance = df["attendance"].isna().sum()

    attendance_mean = df["attendance"].mean()
    df["attendance"] = df["attendance"].fillna(attendance_mean)
    invalid_attendance = (
        (df["attendance"] < 0) |
        (df["attendance"] > 100)
    ).sum()

    df.loc[df["attendance"] < 0, "attendance"] = 0
    df.loc[df["attendance"] > 100, "attendance"] = 100

    report["missing_attendance_filled"] = int(missing_attendance)
    report["invalid_attendance_corrected"] = int(invalid_attendance)
    missing_emails = df["email"].isna().sum()

    df["email"] = df["email"].fillna("unknown@example.com")

    report["missing_emails_filled"] = int(missing_emails)
    return df, report
