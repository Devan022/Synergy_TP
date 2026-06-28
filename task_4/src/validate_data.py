import pandas as pd


def validate_data(df: pd.DataFrame) -> dict:
    report = {}

    report["duplicate_student_ids"] = int(df["student_id"].duplicated().sum())

    report["missing_names"] = int(df["name"].isna().sum())

    report["missing_emails"] = int(df["email"].isna().sum())

    report["invalid_scores"] = int(
        ((df["score"] < 0) | (df["score"] > 100)).sum()
    )

    report["invalid_attendance"] = int(
        ((df["attendance"] < 0) | (df["attendance"] > 100)).sum()
    )

    report["validation_passed"] = (
        report["duplicate_student_ids"] == 0
        and report["missing_names"] == 0
        and report["missing_emails"] == 0
        and report["invalid_scores"] == 0
        and report["invalid_attendance"] == 0
    )

    return report