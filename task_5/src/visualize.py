import os
import matplotlib.pyplot as plt
import pandas as pd
def plot_average_score_by_domain(df, output_dir):

    avg_scores = df.groupby("domain")["score"].mean()

    plt.figure(figsize=(8, 5))
    avg_scores.plot(kind="bar")

    plt.title("Average Score by Domain")
    plt.xlabel("Domain")
    plt.ylabel("Average Score")

    plt.tight_layout()

    output_path = os.path.join(output_dir, "average_score_by_domain.png")
    plt.savefig(output_path)
    plt.close()

def plot_attendance_distribution(df, output_dir):
    plt.figure(figsize=(8, 5))

    plt.hist(
        df["attendance"],
        bins=5,
        edgecolor="black"
    )

    plt.title("Attendance Distribution")
    plt.xlabel("Attendance (%)")
    plt.ylabel("Number of Students")

    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        "attendance_distribution.png"
    )

    plt.savefig(output_path)
    plt.close()
def plot_score_vs_attendance(df, output_dir):
    plt.figure(figsize=(8, 5))

    plt.scatter(
        df["attendance"],
        df["score"]
    )

    plt.title("Score vs Attendance")
    plt.xlabel("Attendance (%)")
    plt.ylabel("Score")

    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        "score_vs_attendance.png"
    )

    plt.savefig(output_path)
    plt.close()