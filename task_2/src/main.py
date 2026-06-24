import sys
from analyzer import (
    read_submissions,
    get_submitted_students,
    calculate_average_score,
    get_domain_wise_average,
    get_missing_submissions,
    write_summary,
)


def main():
    if len(sys.argv) != 3:
        print("Usage: python task_2/src/main.py <input_csv> <output_json>")
        sys.exit(1)

    input_csv = sys.argv[1]
    output_json = sys.argv[2]

    try:
        submissions = read_submissions(input_csv)

        total_students = len(submissions)
        submitted_students = get_submitted_students(submissions)
        missing_students = get_missing_submissions(submissions)

        submitted_count = len(submitted_students)
        missing_count = len(missing_students)
        average_score = calculate_average_score(submissions)

        highest_scorer = max(submitted_students, key=lambda x: x["score"])["name"] if submitted_students else None
        lowest_scorer = min(submitted_students, key=lambda x: x["score"])["name"] if submitted_students else None

        below_five = [student["name"] for student in submitted_students if student["score"] < 5]

        domain_wise_average = get_domain_wise_average(submissions)

        summary = {
            "total_students": total_students,
            "submitted_count": submitted_count,
            "missing_count": missing_count,
            "average_score": average_score,
            "highest_scorer": highest_scorer,
            "lowest_scorer_among_submitted": lowest_scorer,
            "domain_wise_average": domain_wise_average,
            "missing_submissions": missing_students,
            "students_below_5": below_five,
        }

        print(f"Total students: {total_students}")
        print(f"Submitted count: {submitted_count}")
        print(f"Missing count: {missing_count}")
        print(f"Average score: {average_score}")
        print(f"Highest scorer: {highest_scorer}")
        print(f"Lowest scorer among submitted: {lowest_scorer}")
        print(f"Missing submissions: {missing_students}")
        print(f"Students scoring below 5: {below_five}")

        write_summary(output_json, summary)
        print(f"Summary written to {output_json}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()