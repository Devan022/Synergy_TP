import sys

from train_linear_regression import train_linear_regression
from train_logistic_regression import train_logistic_regression
from train_kmeans import train_kmeans


def main():

    if len(sys.argv) < 3:
        print("Usage:")
        print("python main.py <csv_path> <output_folder>")
        return

    csv_path = sys.argv[1]

    print("=" * 60)
    print("TASK 10 - MACHINE LEARNING PIPELINE")
    print("=" * 60)

    print("\nRunning Linear Regression...")
    train_linear_regression(csv_path)

    print("\nRunning Logistic Regression...")
    train_logistic_regression(csv_path)

    print("\nRunning KMeans Clustering...")
    train_kmeans(csv_path)

    print("\n" + "=" * 60)
    print("TASK COMPLETED SUCCESSFULLY!")
    print("All metrics, predictions, and plots have been generated.")
    print("=" * 60)


if __name__ == "__main__":
    main()