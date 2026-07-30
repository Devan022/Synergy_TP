import os
import matplotlib.pyplot as plt
import numpy as np


os.makedirs("../output", exist_ok=True)


def plot_loss(loss_history, filename, title):

    plt.figure(figsize=(8,5))

    plt.plot(loss_history)

    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")

    plt.grid(True)

    plt.savefig(f"../output/{filename}")

    plt.close()


def plot_actual_vs_predicted(actual, predicted):

    plt.figure(figsize=(6,6))

    plt.scatter(actual, predicted)

    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    plt.title("Actual vs Predicted")

    plt.grid(True)

    plt.savefig("../output/actual_vs_predicted.png")

    plt.close()


def plot_confusion_matrix(cm):

    plt.figure(figsize=(5,5))

    plt.imshow(cm)

    plt.colorbar()

    plt.xticks([0,1],["0","1"])
    plt.yticks([0,1],["0","1"])

    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.title("Confusion Matrix")

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(cm[i][j]),
                ha="center",
                va="center"
            )

    plt.savefig("../output/confusion_matrix.png")

    plt.close()


def plot_clusters(X, labels):

    plt.figure(figsize=(7,6))

    plt.scatter(
        X[:,0],
        X[:,1],
        c=labels
    )

    plt.xlabel("Temperature")
    plt.ylabel("Humidity")

    plt.title("KMeans Clustering")

    plt.grid(True)

    plt.savefig("../output/clustering_plot.png")

    plt.close()