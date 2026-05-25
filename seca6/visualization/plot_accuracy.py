"""
Plot Accuracy Evolution for SECA

This script loads the SECA evolution log and plots
accuracy improvement across generations.
"""

import matplotlib.pyplot as plt
from visualization.dashboard_utils import (
    load_results,
    extract_generations,
    extract_accuracy
)


def plot_accuracy():

    # Load evolution results
    results = load_results()

    if not results:
        print("No evolution log found.")
        return

    generations = extract_generations(results)
    accuracy = extract_accuracy(results)

    plt.figure()

    plt.plot(
        generations,
        accuracy,
        marker="o",
        linewidth=2
    )

    plt.title("SECA Evolution: Accuracy vs Generation")
    plt.xlabel("Generation")
    plt.ylabel("Validation Accuracy")

    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    plot_accuracy()