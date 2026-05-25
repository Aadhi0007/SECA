"""
Plot Evolution Results for SECA

This script reads evolution logs and generates graphs
showing the progress of neural architecture evolution.
"""

import json
import matplotlib.pyplot as plt


LOG_FILE = "logs/evolution_log.json"


def load_log(filepath):
    """Load evolution log file."""

    with open(filepath, "r") as f:
        data = json.load(f)

    return data["results"]


def plot_accuracy(results):
    """Plot accuracy vs generation."""

    generations = list(range(len(results)))
    accuracy = [r["accuracy"] for r in results]

    plt.figure()
    plt.plot(generations, accuracy, marker="o")
    plt.title("SECA Evolution: Accuracy vs Generation")
    plt.xlabel("Generation")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.show()


def plot_fitness(results):
    """Plot fitness vs generation."""

    generations = list(range(len(results)))
    fitness = [r["fitness"] for r in results]

    plt.figure()
    plt.plot(generations, fitness, marker="o")
    plt.title("SECA Evolution: Fitness vs Generation")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.grid(True)
    plt.show()


def plot_parameters(results):
    """Plot model parameters vs generation."""

    generations = list(range(len(results)))
    params = [r["params"] for r in results]

    plt.figure()
    plt.plot(generations, params, marker="o")
    plt.title("SECA Evolution: Model Size vs Generation")
    plt.xlabel("Generation")
    plt.ylabel("Parameters")
    plt.grid(True)
    plt.show()


def main():

    results = load_log(LOG_FILE)

    plot_accuracy(results)
    plot_fitness(results)
    plot_parameters(results)


if __name__ == "__main__":
    main()