"""
Run SECA Evolution Experiment

This script runs the evolutionary neural architecture search
using the SECA framework.
"""

from seca_core.seca_engine import SECAEngine
from data.dataset_loader import load_dataset

from experiments.results_logger import (
    save_evolution_log,
    save_training_csv
)


POPULATION_SIZE = 6
GENERATIONS = 5


def main():

    print("Loading dataset...")

    x_train, x_test, y_train, y_test = load_dataset()

    print("Initializing SECA engine...")

    seca = SECAEngine(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        population_size=POPULATION_SIZE,
        generations=GENERATIONS
    )

    print("Running evolution...")

    best_genome, best_stats, history = seca.run_evolution()

    print("\n=== FINAL BEST ARCHITECTURE ===")

    print("Genome:", best_genome)
    print("Accuracy:", best_stats["accuracy"])
    print("Parameters:", best_stats["params"])
    print("Fitness:", best_stats["fitness"])

    print("\nSaving logs...")

    save_evolution_log(history)
    save_training_csv(history)

    print("\nEvolution experiment completed.")


if __name__ == "__main__":
    main()