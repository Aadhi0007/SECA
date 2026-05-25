"""
SECA - Self Evolving Cognitive Architecture
Main entry point for running the SECA framework.

This script:
1. Loads dataset
2. Initializes SECA engine
3. Runs evolutionary architecture search
4. Logs results
"""

import os
import json
import time
import pandas as pd

from seca_core.seca_engine import SECAEngine
from data.dataset_loader import load_dataset
import tensorflow as tf

def set_memory_growth():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print("GPU memory growth enabled.")
        except RuntimeError as e:
            print(e)

set_memory_growth()


def create_logs():
    """Create required directories."""
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models/saved_models", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)


def save_results(history):
    """Save evolution results to log files."""

    # JSON log
    json_file = "logs/evolution_log.json"

    with open(json_file, "w") as f:
        json.dump({"results": history}, f, indent=4)

    print(f"\nEvolution log saved to {json_file}")

    # CSV log
    csv_file = "logs/training_log.csv"

    df = pd.DataFrame(history)
    df.to_csv(csv_file, index=False)

    print(f"Training log saved to {csv_file}")


def main():

    print("\n===================================")
    print("   SECA - Self Evolving Architecture")
    print("===================================\n")

    start_time = time.time()

    # Create directories
    create_logs()

    # Load dataset
    print("Loading dataset...")
    x_train, x_test, y_train, y_test = load_dataset()

    # Subset the data for a fast demonstration run
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    x_test = x_test[:200]
    y_test = y_test[:200]
    print("Using a subset of 1000 training samples for fast execution.")

    import numpy as np
    input_shape = x_train.shape[1:]
    num_classes = len(np.unique(y_train))

    # Initialize SECA engine with reduced parameters for quick demo
    seca = SECAEngine(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        input_shape=input_shape,
        num_classes=num_classes,
        population_size=2,
        generations=2
    )

    print("\nStarting SECA Evolution...\n")

    # Run evolution
    best_genome, best_stats, history = seca.run_evolution()

    print("\n==============================")
    print("     FINAL BEST MODEL")
    print("==============================")

    print("\nBest Genome:")
    print(json.dumps(best_genome, indent=4))

    print("\nBest Statistics:")
    print(best_stats)

    # Save logs
    save_results(history)

    end_time = time.time()

    print("\nTotal Execution Time: {:.2f} seconds".format(end_time - start_time))
    print("\nSECA evolution completed successfully.")


if __name__ == "__main__":
    main()