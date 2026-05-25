"""
Results Logger for SECA

This module records the results of the evolutionary process
including generation statistics, best architectures, and metrics.
"""

import json
import csv
import os
from datetime import datetime


LOG_DIR = "logs"


def ensure_log_directory():
    """Create logs directory if it does not exist."""

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)


def save_evolution_log(history, filename="evolution_log.json"):
    """
    Save evolution history to JSON file.

    Parameters
    ----------
    history : list
        Evolution history records
    filename : str
    """

    ensure_log_directory()

    filepath = os.path.join(LOG_DIR, filename)

    data = {
        "experiment": "SECA Evolution Run",
        "timestamp": str(datetime.now()),
        "results": history
    }

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Evolution log saved to {filepath}")


def save_training_csv(history, filename="training_log.csv"):
    """
    Save evolution metrics to CSV.

    Parameters
    ----------
    history : list
        Evolution history records
    filename : str
    """

    ensure_log_directory()

    filepath = os.path.join(LOG_DIR, filename)

    with open(filepath, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "generation",
            "accuracy",
            "parameters",
            "fitness"
        ])

        for i, record in enumerate(history):

            writer.writerow([
                i,
                record.get("accuracy"),
                record.get("params"),
                record.get("fitness")
            ])

    print(f"Training log saved to {filepath}")