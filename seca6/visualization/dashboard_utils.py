"""
Visualization Dashboard Utilities for SECA

This module loads evolution logs and prepares
data for visualization (graphs, dashboards, analysis).
"""

import json
import os


LOG_PATH = "logs/evolution_log.json"


def load_results(log_path=LOG_PATH):
    """
    Load evolution results from JSON log.

    Parameters
    ----------
    log_path : str

    Returns
    -------
    results : list
    """

    if not os.path.exists(log_path):
        print("Log file not found:", log_path)
        return []

    with open(log_path, "r") as f:
        data = json.load(f)

    return data.get("results", [])


def extract_generations(results):
    """Return generation numbers."""

    return list(range(len(results)))


def extract_accuracy(results):
    """Return accuracy history."""

    return [r["accuracy"] for r in results]


def extract_fitness(results):
    """Return fitness history."""

    return [r["fitness"] for r in results]


def extract_parameters(results):
    """Return parameter counts."""

    return [r["params"] for r in results]


def get_best_architecture(results):
    """
    Return the best architecture found during evolution.
    """

    if not results:
        return None

    best = max(results, key=lambda x: x["fitness"])

    return best


def summarize_results(results):
    """
    Create summary statistics of evolution run.
    """

    if not results:
        return {}

    accuracies = extract_accuracy(results)
    fitness = extract_fitness(results)
    params = extract_parameters(results)

    summary = {
        "generations": len(results),
        "best_accuracy": max(accuracies),
        "best_fitness": max(fitness),
        "min_parameters": min(params)
    }

    return summary