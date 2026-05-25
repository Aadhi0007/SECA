"""
Introspection Module for SECA

This module analyzes the performance of the population and
produces self-evaluation signals used by the regulation layer.

It evaluates:
- best fitness
- average fitness
- fitness variance
- model complexity trends
"""


def analyze_population(scores):
    """
    Perform introspection on the current population.

    Parameters
    ----------
    scores : list
        List of dictionaries containing evaluation results

    Returns
    -------
    report : dict
    """

    if len(scores) == 0:
        return None

    fitness_values = [s["fitness"] for s in scores]
    accuracy_values = [s["accuracy"] for s in scores]
    param_values = [s["params"] for s in scores]

    best_fitness = max(fitness_values)
    avg_fitness = sum(fitness_values) / len(fitness_values)

    best_accuracy = max(accuracy_values)
    avg_accuracy = sum(accuracy_values) / len(accuracy_values)

    avg_params = sum(param_values) / len(param_values)

    # variance calculation
    fitness_variance = sum(
        (f - avg_fitness) ** 2 for f in fitness_values
    ) / len(fitness_values)

    report = {
        "best_fitness": best_fitness,
        "avg_fitness": avg_fitness,
        "fitness_variance": fitness_variance,
        "best_accuracy": best_accuracy,
        "avg_accuracy": avg_accuracy,
        "avg_params": avg_params
    }

    return report