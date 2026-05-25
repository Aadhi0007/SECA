"""
Metrics Module for SECA

This module calculates additional metrics used for
evaluation, logging, and cognitive introspection.
"""


def compute_accuracy(predictions, labels):
    """
    Compute classification accuracy.

    Parameters
    ----------
    predictions : list or array
    labels : list or array

    Returns
    -------
    accuracy : float
    """

    correct = 0

    for p, l in zip(predictions, labels):
        if p == l:
            correct += 1

    accuracy = correct / len(labels)

    return accuracy


def compute_model_complexity(params):
    """
    Compute model complexity score based on parameters.

    Parameters
    ----------
    params : int
        Number of model parameters

    Returns
    -------
    complexity : float
    """

    complexity = params / 1_000_000

    return complexity


def compute_efficiency(accuracy, params):
    """
    Efficiency metric balancing performance and size.

    Parameters
    ----------
    accuracy : float
    params : int

    Returns
    -------
    efficiency : float
    """

    complexity = compute_model_complexity(params)

    efficiency = accuracy / (1 + complexity)

    return efficiency


def summarize_metrics(accuracy, params, fitness):
    """
    Combine metrics into a summary dictionary.

    Parameters
    ----------
    accuracy : float
    params : int
    fitness : float

    Returns
    -------
    summary : dict
    """

    efficiency = compute_efficiency(accuracy, params)

    summary = {
        "accuracy": accuracy,
        "parameters": params,
        "fitness": fitness,
        "efficiency": efficiency
    }

    return summary