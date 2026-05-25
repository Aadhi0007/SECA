"""
Confidence Estimator for SECA

This module estimates how confident the system is about
the quality of its current best architecture.

Confidence is based on:
- population fitness distribution
- fitness variance
- accuracy stability
"""


def estimate_confidence(report):
    """
    Estimate confidence score from introspection report.

    Parameters
    ----------
    report : dict
        Output of introspection module

    Returns
    -------
    confidence : float
    """

    if report is None:
        return 0.0

    best_fitness = report["best_fitness"]
    avg_fitness = report["avg_fitness"]
    variance = report["fitness_variance"]

    # difference between best and average
    improvement_gap = best_fitness - avg_fitness

    # confidence formula
    confidence = improvement_gap / (variance + 1e-6)

    # normalize to range [0,1]
    confidence = min(max(confidence, 0.0), 1.0)

    return confidence