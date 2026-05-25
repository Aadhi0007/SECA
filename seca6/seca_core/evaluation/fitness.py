"""
Fitness Evaluation for SECA

The fitness function evaluates how good a neural architecture is.
It considers both performance and efficiency.

Fitness = Accuracy - Complexity Penalty
"""

def compute_fitness(accuracy, params):
    """
    Compute fitness score for an evolved model.

    Parameters
    ----------
    accuracy : float
        Validation accuracy of the model

    params : int
        Number of parameters in the model

    Returns
    -------
    fitness : float
    """

    # Normalize parameter penalty
    complexity_penalty = params / 1_000_000

    fitness = accuracy - complexity_penalty

    return fitness