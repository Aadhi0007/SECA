"""
Selection Operator for SECA

Selects the top performing genomes based on fitness.
"""


def select_top(scores, k=2):
    """
    Select top-k genomes from population.

    Parameters
    ----------
    scores : list
        List of evaluated genomes with fitness

    k : int
        Number of parents to select

    Returns
    -------
    parents : list
    """

    # sort by fitness descending
    sorted_scores = sorted(
        scores,
        key=lambda x: x["fitness"],
        reverse=True
    )

    parents = sorted_scores[:k]

    return parents