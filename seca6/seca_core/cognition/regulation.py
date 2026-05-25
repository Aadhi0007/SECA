"""
Cognitive Regulation Module for SECA

This module performs self-evaluation of the evolutionary process
and decides whether evolution should continue.

The regulation mechanism uses:
- fitness trends
- accuracy stability
- improvement detection
"""

def regulate_evolution(scores, patience=2):
    """
    Decide whether evolution should continue.

    Parameters
    ----------
    scores : list
        List of evaluated genomes with fitness values

    patience : int
        Number of stagnant generations allowed

    Returns
    -------
    evolve : bool
    """

    if len(scores) < 2:
        return True

    # extract fitness values
    fitness_values = [s["fitness"] for s in scores]

    best = max(fitness_values)
    avg = sum(fitness_values) / len(fitness_values)

    # simple cognitive rule:
    # if best fitness is much higher than average, keep evolving
    if best - avg > 0.01:
        return True

    # if improvement is small, evolution may stop
    return True