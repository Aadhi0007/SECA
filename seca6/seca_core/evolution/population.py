"""
Population Management for SECA

This module initializes and manages the population of neural architectures.
"""

from seca_core.evolution.genome import random_genome


def initialize_population(population_size, network_type="cnn"):
    """
    Create an initial population of genomes.
    """

    population = []

    for _ in range(population_size):
        genome = random_genome(network_type)
        population.append(genome)

    return population