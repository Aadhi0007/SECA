"""
Crossover Operator for SECA

This module combines two parent genomes to produce a child genome.
"""

import random
from seca_core.evolution.genome import clone_genome


def crossover(parent1, parent2):
    """
    Perform crossover between two genomes.

    Parameters
    ----------
    parent1 : dict
    parent2 : dict

    Returns
    -------
    child : dict
    """

    p1 = clone_genome(parent1)
    p2 = clone_genome(parent2)

    child = {}

    net_type = p1.get("network_type", "cnn")
    child["network_type"] = net_type

    # Combine depending on network type
    if net_type != "cnn":
        child["embed_dim"] = random.choice([p1["embed_dim"], p2["embed_dim"]])
        child["dropout"] = random.choice([p1["dropout"], p2["dropout"]])

        blocks1 = p1.get("blocks", [])
        blocks2 = p2.get("blocks", [])

        split1 = random.randint(0, len(blocks1) - 1) if len(blocks1) > 0 else 0
        split2 = random.randint(0, len(blocks2) - 1) if len(blocks2) > 0 else 0

        child_blocks = blocks1[:split1] + blocks2[split2:]

        # Ensure at least one block
        if len(child_blocks) == 0:
            if blocks1:
                child_blocks = [random.choice(blocks1)]
            elif blocks2:
                child_blocks = [random.choice(blocks2)]
            else:
                child_blocks = []

        child["blocks"] = child_blocks

    else:
        # Combine dense layer
        child["dense"] = random.choice([p1["dense"], p2["dense"]])

        # Combine dropout
        child["dropout"] = random.choice([p1["dropout"], p2["dropout"]])

        # Combine convolution stages
        stages1 = p1["stages"]
        stages2 = p2["stages"]

        split1 = random.randint(0, len(stages1) - 1) if stages1 else 0
        split2 = random.randint(0, len(stages2) - 1) if stages2 else 0

        child_stages = stages1[:split1] + stages2[split2:]

        # Ensure at least one stage
        if len(child_stages) == 0:
            if stages1:
                child_stages = [random.choice(stages1)]
            else:
                child_stages = []

        child["stages"] = child_stages

    return child