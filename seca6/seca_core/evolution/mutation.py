"""
Mutation Operator for SECA

This module mutates neural architecture genomes to introduce variation.
"""

import random
from seca_core.evolution.genome import clone_genome

FILTER_OPTIONS = [16, 32, 48, 64]
KERNEL_OPTIONS = [3, 5]
POOL_OPTIONS = [1, 2]
DENSE_OPTIONS = [32, 64, 128]

EMBED_DIM_OPTIONS = [32, 64, 128]
NUM_HEADS_OPTIONS = [2, 4, 8]
FF_DIM_OPTIONS = [64, 128, 256]

DROPOUT_OPTIONS = [0.0, 0.1, 0.2]


def mutate(genome):
    """
    Apply mutation to a genome.
    """

    g = clone_genome(genome)
    net_type = g.get("network_type", "cnn")

    if net_type == "llm":
        mutation_type = random.choice([
            "embed_dim",
            "dropout",
            "num_heads",
            "ff_dim",
            "add_block",
            "remove_block"
        ])

        if mutation_type == "embed_dim":
            g["embed_dim"] = random.choice(EMBED_DIM_OPTIONS)
        elif mutation_type == "dropout":
            g["dropout"] = random.choice(DROPOUT_OPTIONS)
        elif mutation_type == "num_heads" and len(g["blocks"]) > 0:
            block = random.choice(g["blocks"])
            block["num_heads"] = random.choice(NUM_HEADS_OPTIONS)
        elif mutation_type == "ff_dim" and len(g["blocks"]) > 0:
            block = random.choice(g["blocks"])
            block["ff_dim"] = random.choice(FF_DIM_OPTIONS)
        elif mutation_type == "add_block":
            new_block = {
                "num_heads": random.choice(NUM_HEADS_OPTIONS),
                "ff_dim": random.choice(FF_DIM_OPTIONS)
            }
            g["blocks"].append(new_block)
        elif mutation_type == "remove_block":
            if len(g["blocks"]) > 1:
                idx = random.randrange(len(g["blocks"]))
                g["blocks"].pop(idx)
                
    else:
        # CNN Mutation
        mutation_type = random.choice([
            "filters",
            "kernel",
            "pool",
            "dense",
            "dropout",
            "add_stage",
            "remove_stage"
        ])

        if mutation_type == "filters" and len(g["stages"]) > 0:
            stage = random.choice(g["stages"])
            stage["filters"] = random.choice(FILTER_OPTIONS)
        elif mutation_type == "kernel" and len(g["stages"]) > 0:
            stage = random.choice(g["stages"])
            stage["kernel"] = random.choice(KERNEL_OPTIONS)
        elif mutation_type == "pool" and len(g["stages"]) > 0:
            stage = random.choice(g["stages"])
            stage["pool"] = random.choice(POOL_OPTIONS)
        elif mutation_type == "dense":
            g["dense"] = random.choice(DENSE_OPTIONS)
        elif mutation_type == "dropout":
            g["dropout"] = random.choice(DROPOUT_OPTIONS)
        elif mutation_type == "add_stage":
            new_stage = {
                "filters": random.choice(FILTER_OPTIONS),
                "kernel": random.choice(KERNEL_OPTIONS),
                "pool": random.choice(POOL_OPTIONS)
            }
            g["stages"].append(new_stage)
        elif mutation_type == "remove_stage":
            if len(g["stages"]) > 1:
                idx = random.randrange(len(g["stages"]))
                g["stages"].pop(idx)

    return g