"""
Genome Representation for SECA

A genome defines the architecture of a neural network.
Each genome contains:

- number of dense units
- dropout rate
- convolution stages
"""

import random
import copy


# Allowed architecture search space for CNN
FILTER_OPTIONS = [16, 32, 48, 64]
KERNEL_OPTIONS = [3, 5]
POOL_OPTIONS = [1, 2]
CNN_DENSE_OPTIONS = [32, 64, 128]

# Allowed architecture search space for LLM
EMBED_DIM_OPTIONS = [32, 64, 128]
NUM_HEADS_OPTIONS = [2, 4, 8]
FF_DIM_OPTIONS = [64, 128, 256]
LLM_BLOCKS_MIN = 1
LLM_BLOCKS_MAX = 4

DROPOUT_OPTIONS = [0.0, 0.1, 0.2]

MIN_STAGES = 1
MAX_STAGES = 5


def random_stage():
    """Generate a random convolution stage."""
    return {
        "filters": random.choice(FILTER_OPTIONS),
        "kernel": random.choice(KERNEL_OPTIONS),
        "pool": random.choice(POOL_OPTIONS)
    }

def random_llm_block():
    """Generate a random transformer block."""
    return {
        "num_heads": random.choice(NUM_HEADS_OPTIONS),
        "ff_dim": random.choice(FF_DIM_OPTIONS)
    }

def random_cnn_genome():
    num_stages = random.randint(MIN_STAGES, MAX_STAGES)
    return {
        "network_type": "cnn",
        "dense": random.choice(CNN_DENSE_OPTIONS),
        "dropout": random.choice(DROPOUT_OPTIONS),
        "stages": [random_stage() for _ in range(num_stages)]
    }

def random_llm_genome(network_type="llm"):
    num_blocks = random.randint(LLM_BLOCKS_MIN, LLM_BLOCKS_MAX)
    return {
        "network_type": network_type,
        "embed_dim": random.choice(EMBED_DIM_OPTIONS),
        "dropout": random.choice(DROPOUT_OPTIONS),
        "blocks": [random_llm_block() for _ in range(num_blocks)]
    }

def random_genome(network_type="cnn"):
    """Generate a random neural architecture genome."""
    if network_type in ["llm", "gpt", "bert", "llama"]:
        return random_llm_genome(network_type)
    # Default is CNN
    return random_cnn_genome()


def clone_genome(genome):
    """Create a deep copy of a genome."""
    return copy.deepcopy(genome)


def print_genome(genome):
    """Pretty print genome."""

    print(f"\nGenome Architecture ({genome.get('network_type', 'cnn').upper()})")
    print("-------------------")

    if genome.get("network_type") == "llm":
        print(f"Embed Dim: {genome['embed_dim']}")
        for i, block in enumerate(genome["blocks"]):
            print(f"Block {i}: heads={block['num_heads']} ff_dim={block['ff_dim']}")
    else:
        for i, stage in enumerate(genome["stages"]):
            print(
                f"Conv Stage {i}: "
                f"filters={stage['filters']} "
                f"kernel={stage['kernel']} "
                f"pool={stage['pool']}"
            )
        print(f"Dense Units: {genome['dense']}")
        
    print(f"Dropout: {genome['dropout']}")