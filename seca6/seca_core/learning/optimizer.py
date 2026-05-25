"""
Optimizer Module for SECA

Provides optimizer selection for neural network training.
"""

import tensorflow as tf


def get_optimizer(name="adam", learning_rate=0.001):
    """
    Return optimizer instance.

    Parameters
    ----------
    name : str
        Optimizer name

    learning_rate : float
        Learning rate value

    Returns
    -------
    optimizer : tf.keras.optimizers
    """

    name = name.lower()

    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)

    elif name == "sgd":
        return tf.keras.optimizers.SGD(
            learning_rate=learning_rate,
            momentum=0.9
        )

    elif name == "rmsprop":
        return tf.keras.optimizers.RMSprop(
            learning_rate=learning_rate
        )

    else:
        raise ValueError(f"Unsupported optimizer: {name}")