"""
Dataset Loader for SECA

Loads and preprocesses dataset for training.
"""

import tensorflow as tf


def load_dataset(name="mnist"):
    """
    Load a standardized dataset.

    Parameters
    ----------
    name : str
        Name of the dataset ('mnist', 'fashion_mnist', 'cifar10', 'cifar100', 'imdb', 'reuters')

    Returns
    -------
    x_train, x_test, y_train, y_test
    """

    if name == "mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    elif name == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    elif name == "cifar10":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    elif name == "cifar100":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
    elif name == "imdb":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=10000)
    elif name == "reuters":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.reuters.load_data(num_words=10000)
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    if name in ["imdb", "reuters"]:
        # Process Text Sequences for LLM
        x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=64)
        x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=64)
    else:
        # normalize images
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        # Ensure channel dimension exists
        if len(x_train.shape) == 3:
            # Grayscale (e.g., MNIST)
            x_train = x_train[..., tf.newaxis]
            x_test = x_test[..., tf.newaxis]

    # Flatten labels to 1D if necessary
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    return x_train, x_test, y_train, y_test