"""
Trainer Module for SECA

Responsible for:
1. Training neural architectures
2. Evaluating validation accuracy
3. Returning training history
"""

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

EPOCHS = 1 # Set to 1 for fast demonstration. Increase for better actual results.

def train_model(model, train_ds, test_ds):
    """
    Train a neural network model.

    Parameters
    ----------
    model : keras.Model
        Neural network created from genome
    train_ds : tf.data.Dataset
        Training dataset
    test_ds : tf.data.Dataset
        Validation dataset

    Returns
    -------
    history : keras history object
    val_accuracy : float
    """

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=1,
        restore_best_weights=True
    )

    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=test_ds,
        callbacks=[early_stopping],
        verbose=0
    )

    # Extract validation accuracy (may be shorter than EPOCHS due to early stopping)
    val_accuracy = history.history["val_accuracy"][-1]

    return history, val_accuracy