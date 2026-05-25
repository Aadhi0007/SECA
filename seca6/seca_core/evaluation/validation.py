"""
Validation Module for SECA

This module evaluates trained neural networks on validation/test data.
It provides prediction accuracy and evaluation statistics.
"""

import numpy as np


def evaluate_model(model, x_test, y_test):
    """
    Evaluate model performance on test data.

    Parameters
    ----------
    model : keras.Model
    x_test : array
    y_test : array

    Returns
    -------
    accuracy : float
    loss : float
    """

    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

    return accuracy, loss


def predict_classes(model, x_test):
    """
    Generate class predictions for test data.

    Parameters
    ----------
    model : keras.Model
    x_test : array

    Returns
    -------
    predictions : list
    """

    probs = model.predict(x_test, verbose=0)

    predictions = np.argmax(probs, axis=1)

    return predictions


def validation_summary(model, x_test, y_test):
    """
    Generate a validation report.

    Parameters
    ----------
    model : keras.Model
    x_test : array
    y_test : array

    Returns
    -------
    report : dict
    """

    accuracy, loss = evaluate_model(model, x_test, y_test)

    report = {
        "accuracy": float(accuracy),
        "loss": float(loss),
        "samples": len(x_test)
    }

    return report