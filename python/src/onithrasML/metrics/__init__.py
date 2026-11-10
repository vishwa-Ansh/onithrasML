"""
_metrics.py

Evaluation metrics for regression and classification models.
"""

import numpy as np
from typing import Optional
from ..linear_model._linear_regression import InvalidInputError

# ---------------------------------------------------------
# REGRESSION METRICS
# ---------------------------------------------------------

def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Mean Squared Error (MSE)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise InvalidInputError("y_true and y_pred must have the same shape")
    return float(np.mean((y_true - y_pred) ** 2))

def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the Mean Absolute Error (MAE)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise InvalidInputError("y_true and y_pred must have the same shape")
    return float(np.mean(np.abs(y_true - y_pred)))

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the R-squared (Coefficient of Determination) score."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise InvalidInputError("y_true and y_pred must have the same shape")
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0.0:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))

# ---------------------------------------------------------
# CLASSIFICATION METRICS
# ---------------------------------------------------------

def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the accuracy classification score."""
    y_true, y_pred = np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise InvalidInputError("y_true and y_pred must have the same shape")
    return float(np.mean(y_true == y_pred))