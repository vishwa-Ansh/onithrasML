"""
logistic_regression.py

A complete, from-scratch Logistic Regression implementation for binary classification.

Supports:
    - Batch Gradient Descent
    - Optional L2 regularization
    - Sigmoid activation
    - Probability predictions (predict_proba)
    - Class predictions (predict)
    - Comprehensive input validation and error handling

Author: <your name>
License: MIT
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, Literal

import numpy as np

# Attempt to import base exceptions from your existing linear regression module.
# If not found (e.g., standalone testing), define them here.
try:
    from ._linear_regression import (
        LinearRegressionError,
        NotFittedError,
        InvalidInputError,
    )
except ImportError:
    class LinearRegressionError(Exception):
        pass
    class NotFittedError(LinearRegressionError):
        pass
    class InvalidInputError(LinearRegressionError):
        pass

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LogisticRegressionConfig:
    """
    Configuration for the LogisticRegression model.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent.
    n_iterations : int
        Number of gradient descent iterations.
    tol : float
        Convergence tolerance. Training stops early when the change 
        in loss is smaller than this value.
    l2_penalty : float
        L2 regularization strength. 0.0 means no regularization.
        Note: the intercept is never regularized.
    fit_intercept : bool
        Whether to fit an intercept term.
    standardize : bool
        Whether to standardize features before training. 
        Strongly recommended for gradient descent.
    verbose : bool
        If True, logs loss every `log_every` iterations.
    log_every : int
        Logging frequency.
    """
    learning_rate: float = 0.01
    n_iterations: int = 1000
    tol: float = 1e-7
    l2_penalty: float = 0.0
    fit_intercept: bool = True
    standardize: bool = True
    verbose: bool = False
    log_every: int = 100

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.n_iterations <= 0:
            raise ValueError("n_iterations must be positive")
        if self.tol < 0:
            raise ValueError("tol must be non-negative")
        if self.l2_penalty < 0:
            raise ValueError("l2_penalty must be non-negative")


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class LogisticRegression:
    """
    Logistic Regression model implemented from scratch for binary classification.
    
    Labels (y) must be 0 or 1.
    """
    config: LogisticRegressionConfig = field(default_factory=LogisticRegressionConfig)

    coef_: Optional[np.ndarray] = field(default=None, init=False)
    intercept_: Optional[float] = field(default=None, init=False)
    n_features_in_: Optional[int] = field(default=None, init=False)
    loss_history_: list[float] = field(default_factory=list, init=False)
    is_fitted_: bool = field(default=False, init=False)

    # Internal scaler state
    _feature_mean: Optional[np.ndarray] = field(default=None, init=False, repr=False)
    _feature_std: Optional[np.ndarray] = field(default=None, init=False, repr=False)

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """
        Fit the model to training data using gradient descent.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target labels. Must contain only 0 and 1.

        Returns
        -------
        self : LogisticRegression
            Fitted model.
        """
        X, y = self._validate_fit_input(X, y)

        self.n_features_in_ = X.shape[1]
        self.loss_history_ = []

        # Standardize features
        if self.config.standardize:
            self._feature_mean = X.mean(axis=0)
            self._feature_std = X.std(axis=0)
            self._feature_std[self._feature_std == 0.0] = 1.0
            X_scaled = (X - self._feature_mean) / self._feature_std
        else:
            self._feature_mean = None
            self._feature_std = None
            X_scaled = X

        n_samples, n_features = X_scaled.shape
        
        # Initialize weights (including intercept as weight[0] if fit_intercept is True)
        self.coef_ = np.zeros(n_features, dtype=np.float64)
        self.intercept_ = 0.0

        prev_loss = np.inf

        for i in range(self.config.n_iterations):
            # Linear combination
            z = X_scaled @ self.coef_ + self.intercept_
            
            # Sigmoid activation (probabilities)
            y_pred_proba = self._sigmoid(z)
            
            # Clip probabilities to avoid log(0)
            eps = 1e-15
            y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)

            # Binary Cross-Entropy Loss
            loss = -np.mean(y * np.log(y_pred_proba) + (1 - y) * np.log(1 - y_pred_proba))
            
            # Add L2 penalty (excluding intercept)
            if self.config.l2_penalty > 0:
                loss += (self.config.l2_penalty / 2) * np.sum(self.coef_ ** 2)

            self.loss_history_.append(float(loss))

            if self.config.verbose and (i % self.config.log_every == 0 or i == 0):
                logger.info("Logistic GD iter %d: loss = %.6f", i, loss)

            # Gradients
            error = y_pred_proba - y  # Derivative of BCE w.r.t z
            
            grad_w = (1 / n_samples) * (X_scaled.T @ error)
            grad_b = (1 / n_samples) * np.sum(error)

            # Add L2 penalty gradient to weights (not intercept)
            if self.config.l2_penalty > 0:
                grad_w += self.config.l2_penalty * self.coef_

            # Update weights
            self.coef_ -= self.config.learning_rate * grad_w
            self.intercept_ -= self.config.learning_rate * grad_b

            # Early stopping
            if abs(prev_loss - loss) < self.config.tol:
                if self.config.verbose:
                    logger.info("Converged at iteration %d (loss=%.6f)", i, loss)
                break
            prev_loss = loss

        # Convert weights back to original feature space
        if self.config.standardize:
            self.coef_ = self.coef_ / self._feature_std
            self.intercept_ = self.intercept_ - np.dot(self._feature_mean, self.coef_)

        self.is_fitted_ = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for given features.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)

        Returns
        -------
        proba : np.ndarray of shape (n_samples, 2)
            Probability of class 0 and class 1. 
            proba[:, 0] is P(y=0), proba[:, 1] is P(y=1).
        """
        if not self.is_fitted_:
            raise NotFittedError("This LogisticRegression instance is not fitted yet.")

        X = self._validate_predict_input(X)
        z = X @ self.coef_ + self.intercept_
        prob_1 = self._sigmoid(z)
        prob_0 = 1.0 - prob_1
        return np.column_stack((prob_0, prob_1))

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels (0 or 1) for given features.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
        threshold : float, default=0.5
            Probability threshold for classifying as class 1.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted labels (0 or 1).
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Return the mean accuracy on the given test data and labels.
        """
        y_pred = self.predict(X)
        y = np.asarray(y).ravel()
        return float(np.mean(y_pred == y))

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid function."""
        # Prevent overflow in exp
        z_clipped = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    @staticmethod
    def _validate_fit_input(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if X.ndim == 1:
            X = X.reshape(-1, 1)
            
        if X.ndim != 2:
            raise InvalidInputError(f"X must be 1D or 2D, got shape {X.shape}")
        if X.shape[0] != y.shape[0]:
            raise InvalidInputError(f"X and y have inconsistent sample counts: {X.shape[0]} vs {y.shape[0]}")
        if X.shape[0] == 0 or X.shape[1] == 0:
            raise InvalidInputError("X or y is empty")
        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
            raise InvalidInputError("X or y contains NaN or Inf values")

        # Ensure binary labels
        unique_labels = np.unique(y)
        if not np.all(np.isin(unique_labels, [0, 1])):
            raise InvalidInputError(f"y must contain only 0 and 1. Found: {unique_labels}")

        return X, y

    def _validate_predict_input(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.ndim != 2:
            raise InvalidInputError(f"X must be 1D or 2D, got shape {X.shape}")
        if X.shape[1] != self.n_features_in_:
            raise InvalidInputError(f"X has {X.shape[1]} features, but model was trained on {self.n_features_in_}")
        if not np.all(np.isfinite(X)):
            raise InvalidInputError("X contains NaN or Inf values")
        return X