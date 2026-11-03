"""
_split.py

A from-scratch implementation of train_test_split.
Splits arrays into random train and test subsets.
"""

from __future__ import annotations
import numpy as np
from typing import Union, Tuple, Optional

try:
    from ..linear_model._linear_regression import InvalidInputError
except ImportError:
    class InvalidInputError(Exception):
        pass


def train_test_split(
    *arrays: np.ndarray,
    test_size: Union[float, int] = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True
) -> Tuple[np.ndarray, ...]:
    """
    Split arrays or matrices into random train and test subsets.

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse matrices or pandas dataframes.
    test_size : float or int, default=0.2
        If float, should be between 0.0 and 1.0 and represent the proportion 
        of the dataset to include in the test split. 
        If int, represents the absolute number of test samples.
    random_state : int, default=None
        Controls the shuffling applied to the data before applying the split.
        Pass an int for reproducible output across multiple function calls.
    shuffle : bool, default=True
        Whether or not to shuffle the data before splitting.

    Returns
    -------
    splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.
        (e.g., X_train, X_test, y_train, y_test)
    """
    
    # 1. Validate inputs
    n_arrays = len(arrays)
    if n_arrays == 0:
        raise InvalidInputError("At least one array required as input")
        
    n_samples = len(arrays[0])
    for arr in arrays:
        if len(arr) != n_samples:
            raise InvalidInputError("All input arrays must have the same number of samples")

    # 2. Determine test set size
    if isinstance(test_size, float):
        if test_size <= 0.0 or test_size >= 1.0:
            raise ValueError("test_size as float must be between 0.0 and 1.0")
        n_test = int(np.ceil(test_size * n_samples))
    elif isinstance(test_size, int):
        if test_size <= 0 or test_size >= n_samples:
            raise ValueError("test_size as int must be > 0 and < n_samples")
        n_test = test_size
    else:
        raise TypeError("test_size must be a float or int")
        
    n_train = n_samples - n_test

    # 3. Shuffle indices if required
    if shuffle:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(n_samples)
    else:
        indices = np.arange(n_samples)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    # 4. Split the arrays
    result = []
    for arr in arrays:
        arr = np.asarray(arr)
        result.append(arr[train_indices])
        result.append(arr[test_indices])

    return tuple(result)