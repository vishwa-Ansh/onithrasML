import numpy as np
class SimpeleImputer:
    """
    Impute missing values in numerical and categorical datasets.

    This class provides simple strategies for replacing missing values,
    similar to scikit-learn's SimpleImputer.

    Parameters
    ----------
    strategy : str, default="mean"
        Strategy used to replace missing values.

        Supported strategies are:

        - "mean"
            Replace missing numerical values with the mean of the column.

        - "median"
            Replace missing numerical values with the median of the column.

        - "most_frequent"
            Replace missing values with the most frequently occurring value.

        - "constant"
            Replace missing values with `fill_value`.

    fill_value : object, default=None
        Value used when `strategy="constant"`.

    Attributes
    ----------
    strategy : str
        Imputation strategy selected by the user.

    fill_value : object
        Value used for constant imputation.

    statistics_ : numpy.ndarray or None
        Statistics learned from each column during `fit()`.

        For example, when using the "mean" strategy:

        [30.0, 45000.0]

        means that the first column has a learned mean of 30.0
        and the second column has a learned mean of 45000.0.

    is_fitted : bool
        Indicates whether the imputer has been fitted.

    Examples
    --------
    >>> X = np.array([
    ...     [20, 30000],
    ...     [25, 40000],
    ...     [np.nan, 50000],
    ...     [35, np.nan]
    ... ])

    >>> imputer = SimpeleImputer(strategy="mean")

    >>> X_new = imputer.fit_transform(X)

    >>> print(X_new)
    [[2.0e+01 3.0e+04]
     [2.5e+01 4.0e+04]
     [2.5e+01 5.0e+04]
     [3.5e+01 4.0e+04]]
    """

    def __init__(self, strategy="mean", fill_value=None):
        """
        Initialize the imputer.

        Parameters
        ----------
        strategy : str, default="mean"
            Strategy used for imputing missing values.

        fill_value : object, default=None
            Replacement value used when `strategy="constant"`.
        """

        self.strategy = strategy
        self.fill_value = fill_value

        # Learned statistics are stored here after fit().
        self.statistics_ = None

        # Indicates whether fit() has been called successfully.
        self.is_fitted = False


    def fit(self, X):
        """
        Learn the imputation statistic for each column.

        The method calculates the required statistic from the provided
        data and stores the result in `statistics_`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data containing missing values.

        Returns
        -------
        self
            Returns the fitted imputer.

        Raises
        ------
        ValueError
            If the input data is empty, cannot be converted to a NumPy
            array, or an unsupported strategy is provided.

        TypeError
            If the input data contains an unsupported type.

        Notes
        -----
        The statistics learned during `fit()` should be used to transform
        both training and test data. The test data should only be passed
        to `transform()` to avoid data leakage.
        """

        # Check whether input data is provided.
        if X is None:
            raise ValueError("Required valid data.")

        try:
            X = np.asarray(X, dtype=object)

        except ValueError as e:
            raise ValueError(
                f"Invalid data for array conversion: {e}"
            ) from e

        except TypeError as e:
            raise TypeError(
                f"Unsupported data type for array conversion: {e}"
            ) from e

        # Check for an empty array.
        if X.size == 0:
            raise ValueError("Required valid data.")

        if X.ndim == 1:

            # Convert 1D array into a 2D column vector.
            X = X.reshape(-1, 1)

        valid_strategies = [
            "mean",
            "median",
            "most_frequent",
            "constant"
        ]

        if self.strategy not in valid_strategies:
            raise ValueError(
                f"Invalid strategy: {self.strategy}. "
                f"Expected one of {valid_strategies}."
            )

        statistics = []

        # Iterate over every column.
        for j in range(X.shape[1]):

            column = X[:, j]

            # Identify missing values in the current column.
            missing_mask = self._is_missing_(column)

            # Select only non-missing values.
            valid_value = column[~missing_mask]

            # Handle a completely missing column.
            if len(valid_value) == 0:

                if self.strategy == "constant":
                    statistics.append(self.fill_value)

                else:
                    statistics.append(np.nan)

                continue

            # Mean strategy.
            if self.strategy == "mean":

                try:
                    values = np.array(
                        valid_value,
                        dtype=float
                    )

                    statistic = np.mean(values)

                except (ValueError, TypeError) as e:
                    raise TypeError(
                        "Mean strategy can only be used with "
                        "numerical data."
                    ) from e

            # Median strategy.
            elif self.strategy == "median":

                try:
                    values = np.array(
                        valid_value,
                        dtype=float
                    )

                    statistic = np.median(values)

                except (ValueError, TypeError) as e:
                    raise TypeError(
                        "Median strategy can only be used with "
                        "numerical data."
                    ) from e

            # Constant strategy.
            elif self.strategy == "constant":

                statistic = self.fill_value

            # Most frequent strategy.
            elif self.strategy == "most_frequent":

                values = np.array(valid_value)

                unique_value, count = np.unique(
                    values,
                    return_counts=True
                )

                max_index = np.argmax(count)

                statistic = unique_value[max_index]

            statistics.append(statistic)

        # Store learned statistics after processing all columns.
        self.statistics_ = np.array(
            statistics,
            dtype=object
        )

        # Mark the estimator as fitted.
        self.is_fitted = True

        return self


    def transform(self, X):
        """
        Replace missing values using statistics learned during `fit()`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data containing missing values.

        Returns
        -------
        numpy.ndarray
            Transformed data with missing values replaced.

        Raises
        ------
        ValueError
            If `fit()` has not been called before `transform()`.

        TypeError
            If the input data cannot be converted to a NumPy array.

        Notes
        -----
        This method does not calculate new statistics. It uses the
        statistics learned during `fit()`.
        """

        if not self.is_fitted:
            raise ValueError(
                "Imputer is not fitted. "
                "Call fit() before transform()."
            )

        if X is None:
            raise ValueError("Required valid data.")

        try:
            X = np.asarray(X, dtype=object)

        except ValueError as e:
            raise ValueError(
                f"Invalid data for array conversion: {e}"
            ) from e

        except TypeError as e:
            raise TypeError(
                f"Unsupported data type for array conversion: {e}"
            ) from e

        if X.size == 0:
            raise ValueError("Required valid data.")

        if X.ndim == 1:

            # Convert 1D array into a 2D column vector.
            X = X.reshape(-1, 1)

        X_new = X.copy()

        # Iterate over every column.
        for j in range(X_new.shape[1]):

            column = X_new[:, j]

            # Identify missing values.
            missing_mask = self._is_missing_(column)

            # Get the statistic learned for this column.
            replacement = self.statistics_[j]

            # Replace missing values.
            X_new[missing_mask, j] = replacement

        # Try to return numerical data as float.
        try:
            return X_new.astype(float)

        except (ValueError, TypeError):
            # Return object array for categorical/mixed data.
            return X_new


    def fit_transform(self, X):
        """
        Fit the imputer and transform the data.

        This is equivalent to:

            imputer.fit(X)
            imputer.transform(X)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data containing missing values.

        Returns
        -------
        numpy.ndarray
            Transformed data with missing values replaced.
        """

        self.fit(X)

        return self.transform(X)


    def _is_missing_(self, column):
        """
        Identify missing values in a single column.

        Numerical columns are processed using NumPy's vectorized
        `np.isnan()` operation for better performance.

        Object/categorical columns are processed using a Python loop
        to support values such as `None`.

        Parameters
        ----------
        column : numpy.ndarray
            One-dimensional column from the dataset.

        Returns
        -------
        numpy.ndarray
            Boolean mask where:

            True
                Value is missing.

            False
                Value is not missing.
        """

        # Numerical columns:
        # NumPy performs the operation using optimized low-level code.
        if np.issubdtype(column.dtype, np.number):

            return np.isnan(column)

        # Categorical/object columns:
        # Python-level checking is required because np.isnan()
        # cannot directly process strings and arbitrary objects.
        else:

            mask = []

            for value in column:

                # Python None is considered missing.
                if value is None:
                    mask.append(True)

                # Check for NaN stored inside an object array.
                elif (
                    isinstance(value, (float, np.floating))
                    and np.isnan(value)
                ):
                    mask.append(True)

                # Otherwise, the value is not missing.
                else:
                    mask.append(False)

            return np.array(mask)