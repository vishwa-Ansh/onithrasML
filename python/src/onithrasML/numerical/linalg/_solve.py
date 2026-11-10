import numpy as np
import scipy.linalg


def solve(a, b):
    """
    Solve a linear system of equations.

    The function solves the equation:

        A @ X = B

    for X, where A is a square coefficient matrix and B is
    the right-hand side.

    Parameters
    ----------
    a : array_like, shape (n, n)
        Square coefficient matrix.

    b : array_like, shape (n,) or (n, m)
        Right-hand side of the linear system.

    Returns
    -------
    x : ndarray, shape (n,) or (n, m)
        Solution of the linear system.

    Raises
    ------
    TypeError
        If `a` or `b` cannot be converted to NumPy arrays.

    ValueError
        If `a` is not 2-dimensional.
        If `a` is not a square matrix.
        If the dimensions of `a` and `b` are incompatible.

    Examples
    --------
    Solve a simple 2x2 system:

    >>> import numpy as np
    >>> from onithrasML.numerical.linalg import solve

    >>> A = np.array([
    ...     [2.0, 1.0],
    ...     [1.0, 3.0]
    ... ])

    >>> b = np.array([5.0, 6.0])

    >>> solve(A, b)
    array([1.8, 1.4])

    Notes
    -----
    The current implementation uses SciPy as the numerical backend.
    The backend can later be replaced by an optimized C++/BLAS/LAPACK
    implementation without changing the public API.
    """

    try:
        a = np.asarray(a)
        b = np.asarray(b)

    except (TypeError, ValueError) as exc:
        raise TypeError(
            "a and b must be convertible to NumPy arrays"
        ) from exc

    # A must be a matrix
    if a.ndim != 2:
        raise ValueError(
            f"a must be 2-dimensional, got {a.ndim}-dimensional"
        )

    # A must be square
    if a.shape[0] != a.shape[1]:
        raise ValueError(
            f"a must be square, got shape {a.shape}"
        )

    # B must have compatible first dimension
    if b.ndim == 0:
        raise ValueError(
            "b must be a 1-dimensional or 2-dimensional array"
        )

    if b.shape[0] != a.shape[0]:
        raise ValueError(
            f"Incompatible dimensions: a has {a.shape[0]} rows "
            f"but b has {b.shape[0]} rows"
        )

    return scipy.linalg.solve(a, b)