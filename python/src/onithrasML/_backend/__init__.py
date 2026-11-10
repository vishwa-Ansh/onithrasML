try:
    from ._native import LinearRegression
    NATIVE_AVAILABLE = True
except ImportError:
    LinearRegression = None
    NATIVE_AVAILABLE = False

__all__ = ["LinearRegression", "NATIVE_AVAILABLE"]