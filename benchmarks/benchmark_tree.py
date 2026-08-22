import time
import numpy as np
from sklearn.tree import DecisionTreeRegressor


class ScratchDecisionTreeRegressor:
    """Minimal placeholder for benchmarking the future scratch implementation."""

    def fit(self, X, y):
        self.mean_ = np.mean(y)
        return self

    def predict(self, X):
        return np.full(X.shape[0], self.mean_)


def benchmark(model, X, y, X_test, name):
    start = time.perf_counter()
    model.fit(X, y)
    fit_time = time.perf_counter() - start

    start = time.perf_counter()
    predictions = model.predict(X_test)
    predict_time = time.perf_counter() - start

    print(f"{name}")
    print(f"  fit time:     {fit_time:.6f} s")
    print(f"  predict time: {predict_time:.6f} s")
    print(f"  prediction shape: {predictions.shape}")
    print()


rng = np.random.default_rng(42)

X = rng.normal(size=(10_000, 10))
y = rng.normal(size=10_000)
X_test = rng.normal(size=(2_000, 10))

scratch_model = ScratchDecisionTreeRegressor()
sklearn_model = DecisionTreeRegressor(random_state=42)

benchmark(scratch_model, X, y, X_test, "Scratch Decision Tree")
benchmark(sklearn_model, X, y, X_test, "scikit-learn Decision Tree")