import numpy as np
from onithrasML.linear_model import LinearRegression, LinearRegressionConfig

# 1. Normal equation (fast, exact)
X = np.array([[1.0], [2.0], [3.0], [4.0]])
y = np.array([2.0, 4.0, 6.0, 8.0])

model = LinearRegression(LinearRegressionConfig(method="normal"))
model.fit(X, y)
print("Coef:", model.coef_)          # ~[2.0]
print("Intercept:", model.intercept_) # ~0.0
print("R²:", model.score(X, y))      # 1.0

# 2. Gradient descent (large data)
rng = np.random.default_rng(0)
X_big = rng.normal(size=(10_000, 5))
true_w = np.array([1.5, -2.0, 0.0, 3.0, 0.5])
y_big = X_big @ true_w + 0.1 * rng.normal(size=10_000)

gd_model = LinearRegression(LinearRegressionConfig(
    method="gd",
    learning_rate=0.1,
    n_iterations=500,
    standardize=True,
    verbose=False,
))
gd_model.fit(X_big, y_big)
print("GD coef:", gd_model.coef_)
print("True   :", true_w)