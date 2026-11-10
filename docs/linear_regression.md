# Linear Regression in onithrasML

`LinearRegression` is a foundational supervised learning algorithm implemented entirely from scratch using NumPy. It predicts continuous target values by finding the best-fitting linear relationship between the input features and the target.

This implementation is designed for performance, numerical stability, and educational clarity. It supports two solving methods: the **Normal Equation** (closed-form) and **Batch Gradient Descent** (iterative).

---

## 1. Mathematical Background

The model learns a weight vector \( w \) and an intercept \( b \) such that the prediction \( \hat{y} \) is:

\[
 \hat{y} = X \cdot w + b 
\]

The algorithm optimizes the **Mean Squared Error (MSE)** loss function. To prevent overfitting, an optional **L2 Regularization (Ridge)** penalty is added:

\[
 J(w, b) = \frac{1}{2n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2 + \frac{\lambda}{2} ||w||^2 
\]

*(Where \(\lambda\) is the `l2_penalty` hyperparameter. The intercept \( b \) is never regularized).*

### Two Solvers Available:
1. **Normal Equation (`method="normal"`)**: 
   Solves \( w = (X^T X + \lambda I)^{-1} X^T y \) directly. It is exact, fast for smaller datasets, and requires no hyperparameter tuning like learning rate.
2. **Batch Gradient Descent (`method="gd"`)**:
   Iteratively updates weights using the derivative of the loss function. It scales beautifully to massive datasets and memory constraints.

---

## 2. Quick Start (Basic Usage)

Here is the simplest way to train a Linear Regression model using the Normal Equation.

```python
import numpy as np
import matplotlib.pyplot as plt
from onithrasML.linear_model import LinearRegression, LinearRegressionConfig

# -------------------------------------------------------------------
# STEP 1: Creating a Realistic Synthetic Dataset
# -------------------------------------------------------------------
np.random.seed(42)
n_samples = 1000

# Generating features
area = np.random.randint(800, 5000, n_samples)       # 800 - 5000 sq ft
bedrooms = np.random.randint(1, 6, n_samples)        # 1 - 5 bedrooms
age = np.random.randint(0, 30, n_samples)            # 0 - 30 years old
distance = np.random.uniform(1, 30, n_samples)       # 1 - 30 km from center

# Combine into feature matrix X
X = np.column_stack((area, bedrooms, age, distance))

# Creating the target (y) with a realistic formula + noise
# Formula: Price = (Area * 0.05) + (Bedrooms * 10) - (Age * 0.5) - (Distance * 1.5) + Base Price (30)
# Add noise to make it realistic (market fluctuations, etc.)
noise = np.random.normal(0, 15, n_samples)
y = (area * 0.05) + (bedrooms * 10) - (age * 0.5) - (distance * 1.5) + 30 + noise

print("--- Dataset Overview ---")
print(f"Total Samples: {n_samples}")
print(f"Features Shape (X): {X.shape}")
print(f"Target Shape (y): {y.shape}")
print(f"\nFeature Columns: ['Area', 'Bedrooms', 'Age', 'Distance']")
print(f"\nFirst 3 rows of X:\n{X[:3]}")
print(f"First 3 Prices (Lakhs): {y[:3]}")

# -------------------------------------------------------------------
# STEP 2: Training with Normal Equation (Closed-Form)
# -------------------------------------------------------------------
print("\n--- Training with Normal Equation ---")

# Normal equation is fast and exact, best for smaller datasets
config_normal = LinearRegressionConfig(
    method="normal",     # Uses the mathematical closed-form solution
    l2_penalty=0.1,      # Slight regularization to handle multicollinearity
    standardize=True,
    verbose=False
)

model_normal = LinearRegression(config_normal)
model_normal.fit(X, y)

print("Normal Equation Model Trained!")
print(f"Learned Coefficients: {model_normal.coef_}")
print(f"Learned Intercept: {model_normal.intercept_:.4f}")
print(f"Training R² Score: {model_normal.score(X, y):.4f}")

# -------------------------------------------------------------------
# STEP 3: Training with Gradient Descent (Iterative)
# -------------------------------------------------------------------
print("\n--- Training with Gradient Descent ---")

# Gradient descent is slower but scales to massive datasets and allows for partial updates
config_gd = LinearRegressionConfig(
    method="gd",           # Uses iterative optimization
    learning_rate=0.1,     # Step size (needs tuning)
    n_iterations=5000,     # More iterations for convergence
    l2_penalty=0.1,
    standardize=True,      # CRITICAL for gradient descent
    verbose=False
)

model_gd = LinearRegression(config_gd)
model_gd.fit(X, y)

print("Gradient Descent Model Trained!")
print(f"Learned Coefficients: {model_gd.coef_}")
print(f"Learned Intercept: {model_gd.intercept_:.4f}")
print(f"Training R² Score: {model_gd.score(X, y):.4f}")

# -------------------------------------------------------------------
# STEP 4: Visualizing Loss Convergence (GD only)
# -------------------------------------------------------------------
# The normal equation doesn't have a loss history (it's exact),
# but GD keeps a history of MSE at each step.

plt.figure(figsize=(10, 4))
plt.plot(model_gd.loss_history_, color='green', linewidth=2)
plt.title("Gradient Descent: MSE Loss over Iterations")
plt.xlabel("Iterations (Epochs)")
plt.ylabel("Mean Squared Error (MSE)")
plt.grid(True, alpha=0.3)
plt.show()

# -------------------------------------------------------------------
# STEP 5: Predicting on New Houses
# -------------------------------------------------------------------
print("\n--- Predicting Prices for New Houses ---")

# Let's test on 3 new properties
# House 1: Large, new, close to city (Expected: Expensive)
# House 2: Small, old, far from city (Expected: Cheap)
# House 3: Medium everything (Expected: Average)

new_houses = np.array([
    [4000, 5, 2, 3.0],    # House 1: Premium
    [900, 1, 25, 25.0],   # House 2: Budget
    [2500, 3, 10, 12.0]   # House 3: Mid-range
])

# Predict using both models (should be very close)
prices_normal = model_normal.predict(new_houses)
prices_gd = model_gd.predict(new_houses)

for i in range(len(new_houses)):
    print(f"\nHouse {i+1}: Area={new_houses[i][0]} sqft, Bed={new_houses[i][1]}, Age={new_houses[i][2]}yrs, Dist={new_houses[i][3]}km")
    print(f"  Predicted Price (Normal Eq): ₹{prices_normal[i]:.2f} Lakhs")
    print(f"  Predicted Price (Gradient D): ₹{prices_gd[i]:.2f} Lakhs")

# -------------------------------------------------------------------
# STEP 6: Single Feature Visualization (Area vs Price)
# -------------------------------------------------------------------
# To understand the relationship better, let's plot Area vs Price
# (holding other features constant at their mean)

# Create a range of areas
area_range = np.linspace(800, 5000, 100)

# Create a test matrix where only Area changes
mean_features = X.mean(axis=0)
X_test = np.tile(mean_features, (100, 1))
X_test[:, 0] = area_range  # Only vary the Area column

# Predict prices for these areas
predicted_prices = model_normal.predict(X_test)

plt.figure(figsize=(10, 5))
plt.scatter(X[:, 0], y, alpha=0.3, label="Actual Data", color='blue')
plt.plot(area_range, predicted_prices, color='red', linewidth=3, label="Model Prediction")
plt.title("House Price vs Area (Other Features Constant)")
plt.xlabel("Area (sq ft)")
plt.ylabel("Price (Lakhs)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# -------------------------------------------------------------------
# STEP 7: Handling Errors Robustly
# -------------------------------------------------------------------
print("\n--- Error Handling Demo ---")

# 1. Trying to predict with wrong number of features
try:
    model_normal.predict(np.array([[2000, 3, 5]]))  # Missing 'Distance'
except Exception as e:
    print(f"Error 1 Caught: {e}")

# 2. Trying to fit with NaN values
try:
    bad_X = X.copy()
    bad_X[0, 0] = np.nan
    bad_model = LinearRegression()
    bad_model.fit(bad_X, y)
except Exception as e:
    print(f"Error 2 Caught: {e}")

# 3. Trying to predict before fitting
try:
    new_model = LinearRegression()
    new_model.predict(new_houses)
except Exception as e:
    print(f"Error 3 Caught: {e}")

# 4. Singular matrix scenario (forcing an error in normal equation)
try:
    # Creating duplicate features (collinearity) with no regularization
    bad_X2 = np.column_stack((X[:, 0], X[:, 0], X[:, 1]))  # Area duplicated
    bad_config = LinearRegressionConfig(method="normal", l2_penalty=0.0, standardize=False)
    bad_model2 = LinearRegression(bad_config)
    bad_model2.fit(bad_X2, y)
except Exception as e:
    print(f"Error 4 Caught (Singular Matrix): {e}")
```
