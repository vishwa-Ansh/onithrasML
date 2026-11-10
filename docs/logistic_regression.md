#Logistic Regression in onithrasML

`LogisticRegression` is a classification algorithm implemented from scratch in NumPy. Despite its name, it is used for **binary classification** (predicting whether something belongs to Class 0 or Class 1), not regression.

It works by applying the **Sigmoid function** to a linear combination of features, squashing the output into a probability between 0 and 1. It is trained using **Batch Gradient Descent** and **Binary Cross-Entropy Loss**.

---

## 1. Mathematical Background

The model learns weights \( w \) and bias \( b \) such that:

\[
 z = X \cdot w + b 
\]

This linear output \( z \) is passed through the **Sigmoid function** to get the probability of Class 1:

\[
 \hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}} 
\]

The model optimizes the **Binary Cross-Entropy Loss** (Log Loss):

\[
 J(w, b) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right] + \frac{\lambda}{2} ||w||^2 
\]

*(Where \(\lambda\) is the L2 regularization penalty, which helps prevent overfitting).*

---

## 2. Quick Start (Basic Usage)

Here is the simplest way to train a Logistic Regression model.

```python
import numpy as np
import matplotlib.pyplot as plt
from onithrasML.linear_model import LogisticRegression, LogisticRegressionConfig

# -------------------------------------------------------------------
# STEP 1: Creating a Realistic Synthetic Dataset
# -------------------------------------------------------------------
np.random.seed(42)
n_samples = 500

# Generating features
age = np.random.randint(21, 60, n_samples)
income = np.random.normal(50, 20, n_samples)  # Mean 50k, std 20k
credit_score = np.random.randint(300, 900, n_samples)

# Combine into feature matrix X
X = np.column_stack((age, income, credit_score))

# Creating the target (y) with a realistic rule + noise
# Rule: Good credit score AND good income = Approve
# We add some random noise so the data isn't perfectly separable
score_calc = (credit_score * 0.05) + (income * 0.1) - (age * 0.05) + np.random.normal(0, 5, n_samples)
y = (score_calc > 35).astype(int) 

print("--- Dataset Overview ---")
print(f"Total Samples: {n_samples}")
print(f"Features Shape (X): {X.shape}")
print(f"Target Shape (y): {y.shape}")
print(f"Class Distribution: {np.bincount(y)} (0 = Reject, 1 = Approve)")
print(f"\nFirst 3 rows of X:\n{X[:3]}")

# -------------------------------------------------------------------
# STEP 2: Model Configuration & Training
# -------------------------------------------------------------------
print("\n--- Training Model ---")

# We use standardization because Income is in 10,000s and Credit Score is in 100s
config = LogisticRegressionConfig(
    learning_rate=0.1,     # Step size
    n_iterations=5000,     # Max epochs
    l2_penalty=0.01,       # Slight regularization to prevent overfitting
    standardize=True,      # CRITICAL: Scales features so GD converges fast
    verbose=False
)

model = LogisticRegression(config)

# Train the model
model.fit(X, y)

print("Model trained successfully!")
print(f"Learned Coefficients: {model.coef_}")
print(f"Learned Intercept: {model.intercept_:.4f}")

# Check Accuracy
accuracy = model.score(X, y)
print(f"Training Accuracy: {accuracy * 100:.2f}%")

# -------------------------------------------------------------------
# STEP 3: Visualizing the Loss Convergence
# -------------------------------------------------------------------
# We can plot the loss_history_ to see if the model learned properly
plt.figure(figsize=(8, 4))
plt.plot(model.loss_history_, color='blue', linewidth=2)
plt.title("Binary Cross-Entropy Loss over Iterations")
plt.xlabel("Iterations (Epochs)")
plt.ylabel("Loss")
plt.grid(True, alpha=0.3)
plt.show()

# -------------------------------------------------------------------
# STEP 4: Predicting on New Unseen Customers
# -------------------------------------------------------------------
print("\n--- Predicting New Customers ---")

# Let's test on 3 new customers
# Customer 1: Young, low income, bad credit
# Customer 2: Middle-aged, high income, good credit
# Customer 3: Borderline case

new_customers = np.array([
    [25, 20.0, 450],   # Customer 1 (Expected: Reject)
    [45, 90.0, 800],   # Customer 2 (Expected: Approve)
    [35, 55.0, 650]    # Customer 3 (Borderline)
])

# Get probabilities
probabilities = model.predict_proba(new_customers)

# Get predictions (default threshold 0.5)
predictions = model.predict(new_customers)

for i, (proba, pred) in enumerate(zip(probabilities, predictions)):
    print(f"\nCustomer {i+1}: Age={new_customers[i][0]}, Income={new_customers[i][1]}k, Score={new_customers[i][2]}")
    print(f"  Probability of Reject (0): {proba[0]*100:.2f}%")
    print(f"  Probability of Approve (1): {proba[1]*100:.2f}%")
    decision = "APPROVED ✅" if pred == 1 else "REJECTED ❌"
    print(f"  Final Decision (Threshold 0.5): {decision}")

# -------------------------------------------------------------------
# STEP 5: Custom Thresholding (Business Logic)
# -------------------------------------------------------------------
print("\n--- Strict Bank Policy (Threshold = 0.85) ---")
# Banks are very careful. They might only approve if the model is 85%+ confident.
strict_predictions = model.predict(new_customers, threshold=0.85)

for i, pred in enumerate(strict_predictions):
    decision = "APPROVED ✅" if pred == 1 else "REJECTED ❌"
    print(f"Customer {i+1} Strict Decision: {decision}")

# -------------------------------------------------------------------
# STEP 6: Handling Errors Robustly
# -------------------------------------------------------------------
print("\n--- Error Handling Demo ---")

# 1. Trying to predict with wrong number of features
try:
    model.predict(np.array([[25, 50.0]])) # Missing Credit Score
except Exception as e:
    print(f"Error 1 Caught: {e}")

# 2. Trying to fit with invalid labels
try:
    bad_model = LogisticRegression()
    bad_y = np.array([0, 1, 2, 1, 0] * 100) # Contains '2'
    bad_model.fit(X, bad_y)
except Exception as e:
    print(f"Error 2 Caught: {e}")

# 3. Trying to predict before fitting
try:
    new_model = LogisticRegression()
    new_model.predict(new_customers)
except Exception as e:
    print(f"Error 3 Caught: {e}")
```
