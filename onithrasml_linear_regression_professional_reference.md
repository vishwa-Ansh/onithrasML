# OnithrasML — Linear Regression: Complete Mathematical, Statistical, and Implementation Reference

> A comprehensive mathematical, statistical, diagnostic, and implementation specification for Linear Regression in **OnithrasML**.

[![Python](https://img.shields.io/badge/Python-3.9_and_above-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/Docs-OnithrasML-blue.svg)](https://onithrasml.vercel.app)

| Document Specification | Details |
| :--- | :--- |
| **Library & Module** | `onithrasML.linear_model.LinearRegression` |
| **Scope** | Classical Ordinary Least Squares ($\text{OLS}$), Matrix Calculus, Statistical Inference, Diagnostics, Regularization, and Numerical Implementation |
| **Target Audience** | Machine Learning Engineers, Data Scientists, Statisticians, and Researchers |
| **Document Version** | Production Reference $2.0$ (Expanded Edition) |

Linear Regression is one of the most important foundations of classical machine learning and econometrics. This reference guide progresses systematically from the basic univariate model all the way to **Ordinary Least Squares ($\text{OLS}$), matrix calculus derivations, Maximum Likelihood Estimation ($\text{MLE}$), statistical inference, residual and influence diagnostics, Weighted Least Squares ($\text{WLS}$), Generalized Least Squares ($\text{GLS}$), Ridge, Lasso, Elastic Net, Gradient Descent, QR/SVD numerical stability, and full implementation from scratch**.

---

## Table of Contents

### Part I: Mathematical Foundations & Ordinary Least Squares
1. [What Is Linear Regression?](#1-what-is-linear-regression)
2. [Why Linear Regression Matters](#2-why-linear-regression-matters)
3. [Mathematical Setup](#3-mathematical-setup)
4. [Simple Linear Regression](#4-simple-linear-regression)
5. [Deriving the Simple Regression Coefficients](#5-deriving-the-simple-regression-coefficients)
6. [Multiple Linear Regression](#6-multiple-linear-regression)
7. [The Design Matrix](#7-the-design-matrix)
8. [The Statistical Model](#8-the-statistical-model)
9. [Residuals](#9-residuals)
10. [Least Squares](#10-least-squares)
11. [Normal Equation — Full Derivation](#11-normal-equation--full-derivation)
12. [Geometric Interpretation](#12-geometric-interpretation)
13. [Worked Numerical Example](#13-worked-numerical-example)

### Part II: Evaluation Metrics & Statistical Inference
14. [Regression Metrics](#14-regression-metrics)
15. [$R^2$ and Adjusted $R^2$](#15-r2-and-adjusted-r2)
16. [Degrees of Freedom](#16-degrees-of-freedom)
17. [Error Variance](#17-error-variance)
18. [Standard Error — Derivation](#18-standard-error--derivation)
19. [$t$-Test and $p$-Value](#19-t-test-and-p-value)
20. [Confidence Intervals and Prediction Intervals](#20-confidence-intervals-and-prediction-intervals)
21. [$z$-Test](#21-z-test)
22. [$F$-Test](#22-f-test)
23. [Regression Assumptions](#23-regression-assumptions)
24. [Gauss–Markov Theorem, BLUE, and Maximum Likelihood](#24-gaussmarkov-theorem-blue-and-maximum-likelihood)
25. [Unbiasedness — Derivation](#25-unbiasedness--derivation)
26. [Variance of OLS — Derivation and Robust Standard Errors](#26-variance-of-ols--derivation-and-robust-standard-errors)

### Part III: Diagnostics, Multicollinearity & Influential Observations
27. [Multicollinearity](#27-multicollinearity)
28. [Variance Inflation Factor (VIF)](#28-variance-inflation-factor-vif)
29. [Residual Diagnostics](#29-residual-diagnostics)
30. [Hat Matrix and Leverage](#30-hat-matrix-and-leverage)
31. [Cook's Distance](#31-cooks-distance)

### Part IV: Optimization, Numerical Stability & Extended Estimators
32. [Gradient Descent](#32-gradient-descent)
33. [Gradient Descent — Full Derivation](#33-gradient-descent--full-derivation)
34. [Feature Scaling](#34-feature-scaling)
35. [Numerical Stability (QR and SVD Decompositions)](#35-numerical-stability-qr-and-svd-decompositions)
36. [Weighted Least Squares (WLS)](#36-weighted-least-squares-wls)
37. [Generalized Least Squares (GLS)](#37-generalized-least-squares-gls)
38. [Ridge Regression ($L_2$ Regularization)](#38-ridge-regression-l2-regularization)
39. [Lasso ($L_1$) and Elastic Net Regression](#39-lasso-l1-and-elastic-net-regression)
40. [Polynomial Regression](#40-polynomial-regression)
41. [Interaction Terms](#41-interaction-terms)

### Part V: Machine Learning Validation, Implementation & Reference
42. [Train/Test Split](#42-traintest-split)
43. [Cross-Validation](#43-cross-validation)
44. [Data Leakage](#44-data-leakage)
45. [Bias–Variance Tradeoff](#45-biasvariance-tradeoff)
46. [Implementation From Scratch](#46-implementation-from-scratch)
47. [OnithrasML Usage](#47-onithrasml-usage)
48. [Complete Regression Workflow](#48-complete-regression-workflow)
49. [Common Mistakes](#49-common-mistakes)
50. [Formula Sheet](#50-formula-sheet)
51. [Concept Map](#51-concept-map)
52. [Learning Checklist](#52-learning-checklist)

---

## Part I: Mathematical Foundations & Ordinary Least Squares

### 1. What Is Linear Regression?

Linear Regression models a continuous target variable as a linear combination of one or more predictors.

For a single predictor variable $x$:

$$
\hat{y} = \beta_0 + \beta_1 x
$$

For several predictors $x_1, x_2, \dots, x_p$:

$$
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p
$$

The defining requirement of the model is that it is **linear in the parameters** $\beta_0, \beta_1, \dots, \beta_p$. A specification such as

$$
\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2
$$

is still a linear regression model because the partial derivatives of $\hat{y}$ with respect to each parameter $\beta_j$ do not depend on any parameter $\beta_k$:

$$
\frac{\partial \hat{y}}{\partial \beta_0} = 1, \qquad \frac{\partial \hat{y}}{\partial \beta_1} = x, \qquad \frac{\partial \hat{y}}{\partial \beta_2} = x^2
$$

By contrast, a model such as $\hat{y} = \beta_0 \exp(\beta_1 x)$ is nonlinear in the parameters because $\frac{\partial \hat{y}}{\partial \beta_1} = \beta_0 x \exp(\beta_1 x)$ depends on $\beta_0$ and $\beta_1$.

The objective of Ordinary Least Squares ($\text{OLS}$) is to find the parameter vector $\hat{\beta}$ that minimizes the sum of squared residuals across all $n$ observations:

$$
\boxed{
\hat{\beta}
=
\arg\min_{\beta}
\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
}
$$

---

### 2. Why Linear Regression Matters

Linear Regression serves as the foundational pillar of statistical modeling and supervised machine learning because it provides a direct bridge between six core disciplines:

- **Linear Algebra** — vector spaces, matrices, orthogonal projections, subspaces, and matrix rank.
- **Calculus** — partial derivatives, gradients, Hessian matrices, and convexity.
- **Statistics** — point estimation, sampling distributions, hypothesis testing, and confidence intervals.
- **Probability** — conditional expectation, Gaussian noise processes, and likelihood functions.
- **Optimization** — closed-form analytical solutions versus iterative first-order methods.
- **Machine Learning** — feature representation, regularization, cross-validation, and out-of-sample generalization.

#### End-to-End Conceptual Pipeline

| Step | Stage | Mathematical Expression | Core Operation |
| :---: | :--- | :--- | :--- |
| $1$ | **Raw Data** | $\{(x_i, y_i)\}_{i=1}^n$ | Collect $n$ observations of predictors and continuous target values. |
| $2$ | **Design Matrix** | $X \in \mathbb{R}^{n \times (p+1)}$ | Encode predictors alongside a leading column of ones for the intercept. |
| $3$ | **Statistical Model** | $y = X\beta + \epsilon$ | Formulate the population relationship with unobserved error $\epsilon$. |
| $4$ | **Least Squares Objective** | $\min_{\beta} \|y - X\beta\|_2^2$ | Choose $\beta$ that minimizes the sum of squared errors. |
| $5$ | **Parameter Estimate** | $\hat{\beta} = (X^T X)^{-1} X^T y$ | Compute the optimal coefficient vector via the Normal Equation. |
| $6$ | **Predictions** | $\hat{y} = X\hat{\beta} = Hy$ | Project $y$ onto the column space of $X$. |
| $7$ | **Residuals** | $e = y - \hat{y}$ | Compute the orthogonal error component for diagnostics and inference. |
| $8$ | **Evaluation & Inference** | $R^2, \text{RMSE}, t, F, \text{VIF}$ | Quantify predictive accuracy, parameter uncertainty, and model validity. |

---

### 3. Mathematical Setup

Let us define the core notation used throughout this specification:

- $n$ = number of observations (samples)
- $p$ = number of predictors (features), excluding the intercept
- $X$ = design matrix containing the intercept column and all $p$ predictor columns
- $y$ = observed target vector
- $\beta$ = true population coefficient vector
- $\hat{\beta}$ = estimated sample coefficient vector
- $\epsilon$ = unobserved theoretical error vector (disturbance term)
- $e$ = observed sample residual vector

#### Dimensional Analysis

$$
X \in \mathbb{R}^{n \times (p+1)}, \qquad y \in \mathbb{R}^{n \times 1}, \qquad \beta \in \mathbb{R}^{(p+1) \times 1}, \qquad \epsilon \in \mathbb{R}^{n \times 1}, \qquad e \in \mathbb{R}^{n \times 1}
$$

The classical linear regression model is written in matrix notation as:

$$
\boxed{
y = X\beta + \epsilon
}
$$

After estimating the parameters $\hat{\beta}$ from data, the fitted vector of predictions is:

$$
\boxed{
\hat{y} = X\hat{\beta}
}
$$

---

### 4. Simple Linear Regression

When $p = 1$, the model relates a single predictor $x_i$ to the target $y_i$:

$$
\boxed{
y_i = \beta_0 + \beta_1 x_i + \epsilon_i
}
$$

The fitted regression line is:

$$
\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i
$$

#### Meaning of $\beta_0$ (The Intercept)

$\beta_0$ is the model's expected target value $E[y \mid x = 0]$, assuming $x = 0$ lies within the meaningful domain of the problem. If $x = 0$ is outside the observed range of data (for example, predicting house price from square footage), $\beta_0$ serves as a mathematical anchor for the line rather than a physically interpretable quantity, unless the predictor is centered around its mean ($x_i - \bar{x}$).

#### Meaning of $\beta_1$ (The Slope)

$\beta_1$ represents the marginal change in predicted $y$ for a one-unit increase in $x$:

$$
\beta_1 = \frac{d\hat{y}}{dx}
$$

For example, in the fitted equation:

$$
\hat{y} = 50 + 3x
$$

increasing $x$ by $1$ unit increases the predicted value $\hat{y}$ by $3$ units.

---

### 5. Deriving the Simple Regression Coefficients

We seek the estimators $(\hat{\beta}_0, \hat{\beta}_1)$ that minimize the residual sum of squares objective function:

$$
S(\beta_0, \beta_1)
=
\sum_{i=1}^{n}
(y_i - \beta_0 - \beta_1 x_i)^2
$$

#### Step 1: Differentiate with Respect to $\beta_0$

Take the partial derivative of $S(\beta_0, \beta_1)$ with respect to $\beta_0$:

$$
\frac{\partial S}{\partial \beta_0}
=
-2\sum_{i=1}^{n}
(y_i - \beta_0 - \beta_1 x_i)
$$

At the stationary point, set the derivative to zero:

$$
\sum_{i=1}^{n}(y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i) = 0
$$

Distributing the summation yields the first normal equation:

$$
n\hat{\beta}_0 + \hat{\beta}_1\sum_{i=1}^{n} x_i = \sum_{i=1}^{n} y_i
$$

Dividing both sides by $n$ and using the sample means $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$ and $\bar{y} = \frac{1}{n}\sum_{i=1}^n y_i$:

$$
\boxed{
\hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x}
}
$$

> **Geometric Consequence:** Because $\bar{y} = \hat{\beta}_0 + \hat{\beta}_1\bar{x}$, every Ordinary Least Squares regression line with an intercept passes directly through the sample centroid $(\bar{x}, \bar{y})$.

#### Step 2: Differentiate with Respect to $\beta_1$

Now take the partial derivative of $S(\beta_0, \beta_1)$ with respect to $\beta_1$:

$$
\frac{\partial S}{\partial \beta_1}
=
-2\sum_{i=1}^{n} x_i(y_i - \beta_0 - \beta_1 x_i)
$$

Set equal to zero:

$$
\sum_{i=1}^{n} x_i(y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i) = 0
$$

Substitute $\hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x}$ into the equation:

$$
\sum_{i=1}^{n} x_i \left[ (y_i - \bar{y}) - \hat{\beta}_1(x_i - \bar{x}) \right] = 0
$$

Rearranging terms:

$$
\sum_{i=1}^{n} x_i(y_i - \bar{y}) = \hat{\beta}_1 \sum_{i=1}^{n} x_i(x_i - \bar{x})
$$

Using the algebraic identities $\sum_{i=1}^n \bar{x}(y_i - \bar{y}) = 0$ and $\sum_{i=1}^n \bar{x}(x_i - \bar{x}) = 0$, we can subtract zero from both sides to center both factors:

$$
\boxed{
\hat{\beta}_1
=
\frac{
\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})
}{
\sum_{i=1}^{n}(x_i - \bar{x})^2
}
=
\frac{S_{xy}}{S_{xx}}
}
$$

and:

$$
\boxed{
\hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x}
}
$$

#### Connection to Sample Covariance and Correlation

Dividing the numerator and denominator of $\hat{\beta}_1$ by $n - 1$ reveals the relationship between the slope, sample covariance $\text{Cov}(x, y)$, sample standard deviations $s_x, s_y$, and Pearson correlation coefficient $r_{xy}$:

$$
\hat{\beta}_1
=
\frac{\text{Cov}(x, y)}{\text{Var}(x)}
=
r_{xy} \frac{s_y}{s_x}
$$

This is the exact closed-form solution for simple $\text{OLS}$.

---

### 6. Multiple Linear Regression

For $p$ predictors, the fitted regression hyperplane is:

$$
\boxed{
\hat{y}
=
\beta_0 +
\beta_1 x_1 +
\beta_2 x_2 +
\cdots +
\beta_p x_p
}
$$

In multiple regression, the interpretation of each slope coefficient $\beta_j$ is **conditional on all other included predictors** (*ceteris paribus*):

$$
\beta_j = \frac{\partial E[y \mid x_1, \dots, x_p]}{\partial x_j}
$$

For example, consider the fitted model:

$$
\hat{y} = 100 + 5x_1 - 2x_2
$$

The coefficient $5$ means:

> Holding $x_2$ constant, a one-unit increase in $x_1$ increases the model's predicted target $\hat{y}$ by $5$ units.

This "holding other variables constant" (partial effect) interpretation is the primary reason multiple regression is far more informative than fitting separate simple regressions on each predictor individually. By the **Frisch–Waugh–Lovell ($\text{FWL}$) Theorem**, $\hat{\beta}_j$ in multiple regression is mathematically identical to regressing the residualized target on the residualized predictor $x_j$ after linearly removing the effects of all other $p - 1$ predictors from both.

---

### 7. The Design Matrix

For a multiple regression model with two predictors:

$$
y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \epsilon_i, \qquad i = 1, 2, \dots, n
$$

the design matrix $X \in \mathbb{R}^{n \times 3}$ is constructed as:

$$
X =
\begin{bmatrix}
1 & x_{11} & x_{12}\\
1 & x_{21} & x_{22}\\
\vdots & \vdots & \vdots\\
1 & x_{n1} & x_{n2}
\end{bmatrix}
$$

and the parameter vector $\beta \in \mathbb{R}^{3 \times 1}$ is:

$$
\beta =
\begin{bmatrix}
\beta_0\\
\beta_1\\
\beta_2
\end{bmatrix}
$$

Matrix-vector multiplication yields the vector of linear predictors across all $n$ observations simultaneously:

$$
X\beta
=
\begin{bmatrix}
\beta_0 + \beta_1 x_{11} + \beta_2 x_{12}\\
\beta_0 + \beta_1 x_{21} + \beta_2 x_{22}\\
\vdots\\
\beta_0 + \beta_1 x_{n1} + \beta_2 x_{n2}
\end{bmatrix}
$$

The leading column of ones $\mathbf{1}_n = [1, 1, \dots, 1]^T$ is therefore not merely cosmetic—it provides the constant basis vector required for matrix multiplication to include the intercept $\beta_0$. If the column of ones is omitted, the regression hyperplane is forced to pass through the origin $\mathbf{0}$.

---

### 8. The Statistical Model

The population data-generating process is represented by the statistical model:

$$
\boxed{
y = X\beta + \epsilon
}
$$

This equation decomposes the observed response vector $y$ into two distinct conceptual components:

1. **Systematic (Deterministic) Component:**

$$
X\beta
$$

2. **Random (Stochastic) Component:**

$$
\epsilon
$$

When the zero-conditional-mean assumption holds:

$$
E[\epsilon \mid X] = \mathbf{0}
$$

the population regression function equals the conditional expectation of $y$ given $X$:

$$
E[y \mid X] = E[X\beta + \epsilon \mid X] = X\beta + E[\epsilon \mid X] = X\beta
$$

#### Crucial Distinction Between $\epsilon$ and $e$

| Quantity | Symbol | Formula | Observability | Meaning |
| :--- | :---: | :---: | :--- | :--- |
| **Population Error** | $\epsilon$ | $\epsilon = y - X\beta$ | Unobserved | Theoretical random disturbance from the true population hyperplane. |
| **Sample Residual** | $e$ | $e = y - X\hat{\beta}$ | Observed | Empirical discrepancy between observed data and the sample-fitted hyperplane. |

---

### 9. Residuals

For each individual observation $i \in \{1, \dots, n\}$, the sample residual is:

$$
\boxed{
e_i = y_i - \hat{y}_i
}
$$

In vector notation across all $n$ observations:

$$
\boxed{
e = y - \hat{y} = y - X\hat{\beta}
}
$$

- A **positive residual** ($e_i > 0$) means the actual observed value $y_i$ lies above the model's prediction $\hat{y}_i$ (underprediction).
- A **negative residual** ($e_i < 0$) means the actual observed value $y_i$ lies below the model's prediction $\hat{y}_i$ (overprediction).

#### Algebraic Properties of OLS Residuals

When an intercept is included in $X$, the $\text{OLS}$ residual vector $e$ satisfies four exact algebraic identities in every sample:

1. **Zero Sample Sum and Mean:**

$$
\sum_{i=1}^{n} e_i = 0 \implies \bar{e} = 0
$$

2. **Orthogonality to Every Predictor:**

$$
X^T e = \mathbf{0} \implies \sum_{i=1}^{n} x_{ij} e_i = 0 \quad \forall j \in \{1, \dots, p\}
$$

3. **Orthogonality to Fitted Values:**

$$
\hat{y}^T e = (X\hat{\beta})^T e = \hat{\beta}^T (X^T e) = 0
$$

4. **Conservation of Sample Mean:**

$$
\sum_{i=1}^{n} y_i = \sum_{i=1}^{n} \hat{y}_i \implies \bar{y} = \bar{\hat{y}}
$$

---

### 10. Least Squares

The Residual Sum of Squares ($\text{SSE}$, also denoted $\text{RSS}$) is defined as:

$$
\boxed{
SSE
=
\sum_{i=1}^{n}
(y_i - \hat{y}_i)^2
=
\sum_{i=1}^{n} e_i^2
}
$$

Expressed as a function of the candidate parameter vector $\beta$ during optimization:

$$
SSE(\beta)
=
\|y - X\beta\|_2^2
=
(y - X\beta)^T(y - X\beta)
$$

#### Why Minimize Squared Errors?

1. **Avoiding Sign Cancellation:** If we attempted to minimize the raw sum $\sum_{i=1}^n (y_i - \hat{y}_i)$, large positive errors ($+100$) and large negative errors ($-100$) would cancel out to zero, allowing arbitrarily poor lines to appear optimal.
2. **Smooth Differentiability and Closed-Form Solution:** Minimizing absolute errors $\sum_{i=1}^n |y_i - \hat{y}_i|$ ($L_1$ / quantile regression) avoids sign cancellation and is more robust to outliers, but the absolute value function is non-differentiable at zero and lacks a closed-form matrix formula.
3. **Statistical Optimality:** Squaring the errors yields a smooth, strictly convex quadratic objective (when $X$ has full column rank) with a unique closed-form analytical solution that is both the Best Linear Unbiased Estimator ($\text{BLUE}$) by the Gauss–Markov Theorem and the Maximum Likelihood Estimator ($\text{MLE}$) under Gaussian noise.

---

### 11. Normal Equation — Full Derivation

Start with the squared-error objective function:

$$
J(\beta)
=
(y - X\beta)^T(y - X\beta)
$$

Expand the transpose product using $(A - B)^T = A^T - B^T$ and $(X\beta)^T = \beta^T X^T$:

$$
J(\beta)
=
(y^T - \beta^T X^T)(y - X\beta)
$$

Distribute the matrix multiplication:

$$
J(\beta)
=
y^T y
- y^T X\beta
- \beta^T X^T y
+ \beta^T X^T X\beta
$$

Notice that $y^T X\beta$ is a $1 \times 1$ scalar, and a scalar equals its own transpose:

$$
(y^T X\beta)^T = \beta^T X^T y \implies y^T X\beta = \beta^T X^T y
$$

Combining the two middle terms gives:

$$
J(\beta)
=
y^T y
- 2\beta^T X^T y
+ \beta^T X^T X\beta
$$

Apply the standard matrix calculus differentiation identities $\nabla_\beta (\beta^T a) = a$ and $\nabla_\beta (\beta^T A\beta) = 2A\beta$ (for symmetric $A = X^T X$):

$$
\nabla_\beta J(\beta)
=
-2X^T y
+
2X^T X\beta
$$

At the stationary point $\hat{\beta}$, set the gradient vector equal to the zero vector:

$$
\nabla_\beta J(\hat{\beta}) = \mathbf{0}
\implies
-2X^T y + 2X^T X\hat{\beta} = \mathbf{0}
$$

Dividing by $2$ yields the **Normal Equation**:

$$
\boxed{
X^T X\hat{\beta} = X^T y
}
$$

If $X$ has full column rank $\text{rank}(X) = p + 1$, the Gram matrix $X^T X$ is symmetric positive definite and therefore invertible, giving the unique closed-form $\text{OLS}$ estimator:

$$
\boxed{
\hat{\beta}
=
(X^T X)^{-1} X^T y
}
$$

#### Second-Order Sufficiency Condition (Hessian Verification)

To confirm that $\hat{\beta}$ is a global minimum rather than a saddle point or maximum, compute the Hessian matrix of second partial derivatives:

$$
\nabla^2_\beta J(\beta) = 2X^T X
$$

For any non-zero vector $v \in \mathbb{R}^{p+1}$, the quadratic form is:

$$
v^T (2X^T X) v = 2(Xv)^T(Xv) = 2\|Xv\|_2^2 \ge 0
$$

Thus, $2X^T X$ is always positive semi-definite (and strictly positive definite when $X$ has full column rank, since $Xv \ne \mathbf{0}$ for $v \ne \mathbf{0}$), proving that $\hat{\beta}$ is the unique global minimizer.

#### Important Numerical Note

While $(X^T X)^{-1}X^T y$ is essential for theoretical derivations, production numerical software should avoid explicitly computing the matrix inverse $(X^T X)^{-1}$. Prefer solving the linear system directly or using a least-squares decomposition solver:

```python
# Preferred linear solver when X.T @ X is well-conditioned:
beta = np.linalg.solve(X.T @ X, X.T @ y)

# Preferred general least-squares solver (uses SVD under the hood):
beta = np.linalg.lstsq(X, y, rcond=None)[0]
```

---

### 12. Geometric Interpretation

Ordinary Least Squares has an elegant geometric interpretation in $n$-dimensional Euclidean space $\mathbb{R}^n$.

The columns of the design matrix $X = [x_{\cdot 0}, x_{\cdot 1}, \dots, x_{\cdot p}]$ span a $(p+1)$-dimensional subspace known as the **column space** $\text{Col}(X) \subset \mathbb{R}^n$. Every possible prediction vector $\hat{y} = X\beta$ is a linear combination of the columns of $X$ and must therefore lie inside $\text{Col}(X)$.

Because the observed target vector $y \in \mathbb{R}^n$ generally does not lie entirely within $\text{Col}(X)$ due to random noise, minimizing the Euclidean distance $\|y - \hat{y}\|_2$ is equivalent to finding the **orthogonal projection** of $y$ onto $\text{Col}(X)$:

$$
\hat{y} = Hy
$$

where $H$ is the orthogonal projection matrix (the **Hat Matrix**):

$$
\boxed{
H = X(X^T X)^{-1} X^T
}
$$

The residual vector $e = y - \hat{y} = (I - H)y$ represents the perpendicular drop from $y$ onto the subspace $\text{Col}(X)$. At the $\text{OLS}$ solution:

$$
X^T e = \mathbf{0} \implies e \perp \text{Col}(X)
$$

This means the residual vector $e$ is orthogonal to every column of $X$ and hence orthogonal to the fitted vector $\hat{y}$.

#### Orthogonal Decomposition Summary

| Vector / Relationship | Mathematical Expression | Geometric Role in $\mathbb{R}^n$ |
| :--- | :---: | :--- |
| **Target Vector** | $y \in \mathbb{R}^n$ | Hypotenuse of the right triangle in observation space. |
| **Fitted Projection** | $\hat{y} = Hy \in \text{Col}(X)$ | Orthogonal shadow of $y$ cast onto the model subspace $\text{Col}(X)$. |
| **Residual Vector** | $e = (I - H)y \in \text{Col}(X)^\perp$ | Perpendicular error vector orthogonal to $\text{Col}(X)$. |
| **Pythagorean Theorem** | $\|y\|_2^2 = \|\hat{y}\|_2^2 + \|e\|_2^2$ | Holds exactly because $\hat{y}^T e = 0$ ($\hat{y} \perp e$). |

---

### 13. Worked Numerical Example

Consider the following dataset of $n = 5$ observations:

$$
x = [1, 2, 3, 4, 5]^T, \qquad y = [3, 5, 4, 8, 7]^T
$$

We fit the simple linear regression model:

$$
\hat{y} = \beta_0 + \beta_1 x
$$

#### Step 1 — Calculate Sample Means

$$
\bar{x} = \frac{1 + 2 + 3 + 4 + 5}{5} = 3
$$

$$
\bar{y} = \frac{3 + 5 + 4 + 8 + 7}{5} = 5.4
$$

#### Step 2 — Calculate Centered Deviations and Products

| $i$ | $x_i$ | $y_i$ | $x_i - \bar{x}$ | $y_i - \bar{y}$ | $(x_i - \bar{x})(y_i - \bar{y})$ | $(x_i - \bar{x})^2$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $1$ | $1$ | $3$ | $-2$ | $-2.4$ | $4.8$ | $4$ |
| $2$ | $2$ | $5$ | $-1$ | $-0.4$ | $0.4$ | $1$ |
| $3$ | $3$ | $4$ | $0$ | $-1.4$ | $0.0$ | $0$ |
| $4$ | $4$ | $8$ | $1$ | $2.6$ | $2.6$ | $1$ |
| $5$ | $5$ | $7$ | $2$ | $1.6$ | $3.2$ | $4$ |
| **Total** | **$15$** | **$27$** | **$0$** | **$0.0$** | **$S_{xy} = 11.0$** | **$S_{xx} = 10$** |

Therefore:

$$
S_{xy} = 11.0, \qquad S_{xx} = 10
$$

#### Step 3 — Compute Slope $\hat{\beta}_1$

$$
\hat{\beta}_1
=
\frac{S_{xy}}{S_{xx}}
=
\frac{11}{10}
=
\boxed{1.1}
$$

#### Step 4 — Compute Intercept $\hat{\beta}_0$

$$
\hat{\beta}_0
=
\bar{y} - \hat{\beta}_1\bar{x}
=
5.4 - (1.1)(3)
=
5.4 - 3.3
=
\boxed{2.1}
$$

Therefore, the fitted regression equation is:

$$
\boxed{
\hat{y} = 2.1 + 1.1x
}
$$

#### Step 5 — Compute Predictions and Residuals

| $x_i$ | $y_i$ | $\hat{y}_i = 2.1 + 1.1x_i$ | Residual $e_i = y_i - \hat{y}_i$ | Squared Residual $e_i^2$ |
| :---: | :---: | :---: | :---: | :---: |
| $1$ | $3$ | $3.2$ | $-0.2$ | $0.04$ |
| $2$ | $5$ | $4.3$ | $0.7$ | $0.49$ |
| $3$ | $4$ | $5.4$ | $-1.4$ | $1.96$ |
| $4$ | $8$ | $6.5$ | $1.5$ | $2.25$ |
| $5$ | $7$ | $7.6$ | $-0.6$ | $0.36$ |
| **Sum** | **$27$** | **$27.0$** | **$0.0$** | **$5.10$** |

Notice that $\sum_{i=1}^5 e_i = -0.2 + 0.7 - 1.4 + 1.5 - 0.6 = 0.0$, confirming the exact orthogonality property of $\text{OLS}$ residuals.

#### Step 6 — Compute SSE

$$
SSE
=
(-0.2)^2 + (0.7)^2 + (-1.4)^2 + (1.5)^2 + (-0.6)^2
=
0.04 + 0.49 + 1.96 + 2.25 + 0.36
=
\boxed{5.10}
$$

#### Step 7 — Compute Prediction MSE vs. Unbiased Error Variance $\hat{\sigma}^2$

There are $n = 5$ observations. For descriptive prediction error across the sample:

$$
MSE = \frac{SSE}{n} = \frac{5.10}{5} = \boxed{1.02}
$$

For an unbiased estimator of the population error variance $\sigma^2$ in statistical inference, we divide by the residual degrees of freedom $df = n - p - 1$:

$$
\hat{\sigma}^2
=
\frac{SSE}{n - p - 1}
$$

Here $p = 1$, so $df = 5 - 1 - 1 = 3$:

$$
\hat{\sigma}^2
=
\frac{5.10}{3}
=
\boxed{1.70}
$$

This distinction is critical in practice:
- **Predictive evaluation metrics** typically define $\text{MSE} = \frac{SSE}{n}$.
- **Classical statistical inference** defines the unbiased residual variance estimator as $\hat{\sigma}^2 = \frac{SSE}{n - p - 1}$.

---

## Part II: Evaluation Metrics & Statistical Inference

### 14. Regression Metrics

#### Mean Absolute Error (MAE)

$$
\boxed{
MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
}
$$

For our worked numerical example:

$$
MAE = \frac{0.2 + 0.7 + 1.4 + 1.5 + 0.6}{5} = \frac{4.4}{5} = \boxed{0.88}
$$

#### Mean Squared Error (MSE)

$$
\boxed{
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
}
$$

For the worked example:

$$
MSE = \boxed{1.02}
$$

#### Root Mean Squared Error (RMSE)

$$
\boxed{
RMSE = \sqrt{MSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
}
$$

For the worked example:

$$
RMSE = \sqrt{1.02} \approx \boxed{1.01}
$$

#### Mean Absolute Percentage Error (MAPE)

When $y_i \ne 0$ for all observations, relative scale-free error can be measured via $\text{MAPE}$:

$$
\boxed{
MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|
}
$$

#### Comparison of Regression Metrics

| Metric | Units | Outlier Sensitivity | Key Interpretation |
| :--- | :--- | :--- | :--- |
| **$\text{MAE}$** | Target units ($y$) | Low (Linear penalty) | Average absolute magnitude of prediction error. |
| **$\text{MSE}$** | Squared target units ($y^2$) | High (Quadratic penalty) | Differentiable squared loss directly minimized by $\text{OLS}$. |
| **$\text{RMSE}$** | Target units ($y$) | High (Quadratic penalty) | Standard deviation-like error metric in the original scale of $y$. |
| **$\text{MAPE}$** | Percentage ($\%$) | Asymmetric near $y_i \to 0$ | Scale-independent relative percentage error. |
| **$R^2$** | Dimensionless | High | Proportion of target variance explained relative to the mean baseline $\bar{y}$. |

---

### 15. $R^2$ and Adjusted $R^2$

We partition the total variation in the target variable $y$ into two orthogonal components.

1. **Total Sum of Squares ($\text{SST}$):** Variation of observed $y_i$ around its sample mean $\bar{y}$:

$$
\boxed{
SST = \sum_{i=1}^{n}(y_i - \bar{y})^2
}
$$

2. **Regression Sum of Squares ($\text{SSR}$):** Variation of fitted predictions $\hat{y}_i$ around the sample mean $\bar{y}$:

$$
\boxed{
SSR = \sum_{i=1}^{n}(\hat{y}_i - \bar{y})^2
}
$$

3. **Residual Sum of Squares ($\text{SSE}$):** Unexplained variation of $y_i$ around the fitted predictions $\hat{y}_i$:

$$
\boxed{
SSE = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2
}
$$

#### Proof of the Variance Decomposition Identity ($SST = SSR + SSE$)

When the model includes an intercept, write $y_i - \bar{y} = (\hat{y}_i - \bar{y}) + (y_i - \hat{y}_i) = (\hat{y}_i - \bar{y}) + e_i$. Squaring and summing over $i = 1, \dots, n$:

$$
\sum_{i=1}^{n}(y_i - \bar{y})^2
=
\sum_{i=1}^{n}(\hat{y}_i - \bar{y})^2
+
2\sum_{i=1}^{n}(\hat{y}_i - \bar{y})e_i
+
\sum_{i=1}^{n} e_i^2
$$

Because $\sum_{i=1}^n \hat{y}_i e_i = 0$ and $\bar{y}\sum_{i=1}^n e_i = 0$, the cross-product term vanishes identically, yielding:

$$
\boxed{
SST = SSR + SSE
}
$$

Therefore, the **Coefficient of Determination ($R^2$)** is:

$$
\boxed{
R^2
=
\frac{SSR}{SST}
=
1 - \frac{SSE}{SST}
}
$$

In our worked numerical example ($SST = 17.20$, $SSR = 12.10$, $SSE = 5.10$):

$$
R^2 = 1 - \frac{5.10}{17.20} = \frac{12.10}{17.20} \approx \boxed{0.7035}
$$

#### Adjusted $R^2$

Adding any new predictor to an $\text{OLS}$ model—even pure random noise—can never increase $SSE$ on the training data, meaning $R^2$ is monotonically non-decreasing with $p$. **Adjusted $R^2$** divides both $SSE$ and $SST$ by their respective degrees of freedom to penalize unnecessary predictors:

$$
\boxed{
R^2_{adj}
=
1 - \frac{SSE / (n - p - 1)}{SST / (n - 1)}
=
1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}
}
$$

#### Information Criteria: AIC and BIC

Beyond $R^2_{adj}$, model selection among candidate feature subsets is often guided by the **Akaike Information Criterion ($\text{AIC}$)** and **Bayesian Information Criterion ($\text{BIC}$)**, which balance goodness-of-fit against model complexity $k = p + 1$:

$$
\text{AIC} = n \ln\left(\frac{SSE}{n}\right) + 2k, \qquad \text{BIC} = n \ln\left(\frac{SSE}{n}\right) + k \ln(n)
$$

Lower values of $\text{AIC}$ and $\text{BIC}$ indicate a superior balance between model fit and parsimony.

---

### 16. Degrees of Freedom

Degrees of freedom quantify the number of independent pieces of information remaining in the sample after estimating model parameters.

For a dataset with $n$ observations and $p$ predictors plus an intercept ($p + 1$ total parameters):

$$
\boxed{
df_{total} = n - 1, \qquad df_{model} = p, \qquad df_{residual} = n - p - 1
}
$$

Geometrically, the response vector $y$ lives in $\mathbb{R}^n$, the fitted projection $\hat{y}$ is constrained to a $(p+1)$-dimensional subspace $\text{Col}(X)$, and the residual vector $e = (I - H)y$ is constrained to the orthogonal complement $\text{Col}(X)^\perp$, which has dimension:

$$
\dim\left(\text{Col}(X)^\perp\right) = \text{tr}(I_n - H) = n - (p + 1) = n - p - 1
$$

This subspace dimension is the exact mathematical reason why the unbiased residual variance estimator divides $SSE$ by $n - p - 1$ rather than $n$.

---

### 17. Error Variance

Under the homoscedastic spherical error assumption:

$$
\text{Var}(\epsilon \mid X) = \sigma^2 I_n
$$

the unknown population error variance $\sigma^2$ is estimated unbiasedly by the Residual Mean Square:

$$
\boxed{
\hat{\sigma}^2 = \frac{SSE}{n - p - 1}
}
$$

Because $E[SSE \mid X] = \sigma^2 \text{tr}(I_n - H) = (n - p - 1)\sigma^2$, taking the expectation confirms unbiasedness:

$$
E[\hat{\sigma}^2 \mid X] = \frac{(n - p - 1)\sigma^2}{n - p - 1} = \sigma^2
$$

The square root is the **Residual Standard Error ($\text{RSE}$)**:

$$
\boxed{
\hat{\sigma} = \sqrt{\frac{SSE}{n - p - 1}}
}
$$

---

### 18. Standard Error — Derivation

Start from the closed-form $\text{OLS}$ estimator:

$$
\hat{\beta} = (X^T X)^{-1} X^T y
$$

Substitute the population model $y = X\beta + \epsilon$:

$$
\hat{\beta} = (X^T X)^{-1} X^T (X\beta + \epsilon)
$$

Distribute the matrix product:

$$
\hat{\beta} = (X^T X)^{-1} X^T X\beta + (X^T X)^{-1} X^T \epsilon
$$

Since $(X^T X)^{-1}(X^T X) = I_{p+1}$, this simplifies directly to the fundamental sampling error representation:

$$
\boxed{
\hat{\beta} = \beta + (X^T X)^{-1} X^T \epsilon
}
$$

Therefore, the estimation error vector is linear in $\epsilon$:

$$
\hat{\beta} - \beta = (X^T X)^{-1} X^T \epsilon
$$

Under $\text{Var}(\epsilon \mid X) = \sigma^2 I_n$, the conditional variance–covariance matrix of $\hat{\beta}$ is:

$$
\boxed{
\text{Var}(\hat{\beta} \mid X) = \sigma^2 (X^T X)^{-1}
}
$$

Replacing the unknown $\sigma^2$ with its unbiased estimator $\hat{\sigma}^2 = \frac{SSE}{n - p - 1}$ yields the estimated covariance matrix:

$$
\boxed{
\widehat{\text{Var}}(\hat{\beta}) = \hat{\sigma}^2 (X^T X)^{-1}
}
$$

The **Standard Error ($\text{SE}$)** of the $j$-th coefficient $\hat{\beta}_j$ is the square root of the $j$-th diagonal element of this matrix:

$$
\boxed{
SE(\hat{\beta}_j)
=
\sqrt{
\hat{\sigma}^2
\left[(X^T X)^{-1}\right]_{jj}
}
}
$$

---

### 19. $t$-Test and $p$-Value

To test the null hypothesis that the $j$-th population coefficient equals a hypothesized value $\beta_{j,0}$:

$$
H_0: \beta_j = \beta_{j,0} \quad \text{vs.} \quad H_1: \beta_j \ne \beta_{j,0}
$$

we compute the studentized $t$-statistic:

$$
\boxed{
t
=
\frac{\hat{\beta}_j - \beta_{j,0}}{SE(\hat{\beta}_j)}
\sim t_{n - p - 1} \quad (\text{under } H_0)
}
$$

For the standard significance test of whether predictor $x_j$ has no linear effect given the other predictors ($H_0: \beta_j = 0$), the statistic reduces to:

$$
\boxed{
t = \frac{\hat{\beta}_j}{SE(\hat{\beta}_j)}
}
$$

#### Numerical Example

Suppose a multiple regression model with $n = 30$ observations and $p = 2$ predictors yields:

$$
\hat{\beta}_1 = 8, \qquad SE(\hat{\beta}_1) = 2
$$

Then the $t$-statistic is:

$$
t = \frac{8}{2} = 4.0
$$

The residual degrees of freedom are:

$$
df = n - p - 1 = 30 - 2 - 1 = 27
$$

The two-sided $p$-value is computed from the Student's $t$-distribution with $27$ degrees of freedom:

$$
p\text{-value} = 2 P(T_{27} \ge |4.0|) \approx 0.00044
$$

#### Correct Interpretation of the $p$-Value

A $p$-value is the probability, assuming the null hypothesis $H_0$ and all underlying model assumptions hold true, of observing a test statistic at least as extreme as the value calculated from the sample.

It is **not**:
- the probability that the null hypothesis $H_0$ is true, $P(H_0 \mid \text{Data})$;
- a direct measure of practical or economic effect size;
- proof that the causal specification of the model is valid.

---

### 20. Confidence Intervals and Prediction Intervals

#### Confidence Interval for a Coefficient $\beta_j$

A $100(1 - \alpha)\%$ confidence interval for parameter $\beta_j$ is:

$$
\boxed{
\hat{\beta}_j \pm t_{1 - \alpha/2, \, df} \cdot SE(\hat{\beta}_j)
}
$$

**Example:** Suppose $\hat{\beta}_1 = 10$, $SE(\hat{\beta}_1) = 2$, and the critical value is $t_{0.975, \, df} = 2.06$. Then:

$$
10 \pm (2.06)(2) = 10 \pm 4.12 \implies \boxed{[5.88, \; 14.12]}
$$

Because $0 \notin [5.88, 14.12]$, we reject $H_0: \beta_1 = 0$ at the $\alpha = 0.05$ significance level.

#### Confidence Interval for the Mean Response vs. Prediction Interval for a New Observation

Given a new feature vector $x_* \in \mathbb{R}^{p+1}$ (including the leading $1$), the point prediction is $\hat{y}_* = x_*^T \hat{\beta}$. However, there are two distinct uncertainty intervals around $\hat{y}_*$:

1. **Confidence Interval for the Expected Mean Response $E[y_* \mid x_*] = x_*^T \beta$:**
   Measures only the uncertainty in estimating the regression surface $X\hat{\beta}$:

$$
\boxed{
\hat{y}_* \pm t_{1 - \alpha/2, \, n-p-1} \, \hat{\sigma} \sqrt{x_*^T (X^T X)^{-1} x_*}
}
$$

2. **Prediction Interval for an Individual Future Observation $y_* = x_*^T \beta + \epsilon_*$:**
   Must account for both the uncertainty in the regression surface **plus** the irreducible variance $\sigma^2$ of the new observation's random shock $\epsilon_*$:

$$
\boxed{
\hat{y}_* \pm t_{1 - \alpha/2, \, n-p-1} \, \hat{\sigma} \sqrt{1 + x_*^T (X^T X)^{-1} x_*}
}
$$

Because of the extra $1$ inside the square root, a **Prediction Interval is always wider than a Confidence Interval** at the same point $x_*$.

---

### 21. $z$-Test

The $z$-statistic takes the standardized form:

$$
\boxed{
z = \frac{\hat{\theta} - \theta_0}{SE(\hat{\theta})}
}
$$

A $z$-distribution ($N(0, 1)$) reference is used when either:
1. The true population error variance $\sigma^2$ is known exactly *a priori*, or
2. The sample size $n$ is large enough ($n - p - 1 \to \infty$) that by the Central Limit Theorem and Slutsky's Theorem, the $t_{n-p-1}$ distribution converges in distribution to the standard normal distribution $N(0, 1)$.

In finite samples with unknown error variance $\sigma^2$, the $t$-distribution should always be preferred because it accounts for the heavier tails introduced by estimating $\hat{\sigma}^2$.

---

### 22. $F$-Test

#### Global (Omnibus) $F$-Test

The overall regression $F$-test evaluates the joint null hypothesis that **all slope coefficients are simultaneously zero**:

$$
H_0: \beta_1 = \beta_2 = \cdots = \beta_p = 0 \quad \text{vs.} \quad H_1: \exists j \in \{1, \dots, p\} \text{ such that } \beta_j \ne 0
$$

The global $F$-statistic compares the explained mean square to the residual mean square:

$$
\boxed{
F
=
\frac{SSR / p}{SSE / (n - p - 1)}
=
\frac{R^2 / p}{(1 - R^2) / (n - p - 1)}
\sim F_{p, \, n - p - 1}
}
$$

with numerator degrees of freedom $df_1 = p$ and denominator degrees of freedom $df_2 = n - p - 1$.

For a single predictor ($p = 1$) or a single linear restriction, the $F$-statistic equals the square of the corresponding $t$-statistic:

$$
\boxed{
F_{1, \, n-2} = t_{n-2}^2
}
$$

#### Partial $F$-Test for Nested Models

To test whether a subset of $q$ predictors can be excluded from a full model ($p$ predictors) to form a reduced model ($p - q$ predictors):

$$
\boxed{
F
=
\frac{(SSE_{reduced} - SSE_{full}) / q}{SSE_{full} / (n - p - 1)}
\sim F_{q, \, n - p - 1}
}
$$

---

### 23. Regression Assumptions

The classical linear regression assumptions govern when $\text{OLS}$ is unbiased, efficient, and valid for exact finite-sample inference:

#### 23.1 Linearity in Parameters
The conditional expectation of $y$ is a linear function of the parameter vector $\beta$:

$$
E[y \mid X] = X\beta
$$

Polynomial features ($x^2, x^3$), logarithmic transforms ($\ln x$), and interaction terms ($x_1 x_2$) satisfy this assumption because the model remains linear in $\beta$.

#### 23.2 Strict Exogeneity
The conditional mean of the errors given the design matrix is zero:

$$
\boxed{
E[\epsilon \mid X] = \mathbf{0}
}
$$

This implies $E[\epsilon_i] = 0$ and $\text{Cov}(x_{ij}, \epsilon_i) = 0$. Violating exogeneity (due to omitted variables, measurement error in $X$, or simultaneous causality) induces **omitted variable bias** and inconsistency in $\hat{\beta}$.

#### 23.3 Independence (No Autocorrelation)
Conditional on $X$, errors across distinct observations are uncorrelated:

$$
\text{Cov}(\epsilon_i, \epsilon_j \mid X) = 0 \quad \forall i \ne j
$$

Time-series data and spatially or hierarchically clustered observations frequently violate this assumption, requiring $\text{GLS}$, Newey–West standard errors, or cluster-robust inference.

#### 23.4 Homoscedasticity
Every error term has constant conditional variance across all observations:

$$
\boxed{
\text{Var}(\epsilon_i \mid X) = \sigma^2 \quad \forall i \in \{1, \dots, n\}
}
$$

Combined with independence, this gives the spherical covariance matrix $\text{Var}(\epsilon \mid X) = \sigma^2 I_n$. When $\text{Var}(\epsilon_i \mid X) = \sigma_i^2$ varies across observations, **heteroscedasticity** is present.

#### 23.5 Normality of Errors
For exact finite-sample $t$-tests, $F$-tests, and confidence intervals:

$$
\boxed{
\epsilon \mid X \sim N(\mathbf{0}, \sigma^2 I_n)
}
$$

Normality is **not required** to compute $\hat{\beta}$, nor is it required for unbiasedness or the Gauss–Markov theorem.

#### 23.6 Full Column Rank (No Perfect Multicollinearity)
No predictor column in $X$ is an exact linear combination of the other columns:

$$
\text{rank}(X) = p + 1 \iff \det(X^T X) \ne 0
$$

---

### 24. Gauss–Markov Theorem, BLUE, and Maximum Likelihood

#### The Gauss–Markov Theorem

Under the classical Gauss–Markov assumptions—linearity ($y = X\beta + \epsilon$), full rank ($\text{rank}(X) = p+1$), strict exogeneity ($E[\epsilon \mid X] = \mathbf{0}$), and spherical errors ($\text{Var}(\epsilon \mid X) = \sigma^2 I_n$)—the $\text{OLS}$ estimator $\hat{\beta} = (X^T X)^{-1} X^T y$ is:

$$
\boxed{\text{BLUE: Best Linear Unbiased Estimator}}
$$

- **Linear:** $\hat{\beta} = Cy$ is a linear function of the target vector $y$, where $C = (X^T X)^{-1} X^T$.
- **Unbiased:** $E[\hat{\beta} \mid X] = \beta$.
- **Best (Minimum Variance):** For any other linear unbiased estimator $\tilde{\beta} = \tilde{C}y$, the covariance difference matrix $\text{Var}(\tilde{\beta} \mid X) - \text{Var}(\hat{\beta} \mid X)$ is positive semi-definite. Consequently, for every individual component $j$, $\text{Var}(\hat{\beta}_j \mid X) \le \text{Var}(\tilde{\beta}_j \mid X)$.

> **Scope Note:** The Gauss–Markov theorem applies strictly to the class of *linear unbiased* estimators. Biased estimators such as Ridge Regression can achieve lower Mean Squared Error ($\text{MSE} = \text{Variance} + \text{Bias}^2$) when multicollinearity is high.

#### Equivalence of OLS and Maximum Likelihood Estimation (MLE)

If we additionally assume Gaussian errors $\epsilon \mid X \sim N(\mathbf{0}, \sigma^2 I_n)$, then $y \mid X \sim N(X\beta, \sigma^2 I_n)$. The log-likelihood function for $(\beta, \sigma^2)$ is:

$$
\ell(\beta, \sigma^2)
=
-\frac{n}{2}\ln(2\pi)
-
\frac{n}{2}\ln(\sigma^2)
-
\frac{1}{2\sigma^2}\|y - X\beta\|_2^2
$$

Maximizing $\ell(\beta, \sigma^2)$ with respect to $\beta$ is identical to minimizing $\|y - X\beta\|_2^2$. Therefore:

$$
\boxed{
\hat{\beta}_{MLE} = \hat{\beta}_{OLS} = (X^T X)^{-1} X^T y
}
$$

Meanwhile, differentiating $\ell(\beta, \sigma^2)$ with respect to $\sigma^2$ yields $\hat{\sigma}^2_{MLE} = \frac{SSE}{n}$, which is slightly downward biased in finite samples compared to the unbiased estimator $\hat{\sigma}^2_{OLS} = \frac{SSE}{n - p - 1}$.

---

### 25. Unbiasedness — Derivation

Start with the $\text{OLS}$ estimator:

$$
\hat{\beta} = (X^T X)^{-1} X^T y
$$

Substitute the true model $y = X\beta + \epsilon$:

$$
\hat{\beta} = \beta + (X^T X)^{-1} X^T \epsilon
$$

Take the conditional expectation given $X$:

$$
E[\hat{\beta} \mid X]
=
E[\beta \mid X] + E\left[(X^T X)^{-1} X^T \epsilon \,\middle|\, X\right]
$$

Because $(X^T X)^{-1} X^T$ is deterministic conditional on $X$, it factors outside the expectation:

$$
E[\hat{\beta} \mid X]
=
\beta + (X^T X)^{-1} X^T E[\epsilon \mid X]
$$

Invoking strict exogeneity $E[\epsilon \mid X] = \mathbf{0}$:

$$
\boxed{
E[\hat{\beta} \mid X] = \beta
}
$$

By the Law of Iterated Expectations, unconditional unbiasedness follows immediately: $E[\hat{\beta}] = E\left[E[\hat{\beta} \mid X]\right] = \beta$.

---

### 26. Variance of OLS — Derivation and Robust Standard Errors

From the sampling error equation:

$$
\hat{\beta} - \beta = (X^T X)^{-1} X^T \epsilon
$$

define the $(p+1) \times n$ matrix $A = (X^T X)^{-1} X^T$. Then $\hat{\beta} - \beta = A\epsilon$. By the matrix covariance transformation property:

$$
\text{Var}(\hat{\beta} \mid X)
=
A \, \text{Var}(\epsilon \mid X) \, A^T
$$

Under homoscedasticity and independence, $\text{Var}(\epsilon \mid X) = \sigma^2 I_n$:

$$
\text{Var}(\hat{\beta} \mid X)
=
\sigma^2 A A^T
=
\sigma^2 (X^T X)^{-1} X^T X (X^T X)^{-1}
$$

Since $(X^T X)^{-1}(X^T X) = I_{p+1}$, this simplifies to:

$$
\boxed{
\text{Var}(\hat{\beta} \mid X) = \sigma^2 (X^T X)^{-1}
}
$$

#### Heteroscedasticity-Consistent (White / Sandwich) Robust Standard Errors

When heteroscedasticity is present ($\text{Var}(\epsilon \mid X) = \Omega = \text{diag}(\sigma_1^2, \dots, \sigma_n^2)$), $\hat{\beta}_{OLS}$ remains unbiased, but $\sigma^2(X^T X)^{-1}$ is no longer valid. The true covariance takes the **sandwich form**:

$$
\text{Var}(\hat{\beta} \mid X) = (X^T X)^{-1} \left( X^T \Omega X \right) (X^T X)^{-1}
$$

Halbert White's ($\text{HC0}$ / $\text{HC1}$) heteroscedasticity-consistent covariance estimator replaces $\Omega$ with the squared observed residuals $\hat{\Omega} = \text{diag}(e_1^2, e_2^2, \dots, e_n^2)$:

$$
\boxed{
\widehat{\text{Var}}_{HC1}(\hat{\beta})
=
\frac{n}{n - p - 1}
(X^T X)^{-1}
\left( \sum_{i=1}^{n} e_i^2 x_i x_i^T \right)
(X^T X)^{-1}
}
$$

---

## Part III: Diagnostics, Multicollinearity & Influential Observations

### 27. Multicollinearity

**Multicollinearity** occurs when two or more predictors in $X$ share strong overlapping linear relationships, such as:

$$
x_2 \approx 2x_1 + 0.5x_3
$$

#### Consequences of High Multicollinearity

- **Inflated Standard Errors:** As columns of $X$ become nearly collinear, the smallest eigenvalue $\lambda_{\min}(X^T X) \to 0$, causing the diagonal entries of $(X^T X)^{-1}$ to grow very large.
- **Unstable Coefficient Estimates:** Small perturbations in the training data cause large swings in $\hat{\beta}$, including sign reversals.
- **Wide Confidence Intervals and Low $t$-Statistics:** Individual $p$-values may appear non-significant even when the overall $F$-test and $R^2$ are highly significant.
- **Unimpaired In-Distribution Predictions:** Critically, collinearity among features does not bias $\hat{y} = X\hat{\beta}$ as long as new test points follow the same collinear structure.

---

### 28. Variance Inflation Factor (VIF)

To quantify the severity of multicollinearity for predictor $j \in \{1, \dots, p\}$, regress $x_j$ on all remaining $p - 1$ predictors and compute the auxiliary coefficient of determination $R_j^2$.

The **Variance Inflation Factor ($\text{VIF}_j$)** is defined as:

$$
\boxed{
VIF_j = \frac{1}{1 - R_j^2}
}
$$

Explicitly, the variance of $\hat{\beta}_j$ can be written in terms of $\text{VIF}_j$ as:

$$
\text{Var}(\hat{\beta}_j \mid X)
=
\frac{\sigma^2}{\sum_{i=1}^{n}(x_{ij} - \bar{x}_j)^2} \cdot VIF_j
$$

Thus, $\text{VIF}_j$ measures the exact multiplicative factor by which $\text{Var}(\hat{\beta}_j)$ is inflated compared to an ideal setting where predictor $x_j$ is completely orthogonal to all other predictors ($R_j^2 = 0 \implies VIF_j = 1$).

#### Numerical Example and Diagnostic Thresholds

If $R_j^2 = 0.80$, then:

$$
VIF_j = \frac{1}{1 - 0.80} = \frac{1}{0.20} = 5.0
$$

| $\text{VIF}_j$ Value | Auxiliary $R_j^2$ | Practical Diagnostic Interpretation |
| :---: | :---: | :--- |
| $VIF_j = 1$ | $0.00$ | Complete orthogonality; zero collinearity inflation. |
| $1 < VIF_j < 5$ | $0.00 \text{ to } 0.80$ | Mild to moderate correlation; generally acceptable. |
| $5 \le VIF_j < 10$ | $0.80 \text{ to } 0.90$ | High multicollinearity; interpret individual coefficients with caution. |
| $VIF_j \ge 10$ | $\ge 0.90$ | Severe multicollinearity; consider Ridge regularization, feature removal, or PCA. |

---

### 29. Residual Diagnostics

A regression model must never be evaluated by summary metrics alone (as demonstrated by **Anscombe's Quartet**, where four vastly different datasets share identical $\hat{\beta}$, $MSE$, and $R^2$). Comprehensive residual diagnostics include:

| Diagnostic Tool | Axes / Formulation | Target Assumption | Pathology Signature |
| :--- | :--- | :--- | :--- |
| **Residuals vs. Fitted Plot** | $e_i$ vs. $\hat{y}_i$ | Linearity & Homoscedasticity | Curved U-shape indicates missing nonlinear terms; funnel/cone shape indicates heteroscedasticity. |
| **Normal Q–Q Plot** | Standardized $e_i$ quantiles vs. $N(0,1)$ quantiles | Normality of $\epsilon$ | S-curves or heavy tail deviations from the $45^\circ$ reference line indicate skewness or kurtosis. |
| **Scale–Location Plot** | $\sqrt{|r_i|}$ vs. $\hat{y}_i$ | Homoscedasticity | Upward or downward sloping trend line signals non-constant error variance. |
| **Residuals vs. Predictor Plot** | $e_i$ vs. $x_{ij}$ | Functional Form | Systematic patterns reveal unmodeled nonlinearities in specific predictors. |
| **Durbin–Watson Statistic** | $DW = \frac{\sum_{t=2}^n (e_t - e_{t-1})^2}{\sum_{t=1}^n e_t^2} \in [0, 4]$ | Serial Independence | $DW \approx 2$ indicates zero autocorrelation; $DW \ll 2$ indicates positive serial correlation. |

---

### 30. Hat Matrix and Leverage

The **Hat Matrix** $H \in \mathbb{R}^{n \times n}$ maps the observed response vector $y$ directly to the fitted vector $\hat{y}$ ("putting the hat on $y$"):

$$
\boxed{
H = X(X^T X)^{-1} X^T, \qquad \hat{y} = Hy
}
$$

#### Key Algebraic Properties of $H$ and $M = I_n - H$

1. **Symmetry:** $H^T = H$ and $M^T = M$.
2. **Idempotence (Projection Property):** $H^2 = H$ and $M^2 = M$.
3. **Orthogonality:** $HM = H(I_n - H) = H - H^2 = \mathbf{0}$.
4. **Trace and Rank:** $\text{tr}(H) = \text{rank}(H) = p + 1$.

The $i$-th diagonal entry of $H$, denoted $h_{ii}$, is the **leverage** of observation $i$:

$$
h_{ii} = x_i^T (X^T X)^{-1} x_i
$$

When an intercept is included, leverage satisfies the bounds and sum identity:

$$
\frac{1}{n} \le h_{ii} \le 1, \qquad \sum_{i=1}^{n} h_{ii} = p + 1, \qquad \bar{h} = \frac{p + 1}{n}
$$

Observations with $h_{ii} > \frac{2(p + 1)}{n}$ (or $\frac{3(p + 1)}{n}$) lie far from the center of the predictor space and possess **high leverage**—meaning they have the potential to pull the regression hyperplane strongly toward themselves. Because $\text{Var}(e_i \mid X) = \sigma^2(1 - h_{ii})$, high-leverage points naturally force their own raw residuals $e_i$ to be small!

---

### 31. Cook's Distance

To account for the fact that high-leverage points shrink their own raw residuals, we first define the **internally studentized residual** $r_i$:

$$
r_i = \frac{e_i}{\hat{\sigma}\sqrt{1 - h_{ii}}}
$$

**Cook's Distance ($D_i$)** combines the studentized residual size $r_i^2$ with the leverage ratio $\frac{h_{ii}}{1 - h_{ii}}$ to measure how much the entire vector of fitted values $\hat{y}$ changes if observation $i$ is deleted from the training set ($\hat{y}_{(-i)}$):

$$
\boxed{
D_i
=
\frac{\|\hat{y} - \hat{y}_{(-i)}\|_2^2}{(p + 1)\hat{\sigma}^2}
=
\frac{e_i^2}{(p + 1)\hat{\sigma}^2}
\frac{h_{ii}}{(1 - h_{ii})^2}
=
\frac{r_i^2}{p + 1} \left( \frac{h_{ii}}{1 - h_{ii}} \right)
}
$$

Common diagnostic thresholds for flagging influential points are $D_i > \frac{4}{n}$ or $D_i > 1$. High Cook's Distance is an investigative signal—not an automatic deletion rule.

---

## Part IV: Optimization, Numerical Stability & Extended Estimators

### 32. Gradient Descent

When the number of features $p$ is extremely large (for instance, $p > 10^5$), forming and factoring the $(p+1) \times (p+1)$ matrix $X^T X$ in $\mathcal{O}(np^2 + p^3)$ time becomes computationally prohibitive. First-order iterative optimization solves linear regression in $\mathcal{O}(np)$ per iteration.

Define the scaled half-mean-squared-error objective function:

$$
\boxed{
J(\beta)
=
\frac{1}{2n}
\|X\beta - y\|_2^2
}
$$

Gradient Descent iteratively updates the parameter vector in the direction opposite to the gradient:

$$
\boxed{
\beta^{(t+1)}
=
\beta^{(t)}
-
\alpha \nabla J(\beta^{(t)})
}
$$

where $\alpha > 0$ is the learning rate (step size).

| Variant | Batch Size per Update | Update Cost | Convergence Behavior |
| :--- | :---: | :---: | :--- |
| **Batch Gradient Descent** | All $n$ samples | $\mathcal{O}(np)$ | Deterministic, smooth trajectory to the exact minimum. |
| **Mini-Batch Gradient Descent** | $B \in [32, 512]$ samples | $\mathcal{O}(Bp)$ | Vectorized hardware efficiency with low-variance stochastic updates. |
| **Stochastic Gradient Descent ($\text{SGD}$)** | $1$ sample | $\mathcal{O}(p)$ | Fast online updates; oscillates around minimum unless $\alpha_t$ decays. |

---

### 33. Gradient Descent — Full Derivation

Start with the scaled objective function:

$$
J(\beta)
=
\frac{1}{2n}
(X\beta - y)^T(X\beta - y)
$$

Expand the quadratic product:

$$
J(\beta)
=
\frac{1}{2n}
\left[
\beta^T X^T X\beta
- 2\beta^T X^T y
+ y^T y
\right]
$$

Differentiate with respect to $\beta$:

$$
\nabla J(\beta)
=
\frac{1}{2n}
\left[
2X^T X\beta
- 2X^T y
\right]
$$

Factoring out $2$ and $X^T$ gives the exact gradient vector:

$$
\boxed{
\nabla J(\beta)
=
\frac{1}{n}
X^T(X\beta - y)
=
\frac{1}{n}
X^T(\hat{y} - y)
}
$$

Thus, the Batch Gradient Descent update rule is:

$$
\boxed{
\beta
\leftarrow
\beta
-
\frac{\alpha}{n}
X^T(X\beta - y)
}
$$

#### Step-Size Stability Condition

The Hessian of $J(\beta)$ is $H_J = \frac{1}{n}X^T X$, which has maximum eigenvalue $L = \lambda_{\max}\left(\frac{1}{n}X^T X\right)$ (the Lipschitz constant of $\nabla J$). Batch Gradient Descent is guaranteed to converge to the global minimum if and only if the learning rate satisfies:

$$
\boxed{
0 < \alpha < \frac{2}{\lambda_{\max}\left(\frac{1}{n}X^T X\right)}
}
$$

---

### 34. Feature Scaling

Gradient Descent converges slowly when predictors span vastly different numerical scales—for example:

$$
x_1 \in [0, 1] \quad \text{while} \quad x_2 \in [0, 10^6]
$$

Disparate feature scales elongate the elliptical level sets of $J(\beta)$, creating a large condition number $\kappa = \frac{\lambda_{\max}}{\lambda_{\min}} \gg 1$ and forcing Gradient Descent into a slow zig-zagging trajectory.

**Z-Score Standardization** centers and rescales each predictor $j$ to have zero mean and unit variance:

$$
\boxed{
z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}
}
$$

Feature scaling is essential for:
- **Gradient Descent** (accelerates convergence by making level curves nearly spherical);
- **Ridge, Lasso, and Elastic Net** (ensures the penalty $\lambda$ treats all predictors fairly regardless of measurement units).

> **Leakage Rule:** In train/test workflows, always compute $\mu_j$ and $\sigma_j$ on the **training set only**, and apply those exact training statistics to standardize the validation and test sets.

---

### 35. Numerical Stability (QR and SVD Decompositions)

Why do numerical analysts discourage forming $X^T X$ explicitly when $X$ is ill-conditioned? Because squaring the design matrix **squares the condition number**:

$$
\kappa(X^T X) = \left[\kappa(X)\right]^2
$$

If $\kappa(X) = 10^8$, then $\kappa(X^T X) = 10^{16}$, which exhausts standard $64$-bit IEEE floating-point precision ($\epsilon_{mach} \approx 2.22 \times 10^{-16}$). Production solvers instead operate directly on $X$ via orthogonal or singular value decompositions:

#### 1. Thin QR Decomposition

Factor $X \in \mathbb{R}^{n \times (p+1)}$ as $X = QR$, where $Q \in \mathbb{R}^{n \times (p+1)}$ has orthonormal columns ($Q^T Q = I_{p+1}$) and $R \in \mathbb{R}^{(p+1) \times (p+1)}$ is upper triangular. Substituting into the Normal Equation:

$$
(QR)^T (QR)\hat{\beta} = (QR)^T y \implies R^T (Q^T Q) R\hat{\beta} = R^T Q^T y \implies \boxed{R\hat{\beta} = Q^T y}
$$

Since $R$ is upper triangular, $R\hat{\beta} = Q^T y$ is solved rapidly via **back-substitution** without ever forming $X^T X$.

#### 2. Singular Value Decomposition (SVD) and Pseudoinverse

Factor $X = U \Sigma V^T$, where $U \in \mathbb{R}^{n \times (p+1)}$ and $V \in \mathbb{R}^{(p+1) \times (p+1)}$ are orthogonal and $\Sigma = \text{diag}(\sigma_1, \dots, \sigma_{p+1})$ contains the singular values. The minimum-norm least-squares estimator is given by the **Moore–Penrose Pseudoinverse** $X^+$:

$$
\boxed{
\hat{\beta} = X^+ y = V \Sigma^+ U^T y = \sum_{\sigma_j > \text{tol}} \frac{u_j^T y}{\sigma_j} v_j
}
$$

This is what `np.linalg.lstsq(X, y, rcond=None)` computes, making it completely resilient even when $\text{rank}(X) < p + 1$ (rank-deficient design matrices).

---

### 36. Weighted Least Squares (WLS)

When errors are independent but heteroscedastic with known (or estimated) variances $\text{Var}(\epsilon_i \mid X) = \sigma^2 / w_i$, **Weighted Least Squares ($\text{WLS}$)** assigns positive weights $w_i > 0$ via the diagonal weight matrix $W = \text{diag}(w_1, w_2, \dots, w_n)$ and minimizes:

$$
\boxed{
J_{WLS}(\beta)
=
\sum_{i=1}^{n} w_i (y_i - x_i^T \beta)^2
=
(y - X\beta)^T W (y - X\beta)
}
$$

Differentiating with respect to $\beta$:

$$
\nabla J_{WLS}(\beta) = -2X^T Wy + 2X^T WX\beta = \mathbf{0}
$$

Setting the gradient to zero yields the Weighted Normal Equation $X^T WX\hat{\beta}_{WLS} = X^T Wy$ and the closed-form $\text{WLS}$ estimator:

$$
\boxed{
\hat{\beta}_{WLS}
=
(X^T WX)^{-1} X^T Wy
}
$$

When $w_i = 1 / \sigma_i^2$, $\text{WLS}$ downweights high-variance noisy observations and restores full Gauss–Markov efficiency ($\text{BLUE}$).

---

### 37. Generalized Least Squares (GLS)

Suppose the error covariance matrix takes a general symmetric positive definite form (allowing both heteroscedasticity and serial/spatial correlation):

$$
\text{Var}(\epsilon \mid X) = \Omega
$$

By factoring $\Omega^{-1} = P^T P$ (via Cholesky decomposition) and pre-multiplying the regression equation by $P$ ($Py = PX\beta + P\epsilon$), the transformed error $P\epsilon$ satisfies $\text{Var}(P\epsilon \mid X) = P \Omega P^T = I_n$. Applying $\text{OLS}$ to the whitened system yields the **Generalized Least Squares ($\text{GLS}$)** estimator:

$$
\boxed{
\hat{\beta}_{GLS}
=
(X^T \Omega^{-1} X)^{-1} X^T \Omega^{-1} y
}
$$

with covariance matrix:

$$
\text{Var}(\hat{\beta}_{GLS} \mid X) = (X^T \Omega^{-1} X)^{-1}
$$

- **$\text{OLS}$** is the special case where $\Omega = \sigma^2 I_n$.
- **$\text{WLS}$** is the special case where $\Omega = \text{diag}(\sigma_1^2, \dots, \sigma_n^2)$ is diagonal.

---

### 38. Ridge Regression ($L_2$ Regularization)

**Ridge Regression (Tikhonov Regularization)** adds a squared $L_2$ penalty on the slope coefficients to control variance and resolve multicollinearity:

$$
\boxed{
J_{ridge}(\beta)
=
\|y - X\beta\|_2^2
+
\lambda \sum_{j=1}^{p} \beta_j^2
=
(y - X\beta)^T(y - X\beta) + \lambda \beta^T I_0 \beta
}
$$

where $\lambda \ge 0$ is the regularization hyperparameter and $I_0 = \text{diag}(0, 1, 1, \dots, 1)$ ensures the intercept $\beta_0$ is not penalized.

Differentiating and setting the gradient to zero ($\nabla J_{ridge} = -2X^T y + 2(X^T X + \lambda I_0)\hat{\beta}_{ridge} = \mathbf{0}$) yields the closed-form Ridge estimator:

$$
\boxed{
\hat{\beta}_{ridge}
=
(X^T X + \lambda I_0)^{-1} X^T y
}
$$

#### Mathematical Properties of Ridge Regression
- **Guaranteed Invertibility:** For any $\lambda > 0$, adding $\lambda$ shifts all penalized eigenvalues to $\lambda_j + \lambda > 0$, making $X^T X + \lambda I_0$ strictly positive definite even when $p > n$ or columns are collinear.
- **Bias–Variance Tradeoff:** As $\lambda$ increases from $0 \to \infty$, $\hat{\beta}_{ridge}$ shrinks smoothly toward $\mathbf{0}$, introducing a small controlled bias in exchange for a substantial reduction in estimator variance.

---

### 39. Lasso ($L_1$) and Elastic Net Regression

#### Lasso Regression ($L_1$ Regularization)

**Lasso (Least Absolute Shrinkage and Selection Operator)** replaces the squared penalty with an $L_1$ absolute value penalty:

$$
\boxed{
J_{lasso}(\beta)
=
\frac{1}{2n}\|y - X\beta\|_2^2
+
\lambda \sum_{j=1}^{p} |\beta_j|
}
$$

Because the $L_1$ norm has sharp corners along the coordinate axes (its subdifferential at $\beta_j = 0$ is the interval $[-\lambda, +\lambda]$), Lasso drives weakly informative coefficients **exactly to zero**, performing automatic feature selection and producing sparse models. It is typically solved via **Coordinate Descent** using the soft-thresholding operator:

$$
S(z, \gamma) = \text{sign}(z) \max(|z| - \gamma, \, 0)
$$

#### Elastic Net Regression ($L_1 + L_2$ Hybrid)

When a group of predictors is highly correlated, Lasso tends to select one predictor arbitrarily and drop the others. **Elastic Net** combines both penalties via mixing ratio $\rho \in [0, 1]$:

$$
\boxed{
J_{enet}(\beta)
=
\frac{1}{2n}\|y - X\beta\|_2^2
+
\lambda \left( \rho \sum_{j=1}^{p} |\beta_j| + \frac{1 - \rho}{2} \sum_{j=1}^{p} \beta_j^2 \right)
}
$$

---

### 40. Polynomial Regression

Polynomial regression captures nonlinear curves in predictor $x$ by augmenting the design matrix with powers of $x$:

$$
\hat{y}
=
\beta_0 +
\beta_1 x +
\beta_2 x^2 +
\cdots +
\beta_d x^d
$$

This model is nonlinear in $x$ but remains **linear in $\beta$**. The design matrix becomes the $(n \times (d+1))$ **Vandermonde matrix**:

$$
X =
\begin{bmatrix}
1 & x_1 & x_1^2 & \cdots & x_1^d\\
1 & x_2 & x_2^2 & \cdots & x_2^d\\
\vdots & \vdots & \vdots & \ddots & \vdots\\
1 & x_n & x_n^2 & \cdots & x_n^d
\end{bmatrix}
$$

> **Practical Warning:** High-degree polynomials ($d \ge 5$) induce severe multicollinearity among $x, x^2, \dots, x^d$ and suffer from wild oscillations near the boundaries (**Runge's Phenomenon**). Standardize $x$ before taking powers, or use orthogonal polynomials / splines.

---

### 41. Interaction Terms

An interaction model allows the effect of one predictor to depend on the level of another predictor:

$$
\hat{y}
=
\beta_0 +
\beta_1 x_1 +
\beta_2 x_2 +
\beta_3 (x_1 x_2)
$$

Taking the partial derivative with respect to $x_1$ reveals the conditional marginal effect:

$$
\boxed{
\frac{\partial \hat{y}}{\partial x_1}
=
\beta_1 + \beta_3 x_2
}
$$

Here, $\beta_1$ is the marginal effect of $x_1$ only when $x_2 = 0$, and $\beta_3$ measures how much the slope of $x_1$ changes per one-unit increase in $x_2$.

> **Principle of Marginality (Hierarchy Principle):** Whenever an interaction term $x_1 x_2$ is included in a regression model, the lower-order main effects $x_1$ and $x_2$ should remain in the model even if their individual $p$-values appear large; omitting main effects forces an arbitrary constraint on the interaction surface under linear shifts of the origin.

---

## Part V: Machine Learning Validation, Implementation & Reference

### 42. Train/Test Split

A disciplined supervised-learning workflow partitions the dataset into disjoint subsets before any modeling takes place:

| Partition | Typical Proportion | Purpose | Rules of Engagement |
| :--- | :---: | :--- | :--- |
| **Training Set** | $70\% \text{ to } 80\%$ | Parameter estimation ($\hat{\beta}$) and fitting preprocessors ($\mu, \sigma$). | Used directly by `.fit(X_train, y_train)`. |
| **Validation Set / CV Folds** | $10\% \text{ to } 20\%$ | Hyperparameter tuning ($\lambda$, polynomial degree $d$) and feature selection. | Used iteratively to compare candidate models. |
| **Test Set (Holdout)** | $10\% \text{ to } 20\%$ | Final, unbiased estimate of out-of-sample generalization error. | Evaluated **once** at the very end; never used for tuning. |

---

### 43. Cross-Validation

**$K$-Fold Cross-Validation** partitions the training dataset into $K$ equal-sized mutually exclusive folds $F_1, F_2, \dots, F_K$. For each iteration $k \in \{1, \dots, K\}$:
1. Train the preprocessing pipeline and regression estimator on the $K - 1$ folds $\mathcal{D}_{train}^{(-k)} = \bigcup_{j \ne k} F_j$.
2. Compute the validation loss $L_k$ on the held-out fold $F_k$.

The overall cross-validation score is the average across all $K$ folds:

$$
\boxed{
CV_{(K)} = \frac{1}{K} \sum_{k=1}^{K} MSE_k
}
$$

#### Leave-One-Out Cross-Validation (LOOCV) Analytical Shortcut for OLS

When $K = n$ ($\text{LOOCV}$), linear regression possesses a remarkable closed-form identity: the leave-one-out residual $y_i - \hat{y}_{i,(-i)}$ can be computed from a **single** full-dataset $\text{OLS}$ fit using only the raw residual $e_i$ and the Hat Matrix leverage $h_{ii}$:

$$
\boxed{
CV_{(n)}
=
\frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_{i,(-i)} \right)^2
=
\frac{1}{n} \sum_{i=1}^{n} \left( \frac{e_i}{1 - h_{ii}} \right)^2
}
$$

This numerator $\sum_{i=1}^n \left(\frac{e_i}{1 - h_{ii}}\right)^2$ is known as the **Predicted Residual Error Sum of Squares ($\text{PRESS}$)** statistic.

---

### 44. Data Leakage

**Data leakage** occurs whenever information from outside the training set (such as test-set statistics or future target values) contaminates the training pipeline, producing overly optimistic validation metrics that fail in production.

| Pipeline Step | Incorrect (Leaked) Workflow | Correct (Leakage-Free) Workflow |
| :--- | :--- | :--- |
| **1. Splitting vs. Scaling** | Fit scaler on full dataset $(X)$, then split into train/test. | Split into $(X_{train}, X_{test})$ first; fit scaler on $X_{train}$ only. |
| **2. Missing Value Imputation** | Impute missing values using the global mean/median of the entire dataset. | Compute mean/median on $X_{train}$, and apply that value to $X_{train}$ and $X_{test}$. |
| **3. Feature Selection / VIF** | Select top features using correlation with $y$ across all $n$ samples. | Perform feature selection strictly inside the training set (or inside each CV fold). |
| **4. Time-Series Forecasting** | Use random $K$-fold shuffling on sequential time-series observations. | Use temporal walk-forward splits so training always precedes testing in time. |

---

### 45. Bias–Variance Tradeoff

For any unseen test point $x_*$ where $y_* = f(x_*) + \epsilon_*$ with $E[\epsilon_*] = 0$ and $\text{Var}(\epsilon_*) = \sigma^2$, the expected out-of-sample squared prediction error of an estimator $\hat{f}(x_*)$ decomposes into three fundamental terms:

$$
\boxed{
E\left[\left(y_* - \hat{f}(x_*)\right)^2\right]
=
\underbrace{\left(E[\hat{f}(x_*)] - f(x_*)\right)^2}_{\text{Bias}^2}
+
\underbrace{E\left[\left(\hat{f}(x_*) - E[\hat{f}(x_*)]\right)^2\right]}_{\text{Variance}}
+
\underbrace{\sigma^2}_{\text{Irreducible Noise}}
}
$$

- **Underfitting (High Bias, Low Variance):** Model is too restrictive (e.g., fitting a straight line to a quadratic relationship or setting $\lambda \to \infty$ in Ridge).
- **Optimal Generalization:** Balanced complexity that minimizes $\text{Bias}^2 + \text{Variance}$ on unseen data.
- **Overfitting (Low Bias, High Variance):** Model is overly flexible (e.g., high-degree polynomial $d \approx n$ or unregularized $\text{OLS}$ with $p \approx n$), memorizing training noise $\epsilon$.

---

### 46. Implementation From Scratch

#### 46.1 Functional OLS Solver Using a Stable Least-Squares Decomposition

```python
import numpy as np


def fit_linear_regression(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    Xb = np.column_stack([np.ones(X.shape[0]), X])
    beta = np.linalg.lstsq(Xb, y, rcond=None)[0]

    return beta[0], beta[1:]


def predict_linear_regression(X, intercept, coefficients):
    X = np.asarray(X, dtype=float)
    return intercept + X @ coefficients
```

#### 46.2 Functional Batch Gradient Descent Solver

```python
import numpy as np


def linear_regression_gradient_descent(
    X,
    y,
    learning_rate=0.01,
    epochs=10000
):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    Xb = np.column_stack([np.ones(X.shape[0]), X])
    beta = np.zeros(Xb.shape[1])

    for _ in range(epochs):
        error = Xb @ beta - y
        gradient = (Xb.T @ error) / len(y)
        beta -= learning_rate * gradient

    return beta
```

#### 46.3 Production-Style Reference Estimator Class (With Inference & Diagnostics)

```python
import numpy as np


class ReferenceLinearRegression:
    """
    Complete reference implementation of Ordinary and Ridge Least Squares
    supporting SVD/QR solvers, statistical inference, and leverage diagnostics.
    """

    def __init__(self, fit_intercept=True, alpha=0.0):
        self.fit_intercept = fit_intercept
        self.alpha = float(alpha)
        self.intercept_ = 0.0
        self.coef_ = None
        self.se_ = None
        self.t_stats_ = None
        self.r2_ = None
        self.adj_r2_ = None
        self.sigma2_ = None
        self.leverage_ = None
        self.cooks_d_ = None

    def _design_matrix(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if self.fit_intercept:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X

    def fit(self, X, y):
        y = np.asarray(y, dtype=float).reshape(-1)
        Xd = self._design_matrix(X)
        n, k = Xd.shape
        p = k - 1 if self.fit_intercept else k

        if self.alpha == 0.0:
            beta = np.linalg.lstsq(Xd, y, rcond=None)[0]
        else:
            penalty = np.eye(k) * self.alpha
            if self.fit_intercept:
                penalty[0, 0] = 0.0
            beta = np.linalg.solve(Xd.T @ Xd + penalty, Xd.T @ y)

        if self.fit_intercept:
            self.intercept_ = float(beta[0])
            self.coef_ = beta[1:]
        else:
            self.intercept_ = 0.0
            self.coef_ = beta

        # Compute predictions, residuals, and statistical diagnostics
        y_hat = Xd @ beta
        residuals = y - y_hat
        sse = float(residuals @ residuals)
        sst = float(np.sum((y - np.mean(y)) ** 2))
        df_res = max(n - k, 1)

        self.sigma2_ = sse / df_res
        self.r2_ = 1.0 - (sse / sst) if sst > 0 else 0.0
        self.adj_r2_ = 1.0 - ((1.0 - self.r2_) * (n - 1) / df_res)

        # Covariance matrix, standard errors, Leverage, and Cook's Distance
        xtx_inv = np.linalg.pinv(Xd.T @ Xd)
        cov_beta = self.sigma2_ * xtx_inv
        self.se_ = np.sqrt(np.maximum(np.diag(cov_beta), 1e-15))
        self.t_stats_ = beta / self.se_

        self.leverage_ = np.einsum("ij,jk,ik->i", Xd, xtx_inv, Xd)
        denom = np.maximum(1.0 - self.leverage_, 1e-12)
        self.cooks_d_ = (residuals ** 2 / (k * self.sigma2_)) * (
            self.leverage_ / (denom ** 2)
        )
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self.intercept_ + X @ self.coef_
```

---

### 47. OnithrasML Usage

The standard user-facing workflow in `OnithrasML` follows an intuitive estimator interface:

```python
import numpy as np
from onithrasML.linear_model import LinearRegression

# 1. Prepare feature matrix X and target vector y
X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
])

y = np.array([3, 5, 7, 9, 11])

# 2. Instantiate and fit the LinearRegression estimator
model = LinearRegression()
model.fit(X, y)

# 3. Generate predictions and inspect learned parameters
predictions = model.predict(X)

print("Intercept (beta_0):", model.intercept_)
print("Slope Coefficients (beta):", model.coef_)
print("Fitted Predictions:", predictions)
```

For the exact current API, constructor parameters and learned attributes should be kept synchronized with the implementation in the `OnithrasML` source repository.

---

### 48. Complete Regression Workflow

A disciplined, production-grade regression workflow follows $16$ sequential stages:

| Step | Workflow Stage | Key Action & Verification |
| :---: | :--- | :--- |
| $1$ | **Problem Formulation** | Define whether the primary goal is causal inference, parameter interpretation, or out-of-sample prediction. |
| $2$ | **Exploratory Data Analysis** | Inspect distributions, missingness, measurement units, and domain validity of variables. |
| $3$ | **Specification of $X$ and $y$** | Define predictors $X \in \mathbb{R}^{n \times p}$ and continuous response $y \in \mathbb{R}^n$. |
| $4$ | **Train/Validation/Test Split** | Partition data before fitting any scalers, imputers, or feature selectors. |
| $5$ | **Leakage-Free Preprocessing** | Fit standardization ($\mu_j, \sigma_j$) and imputation strictly on the training partition. |
| $6$ | **Baseline Establishment** | Compute the intercept-only baseline prediction $\hat{y}_i = \bar{y}_{train}$ and its baseline $\text{RMSE}$. |
| $7$ | **Model Fitting** | Fit `LinearRegression` via a numerically stable solver (`lstsq` / QR / SVD). |
| $8$ | **Metric Evaluation** | Compute $\text{MAE}$, $\text{MSE}$, $\text{RMSE}$, $R^2$, and $R^2_{adj}$ on training and validation folds. |
| $9$ | **Residual Diagnostics** | Inspect $e_i$ vs. $\hat{y}_i$, Normal Q–Q plots, and scale–location plots for nonlinearity or heteroscedasticity. |
| $10$ | **Multicollinearity Audit** | Compute $\text{VIF}_j = \frac{1}{1 - R_j^2}$ and the condition number $\kappa(X)$ across predictors. |
| $11$ | **Influence Analysis** | Check Hat Matrix leverages $h_{ii}$ and Cook's Distances $D_i$ for high-impact observations. |
| $12$ | **Statistical Inference** | Evaluate $SE(\hat{\beta}_j)$ (classical or $\text{HC1}$ robust), $t$-tests, confidence intervals, and the global $F$-test. |
| $13$ | **Regularization & Extensions** | Compare $\text{WLS}$/$\text{GLS}$ (if heteroscedastic) or Ridge/Lasso/Elastic Net (if collinear or high-dimensional). |
| $14$ | **Cross-Validation** | Select hyperparameters ($\lambda$, polynomial degree $d$) via $K$-fold cross-validation or $\text{LOOCV}$ $\text{PRESS}$. |
| $15$ | **Final Holdout Evaluation** | Evaluate the locked pipeline **once** on the untouched test set to report honest generalization performance. |
| $16$ | **Documentation** | Record model assumptions, confidence bounds, domain limitations, and monitoring thresholds. |

---

### 49. Common Mistakes

| Mistake | Why It Is Incorrect | Best Practice / Remediation |
| :--- | :--- | :--- |
| **1. Treating $R^2$ as classification accuracy** | $R^2 = 1 - \frac{SSE}{SST}$ measures variance reduction relative to $\bar{y}$ on a continuous scale, not the fraction of "correct" predictions. | Pair $R^2$ and $R^2_{adj}$ with scale-meaningful error metrics ($\text{RMSE}$ and $\text{MAE}$). |
| **2. Assuming high $R^2$ proves a valid model** | Spurious time trends, data leakage, overfitting ($p \approx n$), or severe nonlinearity can coexist with $R^2 > 0.95$. | Always inspect residual plots and out-of-sample cross-validation error alongside $R^2$. |
| **3. Equating a small $p$-value with a large practical effect** | With huge sample sizes ($n > 10^6$), tiny, practically irrelevant slopes produce minuscule $p$-values because $SE(\hat{\beta}_j) \propto 1/\sqrt{n}$. | Report point estimates $\hat{\beta}_j$ and $95\%$ confidence intervals in meaningful domain units. |
| **4. Confusing sample residuals $e$ with theoretical errors $\epsilon$** | $\epsilon = y - X\beta$ is unobserved population noise; $e = (I - H)\epsilon$ is the observed projection onto $\text{Col}(X)^\perp$. | Account for leverage $h_{ii}$ using studentized residuals $r_i = \frac{e_i}{\hat{\sigma}\sqrt{1 - h_{ii}}}$. |
| **5. Explicitly inverting $X^T X$ in production code** | Computing `np.linalg.inv(X.T @ X)` squares the condition number $\kappa(X)^2$ and amplifies floating-point rounding errors. | Use `np.linalg.lstsq(X, y, rcond=None)` or QR/SVD decompositions. |
| **6. Ignoring feature scale in Gradient Descent or Ridge/Lasso** | Unscaled predictors cause elongated loss contours in Gradient Descent and unequal unit-dependent penalties in Ridge/Lasso. | Standardize predictors ($z = \frac{x - \mu}{\sigma}$) using training-set parameters. |
| **7. Tuning hyperparameters repeatedly on the test set** | Iteratively modifying features or $\lambda$ to improve test-set scores leaks test information into model selection. | Tune exclusively via $K$-fold cross-validation on the training set; touch the test set once. |
| **8. Deleting high-$D_i$ outliers automatically** | An influential observation may be a data entry error, a valid extreme regime, or evidence of a missing nonlinear feature. | Investigate data provenance and compare sensitivity between $\text{OLS}$ and robust estimators before removal. |

---

### 50. Formula Sheet

| Concept | Mathematical Formula |
| :--- | :--- |
| **1. Population Linear Model** | $$y = X\beta + \epsilon$$ |
| **2. Fitted Prediction Vector** | $$\hat{y} = X\hat{\beta}$$ |
| **3. Sample Residual Vector** | $$e = y - \hat{y} = (I_n - H)y$$ |
| **4. Residual Sum of Squares ($\text{SSE}$)** | $$SSE = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = (y - X\hat{\beta})^T(y - X\hat{\beta})$$ |
| **5. Mean Squared Error ($\text{MSE}$)** | $$MSE = \frac{SSE}{n}$$ |
| **6. Root Mean Squared Error ($\text{RMSE}$)** | $$RMSE = \sqrt{MSE}$$ |
| **7. Mean Absolute Error ($\text{MAE}$)** | $$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$ |
| **8. Simple Regression Slope ($\hat{\beta}_1$)** | $$\hat{\beta}_1 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}$$ |
| **9. Simple Regression Intercept ($\hat{\beta}_0$)** | $$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x}$$ |
| **10. Normal Equation** | $$X^T X\hat{\beta} = X^T y$$ |
| **11. Closed-Form OLS Estimator** | $$\hat{\beta} = (X^T X)^{-1}X^T y$$ |
| **12. Unbiased Error Variance ($\hat{\sigma}^2$)** | $$\hat{\sigma}^2 = \frac{SSE}{n - p - 1}$$ |
| **13. Coefficient Covariance Matrix** | $$\widehat{\text{Var}}(\hat{\beta}) = \hat{\sigma}^2(X^T X)^{-1}$$ |
| **14. Standard Error of $\hat{\beta}_j$** | $$SE(\hat{\beta}_j) = \sqrt{\hat{\sigma}^2\left[(X^T X)^{-1}\right]_{jj}}$$ |
| **15. Student's $t$-Statistic** | $$t = \frac{\hat{\beta}_j - \beta_{j,0}}{SE(\hat{\beta}_j)}$$ |
| **16. $(1-\alpha)$ Confidence Interval** | $$\hat{\beta}_j \pm t_{1-\alpha/2, \, df}SE(\hat{\beta}_j)$$ |
| **17. Coefficient of Determination ($R^2$)** | $$R^2 = \frac{SSR}{SST} = 1 - \frac{SSE}{SST}$$ |
| **18. Adjusted $R^2$** | $$R^2_{adj} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$$ |
| **19. Omnibus $F$-Statistic** | $$F = \frac{SSR / p}{SSE / (n - p - 1)}$$ |
| **20. Variance Inflation Factor ($\text{VIF}_j$)** | $$VIF_j = \frac{1}{1 - R_j^2}$$ |
| **21. Hat Matrix ($H$) and Leverage ($h_{ii}$)** | $$H = X(X^T X)^{-1}X^T, \qquad h_{ii} = x_i^T(X^T X)^{-1}x_i$$ |
| **22. Cook's Distance ($D_i$)** | $$D_i = \frac{e_i^2}{(p + 1)\hat{\sigma}^2}\frac{h_{ii}}{(1 - h_{ii})^2}$$ |
| **23. Batch Gradient Vector** | $$\nabla J(\beta) = \frac{1}{n}X^T(X\beta - y)$$ |
| **24. Gradient Descent Update** | $$\beta \leftarrow \beta - \alpha \nabla J(\beta)$$ |
| **25. Weighted Least Squares ($\text{WLS}$)** | $$\hat{\beta}_{WLS} = (X^T WX)^{-1}X^T Wy$$ |
| **26. Generalized Least Squares ($\text{GLS}$)** | $$\hat{\beta}_{GLS} = (X^T \Omega^{-1}X)^{-1}X^T \Omega^{-1}y$$ |
| **27. Ridge Estimator ($L_2$)** | $$\hat{\beta}_{ridge} = (X^T X + \lambda I_0)^{-1}X^T y$$ |

---

### 51. Concept Map

The conceptual architecture of Linear Regression unifies three foundational pillars into a single coherent estimation framework:

| Pillar | Core Subfields | Key Mathematical Objects | Unified Estimator & Output |
| :--- | :--- | :--- | :--- |
| **1. Mathematics** | Matrix Algebra, Multivariable Calculus, Euclidean Geometry | Gram Matrix $X^T X$, Gradient $\nabla_\beta J$, Projection Matrix $H = X(X^T X)^{-1}X^T$, Orthogonality $e \perp \text{Col}(X)$ | $$\hat{\beta} = (X^T X)^{-1}X^T y$$ |
| **2. Statistics** | Sampling Theory, Hypothesis Inference, Residual Diagnostics | Unbiased Variance $\hat{\sigma}^2$, Covariance $\sigma^2(X^T X)^{-1}$, $t / F / z$ Tests, $\text{VIF}_j$, Leverage $h_{ii}$, Cook's $D_i$ | $$\hat{\beta}_j \pm t_{1-\alpha/2, \, df}SE(\hat{\beta}_j)$$ |
| **3. Implementation** | Numerical Linear Algebra, Regularization, ML Validation | QR / SVD Solvers (`lstsq`), Gradient Descent, Ridge / Lasso / $\text{WLS}$ / $\text{GLS}$, Cross-Validation, `fit(X, y)` / `predict(X)` | $$\hat{y} = X\hat{\beta}$$ |

---

### 52. Learning Checklist

A complete mastery of Linear Regression in **OnithrasML** encompasses the following $56$ competencies:

- [ ] Simple Linear Regression
- [ ] Multiple Linear Regression
- [ ] Design Matrix $X \in \mathbb{R}^{n \times (p+1)}$
- [ ] Intercept column $\mathbf{1}_n$
- [ ] Statistical model $y = X\beta + \epsilon$
- [ ] Residuals $e = y - \hat{y}$
- [ ] Residual Sum of Squares ($\text{SSE}$)
- [ ] Mean Squared Error ($\text{MSE}$)
- [ ] Root Mean Squared Error ($\text{RMSE}$)
- [ ] Mean Absolute Error ($\text{MAE}$)
- [ ] Principle of Least Squares
- [ ] Ordinary Least Squares ($\text{OLS}$)
- [ ] Normal Equation $X^T X\hat{\beta} = X^T y$
- [ ] Full matrix calculus derivation
- [ ] Geometric interpretation ($\text{Col}(X)$ projection)
- [ ] Gradient Descent optimization
- [ ] Gradient vector derivation $\nabla J(\beta)$
- [ ] Learning rate $\alpha$ and step-size stability
- [ ] Feature scaling (Z-score standardization)
- [ ] Coefficient of Determination ($R^2$)
- [ ] Adjusted $R^2$
- [ ] Degrees of freedom ($df = n - p - 1$)
- [ ] Unbiased error variance estimator $\hat{\sigma}^2$
- [ ] Standard Error $SE(\hat{\beta}_j)$
- [ ] Student's $t$-Test
- [ ] $p$-Value interpretation
- [ ] Confidence Intervals and Prediction Intervals
- [ ] $z$-Test asymptotic connection
- [ ] Omnibus and Partial $F$-Tests
- [ ] Classical regression assumptions
- [ ] Strict exogeneity $E[\epsilon \mid X] = \mathbf{0}$
- [ ] Homoscedasticity vs. Heteroscedasticity
- [ ] Multicollinearity analysis
- [ ] Variance Inflation Factor ($\text{VIF}$)
- [ ] Residual diagnostic plots
- [ ] Hat Matrix $H = X(X^T X)^{-1}X^T$
- [ ] Leverage $h_{ii}$
- [ ] Cook's Distance $D_i$
- [ ] Gauss–Markov Theorem and $\text{BLUE}$
- [ ] Conditional unbiasedness derivation
- [ ] $\text{OLS}$ variance–covariance matrix derivation
- [ ] Weighted Least Squares ($\text{WLS}$)
- [ ] Generalized Least Squares ($\text{GLS}$)
- [ ] Ridge Regression ($L_2$ penalty)
- [ ] Lasso ($L_1$) and Elastic Net Regression
- [ ] Polynomial Regression
- [ ] Interaction terms and marginal effects
- [ ] Train/Test Split discipline
- [ ] $K$-Fold Cross-Validation and $\text{LOOCV}$ $\text{PRESS}$
- [ ] Data Leakage prevention
- [ ] Bias–Variance Tradeoff decomposition
- [ ] Numerical Stability (QR and SVD vs. matrix inversion)
- [ ] Matrix Rank and identifiability
- [ ] NumPy least-squares implementation from scratch
- [ ] Batch Gradient Descent implementation from scratch
- [ ] `OnithrasML` estimator API usage

---

## Final Mental Model

If you understand these five equations, you understand the mathematical core of Linear Regression:

1. **Statistical Model:**

$$
y = X\beta + \epsilon
$$

2. **Least-Squares Objective:**

$$
\min_\beta \|y - X\beta\|_2^2
$$

3. **Normal Equation:**

$$
X^T X\hat{\beta} = X^T y
$$

4. **Parameter Estimate:**

$$
\hat{\beta} = (X^T X)^{-1}X^T y
$$

5. **Fitted Prediction:**

$$
\hat{y} = X\hat{\beta}
$$

Everything else—residuals, evaluation metrics, standard errors, hypothesis tests, confidence intervals, leverage diagnostics, Ridge, Lasso, $\text{WLS}$, $\text{GLS}$, and Gradient Descent—builds directly on this foundation.

---

## Documentation Principle

`OnithrasML` documentation answers three essential questions for every major estimator:

1. **How do I use it?** (Clean, reproducible API examples and workflows)
2. **Why does it work mathematically?** (Complete first-principles derivations and geometric intuition)
3. **When should I trust or question the result?** (Assumptions, diagnostics, numerical conditioning, and statistical inference)

That makes the documentation valuable both as a production package reference and as a rigorous mathematical learning resource.

---

**OnithrasML — Linear Regression: from `fit(X, y)` to the mathematics behind the estimator.**