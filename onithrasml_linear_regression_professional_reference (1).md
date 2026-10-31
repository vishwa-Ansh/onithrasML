# OnithrasML — Linear Regression: Complete Mathematical, Statistical, and Implementation Reference

> A complete mathematical, statistical, diagnostic, and implementation reference for Linear Regression in **OnithrasML**.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/Docs-OnithrasML-blue.svg)](https://onithrasml.vercel.app)

| Document Specification | Details |
| :--- | :--- |
| **Library & Module** | `onithrasML.linear_model.LinearRegression` |
| **Scope** | Classical Ordinary Least Squares ($\text{OLS}$), Matrix Calculus, Statistical Inference, Diagnostics, Regularization, and Numerical Implementation |
| **Target Audience** | Machine Learning Engineers, Data Scientists, Statisticians, and Researchers |
| **Document Version** | Production Reference $2.0$ (GitHub-Optimized Edition) |

Linear Regression is one of the most important foundations of classical machine learning and econometrics. This reference guide goes from the basic univariate model all the way to **Ordinary Least Squares ($\text{OLS}$), matrix derivations, Maximum Likelihood Estimation ($\text{MLE}$), statistical inference, residual and influence diagnostics, Weighted Least Squares ($\text{WLS}$), Generalized Least Squares ($\text{GLS}$), Ridge, Lasso, Elastic Net, Gradient Descent, QR/SVD numerical stability, and full implementation from scratch**.

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
15. [R-Squared and Adjusted R-Squared](#15-r-squared-and-adjusted-r-squared)
16. [Degrees of Freedom](#16-degrees-of-freedom)
17. [Error Variance](#17-error-variance)
18. [Standard Error — Derivation](#18-standard-error--derivation)
19. [t-Test and p-Value](#19-t-test-and-p-value)
20. [Confidence Intervals and Prediction Intervals](#20-confidence-intervals-and-prediction-intervals)
21. [z-Test](#21-z-test)
22. [F-Test](#22-f-test)
23. [Regression Assumptions](#23-regression-assumptions)
24. [Gauss–Markov Theorem, BLUE, and Maximum Likelihood](#24-gaussmarkov-theorem-blue-and-maximum-likelihood)
25. [Unbiasedness — Derivation](#25-unbiasedness--derivation)
26. [Variance of OLS — Derivation and Robust Standard Errors](#26-variance-of-ols--derivation-and-robust-standard-errors)

### Part III: Diagnostics, Multicollinearity & Influential Observations
27. [Multicollinearity](#27-multicollinearity)
28. [VIF (Variance Inflation Factor)](#28-vif-variance-inflation-factor)
29. [Residual Diagnostics](#29-residual-diagnostics)
30. [Hat Matrix and Leverage](#30-hat-matrix-and-leverage)
31. [Cook's Distance](#31-cooks-distance)

### Part IV: Optimization, Numerical Stability & Extended Estimators
32. [Gradient Descent](#32-gradient-descent)
33. [Gradient Descent — Full Derivation](#33-gradient-descent--full-derivation)
34. [Feature Scaling](#34-feature-scaling)
35. [Numerical Stability (QR and SVD Decompositions)](#35-numerical-stability-qr-and-svd-decompositions)
36. [WLS (Weighted Least Squares)](#36-wls-weighted-least-squares)
37. [GLS (Generalized Least Squares)](#37-generalized-least-squares-gls)
38. [Ridge Regression](#38-ridge-regression)
39. [Lasso and Elastic Net Regression](#39-lasso-and-elastic-net-regression)
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

For one predictor variable $x$:

$$
\hat{y} = \beta_0 + \beta_1 x
$$

For several predictors $x_1, x_2, \dots, x_p$:

$$
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p
$$

The important phrase is **linear in the parameters** $\beta_0, \beta_1, \dots, \beta_p$. A model such as:

$$
\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2
$$

is still a linear regression model because it is linear in the coefficients $\beta_0, \beta_1, \beta_2$. Specifically, the partial derivatives with respect to each parameter do not depend on any parameter $\beta_k$:

$$
\frac{\partial \hat{y}}{\partial \beta_0} = 1, \qquad \frac{\partial \hat{y}}{\partial \beta_1} = x, \qquad \frac{\partial \hat{y}}{\partial \beta_2} = x^2
$$

By contrast, a specification such as $\hat{y} = \beta_0 \exp(\beta_1 x)$ is nonlinear in the parameters because $\frac{\partial \hat{y}}{\partial \beta_1} = \beta_0 x \exp(\beta_1 x)$ depends on $\beta_0$ and $\beta_1$.

The objective of Ordinary Least Squares ($\text{OLS}$) is to find the coefficient vector $\hat{\beta}$ that minimizes the sum of squared residuals across all $n$ observations:

$$
\boxed{\hat{\beta} = \arg\min_{\beta} \sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

---

### 2. Why Linear Regression Matters

Linear Regression is foundational because it provides a direct connection between six core disciplines:

- **Linear Algebra** — vector spaces, matrices, orthogonal projections, subspaces, and matrix rank
- **Calculus** — partial derivatives, gradients, Hessian matrices, and convexity
- **Statistics** — point estimation, sampling distributions, hypothesis testing, and uncertainty quantification
- **Probability** — conditional expectation, error assumptions, and likelihood functions
- **Optimization** — closed-form analytical solutions versus iterative gradient methods
- **Machine Learning** — feature engineering, regularization, training, and out-of-sample generalization

A useful mental model of the pipeline is:

```text
Data
  |
  v
Design Matrix X
  |
  v
Model: y = X * beta + epsilon
  |
  v
Choose beta that minimizes squared error
  |
  v
Estimated Coefficients (beta_hat)
  |
  v
Predictions: y_hat = X * beta_hat
  |
  v
Residuals: e = y - y_hat
  |
  v
Metrics + Diagnostics + Statistical Inference
```

| Stage | Mathematical Object | Core Operation |
| :--- | :--- | :--- |
| **1. Raw Data** | $\{(x_i, y_i)\}_{i=1}^n$ | Collect $n$ observations of predictors and continuous target values. |
| **2. Design Matrix** | $X \in \mathbb{R}^{n \times (p+1)}$ | Encode predictors alongside a leading column of ones for the intercept. |
| **3. Statistical Model** | $y = X\beta + \epsilon$ | Formulate the population relationship with unobserved error $\epsilon$. |
| **4. Least Squares Objective** | $\min_{\beta} \Vert y - X\beta \Vert_2^2$ | Choose $\beta$ that minimizes the sum of squared errors. |
| **5. Parameter Estimate** | $\hat{\beta} = (X^T X)^{-1} X^T y$ | Compute the optimal coefficient vector via the Normal Equation. |
| **6. Predictions** | $\hat{y} = X\hat{\beta} = Hy$ | Project $y$ orthogonally onto the column space of $X$. |
| **7. Residuals** | $e = y - \hat{y}$ | Compute the orthogonal error component for diagnostics and inference. |
| **8. Evaluation & Inference** | $R^2, \text{RMSE}, t, F, \text{VIF}$ | Quantify predictive accuracy, parameter uncertainty, and model validity. |

---

### 3. Mathematical Setup

Let us define the core mathematical objects used throughout this reference:

- $n$ = number of observations (samples)
- $p$ = number of predictors (features), excluding the intercept
- $X$ = design matrix containing the intercept column and all $p$ predictor columns
- $y$ = observed target vector
- $\beta$ = true population/model coefficient vector
- $\hat{\beta}$ = estimated sample coefficient vector
- $\epsilon$ = unobserved theoretical error vector
- $e$ = observed sample residual vector

#### Dimensions

$$
X \in \mathbb{R}^{n \times (p+1)}, \qquad y \in \mathbb{R}^{n \times 1}, \qquad \beta \in \mathbb{R}^{(p+1) \times 1}, \qquad \epsilon \in \mathbb{R}^{n \times 1}, \qquad e \in \mathbb{R}^{n \times 1}
$$

The classical linear regression model is:

$$
\boxed{y = X\beta + \epsilon}
$$

After fitting the model on sample data:

$$
\boxed{\hat{y} = X\hat{\beta}}
$$

---

### 4. Simple Linear Regression

When $p = 1$, the simple linear regression model is:

$$
\boxed{y_i = \beta_0 + \beta_1 x_i + \epsilon_i}
$$

The predicted value for observation $i$ is:

$$
\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i
$$

#### Meaning of $\beta_0$ (Intercept)

$\beta_0$ is the model's expected target value $E[y \mid x = 0]$ when $x = 0$, assuming $x = 0$ is meaningful in the problem domain. If $x = 0$ lies outside the realistic range of data, $\beta_0$ acts as a mathematical vertical anchor unless the predictor is centered around its mean ($x_i - \bar{x}$).

#### Meaning of $\beta_1$ (Slope)

$\beta_1$ is the change in predicted $y$ for a one-unit increase in $x$:

$$
\beta_1 = \frac{d\hat{y}}{dx}
$$

For example:

$$
\hat{y} = 50 + 3x
$$

means that increasing $x$ by $1$ unit changes the predicted value $\hat{y}$ by $3$ units.

---

### 5. Deriving the Simple Regression Coefficients

We want to minimize the residual sum of squares objective function with respect to $\beta_0$ and $\beta_1$:

$$
S(\beta_0, \beta_1) = \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i)^2
$$

#### Step 1: Partial Derivative with Respect to $\beta_0$

Take the partial derivative with respect to $\beta_0$:

$$
\frac{\partial S}{\partial \beta_0} = -2\sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i)
$$

At the minimum, set the derivative equal to zero:

$$
\sum_{i=1}^{n}(y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i) = 0
$$

Therefore:

$$
n\hat{\beta}_0 + \hat{\beta}_1\sum_{i=1}^{n} x_i = \sum_{i=1}^{n} y_i
$$

Divide through by $n$ using $\bar{x} = \frac{1}{n}\sum_{i=1}^n x_i$ and $\bar{y} = \frac{1}{n}\sum_{i=1}^n y_i$:

$$
\boxed{\hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x}}
$$

> **Centroid Property:** Because $\bar{y} = \hat{\beta}_0 + \hat{\beta}_1\bar{x}$, the fitted simple $\text{OLS}$ regression line always passes through the point of sample means $(\bar{x}, \bar{y})$.

#### Step 2: Partial Derivative with Respect to $\beta_1$

Now differentiate $S(\beta_0, \beta_1)$ with respect to $\beta_1$:

$$
\frac{\partial S}{\partial \beta_1} = -2\sum_{i=1}^{n} x_i(y_i - \beta_0 - \beta_1 x_i)
$$

Set equal to zero:

$$
\sum_{i=1}^{n} x_i(y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i) = 0
$$

Substitute $\hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x}$:

$$
\sum_{i=1}^{n} x_i\left[(y_i - \bar{y}) - \hat{\beta}_1(x_i - \bar{x})\right] = 0
$$

Using the algebraic identities $\sum_{i=1}^n \bar{x}(y_i - \bar{y}) = 0$ and $\sum_{i=1}^n \bar{x}(x_i - \bar{x}) = 0$, we obtain the exact closed-form solution for simple $\text{OLS}$:

$$
\boxed{\hat{\beta}_1 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2} = \frac{S_{xy}}{S_{xx}}}
$$

and:

$$
\boxed{\hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x}}
$$

#### Connection to Sample Covariance and Pearson Correlation

Dividing the numerator and denominator of $\hat{\beta}_1$ by $n - 1$ links the slope directly to the sample covariance $\text{Cov}(x, y)$, sample standard deviations $s_x, s_y$, and Pearson correlation $r_{xy}$:

$$
\hat{\beta}_1 = \frac{\text{Cov}(x, y)}{\text{Var}(x)} = r_{xy}\frac{s_y}{s_x}
$$

---

### 6. Multiple Linear Regression

For $p$ predictors, the model is:

$$
\boxed{\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_p x_p}
$$

The coefficient interpretation is conditional on all other included predictors (*ceteris paribus*):

$$
\beta_j = \frac{\partial E[y \mid x_1, \dots, x_p]}{\partial x_j}
$$

For example:

$$
\hat{y} = 100 + 5x_1 - 2x_2
$$

The coefficient $5$ means:

> Holding $x_2$ constant, a one-unit increase in $x_1$ changes the model's predicted target by $5$ units.

This "holding other variables constant" interpretation is one major reason multiple regression is more powerful than fitting separate simple regressions. By the **Frisch–Waugh–Lovell ($\text{FWL}$) Theorem**, $\hat{\beta}_j$ in multiple regression is identical to the slope obtained by regressing residualized $y$ on residualized $x_j$ after linearly partialling out all other $p - 1$ predictors.

---

### 7. The Design Matrix

For a two-predictor regression model:

$$
y_i = \beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \epsilon_i
$$

the design matrix $X \in \mathbb{R}^{n \times 3}$ is:

$$
X = \begin{bmatrix} 1 & x_{11} & x_{12} \\ 1 & x_{21} & x_{22} \\ \vdots & \vdots & \vdots \\ 1 & x_{n1} & x_{n2} \end{bmatrix}
$$

and the parameter vector $\beta \in \mathbb{R}^{3 \times 1}$ is:

$$
\beta = \begin{bmatrix} \beta_0 \\ \beta_1 \\ \beta_2 \end{bmatrix}
$$

Then matrix-vector multiplication produces the linear predictions across all $n$ observations simultaneously:

$$
X\beta = \begin{bmatrix} \beta_0 + \beta_1 x_{11} + \beta_2 x_{12} \\ \beta_0 + \beta_1 x_{21} + \beta_2 x_{22} \\ \vdots \\ \beta_0 + \beta_1 x_{n1} + \beta_2 x_{n2} \end{bmatrix}
$$

The leading intercept column of ones $\mathbf{1}_n$ is therefore not cosmetic—it is what allows matrix multiplication to estimate $\beta_0$.

---

### 8. The Statistical Model

The statistical population model is:

$$
\boxed{y = X\beta + \epsilon}
$$

This separates two fundamental components:

1. **Systematic component:** $X\beta$
2. **Random component:** $\epsilon$

The population regression function is:

$$
E[y \mid X] = X\beta \quad \text{when} \quad E[\epsilon \mid X] = \mathbf{0}
$$

This distinction is essential:

- $\epsilon = y - X\beta$ is the **unobserved theoretical error** from the true population hyperplane.
- $e = y - \hat{y} = y - X\hat{\beta}$ is the **observed sample residual** from the fitted hyperplane.

---

### 9. Residuals

For observation $i$:

$$
\boxed{e_i = y_i - \hat{y}_i}
$$

In vector form:

$$
\boxed{e = y - X\hat{\beta}}
$$

- A **positive residual** ($e_i > 0$) means the actual value $y_i$ is above the prediction $\hat{y}_i$.
- A **negative residual** ($e_i < 0$) means the actual value $y_i$ is below the prediction $\hat{y}_i$.

#### Algebraic Properties of OLS Residuals

When an intercept column is included in $X$, the $\text{OLS}$ residuals satisfy four exact identities in every sample:

1. **Zero Sum and Zero Mean:** $\sum_{i=1}^n e_i = 0 \implies \bar{e} = 0$.
2. **Orthogonality to Predictors:** $X^T e = \mathbf{0} \implies \sum_{i=1}^n x_{ij} e_i = 0$ for each predictor $j$.
3. **Orthogonality to Predictions:** $\hat{y}^T e = (X\hat{\beta})^T e = \hat{\beta}^T X^T e = 0$.
4. **Equality of Target and Prediction Means:** $\sum_{i=1}^n y_i = \sum_{i=1}^n \hat{y}_i \implies \bar{y} = \bar{\hat{y}}$.

---

### 10. Least Squares

The Residual Sum of Squares ($\text{SSE}$) is:

$$
\boxed{SSE = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} e_i^2}
$$

Using $\hat{y} = X\beta$ during optimization:

$$
SSE(\beta) = (y - X\beta)^T(y - X\beta) = \Vert y - X\beta \Vert_2^2
$$

#### Why Square the Errors?

1. **Prevents Sign Cancellation:** Minimizing $\sum_{i=1}^n (y_i - \hat{y}_i)$ fails because positive and negative errors cancel each other out.
2. **Differentiability & Closed-Form Solution:** Minimizing absolute error $\sum_{i=1}^n \vert y_i - \hat{y}_i \vert$ avoids cancellation and is more robust to outliers, but its derivative is undefined at zero and it lacks a closed-form matrix formula.
3. **Statistical Efficiency:** Squared error yields a smooth, strictly convex quadratic objective (when $X$ has full column rank) with a unique closed-form solution that is both $\text{BLUE}$ under the Gauss–Markov assumptions and the Maximum Likelihood Estimator ($\text{MLE}$) under Gaussian errors.

---

### 11. Normal Equation — Full Derivation

Start with the matrix squared-error objective:

$$
J(\beta) = (y - X\beta)^T(y - X\beta)
$$

Expand the transpose:

$$
J(\beta) = (y^T - \beta^T X^T)(y - X\beta)
$$

Multiply out the terms:

$$
J(\beta) = y^T y - y^T X\beta - \beta^T X^T y + \beta^T X^T X\beta
$$

Because $y^T X\beta$ is a $1 \times 1$ scalar, it equals its own transpose:

$$
y^T X\beta = (y^T X\beta)^T = \beta^T X^T y
$$

Therefore:

$$
J(\beta) = y^T y - 2\beta^T X^T y + \beta^T X^T X\beta
$$

Differentiate with respect to the vector $\beta$:

$$
\nabla_\beta J(\beta) = -2X^T y + 2X^T X\beta
$$

At the minimum, set the gradient to the zero vector:

$$
\nabla_\beta J(\hat{\beta}) = \mathbf{0} \implies -2X^T y + 2X^T X\hat{\beta} = \mathbf{0}
$$

Divide by $2$ to obtain the **Normal Equation**:

$$
\boxed{X^T X\hat{\beta} = X^T y}
$$

If $X$ has full column rank $\text{rank}(X) = p + 1$, then $X^T X$ is symmetric positive definite and invertible:

$$
\boxed{\hat{\beta} = (X^T X)^{-1} X^T y}
$$

#### Second-Order Sufficiency Check (Hessian Matrix)

Taking the second derivative gives the Hessian matrix:

$$
\nabla^2_\beta J(\beta) = 2X^T X
$$

For any non-zero vector $v \in \mathbb{R}^{p+1}$, $v^T(2X^T X)v = 2\Vert Xv \Vert_2^2 \ge 0$ (and strictly $> 0$ when $\text{rank}(X) = p + 1$), confirming that $\hat{\beta}$ is the unique global minimizer.

#### Important Numerical Note

The matrix inverse expression $(X^T X)^{-1} X^T y$ is ideal for theoretical derivations, but production numerical code should avoid explicitly computing the inverse. Prefer solving the linear system or using an SVD/QR least-squares solver:

```python
# Preferred linear system solver when X.T @ X is well-conditioned:
beta = np.linalg.solve(X.T @ X, X.T @ y)

# Preferred general least-squares solver (uses SVD):
beta = np.linalg.lstsq(X, y, rcond=None)[0]
```

---

### 12. Geometric Interpretation

Ordinary Least Squares has a clear geometric interpretation in $\mathbb{R}^n$.

The prediction vector $\hat{y}$ is the **orthogonal projection** of $y$ onto the column space $\text{Col}(X)$:

$$
\hat{y} = Hy
$$

where $H$ is the projection (Hat) matrix:

$$
\boxed{H = X(X^T X)^{-1} X^T}
$$

The residual vector is:

$$
e = y - \hat{y} = (I_n - H)y
$$

At the $\text{OLS}$ solution:

$$
X^T e = \mathbf{0} \implies e \perp \text{Col}(X)
$$

This means the residual vector $e$ is orthogonal to every column of $X$ and to $\hat{y}$.

```text
              y
             /|
            / |
           /  |
          /   |  e = y - y_hat
         /    |  (orthogonal to Col(X))
        /     |
       /______|
        y_hat = H * y  (in Col(X))

y_hat = orthogonal projection of y onto Col(X)
e     = y - y_hat
e is perpendicular to Col(X)
```

---

### 13. Worked Numerical Example

Consider the dataset of $n = 5$ observations:

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

#### Step 2 — Calculate the Centered Values

| $x_i$ | $y_i$ | $x_i - \bar{x}$ | $y_i - \bar{y}$ | Product $(x_i - \bar{x})(y_i - \bar{y})$ | Squared Deviation $(x_i - \bar{x})^2$ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| $1$ | $3$ | $-2$ | $-2.4$ | $4.8$ | $4$ |
| $2$ | $5$ | $-1$ | $-0.4$ | $0.4$ | $1$ |
| $3$ | $4$ | $0$ | $-1.4$ | $0.0$ | $0$ |
| $4$ | $8$ | $1$ | $2.6$ | $2.6$ | $1$ |
| $5$ | $7$ | $2$ | $1.6$ | $3.2$ | $4$ |
| **Total** | **$27$** | **$0$** | **$0.0$** | **$S_{xy} = 11.0$** | **$S_{xx} = 10$** |

Therefore:

$$
S_{xy} = 11, \qquad S_{xx} = 10
$$

#### Step 3 — Slope

$$
\hat{\beta}_1 = \frac{S_{xy}}{S_{xx}} = \frac{11}{10} = \boxed{1.1}
$$

#### Step 4 — Intercept

$$
\hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x} = 5.4 - (1.1)(3) = \boxed{2.1}
$$

Therefore, the fitted regression line is:

$$
\boxed{\hat{y} = 2.1 + 1.1x}
$$

#### Step 5 — Predictions and Residuals

| $x_i$ | $y_i$ | $\hat{y}_i = 2.1 + 1.1x_i$ | Residual $e_i = y_i - \hat{y}_i$ | Squared Residual $e_i^2$ |
| ---: | ---: | ---: | ---: | ---: |
| $1$ | $3$ | $3.2$ | $-0.2$ | $0.04$ |
| $2$ | $5$ | $4.3$ | $0.7$ | $0.49$ |
| $3$ | $4$ | $5.4$ | $-1.4$ | $1.96$ |
| $4$ | $8$ | $6.5$ | $1.5$ | $2.25$ |
| $5$ | $7$ | $7.6$ | $-0.6$ | $0.36$ |
| **Total** | **$27$** | **$27.0$** | **$0.0$** | **$5.10$** |

#### Step 6 — SSE

$$
SSE = (-0.2)^2 + (0.7)^2 + (-1.4)^2 + (1.5)^2 + (-0.6)^2 = 0.04 + 0.49 + 1.96 + 2.25 + 0.36 = \boxed{5.10}
$$

#### Step 7 — Prediction MSE vs. Unbiased Error Variance

There are $n = 5$ observations. For descriptive prediction error:

$$
MSE = \frac{SSE}{n} = \frac{5.10}{5} = \boxed{1.02}
$$

For an unbiased estimator of the population error variance $\sigma^2$ in classical regression inference, divide by the residual degrees of freedom $df = n - p - 1$ (here $p = 1$, so $df = 5 - 1 - 1 = 3$):

$$
\hat{\sigma}^2 = \frac{SSE}{n - p - 1} = \frac{5.10}{3} = \boxed{1.70}
$$

This distinction is important:

- **Prediction $\text{MSE}$** uses $SSE / n$.
- **Residual variance estimate for inference ($\hat{\sigma}^2$)** uses $SSE / (n - p - 1)$.

---

## Part II: Evaluation Metrics & Statistical Inference

### 14. Regression Metrics

#### Mean Absolute Error (MAE)

$$
\boxed{MAE = \frac{1}{n}\sum_{i=1}^{n}\vert y_i - \hat{y}_i \vert}
$$

For the worked example:

$$
MAE = \frac{0.2 + 0.7 + 1.4 + 1.5 + 0.6}{5} = \boxed{0.88}
$$

#### Mean Squared Error (MSE)

$$
\boxed{MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

For the worked example:

$$
MSE = \boxed{1.02}
$$

#### Root Mean Squared Error (RMSE)

$$
\boxed{RMSE = \sqrt{MSE}}
$$

For the worked example:

$$
RMSE = \sqrt{1.02} \approx \boxed{1.01}
$$

#### Mean Absolute Percentage Error (MAPE)

When $y_i \ne 0$ for all observations:

$$
\boxed{MAPE = \frac{100\%}{n}\sum_{i=1}^{n}\left\vert \frac{y_i - \hat{y}_i}{y_i} \right\vert}
$$

#### Metric Intuition

| Metric | Units | Outlier Sensitivity | Interpretation |
| :--- | :--- | :--- | :--- |
| $\text{MAE}$ | Target units ($y$) | Low (Linear penalty) | Average absolute error magnitude. |
| $\text{MSE}$ | Squared units ($y^2$) | High (Quadratic penalty) | Average squared error minimized by $\text{OLS}$. |
| $\text{RMSE}$ | Target units ($y$) | High (Quadratic penalty) | Error in target units with extra sensitivity to large errors. |
| $\text{MAPE}$ | Percentage ($\%$) | Asymmetric near $y_i \to 0$ | Scale-free relative percentage error. |
| $R^2$ | Dimensionless | High | Proportion of variance explained relative to the intercept-only baseline $\bar{y}$. |

---

### 15. R-Squared and Adjusted R-Squared

Define the three sums of squares:

1. **Total Sum of Squares ($\text{SST}$):**

$$
\boxed{SST = \sum_{i=1}^{n}(y_i - \bar{y})^2}
$$

2. **Regression Sum of Squares ($\text{SSR}$):**

$$
\boxed{SSR = \sum_{i=1}^{n}(\hat{y}_i - \bar{y})^2}
$$

3. **Residual Sum of Squares ($\text{SSE}$):**

$$
\boxed{SSE = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2}
$$

With an intercept included in the model, the orthogonality of $\text{OLS}$ residuals guarantees the exact variance decomposition:

$$
\boxed{SST = SSR + SSE}
$$

Therefore, the **Coefficient of Determination ($R^2$)** is:

$$
\boxed{R^2 = \frac{SSR}{SST} = 1 - \frac{SSE}{SST}}
$$

For our worked example ($SST = 17.20$, $SSR = 12.10$, $SSE = 5.10$):

$$
R^2 = 1 - \frac{5.10}{17.20} \approx \boxed{0.7035}
$$

#### Adjusted R-Squared

Because ordinary $R^2$ never decreases when adding predictors, **Adjusted $R^2$** normalizes by degrees of freedom to penalize unnecessary predictors:

$$
\boxed{R^2_{adj} = 1 - \frac{SSE / (n - p - 1)}{SST / (n - 1)} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}}
$$

#### Information Criteria (AIC and BIC)

For model selection across candidate predictor subsets with $k = p + 1$ parameters:

$$
\text{AIC} = n\ln\left(\frac{SSE}{n}\right) + 2k, \qquad \text{BIC} = n\ln\left(\frac{SSE}{n}\right) + k\ln(n)
$$

---

### 16. Degrees of Freedom

Degrees of freedom represent the amount of independent information remaining in the sample after estimating parameters.

For a model with $n$ observations and $p$ predictors plus an intercept ($p + 1$ estimated parameters):

$$
\boxed{df_{total} = n - 1, \qquad df_{model} = p, \qquad df_{residual} = n - p - 1}
$$

There are:

- $n$ observations;
- $p + 1$ fitted coefficients;
- therefore $n - (p + 1) = n - p - 1$ residual degrees of freedom, matching the dimension of the orthogonal complement subspace $\text{Col}(X)^\perp$ ($\text{tr}(I_n - H) = n - p - 1$).

This is why the unbiased residual variance estimator uses $\frac{SSE}{n - p - 1}$ rather than $\frac{SSE}{n}$.

---

### 17. Error Variance

Under the homoscedastic spherical error assumption:

$$
\text{Var}(\epsilon \mid X) = \sigma^2 I_n
$$

the unknown population variance $\sigma^2$ is estimated unbiasedly using:

$$
\boxed{\hat{\sigma}^2 = \frac{SSE}{n - p - 1}}
$$

and the **Residual Standard Error ($\text{RSE}$)** is:

$$
\boxed{\hat{\sigma} = \sqrt{\frac{SSE}{n - p - 1}}}
$$

---

### 18. Standard Error — Derivation

Start from the closed-form $\text{OLS}$ estimator:

$$
\hat{\beta} = (X^T X)^{-1} X^T y
$$

Substitute the population model $y = X\beta + \epsilon$:

$$
\hat{\beta} = (X^T X)^{-1} X^T(X\beta + \epsilon)
$$

Distribute the matrix product:

$$
\hat{\beta} = (X^T X)^{-1} X^T X\beta + (X^T X)^{-1} X^T \epsilon
$$

Since $(X^T X)^{-1} X^T X = I_{p+1}$, we obtain:

$$
\boxed{\hat{\beta} = \beta + (X^T X)^{-1} X^T \epsilon}
$$

Therefore:

$$
\hat{\beta} - \beta = (X^T X)^{-1} X^T \epsilon
$$

Under $\text{Var}(\epsilon \mid X) = \sigma^2 I_n$, the coefficient covariance matrix becomes:

$$
\boxed{\text{Var}(\hat{\beta} \mid X) = \sigma^2 (X^T X)^{-1}}
$$

Replacing $\sigma^2$ by $\hat{\sigma}^2$:

$$
\boxed{\widehat{\text{Var}}(\hat{\beta}) = \hat{\sigma}^2 (X^T X)^{-1}}
$$

The standard error of coefficient $j$ is the square root of the $j$-th diagonal entry:

$$
\boxed{SE(\hat{\beta}_j) = \sqrt{\hat{\sigma}^2 \left[(X^T X)^{-1}\right]_{jj}}}
$$

---

### 19. t-Test and p-Value

To test the null hypothesis:

$$
H_0: \beta_j = \beta_{j,0} \quad \text{vs.} \quad H_1: \beta_j \ne \beta_{j,0}
$$

calculate the $t$-statistic:

$$
\boxed{t = \frac{\hat{\beta}_j - \beta_{j,0}}{SE(\hat{\beta}_j)} \sim t_{n - p - 1}}
$$

For the standard significance test $H_0: \beta_j = 0$, the statistic simplifies to:

$$
\boxed{t = \frac{\hat{\beta}_j}{SE(\hat{\beta}_j)}}
$$

#### Numerical Example

Suppose $\hat{\beta}_1 = 8$ and $SE(\hat{\beta}_1) = 2$. Then:

$$
t = \frac{8}{2} = 4
$$

If $n = 30$ and $p = 2$, then $df = 30 - 2 - 1 = 27$. The two-sided $p$-value is:

$$
p\text{-value} = 2P(T_{27} \ge \vert 4 \vert) \approx 0.00044
$$

#### What Does the p-Value Mean?

A $p$-value is the probability, under the null hypothesis $H_0$ and the assumed statistical model, of observing a test statistic at least as extreme as the one obtained.

It is **not**:

- the probability that $H_0$ is true;
- a measure of whether the coefficient is practically large or important;
- proof that the model specification is causally correct.

---

### 20. Confidence Intervals and Prediction Intervals

#### Confidence Interval for Coefficient $\beta_j$

A $100(1 - \alpha)\%$ confidence interval for coefficient $j$ is:

$$
\boxed{\hat{\beta}_j \pm t_{1 - \alpha/2, \, df} \cdot SE(\hat{\beta}_j)}
$$

**Example:** Suppose $\hat{\beta}_1 = 10$, $SE(\hat{\beta}_1) = 2$, and $t_{0.975, \, df} = 2.06$. Then:

$$
10 \pm (2.06)(2) = 10 \pm 4.12 \implies \boxed{[5.88, \; 14.12]}
$$

#### Confidence Interval for Mean Response vs. Prediction Interval for a New Observation

For a new feature vector $x_* \in \mathbb{R}^{p+1}$ with point prediction $\hat{y}_* = x_*^T \hat{\beta}$:

1. **Confidence Interval for the Mean Response $E[y_* \mid x_*] = x_*^T \beta$:**

$$
\boxed{\hat{y}_* \pm t_{1 - \alpha/2, \, n - p - 1} \, \hat{\sigma}\sqrt{x_*^T(X^T X)^{-1}x_*}}
$$

2. **Prediction Interval for an Individual New Observation $y_* = x_*^T \beta + \epsilon_*$:**

$$
\boxed{\hat{y}_* \pm t_{1 - \alpha/2, \, n - p - 1} \, \hat{\sigma}\sqrt{1 + x_*^T(X^T X)^{-1}x_*}}
$$

Because the prediction interval incorporates both parameter uncertainty and the variance $\sigma^2$ of the new random shock $\epsilon_*$, it is always wider than the confidence interval.

---

### 21. z-Test

The $z$-statistic is:

$$
\boxed{z = \frac{\hat{\theta} - \theta_0}{SE(\hat{\theta})}}
$$

A standard normal $z$-reference is appropriate when the population variance $\sigma^2$ is known *a priori* or when the sample size $n$ is large enough ($df = n - p - 1 \to \infty$) that $t_{df}$ converges in distribution to $N(0, 1)$. For finite-sample regression with unknown error variance, the $t$-distribution is preferred.

---

### 22. F-Test

#### Global Omnibus F-Test

The overall regression null hypothesis is:

$$
H_0: \beta_1 = \beta_2 = \cdots = \beta_p = 0
$$

The $F$-statistic is:

$$
\boxed{F = \frac{SSR / p}{SSE / (n - p - 1)} = \frac{R^2 / p}{(1 - R^2) / (n - p - 1)} \sim F_{p, \, n - p - 1}}
$$

with degrees of freedom $df_1 = p$ and $df_2 = n - p - 1$.

For a single predictor ($p = 1$), the $F$-statistic equals the square of the $t$-statistic:

$$
\boxed{F = t^2}
$$

#### Partial F-Test for Nested Models

To test whether $q$ predictors can be dropped from a full model ($p$ predictors) to a reduced model ($p - q$ predictors):

$$
\boxed{F = \frac{(SSE_{reduced} - SSE_{full}) / q}{SSE_{full} / (n - p - 1)} \sim F_{q, \, n - p - 1}}
$$

---

### 23. Regression Assumptions

#### 23.1 Linearity in Parameters

The conditional expectation is linear in $\beta$:

$$
E[y \mid X] = X\beta
$$

Polynomial features and interaction features still satisfy this assumption because the model remains linear in $\beta$.

#### 23.2 Exogeneity

A critical requirement for unbiasedness is zero conditional mean of the errors:

$$
\boxed{E[\epsilon \mid X] = \mathbf{0}}
$$

Violating this condition (for example, via omitted variables correlated with $X$) produces biased coefficient estimates.

#### 23.3 Independence

Errors across observations are conditionally uncorrelated: $\text{Cov}(\epsilon_i, \epsilon_j \mid X) = 0$ for $i \ne j$. Time-series and clustered data often violate simple independence assumptions.

#### 23.4 Homoscedasticity

Constant conditional variance across observations:

$$
\boxed{\text{Var}(\epsilon_i \mid X) = \sigma^2}
$$

Heteroscedasticity means $\text{Var}(\epsilon_i \mid X) = \sigma_i^2$ changes across observations or predictor values.

#### 23.5 Normality

For exact finite-sample $t$ and $F$ inference:

$$
\boxed{\epsilon \mid X \sim N(\mathbf{0}, \sigma^2 I_n)}
$$

Normality is not required merely to compute the $\text{OLS}$ point estimate or to establish $\text{BLUE}$.

#### 23.6 No Perfect Multicollinearity

No predictor column in $X$ is an exact linear combination of the others: $\text{rank}(X) = p + 1$.

---

### 24. Gauss–Markov Theorem, BLUE, and Maximum Likelihood

#### Gauss–Markov Theorem

Under linearity, full rank, strict exogeneity $E[\epsilon \mid X] = \mathbf{0}$, and spherical errors $\text{Var}(\epsilon \mid X) = \sigma^2 I_n$, the $\text{OLS}$ estimator is:

$$
\boxed{\text{BLUE: Best Linear Unbiased Estimator}}
$$

- **Linear** — estimator $\hat{\beta} = Cy$ is a linear function of the observed target $y$.
- **Unbiased** — expected estimate equals the true parameter: $E[\hat{\beta} \mid X] = \beta$.
- **Best** — has the smallest covariance matrix (in the Loewner positive semi-definite ordering) among all linear unbiased estimators.

#### Equivalence with Maximum Likelihood Estimation (MLE)

Under Gaussian errors $y \mid X \sim N(X\beta, \sigma^2 I_n)$, the log-likelihood is:

$$
\ell(\beta, \sigma^2) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln(\sigma^2) - \frac{1}{2\sigma^2}\Vert y - X\beta \Vert_2^2
$$

Maximizing $\ell(\beta, \sigma^2)$ over $\beta$ is identical to minimizing $\Vert y - X\beta \Vert_2^2$, so $\hat{\beta}_{MLE} = \hat{\beta}_{OLS} = (X^T X)^{-1} X^T y$.

---

### 25. Unbiasedness — Derivation

Start with:

$$
\hat{\beta} = (X^T X)^{-1} X^T y
$$

Substitute $y = X\beta + \epsilon$:

$$
\hat{\beta} = \beta + (X^T X)^{-1} X^T \epsilon
$$

Take conditional expectation given $X$:

$$
E[\hat{\beta} \mid X] = \beta + (X^T X)^{-1} X^T E[\epsilon \mid X]
$$

If $E[\epsilon \mid X] = \mathbf{0}$, then:

$$
\boxed{E[\hat{\beta} \mid X] = \beta}
$$

Therefore, $\text{OLS}$ is conditionally and unconditionally unbiased.

---

### 26. Variance of OLS — Derivation and Robust Standard Errors

From the sampling error identity:

$$
\hat{\beta} - \beta = (X^T X)^{-1} X^T \epsilon
$$

Let $A = (X^T X)^{-1} X^T$. Then $\hat{\beta} - \beta = A\epsilon$, so:

$$
\text{Var}(\hat{\beta} \mid X) = A \, \text{Var}(\epsilon \mid X) \, A^T
$$

Under $\text{Var}(\epsilon \mid X) = \sigma^2 I_n$, we get $\text{Var}(\hat{\beta} \mid X) = \sigma^2 A A^T$. Since $A A^T = (X^T X)^{-1} X^T X (X^T X)^{-1} = (X^T X)^{-1}$:

$$
\boxed{\text{Var}(\hat{\beta} \mid X) = \sigma^2 (X^T X)^{-1}}
$$

#### White's Heteroscedasticity-Consistent (HC1) Robust Standard Errors

When heteroscedasticity is present ($\text{Var}(\epsilon \mid X) = \Omega = \text{diag}(\sigma_1^2, \dots, \sigma_n^2)$), the true covariance is $(X^T X)^{-1}(X^T \Omega X)(X^T X)^{-1}$, which is estimated consistently via the $\text{HC1}$ sandwich estimator:

$$
\boxed{\widehat{\text{Var}}_{HC1}(\hat{\beta}) = \frac{n}{n - p - 1}(X^T X)^{-1}\left(\sum_{i=1}^{n} e_i^2 x_i x_i^T\right)(X^T X)^{-1}}
$$

---

## Part III: Diagnostics, Multicollinearity & Influential Observations

### 27. Multicollinearity

Multicollinearity occurs when predictors contain strong overlapping linear information, such as:

$$
x_2 \approx 2x_1
$$

#### Consequences

- Large standard errors $SE(\hat{\beta}_j)$ because $\left[(X^T X)^{-1}\right]_{jj}$ becomes inflated;
- Unstable coefficient magnitudes and signs across samples;
- Individual predictors appearing statistically non-significant despite a high overall $R^2$ and significant $F$-test;
- In-distribution predictions $\hat{y}$ remain unbiased, though individual coefficients become difficult to interpret separately.

---

### 28. VIF (Variance Inflation Factor)

For predictor $j$, regress $x_j$ on all remaining $p - 1$ predictors and obtain the auxiliary $R_j^2$. Then:

$$
\boxed{VIF_j = \frac{1}{1 - R_j^2}}
$$

For example, if $R_j^2 = 0.8$:

$$
VIF_j = \frac{1}{1 - 0.8} = 5
$$

| $\text{VIF}_j$ Range | Auxiliary $R_j^2$ | Diagnostic Interpretation |
| :--- | :--- | :--- |
| $VIF_j = 1$ | $0.00$ | Orthogonal predictor; zero variance inflation. |
| $1 < VIF_j < 5$ | $0.00 \text{ to } 0.80$ | Mild to moderate correlation; generally acceptable. |
| $5 \le VIF_j < 10$ | $0.80 \text{ to } 0.90$ | Elevated multicollinearity; exercise caution in coefficient interpretation. |
| $VIF_j \ge 10$ | $\ge 0.90$ | Severe multicollinearity; consider Ridge regularization or feature pruning. |

---

### 29. Residual Diagnostics

A regression model should never be judged by a single summary number. Essential diagnostic checks include:

| Diagnostic Plot / Test | Axes / Statistic | What to Look For |
| :--- | :--- | :--- |
| **Residuals vs. Fitted Values** | $e_i$ vs. $\hat{y}_i$ | Curvature indicates nonlinearity; funnel shape indicates heteroscedasticity. |
| **Normal Q–Q Plot** | Standardized $e_i$ vs. $N(0, 1)$ quantiles | Deviations from the $45^\circ$ line indicate heavy tails or skewness. |
| **Scale–Location Plot** | $\sqrt{\vert r_i \vert}$ vs. $\hat{y}_i$ | Sloping trend line indicates non-constant residual variance. |
| **Residuals vs. Predictors** | $e_i$ vs. $x_{ij}$ | Unmodeled nonlinear patterns in specific features. |
| **Durbin–Watson Statistic** | $DW = \frac{\sum_{t=2}^n(e_t - e_{t-1})^2}{\sum_{t=1}^n e_t^2}$ | $DW \approx 2$ indicates no serial correlation; $DW \ll 2$ signals positive autocorrelation. |

---

### 30. Hat Matrix and Leverage

The Hat Matrix is:

$$
\boxed{H = X(X^T X)^{-1} X^T, \qquad \hat{y} = Hy}
$$

$H$ is symmetric ($H^T = H$) and idempotent ($H^2 = H$), with $\text{tr}(H) = p + 1$. Its diagonal entries $h_{ii}$ are the **leverage** values:

$$
\boxed{h_{ii} = x_i^T (X^T X)^{-1} x_i, \qquad \frac{1}{n} \le h_{ii} \le 1, \qquad \sum_{i=1}^{n} h_{ii} = p + 1}
$$

High leverage ($h_{ii} > \frac{2(p + 1)}{n}$) means observation $i$ has unusual predictor values relative to the center of the dataset.

---

### 31. Cook's Distance

Cook's Distance measures the overall influence of observation $i$ on all fitted values simultaneously:

$$
\boxed{D_i = \frac{e_i^2}{(p + 1)\hat{\sigma}^2}\frac{h_{ii}}{(1 - h_{ii})^2} = \frac{r_i^2}{p + 1}\left(\frac{h_{ii}}{1 - h_{ii}}\right)}
$$

where $r_i = \frac{e_i}{\hat{\sigma}\sqrt{1 - h_{ii}}}$ is the internally studentized residual. Observations with $D_i > \frac{4}{n}$ or $D_i > 1$ warrant close inspection, though high $D_i$ is a diagnostic signal rather than an automatic deletion rule.

---

## Part IV: Optimization, Numerical Stability & Extended Estimators

### 32. Gradient Descent

Closed-form matrix factorization is not the only way to fit linear regression. For very high-dimensional problems, we minimize the scaled objective:

$$
\boxed{J(\beta) = \frac{1}{2n}\Vert X\beta - y \Vert_2^2}
$$

Gradient Descent repeatedly updates the parameter vector:

$$
\boxed{\beta^{(t+1)} = \beta^{(t)} - \alpha\nabla J(\beta^{(t)})}
$$

where $\alpha > 0$ is the learning rate.

---

### 33. Gradient Descent — Full Derivation

Start with:

$$
J(\beta) = \frac{1}{2n}(X\beta - y)^T(X\beta - y)
$$

Expand the quadratic form:

$$
J(\beta) = \frac{1}{2n}\left[\beta^T X^T X\beta - 2\beta^T X^T y + y^T y\right]
$$

Differentiate with respect to $\beta$:

$$
\nabla J(\beta) = \frac{1}{2n}\left[2X^T X\beta - 2X^T y\right]
$$

Therefore, the exact gradient is:

$$
\boxed{\nabla J(\beta) = \frac{1}{n}X^T(X\beta - y)}
$$

and the Batch Gradient Descent update rule is:

$$
\boxed{\beta \leftarrow \beta - \frac{\alpha}{n}X^T(X\beta - y)}
$$

Convergence is guaranteed when $0 < \alpha < \frac{2}{\lambda_{\max}\left(\frac{1}{n}X^T X\right)}$.

---

### 34. Feature Scaling

Gradient Descent converges slowly when features have drastically different scales, such as:

$$
x_1 \in [0, 1] \quad \text{while} \quad x_2 \in [0, 1000000]
$$

Standardization transforms each predictor to zero mean and unit variance:

$$
\boxed{z = \frac{x - \mu}{\sigma}}
$$

Scaling is especially important for:

- Gradient Descent (improves conditioning of the loss surface);
- Ridge, Lasso, and Elastic Net (ensures penalties are applied equally across features);
- Distance-based algorithms.

For train/test workflows, always calculate $\mu$ and $\sigma$ using the **training data only**.

---

### 35. Numerical Stability (QR and SVD Decompositions)

Explicitly forming $X^T X$ squares the condition number of the design matrix:

$$
\kappa(X^T X) = \left[\kappa(X)\right]^2
$$

Avoid explicitly calling `np.linalg.inv(X.T @ X)`. Instead, use stable decompositions:

1. **QR Decomposition:** Factor $X = QR$ where $Q^T Q = I_{p+1}$ and $R$ is upper triangular, then solve the triangular system:

$$
\boxed{R\hat{\beta} = Q^T y}
$$

2. **Singular Value Decomposition ($\text{SVD}$):** Factor $X = U\Sigma V^T$ and compute the minimum-norm pseudoinverse solution:

$$
\boxed{\hat{\beta} = X^+ y = V\Sigma^+ U^T y}
$$

If the columns of $X$ are linearly dependent ($\text{rank}(X) < p + 1$), parameters are not uniquely identifiable in the classical inverse formula, whereas $\text{SVD}$ (`np.linalg.lstsq`) still returns the unique minimum-norm solution.

---

### 36. WLS (Weighted Least Squares)

Weighted Least Squares minimizes the weighted sum of squared residuals for diagonal weight matrix $W = \text{diag}(w_1, \dots, w_n)$:

$$
\boxed{J_{WLS}(\beta) = (y - X\beta)^T W(y - X\beta)}
$$

Differentiate with respect to $\beta$:

$$
\nabla J_{WLS}(\beta) = -2X^T Wy + 2X^T WX\beta = \mathbf{0}
$$

Solving the weighted normal equation $X^T WX\hat{\beta}_{WLS} = X^T Wy$ yields:

$$
\boxed{\hat{\beta}_{WLS} = (X^T WX)^{-1} X^T Wy}
$$

$\text{WLS}$ is optimal when observations have unequal error variances $\sigma_i^2$ and weights are chosen as $w_i = 1 / \sigma_i^2$.

---

### 37. GLS (Generalized Least Squares)

Suppose the error covariance matrix is a general symmetric positive definite matrix:

$$
\text{Var}(\epsilon \mid X) = \Omega
$$

Then Generalized Least Squares ($\text{GLS}$) gives the $\text{BLUE}$ estimator:

$$
\boxed{\hat{\beta}_{GLS} = (X^T \Omega^{-1} X)^{-1} X^T \Omega^{-1} y}
$$

- $\text{OLS}$ is the special case $\Omega = \sigma^2 I_n$.
- $\text{WLS}$ is the special case where $\Omega$ is diagonal.

---

### 38. Ridge Regression

Ridge Regression adds an $L_2$ penalty on the slope coefficients (leaving the intercept $\beta_0$ unpenalized via $I_0 = \text{diag}(0, 1, \dots, 1)$):

$$
\boxed{J_{ridge}(\beta) = SSE + \lambda\sum_{j=1}^{p}\beta_j^2}
$$

The closed-form solution is:

$$
\boxed{\hat{\beta}_{ridge} = (X^T X + \lambda I_0)^{-1} X^T y}
$$

- $\lambda = 0$ recovers ordinary least squares ($\text{OLS}$).
- Larger $\lambda > 0$ increases coefficient shrinkage, reducing variance under multicollinearity at the cost of a small bias.

---

### 39. Lasso and Elastic Net Regression

#### Lasso Regression ($L_1$ Penalty)

Lasso uses an $L_1$ absolute-value penalty:

$$
\boxed{J_{lasso}(\beta) = SSE + \lambda\sum_{j=1}^{p}\vert \beta_j \vert}
$$

Because of the sharp geometry of the $L_1$ norm, Lasso can drive coefficients **exactly to zero**, performing automatic feature selection.

#### Elastic Net Regression ($L_1 + L_2$ Penalty)

Elastic Net combines $L_1$ sparsity and $L_2$ stability for correlated predictors:

$$
\boxed{J_{enet}(\beta) = \frac{1}{2n}\Vert y - X\beta \Vert_2^2 + \lambda\left(\rho\sum_{j=1}^{p}\vert \beta_j \vert + \frac{1 - \rho}{2}\sum_{j=1}^{p}\beta_j^2\right)}
$$

---

### 40. Polynomial Regression

Polynomial regression adds powers of $x$ as engineered features:

$$
\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3
$$

This is nonlinear in $x$ but **linear in $\beta$**. The design matrix is the Vandermonde matrix:

$$
X = \begin{bmatrix} 1 & x_1 & x_1^2 & x_1^3 \\ 1 & x_2 & x_2^2 & x_2^3 \\ \vdots & \vdots & \vdots & \vdots \end{bmatrix}
$$

---

### 41. Interaction Terms

An interaction model includes product terms between features:

$$
\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1 x_2
$$

The marginal effect of $x_1$ is:

$$
\boxed{\frac{\partial \hat{y}}{\partial x_1} = \beta_1 + \beta_3 x_2}
$$

Therefore, the slope with respect to $x_1$ depends on the value of $x_2$.

---

## Part V: Machine Learning Validation, Implementation & Reference

### 42. Train/Test Split

A disciplined supervised-learning workflow separates data before fitting any parameters:

```text
                 Dataset
                    |
             +------+------+
             |             |
          Train           Test
             |             |
          Fit model     Final evaluation (once)
```

The test set represents unseen data and must never be used to iteratively tune the model.

---

### 43. Cross-Validation

$K$-fold cross-validation divides the training data into $k$ folds:

```text
Fold 1    -> Validation
Folds 2..k -> Training
```

Rotating the validation fold across all $k$ partitions yields the mean validation error. For $\text{OLS}$, Leave-One-Out Cross-Validation ($\text{LOOCV}$) has an exact closed-form shortcut ($\text{PRESS}$ statistic) using the Hat Matrix leverages $h_{ii}$:

$$
\boxed{CV_{(n)} = \frac{1}{n}\sum_{i=1}^{n}\left(\frac{e_i}{1 - h_{ii}}\right)^2}
$$

---

### 44. Data Leakage

Data leakage happens when information from outside the training set influences the training process.

```text
INCORRECT (Data Leakage):
Full dataset -> Fit scaler -> Train/Test split

CORRECT (Leakage-Free):
Train/Test split -> Fit scaler on train only -> Transform train -> Transform test using train parameters
```

The same strict rule applies to imputation, feature selection, target encoding, and dimensionality reduction.

---

### 45. Bias–Variance Tradeoff

Expected out-of-sample squared error decomposes into three components:

$$
\boxed{E\left[(y_* - \hat{f}(x_*))^2\right] = \text{Bias}^2 + \text{Variance} + \sigma^2}
$$

```text
Too simple      -> High bias (Underfitting)
Good balance    -> Good generalization
Too flexible    -> High variance (Overfitting)
```

---

### 46. Implementation From Scratch

#### OLS Using a Stable Least-Squares Solver

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

#### Batch Gradient Descent

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

#### Full Reference Estimator Class (With Inference, Ridge, Leverage & Cook's Distance)

```python
import numpy as np


class ReferenceLinearRegression:
    """Reference OLS / Ridge estimator with statistical inference and diagnostics."""

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

        y_hat = Xd @ beta
        residuals = y - y_hat
        sse = float(residuals @ residuals)
        sst = float(np.sum((y - np.mean(y)) ** 2))
        df_res = max(n - k, 1)

        self.sigma2_ = sse / df_res
        self.r2_ = 1.0 - (sse / sst) if sst > 0 else 0.0
        self.adj_r2_ = 1.0 - ((1.0 - self.r2_) * (n - 1) / df_res)

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

The standard user-facing flow in `OnithrasML` is:

```python
import numpy as np
from onithrasML.linear_model import LinearRegression

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5]
])

y = np.array([3, 5, 7, 9, 11])

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)

print(model.intercept_)
print(model.coef_)
print(predictions)
```

---

### 48. Complete Regression Workflow

A production-minded regression workflow:

```text
1. Understand the problem
        |
        v
2. Inspect the dataset
        |
        v
3. Define X and y
        |
        v
4. Split train/test
        |
        v
5. Fit preprocessing on train only
        |
        v
6. Establish a baseline
        |
        v
7. Fit Linear Regression
        |
        v
8. Evaluate MAE / MSE / RMSE / R^2
        |
        v
9. Inspect residuals
        |
        v
10. Check multicollinearity (VIF)
        |
        v
11. Check influential observations (Leverage & Cook's Distance)
        |
        v
12. Perform statistical inference when appropriate
        |
        v
13. Compare Ridge / Lasso / WLS / GLS if needed
        |
        v
14. Cross-validate model choices
        |
        v
15. Evaluate once on the untouched test set
        |
        v
16. Document assumptions and limitations
```

---

### 49. Common Mistakes

| Mistake | Why It Is Problematic | Correct Practice |
| :--- | :--- | :--- |
| **1. Treating $R^2$ as accuracy** | $R^2$ measures continuous variance reduction, not classification accuracy. | Report $\text{RMSE}$ and $\text{MAE}$ in target units alongside $R^2$. |
| **2. Thinking high $R^2$ proves a good model** | High $R^2$ can coexist with data leakage, nonlinearity, or severe overfitting. | Inspect residual plots and out-of-sample validation error. |
| **3. Treating $p$-value as effect size** | A tiny $p$-value in a large sample does not mean the slope is practically large. | Report point estimates $\hat{\beta}_j$ and confidence intervals. |
| **4. Confusing residuals $e$ and errors $\epsilon$** | $e = y - \hat{y}$ is observed; $\epsilon = y - X\beta$ is the unobserved population disturbance. | Use studentized residuals $r_i$ to account for leverage $h_{ii}$. |
| **5. Explicit matrix inversion** | `np.linalg.inv(X.T @ X)` squares the condition number and loses precision. | Use `np.linalg.lstsq(X, y, rcond=None)` or `np.linalg.solve`. |
| **6. Ignoring feature scale in optimization** | Unscaled predictors slow down Gradient Descent and distort Ridge/Lasso penalties. | Standardize features ($z = \frac{x - \mu}{\sigma}$) using training statistics. |
| **7. Using the test set repeatedly** | Repeatedly tuning against the test set causes test-set leakage. | Use $K$-fold cross-validation for tuning; evaluate on test once. |
| **8. Removing outliers automatically** | High-$D_i$ points may be valid rare regimes or signs of model misspecification. | Investigate data provenance and compare sensitivity before removal. |

---

### 50. Formula Sheet

| Concept | Mathematical Formula |
| :--- | :--- |
| **1. Model** | $y = X\beta + \epsilon$ |
| **2. Prediction** | $\hat{y} = X\hat{\beta}$ |
| **3. Residual** | $e = y - \hat{y} = (I_n - H)y$ |
| **4. SSE** | $\displaystyle SSE = \sum_{i=1}^{n}(y_i - \hat{y}_i)^2$ |
| **5. MSE** | $\displaystyle MSE = \frac{SSE}{n}$ |
| **6. RMSE** | $\displaystyle RMSE = \sqrt{MSE}$ |
| **7. MAE** | $\displaystyle MAE = \frac{1}{n}\sum_{i=1}^{n}\vert y_i - \hat{y}_i \vert$ |
| **8. Simple Slope** | $\displaystyle \hat{\beta}_1 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2}$ |
| **9. Simple Intercept** | $\hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x}$ |
| **10. Normal Equation** | $X^T X\hat{\beta} = X^T y$ |
| **11. Closed-Form OLS** | $\hat{\beta} = (X^T X)^{-1} X^T y$ |
| **12. Error Variance** | $\displaystyle \hat{\sigma}^2 = \frac{SSE}{n - p - 1}$ |
| **13. Coefficient Covariance** | $\widehat{\text{Var}}(\hat{\beta}) = \hat{\sigma}^2(X^T X)^{-1}$ |
| **14. Standard Error** | $\displaystyle SE(\hat{\beta}_j) = \sqrt{\hat{\sigma}^2\left[(X^T X)^{-1}\right]_{jj}}$ |
| **15. t-Statistic** | $\displaystyle t = \frac{\hat{\beta}_j - \beta_{j,0}}{SE(\hat{\beta}_j)}$ |
| **16. Confidence Interval** | $\hat{\beta}_j \pm t_{1 - \alpha/2, \, df}SE(\hat{\beta}_j)$ |
| **17. R-Squared** | $\displaystyle R^2 = \frac{SSR}{SST} = 1 - \frac{SSE}{SST}$ |
| **18. Adjusted R-Squared** | $\displaystyle R^2_{adj} = 1 - \frac{(1 - R^2)(n - 1)}{n - p - 1}$ |
| **19. F-Statistic** | $\displaystyle F = \frac{SSR / p}{SSE / (n - p - 1)}$ |
| **20. VIF** | $\displaystyle VIF_j = \frac{1}{1 - R_j^2}$ |
| **21. Hat Matrix** | $H = X(X^T X)^{-1} X^T$ |
| **22. Cook's Distance** | $\displaystyle D_i = \frac{e_i^2}{(p + 1)\hat{\sigma}^2}\frac{h_{ii}}{(1 - h_{ii})^2}$ |
| **23. Gradient Vector** | $\displaystyle \nabla J(\beta) = \frac{1}{n}X^T(X\beta - y)$ |
| **24. Gradient Update** | $\beta \leftarrow \beta - \alpha\nabla J(\beta)$ |
| **25. WLS Estimator** | $\hat{\beta}_{WLS} = (X^T WX)^{-1} X^T Wy$ |
| **26. GLS Estimator** | $\hat{\beta}_{GLS} = (X^T \Omega^{-1} X)^{-1} X^T \Omega^{-1} y$ |
| **27. Ridge Estimator** | $\hat{\beta}_{ridge} = (X^T X + \lambda I_0)^{-1} X^T y$ |

---

### 51. Concept Map

```text
                         LINEAR REGRESSION
                                |
              +-----------------+-----------------+
              |                 |                 |
          Mathematics       Statistics        Implementation
              |                 |                 |
      +-------+-------+    +----+----+       +----+----+
      |       |       |    |         |       |         |
   Matrix  Calculus  Geometry  Inference  Diagnostics  API
      |       |       |    |         |       |         |
     X^TX  Gradient Projection t/F/z     VIF/Cook   fit/predict
      |       |       |    |         |
      +-------+-------+----+---------+
                      |
                 OLS Estimate
                      |
                      v
            beta_hat = (X^T X)^(-1) X^T y
```

| Pillar | Core Subfields | Key Mathematical Objects | Primary Output |
| :--- | :--- | :--- | :--- |
| **1. Mathematics** | Matrix Algebra, Calculus, Geometry | Gram Matrix $X^T X$, Gradient $\nabla_\beta J$, Projection $H = X(X^T X)^{-1}X^T$, Orthogonality $e \perp \text{Col}(X)$ | $\hat{\beta} = (X^T X)^{-1}X^T y$ |
| **2. Statistics** | Estimation, Inference, Diagnostics | Variance $\hat{\sigma}^2$, Covariance $\sigma^2(X^T X)^{-1}$, $t / F / z$ Tests, $\text{VIF}_j$, Leverage $h_{ii}$, Cook's $D_i$ | $\hat{\beta}_j \pm t_{1-\alpha/2, \, df}SE(\hat{\beta}_j)$ |
| **3. Implementation** | Numerical Solvers, Regularization, Validation | QR / SVD Solvers (`lstsq`), Gradient Descent, Ridge / Lasso / $\text{WLS}$ / $\text{GLS}$, Cross-Validation | $\hat{y} = X\hat{\beta}$ |

---

### 52. Learning Checklist

A strong understanding of Linear Regression includes:

- [ ] Simple Linear Regression
- [ ] Multiple Linear Regression
- [ ] Design Matrix $X \in \mathbb{R}^{n \times (p+1)}$
- [ ] Intercept column $\mathbf{1}_n$
- [ ] Statistical model $y = X\beta + \epsilon$
- [ ] Residuals $e = y - \hat{y}$
- [ ] SSE
- [ ] MSE
- [ ] RMSE
- [ ] MAE
- [ ] Least Squares
- [ ] OLS
- [ ] Normal Equation $X^T X\hat{\beta} = X^T y$
- [ ] Full matrix derivation
- [ ] Geometric interpretation
- [ ] Gradient Descent
- [ ] Gradient derivation
- [ ] Learning rate $\alpha$
- [ ] Feature scaling
- [ ] R-Squared ($R^2$)
- [ ] Adjusted R-Squared ($R^2_{adj}$)
- [ ] Degrees of freedom ($df = n - p - 1$)
- [ ] Error variance $\hat{\sigma}^2$
- [ ] Standard Error $SE(\hat{\beta}_j)$
- [ ] t-Test
- [ ] p-Value
- [ ] Confidence Interval & Prediction Interval
- [ ] z-Test
- [ ] F-Test
- [ ] Regression assumptions
- [ ] Exogeneity $E[\epsilon \mid X] = \mathbf{0}$
- [ ] Homoscedasticity vs. Heteroscedasticity
- [ ] Multicollinearity
- [ ] VIF
- [ ] Residual diagnostics
- [ ] Hat Matrix $H$
- [ ] Leverage $h_{ii}$
- [ ] Cook's Distance $D_i$
- [ ] Gauss–Markov theorem & BLUE
- [ ] Unbiasedness derivation
- [ ] OLS variance derivation & White's robust standard errors
- [ ] WLS
- [ ] GLS
- [ ] Ridge
- [ ] Lasso & Elastic Net
- [ ] Polynomial Regression
- [ ] Interaction terms
- [ ] Train/Test Split
- [ ] Cross-Validation & LOOCV PRESS
- [ ] Data Leakage
- [ ] Bias–Variance Tradeoff
- [ ] Numerical Stability (QR & SVD)
- [ ] Rank
- [ ] NumPy implementation
- [ ] Gradient Descent implementation
- [ ] OnithrasML API usage

---

## Final Mental Model

If you understand these five equations, you understand the mathematical core of Linear Regression:

### 1. Model

$$
y = X\beta + \epsilon
$$

### 2. Objective

$$
\min_\beta \Vert y - X\beta \Vert_2^2
$$

### 3. Normal Equation

$$
X^T X\hat{\beta} = X^T y
$$

### 4. Estimate

$$
\hat{\beta} = (X^T X)^{-1} X^T y
$$

### 5. Prediction

$$
\hat{y} = X\hat{\beta}
$$

Everything else — residuals, metrics, standard errors, hypothesis tests, confidence intervals, diagnostics, Ridge, Lasso, $\text{WLS}$, $\text{GLS}$, and Gradient Descent — builds on this foundation.

---

## Documentation Principle

`OnithrasML` documentation answers three questions for every major feature:

1. **How do I use it?**
2. **Why does it work mathematically?**
3. **When should I trust or question the result?**

That makes the documentation useful both as a package reference and as a learning resource.

---

**OnithrasML — Linear Regression: from `fit(X, y)` to the mathematics behind the estimator.**