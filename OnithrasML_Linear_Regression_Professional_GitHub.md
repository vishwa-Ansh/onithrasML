# OnithrasML --- Linear Regression

> A production-oriented Linear Regression reference covering usage,
> mathematics, estimation, diagnostics, statistical inference,
> optimization, and implementation.

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Machine
Learning](https://img.shields.io/badge/Type-Supervised%20Learning-purple.svg)](#overview)
[![Status](https://img.shields.io/badge/Documentation-Reference-success.svg)](#)

------------------------------------------------------------------------

## Table of Contents

-   [Overview](#overview)
-   [Why Linear Regression](#why-linear-regression)
-   [Quick Start](#quick-start)
-   [API Reference](#api-reference)
-   [Problem Formulation](#problem-formulation)
-   [Simple Linear Regression](#simple-linear-regression)
-   [Multiple Linear Regression](#multiple-linear-regression)
-   [Design Matrix](#design-matrix)
-   [Least Squares and OLS](#least-squares-and-ols)
-   [Normal Equation](#normal-equation)
-   [Worked Numerical Example](#worked-numerical-example)
-   [Prediction and Residuals](#prediction-and-residuals)
-   [Regression Metrics](#regression-metrics)
-   [R² and Adjusted R²](#r²-and-adjusted-r²)
-   [Statistical Inference](#statistical-inference)
-   [Standard Error](#standard-error)
-   [t-Test and p-Value](#t-test-and-p-value)
-   [Confidence Intervals](#confidence-intervals)
-   [z-Test](#z-test)
-   [F-Test](#f-test)
-   [Regression Assumptions](#regression-assumptions)
-   [Multicollinearity and VIF](#multicollinearity-and-vif)
-   [Residual Diagnostics](#residual-diagnostics)
-   [Leverage and Hat Matrix](#leverage-and-hat-matrix)
-   [Cook's Distance](#cooks-distance)
-   [Gradient Descent](#gradient-descent)
-   [Numerical Stability](#numerical-stability)
-   [WLS and GLS](#wls-and-gls)
-   [Regularization](#regularization)
-   [Polynomial and Interaction
    Terms](#polynomial-and-interaction-terms)
-   [Model Validation](#model-validation)
-   [Implementation From Scratch](#implementation-from-scratch)
-   [End-to-End Workflow](#end-to-end-workflow)
-   [Choosing the Right Approach](#choosing-the-right-approach)
-   [Common Mistakes](#common-mistakes)
-   [Formula Reference](#formula-reference)
-   [Glossary](#glossary)
-   [Learning Checklist](#learning-checklist)

------------------------------------------------------------------------

# Overview

Linear Regression is a supervised learning method for modeling a
continuous target as a linear function of one or more predictors.

For a feature matrix (X), target vector (y), and coefficient vector
(`\beta`):

$$
y = X\beta + \epsilon
$$

After fitting the model, the estimated prediction is:

$$
\hat{y}=X\hat{\beta}
$$

The central estimation problem is:

$$
\hat{\beta}
=
\arg\min_{\beta}
\|y-X\beta\|_2^2
$$

This connects:

-   **Linear Algebra** --- vectors, matrices, rank, projections
-   **Calculus** --- gradients and optimization
-   **Statistics** --- estimation, uncertainty, hypothesis testing
-   **Probability** --- error distributions and conditional expectations
-   **Numerical Methods** --- stable solutions of linear systems
-   **Machine Learning** --- generalization, validation, regularization

> **Key idea:** Linear Regression is not just the line
> `model.fit(X, y)`. That call represents a complete estimation
> pipeline.

------------------------------------------------------------------------

# Why Linear Regression

Linear Regression is useful when:

-   the target is continuous;
-   interpretability matters;
-   a strong statistical baseline is required;
-   the relationship is approximately linear in the model parameters;
-   coefficient-level inference is useful;
-   training speed and simplicity are important.

It is also an excellent baseline for comparing more complex models.

------------------------------------------------------------------------

# Quick Start

## Installation

``` bash
pip install onithrasML
```

## Minimal Example

``` python
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

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
print("Predictions:", predictions)
```

The underlying relationship is:

$$
y=1+2x
$$

so the fitted model should recover coefficients close to:

$$
\hat{\beta}_0=1,\qquad \hat{\beta}_1=2
$$

------------------------------------------------------------------------

# API Reference

## `LinearRegression`

The primary estimator for fitting a linear regression model.

### Constructor

``` python
LinearRegression(...)
```

The exact constructor arguments depend on the current OnithrasML
implementation.

### Core methods

  Method          Purpose
  --------------- ------------------------------------------------------
  `fit(X, y)`     Estimate model parameters from training data
  `predict(X)`    Generate predictions
  `score(X, y)`   Return the implementation's model score, if provided

### Common learned attributes

  Attribute      Meaning
  -------------- --------------------------------
  `coef_`        Estimated feature coefficients
  `intercept_`   Estimated intercept

> **Implementation note:** Keep the public API stable and document
> constructor parameters directly from the actual OnithrasML source so
> that this reference never becomes out of sync with the package.

------------------------------------------------------------------------

# Problem Formulation

The classical linear model is:

$$
y=X\beta+\epsilon
$$

where:

-   (y`\in``\mathbb{R}`\^{n}) is the observed target vector;
-   (X`\in``\mathbb{R}`\^{n`\times `p}) is the design
    matrix;
-   (`\beta``\in``\mathbb{R}`\^{p}) is the coefficient
    vector;
-   (`\epsilon``\in``\mathbb{R}`\^{n}) is the error
    vector.

With an intercept, the design matrix contains one additional column:

$$
X\in\mathbb{R}^{n\times(p+1)}
$$

and:

$$
\beta=
\begin{bmatrix}
\beta_0\\
\beta_1\\
\vdots\\
\beta_p
\end{bmatrix}
$$

------------------------------------------------------------------------

# Simple Linear Regression

With one predictor:

$$
\boxed{\hat{y}=\beta_0+\beta_1x}
$$

## Intercept

(`\beta`\_0) is the predicted value when (x=0), provided that
(x=0) is meaningful in the problem domain.

## Slope

(`\beta`\_1) describes the change in predicted (y) associated with
a one-unit increase in (x).

For simple regression:

$$
\boxed{
\hat{\beta}_1=
\frac{\sum_i(x_i-\bar{x})(y_i-\bar{y})}
{\sum_i(x_i-\bar{x})^2}
}
$$

and:

$$
\boxed{
\hat{\beta}_0=\bar{y}-\hat{\beta}_1\bar{x}
}
$$

Equivalently:

$$
\hat{\beta}_1=
\frac{\operatorname{Cov}(X,Y)}
{\operatorname{Var}(X)}
$$

when covariance and variance use consistent normalization.

------------------------------------------------------------------------

# Multiple Linear Regression

With (p) predictors:

$$
\boxed{
\hat{y}
=
\beta_0+
\beta_1x_1+
\beta_2x_2+
\cdots+
\beta_px_p
}
$$

For example:

$$
\hat{y}=100+5x_1-2x_2+10x_3
$$

The coefficient (5) means that, within the model, a one-unit increase in
(x_1) is associated with a five-unit increase in predicted (y),
**holding the other included predictors constant**.

------------------------------------------------------------------------

# Design Matrix

Suppose:

$$
\hat{y}=\beta_0+\beta_1x_1+\beta_2x_2
$$

Then:

$$
X=
\begin{bmatrix}
1&x_{11}&x_{12}\\
1&x_{21}&x_{22}\\
\vdots&\vdots&\vdots\\
1&x_{n1}&x_{n2}
\end{bmatrix}
$$

and:

$$
\beta=
\begin{bmatrix}
\beta_0\\
\beta_1\\
\beta_2
\end{bmatrix}
$$

Therefore:

$$
\hat{y}=X\beta
$$

The first column of ones represents the intercept.

------------------------------------------------------------------------

# Least Squares and OLS

For observation (i), the residual is:

$$
e_i=y_i-\hat{y}_i
$$

If residuals were simply added:

$$
\sum_i e_i
$$

positive and negative errors could cancel.

OLS therefore minimizes squared residuals:

$$
\boxed{
SSE=
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
$$

In matrix form:

$$
\boxed{
SSE=
(y-X\beta)^T(y-X\beta)
}
$$

Ordinary Least Squares estimates:

$$
\boxed{
\hat{\beta}
=
\arg\min_\beta
(y-X\beta)^T(y-X\beta)
}
$$

------------------------------------------------------------------------

# Normal Equation

Starting from:

$$
S(\beta)=(y-X\beta)^T(y-X\beta)
$$

expand:

$$
S(\beta)
=
y^Ty-2\beta^TX^Ty+\beta^TX^TX\beta
$$

Differentiate with respect to (`\beta`):

$$
\nabla S(\beta)
=
-2X^Ty+2X^TX\beta
$$

At the optimum:

$$
\nabla S(\beta)=0
$$

Therefore:

$$
X^TX\hat{\beta}=X^Ty
$$

This is the **Normal Equation**.

If (X\^TX) is invertible:

$$
\boxed{
\hat{\beta}=(X^TX)^{-1}X^Ty
}
$$

> **Numerical practice:** The equation is mathematically correct, but
> production code should usually solve the linear system directly rather
> than explicitly forming ((X^TX)^{-1}).

------------------------------------------------------------------------

# Worked Numerical Example

Consider:

$$
x=[1,2,3,4,5]
$$

and:

$$
y=[3,5,4,8,7]
$$

We fit:

$$
\hat{y}=\beta_0+\beta_1x
$$

## Step 1 --- Means

$$
\bar{x}=3
$$

$$
\bar{y}=5.4
$$

## Step 2 --- Slope

$$
S_{xy}
=
\sum_i(x_i-\bar{x})(y_i-\bar{y})
=11
$$

$$
S_{xx}
=
\sum_i(x_i-\bar{x})^2
=10
$$

Therefore:

$$
\hat{\beta}_1=\frac{11}{10}=1.1
$$

## Step 3 --- Intercept

$$
\hat{\beta}_0
=
\bar{y}-\hat{\beta}_1\bar{x}
$$

$$
=5.4-(1.1)(3)=2.1
$$

The fitted model is:

$$
\boxed{\hat{y}=2.1+1.1x}
$$

## Step 4 --- Predictions

$$
\hat{y}=[3.2,4.3,5.4,6.5,7.6]
$$

## Step 5 --- Residuals

$$
e=y-\hat{y}
$$

$$
e=[-0.2,0.7,-1.4,1.5,-0.6]
$$

## Step 6 --- SSE

$$
SSE=
(-0.2)^2+
(0.7)^2+
(-1.4)^2+
(1.5)^2+
(-0.6)^2
$$

$$
\boxed{SSE=5.10}
$$

## Step 7 --- MSE

$$
MSE=\frac{5.10}{5}=1.02
$$

## Step 8 --- RMSE

$$
RMSE=\sqrt{1.02}\approx1.01
$$

## Step 9 --- MAE

$$
MAE=
\frac{0.2+0.7+1.4+1.5+0.6}{5}
=0.88
$$

------------------------------------------------------------------------

# Prediction and Residuals

Prediction:

$$
\hat{y}=X\hat{\beta}
$$

Residual:

$$
\boxed{e=y-\hat{y}}
$$

A positive residual means:

$$
y>\hat{y}
$$

so the model under-predicted.

A negative residual means:

$$
y<\hat{y}
$$

so the model over-predicted.

------------------------------------------------------------------------

# Regression Metrics

## MAE

$$
\boxed{
MAE=
\frac1n\sum_{i=1}^{n}|y_i-\hat{y}_i|
}
$$

MAE is easy to interpret because it has the same unit as the target.

## MSE

$$
\boxed{
MSE=
\frac1n\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
}
$$

MSE penalizes large errors more strongly because the error is squared.

## RMSE

$$
\boxed{
RMSE=\sqrt{MSE}
}
$$

RMSE has the same unit as the target.

### Which metric should be used?

  Metric   Useful when
  -------- ----------------------------------------------------------
  MAE      Robust, easy-to-interpret average error
  MSE      Squared-error optimization
  RMSE     Want target-unit error with stronger large-error penalty
  R²       Want variance-explained style comparison

------------------------------------------------------------------------

# R² and Adjusted R²

Define:

$$
SST=\sum_i(y_i-\bar{y})^2
$$

and:

$$
SSE=\sum_i(y_i-\hat{y}_i)^2
$$

For a model with an intercept:

$$
SST=SSR+SSE
$$

R² is:

$$
\boxed{
R^2=1-\frac{SSE}{SST}
}
$$

It measures improvement relative to the intercept-only baseline.

## Adjusted R²

For (n) observations and (p) predictors:

$$
\boxed{
R^2_{adj}
=
1-
\frac{(1-R^2)(n-1)}
{n-p-1}
}
$$

Unlike ordinary R², adjusted R² can decrease when an additional
predictor does not provide enough explanatory value.

> **Important:** Neither R² nor adjusted R² proves causation, validates
> every model assumption, or guarantees good out-of-sample predictions.

------------------------------------------------------------------------

# Statistical Inference

Prediction and inference answer different questions.

### Prediction

> What value of (y) does the model predict?

### Inference

> What can we say about the underlying coefficients and uncertainty?

Under the classical linear model:

$$
E[\epsilon\mid X]=0
$$

and, for the usual homoscedastic setup:

$$
Var(\epsilon\mid X)=\sigma^2I
$$

the OLS estimator has:

$$
E[\hat{\beta}\mid X]=\beta
$$

and:

$$
\boxed{
Var(\hat{\beta}\mid X)
=
\sigma^2(X^TX)^{-1}
}
$$

------------------------------------------------------------------------

# Standard Error

The residual degrees of freedom are:

$$
\boxed{
df=n-p-1
}
$$

Estimate the error variance:

$$
\boxed{
\hat{\sigma}^2=
\frac{SSE}{n-p-1}
}
$$

Then:

$$
\widehat{Var}(\hat{\beta})
=
\hat{\sigma}^2(X^TX)^{-1}
$$

The standard error of coefficient (j) is:

$$
\boxed{
SE(\hat{\beta}_j)
=
\sqrt{
\hat{\sigma}^2
[(X^TX)^{-1}]_{jj}
}
}
$$

### Interpretation

A coefficient estimate without its uncertainty can be misleading.

For example:

$$
\hat{\beta}=10,\qquad SE=2
$$

is much more precise than:

$$
\hat{\beta}=10,\qquad SE=20
$$

------------------------------------------------------------------------

# t-Test and p-Value

To test:

$$
H_0:\beta_j=\beta_{j,0}
$$

use:

$$
\boxed{
t=
\frac{
\hat{\beta}_j-\beta_{j,0}
}{
SE(\hat{\beta}_j)
}
}
$$

For the common null hypothesis:

$$
H_0:\beta_j=0
$$

we get:

$$
t=
\frac{\hat{\beta}_j}{SE(\hat{\beta}_j)}
$$

### Numerical example

Suppose:

$$
\hat{\beta}=8,\qquad SE=2
$$

Then:

$$
t=\frac{8}{2}=4
$$

The two-sided p-value is determined from the t-distribution with:

$$
df=n-p-1
$$

A p-value measures how unusual the observed statistic would be under the
null model. It is **not** the probability that the null hypothesis is
true.

------------------------------------------------------------------------

# Confidence Intervals

A (100(1-`\alpha`)%) confidence interval is:

$$
\boxed{
\hat{\beta}_j
\pm
t_{1-\alpha/2,df}
SE(\hat{\beta}_j)
}
$$

For example, if:

$$
\hat{\beta}=10
$$

$$
SE=2
$$

and:

$$
t_{0.975,df}=2.06
$$

then:

$$
10\pm2.06(2)
$$

so:

$$
\boxed{[5.88,\;14.12]}
$$

------------------------------------------------------------------------

# z-Test

A z-statistic is:

$$
\boxed{
z=
\frac{
\hat{\theta}-\theta_0
}{
SE(\hat{\theta})
}
}
$$

For regression coefficients, a z reference is appropriate when the
population error variance is known or when an appropriate asymptotic
normal approximation is being used.

For ordinary finite-sample regression with unknown variance, the
t-distribution is typically the appropriate reference distribution.

------------------------------------------------------------------------

# F-Test

The overall regression test can be written as:

$$
H_0:
\beta_1=\beta_2=\cdots=\beta_p=0
$$

For a model with an intercept:

$$
\boxed{
F=
\frac{SSR/p}
{SSE/(n-p-1)}
}
$$

Degrees of freedom:

$$
df_1=p
$$

$$
df_2=n-p-1
$$

For a single restriction, the corresponding relationship is:

$$
\boxed{F=t^2}
$$

when both tests use the same restriction and model.

------------------------------------------------------------------------

# Regression Assumptions

## 1. Linearity

The conditional mean is linear in the parameters:

$$
E[y\mid X]=X\beta
$$

This does not mean every raw feature-to-target plot must be a straight
line. Polynomial and interaction terms can still produce a model that is
linear in its coefficients.

## 2. Exogeneity

A central condition is:

$$
\boxed{
E[\epsilon\mid X]=0
}
$$

If predictors are related to omitted error components, OLS can be
biased.

## 3. Independence

The error structure must satisfy the independence assumptions required
for the intended inference.

Time-series, repeated-measures, and clustered data often require
specialized treatment.

## 4. Homoscedasticity

Classical homoscedasticity means:

$$
\boxed{
Var(\epsilon\mid X)=\sigma^2
}
$$

If the variance changes with (X), the errors are heteroscedastic.

## 5. Normality

For exact small-sample t and F inference, a common assumption is:

$$
\epsilon\mid X\sim N(0,\sigma^2I)
$$

Normality is not required merely to compute OLS coefficients.

## 6. No Perfect Multicollinearity

Predictors must not contain exact linear dependencies.

------------------------------------------------------------------------

# Multicollinearity and VIF

Suppose:

$$
x_2\approx2x_1
$$

Then the predictors contain overlapping information.

The model may still predict accurately, but individual coefficient
estimates can become unstable.

## Variance Inflation Factor

For predictor (j):

$$
\boxed{
VIF_j=
\frac{1}{1-R_j^2}
}
$$

where (R_j\^2) is obtained by regressing predictor (x_j) on the
remaining predictors.

Example:

$$
R_j^2=0.8
$$

gives:

$$
VIF_j=\frac{1}{1-0.8}=5
$$

Thresholds such as 5 or 10 are heuristics, not universal laws.

------------------------------------------------------------------------

# Residual Diagnostics

A model should not be evaluated using a single score alone.

Useful diagnostics include:

-   residuals vs fitted values;
-   residuals vs predictors;
-   Q-Q plot;
-   scale-location plot;
-   leverage;
-   Cook's Distance.

### Residual vs fitted

A random-looking cloud around zero is generally desirable.

Systematic patterns can indicate:

-   nonlinear relationships;
-   heteroscedasticity;
-   omitted variables;
-   interactions;
-   model misspecification.

------------------------------------------------------------------------

# Leverage and Hat Matrix

The fitted values can be expressed as:

$$
\hat{y}=Hy
$$

where:

$$
\boxed{
H=X(X^TX)^{-1}X^T
}
$$

is the **hat matrix**.

The diagonal element:

$$
h_{ii}
$$

measures the leverage of observation (i).

High leverage means an observation lies in an unusual region of
predictor space.

High leverage does not automatically mean the observation is erroneous.

------------------------------------------------------------------------

# Cook's Distance

Cook's Distance measures how strongly an observation can affect the
fitted regression.

A commonly used form is:

$$
\boxed{
D_i=
\frac{e_i^2}{(p+1)MSE}
\frac{h_{ii}}{(1-h_{ii})^2}
}
$$

Conventions differ slightly in how parameter counts are represented.

Cook's Distance is a diagnostic, not an automatic deletion rule.

------------------------------------------------------------------------

# Gradient Descent

OLS can be solved using a closed-form least-squares method, but
optimization methods can also minimize the objective.

Define:

$$
J(\beta)
=
\frac{1}{2n}
\|X\beta-y\|_2^2
$$

The gradient is:

$$
\boxed{
\nabla J(\beta)
=
\frac{1}{n}X^T(X\beta-y)
}
$$

Gradient Descent updates:

$$
\boxed{
\beta
\leftarrow
\beta-\alpha\nabla J(\beta)
}
$$

where (`\alpha`) is the learning rate.

## Why the (1/2)?

The derivative of a square introduces a factor of 2. The (1/2) cancels
that factor and does not change the minimizer.

------------------------------------------------------------------------

# Gradient Descent --- Numerical Example

Use:

$$
X=
\begin{bmatrix}
1&1\\
1&2\\
1&3
\end{bmatrix}
$$

and:

$$
y=
\begin{bmatrix}
2\\
4\\
6
\end{bmatrix}
$$

Start with:

$$
\beta=
\begin{bmatrix}
0\\
0
\end{bmatrix}
$$

and:

$$
\alpha=0.1
$$

Initial error:

$$
X\beta-y=
\begin{bmatrix}
-2\\
-4\\
-6
\end{bmatrix}
$$

Then:

$$
X^T(X\beta-y)
=
\begin{bmatrix}
-12\\
-28
\end{bmatrix}
$$

Divide by (n=3):

$$
\nabla J=
\begin{bmatrix}
-4\\
-9.3333
\end{bmatrix}
$$

Update:

$$
\beta_{new}
=
\beta-0.1\nabla J
$$

Therefore:

$$
\boxed{
\beta_{new}
=
\begin{bmatrix}
0.4\\
0.9333
\end{bmatrix}
}
$$

The process repeats until the objective converges.

------------------------------------------------------------------------

# Numerical Stability

The mathematical expression:

$$
(X^TX)^{-1}X^Ty
$$

does not imply that production code should explicitly calculate the
inverse.

Prefer solving:

$$
(X^TX)\beta=X^Ty
$$

with a numerical linear-system solver.

For example:

``` python
beta = np.linalg.solve(X.T @ X, X.T @ y)
```

or:

``` python
beta = np.linalg.lstsq(X, y, rcond=None)[0]
```

For difficult numerical problems, QR and SVD methods are especially
important because they provide better numerical stability than
explicitly forming an inverse.

------------------------------------------------------------------------

# Rank and Conditioning

If the columns of (X) are linearly dependent, the matrix can be rank
deficient.

Example:

$$
x_2=2x_1
$$

Then (X\^TX) may be singular.

Even when a matrix is technically invertible, it can be
**ill-conditioned**, meaning small numerical perturbations can produce
relatively large changes in the computed solution.

A condition number is one way to quantify this sensitivity.

------------------------------------------------------------------------

# WLS and GLS

## Weighted Least Squares

WLS minimizes:

$$
\boxed{
(y-X\beta)^TW(y-X\beta)
}
$$

The estimator is:

$$
\boxed{
\hat{\beta}_{WLS}
=
(X^TWX)^{-1}X^TWy
}
$$

WLS is useful when observations have different error variances and
appropriate weights are known or estimated.

## Generalized Least Squares

If:

$$
Var(\epsilon)=\Omega
$$

then:

$$
\boxed{
\hat{\beta}_{GLS}
=
(X^T\Omega^{-1}X)^{-1}
X^T\Omega^{-1}y
}
$$

OLS is the special case:

$$
\Omega=\sigma^2I
$$

WLS is a common special case where the covariance matrix is diagonal.

------------------------------------------------------------------------

# Regularization

Regularization modifies the objective to control coefficient magnitude
and improve generalization in settings such as multicollinearity or
high-dimensional data.

## Ridge

Ridge uses L2 regularization:

$$
\boxed{
\min_\beta
\left[
SSE+\lambda\sum_{j=1}^{p}\beta_j^2
\right]
}
$$

For the standard matrix formulation:

$$
\boxed{
\hat{\beta}_{ridge}
=
(X^TX+\lambda I)^{-1}X^Ty
}
$$

The intercept is usually excluded from the penalty.

## Lasso

Lasso uses L1 regularization:

$$
\boxed{
\min_\beta
\left[
SSE+\lambda\sum_{j=1}^{p}|\beta_j|
\right]
}
$$

L1 regularization can produce exact zero coefficients, making Lasso
useful for sparse feature selection.

## Ridge vs Lasso

  Property              Ridge                      Lasso
  --------------------- -------------------------- ---------------------
  Penalty               L2                         L1
  Shrinkage             Yes                        Yes
  Exact zeros           Usually no                 Often yes
  Feature selection     Limited                    Stronger
  Correlated features   Often distributes weight   May select a subset

------------------------------------------------------------------------

# Polynomial and Interaction Terms

## Polynomial Regression

A polynomial model can be:

$$
\hat{y}
=
\beta_0+
\beta_1x+
\beta_2x^2+
\beta_3x^3
$$

Although this is nonlinear in (x), it is still linear in the
coefficients.

## Interaction Terms

An interaction model can be:

$$
\hat{y}
=
\beta_0+
\beta_1x_1+
\beta_2x_2+
\beta_3x_1x_2
$$

The marginal effect of (x_1) is:

$$
\frac{\partial\hat{y}}{\partial x_1}
=
\beta_1+\beta_3x_2
$$

Therefore the effect of (x_1) depends on (x_2).

------------------------------------------------------------------------

# Model Validation

## Train/Test Split

A basic workflow is:

``` text
Dataset
   |
   +---- Training set ----> Fit model
   |
   +---- Test set --------> Final evaluation
```

The test set should represent unseen data.

## Data Leakage

Avoid learning preprocessing information from the test set.

Correct:

``` text
Training data
    |
    +--> fit scaler/imputer
    |
    +--> transform training data

Test data
    |
    +--> transform using training parameters
```

Incorrect:

``` text
Full dataset
    |
    +--> fit scaler
    |
    +--> split
```

## Cross-Validation

K-fold cross-validation repeatedly trains and validates on different
folds.

It is useful for:

-   model comparison;
-   hyperparameter selection;
-   estimating validation performance.

The final test set should remain untouched until final evaluation.

------------------------------------------------------------------------

# Implementation From Scratch

## Closed-Form Least Squares

``` python
import numpy as np

def fit_linear_regression(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    X = np.column_stack([np.ones(X.shape[0]), X])

    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    intercept = beta[0]
    coefficients = beta[1:]

    return intercept, coefficients
```

## Prediction

``` python
def predict_linear_regression(X, intercept, coefficients):
    X = np.asarray(X, dtype=float)
    return intercept + X @ coefficients
```

## Gradient Descent

``` python
import numpy as np

def linear_regression_gd(X, y, learning_rate=0.01, epochs=10000):
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

------------------------------------------------------------------------

# End-to-End Workflow

A robust regression workflow is:

``` text
Raw Dataset
     |
     v
Understand target + predictors
     |
     v
Train/Test Split
     |
     v
Fit preprocessing on training data only
     |
     v
Fit Linear Regression
     |
     v
Generate predictions
     |
     +------------------+
     |                  |
     v                  v
Metrics            Diagnostics
     |                  |
     |                  +--> Residuals
     |                  +--> Leverage
     |                  +--> VIF
     |                  +--> Influence
     |
     v
Statistical interpretation
     |
     v
Cross-validation / model selection
     |
     v
Final test evaluation
```

------------------------------------------------------------------------

# Choosing the Right Approach

  Situation                              Reasonable starting point
  -------------------------------------- -----------------------------
  Simple interpretable relationship      OLS
  Many correlated predictors             Ridge
  Sparse feature selection               Lasso
  Unequal known error variances          WLS
  Correlated / structured errors         GLS
  Nonlinear curve but structured basis   Polynomial Regression
  Effect depends on another feature      Interaction terms
  Strongly nonlinear unknown structure   Tree-based or neural models

------------------------------------------------------------------------

# Common Mistakes

### 1. Treating R² as accuracy

R² is not classification accuracy.

### 2. Assuming high R² proves a model is good

A model can have high R² and still have poor residual behavior or poor
out-of-sample performance.

### 3. Treating p-value as effect size

Statistical significance does not measure practical importance.

### 4. Assuming correlation proves causation

Regression coefficients alone do not establish causal relationships.

### 5. Deleting every outlier

Investigate observations before removing them.

### 6. Explicitly inverting matrices unnecessarily

Use stable numerical solvers.

### 7. Ignoring multicollinearity

Strong predictor correlation can make coefficient estimates unstable.

### 8. Scaling before train/test splitting

This can leak information from the test set.

### 9. Selecting models using the test set

Repeated test-set tuning destroys the purpose of the test set.

### 10. Confusing residuals and theoretical errors

Residual:

$$
e=y-\hat{y}
$$

Theoretical error:

$$
\epsilon
$$

They are related but not identical concepts.

------------------------------------------------------------------------

# Formula Reference

## Model

$$
y=X\beta+\epsilon
$$

## Prediction

$$
\hat{y}=X\hat{\beta}
$$

## Residual

$$
e=y-\hat{y}
$$

## SSE

$$
SSE=\sum_i(y_i-\hat{y}_i)^2
$$

## MSE

$$
MSE=\frac{SSE}{n}
$$

## RMSE

$$
RMSE=\sqrt{MSE}
$$

## MAE

$$
MAE=\frac1n\sum_i|y_i-\hat{y}_i|
$$

## OLS

$$
\hat{\beta}
=
\arg\min_\beta
\|y-X\beta\|_2^2
$$

## Normal Equation

$$
X^TX\hat{\beta}=X^Ty
$$

## Closed Form

$$
\hat{\beta}=(X^TX)^{-1}X^Ty
$$

## Gradient

$$
\nabla J(\beta)
=
\frac1nX^T(X\beta-y)
$$

## Gradient Descent

$$
\beta\leftarrow\beta-\alpha\nabla J(\beta)
$$

## Error Variance

$$
\hat{\sigma}^2=
\frac{SSE}{n-p-1}
$$

## Coefficient Variance

$$
\widehat{Var}(\hat{\beta})
=
\hat{\sigma}^2(X^TX)^{-1}
$$

## Standard Error

$$
SE(\hat{\beta}_j)
=
\sqrt{
\hat{\sigma}^2
[(X^TX)^{-1}]_{jj}
}
$$

## t-Statistic

$$
t=
\frac{\hat{\beta}_j-\beta_{j,0}}
{SE(\hat{\beta}_j)}
$$

## Confidence Interval

$$
\hat{\beta}_j
\pm
t_{1-\alpha/2,df}SE(\hat{\beta}_j)
$$

## R²

$$
R^2=1-\frac{SSE}{SST}
$$

## Adjusted R²

$$
R^2_{adj}
=
1-
\frac{(1-R^2)(n-1)}
{n-p-1}
$$

## F-Statistic

$$
F=
\frac{SSR/p}
{SSE/(n-p-1)}
$$

## VIF

$$
VIF_j=
\frac{1}{1-R_j^2}
$$

## Hat Matrix

$$
H=X(X^TX)^{-1}X^T
$$

## WLS

$$
\hat{\beta}_{WLS}
=
(X^TWX)^{-1}X^TWy
$$

## GLS

$$
\hat{\beta}_{GLS}
=
(X^T\Omega^{-1}X)^{-1}
X^T\Omega^{-1}y
$$

## Ridge

$$
\hat{\beta}_{ridge}
=
(X^TX+\lambda I)^{-1}X^Ty
$$

------------------------------------------------------------------------

# Glossary

  Term                    Meaning
  ----------------------- --------------------------------
  OLS                     Ordinary Least Squares
  SSE                     Sum of Squared Errors
  MSE                     Mean Squared Error
  RMSE                    Root Mean Squared Error
  MAE                     Mean Absolute Error
  SST                     Total Sum of Squares
  SSR                     Regression Sum of Squares
  SE                      Standard Error
  VIF                     Variance Inflation Factor
  WLS                     Weighted Least Squares
  GLS                     Generalized Least Squares
  BLUE                    Best Linear Unbiased Estimator
  \(X\)                   Design matrix
  \(y\)                   Observed target
  (`\beta`)         Model coefficients
  (`\hat{\beta}`)   Estimated coefficients
  (`\epsilon`)      Theoretical error
  \(e\)                   Observed residual
  \(H\)                   Hat/projection matrix
  (`\lambda`)       Regularization strength

------------------------------------------------------------------------

# Learning Checklist

A complete understanding of Linear Regression should include:

-   [ ] Simple Linear Regression
-   [ ] Multiple Linear Regression
-   [ ] Design Matrix
-   [ ] Intercept column
-   [ ] Residuals
-   [ ] SSE
-   [ ] MSE
-   [ ] RMSE
-   [ ] MAE
-   [ ] Least Squares
-   [ ] OLS
-   [ ] Normal Equation
-   [ ] Matrix derivation
-   [ ] Gradient Descent
-   [ ] Learning Rate
-   [ ] Feature Scaling
-   [ ] Covariance / Variance interpretation
-   [ ] R²
-   [ ] Adjusted R²
-   [ ] Degrees of Freedom
-   [ ] Error Variance
-   [ ] Standard Error
-   [ ] t-Test
-   [ ] p-Value
-   [ ] Confidence Interval
-   [ ] z-Test
-   [ ] F-Test
-   [ ] Exogeneity
-   [ ] Homoscedasticity
-   [ ] Multicollinearity
-   [ ] VIF
-   [ ] Residual Diagnostics
-   [ ] Leverage
-   [ ] Hat Matrix
-   [ ] Cook's Distance
-   [ ] WLS
-   [ ] GLS
-   [ ] Ridge
-   [ ] Lasso
-   [ ] Polynomial Regression
-   [ ] Interaction Terms
-   [ ] Train/Test Split
-   [ ] Data Leakage
-   [ ] Cross-Validation
-   [ ] Numerical Stability
-   [ ] Rank
-   [ ] QR/SVD concepts
-   [ ] NumPy implementation
-   [ ] Gradient Descent implementation
-   [ ] OnithrasML implementation

------------------------------------------------------------------------

# Final Perspective

The complete Linear Regression pipeline can be summarized as:

$$
\boxed{
\text{Data}
\rightarrow
\text{Design Matrix}
\rightarrow
\text{Model}
\rightarrow
\text{Least Squares}
\rightarrow
\text{Coefficient Estimates}
\rightarrow
\text{Prediction}
\rightarrow
\text{Residuals}
\rightarrow
\text{Metrics}
\rightarrow
\text{Diagnostics}
\rightarrow
\text{Inference}
}
$$

The central mathematical chain is:

$$
\boxed{
\text{Squared Error}
\rightarrow
\text{Least Squares}
\rightarrow
\text{Derivative}
\rightarrow
\text{Normal Equation}
\rightarrow
\hat{\beta}
\rightarrow
\hat{y}
\rightarrow
e
}
$$

This is the foundation on which the rest of the classical regression
toolkit is built.

------------------------------------------------------------------------

## Documentation Philosophy

OnithrasML documentation should answer three questions for every major
feature:

1.  **How do I use it?**
2.  **Why does it work?**
3.  **What should I check before trusting the result?**

That means the documentation should remain both an API reference for
users and a mathematical reference for readers who want to understand
the implementation.

> **OnithrasML Linear Regression:** from `fit(X, y)` to the mathematics
> behind OLS.
