# OnithrasML — Linear Regression

> A complete mathematical, statistical, and implementation reference for Linear Regression in OnithrasML.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/Docs-OnithrasML-blue.svg)](https://onithrasml.vercel.app)

Linear Regression is one of the most important foundations of classical machine learning. This guide goes from the basic model all the way to **OLS, matrix derivations, statistical inference, diagnostics, WLS, GLS, Ridge, Lasso, Gradient Descent, numerical stability, and implementation from scratch**.

---

## Table of Contents

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
14. [Regression Metrics](#14-regression-metrics)
15. [R² and Adjusted R²](#15-r2-and-adjusted-r2)
16. [Degrees of Freedom](#16-degrees-of-freedom)
17. [Error Variance](#17-error-variance)
18. [Standard Error — Derivation](#18-standard-error--derivation)
19. [t-Test and p-Value](#19-t-test-and-p-value)
20. [Confidence Intervals](#20-confidence-intervals)
21. [z-Test](#21-z-test)
22. [F-Test](#22-f-test)
23. [Regression Assumptions](#23-regression-assumptions)
24. [Gauss–Markov Theorem and BLUE](#24-gaussmarkov-theorem-and-blue)
25. [Unbiasedness — Derivation](#25-unbiasedness--derivation)
26. [Variance of OLS — Derivation](#26-variance-of-ols--derivation)
27. [Multicollinearity](#27-multicollinearity)
28. [VIF](#28-vif)
29. [Residual Diagnostics](#29-residual-diagnostics)
30. [Hat Matrix and Leverage](#30-hat-matrix-and-leverage)
31. [Cook's Distance](#31-cooks-distance)
32. [Gradient Descent](#32-gradient-descent)
33. [Gradient Descent — Full Derivation](#33-gradient-descent--full-derivation)
34. [Feature Scaling](#34-feature-scaling)
35. [Numerical Stability](#35-numerical-stability)
36. [WLS](#36-wls)
37. [GLS](#37-gls)
38. [Ridge Regression](#38-ridge-regression)
39. [Lasso Regression](#39-lasso-regression)
40. [Polynomial Regression](#40-polynomial-regression)
41. [Interaction Terms](#41-interaction-terms)
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

# 1. What Is Linear Regression?

Linear Regression models a continuous target variable as a linear combination of one or more predictors.

For one predictor:

$$
\hat{y} = \beta_0 + \beta_1 x
$$

For several predictors:

$$
\hat{y}
=
\beta_0+\beta_1x_1+\beta_2x_2+\cdots+\beta_px_p
$$

The important word is **linear in the parameters**. A model such as

$$
\hat{y}=\beta_0+\beta_1x+\beta_2x^2
$$

is still a linear regression model because it is linear in the coefficients $\beta_0,\beta_1,\beta_2$.

The objective of ordinary least squares is:

$$
\boxed{
\hat{\beta}
=
\arg\min_{\beta}
\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
}
$$

---

# 2. Why Linear Regression Matters

Linear Regression is important because it gives a direct connection between:

- **Linear algebra** — vectors, matrices, projections, rank
- **Calculus** — derivatives and gradients
- **Statistics** — estimation and uncertainty
- **Probability** — error assumptions
- **Optimization** — minimizing an objective
- **Machine learning** — training and generalization

A useful mental model is:

```text
Data
  ↓
Design Matrix X
  ↓
Model y = Xβ + ε
  ↓
Choose β that minimizes squared error
  ↓
β̂
  ↓
Predictions Xβ̂
  ↓
Residuals y - ŷ
  ↓
Metrics + Diagnostics + Inference
```

---

# 3. Mathematical Setup

Let:

- $n$ = number of observations
- $p$ = number of predictors, excluding the intercept
- $X$ = design matrix
- $y$ = observed target vector
- $\beta$ = population/model coefficient vector
- $\hat{\beta}$ = estimated coefficient vector
- $\epsilon$ = theoretical error vector
- $e$ = observed residual vector

Dimensions:

$$
X\in\mathbb{R}^{n\times(p+1)}
$$

$$
y\in\mathbb{R}^{n}
$$

$$
\beta\in\mathbb{R}^{p+1}
$$

The classical model is:

$$
\boxed{
y=X\beta+\epsilon
}
$$

After fitting:

$$
\boxed{
\hat{y}=X\hat{\beta}
}
$$

---

# 4. Simple Linear Regression

The simple model is:

$$
\boxed{
y_i=\beta_0+\beta_1x_i+\epsilon_i
}
$$

The predicted value is:

$$
\hat{y}_i=\hat{\beta}_0+\hat{\beta}_1x_i
$$

## Meaning of $\beta_0$

$\beta_0$ is the model's expected target value when $x=0$, assuming $x=0$ is meaningful in the problem.

## Meaning of $\beta_1$

$\beta_1$ is the change in predicted $y$ for a one-unit increase in $x$.

For example:

$$
\hat{y}=50+3x
$$

means that increasing $x$ by one unit changes the predicted value by 3 units.

---

# 5. Deriving the Simple Regression Coefficients

We want to minimize:

$$
S(\beta_0,\beta_1)
=
\sum_{i=1}^{n}
(y_i-\beta_0-\beta_1x_i)^2
$$

Take the partial derivative with respect to $\beta_0$:

$$
\frac{\partial S}{\partial\beta_0}
=
-2\sum_{i=1}^{n}
(y_i-\beta_0-\beta_1x_i)
$$

At the minimum:

$$
\sum_i(y_i-\beta_0-\beta_1x_i)=0
$$

Therefore:

$$
n\beta_0+\beta_1\sum_i x_i=\sum_i y_i
$$

Divide by $n$:

$$
\boxed{
\beta_0=\bar{y}-\beta_1\bar{x}
}
$$

Now differentiate with respect to $\beta_1$:

$$
\frac{\partial S}{\partial\beta_1}
=
-2\sum_i x_i(y_i-\beta_0-\beta_1x_i)
$$

Set equal to zero:

$$
\sum_i x_i(y_i-\beta_0-\beta_1x_i)=0
$$

Substitute:

$$
\beta_0=\bar{y}-\beta_1\bar{x}
$$

After simplification:

$$
\boxed{
\hat{\beta}_1
=
\frac{
\sum_i(x_i-\bar{x})(y_i-\bar{y})
}{
\sum_i(x_i-\bar{x})^2
}
}
$$

and:

$$
\boxed{
\hat{\beta}_0=\bar{y}-\hat{\beta}_1\bar{x}
}
$$

This is the exact closed-form solution for simple OLS.

---

# 6. Multiple Linear Regression

For $p$ predictors:

$$
\boxed{
\hat{y}
=
\beta_0+
\beta_1x_1+
\cdots+
\beta_px_p
}
$$

The coefficient interpretation is conditional on the other predictors.

For example:

$$
\hat{y}=100+5x_1-2x_2
$$

The coefficient $5$ means:

> Holding $x_2$ constant, a one-unit increase in $x_1$ changes the model's predicted target by 5 units.

This "holding other variables constant" interpretation is one reason multiple regression is more powerful than fitting separate simple regressions.

---

# 7. The Design Matrix

For:

$$
y_i=\beta_0+\beta_1x_{i1}+\beta_2x_{i2}+\epsilon_i
$$

the design matrix is:

$$
X=
\begin{bmatrix}
1 & x_{11} & x_{12}\\
1 & x_{21} & x_{22}\\
\vdots & \vdots & \vdots\\
1 & x_{n1} & x_{n2}
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

Then:

$$
X\beta
=
\begin{bmatrix}
\beta_0+\beta_1x_{11}+\beta_2x_{12}\\
\beta_0+\beta_1x_{21}+\beta_2x_{22}\\
\vdots\\
\beta_0+\beta_1x_{n1}+\beta_2x_{n2}
\end{bmatrix}
$$

The intercept column of ones is therefore not cosmetic. It is what allows the model to estimate $\beta_0$.

---

# 8. The Statistical Model

The statistical model is:

$$
\boxed{
y=X\beta+\epsilon
}
$$

This separates two concepts:

### Systematic component

$$
X\beta
$$

### Random component

$$
\epsilon
$$

The regression function is:

$$
E[y\mid X]=X\beta
$$

when:

$$
E[\epsilon\mid X]=0
$$

This distinction is essential:

- $\epsilon$ is the unobserved theoretical error.
- $e=y-\hat{y}$ is the observed residual.

---

# 9. Residuals

For observation $i$:

$$
\boxed{
e_i=y_i-\hat{y}_i
}
$$

Vector form:

$$
\boxed{
e=y-X\hat{\beta}
}
$$

A positive residual means the actual value is above the prediction.

A negative residual means the actual value is below the prediction.

OLS chooses coefficients so that the squared residuals are collectively minimized.

---

# 10. Least Squares

The residual sum of squares is:

$$
\boxed{
SSE
=
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
}
$$

Using $\hat{y}=X\beta$ during optimization:

$$
SSE(\beta)
=
(y-X\beta)^T(y-X\beta)
$$

Why square the errors?

If we minimized:

$$
\sum_i(y_i-\hat{y}_i)
$$

positive and negative errors could cancel.

Absolute error would avoid cancellation, but squared error gives a smooth differentiable objective and leads to a powerful closed-form solution.

---

# 11. Normal Equation — Full Derivation

Start with:

$$
J(\beta)
=
(y-X\beta)^T(y-X\beta)
$$

Expand:

$$
J(\beta)
=
(y^T-\beta^TX^T)(y-X\beta)
$$

Multiply:

$$
J(\beta)
=
y^Ty
-y^TX\beta
-\beta^TX^Ty
+\beta^TX^TX\beta
$$

Because $y^TX\beta$ is a scalar:

$$
y^TX\beta=\beta^TX^Ty
$$

Therefore:

$$
J(\beta)
=
y^Ty
-2\beta^TX^Ty
+\beta^TX^TX\beta
$$

Differentiate:

$$
\nabla_\beta J
=
-2X^Ty
+
2X^TX\beta
$$

At the minimum:

$$
\nabla_\beta J=0
$$

Therefore:

$$
-2X^Ty+2X^TX\hat{\beta}=0
$$

Divide by 2:

$$
X^TX\hat{\beta}=X^Ty
$$

This is the **Normal Equation**:

$$
\boxed{
X^TX\hat{\beta}=X^Ty
}
$$

If $X^TX$ is invertible:

$$
\boxed{
\hat{\beta}
=
(X^TX)^{-1}X^Ty
}
$$

## Important numerical note

The inverse expression is mathematically useful for derivation, but production numerical code should generally avoid explicitly calculating the inverse.

Prefer:

```python
beta = np.linalg.solve(X.T @ X, X.T @ y)
```

or, more generally:

```python
beta = np.linalg.lstsq(X, y, rcond=None)[0]
```

---

# 12. Geometric Interpretation

OLS has a beautiful geometric interpretation.

The prediction $\hat{y}$ is the projection of $y$ onto the column space of $X$:

$$
\hat{y}=Hy
$$

where:

$$
\boxed{
H=X(X^TX)^{-1}X^T
}
$$

The residual is:

$$
e=y-\hat{y}
$$

At the OLS solution:

$$
X^Te=0
$$

This means the residual vector is orthogonal to every column of $X$.

So geometrically:

```text
              y
             /
            /
           /|
          / |
         /  | e
        /   |
       /____|
      ŷ

ŷ = projection of y onto Col(X)
e = y - ŷ
e ⟂ Col(X)
```

This is the geometric meaning of the normal equation.

---

# 13. Worked Numerical Example

Consider:

$$
x=[1,2,3,4,5]
$$

$$
y=[3,5,4,8,7]
$$

We fit:

$$
\hat{y}=\beta_0+\beta_1x
$$

## Step 1 — Calculate means

$$
\bar{x}
=
\frac{1+2+3+4+5}{5}
=
3
$$

$$
\bar{y}
=
\frac{3+5+4+8+7}{5}
=
5.4
$$

## Step 2 — Calculate the centered values

| $x_i$ | $y_i$ | $x_i-\bar{x}$ | $y_i-\bar{y}$ | Product | Squared $x$ deviation |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | -2 | -2.4 | 4.8 | 4 |
| 2 | 5 | -1 | -0.4 | 0.4 | 1 |
| 3 | 4 | 0 | -1.4 | 0 | 0 |
| 4 | 8 | 1 | 2.6 | 2.6 | 1 |
| 5 | 7 | 2 | 1.6 | 3.2 | 4 |
| **Total** | | | | **11** | **10** |

Therefore:

$$
S_{xy}=11
$$

and:

$$
S_{xx}=10
$$

## Step 3 — Slope

$$
\hat{\beta}_1
=
\frac{S_{xy}}{S_{xx}}
=
\frac{11}{10}
=
1.1
$$

## Step 4 — Intercept

$$
\hat{\beta}_0
=
\bar{y}-\hat{\beta}_1\bar{x}
$$

$$
=5.4-(1.1)(3)
$$

$$
\boxed{\hat{\beta}_0=2.1}
$$

Therefore:

$$
\boxed{
\hat{y}=2.1+1.1x
}
$$

## Step 5 — Predictions

| $x$ | $y$ | $\hat{y}$ | Residual $e=y-\hat{y}$ |
|---:|---:|---:|---:|
| 1 | 3 | 3.2 | -0.2 |
| 2 | 5 | 4.3 | 0.7 |
| 3 | 4 | 5.4 | -1.4 |
| 4 | 8 | 6.5 | 1.5 |
| 5 | 7 | 7.6 | -0.6 |

## Step 6 — SSE

$$
SSE
=
(-0.2)^2
+(0.7)^2
+(-1.4)^2
+(1.5)^2
+(-0.6)^2
$$

$$
=0.04+0.49+1.96+2.25+0.36
$$

$$
\boxed{SSE=5.10}
$$

## Step 7 — MSE

There are $n=5$ observations.

For prediction error:

$$
MSE=\frac{SSE}{n}
$$

so:

$$
\boxed{MSE=1.02}
$$

For an unbiased estimator of the error variance in classical regression inference, however, use residual degrees of freedom:

$$
\hat{\sigma}^2
=
\frac{SSE}{n-p-1}
$$

Here $p=1$:

$$
\hat{\sigma}^2
=
\frac{5.10}{5-1-1}
=
\frac{5.10}{3}
=
1.70
$$

This distinction is important:

- **prediction MSE** often uses $SSE/n$;
- **residual variance estimate for inference** uses $SSE/(n-p-1)$.

---

# 14. Regression Metrics

## MAE

$$
\boxed{
MAE=
\frac{1}{n}
\sum_i|y_i-\hat{y}_i|
}
$$

For the example:

$$
MAE
=
\frac{0.2+0.7+1.4+1.5+0.6}{5}
=
\boxed{0.88}
$$

## MSE

$$
\boxed{
MSE=
\frac{1}{n}
\sum_i(y_i-\hat{y}_i)^2
}
$$

Example:

$$
MSE=1.02
$$

## RMSE

$$
\boxed{
RMSE=\sqrt{MSE}
}
$$

Example:

$$
RMSE=\sqrt{1.02}\approx\boxed{1.01}
$$

### Metric intuition

| Metric | Interpretation |
|---|---|
| MAE | Average absolute error |
| MSE | Average squared error |
| RMSE | Error in target units with extra sensitivity to large errors |
| R² | Improvement relative to the intercept-only baseline |

---

# 15. R² and Adjusted R²

Define total variation:

$$
\boxed{
SST=
\sum_i(y_i-\bar{y})^2
}
$$

Regression sum of squares:

$$
\boxed{
SSR=
\sum_i(\hat{y}_i-\bar{y})^2
}
$$

Residual sum of squares:

$$
\boxed{
SSE=
\sum_i(y_i-\hat{y}_i)^2
}
$$

With an intercept:

$$
\boxed{
SST=SSR+SSE
}
$$

Therefore:

$$
\boxed{
R^2
=
\frac{SSR}{SST}
=
1-\frac{SSE}{SST}
}
$$

## Adjusted R²

For $p$ predictors:

$$
\boxed{
R^2_{adj}
=
1-
\frac{(1-R^2)(n-1)}
{n-p-1}
}
$$

Adjusted R² penalizes unnecessary predictors.

---

# 16. Degrees of Freedom

Degrees of freedom are the amount of independent information remaining after estimating parameters.

For a model with $n$ observations and $p$ predictors plus an intercept:

$$
\boxed{
df_{residual}=n-p-1
}
$$

There are:

- $n$ observations;
- $p+1$ fitted coefficients;
- therefore $n-(p+1)$ residual degrees of freedom.

This is why the residual variance estimator uses:

$$
\frac{SSE}{n-p-1}
$$

rather than simply $SSE/n$.

---

# 17. Error Variance

Under:

$$
Var(\epsilon\mid X)=\sigma^2I
$$

the unknown population variance $\sigma^2$ can be estimated using:

$$
\boxed{
\hat{\sigma}^2
=
\frac{SSE}{n-p-1}
}
$$

and:

$$
\boxed{
\hat{\sigma}
=
\sqrt{\frac{SSE}{n-p-1}}
}
$$

The square root is the residual standard deviation.

---

# 18. Standard Error — Derivation

Start from:

$$
\hat{\beta}
=
(X^TX)^{-1}X^Ty
$$

Substitute:

$$
y=X\beta+\epsilon
$$

Then:

$$
\hat{\beta}
=
(X^TX)^{-1}X^T(X\beta+\epsilon)
$$

Distribute:

$$
\hat{\beta}
=
(X^TX)^{-1}X^TX\beta
+
(X^TX)^{-1}X^T\epsilon
$$

Since:

$$
(X^TX)^{-1}X^TX=I
$$

we get:

$$
\boxed{
\hat{\beta}
=
\beta+
(X^TX)^{-1}X^T\epsilon
}
$$

Therefore:

$$
\hat{\beta}-\beta
=
(X^TX)^{-1}X^T\epsilon
$$

Under:

$$
Var(\epsilon\mid X)=\sigma^2I
$$

the coefficient covariance matrix becomes:

$$
\boxed{
Var(\hat{\beta}\mid X)
=
\sigma^2(X^TX)^{-1}
}
$$

Replacing $\sigma^2$ by $\hat{\sigma}^2$:

$$
\boxed{
\widehat{Var}(\hat{\beta})
=
\hat{\sigma}^2(X^TX)^{-1}
}
$$

The standard error of coefficient $j$ is:

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

---

# 19. t-Test and p-Value

To test:

$$
H_0:\beta_j=\beta_{j,0}
$$

against an alternative hypothesis, calculate:

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

For the common test:

$$
H_0:\beta_j=0
$$

the statistic becomes:

$$
\boxed{
t=
\frac{\hat{\beta}_j}
{SE(\hat{\beta}_j)}
}
$$

## Numerical example

Suppose:

$$
\hat{\beta}_1=8
$$

and:

$$
SE(\hat{\beta}_1)=2
$$

Then:

$$
t=\frac{8}{2}=4
$$

If:

$$
n=30,\qquad p=2
$$

then:

$$
df=30-2-1=27
$$

The two-sided p-value is:

$$
p
=
2P(T_{27}\ge |4|)
$$

The exact numerical value is obtained from the t-distribution.

### What does the p-value mean?

A p-value is the probability, under the null hypothesis and the assumed reference model, of observing a test statistic at least as extreme as the one obtained.

It is **not**:

- the probability that $H_0$ is true;
- the probability that the coefficient is practically important;
- the probability that the model is correct.

---

# 20. Confidence Intervals

A $100(1-\alpha)\%$ confidence interval for coefficient $j$ is:

$$
\boxed{
\hat{\beta}_j
\pm
t_{1-\alpha/2,df}
SE(\hat{\beta}_j)
}
$$

Example:

$$
\hat{\beta}=10
$$

$$
SE=2
$$

and suppose:

$$
t_{0.975,df}=2.06
$$

Then:

$$
10\pm(2.06)(2)
$$

$$
10\pm4.12
$$

Therefore:

$$
\boxed{
[5.88,\;14.12]
}
$$

The interval gives a range of parameter values compatible with the model and confidence procedure.

---

# 21. z-Test

The z-statistic is:

$$
\boxed{
z=
\frac{\hat{\theta}-\theta_0}
{SE(\hat{\theta})}
}
$$

A z reference is appropriate when the population variance is known or when an appropriate large-sample normal approximation is justified.

For ordinary finite-sample regression with unknown error variance, the t-distribution is generally used.

---

# 22. F-Test

The overall regression hypothesis is:

$$
H_0:
\beta_1=\beta_2=\cdots=\beta_p=0
$$

The F-statistic is:

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

Interpretation:

> Does the fitted group of predictors explain statistically detectable variation beyond the intercept-only model?

For one restriction:

$$
\boxed{
F=t^2
}
$$

under the corresponding standard conditions.

---

# 23. Regression Assumptions

## 23.1 Linearity in Parameters

The conditional mean is modeled as:

$$
E[y\mid X]=X\beta
$$

Polynomial features and interaction features can still satisfy this because the model remains linear in $\beta$.

## 23.2 Exogeneity

A key condition is:

$$
\boxed{
E[\epsilon\mid X]=0
}
$$

Violating this can produce biased coefficient estimates.

## 23.3 Independence

Observations/errors must have the independence structure required by the intended estimator and inference.

Time series and clustered observations often violate simple iid assumptions.

## 23.4 Homoscedasticity

Classical homoscedasticity:

$$
\boxed{
Var(\epsilon_i\mid X)=\sigma^2
}
$$

Heteroscedasticity means the variance changes across observations or predictor values.

## 23.5 Normality

For exact small-sample t and F inference, a common assumption is:

$$
\epsilon\mid X\sim N(0,\sigma^2I)
$$

Normality is not required merely to compute the OLS point estimate.

## 23.6 No Perfect Multicollinearity

No predictor should be an exact linear combination of the others.

---

# 24. Gauss–Markov Theorem and BLUE

Under the standard linear-model conditions, OLS is:

$$
\boxed{\text{BLUE}}
$$

meaning:

**Best Linear Unbiased Estimator**

- **Linear** — estimator is linear in the observed target $y$.
- **Unbiased** — expected estimate equals the true parameter.
- **Best** — has the smallest covariance among linear unbiased estimators.

This theorem does not say that OLS is always the best possible machine-learning predictor.

It says something more specific about the class of **linear unbiased estimators** under the theorem's assumptions.

---

# 25. Unbiasedness — Derivation

Start with:

$$
\hat{\beta}
=
(X^TX)^{-1}X^Ty
$$

Substitute:

$$
y=X\beta+\epsilon
$$

Then:

$$
\hat{\beta}
=
\beta+
(X^TX)^{-1}X^T\epsilon
$$

Take conditional expectation:

$$
E[\hat{\beta}\mid X]
=
\beta+
(X^TX)^{-1}X^TE[\epsilon\mid X]
$$

If:

$$
E[\epsilon\mid X]=0
$$

then:

$$
\boxed{
E[\hat{\beta}\mid X]=\beta
}
$$

Therefore OLS is conditionally unbiased.

---

# 26. Variance of OLS — Derivation

We have:

$$
\hat{\beta}-\beta
=
(X^TX)^{-1}X^T\epsilon
$$

Let:

$$
A=(X^TX)^{-1}X^T
$$

Then:

$$
\hat{\beta}-\beta=A\epsilon
$$

Therefore:

$$
Var(\hat{\beta}\mid X)
=
A\,Var(\epsilon\mid X)\,A^T
$$

Under:

$$
Var(\epsilon\mid X)=\sigma^2I
$$

we get:

$$
Var(\hat{\beta}\mid X)
=
\sigma^2AA^T
$$

Now:

$$
AA^T
=
(X^TX)^{-1}X^TX(X^TX)^{-1}
$$

so:

$$
\boxed{
Var(\hat{\beta}\mid X)
=
\sigma^2(X^TX)^{-1}
}
$$

This equation explains why multicollinearity increases coefficient uncertainty: the matrix $(X^TX)^{-1}$ becomes large or unstable when predictors contain nearly redundant information.

---

# 27. Multicollinearity

Multicollinearity occurs when predictors contain strong overlapping linear information.

Example:

$$
x_2\approx2x_1
$$

Possible consequences:

- large standard errors;
- unstable coefficients;
- coefficient signs that change across samples;
- difficulty interpreting individual predictors.

It does not necessarily mean predictions are poor.

The model can sometimes predict well while individual coefficients are difficult to estimate precisely.

---

# 28. VIF

For predictor $j$, regress it on all remaining predictors and obtain $R_j^2$.

Then:

$$
\boxed{
VIF_j=
\frac{1}{1-R_j^2}
}
$$

Example:

$$
R_j^2=0.8
$$

gives:

$$
VIF_j
=
\frac{1}{1-0.8}
=
5
$$

A large VIF indicates that the predictor's variance is inflated by its linear relationship with the other predictors.

Rules such as 5 or 10 are practical heuristics, not universal statistical laws.

---

# 29. Residual Diagnostics

A regression model should not be judged by one number.

Useful diagnostic plots include:

### Residuals vs fitted values

Look for:

- nonlinearity;
- changing variance;
- systematic structure.

### Q-Q plot

Useful for assessing whether residuals approximately follow a normal reference distribution.

### Scale-location plot

Useful for detecting non-constant variance.

### Residuals vs predictors

Can reveal relationships that the fitted model failed to capture.

### Influence diagnostics

Help identify observations that strongly affect the fitted model.

---

# 30. Hat Matrix and Leverage

The hat matrix is:

$$
\boxed{
H=X(X^TX)^{-1}X^T
}
$$

and:

$$
\boxed{
\hat{y}=Hy
}
$$

The diagonal entries:

$$
h_{ii}
$$

are leverage values.

High leverage means the predictor values of an observation are unusual relative to the rest of the dataset.

High leverage does not automatically mean an observation is wrong.

---

# 31. Cook's Distance

Cook's Distance measures the influence of observation $i$ on the fitted regression.

A common form is:

$$
\boxed{
D_i
=
\frac{e_i^2}{(p+1)MSE}
\frac{h_{ii}}{(1-h_{ii})^2}
}
$$

Conventions can differ in the exact parameter-count notation.

High Cook's Distance means the fitted model may change substantially if that observation is removed.

It is a diagnostic signal, not an automatic deletion rule.

---

# 32. Gradient Descent

The closed-form OLS solution is not the only way to fit linear regression.

Define:

$$
\boxed{
J(\beta)
=
\frac{1}{2n}
\|X\beta-y\|_2^2
}
$$

Gradient Descent repeatedly updates:

$$
\boxed{
\beta^{(t+1)}
=
\beta^{(t)}
-
\alpha\nabla J(\beta^{(t)})
}
$$

where $\alpha$ is the learning rate.

---

# 33. Gradient Descent — Full Derivation

Start with:

$$
J(\beta)
=
\frac{1}{2n}
(X\beta-y)^T(X\beta-y)
$$

Expand:

$$
J(\beta)
=
\frac{1}{2n}
\left[
\beta^TX^TX\beta
-2\beta^TX^Ty
+y^Ty
\right]
$$

Differentiate:

$$
\nabla J(\beta)
=
\frac{1}{2n}
\left[
2X^TX\beta
-2X^Ty
\right]
$$

Therefore:

$$
\boxed{
\nabla J(\beta)
=
\frac{1}{n}
X^T(X\beta-y)
}
$$

So the update becomes:

$$
\boxed{
\beta
\leftarrow
\beta
-
\frac{\alpha}{n}
X^T(X\beta-y)
}
$$

This is the exact gradient used for batch Gradient Descent under this objective definition.

---

# 34. Feature Scaling

Gradient Descent can converge slowly when features have very different scales.

For example:

$$
x_1\in[0,1]
$$

while:

$$
x_2\in[0,1,000,000]
$$

The optimization landscape can become poorly conditioned.

A common standardization is:

$$
\boxed{
z=
\frac{x-\mu}{\sigma}
}
$$

Scaling is especially useful for:

- Gradient Descent;
- Ridge;
- Lasso;
- distance-based algorithms.

For train/test workflows, calculate $\mu$ and $\sigma$ using the training data only.

---

# 35. Numerical Stability

The expression:

$$
\hat{\beta}=(X^TX)^{-1}X^Ty
$$

is excellent for mathematical derivation but not always ideal for numerical implementation.

Avoid explicitly calculating:

```python
np.linalg.inv(X.T @ X)
```

when a solver can be used.

Prefer:

```python
beta = np.linalg.solve(X.T @ X, X.T @ y)
```

or:

```python
beta = np.linalg.lstsq(X, y, rcond=None)[0]
```

For difficult problems, QR and SVD are important because they provide more stable numerical solutions.

## Rank

If the columns of $X$ are linearly dependent:

$$
rank(X)<p+1
$$

then the parameters are not uniquely identifiable in the ordinary full-rank formulation.

---

# 36. WLS

Weighted Least Squares minimizes:

$$
\boxed{
(y-X\beta)^TW(y-X\beta)
}
$$

Different observations receive different weights.

Differentiate:

$$
J(\beta)
=
(y-X\beta)^TW(y-X\beta)
$$

The derivative is:

$$
\nabla J
=
-2X^TWy
+
2X^TWX\beta
$$

Set equal to zero:

$$
X^TWX\hat{\beta}=X^TWy
$$

Therefore:

$$
\boxed{
\hat{\beta}_{WLS}
=
(X^TWX)^{-1}X^TWy
}
$$

WLS is useful when observations have unequal known or estimated error variances.

---

# 37. GLS

Suppose:

$$
Var(\epsilon\mid X)=\Omega
$$

where $\Omega$ is not necessarily $\sigma^2I$.

Then GLS gives:

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

WLS is a common special case where $\Omega$ is diagonal.

---

# 38. Ridge Regression

Ridge adds an L2 penalty:

$$
\boxed{
J(\beta)
=
SSE+
\lambda
\sum_{j=1}^{p}\beta_j^2
}
$$

The intercept is normally not penalized.

For the standard matrix formulation:

$$
\boxed{
\hat{\beta}_{ridge}
=
(X^TX+\lambda I)^{-1}X^Ty
}
$$

where $I$ is adjusted so the intercept is not penalized when appropriate.

### Effect of $\lambda$

- $\lambda=0$ approaches OLS.
- Larger $\lambda$ increases shrinkage.
- Very large $\lambda$ can create excessive bias.

Ridge is especially useful under multicollinearity.

---

# 39. Lasso Regression

Lasso uses an L1 penalty:

$$
\boxed{
J(\beta)
=
SSE+
\lambda
\sum_{j=1}^{p}|\beta_j|
}
$$

Lasso can drive some coefficients exactly to zero.

This makes it useful for sparse models and feature selection.

Unlike Ridge, Lasso does not generally have the same simple closed-form matrix solution.

---

# 40. Polynomial Regression

Polynomial regression adds transformed features.

For example:

$$
\hat{y}
=
\beta_0+
\beta_1x+
\beta_2x^2+
\beta_3x^3
$$

This is nonlinear in $x$ but linear in $\beta$.

The design matrix becomes:

$$
X=
\begin{bmatrix}
1 & x_1 & x_1^2 & x_1^3\\
1 & x_2 & x_2^2 & x_2^3\\
\vdots & \vdots & \vdots & \vdots
\end{bmatrix}
$$

The same least-squares machinery can then be used.

---

# 41. Interaction Terms

An interaction model can be:

$$
\hat{y}
=
\beta_0+
\beta_1x_1+
\beta_2x_2+
\beta_3x_1x_2
$$

The marginal effect of $x_1$ is:

$$
\frac{\partial\hat{y}}{\partial x_1}
=
\beta_1+\beta_3x_2
$$

Therefore the effect of $x_1$ depends on the value of $x_2$.

This is fundamentally different from a model containing only additive effects.

---

# 42. Train/Test Split

A basic supervised-learning workflow separates data:

```text
                 Dataset
                    |
             +------+------+
             |             |
          Train           Test
             |             |
          Fit model     Final evaluation
```

The test set should represent unseen data.

Do not repeatedly tune the model against the test set.

---

# 43. Cross-Validation

K-fold cross-validation divides training data into $k$ folds.

For each fold:

```text
Fold 1 → validation
Folds 2..k → training
```

Then rotate the validation fold.

The final validation score is commonly summarized by the mean across folds.

Cross-validation is useful for:

- comparing models;
- selecting hyperparameters;
- choosing Ridge/Lasso $\lambda$;
- estimating validation performance.

---

# 44. Data Leakage

Data leakage happens when information that should be unavailable during training influences the training process.

Incorrect:

```text
Full dataset
    ↓
Fit scaler
    ↓
Train/Test split
```

Correct:

```text
Train/Test split
       ↓
Fit scaler on train only
       ↓
Transform train
       ↓
Transform test using train parameters
```

The same rule applies to:

- imputation;
- feature selection;
- target encoding;
- dimensionality reduction;
- model-based preprocessing.

---

# 45. Bias–Variance Tradeoff

Prediction error can be understood through:

- bias;
- variance;
- irreducible noise.

A flexible model can reduce bias but increase variance.

A highly constrained model can reduce variance but increase bias.

Regularization deliberately introduces bias to reduce variance.

Conceptually:

```text
Too simple
   ↓
High bias

Good balance
   ↓
Good generalization

Too flexible
   ↓
High variance
```

---

# 46. Implementation From Scratch

## OLS Using a Stable Least-Squares Solver

```python
import numpy as np

def fit_linear_regression(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    Xb = np.column_stack([np.ones(X.shape[0]), X])

    beta = np.linalg.lstsq(Xb, y, rcond=None)[0]

    return beta[0], beta[1:]
```

## Prediction

```python
def predict_linear_regression(X, intercept, coefficients):
    X = np.asarray(X, dtype=float)
    return intercept + X @ coefficients
```

## Batch Gradient Descent

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

---

# 47. OnithrasML Usage

The intended user-facing flow is:

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

For the exact current API, constructor parameters and learned attributes should be kept synchronized with the implementation in the OnithrasML source.

---

# 48. Complete Regression Workflow

A production-minded workflow:

```text
1. Understand the problem
        ↓
2. Inspect the dataset
        ↓
3. Define X and y
        ↓
4. Split train/test
        ↓
5. Fit preprocessing on train only
        ↓
6. Establish a baseline
        ↓
7. Fit Linear Regression
        ↓
8. Evaluate MAE / MSE / RMSE / R²
        ↓
9. Inspect residuals
        ↓
10. Check multicollinearity
        ↓
11. Check influential observations
        ↓
12. Perform inference when appropriate
        ↓
13. Compare Ridge/Lasso if needed
        ↓
14. Cross-validate model choices
        ↓
15. Evaluate once on the untouched test set
        ↓
16. Document assumptions and limitations
```

---

# 49. Common Mistakes

## Mistake 1 — Treating R² as accuracy

R² is not classification accuracy.

## Mistake 2 — Thinking high R² proves a good model

A model can have high R² and still suffer from:

- leakage;
- poor residual structure;
- multicollinearity;
- influential observations;
- poor test performance.

## Mistake 3 — Treating p-value as effect size

A small p-value does not tell you whether the effect is practically large.

## Mistake 4 — Confusing residuals and errors

Residual:

$$
e=y-\hat{y}
$$

Theoretical error:

$$
\epsilon=y-X\beta
$$

They are related but conceptually different.

## Mistake 5 — Explicit matrix inversion

Prefer stable solvers.

## Mistake 6 — Ignoring feature scale for optimization

Scaling can dramatically improve Gradient Descent behavior.

## Mistake 7 — Using the test set repeatedly

The test set should remain an honest final evaluation.

## Mistake 8 — Removing outliers automatically

Investigate whether an observation is:

- an error;
- a legitimate rare case;
- a high-leverage point;
- influential because the model is misspecified.

---

# 50. Formula Sheet

### Model

$$
y=X\beta+\epsilon
$$

### Prediction

$$
\hat{y}=X\hat{\beta}
$$

### Residual

$$
e=y-\hat{y}
$$

### SSE

$$
SSE=\sum_i(y_i-\hat{y}_i)^2
$$

### MSE

$$
MSE=\frac{SSE}{n}
$$

### RMSE

$$
RMSE=\sqrt{MSE}
$$

### MAE

$$
MAE=\frac{1}{n}\sum_i|y_i-\hat{y}_i|
$$

### Simple slope

$$
\hat{\beta}_1=
\frac{\sum_i(x_i-\bar{x})(y_i-\bar{y})}
{\sum_i(x_i-\bar{x})^2}
$$

### Simple intercept

$$
\hat{\beta}_0=\bar{y}-\hat{\beta}_1\bar{x}
$$

### Normal Equation

$$
X^TX\hat{\beta}=X^Ty
$$

### Closed form

$$
\hat{\beta}=(X^TX)^{-1}X^Ty
$$

### Error variance

$$
\hat{\sigma}^2=
\frac{SSE}{n-p-1}
$$

### Coefficient covariance

$$
\widehat{Var}(\hat{\beta})
=
\hat{\sigma}^2(X^TX)^{-1}
$$

### Standard error

$$
SE(\hat{\beta}_j)
=
\sqrt{
\hat{\sigma}^2
[(X^TX)^{-1}]_{jj}
}
$$

### t-statistic

$$
t=
\frac{\hat{\beta}_j-\beta_{j,0}}
{SE(\hat{\beta}_j)}
$$

### Confidence interval

$$
\hat{\beta}_j
\pm
t_{1-\alpha/2,df}SE(\hat{\beta}_j)
$$

### R²

$$
R^2=1-\frac{SSE}{SST}
$$

### Adjusted R²

$$
R^2_{adj}
=
1-
\frac{(1-R^2)(n-1)}
{n-p-1}
$$

### F-statistic

$$
F=
\frac{SSR/p}
{SSE/(n-p-1)}
$$

### VIF

$$
VIF_j=\frac{1}{1-R_j^2}
$$

### Hat matrix

$$
H=X(X^TX)^{-1}X^T
$$

### Gradient

$$
\nabla J(\beta)
=
\frac{1}{n}X^T(X\beta-y)
$$

### Gradient update

$$
\beta
\leftarrow
\beta-\alpha\nabla J(\beta)
$$

### WLS

$$
\hat{\beta}_{WLS}
=
(X^TWX)^{-1}X^TWy
$$

### GLS

$$
\hat{\beta}_{GLS}
=
(X^T\Omega^{-1}X)^{-1}
X^T\Omega^{-1}y
$$

### Ridge

$$
\hat{\beta}_{ridge}
=
(X^TX+\lambda I)^{-1}X^Ty
$$

---

# 51. Concept Map

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
      |       |       |    |         |       |         |
   XᵀX     Gradient  Projection   t/F/z    VIF/Cook   fit/predict
      |       |       |    |         |
      +-------+-------+----+---------+
                      |
                 OLS Estimate
                      |
                      v
               β̂ = (XᵀX)⁻¹Xᵀy
```

---

# 52. Learning Checklist

A strong understanding of Linear Regression should include:

- [ ] Simple Linear Regression
- [ ] Multiple Linear Regression
- [ ] Design Matrix
- [ ] Intercept column
- [ ] Statistical model $y=X\beta+\epsilon$
- [ ] Residuals
- [ ] SSE
- [ ] MSE
- [ ] RMSE
- [ ] MAE
- [ ] Least Squares
- [ ] OLS
- [ ] Normal Equation
- [ ] Full matrix derivation
- [ ] Geometric interpretation
- [ ] Gradient Descent
- [ ] Gradient derivation
- [ ] Learning rate
- [ ] Feature scaling
- [ ] R²
- [ ] Adjusted R²
- [ ] Degrees of freedom
- [ ] Error variance
- [ ] Standard Error
- [ ] t-Test
- [ ] p-Value
- [ ] Confidence Interval
- [ ] z-Test
- [ ] F-Test
- [ ] Regression assumptions
- [ ] Exogeneity
- [ ] Homoscedasticity
- [ ] Multicollinearity
- [ ] VIF
- [ ] Residual diagnostics
- [ ] Hat Matrix
- [ ] Leverage
- [ ] Cook's Distance
- [ ] Gauss–Markov theorem
- [ ] Unbiasedness derivation
- [ ] OLS variance derivation
- [ ] WLS
- [ ] GLS
- [ ] Ridge
- [ ] Lasso
- [ ] Polynomial Regression
- [ ] Interaction terms
- [ ] Train/Test Split
- [ ] Cross-Validation
- [ ] Data Leakage
- [ ] Bias–Variance Tradeoff
- [ ] Numerical Stability
- [ ] Rank
- [ ] NumPy implementation
- [ ] Gradient Descent implementation
- [ ] OnithrasML API usage

---

## Final Mental Model

If you understand these five equations, you understand the mathematical core of Linear Regression:

### 1. Model

$$
y=X\beta+\epsilon
$$

### 2. Objective

$$
\min_\beta
\|y-X\beta\|^2
$$

### 3. Normal Equation

$$
X^TX\hat{\beta}=X^Ty
$$

### 4. Estimate

$$
\hat{\beta}=(X^TX)^{-1}X^Ty
$$

### 5. Prediction

$$
\hat{y}=X\hat{\beta}
$$

Everything else — residuals, metrics, standard errors, hypothesis tests, confidence intervals, diagnostics, Ridge, WLS, GLS, and Gradient Descent — builds on this foundation.

---

## Documentation Principle

OnithrasML documentation should answer three questions for every major feature:

1. **How do I use it?**
2. **Why does it work mathematically?**
3. **When should I trust or question the result?**

That makes the documentation useful both as a package reference and as a learning resource.

---

**OnithrasML — Linear Regression: from `fit(X, y)` to the mathematics behind the estimator.**
