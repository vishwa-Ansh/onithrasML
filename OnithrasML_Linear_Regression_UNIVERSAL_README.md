# OnithrasML — Linear Regression

> A complete mathematical, statistical, and implementation reference for Linear Regression in OnithrasML.

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Documentation](https://img.shields.io/badge/Docs-OnithrasML-blue.svg)](https://onithrasml.vercel.app)

Linear Regression is one of the most important foundations of classical machine learning. This guide goes from the basic model all the way to **OLS, matrix derivations, statistical inference, diagnostics, WLS, GLS, Ridge, Lasso, Gradient Descent, numerical stability, and implementation from scratch**.

---

> **Rendering note:** All mathematical expressions in this README use Unicode symbols and plain Markdown code blocks rather than LaTeX delimiters. This makes the equations readable in GitHub, VS Code Markdown Preview, and Markdown renderers that do not support KaTeX/MathJax.

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


```text
ŷ = β₀ + β₁ x
```


For several predictors:


```text
ŷ
=
β₀+β₁x₁+β₂x₂+⋯+βₚxₚ
```


The important word is **linear in the parameters**. A model such as


```text
ŷ=β₀+β₁x+β₂x²
```


is still a linear regression model because it is linear in the coefficients `β₀,β₁,β₂`.

The objective of ordinary least squares is:


```text
β̂
=
arg minβ
Σᵢ₌₁ⁿ(yᵢ-ŷᵢ)²
```


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

- `n` = number of observations
- `p` = number of predictors, excluding the intercept
- `X` = design matrix
- `y` = observed target vector
- `β` = population/model coefficient vector
- `β̂` = estimated coefficient vector
- `ε` = theoretical error vector
- `e` = observed residual vector

Dimensions:


```text
X∈ℝⁿ×⁽ᵖ⁺¹⁾
```



```text
y∈ℝⁿ
```



```text
β∈ℝᵖ⁺¹
```


The classical model is:


```text
y=Xβ+ε
```


After fitting:


```text
ŷ=Xβ̂
```


---

# 4. Simple Linear Regression

The simple model is:


```text
yᵢ=β₀+β₁xᵢ+εᵢ
```


The predicted value is:


```text
ŷᵢ=β̂₀+β̂₁xᵢ
```


## Meaning of `β₀`

`β₀` is the model's expected target value when `x=0`, assuming `x=0` is meaningful in the problem.

## Meaning of `β₁`

`β₁` is the change in predicted `y` for a one-unit increase in `x`.

For example:


```text
ŷ=50+3x
```


means that increasing `x` by one unit changes the predicted value by 3 units.

---

# 5. Deriving the Simple Regression Coefficients

We want to minimize:


```text
S(β₀,β₁)
=
Σᵢ₌₁ⁿ
(yᵢ-β₀-β₁xᵢ)²
```


Take the partial derivative with respect to `β₀`:


```text
(∂ S) / (∂β₀)
=
-2Σᵢ₌₁ⁿ
(yᵢ-β₀-β₁xᵢ)
```


At the minimum:


```text
Σᵢ(yᵢ-β₀-β₁xᵢ)=0
```


Therefore:


```text
nβ₀+β₁Σᵢ xᵢ=Σᵢ yᵢ
```


Divide by `n`:


```text
β₀=ȳ-β₁x̄
```


Now differentiate with respect to `β₁`:


```text
(∂ S) / (∂β₁)
=
-2Σᵢ xᵢ(yᵢ-β₀-β₁xᵢ)
```


Set equal to zero:


```text
Σᵢ xᵢ(yᵢ-β₀-β₁xᵢ)=0
```


Substitute:


```text
β₀=ȳ-β₁x̄
```


After simplification:


```text
β̂₁
=
frac
Σᵢ(xᵢ-x̄)(yᵢ-ȳ)
Σᵢ(xᵢ-x̄)²
```


and:


```text
β̂₀=ȳ-β̂₁x̄
```


This is the exact closed-form solution for simple OLS.

---

# 6. Multiple Linear Regression

For `p` predictors:


```text
y
=
β₀+
β₁x₁+
⋯+
βₚxₚ
̂
```


The coefficient interpretation is conditional on the other predictors.

For example:


```text
ŷ=100+5x₁-2x₂
```


The coefficient `5` means:

> Holding `x₂` constant, a one-unit increase in `x₁` changes the model's predicted target by 5 units.

This "holding other variables constant" interpretation is one reason multiple regression is more powerful than fitting separate simple regressions.

---

# 7. The Design Matrix

For:


```text
yᵢ=β₀+β₁xᵢ₁+β₂xᵢ₂+εᵢ
```


the design matrix is:


```text
[
  1 & x₁₁ & x₁₂
  1 & x₂₁ & x₂₂
  vdots & vdots & vdots
  1 & xₙ₁ & xₙ₂
]
```


and:


```text
[
  β₀
  β₁
  β₂
]
```


Then:


```text
[
  β₀+β₁x₁₁+β₂x₁₂
  β₀+β₁x₂₁+β₂x₂₂
  vdots
  β₀+β₁xₙ₁+β₂xₙ₂
]
```


The intercept column of ones is therefore not cosmetic. It is what allows the model to estimate `β₀`.

---

# 8. The Statistical Model

The statistical model is:


```text
y=Xβ+ε
```


This separates two concepts:

### Systematic component


```text
Xβ
```


### Random component


```text
ε
```


The regression function is:


```text
E[ymid X]=Xβ
```


when:


```text
E[εmid X]=0
```


This distinction is essential:

- `ε` is the unobserved theoretical error.
- `e=y-ŷ` is the observed residual.

---

# 9. Residuals

For observation `i`:


```text
eᵢ=yᵢ-yᵢ
̂
```


Vector form:


```text
e=y-Xβ
̂
```


A positive residual means the actual value is above the prediction.

A negative residual means the actual value is below the prediction.

OLS chooses coefficients so that the squared residuals are collectively minimized.

---

# 10. Least Squares

The residual sum of squares is:


```text
SSE
=
Σ_i=1ⁿ
(yᵢ-ŷᵢ)²
```


Using `ŷ=Xβ` during optimization:


```text
SSE(β)
=
(y-Xβ)ᵀ(y-Xβ)
```


Why square the errors?

If we minimized:


```text
Σᵢ(yᵢ-ŷᵢ)
```


positive and negative errors could cancel.

Absolute error would avoid cancellation, but squared error gives a smooth differentiable objective and leads to a powerful closed-form solution.

---

# 11. Normal Equation — Full Derivation

Start with:


```text
J(β)
=
(y-Xβ)ᵀ(y-Xβ)
```


Expand:


```text
J(β)
=
(yᵀ-βᵀXᵀ)(y-Xβ)
```


Multiply:


```text
J(β)
=
yᵀy
-yᵀXβ
-βᵀXᵀy
+βᵀXᵀXβ
```


Because `yᵀXβ` is a scalar:


```text
yᵀXβ=βᵀXᵀy
```


Therefore:


```text
J(β)
=
yᵀy
-2βᵀXᵀy
+βᵀXᵀXβ
```


Differentiate:


```text
∇_β J
=
-2Xᵀy
+
2XᵀXβ
```


At the minimum:


```text
∇_β J=0
```


Therefore:


```text
-2Xᵀy+2XᵀXβ̂=0
```


Divide by 2:


```text
XᵀXβ̂=Xᵀy
```


This is the **Normal Equation**:


```text
XᵀXβ=Xᵀy
̂
```


If `XᵀX` is invertible:


```text
β̂
=
(XᵀX)⁻¹Xᵀy
```


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

The prediction `ŷ` is the projection of `y` onto the column space of `X`:


```text
ŷ=Hy
```


where:


```text
H=X(XᵀX)⁻¹X^ᵀ
```


The residual is:


```text
e=y-ŷ
```


At the OLS solution:


```text
Xᵀe=0
```


This means the residual vector is orthogonal to every column of `X`.

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


```text
x=[1,2,3,4,5]
```



```text
y=[3,5,4,8,7]
```


We fit:


```text
ŷ=β₀+β₁x
```


## Step 1 — Calculate means


```text
x̄
=
(1+2+3+4+5) / (5)
=
3
```



```text
ȳ
=
(3+5+4+8+7) / (5)
=
5.4
```


## Step 2 — Calculate the centered values

| `xᵢ` | `yᵢ` | `xᵢ-x̄` | `yᵢ-ȳ` | Product | Squared `x` deviation |
|---:|---:|---:|---:|---:|---:|
| 1 | 3 | -2 | -2.4 | 4.8 | 4 |
| 2 | 5 | -1 | -0.4 | 0.4 | 1 |
| 3 | 4 | 0 | -1.4 | 0 | 0 |
| 4 | 8 | 1 | 2.6 | 2.6 | 1 |
| 5 | 7 | 2 | 1.6 | 3.2 | 4 |
| **Total** | | | | **11** | **10** |

Therefore:


```text
Sₓᵧ=11
```


and:


```text
Sₓₓ=10
```


## Step 3 — Slope


```text
β̂₁
=
fracSₓᵧSₓₓ
=
(11) / (10)
=
1.1
```


## Step 4 — Intercept


```text
β̂₀
=
ȳ-β̂₁x̄
```



```text
=5.4-(1.1)(3)
```



```text
β₀=2.1̂
```


Therefore:


```text
y=2.1+1.1x
̂
```


## Step 5 — Predictions

| `x` | `y` | `ŷ` | Residual `e=y-ŷ` |
|---:|---:|---:|---:|
| 1 | 3 | 3.2 | -0.2 |
| 2 | 5 | 4.3 | 0.7 |
| 3 | 4 | 5.4 | -1.4 |
| 4 | 8 | 6.5 | 1.5 |
| 5 | 7 | 7.6 | -0.6 |

## Step 6 — SSE


```text
SSE
=
(-0.2)²
+(0.7)²
+(-1.4)²
+(1.5)²
+(-0.6)²
```



```text
=0.04+0.49+1.96+2.25+0.36
```



```text
SSE=5.10
```


## Step 7 — MSE

There are `n=5` observations.

For prediction error:


```text
MSE=(SSE) / (n)
```


so:


```text
MSE=1.02
```


For an unbiased estimator of the error variance in classical regression inference, however, use residual degrees of freedom:


```text
σ̂²
=
(SSE) / (n-p-1)
```


Here `p=1`:


```text
σ̂²
=
(5.10) / (5-1-1)
=
(5.10) / (3)
=
1.70
```


This distinction is important:

- **prediction MSE** often uses `SSE/n`;
- **residual variance estimate for inference** uses `SSE/(n-p-1)`.

---

# 14. Regression Metrics

## MAE


```text
MAE=
frac1n
Σᵢ|yᵢ-ŷᵢ|
```


For the example:


```text
MAE
=
(0.2+0.7+1.4+1.5+0.6) / (5)
=
0.88
```


## MSE


```text
MSE=
frac1n
Σᵢ(yᵢ-ŷᵢ)²
```


Example:


```text
MSE=1.02
```


## RMSE


```text
RMSE=√(MSE
)
```


Example:


```text
RMSE=√(1.02)≈1.01
```


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


```text
SST=
Σᵢ(yᵢ-y)²
̄
```


Regression sum of squares:


```text
SSR=
Σᵢ(ŷᵢ-ȳ)²
```


Residual sum of squares:


```text
SSE=
Σᵢ(yᵢ-yᵢ)²
̂
```


With an intercept:


```text
SST=SSR+SSE
```


Therefore:


```text
R²
=
fracSSRSST
=
1-(SSE) / (SST)
```


## Adjusted R²

For `p` predictors:


```text
R²_adj
=
1-
frac(1-R²)(n-1)
n-p-1
```


Adjusted R² penalizes unnecessary predictors.

---

# 16. Degrees of Freedom

Degrees of freedom are the amount of independent information remaining after estimating parameters.

For a model with `n` observations and `p` predictors plus an intercept:


```text
dfᵣesᵢdual₌ₙ₋ₚ₋₁
```


There are:

- `n` observations;
- `p+1` fitted coefficients;
- therefore `n-(p+1)` residual degrees of freedom.

This is why the residual variance estimator uses:


```text
(SSE) / (n-p-1)
```


rather than simply `SSE/n`.

---

# 17. Error Variance

Under:


```text
Var(εmid X)=σ²I
```


the unknown population variance `σ²` can be estimated using:


```text
σ²
=
(SSE) / (n-p-1)
̂
```


and:


```text
σ
=
√((SSE) / (n-p-1))
̂
```


The square root is the residual standard deviation.

---

# 18. Standard Error — Derivation

Start from:


```text
β̂
=
(XᵀX)⁻¹Xᵀy
```


Substitute:


```text
y=Xβ+ε
```


Then:


```text
β̂
=
(XᵀX)⁻¹Xᵀ(Xβ+ε)
```


Distribute:


```text
β̂
=
(XᵀX)⁻¹XᵀXβ
+
(XᵀX)⁻¹Xᵀε
```


Since:


```text
(XᵀX)⁻¹XᵀX=I
```


we get:


```text
β̂
=
β+
(XᵀX)⁻¹Xᵀε
```


Therefore:


```text
β̂-β
=
(XᵀX)⁻¹Xᵀε
```


Under:


```text
Var(εmid X)=σ²I
```


the coefficient covariance matrix becomes:


```text
Var(β̂mid X)
=
σ²(XᵀX)⁻¹
```


Replacing `σ²` by `σ̂²`:


```text
widehatVar(β̂)
=
σ̂²(XᵀX)⁻¹
```


The standard error of coefficient `j` is:


```text
SE(β̂ⱼ)
=
sqrt
σ̂²
[(XᵀX)⁻¹]ⱼⱼ
```


---

# 19. t-Test and p-Value

To test:


```text
H₀:βⱼ=βⱼ,₀
```


against an alternative hypothesis, calculate:


```text
t=
frac
β̂ⱼ-βⱼ,₀
SE(β̂ⱼ)
```


For the common test:


```text
H₀:βⱼ=0
```


the statistic becomes:


```text
t=
fracβⱼ̂
SE(β̂ⱼ)
```


## Numerical example

Suppose:


```text
β̂₁=8
```


and:


```text
SE(β̂₁)=2
```


Then:


```text
t=(8) / (2)=4
```


If:


```text
n=30,qquad p=2
```


then:


```text
df=30-2-1=27
```


The two-sided p-value is:


```text
p
=
2P(T₂₇ge |4|)
```


The exact numerical value is obtained from the t-distribution.

### What does the p-value mean?

A p-value is the probability, under the null hypothesis and the assumed reference model, of observing a test statistic at least as extreme as the one obtained.

It is **not**:

- the probability that `H₀` is true;
- the probability that the coefficient is practically important;
- the probability that the model is correct.

---

# 20. Confidence Intervals

A `100(1-α)%` confidence interval for coefficient `j` is:


```text
β̂ⱼ
±
t₁₋α/₂,df
SE(β̂ⱼ)
```


Example:


```text
β̂=10
```



```text
SE=2
```


and suppose:


```text
t₀.₉₇₅,df=2.06
```


Then:


```text
10±(2.06)(2)
```



```text
10±4.12
```


Therefore:


```text
[5.88, 14.12]
```


The interval gives a range of parameter values compatible with the model and confidence procedure.

---

# 21. z-Test

The z-statistic is:


```text
z=
fracθ-θ₀̂
SE(θ̂)
```


A z reference is appropriate when the population variance is known or when an appropriate large-sample normal approximation is justified.

For ordinary finite-sample regression with unknown error variance, the t-distribution is generally used.

---

# 22. F-Test

The overall regression hypothesis is:


```text
H₀:
β₁=β₂=⋯=βₚ=0
```


The F-statistic is:


```text
F=
fracSSR/p
SSE/(n-p-1)
```


Degrees of freedom:


```text
df₁=p
```



```text
df₂=n-p-1
```


Interpretation:

> Does the fitted group of predictors explain statistically detectable variation beyond the intercept-only model?

For one restriction:


```text
F=t²
```


under the corresponding standard conditions.

---

# 23. Regression Assumptions

## 23.1 Linearity in Parameters

The conditional mean is modeled as:


```text
E[ymid X]=Xβ
```


Polynomial features and interaction features can still satisfy this because the model remains linear in `β`.

## 23.2 Exogeneity

A key condition is:


```text
E[εmid X]=0
```


Violating this can produce biased coefficient estimates.

## 23.3 Independence

Observations/errors must have the independence structure required by the intended estimator and inference.

Time series and clustered observations often violate simple iid assumptions.

## 23.4 Homoscedasticity

Classical homoscedasticity:


```text
Var(εᵢmid X)=σ²
```


Heteroscedasticity means the variance changes across observations or predictor values.

## 23.5 Normality

For exact small-sample t and F inference, a common assumption is:


```text
εmid Xsim N(0,σ²I)
```


Normality is not required merely to compute the OLS point estimate.

## 23.6 No Perfect Multicollinearity

No predictor should be an exact linear combination of the others.

---

# 24. Gauss–Markov Theorem and BLUE

Under the standard linear-model conditions, OLS is:


```text
BLUE
```


meaning:

**Best Linear Unbiased Estimator**

- **Linear** — estimator is linear in the observed target `y`.
- **Unbiased** — expected estimate equals the true parameter.
- **Best** — has the smallest covariance among linear unbiased estimators.

This theorem does not say that OLS is always the best possible machine-learning predictor.

It says something more specific about the class of **linear unbiased estimators** under the theorem's assumptions.

---

# 25. Unbiasedness — Derivation

Start with:


```text
β̂
=
(XᵀX)⁻¹Xᵀy
```


Substitute:


```text
y=Xβ+ε
```


Then:


```text
β̂
=
β+
(XᵀX)⁻¹Xᵀε
```


Take conditional expectation:


```text
E[β̂mid X]
=
β+
(XᵀX)⁻¹XᵀE[εmid X]
```


If:


```text
E[εmid X]=0
```


then:


```text
E[βmid X]=β
̂
```


Therefore OLS is conditionally unbiased.

---

# 26. Variance of OLS — Derivation

We have:


```text
β̂-β
=
(XᵀX)⁻¹Xᵀε
```


Let:


```text
A=(XᵀX)⁻¹Xᵀ
```


Then:


```text
β̂-β=Aε
```


Therefore:


```text
Var(β̂mid X)
=
A Var(εmid X) Aᵀ
```


Under:


```text
Var(εmid X)=σ²I
```


we get:


```text
Var(β̂mid X)
=
σ²AAᵀ
```


Now:


```text
AAᵀ
=
(XᵀX)⁻¹XᵀX(XᵀX)⁻¹
```


so:


```text
Var(β̂mid X)
=
σ²(XᵀX)⁻¹
```


This equation explains why multicollinearity increases coefficient uncertainty: the matrix `(XᵀX)⁻¹` becomes large or unstable when predictors contain nearly redundant information.

---

# 27. Multicollinearity

Multicollinearity occurs when predictors contain strong overlapping linear information.

Example:


```text
x₂≈2x₁
```


Possible consequences:

- large standard errors;
- unstable coefficients;
- coefficient signs that change across samples;
- difficulty interpreting individual predictors.

It does not necessarily mean predictions are poor.

The model can sometimes predict well while individual coefficients are difficult to estimate precisely.

---

# 28. VIF

For predictor `j`, regress it on all remaining predictors and obtain `Rⱼ²`.

Then:


```text
VIFⱼ=
frac11-Rⱼ²
```


Example:


```text
Rⱼ²=0.8
```


gives:


```text
VIFⱼ
=
(1) / (1-0.8)
=
5
```


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


```text
H=X(XᵀX)⁻¹X^ᵀ
```


and:


```text
y=Hy
̂
```


The diagonal entries:


```text
hᵢᵢ
```


are leverage values.

High leverage means the predictor values of an observation are unusual relative to the rest of the dataset.

High leverage does not automatically mean an observation is wrong.

---

# 31. Cook's Distance

Cook's Distance measures the influence of observation `i` on the fitted regression.

A common form is:


```text
Dᵢ
=
fraceᵢ²(p+1)MSE
frachᵢᵢ(1-hᵢᵢ)²
```


Conventions can differ in the exact parameter-count notation.

High Cook's Distance means the fitted model may change substantially if that observation is removed.

It is a diagnostic signal, not an automatic deletion rule.

---

# 32. Gradient Descent

The closed-form OLS solution is not the only way to fit linear regression.

Define:


```text
J(β)
=
frac12n
|Xβ-y|₂²
```


Gradient Descent repeatedly updates:


```text
β^(t+1)
=
β⁽ᵗ⁾
-
α∇ J(β⁽ᵗ⁾)
```


where `α` is the learning rate.

---

# 33. Gradient Descent — Full Derivation

Start with:


```text
J(β)
=
(1) / (2n)
(Xβ-y)ᵀ(Xβ-y)
```


Expand:


```text
J(β)
=
(1) / (2n)
[
βᵀXᵀXβ
-2βᵀXᵀy
+yᵀy
]
```


Differentiate:


```text
∇ J(β)
=
(1) / (2n)
[
2XᵀXβ
-2Xᵀy
]
```


Therefore:


```text
∇ J(β)
=
frac1n
Xᵀ(Xβ-y)
```


So the update becomes:


```text
β
arrow
β
-
fracαn
Xᵀ(Xβ-y)
```


This is the exact gradient used for batch Gradient Descent under this objective definition.

---

# 34. Feature Scaling

Gradient Descent can converge slowly when features have very different scales.

For example:


```text
x₁∈[0,1]
```


while:


```text
x₂∈[0,1,000,000]
```


The optimization landscape can become poorly conditioned.

A common standardization is:


```text
z=
fracx-μσ
```


Scaling is especially useful for:

- Gradient Descent;
- Ridge;
- Lasso;
- distance-based algorithms.

For train/test workflows, calculate `μ` and `σ` using the training data only.

---

# 35. Numerical Stability

The expression:


```text
β̂=(XᵀX)⁻¹Xᵀy
```


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

If the columns of `X` are linearly dependent:


```text
rank(X)<p+1
```


then the parameters are not uniquely identifiable in the ordinary full-rank formulation.

---

# 36. WLS

Weighted Least Squares minimizes:


```text
(y-Xβ)ᵀW(y-Xβ)
```


Different observations receive different weights.

Differentiate:


```text
J(β)
=
(y-Xβ)ᵀW(y-Xβ)
```


The derivative is:


```text
∇ J
=
-2XᵀWy
+
2XᵀWXβ
```


Set equal to zero:


```text
XᵀWXβ̂=XᵀWy
```


Therefore:


```text
β̂WLS
=
(XᵀWX)⁻¹XᵀWy
```


WLS is useful when observations have unequal known or estimated error variances.

---

# 37. GLS

Suppose:


```text
Var(εmid X)=Omega
```


where `Omega` is not necessarily `σ²I`.

Then GLS gives:


```text
β̂GLS
=
(XᵀOmega⁻¹X)⁻¹
XᵀOmega⁻¹y
```


OLS is the special case:


```text
Omega=σ²I
```


WLS is a common special case where `Omega` is diagonal.

---

# 38. Ridge Regression

Ridge adds an L2 penalty:


```text
J(β)
=
SSE+
λ
Σ_j=1ᵖβⱼ²
```


The intercept is normally not penalized.

For the standard matrix formulation:


```text
β̂ᵣᵢdge
=
(XᵀX+λ I)⁻¹Xᵀy
```


where `I` is adjusted so the intercept is not penalized when appropriate.

### Effect of `λ`

- `λ=0` approaches OLS.
- Larger `λ` increases shrinkage.
- Very large `λ` can create excessive bias.

Ridge is especially useful under multicollinearity.

---

# 39. Lasso Regression

Lasso uses an L1 penalty:


```text
J(β)
=
SSE+
λ
Σ_j=1ᵖ|βⱼ|
```


Lasso can drive some coefficients exactly to zero.

This makes it useful for sparse models and feature selection.

Unlike Ridge, Lasso does not generally have the same simple closed-form matrix solution.

---

# 40. Polynomial Regression

Polynomial regression adds transformed features.

For example:


```text
ŷ
=
β₀+
β₁x+
β₂x²+
β₃x³
```


This is nonlinear in `x` but linear in `β`.

The design matrix becomes:


```text
[
  1 & x₁ & x₁² & x₁³
  1 & x₂ & x₂² & x₂³
  vdots & vdots & vdots & vdots
]
```


The same least-squares machinery can then be used.

---

# 41. Interaction Terms

An interaction model can be:


```text
ŷ
=
β₀+
β₁x₁+
β₂x₂+
β₃x₁x₂
```


The marginal effect of `x₁` is:


```text
frac∂ŷ∂ x₁
=
β₁+β₃x₂
```


Therefore the effect of `x₁` depends on the value of `x₂`.

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

K-fold cross-validation divides training data into `k` folds.

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
- choosing Ridge/Lasso `λ`;
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


```text
e=y-ŷ
```


Theoretical error:


```text
ε=y-Xβ
```


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


```text
y=Xβ+ε
```


### Prediction


```text
ŷ=Xβ̂
```


### Residual


```text
e=y-ŷ
```


### SSE


```text
SSE=Σᵢ(yᵢ-ŷᵢ)²
```


### MSE


```text
MSE=(SSE) / (n)
```


### RMSE


```text
RMSE=√(MSE)
```


### MAE


```text
MAE=(1) / (n)Σᵢ|yᵢ-ŷᵢ|
```


### Simple slope


```text
β̂₁=
fracΣᵢ(xᵢ-x̄)(yᵢ-ȳ)
Σᵢ(xᵢ-x̄)²
```


### Simple intercept


```text
β̂₀=ȳ-β̂₁x̄
```


### Normal Equation


```text
XᵀXβ̂=Xᵀy
```


### Closed form


```text
β̂=(XᵀX)⁻¹Xᵀy
```


### Error variance


```text
σ̂²=
(SSE) / (n-p-1)
```


### Coefficient covariance


```text
widehatVar(β̂)
=
σ̂²(XᵀX)⁻¹
```


### Standard error


```text
SE(β̂ⱼ)
=
sqrt
σ̂²
[(XᵀX)⁻¹]ⱼⱼ
```


### t-statistic


```text
t=
fracβ̂ⱼ-βⱼ,₀
SE(β̂ⱼ)
```


### Confidence interval


```text
β̂ⱼ
±
t₁₋α/₂,dfSE(β̂ⱼ)
```


### R²


```text
R²=1-(SSE) / (SST)
```


### Adjusted R²


```text
R²adⱼ
=
1-
frac(1-R²)(n-1)
n-p-1
```


### F-statistic


```text
F=
fracSSR/p
SSE/(n-p-1)
```


### VIF


```text
VIFⱼ=(1) / (1-Rⱼ²)
```


### Hat matrix


```text
H=X(XᵀX)⁻¹Xᵀ
```


### Gradient


```text
∇ J(β)
=
(1) / (n)Xᵀ(Xβ-y)
```


### Gradient update


```text
β
arrow
β-α∇ J(β)
```


### WLS


```text
β̂WLS
=
(XᵀWX)⁻¹XᵀWy
```


### GLS


```text
β̂GLS
=
(XᵀOmega⁻¹X)⁻¹
XᵀOmega⁻¹y
```


### Ridge


```text
β̂ᵣᵢdge
=
(XᵀX+λ I)⁻¹Xᵀy
```


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
- [ ] Statistical model `y=Xβ+ε`
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


```text
y=Xβ+ε
```


### 2. Objective


```text
min_β
|y-Xβ|²
```


### 3. Normal Equation


```text
XᵀXβ̂=Xᵀy
```


### 4. Estimate


```text
β̂=(XᵀX)⁻¹Xᵀy
```


### 5. Prediction


```text
ŷ=Xβ̂
```


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
