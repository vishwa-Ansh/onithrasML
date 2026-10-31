# Linear Regression

> **OnithrasML --- Linear Regression Documentation** A mathematical,
> statistical, and implementation-focused guide to Linear Regression.

------------------------------------------------------------------------

## 1. Overview

Linear Regression is one of the most important supervised learning
algorithms for predicting a **continuous numerical target** from one or
more input features.

It is simple enough to understand mathematically, but deep enough to
connect several areas of machine learning:

-   Linear Algebra
-   Calculus
-   Probability
-   Statistics
-   Optimization
-   Numerical Computing
-   Model Evaluation

The central idea is:

$$ \hat{y}=X\hat{\beta} $$

where:

-   \(X\) = design matrix
-   (`\hat{\beta}`) = estimated coefficients
-   (`\hat{y}`) = predicted values

The model chooses coefficients that make predictions as close as
possible to the observed target values according to a chosen objective.

------------------------------------------------------------------------

# 2. Quick Start

## Installation

``` bash
pip install onithrasML
```

## Basic Example

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

print(model.coef_)
print(model.intercept_)
print(predictions)
```

### Expected relationship

The data follows:

$$ y=1+2x $$

Therefore the fitted model should be approximately:

$$ \hat{y}=1+2x $$

------------------------------------------------------------------------

# 3. What Linear Regression Actually Does

Suppose we have observations:

    x    y
  --- ----
    1    3
    2    5
    3    7
    4    9
    5   11

We want to find a line:

$$ \hat{y}=b_0+b_1x $$

such that its predictions are close to the observed (y).

The two unknown parameters are:

-   (b_0): intercept
-   (b_1): slope

Linear Regression estimates these parameters from data.

------------------------------------------------------------------------

# 4. Simple Linear Regression

The simplest model has one feature.

$$ \boxed{\hat{y}=b_0+b_1x} $$

## 4.1 Intercept

(b_0) is the predicted value of (y) when:

$$ x=0 $$

So:

$$ \hat{y}\|_{x=0}=b_0 $$

## 4.2 Slope

(b_1) describes the expected change in (y) for a one-unit increase in
(x).

If:

$$ b_1=2 $$

then increasing (x) by one unit increases predicted (y) by 2.

------------------------------------------------------------------------

# 5. Multiple Linear Regression

With multiple features:

$$ \boxed{
\hat{y}
=
b_0+b_1x_1+b_2x_2+\cdots+b_px_p
} $$

Example:

$$ \hat{y} = 100+5x_1-2x_2+10x_3 $$

Interpretation:

-   (b_0=100)
-   (b_1=5)
-   (b_2=-2)
-   (b_3=10)

The coefficient of a feature represents its expected contribution while
the other included features are held constant.

------------------------------------------------------------------------

# 6. Statistical Model

Linear Regression can be written as:

$$ \boxed{y=X\beta+\epsilon} $$

where:

-   \(y\) = observed target
-   \(X\) = feature/design matrix
-   (`\beta`) = true coefficient vector
-   (`\epsilon`) = random error

The fitted model is:

$$ \hat{y}=X\hat{\beta} $$

The residual is:

$$ \boxed{e=y-\hat{y}} $$

The distinction is important:

$$ `\epsilon` $$

is the theoretical/random error in the statistical model, while

$$ e=y-\hat{y} $$

is the observed residual after fitting.

------------------------------------------------------------------------

# 7. Design Matrix

For multiple linear regression, the features are represented as a
matrix.

Suppose:

$$ X_{features}=
```
\begin{bmatrix}
x_{11}&x_{12}\\
x_{21}&x_{22}\\
x_{31}&x_{32}
\end{bmatrix}
```
$$

If the model contains an intercept, we add a column of ones:

$$ X=
```
\begin{bmatrix}
1&x_{11}&x_{12}\\
1&x_{21}&x_{22}\\
1&x_{31}&x_{32}
\end{bmatrix}
```
$$

This first column represents the intercept.

The coefficient vector becomes:

$$ \beta=
```
\begin{bmatrix}
b_0\\
b_1\\
b_2
\end{bmatrix}
```
$$

Then:

$$ \boxed{\hat{y}=X\beta} $$

------------------------------------------------------------------------

# 8. Why the Column of Ones Exists

Consider:

$$ \hat{y}=b_0+b_1x $$

Matrix multiplication can represent this as:

$$
```
\begin{bmatrix}
1&x_1\\
1&x_2\\
1&x_3
\end{bmatrix}
\begin{bmatrix}
b_0\\
b_1
\end{bmatrix}
```
=
```
\begin{bmatrix}
b_0+b_1x_1\\
b_0+b_1x_2\\
b_0+b_1x_3
\end{bmatrix}
```
$$

The ones column allows the matrix multiplication to produce the
intercept.

------------------------------------------------------------------------

# 9. Residuals

For observation (i):

$$ \boxed{e_i=y_i-\hat{y}_i} $$

Example:

$$ y_i=10 $$

and:

$$ \hat{y}_i=8 $$

Then:

$$ e_i=10-8=2 $$

If:

$$ y_i=8,`\quad `\hat{y}_i=10 $$

then:

$$ e_i=-2 $$

The sign tells us whether the model under-predicted or over-predicted.

------------------------------------------------------------------------

# 10. Why Squared Error?

If we simply add residuals:

$$ `\sum `e_i $$

positive and negative errors can cancel.

Example:

$$ 2+(-2)=0 $$

even though the model made two errors.

Therefore Linear Regression commonly uses squared residuals:

$$ e_i^2 $$

which are always non-negative.

------------------------------------------------------------------------

# 11. Sum of Squared Errors

The Sum of Squared Errors is:

$$ \boxed{
SSE=\sum_{i=1}^{n}(y_i-\hat{y}_i)^2
} $$

In matrix notation:

$$ \boxed{
SSE=(y-X\beta)^T(y-X\beta)
} $$

The objective of Ordinary Least Squares is to find the coefficients that
minimize this quantity.

$$ \boxed{
\hat{\beta}
=
\arg\min_\beta
(y-X\beta)^T(y-X\beta)
} $$

------------------------------------------------------------------------

# 12. Mean Squared Error

MSE is:

$$ \boxed{
MSE=\frac{SSE}{n}
} $$

or:

$$ \boxed{
MSE=
\frac1n
\sum_{i=1}^{n}
(y_i-\hat{y}_i)^2
} $$

MSE measures the average squared prediction error.

A smaller MSE generally means predictions are closer to the observed
targets.

------------------------------------------------------------------------

# 13. RMSE

Root Mean Squared Error:

$$ \boxed{
RMSE=\sqrt{MSE}
} $$

RMSE has the same unit as the target variable.

If the target is measured in rupees, RMSE is also measured in rupees.

------------------------------------------------------------------------

# 14. MAE

Mean Absolute Error:

$$ \boxed{
MAE=
\frac1n
\sum_{i=1}^{n}|y_i-\hat{y}_i|
} $$

Unlike MSE, MAE does not square errors.

Therefore large errors are penalized less aggressively than in MSE.

------------------------------------------------------------------------

# 15. Worked Example

Consider:

$$ x=$$`<!-- -->`{=html}1,2,3,4,5$$ $$

and:

$$ y=$$`<!-- -->`{=html}3,5,4,8,7$$ $$

We want to fit:

$$ \hat{y}=b_0+b_1x $$

------------------------------------------------------------------------

## 15.1 Calculate the Means

$$ `\bar `x=`\frac{1+2+3+4+5}{5}`=3 $$

$$ `\bar `y=`\frac{3+5+4+8+7}{5}`=5.4 $$

------------------------------------------------------------------------

## 15.2 Calculate (S_{xy})

$$ S_{xy} = `\sum`(x_i-`\bar `x)(y_i-`\bar `y) $$

The calculation gives:

$$ S_{xy}=11 $$

------------------------------------------------------------------------

## 15.3 Calculate (S_{xx})

$$ S_{xx} = `\sum`(x_i-`\bar `x)^2 $$

Therefore:

$$ S_{xx}=10 $$

------------------------------------------------------------------------

## 15.4 Calculate the Slope

For simple regression:

$$ \boxed{
\hat b_1=\frac{S_{xy}}{S_{xx}}
} $$

Therefore:

$$ \hat b_1=`\frac{11}{10}`=1.1 $$

------------------------------------------------------------------------

## 15.5 Calculate the Intercept

$$ \boxed{
\hat b_0=\bar y-\hat b_1\bar x
} $$

$$ \hat b_0=5.4-(1.1)(3) $$

$$ \hat b_0=2.1 $$

Therefore:

$$ \boxed{
\hat{y}=2.1+1.1x
} $$

------------------------------------------------------------------------

# 16. Predictions

For (x=1):

$$ \hat{y}=2.1+1.1(1)=3.2 $$

For (x=2):

$$ \hat{y}=4.3 $$

For (x=3):

$$ \hat{y}=5.4 $$

For (x=4):

$$ \hat{y}=6.5 $$

For (x=5):

$$ \hat{y}=7.6 $$

Therefore:

$$ \hat{y}=$$`<!-- -->`{=html}3.2,4.3,5.4,6.5,7.6$$ $$

------------------------------------------------------------------------

# 17. Residual Calculation

Observed:

$$ y=$$`<!-- -->`{=html}3,5,4,8,7$$ $$

Predicted:

$$ \hat{y}=$$`<!-- -->`{=html}3.2,4.3,5.4,6.5,7.6$$ $$

Residuals:

$$ e=y-\hat{y} $$

Therefore:

$$ e=$$-0.2,0.7,-1.4,1.5,-0.6$$ $$

------------------------------------------------------------------------

# 18. SSE Calculation

Square every residual:

$$ (-0.2)^2=0.04 $$

$$ (0.7)^2=0.49 $$

$$ (-1.4)^2=1.96 $$

$$ (1.5)^2=2.25 $$

$$ (-0.6)^2=0.36 $$

Therefore:

$$ SSE=0.04+0.49+1.96+2.25+0.36 $$

$$ \boxed{SSE=5.10} $$

------------------------------------------------------------------------

# 19. MSE Calculation

$$ MSE=`\frac{SSE}{n}` $$

$$ MSE=`\frac{5.10}{5}` $$

$$ \boxed{MSE=1.02} $$

------------------------------------------------------------------------

# 20. RMSE Calculation

$$ RMSE=`\sqrt{1.02}` $$

$$ \boxed{RMSE\approx1.01} $$

------------------------------------------------------------------------

# 21. MAE Calculation

$$ MAE= `\frac{
0.2+0.7+1.4+1.5+0.6
}{5}` $$

$$ \boxed{MAE=0.88} $$

------------------------------------------------------------------------

# 22. Ordinary Least Squares

Ordinary Least Squares, or OLS, estimates coefficients by minimizing:

$$ SSE $$

Therefore:

$$ \boxed{
\hat{\beta}
=
\arg\min_\beta
(y-X\beta)^T(y-X\beta)
} $$

The word **least squares** comes from minimizing the sum of squared
residuals.

------------------------------------------------------------------------

# 23. Deriving the Normal Equation

Start with:

$$ S(\beta) = (y-X\beta)^T(y-X\beta) $$

Expand:

$$ S(\beta) =
y^Ty-y^TX\beta-\beta^TX^Ty+\beta^TX^TX\beta
$$

Because the middle two terms are scalars:

$$ y^TX\beta=\beta^TX^Ty $$

so:

$$ S(\beta) =
y^Ty-2\beta^TX^Ty+\beta^TX^TX\beta $$

Differentiate with respect to (`\beta`):

$$ `\nabla `S(\beta) = -2X^Ty+2X^TX\beta $$

At the minimum:

$$ `\nabla `S(\beta)=0 $$

Therefore:

$$ -2X^Ty+2X^TX\beta=0 $$

Divide by 2:

$$ X^TX\beta=X^Ty $$

This is the **Normal Equation**.

If (X^TX) is invertible:

$$ \boxed{
\hat{\beta}=(X^TX)^{-1}X^Ty
} $$

------------------------------------------------------------------------

# 24. Why the Normal Equation Works

The Normal Equation is not a random formula.

It comes directly from:

1.  Define squared error.
2.  Expand the squared error.
3.  Differentiate with respect to coefficients.
4.  Set derivative equal to zero.
5.  Solve the resulting linear system.

So:

$$ \text{Least Squares} `\rightarrow`
\text{Derivative} `\rightarrow`
\text{Normal Equation} $$

------------------------------------------------------------------------

# 25. Numerical Stability

Although:

$$ \hat{\beta}=(X^TX)^{-1}X^Ty $$

is mathematically correct, explicitly calculating an inverse is often
not the best numerical implementation.

Prefer:

``` python
beta = np.linalg.solve(X.T @ X, X.T @ y)
```

or:

``` python
beta = np.linalg.lstsq(X, y, rcond=None)[0]
```

`lstsq` is generally preferable for practical numerical work.

Why?

Because explicit matrix inversion can amplify numerical errors,
especially when the matrix is ill-conditioned.

------------------------------------------------------------------------

# 26. Rank and Singular Matrices

If columns of (X) are linearly dependent, (X^TX) may be singular.

Example:

$$ x_2=2x_1 $$

Then one feature can be perfectly reconstructed from another.

This is a form of perfect multicollinearity.

In that situation:

$$ (X^TX)^{-1} $$

does not exist.

Numerically stable solvers based on QR or SVD can still provide
least-squares solutions.

------------------------------------------------------------------------

# 27. Covariance and Variance View

For simple linear regression:

$$ \boxed{
\hat b_1=
\frac{Cov(X,Y)}
{Var(X)}
} $$

This gives an important intuition.

The slope becomes large when (X) and (Y) move together strongly.

The slope becomes small when their covariance is small.

The denominator measures how much (X) varies.

------------------------------------------------------------------------

# 28. Geometric Interpretation

Linear Regression can also be understood geometrically.

The columns of (X) span a subspace.

The fitted vector:

$$ \hat{y}=X\hat{\beta} $$

is the projection of (y) onto the column space of (X).

Therefore:

$$ y=\hat{y}+e $$

where the residual is orthogonal to the model space.

Mathematically:

$$ \boxed{
X^Te=0
} $$

because:

$$ X^T(y-X\hat{\beta})=0 $$

which is exactly the Normal Equation.

------------------------------------------------------------------------

# 29. Projection Matrix

The fitted values can be written:

$$ \hat{y}=Hy $$

where:

$$ \boxed{
H=X(X^TX)^{-1}X^T
} $$

\(H\) is called the **hat matrix** because it maps:

$$ y`\rightarrow`\hat{y} $$

The diagonal elements:

$$ h_{ii} $$

measure the leverage of observations.

------------------------------------------------------------------------

# 30. Coefficient Interpretation

Suppose:

$$ \hat{y}=500+20x_1-10x_2 $$

Then:

### Intercept

$$ 500 $$

is the predicted target when both features are zero, assuming that value
is meaningful in the data domain.

### (x_1)

A one-unit increase in (x_1) is associated with a 20-unit increase in
predicted (y), holding (x_2) constant.

### (x_2)

A one-unit increase in (x_2) is associated with a 10-unit decrease in
predicted (y), holding (x_1) constant.

------------------------------------------------------------------------

# 31. R²

Total Sum of Squares:

$$ \boxed{
SST=\sum(y_i-\bar y)^2
} $$

Regression Sum of Squares:

$$ \boxed{
SSR=\sum(\hat{y}_i-\bar y)^2
} $$

Residual Sum of Squares:

$$ \boxed{
SSE=\sum(y_i-\hat{y}_i)^2
} $$

For models with an intercept:

$$ \boxed{
SST=SSR+SSE
} $$

R²:

$$ \boxed{
R^2=1-\frac{SSE}{SST}
} $$

Interpretation:

R² measures the proportion of variation in the target explained by the
fitted model relative to an intercept-only baseline.

------------------------------------------------------------------------

# 32. Adjusted R²

Adding more features can increase ordinary R² even when those features
are not useful.

Adjusted R² penalizes unnecessary predictors:

$$ \boxed{
R^2_{adj}
=
1-
\frac{(1-R^2)(n-1)}
{n-p-1}
} $$

where:

-   \(n\) = observations
-   \(p\) = number of predictors excluding the intercept

------------------------------------------------------------------------

# 33. Error Variance

Under the classical linear model:

$$ Var(`\epsilon`\|X)=`\sigma`^2I $$

The unknown error variance can be estimated by:

$$ \boxed{
\hat{\sigma}^2=
\frac{SSE}{n-p-1}
} $$

The denominator is the residual degrees of freedom:

$$ \boxed{
df_{error}=n-p-1
} $$

The subtraction of (p+1) comes from the number of estimated
coefficients, including the intercept.

------------------------------------------------------------------------

# 34. Standard Error of Coefficients

The covariance matrix of OLS coefficients is:

$$ \boxed{
Var(\hat{\beta}|X)
=
\sigma^2(X^TX)^{-1}
} $$

Using estimated error variance:

$$ \boxed{
\widehat{Var}(\hat{\beta})
=
\hat{\sigma}^2(X^TX)^{-1}
} $$

The standard error of coefficient (j) is:

$$ \boxed{
SE(\hat{\beta}_j)
=
\sqrt{
\hat{\sigma}^2
[(X^TX)^{-1}]_{jj}
}
} $$

This tells us how uncertain the estimated coefficient is under the model
assumptions.

------------------------------------------------------------------------

# 35. Why Standard Error Matters

Suppose:

$$ \hat{\beta}=10 $$

and:

$$ SE(\hat{\beta})=2 $$

The estimate is:

$$ 10`\pm2` $$

approximately.

But suppose:

$$ SE(\hat{\beta})=20 $$

Then the estimate 10 is highly uncertain relative to its standard error.

Therefore coefficient magnitude alone is not enough.

We should consider:

$$ `\frac{\hat{\beta}}{SE(\hat{\beta})}` $$

------------------------------------------------------------------------

# 36. t-Statistic

To test:

$$ H_0:\beta*j=\beta*{j,0} $$

the t-statistic is:

$$ \boxed{
t=
\frac{
\hat{\beta}_j-\beta_{j,0}
}{
SE(\hat{\beta}_j)
}
} $$

The most common null hypothesis is:

$$ H_0:\beta_j=0 $$

so:

$$ \boxed{
t=
\frac{\hat{\beta}_j}
{SE(\hat{\beta}_j)}
} $$

------------------------------------------------------------------------

# 37. Manual t-Test Example

Suppose:

$$ \hat{\beta}=8 $$

and:

$$ SE=2 $$

Testing:

$$ H_0:\beta=0 $$

gives:

$$ t=`\frac{8-0}{2}` $$

$$ \boxed{t=4} $$

A large absolute t-statistic means the estimate is many standard errors
away from the null value.

The exact p-value depends on the appropriate t-distribution degrees of
freedom.

------------------------------------------------------------------------

# 38. Degrees of Freedom

For a regression with:

-   \(n\) observations
-   \(p\) predictors
-   one intercept

the residual degrees of freedom are:

$$ \boxed{
df=n-p-1
} $$

Example:

$$ n=100,`\quad `p=5 $$

Then:

$$ df=100-5-1=94 $$

Degrees of freedom determine the reference distribution used for many
inference procedures.

------------------------------------------------------------------------

# 39. p-Value

A p-value is calculated from a test statistic and its reference
distribution.

For a two-sided t-test:

$$ \boxed{
p=2P(T_{df}\ge |t|)
} $$

A small p-value indicates that the observed test statistic would be
unusual under the null hypothesis.

A p-value is **not**:

-   the probability that the null hypothesis is true
-   the probability that the model is correct
-   the probability that the coefficient is practically important

Statistical significance and practical significance are different
concepts.

------------------------------------------------------------------------

# 40. Confidence Interval

A (100(1-`\alpha`)%) confidence interval for a coefficient is:

$$ \boxed{
\hat{\beta}_j
\pm
t_{1-\alpha/2,df}
SE(\hat{\beta}_j)
} $$

For a 95% interval:

$$ `\alpha=0.05` $$

so:

$$ \boxed{
\hat{\beta}_j
\pm
t_{0.975,df}SE(\hat{\beta}_j)
} $$

Example:

$$ \hat{\beta}=10 $$

$$ SE=2 $$

If the critical value is approximately 2.06:

$$ 10`\pm2.06`(2) $$

$$ 10`\pm4.12` $$

Therefore:

$$ \boxed{[5.88,\;14.12]} $$

------------------------------------------------------------------------

# 41. z-Test

A z-statistic can be used when the standard deviation is known or under
conditions where a normal reference is justified.

The statistic is:

$$ \boxed{
z=
\frac{
\hat{\theta}-\theta_0
}{
SE(\hat{\theta})
}
} $$

For regression coefficients, when the population error variance
(`\sigma`^2) is genuinely known:

$$ z= `\frac{\hat{\beta}_j-\beta_{j,0}}`
{SE(\hat{\beta}_j)} $$

In ordinary regression with unknown (`\sigma`^2), the t-distribution is
generally the appropriate finite-sample reference distribution.

------------------------------------------------------------------------

# 42. F-Test

The F-test can test whether a group of regression coefficients is
jointly zero.

For the overall regression:

$$ H_0:
\beta_1=\beta_2=`\cdots`=\beta_p=0 $$

For a model with an intercept:

$$ \boxed{
F=
\frac{SSR/p}
{SSE/(n-p-1)}
} $$

The numerator degrees of freedom are:

$$ df_1=p $$

The denominator degrees of freedom are:

$$ df_2=n-p-1 $$

A large F-statistic indicates that the model explains substantial
variation relative to its residual variation.

------------------------------------------------------------------------

# 43. Relationship Between t and F

For a single restriction:

$$ \boxed{
F=t^2
} $$

when the tests are constructed from the same model and restriction.

This is an important connection between regression inference procedures.

------------------------------------------------------------------------

# 44. Linear Regression Assumptions

Classical OLS inference relies on assumptions.

## 44.1 Linearity

The conditional mean should be linear in the parameters:

$$ E$$y\|X$$=X\beta $$

The relationship does not necessarily have to look like a straight line
in the original variables if transformations and polynomial terms are
included.

------------------------------------------------------------------------

## 44.2 Independence

Observations/errors should have the required independence structure for
the intended inference.

Time-series observations, clustered observations, and repeated
measurements often violate simple independence assumptions.

------------------------------------------------------------------------

## 44.3 Exogeneity

A central condition is:

$$ \boxed{
E[\epsilon|X]=0
} $$

This means the regressors are not systematically related to the
unobserved error.

Violation can create biased coefficient estimates.

------------------------------------------------------------------------

## 44.4 Homoscedasticity

Homoscedasticity means:

$$ \boxed{
Var(\epsilon|X)=\sigma^2
} $$

The error variance is constant across the predictor space.

If:

$$ Var(`\epsilon`\|X) $$

changes with (X), we have heteroscedasticity.

------------------------------------------------------------------------

## 44.5 Normality

Normality of errors is not required for OLS coefficients themselves to
exist.

It is mainly useful for exact small-sample t and F inference.

A common assumption is:

$$ `\epsilon`\|X`\sim `N(0,`\sigma`^2I) $$

------------------------------------------------------------------------

## 44.6 No Perfect Multicollinearity

The predictors cannot contain exact linear dependencies.

If:

$$ x_3=2x_1+x_2 $$

then the design matrix is rank-deficient.

------------------------------------------------------------------------

# 45. Gauss-Markov Theorem

Under the relevant classical linear model assumptions, OLS is **BLUE**:

> Best Linear Unbiased Estimator.

This means that among linear unbiased estimators, OLS has the minimum
variance.

Important:

BLUE does not mean:

-   always lowest prediction error
-   always best machine-learning model
-   always unbiased under every possible data-generating process

It is a precise statistical statement under specified assumptions.

------------------------------------------------------------------------

# 46. Unbiasedness Derivation

Start with:

$$ y=X\beta+`\epsilon` $$

OLS estimator:

$$ \hat{\beta}=(X^TX)^{-1}X^Ty $$

Substitute (y):

$$ \hat{\beta} =
(X^TX)^{-1}X^T(X\beta+`\epsilon`) $$

Expand:

$$ \hat{\beta} = (X^TX)^{-1}X^TX\beta +
(X^TX)^{-1}X^T`\epsilon` $$

Therefore:

$$ \hat{\beta} = \beta+
(X^TX)^{-1}X^T`\epsilon` $$

Take conditional expectation:

$$ E$$`\hat{\beta}`\|X$$ = \beta+
(X^TX)^{-1}X^TE$$`\epsilon`\|X$$ $$

If:

$$ E$$`\epsilon`\|X$$=0 $$

then:

$$ \boxed{
E[\hat{\beta}|X]=\beta
} $$

Thus OLS is conditionally unbiased.

------------------------------------------------------------------------

# 47. Variance of OLS

From:

$$ \hat{\beta} = \beta+
(X^TX)^{-1}X^T`\epsilon` $$

we obtain:

$$ Var(\hat{\beta}\|X) = (X^TX)^{-1}X^T
Var(`\epsilon`\|X) X(X^TX)^{-1} $$

If:

$$ Var(`\epsilon`\|X)=`\sigma`^2I $$

then:

$$ Var(\hat{\beta}\|X) = `\sigma`^2 (X^TX)^{-1} $$

Therefore:

$$ \boxed{
Var(\hat{\beta}|X)
=
\sigma^2(X^TX)^{-1}
} $$

This is the mathematical foundation of regression standard errors.

------------------------------------------------------------------------

# 48. Multicollinearity

Multicollinearity occurs when predictors are strongly related to each
other.

Example:

$$ x_2`\approx`2x_1 $$

The model may still predict well, but individual coefficients can become
unstable.

Symptoms include:

-   large standard errors
-   coefficients changing substantially
-   unexpected coefficient signs
-   difficulty interpreting individual predictors

------------------------------------------------------------------------

# 49. VIF

Variance Inflation Factor:

$$ \boxed{
VIF_j=
\frac{1}{1-R_j^2}
} $$

Here (R_j^2) is obtained by regressing predictor (x_j) on the other
predictors.

Example:

If:

$$ R_j^2=0.8 $$

then:

$$ VIF_j=`\frac1{1-0.8}`=5 $$

Higher VIF indicates stronger linear dependence with other predictors.

There is no universal magic cutoff; thresholds such as 5 or 10 are
heuristics, not laws.

------------------------------------------------------------------------

# 50. Residual Analysis

Residuals should be examined rather than relying only on a single
metric.

Useful plots include:

-   residuals vs fitted values
-   residuals vs predictors
-   Q-Q plot
-   scale-location plot
-   leverage plot

### Residual vs fitted

A random cloud around zero is generally desirable.

Patterns may indicate:

-   nonlinearity
-   heteroscedasticity
-   omitted variables
-   interactions
-   other model misspecification

------------------------------------------------------------------------

# 51. Outliers

An outlier is an observation that is unusual relative to the model.

Not every outlier is influential.

An observation can be:

-   high residual
-   high leverage
-   influential
-   or some combination

These are different concepts.

------------------------------------------------------------------------

# 52. Leverage

The leverage of observation (i) is the diagonal element of the hat
matrix:

$$ \boxed{
h_{ii}=H_{ii}
} $$

where:

$$ H=X(X^TX)^{-1}X^T $$

High leverage means the observation lies in an unusual region of
predictor space.

High leverage does not automatically mean the observation is bad.

------------------------------------------------------------------------

# 53. Cook's Distance

Cook's Distance measures how much the fitted model would be affected by
an observation.

A commonly used expression is:

$$ \boxed{
D_i=
\frac{e_i^2}{p+1}
\frac{h_{ii}}
{(1-h_{ii})^2}
\frac{1}{MSE}
} $$

Exact notation can vary by convention, especially regarding whether the
intercept is included in the parameter count.

Cook's Distance is primarily a diagnostic, not an automatic deletion
rule.

------------------------------------------------------------------------

# 54. Weighted Least Squares

When observations have different error variances, ordinary least squares
may not be efficient.

Weighted Least Squares gives observations different weights.

The objective becomes:

$$ \boxed{
\min_\beta
(y-X\beta)^T
W
(y-X\beta)
} $$

The solution is:

$$ \boxed{
\hat{\beta}_{WLS}
=
(X^TWX)^{-1}X^TWy
} $$

If an observation has high noise, it can receive a lower weight.

------------------------------------------------------------------------

# 55. Generalized Least Squares

GLS handles a more general error covariance structure.

Suppose:

$$ Var(`\epsilon`)=\Omega $$

Then:

$$ \boxed{
\hat{\beta}_{GLS}
=
(X^T\Omega^{-1}X)^{-1}
X^T\Omega^{-1}y
} $$

OLS assumes:

$$ \Omega=`\sigma`^2I $$

WLS is a special case where (`\Omega`) is diagonal.

------------------------------------------------------------------------

# 56. OLS vs WLS vs GLS

  Method   Error structure
  -------- -----------------------------------------------------
  OLS      Equal variance, no covariance under classical setup
  WLS      Different observation variances
  GLS      General covariance structure

Conceptually:

$$ OLS `\rightarrow` WLS `\rightarrow` GLS $$

as the assumed error structure becomes more general.

------------------------------------------------------------------------

# 57. Ridge Regression

Ridge adds an L2 penalty:

$$ \boxed{
\min_\beta
\left[
SSE+\lambda\sum_{j=1}^{p}\beta_j^2
\right]
} $$

The intercept is usually not penalized.

For standardized predictors and appropriate notation:

$$ \boxed{
\hat{\beta}_{ridge}
=
(X^TX+\lambda I)^{-1}X^Ty
} $$

Larger (`\lambda`) causes stronger coefficient shrinkage.

Ridge is useful when predictors are correlated and coefficient variance
needs to be controlled.

------------------------------------------------------------------------

# 58. Lasso Regression

Lasso uses an L1 penalty:

$$ \boxed{
\min_\beta
\left[
SSE+
\lambda\sum_{j=1}^{p}|\beta_j|
\right]
} $$

L1 regularization can drive some coefficients exactly to zero.

This makes Lasso useful for sparse feature selection.

------------------------------------------------------------------------

# 59. Ridge vs Lasso

  -----------------------------------------------------------------------
  Property                Ridge                   Lasso
  ----------------------- ----------------------- -----------------------
  Penalty                 L2                      L1

  Shrinks coefficients    Yes                     Yes

  Exact zeros             Usually no              Often yes

  Feature selection       Limited                 Stronger

  Correlated features     Often distributes       May select one or a
                          weights                 subset
  -----------------------------------------------------------------------

------------------------------------------------------------------------

# 60. Lambda Selection

The regularization parameter:

$$ \lambda $$

controls the strength of regularization.

Small (`\lambda`):

$$ \text{weak regularization} $$

Large (`\lambda`):

$$ \text{strong regularization} $$

It should normally be selected using validation or cross-validation
rather than guessed.

------------------------------------------------------------------------

# 61. Feature Scaling

Feature scaling is especially important for:

-   Gradient Descent
-   Ridge
-   Lasso
-   distance-based algorithms

Example standardization:

$$ \boxed{
z=
\frac{x-\mu}{\sigma}
} $$

Scaling does not change the fundamental meaning of OLS, but it can
dramatically improve optimization behavior and make regularization
penalties comparable across features.

------------------------------------------------------------------------

# 62. Gradient Descent

The Normal Equation gives a closed-form solution.

Gradient Descent instead solves the optimization problem iteratively.

Define:

$$ J(\beta) = `\frac{1}{2n}` \|X\beta-y\|^2 $$

The gradient is:

$$ \boxed{
\nabla J(\beta)
=
\frac1nX^T(X\beta-y)
} $$

The update is:

$$ \boxed{
\beta
\leftarrow
\beta-\alpha\nabla J(\beta)
} $$

where (`\alpha`) is the learning rate.

------------------------------------------------------------------------

# 63. Why the (1/2) Appears

Suppose:

$$
J(\beta)=`\frac1{2n}``\sum`(y_i-\hat{y}_i)^2
$$

Differentiating the square produces a factor of 2.

The factor (1/2) cancels it.

It does not change the location of the minimum.

------------------------------------------------------------------------

# 64. Manual Gradient Descent Example

Take:

$$ X=
```
\begin{bmatrix}
1&1\\
1&2\\
1&3
\end{bmatrix}
```
$$

and:

$$ y=
```
\begin{bmatrix}
2\\
4\\
6
\end{bmatrix}
```
$$

Start:

$$ \beta=
```
\begin{bmatrix}
0\\
0
\end{bmatrix}
```
$$

Learning rate:

$$ `\alpha=0.1` $$

Initial prediction:

$$ X\beta=
```
\begin{bmatrix}
0\\0\\0
\end{bmatrix}
```
$$

Error:

$$ X\beta-y=
```
\begin{bmatrix}
-2\\-4\\-6
\end{bmatrix}
```
$$

Calculate:

$$ X^T(X\beta-y) =
```
\begin{bmatrix}
1&1&1\\
1&2&3
\end{bmatrix}
\begin{bmatrix}
-2\\-4\\-6
\end{bmatrix}
```
$$

Therefore:

# \$\$

    \begin{bmatrix}
    -12\\
    -28
    \end{bmatrix}

\$\$

Divide by (n=3):

$$ `\nabla `J=
```
\begin{bmatrix}
-4\\
-9.3333
\end{bmatrix}
```
$$

Update:

$$ \beta_{new} = \beta-0.1`\nabla `J $$

Therefore:

$$ \boxed{
\beta_{new}
=
\begin{bmatrix}
0.4\\
0.9333
\end{bmatrix}
} $$

The algorithm repeats this process until convergence.

------------------------------------------------------------------------

# 65. Learning Rate

If (`\alpha`) is too small:

-   training becomes slow

If (`\alpha`) is too large:

-   optimization may oscillate
-   loss may increase
-   training may diverge

A useful learning-rate choice depends on the scale and conditioning of
the problem.

------------------------------------------------------------------------

# 66. Closed Form vs Gradient Descent

  Property             Normal Equation    Gradient Descent
  -------------------- ------------------ ------------------
  Type                 Closed form        Iterative
  Large feature sets   Can be expensive   Often preferable
  Learning rate        No                 Yes
  Scaling importance   Lower              Higher
  Implementation       Linear algebra     Optimization

For practical numerical linear regression, QR/SVD-based least-squares
solvers are often preferable to explicitly forming an inverse.

------------------------------------------------------------------------

# 67. Polynomial Regression

Polynomial Regression is still linear in its coefficients.

For example:

$$ \hat{y} = b_0+b_1x+b_2x^2+b_3x^3 $$

It is nonlinear in (x), but linear in:

$$ b_0,b_1,b_2,b_3 $$

Therefore it can still be solved using linear regression methods after
feature expansion.

------------------------------------------------------------------------

# 68. Interaction Terms

Suppose:

$$ \hat{y}=b_0+b_1x_1+b_2x_2+b_3x_1x_2 $$

The interaction term allows the effect of one variable to depend on the
value of another.

The marginal effect of (x_1) is:

$$ `\frac{\partial\hat{y}}{\partial x_1}` = b_1+b_3x_2 $$

So the effect of (x_1) changes with (x_2).

------------------------------------------------------------------------

# 69. Bias-Variance Tradeoff

A model's expected prediction error can conceptually be decomposed into:

$$ \boxed{
Bias^2+Variance+Irreducible\ Error
} $$

A model that is too simple may have high bias.

A model that is too flexible may have high variance.

Regularization attempts to trade some bias for lower variance.

------------------------------------------------------------------------

# 70. Train/Test Split

A model should be evaluated on data that was not used to fit it.

Typical workflow:

$$ Data `\rightarrow` Train/Test $$

Then:

$$ Train `\rightarrow` Fit $$

and:

$$ Test `\rightarrow` Evaluate $$

The test set should represent unseen data.

------------------------------------------------------------------------

# 71. Data Leakage

Data leakage occurs when information unavailable at prediction time
influences model training.

Common examples:

-   scaling the entire dataset before splitting
-   imputing using the full dataset
-   selecting features using the test set
-   using future information

Correct principle:

$$ Train preprocessing `\rightarrow` Test transformation $$

The preprocessing parameters must be learned from training data only.

------------------------------------------------------------------------

# 72. Cross-Validation

K-fold cross-validation divides training data into (K) folds.

For each fold:

1.  Train on (K-1) folds.
2.  Validate on the remaining fold.
3.  Repeat.
4.  Average the validation score.

This is useful for:

-   hyperparameter selection
-   model comparison
-   estimating generalization performance

The final test set should remain untouched until final evaluation.

------------------------------------------------------------------------

# 73. NumPy Implementation

``` python
import numpy as np

X = np.array([
    [1, 1],
    [1, 2],
    [1, 3],
    [1, 4],
    [1, 5]
], dtype=float)

y = np.array([3, 5, 4, 8, 7], dtype=float)

beta = np.linalg.lstsq(X, y, rcond=None)[0]

predictions = X @ beta

residuals = y - predictions

sse = np.sum(residuals ** 2)
mse = np.mean(residuals ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(residuals))
```

------------------------------------------------------------------------

# 74. Gradient Descent Implementation

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

# 75. Evaluation From Scratch

``` python
import numpy as np

def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    error = y_true - y_pred

    mae = np.mean(np.abs(error))
    mse = np.mean(error ** 2)
    rmse = np.sqrt(mse)

    sse = np.sum(error ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)

    r2 = 1 - sse / sst

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }
```

------------------------------------------------------------------------

# 76. scikit-learn Example

``` python
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(model.coef_)
print(model.intercept_)
```

Useful related metrics:

``` python
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)
```

------------------------------------------------------------------------

# 77. OnithrasML Workflow

A typical OnithrasML workflow should follow:

``` text
Raw Data
   ↓
Feature / Target Selection
   ↓
Train/Test Split
   ↓
Preprocessing
   ↓
LinearRegression.fit()
   ↓
Learned Coefficients
   ↓
LinearRegression.predict()
   ↓
Evaluation
   ↓
Residual Diagnostics
   ↓
Statistical Interpretation
```

------------------------------------------------------------------------

# 78. API Design

A production-quality LinearRegression API should clearly distinguish:

### Inputs

-   feature matrix (X)
-   target vector (y)

### Learned parameters

-   coefficients
-   intercept

### Operations

-   `fit()`
-   `predict()`
-   optionally `score()`

### Diagnostics

Depending on the implementation:

-   residual information
-   error metrics
-   coefficient statistics
-   standard errors
-   confidence intervals

------------------------------------------------------------------------

# 79. Complete Mathematical Pipeline

The entire method can be reduced to:

$$ \boxed{
y=X\beta+\epsilon
} $$

Fit:

$$ \boxed{
\hat{\beta}
=
\arg\min_\beta
\|y-X\beta\|^2
} $$

Normal Equation:

$$ \boxed{
X^TX\hat{\beta}=X^Ty
} $$

Closed-form solution when invertible:

$$ \boxed{
\hat{\beta}=(X^TX)^{-1}X^Ty
} $$

Prediction:

$$ \boxed{
\hat{y}=X\hat{\beta}
} $$

Residual:

$$ \boxed{
e=y-\hat{y}
} $$

Error:

$$ \boxed{
SSE=e^Te
} $$

Evaluation:

$$ \boxed{
MSE=\frac{SSE}{n}
} $$

$$ \boxed{
RMSE=\sqrt{MSE}
} $$

$$ \boxed{
R^2=1-\frac{SSE}{SST}
} $$

Inference:

$$ \boxed{
\widehat{Var}(\hat{\beta})
=
\hat{\sigma}^2(X^TX)^{-1}
} $$

$$ \boxed{
SE(\hat{\beta}_j)
=
\sqrt{
\widehat{Var}(\hat{\beta}_j)
}
} $$

$$ \boxed{
t=
\frac{\hat{\beta}_j-\beta_{j,0}}
{SE(\hat{\beta}_j)}
} $$

------------------------------------------------------------------------

# 80. Common Mistakes

## Mistake 1: Thinking R² is accuracy

R² is not classification accuracy.

## Mistake 2: Thinking high R² proves causation

Correlation and prediction do not establish causality.

## Mistake 3: Ignoring residuals

A good R² does not guarantee that model assumptions are appropriate.

## Mistake 4: Treating p-value as effect size

A tiny p-value does not necessarily mean the effect is practically
large.

## Mistake 5: Automatically deleting outliers

Outliers require investigation.

## Mistake 6: Using explicit inverse everywhere

Prefer stable linear algebra solvers.

## Mistake 7: Ignoring multicollinearity

Strongly correlated predictors can make individual coefficients
unstable.

## Mistake 8: Scaling before splitting

This can cause data leakage.

## Mistake 9: Using the test set for model selection

The test set should remain an unbiased final evaluation set.

## Mistake 10: Confusing residual error with model noise

Residuals are observed quantities:

$$ e=y-\hat{y} $$

while (`\epsilon`) is the theoretical error term.

------------------------------------------------------------------------

# 81. When Linear Regression Is a Good Choice

Linear Regression is a strong baseline when:

-   the target is continuous
-   the relationship is reasonably linear in parameters
-   interpretability matters
-   training speed matters
-   the dataset is not excessively high-dimensional
-   statistical inference is important

It is particularly valuable as a baseline even when more complex models
will eventually be used.

------------------------------------------------------------------------

# 82. When Basic Linear Regression May Not Be Enough

Consider other methods when:

-   relationships are strongly nonlinear
-   interactions dominate
-   data contains complex patterns
-   predictors are extremely numerous
-   temporal dependence is important
-   error variance has a complex structure
-   categorical or nonlinear effects need specialized treatment

Possible alternatives include:

-   Polynomial Regression
-   Ridge
-   Lasso
-   Elastic Net
-   Decision Trees
-   Random Forest
-   Gradient Boosting
-   XGBoost
-   Neural Networks

------------------------------------------------------------------------

# 83. Linear Regression Mental Model

Remember the following chain:

``` text
Data
 ↓
Design Matrix X
 ↓
Assume / model y = Xβ + ε
 ↓
Choose coefficients β
 ↓
Minimize squared residuals
 ↓
OLS
 ↓
Normal Equation
 ↓
β̂
 ↓
Predictions Xβ̂
 ↓
Residuals y - ŷ
 ↓
Metrics + Diagnostics
 ↓
Statistical Inference
```

This is the conceptual backbone of classical Linear Regression.

------------------------------------------------------------------------

# 84. Formula Reference

## Model

$$ y=X\beta+`\epsilon` $$

## Prediction

$$ \hat{y}=X\hat{\beta} $$

## Residual

$$ e=y-\hat{y} $$

## SSE

$$ SSE=`\sum`(y_i-\hat{y}_i)^2 $$

## MSE

$$ MSE=`\frac{SSE}{n}` $$

## RMSE

$$ RMSE=`\sqrt{MSE}` $$

## MAE

$$ MAE=`\frac`1n`\sum`\|y_i-\hat{y}_i\| $$

## Normal Equation

$$ \hat{\beta}=(X^TX)^{-1}X^Ty $$

## Gradient

$$ `\nabla `J= `\frac`1nX^T(X\beta-y) $$

## Gradient Descent

$$
\beta`\leftarrow`\beta-`\alpha``\nabla `J
$$

## Error Variance

$$ \hat`\sigma`^2= `\frac{SSE}{n-p-1}` $$

## Coefficient Variance

$$ `\widehat{Var}`(\hat{\beta}) =
\hat`\sigma`^2(X^TX)^{-1} $$

## Standard Error

$$ SE_j= `\sqrt{
\hat{\sigma}^2[(X^TX)^{-1}]_{jj}
}` $$

## t-statistic

$$ t= `\frac{\hat{\beta}_j-\beta_{j,0}}` {SE_j} $$

## Confidence Interval

$$ \hat{\beta}*j `\pm` t*{1-`\alpha`/2,df}SE_j
$$

## R²

$$ R^2=1-`\frac{SSE}{SST}` $$

## Adjusted R²

$$ R^2_{adj} = 1- `\frac{(1-R^2)(n-1)}` {n-p-1} $$

## F-statistic

$$ F= `\frac{SSR/p}` {SSE/(n-p-1)} $$

## VIF

$$ VIF_j=`\frac1{1-R_j^2}` $$

## Hat Matrix

$$ H=X(X^TX)^{-1}X^T $$

## WLS

$$ \hat{\beta}_{WLS} = (X^TWX)^{-1}X^TWy $$

## GLS

$$ \hat{\beta}_{GLS} = (X^T\Omega^{-1}X)^{-1}
X^T\Omega^{-1}y $$

## Ridge

$$ \hat{\beta}_{ridge} =
(X^TX+\lambda I)^{-1}X^Ty $$

------------------------------------------------------------------------

# 85. Final Perspective

Linear Regression is not merely:

``` python
model.fit(X, y)
```

That single line hides a complete mathematical system.

At its core:

$$ \boxed{
\text{Linear Regression}
=
\text{Linear Algebra}
+
\text{Optimization}
+
\text{Statistics}
+
\text{Probability}
} $$

The most important conceptual connection is:

$$ \boxed{
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
\text{Prediction}
\rightarrow
\text{Residuals}
\rightarrow
\text{Inference}
} $$

Once this chain is understood, formulas such as OLS, standard error,
t-test, confidence intervals, F-test, R², WLS, GLS, Ridge, and Lasso
stop being isolated formulas and become parts of the same system.

------------------------------------------------------------------------

# 86. Learning Checklist

Before considering Linear Regression complete, you should be able to
explain and derive:

-   [ ] Simple Linear Regression
-   [ ] Multiple Linear Regression
-   [ ] Design Matrix
-   [ ] Intercept column
-   [ ] Residual
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
-   [ ] Covariance / Variance slope formula
-   [ ] R²
-   [ ] Adjusted R²
-   [ ] Error variance
-   [ ] Degrees of freedom
-   [ ] Standard Error
-   [ ] t-statistic
-   [ ] p-value
-   [ ] Confidence Interval
-   [ ] z-test
-   [ ] F-test
-   [ ] Regression assumptions
-   [ ] Exogeneity
-   [ ] Homoscedasticity
-   [ ] Multicollinearity
-   [ ] VIF
-   [ ] Residual diagnostics
-   [ ] Leverage
-   [ ] Hat matrix
-   [ ] Cook's Distance
-   [ ] WLS
-   [ ] GLS
-   [ ] Ridge
-   [ ] Lasso
-   [ ] Lambda selection
-   [ ] Polynomial Regression
-   [ ] Interaction terms
-   [ ] Bias-Variance Tradeoff
-   [ ] Train/Test Split
-   [ ] Data Leakage
-   [ ] Cross Validation
-   [ ] Numerical Stability
-   [ ] Rank
-   [ ] SVD / QR concepts
-   [ ] NumPy implementation
-   [ ] Gradient Descent implementation
-   [ ] OnithrasML implementation

------------------------------------------------------------------------

## OnithrasML

This documentation is intended to make the Linear Regression
implementation understandable at three levels:

1.  **User level** --- how to use the estimator.
2.  **Mathematical level** --- why the estimator works.
3.  **Implementation level** --- how the mathematics maps to numerical
    computation.

That separation is important for a machine-learning library: users
should be able to start with `fit()` and `predict()`, while advanced
users can continue into OLS, optimization, inference, diagnostics, and
numerical linear algebra.
