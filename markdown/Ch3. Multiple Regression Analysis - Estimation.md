---
jupyter:
  jupytext:
    formats: notebooks//ipynb,markdown//md,scripts//py
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.6
  kernelspec:
    display_name: merino
    language: python
    name: python3
---

# 3. Multiple Regression Analysis: Estimation

```python
%pip install numpy pandas patsy statsmodels wooldridge -q
```

```python
import numpy as np
import pandas as pd
import patsy as pt
import statsmodels.formula.api as smf
import statsmodels.stats.outliers_influence as smo
import wooldridge as wool
```

## 3.1 Multiple Regression in Practice

$$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 +\beta_3 x_3 + \cdots + \beta_k x_k + u$$

### Example 3.1: Determinants of College GPA

$$\text{colGPA} = \beta_0 + \beta_1 \text{hsGPA} + \beta_2 \text{ACT} + u$$

```python
gpa1 = wool.data("gpa1")

reg = smf.ols(formula="colGPA ~ hsGPA + ACT", data=gpa1)
results = reg.fit()
print(f"results.summary(): \n{results.summary()}\n")
```

### Example 3.3 Hourly Wage Equation

$$ \log(\text{wage}) = \beta_0 + \beta_1 \text{educ} + \beta_2 \text{exper} + \beta_3 \text{tenure} + u$$

```python
wage1 = wool.data("wage1")

reg = smf.ols(formula="np.log(wage) ~ educ + exper + tenure", data=wage1)
results = reg.fit()
print(f"results.summary(): \n{results.summary()}\n")
```

### Example 3.4: Participation in 401(k) Pension Plans

$$ \text{prate} = \beta_0 + \beta_1 \text{mrate} + \beta_2 \text{age} + u$$

```python
k401k = wool.data("401k")

reg = smf.ols(formula="prate ~ mrate + age", data=k401k)
results = reg.fit()
print(f"results.summary(): \n{results.summary()}\n")
```

### Example 3.5a: Explaining Arrest Records

$$\text{narr86} = \beta_0 + \beta_1 \text{pcnv} + \beta_2 \text{ptime86} + \beta_3 \text{qemp86} + u$$

```python
crime1 = wool.data("crime1")

# model without avgsen:
reg = smf.ols(formula="narr86 ~ pcnv + ptime86 + qemp86", data=crime1)
results = reg.fit()
print(f"results.summary(): \n{results.summary()}\n")
```

### Example 3.5b: Explaining Arrest Records

$$\text{narr86} = \beta_0 + \beta_1 \text{pcnv} + \beta_2 \text{avgsen} + \beta_3 \text{ptime86} + \beta_4 \text{qemp86} + u$$

```python
crime1 = wool.data("crime1")

# model with avgsen:
reg = smf.ols(formula="narr86 ~ pcnv + avgsen + ptime86 + qemp86", data=crime1)
results = reg.fit()
print(f"results.summary(): \n{results.summary()}\n")
```

### Example 3.6: Hourly Wage Equation

$$ \log(\text{wage}) = \beta_0 + \beta_1 \text{educ} + u$$

```python
wage1 = wool.data("wage1")

reg = smf.ols(formula="np.log(wage) ~ educ", data=wage1)
results = reg.fit()
print(f"results.summary(): \n{results.summary()}\n")
```

## 3.2 OLS in Matrix Form

$$\hat{\beta} = (X'X)^{-1}X'y$$

```python
gpa1 = wool.data("gpa1")

# determine sample size & no. of regressors:
n = len(gpa1)
k = 2

# extract y:
y = gpa1["colGPA"]

# extract X & add a column of ones:
X = pd.DataFrame({"const": 1, "hsGPA": gpa1["hsGPA"], "ACT": gpa1["ACT"]})

# alternative with patsy:
y2, X2 = pt.dmatrices("colGPA ~ hsGPA + ACT", data=gpa1, return_type="dataframe")

# display first rows of X:
print(f"X.head(): \n{X.head()}\n")
```

```python
# parameter estimates:
X = np.array(X)
y = np.array(y).reshape(n, 1)  # creates a row vector
b = np.linalg.inv(X.T @ X) @ X.T @ y
print(f"b: \n{b}\n")
```

$$\hat{u} = y - X\hat{\beta}$$

$$\hat{\sigma}^2 = \frac{1}{n-k-1} \hat{u}'\hat{u}$$

```python
# residuals, estimated variance of u and SER:
u_hat = y - X @ b
sigsq_hat = (u_hat.T @ u_hat) / (n - k - 1)
SER = np.sqrt(sigsq_hat)
print(f"SER: {SER}\n")
```

$$\widehat{\text{var}(\hat{\beta})} = \hat{\sigma}^2 (X'X)^{-1}$$

```python
# estimated variance of the parameter estimators and SE:
Vbeta_hat = sigsq_hat * np.linalg.inv(X.T @ X)
se = np.sqrt(np.diagonal(Vbeta_hat))
print(f"se: {se}\n")
```

## 3.3 Ceteris Paribus Interpretation and Omitted Variable Bias

```python
gpa1 = wool.data("gpa1")

# parameter estimates for full and simple model:
reg = smf.ols(formula="colGPA ~ ACT + hsGPA", data=gpa1)
results = reg.fit()
b = results.params
print(f"b: \n{b}\n")
```

```python
# relation between regressors:
reg_delta = smf.ols(formula="hsGPA ~ ACT", data=gpa1)
results_delta = reg_delta.fit()
delta_tilde = results_delta.params
print(f"delta_tilde: \n{delta_tilde}\n")
```

```python
# omitted variables formula for b1_tilde:
b1_tilde = b["ACT"] + b["hsGPA"] * delta_tilde["ACT"]
print(f"b1_tilde:  \n{b1_tilde}\n")
```

```python
# actual regression with hsGPA omitted:
reg_om = smf.ols(formula="colGPA ~ ACT", data=gpa1)
results_om = reg_om.fit()
b_om = results_om.params
print(f"b_om: \n{b_om}\n")
```

## 3.4 Standard Errors, Multicollinearity, and VIF

```python
gpa1 = wool.data("gpa1")

# full estimation results including automatic SE:
reg = smf.ols(formula="colGPA ~ hsGPA + ACT", data=gpa1)
results = reg.fit()

# extract SER (instead of calculation via residuals):
SER = np.sqrt(results.mse_resid)

# regressing hsGPA on ACT for calculation of R2 & VIF:
reg_hsGPA = smf.ols(formula="hsGPA ~ ACT", data=gpa1)
results_hsGPA = reg_hsGPA.fit()
R2_hsGPA = results_hsGPA.rsquared
VIF_hsGPA = 1 / (1 - R2_hsGPA)
print(f"VIF_hsGPA: {VIF_hsGPA}\n")
```

```python
# manual calculation of SE of hsGPA coefficient:
n = results.nobs
sdx = np.std(gpa1["hsGPA"], ddof=1) * np.sqrt((n - 1) / n)
SE_hsGPA = 1 / np.sqrt(n) * SER / sdx * np.sqrt(VIF_hsGPA)
print(f"SE_hsGPA: {SE_hsGPA}\n")
```

```python
wage1 = wool.data("wage1")

# extract matrices using patsy:
y, X = pt.dmatrices(
    "np.log(wage) ~ educ + exper + tenure",
    data=wage1,
    return_type="dataframe",
)

# get VIF:
K = X.shape[1]
VIF = np.empty(K)
for i in range(K):
    VIF[i] = smo.variance_inflation_factor(X.values, i)
print(f"VIF: \n{VIF}\n")
```
