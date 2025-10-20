---
jupyter:
  jupytext:
    formats: notebooks//ipynb,markdown//md,scripts//py
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: merino
    language: python
    name: python3
---

# Chapter 8: Heteroskedasticity

Non-constant error variance (heteroskedasticity) represents one of the most common violations of classical regression assumptions in applied econometric work. This chapter examines the consequences of heteroskedasticity for OLS estimation and inference, develops formal tests for detecting its presence, and presents both robust inference methods and efficient estimation procedures that account for non-constant variance.

The development proceeds from diagnosis to correction strategies. We establish that heteroskedasticity preserves consistency and unbiasedness of OLS but invalidates standard inference procedures (Section 8.1), introduce heteroskedasticity-robust standard errors as a general solution (Section 8.2), develop formal tests for detecting heteroskedasticity including the Breusch-Pagan and White tests (Section 8.3), examine the linear probability model as a context where heteroskedasticity is inherent (Section 8.4), and conclude with weighted least squares (WLS) as an efficient estimation method when the form of heteroskedasticity is known (Section 8.5). Throughout, we emphasize practical implementation and interpretation using Python's statsmodels library.

**Consequences of Heteroskedasticity:**

Under assumptions MLR.1-MLR.4 (linearity, random sampling, no perfect collinearity, and zero conditional mean), but with heteroskedasticity (violation of MLR.5):

1.  **Consistency and Unbiasedness:** OLS coefficient estimates $\hat{\beta}_j$ remain **unbiased** and **consistent**. That is, $E(\hat{\beta}_j) = \beta_j$ and $\hat{\beta}_j \xrightarrow{p} \beta_j$ as $n \to \infty$. The point estimates themselves are not affected by heteroskedasticity.

2.  **Invalid Inference:** The usual OLS standard errors, $\widehat{\text{SE}}(\hat{\beta}_j)$, are **biased and inconsistent** estimates of the true standard errors, even in large samples. Consequently, the usual t-statistics, F-statistics, confidence intervals, and p-values are **invalid**. They can be either too small or too large, leading to incorrect inference (e.g., rejecting true null hypotheses too often or failing to reject false null hypotheses).

3.  **Loss of Efficiency:** OLS is no longer BLUE (Best Linear Unbiased Estimator). There exist other linear unbiased estimators (e.g., Weighted Least Squares with correct weights) that have smaller variances than OLS. However, OLS remains consistent, which is often sufficient for large samples.

**Solutions:** Use heteroskedasticity-robust standard errors (e.g., White/HC standard errors) for valid inference with OLS estimates, or use Weighted Least Squares (WLS) if the form of heteroskedasticity can be modeled.

We will cover:
*   How to obtain valid inference in the presence of heteroskedasticity using **robust standard errors**.
*   Methods for **testing** whether heteroskedasticity is present.
*   **Weighted Least Squares (WLS)** as an alternative estimation method that can be more efficient than OLS when the form of heteroskedasticity is known or can be reasonably estimated.

First, let's install and import the necessary libraries.

```python
# %pip install numpy pandas patsy statsmodels wooldridge -q
```

```python
import numpy as np
import pandas as pd
import patsy as pt  # Used for creating design matrices easily from formulas
import statsmodels.api as sm  # Provides statistical models and tests
import statsmodels.formula.api as smf  # Convenient formula interface for statsmodels
import wooldridge as wool  # Access to Wooldridge textbook datasets
from IPython.display import display
```

## 8.1 Heteroskedasticity-Robust Inference

Even if heteroskedasticity is present, we can still use OLS coefficient estimates but compute different standard errors that are robust to the presence of heteroskedasticity (of unknown form). These are often called **White standard errors** or **heteroskedasticity-consistent (HC)** standard errors.

Using robust standard errors allows for valid t-tests, F-tests, and confidence intervals based on the OLS estimates, even if the error variance is not constant. `statsmodels` provides several versions of robust standard errors (HC0, HC1, HC2, HC3), which differ mainly in their finite-sample adjustments. HC0 is the original White estimator, while HC1, HC2, and HC3 apply different corrections, often performing better in smaller samples (HC3 is often recommended).

### Example 8.2: Heteroskedasticity-Robust Inference for GPA Equation

We estimate a model for college cumulative GPA (`cumgpa`) using data for students observed in the spring semester (`spring == 1`). We compare the standard OLS results with results using robust standard errors.

```python
# Load the GPA data
gpa3 = wool.data("gpa3")

# Define the regression model using statsmodels formula API
# We predict cumgpa based on SAT score, high school percentile, total credit hours,
# gender, and race dummies.
# We only use data for the spring semester using the 'subset' argument.
reg = smf.ols(
    formula="cumgpa ~ sat + hsperc + tothrs + female + black + white",
    data=gpa3,
    subset=(gpa3["spring"] == 1),  # Use only spring observations
)

# --- Estimate with Default (Homoskedasticity-Assumed) Standard Errors ---
results_default = reg.fit()

# Display the results in a table
# OLS Results with Default Standard Errors
table_default = pd.DataFrame(
    {
        "b": round(results_default.params, 5),  # OLS Coefficients
        "se": round(results_default.bse, 5),  # Default Standard Errors
        "t": round(results_default.tvalues, 5),  # t-statistics based on default SEs
        "pval": round(results_default.pvalues, 5),  # p-values based on default SEs
    },
)
display(table_default)

# Interpretation (Default): Based on these standard errors, variables like sat, hsperc,
# tothrs, female, and black appear statistically significant (p < 0.05).
# However, if heteroskedasticity is present, these SEs and p-values are unreliable.

# --- Estimate with White's Original Robust Standard Errors (HC0) ---
# We fit the same model, but specify cov_type='HC0' to get robust SEs.
results_white = reg.fit(cov_type="HC0")

# Display the results
# OLS Results with Robust (HC0) Standard Errors
table_white = pd.DataFrame(
    {
        "b": round(results_white.params, 5),  # OLS Coefficients (same as default)
        "se": round(results_white.bse, 5),  # Robust (HC0) Standard Errors
        "t": round(results_white.tvalues, 5),  # Robust t-statistics
        "pval": round(results_white.pvalues, 5),  # Robust p-values
    },
)
display(table_white)

# Interpretation (HC0): The coefficient estimates 'b' are identical to the default OLS run.
# However, the standard errors 'se' have changed for most variables compared to the default.
# For example, the SE for 'tothrs' increased from 0.00104 to 0.00121, reducing its t-statistic
# and increasing its p-value (though still significant). The SE for 'black' decreased slightly.
# The conclusions about significance might change depending on the variable and significance level.

# --- Estimate with Refined Robust Standard Errors (HC3) ---
# HC3 applies a different small-sample correction, often preferred over HC0.
results_refined = reg.fit(cov_type="HC3")

# Display the results
# OLS Results with Robust (HC3) Standard Errors
table_refined = pd.DataFrame(
    {
        "b": round(results_refined.params, 5),  # OLS Coefficients (same as default)
        "se": round(results_refined.bse, 5),  # Robust (HC3) Standard Errors
        "t": round(results_refined.tvalues, 5),  # Robust t-statistics
        "pval": round(results_refined.pvalues, 5),  # Robust p-values
    },
)
display(table_refined)

# Interpretation (HC3): The HC3 robust standard errors are slightly different from the HC0 SEs
# (e.g., SE for 'tothrs' is 0.00123 with HC3 vs 0.00121 with HC0). In this specific case,
# the differences between HC0 and HC3 are minor and don't change the conclusions about
# statistical significance compared to HC0. Using robust standard errors confirms that
# sat, hsperc, tothrs, female, and black have statistically significant effects on cumgpa
# in this sample, even if heteroskedasticity is present.
```

Robust standard errors can also be used for hypothesis tests involving multiple restrictions, such as F-tests. We test the joint significance of the race dummies (`black` and `white`), comparing the standard F-test (assuming homoskedasticity) with robust F-tests.

```python
# Reload data if needed
gpa3 = wool.data("gpa3")

# Define the model again
reg = smf.ols(
    formula="cumgpa ~ sat + hsperc + tothrs + female + black + white",
    data=gpa3,
    subset=(gpa3["spring"] == 1),
)
# Define the null hypothesis for the F-test: H0: beta_black = 0 AND beta_white = 0
hypotheses = ["black = 0", "white = 0"]

# --- F-Test using Default (Homoskedasticity-Assumed) VCOV ---
results_default = reg.fit()  # Fit with default SEs
ftest_default = results_default.f_test(hypotheses)
fstat_default = ftest_default.statistic  # Extract F-statistic value
fpval_default = ftest_default.pvalue
# F-Test Comparison
ftest_comparison = pd.DataFrame(
    {
        "Test Type": ["Default F-test"],
        "F-statistic": [f"{fstat_default:.4f}"],
        "p-value": [f"{fpval_default:.4f}"],
    },
)
display(ftest_comparison)

# Interpretation (Default F-Test): The default F-test has F-statistic 0.6796 and p-value 0.5075,
# failing to reject the null hypothesis that race variables are jointly insignificant.

# --- F-Test using Robust (HC3) VCOV ---
results_hc3 = reg.fit(cov_type="HC3")  # Fit with HC3 robust SEs
ftest_hc3 = results_hc3.f_test(
    hypotheses,
)  # Perform F-test using the robust VCOV matrix
fstat_hc3 = ftest_hc3.statistic
fpval_hc3 = ftest_hc3.pvalue
ftest_hc3 = pd.DataFrame(
    {
        "Test Type": ["Robust (HC3) F-test"],
        "F-statistic": [f"{fstat_hc3:.4f}"],
        "p-value": [f"{fpval_hc3:.4f}"],
    },
)
display(ftest_hc3)

# Interpretation (HC3 F-Test): The robust F-statistic (0.6725) is slightly smaller than the
# default (0.6796), and the p-value (0.5111) is slightly larger but consistent with the default.
# The conclusion remains the same: we fail to reject the null hypothesis and conclude that race
# is not jointly statistically significant, even after accounting for potential heteroskedasticity.

# --- F-Test using Robust (HC0) VCOV ---
results_hc0 = reg.fit(cov_type="HC0")  # Fit with HC0 robust SEs
ftest_hc0 = results_hc0.f_test(hypotheses)
fstat_hc0 = ftest_hc0.statistic
fpval_hc0 = ftest_hc0.pvalue
ftest_hc0 = pd.DataFrame(
    {
        "Test Type": ["Robust (HC0) F-test"],
        "F-statistic": [f"{fstat_hc0:.4f}"],
        "p-value": [f"{fpval_hc0:.4f}"],
    },
)
display(ftest_hc0)

# Interpretation (HC0 F-Test): The HC0 robust F-test gives F-statistic 0.7478 and p-value 0.4741,
# very similar results to the HC3 test. In general, if the default and robust test statistics
# lead to different conclusions, the robust result is preferred.
```

## 8.2 Heteroskedasticity Tests

While robust inference provides a way to proceed despite heteroskedasticity, it's often useful to formally test for its presence. Tests can help understand the data better and decide whether alternative estimation methods like WLS might be beneficial (for efficiency).

### Breusch-Pagan Test

The **Breusch-Pagan (BP) test** checks if the error variance is systematically related to the explanatory variables.
*   $H_0$: Homoskedasticity ($Var(u|X) = \sigma^2$, constant variance)
*   $H_1$: Heteroskedasticity ($Var(u|X)$ depends on $X$)

The test involves:
1.  Estimate the original model by OLS and obtain the squared residuals, $\hat{u}^2$.
2.  Regress the squared residuals on the original explanatory variables: $\hat{u}^2 = \delta_0 + \delta_1 x_1 + ... + \delta_k x_k + \text{error}$.
3.  Test the joint significance of $\delta_1, ..., \delta_k$. If they are jointly significant (using an F-test or LM test), we reject $H_0$ and conclude heteroskedasticity is present.

### Example 8.4: Heteroskedasticity in a Housing Price Equation (Levels)

We test for heteroskedasticity in a model explaining house prices (`price`) using lot size (`lotsize`), square footage (`sqrft`), and number of bedrooms (`bdrms`).

```python
# Load housing price data
hprice1 = wool.data("hprice1")

# 1. Estimate the original model (price in levels)
reg = smf.ols(formula="price ~ lotsize + sqrft + bdrms", data=hprice1)
results = reg.fit()

# Display the OLS results (for context)
# --- OLS Results (Levels Model) ---
table_results = pd.DataFrame(
    {
        "b": round(results.params, 4),
        "se": round(results.bse, 4),
        "t": round(results.tvalues, 4),
        "pval": round(results.pvalues, 4),
    },
)
table_results  # Display OLS estimates
```

```python
# --- Breusch-Pagan Test (LM version) using statsmodels function ---
# We need the residuals from the original model and the design matrix (X).
# patsy.dmatrices helps create the X matrix easily from the formula.
y, X = pt.dmatrices(
    "price ~ lotsize + sqrft + bdrms",  # Formula defines X
    data=hprice1,
    return_type="dataframe",
)
# Perform Breusch-Pagan test for heteroskedasticity
# H_0: Var(u|X) = sigma^2 (homoskedasticity)
# H_1: Var(u|X) depends on X (heteroskedasticity)
# Separator line removed
# BREUSCH-PAGAN TEST FOR HETEROSKEDASTICITY
# Separator line removed

# Run the test using statsmodels built-in function
bp_test_results = sm.stats.diagnostic.het_breuschpagan(
    results.resid,  # OLS residuals
    X,  # Design matrix (predictors)
)

# Extract test statistics
bp_lm_statistic = bp_test_results[0]  # Lagrange Multiplier test statistic
bp_lm_pvalue = bp_test_results[1]  # p-value for LM test
bp_f_statistic = bp_test_results[2]  # F-statistic version
bp_f_pvalue = bp_test_results[3]  # p-value for F-test

# Display results with interpretation
bp_results = pd.DataFrame(
    {
        "Test": ["LM Test", "LM Test", "F Test", "F Test"],
        "Metric": ["Statistic", "p-value", "Statistic", "p-value"],
        "Value": [
            f"{bp_lm_statistic:8.4f}",
            f"{bp_lm_pvalue:8.4f}",
            f"{bp_f_statistic:8.4f}",
            f"{bp_f_pvalue:8.4f}",
        ],
    },
)
display(bp_results)

# Interpretation
significance_level = 0.05
test_decision = pd.DataFrame(
    {
        "Decision": [
            f"REJECT H_0 at {significance_level:.0%} level"
            if bp_lm_pvalue < significance_level
            else f"FAIL TO REJECT H_0 at {significance_level:.0%} level",
        ],
        "Conclusion": [
            "Evidence of heteroskedasticity detected"
            if bp_lm_pvalue < significance_level
            else "No significant evidence of heteroskedasticity",
        ],
        "Recommendation": [
            "Use robust standard errors for valid inference"
            if bp_lm_pvalue < significance_level
            else "OLS standard errors are likely valid",
        ],
    },
)
test_decision
```

```python
# --- Breusch-Pagan Test (F version) calculated manually ---
# This demonstrates the underlying steps.

# 2. Get squared residuals
hprice1["resid_sq"] = results.resid**2

# 3. Regress squared residuals on original predictors
reg_resid = smf.ols(formula="resid_sq ~ lotsize + sqrft + bdrms", data=hprice1)
results_resid = reg_resid.fit()

# The F-statistic from this regression is the BP F-test statistic.
bp_F_statistic = results_resid.fvalue
bp_F_pval = results_resid.f_pvalue
# Display Breusch-Pagan test results
pd.DataFrame(
    {
        "Metric": ["BP F statistic", "BP F p-value"],
        "Value": [f"{bp_F_statistic:.4f}", f"{bp_F_pval:.4f}"],
    },
)

# Interpretation (BP F Test): The F-statistic is 5.3389, and the p-value is 0.0020.
# This also leads to rejecting the null hypothesis of homoskedasticity at the 5% level.
# The conclusion matches the LM version. Using robust standard errors for the original
# price model is recommended.
```

### White Test

The **White test** is a more general test for heteroskedasticity that doesn't assume a specific linear relationship between the variance and the predictors. It tests whether the variance depends on any combination of the levels, squares, and cross-products of the original regressors.

A simplified version (often used in practice) regresses the squared residuals $\hat{u}^2$ on the OLS fitted values $\hat{y}$ and squared fitted values $\hat{y}^2$:
$$ \hat{u}^2 = \delta_0 + \delta_1 \hat{y} + \delta_2 \hat{y}^2 + \text{error} $$
An F-test (or LM test) for $H_0: \delta_1 = 0, \delta_2 = 0$ is performed.

### Example 8.5: BP and White test in the Log Housing Price Equation

Often, taking the logarithm of the dependent variable can mitigate heteroskedasticity. We now test the log-log housing price model.

```python
# Load housing price data again if needed
hprice1 = wool.data("hprice1")

# Estimate the model with log(price) as the dependent variable
reg_log = smf.ols(
    formula="np.log(price) ~ np.log(lotsize) + np.log(sqrft) + bdrms",
    data=hprice1,
)
results_log = reg_log.fit()

# --- Breusch-Pagan Test (Log Model) ---
y_log, X_bp_log = pt.dmatrices(
    "np.log(price) ~ np.log(lotsize) + np.log(sqrft) + bdrms",
    data=hprice1,
    return_type="dataframe",
)
# --- Breusch-Pagan Test (Log Model) ---
result_bp_log = sm.stats.diagnostic.het_breuschpagan(results_log.resid, X_bp_log)
bp_statistic_log = result_bp_log[0]
bp_pval_log = result_bp_log[1]
# Display Breusch-Pagan test results (log model)
pd.DataFrame(
    {
        "Metric": ["BP LM statistic", "BP LM p-value"],
        "Value": [f"{bp_statistic_log:.4f}", f"{bp_pval_log:.4f}"],
    },
)

# Interpretation (BP Test, Log Model): The LM statistic is 4.2232, and the p-value is 0.2383.
# We fail to reject the null hypothesis of homoskedasticity at conventional levels (e.g., 5% or 10%).
# This suggests that the BP test does not find evidence of heteroskedasticity related linearly
# to the log predictors in this log-transformed model.
```

```python
# --- White Test (Simplified Version using Fitted Values, Log Model) ---
# Create the design matrix for the White test auxiliary regression
X_wh_log = pd.DataFrame(
    {
        "const": 1,  # Include constant
        "fitted_reg": results_log.fittedvalues,  # OLS fitted values from log model
        "fitted_reg_sq": results_log.fittedvalues**2,  # Squared fitted values
    },
)
# Use the het_breuschpagan function with the new design matrix X_wh_log
# --- White Test (Log Model) ---
result_white_log = sm.stats.diagnostic.het_breuschpagan(results_log.resid, X_wh_log)
white_statistic_log = result_white_log[0]
white_pval_log = result_white_log[1]
# Display White test results (log model)
pd.DataFrame(
    {
        "Metric": ["White LM statistic", "White LM p-value"],
        "Value": [f"{white_statistic_log:.4f}", f"{white_pval_log:.4f}"],
    },
)

# Interpretation (White Test, Log Model): The White test LM statistic is 3.4473, and the
# p-value is 0.1784. Unlike the BP test, the White test does not reject the null hypothesis
# of homoskedasticity at the 5% level. This indicates that the variance might not be
# related to the level of predicted log(price) in a non-linear way captured by the fitted values.
# This suggests that after logging the dependent variable, heteroskedasticity is not
# significantly detected by the White test.
```

## 8.3 Weighted Least Squares (WLS)

If heteroskedasticity is detected and we have an idea about its form, we can use **Weighted Least Squares (WLS)**. WLS is a transformation of the original model that yields estimators that are BLUE (efficient) under heteroskedasticity, provided the variance structure is correctly specified.

The idea is to divide the original equation by the standard deviation of the error term, $\sqrt{h(X)}$, where $h(X) = Var(u|X)$. This gives more weight to observations with smaller error variance and less weight to observations with larger error variance.

In practice, $h(X)$ is unknown, so we use an estimate $\hat{h}(X)$. This leads to **Feasible Generalized Least Squares (FGLS)**. A common approach is to specify a model for the variance function, estimate it, and use the predicted variances to construct the weights $w = 1/\hat{h}(X)$.

### Example 8.6: Financial Wealth Equation (WLS with Assumed Weights)

We model net total financial assets (`nettfa`) for single-person households (`fsize == 1`) as a function of income (`inc`), age (`age`), gender (`male`), and 401k eligibility (`e401k`). It's plausible that the variance of `nettfa` increases with income. We assume $Var(u|inc, age, ...) = \sigma^2 inc$. Thus, the standard deviation is $\sigma \sqrt{inc}$, and the appropriate weight for WLS is $w = 1/inc$.

```python
# Load 401k subsample data
k401ksubs = wool.data("401ksubs")

# Create subset for single-person households
k401ksubs_sub = k401ksubs[
    k401ksubs["fsize"] == 1
].copy()  # Use .copy() to avoid SettingWithCopyWarning

# --- OLS Estimation (with Robust SEs for comparison) ---
# We estimate by OLS first, but use robust standard errors (HC0) because we suspect heteroskedasticity.
reg_ols = smf.ols(
    formula="nettfa ~ inc + I((age-25)**2) + male + e401k",  # Note: age modeled quadratically around 25
    data=k401ksubs_sub,
)
results_ols = reg_ols.fit(cov_type="HC0")  # Use robust SEs

# Display OLS results
# --- OLS Results with Robust (HC0) SEs (Singles Only) ---
table_ols = pd.DataFrame(
    {
        "b": round(results_ols.params, 4),
        "se": round(results_ols.bse, 4),
        "t": round(results_ols.tvalues, 4),
        "pval": round(results_ols.pvalues, 4),
    },
)
table_ols  # Display OLS with robust SEs
```

```python
# --- WLS Estimation (Assuming Var = sigma^2 * inc) ---
# Define the weights as 1/inc. statsmodels expects a list or array of weights.
wls_weight = list(1 / k401ksubs_sub["inc"])

# Estimate using smf.wls, passing the weights vector
reg_wls = smf.wls(
    formula="nettfa ~ inc + I((age-25)**2) + male + e401k",
    weights=wls_weight,
    data=k401ksubs_sub,
)
# By default, WLS provides standard errors assuming the weight specification is correct.
results_wls = reg_wls.fit()

# Display WLS results
# --- WLS Results (Weights = 1/inc) ---
table_wls = pd.DataFrame(
    {
        "b": round(results_wls.params, 4),  # WLS Coefficients
        "se": round(results_wls.bse, 4),  # WLS Standard Errors (assume correct weights)
        "t": round(results_wls.tvalues, 4),  # WLS t-statistics
        "pval": round(results_wls.pvalues, 4),  # WLS p-values
    },
)
table_wls  # Display WLS estimates

# Interpretation (OLS vs WLS):
# Comparing WLS to OLS (robust), the coefficient estimates differ somewhat (e.g., 'inc' coeff
# is 0.7706 in OLS, 0.7404 in WLS). The WLS standard errors are generally smaller than the
# OLS robust SEs (e.g., SE for 'inc' is 0.0643 in WLS vs 0.0994 in OLS robust).
# This suggests WLS is more efficient *if* the assumption Var=sigma^2*inc is correct.
# The coefficient on e401k (eligibility) is positive and significant in both models,
# with point estimate 5.1883 in WLS vs 6.8862 in OLS robust.
```

What if our assumed variance function ($Var = \sigma^2 inc$) is wrong? The WLS estimator will still be consistent (under standard assumptions) but its standard errors might be incorrect, and it might not be efficient. We can compute robust standard errors *for the WLS estimator* to get valid inference even if the weights are misspecified.

```python
# Reload data and prepare WLS if needed
k401ksubs = wool.data("401ksubs")
k401ksubs_sub = k401ksubs[k401ksubs["fsize"] == 1].copy()
wls_weight = list(1 / k401ksubs_sub["inc"])
reg_wls = smf.wls(
    formula="nettfa ~ inc + I((age-25)**2) + male + e401k",
    weights=wls_weight,
    data=k401ksubs_sub,
)

# --- WLS Results with Default (Non-Robust) SEs ---
results_wls_default = reg_wls.fit()
# --- WLS Results with Default (Non-Robust) Standard Errors ---
table_default_wls = pd.DataFrame(
    {
        "b": round(results_wls_default.params, 4),
        "se": round(results_wls_default.bse, 4),
        "t": round(results_wls_default.tvalues, 4),
        "pval": round(results_wls_default.pvalues, 4),
    },
)
table_default_wls  # Display Default WLS SEs
```

```python
# --- WLS Results with Robust (HC3) Standard Errors ---
# Fit the WLS model but request robust standard errors.
results_wls_robust = reg_wls.fit(cov_type="HC3")
# --- WLS Results with Robust (HC3) Standard Errors ---
table_robust_wls = pd.DataFrame(
    {
        "b": round(
            results_wls_robust.params,
            4,
        ),  # WLS Coefficients (same as default WLS)
        "se": round(results_wls_robust.bse, 4),  # Robust SEs applied to WLS
        "t": round(results_wls_robust.tvalues, 4),  # Robust t-statistics
        "pval": round(results_wls_robust.pvalues, 4),  # Robust p-values
    },
)
table_robust_wls  # Display Robust WLS SEs

# Interpretation (Default WLS SE vs Robust WLS SE):
# Comparing the robust WLS SEs to the default WLS SEs, we see some differences, though
# perhaps less dramatic than the OLS vs Robust OLS comparison earlier. For instance, the
# robust SE for 'inc' (0.0752) is slightly larger than the default WLS SE (0.0643).
# The robust SE for e401k (1.5743) is also smaller than the default (1.7034).
# This suggests that the initial assumption Var=sigma^2*inc might not perfectly capture
# the true heteroskedasticity. However, the conclusions about significance remain largely
# unchanged in this case. Using robust standard errors with WLS provides insurance against
# misspecification of the variance function used for weights.
```

### Example 8.7: Demand for Cigarettes (FGLS with Estimated Weights)

Here, we don't assume the form of heteroskedasticity beforehand. Instead, we estimate it using **Feasible GLS (FGLS)**.
1.  Estimate the original model (cigarette demand) by OLS.
2.  Obtain the residuals $\hat{u}$ and square them.
3.  Model the log of squared residuals as a function of the predictors: $\log(\hat{u}^2) = \delta_0 + \delta_1 x_1 + ...$. This models the variance function $h(X)$.
4.  Obtain the fitted values from this regression, $\widehat{\log(u^2)}$. Exponentiate to get estimates of the variance: $\hat{h} = \exp(\widehat{\log(u^2)})$.
5.  Use weights $w = 1/\hat{h}$ in a WLS estimation of the original model.

```python
# Load smoking data
smoke = wool.data("smoke")

# --- Step 1: OLS Estimation of the Cigarette Demand Model ---
reg_ols_smoke = smf.ols(
    formula="cigs ~ np.log(income) + np.log(cigpric) +"
    "educ + age + I(age**2) + restaurn",  # restaurn is restriction dummy
    data=smoke,
)
results_ols_smoke = reg_ols_smoke.fit()
# --- OLS Results (Cigarette Demand) ---
table_ols_smoke = pd.DataFrame(
    {
        "b": round(results_ols_smoke.params, 4),
        "se": round(results_ols_smoke.bse, 4),
        "t": round(results_ols_smoke.tvalues, 4),
        "pval": round(results_ols_smoke.pvalues, 4),
    },
)
table_ols_smoke  # Display OLS estimates
```

```python
# --- Test for Heteroskedasticity (BP Test) ---
y_smoke, X_smoke = pt.dmatrices(
    "cigs ~ np.log(income) + np.log(cigpric) + educ +age + I(age**2) + restaurn",
    data=smoke,
    return_type="dataframe",
)
# --- Breusch-Pagan Test (Cigarette Demand) ---
result_bp_smoke = sm.stats.diagnostic.het_breuschpagan(results_ols_smoke.resid, X_smoke)
bp_statistic_smoke = result_bp_smoke[0]
bp_pval_smoke = result_bp_smoke[1]
# Display BP test results for smoking model
pd.DataFrame(
    {
        "Metric": ["BP LM statistic", "BP LM p-value"],
        "Value": [f"{bp_statistic_smoke:.4f}", f"{bp_pval_smoke:.4f}"],
    },
)

# Interpretation (BP Test): The p-value is 0.0000 (very small), strongly rejecting
# the null of homoskedasticity. FGLS is likely warranted for efficiency.
```

```python
# --- Step 2 & 3: Model the Variance Function ---
# Get residuals, square them, take the log (add small constant if any residuals are zero)
smoke["resid_ols"] = results_ols_smoke.resid
# Ensure no log(0) issues if resid can be exactly zero (unlikely with continuous data)
# A common fix is adding a tiny constant or dropping zero residuals if they exist.
# Here, we assume residuals are non-zero. If errors occur, check this.
smoke["logu2"] = np.log(smoke["resid_ols"] ** 2)

# Regress log(u^2) on the original predictors
reg_varfunc = smf.ols(
    formula="logu2 ~ np.log(income) + np.log(cigpric) +"
    "educ + age + I(age**2) + restaurn",
    data=smoke,  # Need to handle potential -inf if resid^2 was 0
    missing="drop",  # Drop rows where logu2 might be invalid
)
results_varfunc = reg_varfunc.fit()
# --- Variance Function Estimation Results (log(u^2) regressed on X) ---
table_varfunc = pd.DataFrame(
    {
        "b": round(results_varfunc.params, 4),
        "se": round(results_varfunc.bse, 4),
        "t": round(results_varfunc.tvalues, 4),
        "pval": round(results_varfunc.pvalues, 4),
    },
)
table_varfunc  # Display variance function estimates

# Interpretation (Variance Function): This regression tells us which variables are
# significantly related to the log error variance. For instance, log(income) (coefficient 0.2915),
# educ (coefficient -0.0797), age (coefficient 0.2040), and restaurn (coefficient -0.6270)
# appear significant predictors of the variance.
```

```python
# --- Step 4 & 5: FGLS Estimation using Estimated Weights ---
# Get fitted values from the variance function regression
smoke["logh_hat"] = results_varfunc.fittedvalues
# Calculate the estimated variance h_hat = exp(logh_hat)
smoke["h_hat"] = np.exp(smoke["logh_hat"])
# Calculate the weights for WLS: w = 1 / h_hat
wls_weight_fgls = list(1 / smoke["h_hat"])

# Estimate the original demand model using WLS with estimated weights
reg_fgls_wls = smf.wls(
    formula="cigs ~ np.log(income) + np.log(cigpric) +"
    "educ + age + I(age**2) + restaurn",
    weights=wls_weight_fgls,
    data=smoke,
    missing="drop",  # Ensure consistent sample with variance estimation
)
results_fgls_wls = reg_fgls_wls.fit()
# --- FGLS (WLS with Estimated Weights) Results ---
table_fgls_wls = pd.DataFrame(
    {
        "b": round(results_fgls_wls.params, 4),
        "se": round(results_fgls_wls.bse, 4),  # FGLS standard errors
        "t": round(results_fgls_wls.tvalues, 4),
        "pval": round(results_fgls_wls.pvalues, 4),
    },
)
table_fgls_wls  # Display FGLS estimates

# Interpretation (FGLS vs OLS):
# Comparing the FGLS estimates to the original OLS estimates:
# - The coefficient for log(cigpric) is -2.9403 in FGLS vs -0.7509 in OLS. The FGLS estimate is still insignificant.
# - The coefficient for the restaurant smoking restriction ('restaurn') is -3.4611 in FGLS vs -2.8251 in OLS, and remains significant.
# - Standard errors have generally changed. For instance, the SE for log(income) decreased from 0.7278 (OLS) to 0.4370 (FGLS).
# - FGLS estimates are preferred for efficiency if the variance model is reasonably well-specified.
# One could also compute robust standard errors for the FGLS estimates as a further check.
```

## 8.4 The Linear Probability Model Revisited

In Chapter 7, we introduced the **linear probability model (LPM)** for binary dependent variables, where the outcome $y_i \in \{0, 1\}$. We noted that LPM has heteroskedastic errors by construction. In this section, we examine the form of this heteroskedasticity and explore appropriate estimation and inference methods.

### 8.4.1 Why the LPM Has Heteroskedasticity

Recall the LPM specification:

$$
y_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_k x_{ik} + u_i
$$

where $y_i$ is binary (0 or 1). Since $E(y_i | \mathbf{x}_i) = P(y_i = 1 | \mathbf{x}_i) = p_i$, we can write:

$$
u_i = y_i - p_i
$$

The error term $u_i$ can only take two values:

- When $y_i = 1$: $u_i = 1 - p_i$ with probability $p_i$
- When $y_i = 0$: $u_i = -p_i$ with probability $1 - p_i$

**Variance of the error:**

$$
\text{Var}(u_i | \mathbf{x}_i) = E(u_i^2 | \mathbf{x}_i) = p_i(1 - p_i)^2 + (1-p_i)p_i^2 = p_i(1-p_i)
$$

Since $p_i = \beta_0 + \beta_1 x_{i1} + \cdots + \beta_k x_{ik}$ varies across observations (depends on $\mathbf{x}_i$), the variance $\text{Var}(u_i | \mathbf{x}_i) = p_i(1-p_i)$ is **not constant**. This is heteroskedasticity.

**Implications:**

1. **OLS is still unbiased and consistent** for $\boldsymbol{\beta}$
2. **OLS standard errors are incorrect** - must use robust SEs
3. **WLS can improve efficiency** if we model the variance correctly

### 8.4.2 Heteroskedasticity-Robust Inference for LPM

The standard practice for LPM is to estimate with OLS and use **heteroskedasticity-robust standard errors**:

```python
# Example: Labor force participation from Ch7
mroz = wool.data("mroz")

# Estimate LPM with OLS
lpm_ols = smf.ols(
    formula="inlf ~ nwifeinc + educ + exper + I(exper**2) + age + kidslt6 + kidsge6",
    data=mroz,
)
results_lpm_ols = lpm_ols.fit()

# Usual (incorrect) standard errors
table_usual = pd.DataFrame(
    {
        "b": round(results_lpm_ols.params, 6),
        "se_usual": round(results_lpm_ols.bse, 6),
        "t_usual": round(results_lpm_ols.tvalues, 4),
    },
)

# Robust standard errors (HC3)
results_lpm_robust = lpm_ols.fit(cov_type="HC3")
table_robust = pd.DataFrame(
    {
        "b": round(results_lpm_robust.params, 6),
        "se_robust": round(results_lpm_robust.bse, 6),
        "t_robust": round(results_lpm_robust.tvalues, 4),
    },
)

# Compare
comparison_lpm = pd.DataFrame(
    {
        "Coefficient": results_lpm_ols.params.index,
        "Estimate": round(results_lpm_ols.params, 6),
        "SE (usual)": round(results_lpm_ols.bse, 6),
        "SE (robust)": round(results_lpm_robust.bse, 6),
        "Ratio": round(results_lpm_robust.bse / results_lpm_ols.bse, 3),
    },
)
print("LPM: Comparison of Usual vs Robust Standard Errors")
comparison_lpm
```

**Interpretation:**

- The robust SEs can be either larger or smaller than usual SEs
- For LPM, always report robust SEs and use them for t-tests and confidence intervals
- The "Ratio" column shows how much the SEs change when accounting for heteroskedasticity

### 8.4.3 Weighted Least Squares for LPM

Since we know the form of heteroskedasticity in LPM, we can use **WLS** to potentially improve efficiency. The variance is:

$$
\text{Var}(u_i | \mathbf{x}_i) = p_i(1-p_i)
$$

where $p_i = \mathbf{x}_i' \boldsymbol{\beta}$ is unknown. The **feasible WLS** procedure is:

1. Estimate LPM with OLS to get $\hat{p}_i = \hat{\beta}_0 + \hat{\beta}_1 x_{i1} + \cdots$
2. Compute estimated variance: $\hat{h}_i = \hat{p}_i(1 - \hat{p}_i)$
3. Use weights $w_i = 1/\hat{h}_i$ for WLS estimation

**Implementation:**

```python
# Step 1: Get fitted probabilities from OLS
mroz["p_hat"] = results_lpm_ols.fittedvalues

# Step 2: Compute estimated variance h_hat = p_hat * (1 - p_hat)
# To avoid division by zero or negative weights if p_hat is outside [0,1],
# we can either:
# (a) Trim predictions to (0.01, 0.99), or
# (b) Only use observations where 0 < p_hat < 1
mroz["h_hat"] = mroz["p_hat"] * (1 - mroz["p_hat"])

# Remove observations where h_hat <= 0 (would give invalid weights)
mroz_wls = mroz[mroz["h_hat"] > 0].copy()
print(f"Observations with valid weights: {len(mroz_wls)} out of {len(mroz)}")

# Step 3: WLS estimation with weights = 1/h_hat
lpm_wls = smf.wls(
    formula="inlf ~ nwifeinc + educ + exper + I(exper**2) + age + kidslt6 + kidsge6",
    weights=1 / mroz_wls["h_hat"],
    data=mroz_wls,
)
results_lpm_wls = lpm_wls.fit()

# Compare OLS and WLS estimates
comparison_ols_wls = pd.DataFrame(
    {
        "Variable": results_lpm_ols.params.index,
        "OLS_coef": round(results_lpm_ols.params, 6),
        "OLS_se_robust": round(results_lpm_robust.bse, 6),
        "WLS_coef": round(results_lpm_wls.params, 6),
        "WLS_se": round(results_lpm_wls.bse, 6),
    },
)
print("\nLPM: OLS vs WLS Estimation")
comparison_ols_wls
```

**Interpretation:**

- WLS coefficients should be similar to OLS if the model is correctly specified
- WLS standard errors may be smaller (efficiency gain) if variance model is correct
- If OLS and WLS give very different coefficient estimates, this suggests model misspecification

### 8.4.4 Potential Problems with WLS for LPM

While WLS addresses the known heteroskedasticity in LPM, there are practical issues:

**1. Predicted probabilities outside [0, 1]:**

```python
# Check range of fitted probabilities
print(f"Range of OLS fitted values: [{results_lpm_ols.fittedvalues.min():.4f}, {results_lpm_ols.fittedvalues.max():.4f}]")
print(
    f"Observations with p_hat < 0: {(results_lpm_ols.fittedvalues < 0).sum()}",
)
print(
    f"Observations with p_hat > 1: {(results_lpm_ols.fittedvalues > 1).sum()}",
)

# For these observations, h_hat = p_hat(1-p_hat) can be negative or problematic
# This violates the assumption that we know the form of heteroskedasticity
```

**2. Variance model misspecification:**

- The assumed form $h_i = p_i(1-p_i)$ is only correct if the LPM is the true model
- But LPM is a linear approximation to inherently nonlinear probability models
- If the true model is probit or logit, the variance formula is different
- Using incorrect weights can lead to **less** efficient estimates than OLS

**3. WLS standard errors may still be incorrect:**

```python
# Even WLS should use robust SEs as a precaution
results_lpm_wls_robust = lpm_wls.fit(cov_type="HC3")

comparison_wls_se = pd.DataFrame(
    {
        "Variable": results_lpm_wls.params.index,
        "WLS_se_usual": round(results_lpm_wls.bse, 6),
        "WLS_se_robust": round(results_lpm_wls_robust.bse, 6),
        "Ratio": round(results_lpm_wls_robust.bse / results_lpm_wls.bse, 3),
    },
)
print("\nWLS Standard Errors: Usual vs Robust")
comparison_wls_se
```

**Interpretation:**

- If WLS standard errors (usual vs robust) differ substantially, the variance model is likely misspecified
- Safest approach: Use WLS with robust SEs, or just use OLS with robust SEs

### 8.4.5 Practical Recommendations for LPM

Based on the analysis above, here are practical guidelines:

**Simple and reliable approach:**

1. Estimate LPM with OLS
2. **Always use heteroskedasticity-robust standard errors** (HC3)
3. Check fitted values - if many fall outside [0, 1], consider probit/logit (Chapter 17)

**WLS approach (if seeking efficiency):**

1. Estimate WLS using $w_i = 1/[\hat{p}_i(1-\hat{p}_i)]$
2. **Use robust standard errors** for the WLS estimates
3. Compare OLS and WLS coefficients - if very different, model may be misspecified
4. Report both OLS-robust and WLS-robust results for transparency

**When to avoid LPM altogether:**

- If most fitted probabilities fall outside [0, 0.8] or [0.2, 1]
- If the relationship is clearly nonlinear
- For policy analysis requiring precise probability estimates
- $\to$ Use probit or logit models instead (Chapter 17)

```python
# Summary comparison: Which method to use?
methods_comparison = pd.DataFrame(
    {
        "Method": [
            "OLS (usual SE)",
            "OLS (robust SE)",
            "WLS (usual SE)",
            "WLS (robust SE)",
        ],
        "Valid Inference?": ["No (heteroskedasticity)", "Yes", "Maybe", "Yes"],
        "Efficiency": ["No (not BLUE)", "No (not BLUE)", "Yes (if correct)", "Yes (if correct)"],
        "Recommendation": [
            "Never use",
            "Standard practice",
            "Use with caution",
            "Best if WLS desired",
        ],
    },
)
print("\nLPM Estimation Methods Comparison")
display(methods_comparison)
```

**Key Takeaway:**

The LPM has heteroskedasticity by construction with known form $\text{Var}(u_i|\mathbf{x}_i) = p_i(1-p_i)$. While WLS can exploit this structure for potential efficiency gains, practical problems (predictions outside [0,1], model misspecification) mean that **OLS with robust standard errors** remains the most reliable approach for LPM. For better modeling of binary outcomes, consider probit or logit models covered in Chapter 17.

## Chapter Summary

Chapter 8 addressed **heteroskedasticity** - the violation of the constant error variance assumption (MLR.5) in linear regression. While heteroskedasticity does not bias OLS coefficient estimates, it invalidates standard inference procedures and reduces efficiency.

### Key Concepts

**Heteroskedasticity Definition:**

- **Homoskedasticity (MLR.5):** $\text{Var}(u_i | \mathbf{x}_i) = \sigma^2$ for all $i$ (constant variance)
- **Heteroskedasticity:** $\text{Var}(u_i | \mathbf{x}_i) = \sigma_i^2$ varies across observations
- Common in cross-sectional data, especially with:
  - Grouped data (firm-level, city-level aggregates)
  - Income or wealth variables (variance increases with level)
  - Binary dependent variables (LPM)
  - Count data

**Consequences of Heteroskedasticity for OLS:**

1. **Unbiasedness and consistency preserved:**
   - $E(\hat{\beta}_j) = \beta_j$ (unbiased under MLR.1-4)
   - $\hat{\beta}_j \xrightarrow{p} \beta_j$ (consistent)
   - Point estimates are still valid

2. **Invalid standard errors and inference:**
   - Usual OLS standard errors $\widehat{\text{SE}}(\hat{\beta}_j)$ are biased and inconsistent
   - Can be either too small (over-rejection) or too large (under-rejection)
   - t-statistics, F-statistics, confidence intervals, p-values all invalid
   - **Solution:** Use heteroskedasticity-robust standard errors

3. **Loss of efficiency (not BLUE):**
   - OLS no longer has minimum variance among linear unbiased estimators
   - Weighted Least Squares (WLS) with correct weights is more efficient
   - Efficiency loss may be small in practice

**Heteroskedasticity-Robust Inference:**

- **White standard errors (HC):** Asymptotically valid regardless of heteroskedasticity form
- Variants: HC0 (original White), HC1, HC2, HC3 (finite-sample corrections)
- **HC3 recommended** for general use (best small-sample properties)
- In statsmodels: `model.fit(cov_type="HC3")`
- Applies to: t-tests, F-tests, confidence intervals
- **Practical guideline:** Always report robust SEs in cross-sectional regressions

**Testing for Heteroskedasticity:**

1. **Breusch-Pagan (BP) Test:**
   - Tests $H_0: \text{Var}(u_i|\mathbf{x}_i) = \sigma^2$ (homoskedasticity)
   - Regress squared OLS residuals on independent variables: $\hat{u}_i^2 = \delta_0 + \delta_1 x_{i1} + \cdots + \delta_k x_{ik} + v_i$
   - Test statistic: $LM = nR^2_{\hat{u}^2}$ where $R^2_{\hat{u}^2}$ is from this auxiliary regression
   - $LM \sim \chi^2_k$ under $H_0$ (approximately)
   - Rejects if variance depends on any $x$ variables

2. **White Test:**
   - More general - includes squares and cross-products of regressors
   - Auxiliary regression: $\hat{u}_i^2 = \delta_0 + \delta_1 x_{i1} + \delta_2 x_{i1}^2 + \delta_3 x_{i1}x_{i2} + \cdots + v_i$
   - Test statistic: $LM = nR^2_{\hat{u}^2} \sim \chi^2_q$ where $q$ = number of regressors in auxiliary regression
   - Detects heteroskedasticity of unknown form
   - Can also detect functional form misspecification

3. **Interpretation:**
   - Rejecting $H_0$ indicates heteroskedasticity is present
   - Failing to reject doesn't prove homoskedasticity
   - With large $n$, tests may detect economically insignificant heteroskedasticity
   - **Practical approach:** Use robust SEs regardless of test results

**Weighted Least Squares (WLS):**

- **Idea:** Give less weight to observations with higher variance
- Transforms model to satisfy homoskedasticity: $y_i/\sqrt{h_i} = \beta_0/\sqrt{h_i} + \beta_1 x_{i1}/\sqrt{h_i} + \cdots + u_i/\sqrt{h_i}$
- Equivalent to minimizing: $\sum_{i=1}^n w_i (y_i - \beta_0 - \beta_1 x_{i1} - \cdots)^2$ with $w_i = 1/h_i$
- If $h_i$ known: WLS is BLUE (most efficient linear unbiased estimator)
- **Problem:** $h_i$ rarely known in practice

**Feasible GLS (FGLS):**

- **Step 1:** Estimate OLS to get residuals $\hat{u}_i$
- **Step 2:** Model variance function: $\log(\hat{u}_i^2) = \delta_0 + \delta_1 z_{i1} + \cdots + v_i$
  - $z$ variables can be original $x$'s, squares, interactions
- **Step 3:** Get fitted values $\widehat{\log(h_i)}$ and compute $\hat{h}_i = \exp(\widehat{\log(h_i)})$
- **Step 4:** Estimate WLS with weights $w_i = 1/\hat{h}_i$
- **Result:** FGLS estimates are consistent and potentially more efficient than OLS

**FGLS Caveats:**

- Efficiency gain depends on correctly specifying variance function
- Misspecified $h_i$ can give **worse** estimates than OLS
- FGLS standard errors assume correct variance model
- **Best practice:** Compute robust SEs even for FGLS estimates
- If OLS and FGLS differ substantially $\to$ model misspecification

**Linear Probability Model (LPM) and Heteroskedasticity:**

- LPM has heteroskedasticity **by construction:**
  - $\text{Var}(u_i | \mathbf{x}_i) = p_i(1 - p_i)$ where $p_i = E(y_i|\mathbf{x}_i)$
  - Variance is quadratic function of predicted probability
- **Robust inference mandatory:** Always use HC standard errors
- **WLS for LPM:**
  - Known variance form allows FGLS: $w_i = 1/[\hat{p}_i(1-\hat{p}_i)]$
  - But predictions outside [0,1] cause problems
  - Variance model only correct if LPM is true model
- **Practical recommendation:** OLS with robust SEs is most reliable
- **Better alternatives:** Probit or logit models (Chapter 17)

### Python Implementation Patterns

**Heteroskedasticity-robust standard errors:**

- Fit with robust SEs: `results_robust = model.fit(cov_type="HC3")` (HC3 recommended)
- Access robust SEs: `robust_se = results_robust.bse`
- Other options: `cov_type="HC0"`, `"HC1"`, `"HC2"`

**F-tests with robust inference:**

- Define hypotheses: `hypotheses = "x1 = 0, x2 = 0"`
- Robust F-test: `f_test = results_robust.f_test(hypotheses)`

**Breusch-Pagan test:**

- Import: `from statsmodels.stats.diagnostic import het_breuschpagan`
- Test: `bp_test = het_breuschpagan(results.resid, results.model.exog)`

**White test:**

- Import: `from statsmodels.stats.diagnostic import het_white`
- Test: `white_test = het_white(results.resid, results.model.exog)`

**Weighted Least Squares:**

- With known weights: `results_wls = smf.wls(formula="y ~ x1 + x2", weights=weights, data=df).fit()`
- FGLS procedure:
  1. Estimate OLS: `results_ols = smf.ols(...).fit()`
  2. Model variance: Regress `log(resid**2)` on explanatory variables
  3. Compute weights: `weights_fgls = 1 / exp(fitted_values)`
  4. FGLS estimation: `results_fgls = smf.wls(..., weights=weights_fgls).fit()`
  5. Robust SEs: Use `cov_type="HC3"` in WLS fit

### Common Pitfalls

1. **Using usual OLS SEs with heteroskedasticity:**
   - Leads to invalid inference (wrong p-values, confidence intervals)
   - **Fix:** Always use robust SEs in cross-sectional data

2. **Over-interpreting heteroskedasticity tests:**
   - Rejection with large $n$ may indicate trivial heteroskedasticity
   - Non-rejection doesn't guarantee homoskedasticity
   - **Fix:** Use robust SEs regardless of test results

3. **Misspecifying variance function in FGLS:**
   - Can result in less efficient estimates than OLS
   - Standard FGLS SEs invalid if model wrong
   - **Fix:** Use robust SEs even for FGLS; compare OLS and FGLS

4. **Forgetting robust SEs for WLS:**
   - WLS assumes variance model is correct
   - If wrong, usual WLS SEs invalid
   - **Fix:** Report WLS estimates with robust SEs

5. **Using LPM without robust SEs:**
   - LPM always has heteroskedasticity
   - **Fix:** Mandatory robust SEs for LPM

6. **Pursuing efficiency at all costs:**
   - Small efficiency gains may not justify complexity of FGLS
   - Risk of misspecification
   - **Fix:** OLS + robust SEs is simple, robust, and often sufficient

### Decision Framework

**When should you use each approach?**

| Situation | Recommendation | Reasoning |
|-----------|---------------|-----------|
| Default for cross-sectional data | OLS + robust SEs (HC3) | Simple, valid, robust to misspecification |
| Heteroskedasticity confirmed, efficiency important | FGLS + robust SEs | Potential efficiency gains if variance modeled well |
| Known variance structure (rare) | WLS with true weights | Optimal (BLUE) |
| Linear probability model | OLS + robust SEs | Most reliable for LPM |
| Large differences OLS vs FGLS | Investigate model | Suggests misspecification |
| Time series data | Different methods | See Chapters 10-12 |

### Connections

**To previous chapters:**

- **Chapter 3:** Heteroskedasticity violates Gauss-Markov assumption MLR.5
- **Chapter 4:** Invalidates usual t-tests and F-tests
- **Chapter 5:** OLS still consistent, but not asymptotically efficient
- **Chapter 7:** LPM has heteroskedasticity by construction

**To later chapters:**

- **Chapter 9:** Heteroskedasticity tests can signal functional form problems
- **Chapter 12:** Heteroskedasticity in time series contexts
- **Chapter 15:** Robust SEs important for IV estimation
- **Chapter 17:** Probit and logit as alternatives to LPM

### Practical Guidance

**Research workflow:**

1. **Estimate OLS** with usual SEs (for comparison)
2. **Compute robust SEs** (HC3) - always report these
3. **Test for heteroskedasticity** (BP, White) - informative but not decisive
4. **Consider FGLS** if:
   - Efficiency is important (small sample)
   - You can plausibly model variance structure
   - You'll use robust SEs for FGLS too
5. **Compare OLS and FGLS:**
   - Similar coefficients $\to$ heteroskedasticity not severe
   - Very different $\to$ possible misspecification
6. **Report both** OLS-robust and FGLS-robust for transparency

**What to report in papers:**

- Coefficient estimates (OLS or FGLS)
- **Robust standard errors** in parentheses (specify HC0/HC1/HC3)
- Note: "Heteroskedasticity-robust standard errors in parentheses"
- Optionally: Results of BP or White test in footnote
- If using FGLS: Describe variance model specification

**Red flags:**

- OLS and FGLS coefficients differ by more than 1-2 standard errors
- FGLS SEs much larger than OLS SEs (suggests misspecification)
- Many LPM fitted values outside [0, 1] (use probit/logit instead)
- Heteroskedasticity test rejects but robust SEs similar to usual SEs

### Learning Objectives Covered

1. **8.1:** Understand consequences of heteroskedasticity for OLS (unbiased but invalid inference, not BLUE)
2. **8.2:** Explain homoskedasticity assumption and why it matters
3. **8.3:** Compute heteroskedasticity-robust standard errors (HC0, HC1, HC3)
4. **8.4:** Conduct valid t-tests and F-tests using robust SEs
5. **8.5:** Test for heteroskedasticity using Breusch-Pagan and White tests
6. **8.6:** Obtain and interpret p-values from heteroskedasticity tests
7. **8.7:** Explain when and why WLS can improve upon OLS (efficiency gains)
8. **8.8:** Estimate FGLS by modeling variance function
9. **8.9:** Interpret differences between OLS and WLS estimates (misspecification indicator)
10. **8.10:** Understand consequences of misspecifying variance in WLS (inefficiency, invalid SEs)
11. **8.11:** Apply WLS to LPM and understand its shortcomings (predictions outside [0,1], model uncertainty)

### Key Takeaway

Heteroskedasticity is pervasive in cross-sectional economic data but easily addressed through robust inference. **The most important practical lesson: Always use heteroskedasticity-robust standard errors (HC3) for cross-sectional regressions.** This ensures valid inference regardless of whether heteroskedasticity is present. FGLS can improve efficiency but requires careful specification of the variance function and should always be accompanied by robust standard errors. For binary dependent variables, prefer probit or logit models over LPM when possible.
