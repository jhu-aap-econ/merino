# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,markdown//md,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: merino
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 8. Heteroskedasticity
#
# This notebook explores the issue of **heteroskedasticity** in the context of linear regression models estimated using Ordinary Least Squares (OLS). Heteroskedasticity occurs when the variance of the error term, conditional on the explanatory variables, is not constant across observations. This violates one of the Gauss-Markov assumptions required for OLS to be the Best Linear Unbiased Estimator (BLUE).
#
# **Consequences of Heteroskedasticity:**
#
# Under assumptions MLR.1-MLR.4 (linearity, random sampling, no perfect collinearity, and zero conditional mean), but with heteroskedasticity (violation of MLR.5):
#
# 1.  **Consistency and Unbiasedness:** OLS coefficient estimates $\hat{\beta}_j$ remain **unbiased** and **consistent**. That is, $E(\hat{\beta}_j) = \beta_j$ and $\hat{\beta}_j \xrightarrow{p} \beta_j$ as $n \to \infty$. The point estimates themselves are not affected by heteroskedasticity.
#
# 2.  **Invalid Inference:** The usual OLS standard errors, $\widehat{\text{SE}}(\hat{\beta}_j)$, are **biased and inconsistent** estimates of the true standard errors, even in large samples. Consequently, the usual t-statistics, F-statistics, confidence intervals, and p-values are **invalid**. They can be either too small or too large, leading to incorrect inference (e.g., rejecting true null hypotheses too often or failing to reject false null hypotheses).
#
# 3.  **Loss of Efficiency:** OLS is no longer BLUE (Best Linear Unbiased Estimator). There exist other linear unbiased estimators (e.g., Weighted Least Squares with correct weights) that have smaller variances than OLS. However, OLS remains consistent, which is often sufficient for large samples.
#
# **Solutions:** Use heteroskedasticity-robust standard errors (e.g., White/HC standard errors) for valid inference with OLS estimates, or use Weighted Least Squares (WLS) if the form of heteroskedasticity can be modeled.
#
# We will cover:
# *   How to obtain valid inference in the presence of heteroskedasticity using **robust standard errors**.
# *   Methods for **testing** whether heteroskedasticity is present.
# *   **Weighted Least Squares (WLS)** as an alternative estimation method that can be more efficient than OLS when the form of heteroskedasticity is known or can be reasonably estimated.
#
# First, let's install and import the necessary libraries.

# %%
# # %pip install numpy pandas patsy statsmodels wooldridge -q

# %%
import numpy as np
import pandas as pd
import patsy as pt  # Used for creating design matrices easily from formulas
import statsmodels.api as sm  # Provides statistical models and tests
import statsmodels.formula.api as smf  # Convenient formula interface for statsmodels
import wooldridge as wool  # Access to Wooldridge textbook datasets
from IPython.display import display

# %% [markdown]
# ## 8.1 Heteroskedasticity-Robust Inference
#
# Even if heteroskedasticity is present, we can still use OLS coefficient estimates but compute different standard errors that are robust to the presence of heteroskedasticity (of unknown form). These are often called **White standard errors** or **heteroskedasticity-consistent (HC)** standard errors.
#
# Using robust standard errors allows for valid t-tests, F-tests, and confidence intervals based on the OLS estimates, even if the error variance is not constant. `statsmodels` provides several versions of robust standard errors (HC0, HC1, HC2, HC3), which differ mainly in their finite-sample adjustments. HC0 is the original White estimator, while HC1, HC2, and HC3 apply different corrections, often performing better in smaller samples (HC3 is often recommended).
#
# ### Example 8.2: Heteroskedasticity-Robust Inference for GPA Equation
#
# We estimate a model for college cumulative GPA (`cumgpa`) using data for students observed in the spring semester (`spring == 1`). We compare the standard OLS results with results using robust standard errors.
#
# ```
# # Load the GPA data
# gpa3 = wool.data("gpa3")
#
# # Define the regression model using statsmodels formula API
# # We predict cumgpa based on SAT score, high school percentile, total credit hours,
# # gender, and race dummies.
# # We only use data for the spring semester using the 'subset' argument.
# reg = smf.ols(
#     formula="cumgpa ~ sat + hsperc + tothrs + female + black + white",
#     data=gpa3,
#     subset=(gpa3["spring"] == 1),  # Use only spring observations
# )
#
# # --- Estimate with Default (Homoskedasticity-Assumed) Standard Errors ---
# results_default = reg.fit()
#
# # Display the results in a table
# # OLS Results with Default Standard Errors
# table_default = pd.DataFrame(
#     {
#         "b": round(results_default.params, 5),  # OLS Coefficients
#         "se": round(results_default.bse, 5),  # Default Standard Errors
#         "t": round(results_default.tvalues, 5),  # t-statistics based on default SEs
#         "pval": round(results_default.pvalues, 5),  # p-values based on default SEs
#     },
# )
# display(table_default)
#
# # Interpretation (Default): Based on these standard errors, variables like sat, hsperc,
# # tothrs, female, and black appear statistically significant (p < 0.05).
# # However, if heteroskedasticity is present, these SEs and p-values are unreliable.
#
# # --- Estimate with White's Original Robust Standard Errors (HC0) ---
# # We fit the same model, but specify cov_type='HC0' to get robust SEs.
# results_white = reg.fit(cov_type="HC0")
#
# # Display the results
# # OLS Results with Robust (HC0) Standard Errors
# table_white = pd.DataFrame(
#     {
#         "b": round(results_white.params, 5),  # OLS Coefficients (same as default)
#         "se": round(results_white.bse, 5),  # Robust (HC0) Standard Errors
#         "t": round(results_white.tvalues, 5),  # Robust t-statistics
#         "pval": round(results_white.pvalues, 5),  # Robust p-values
#     },
# )
# display(table_white)
#
# # Interpretation (HC0): The coefficient estimates 'b' are identical to the default OLS run.
# # However, the standard errors 'se' have changed for most variables compared to the default.
# # For example, the SE for 'tothrs' increased from 0.00104 to 0.00121, reducing its t-statistic
# # and increasing its p-value (though still significant). The SE for 'black' decreased slightly.
# # The conclusions about significance might change depending on the variable and significance level.
#
# # --- Estimate with Refined Robust Standard Errors (HC3) ---
# # HC3 applies a different small-sample correction, often preferred over HC0.
# results_refined = reg.fit(cov_type="HC3")
#
# # Display the results
# # OLS Results with Robust (HC3) Standard Errors
# table_refined = pd.DataFrame(
#     {
#         "b": round(results_refined.params, 5),  # OLS Coefficients (same as default)
#         "se": round(results_refined.bse, 5),  # Robust (HC3) Standard Errors
#         "t": round(results_refined.tvalues, 5),  # Robust t-statistics
#         "pval": round(results_refined.pvalues, 5),  # Robust p-values
#     },
# )
# display(table_refined)
#
# # Interpretation (HC3): The HC3 robust standard errors are slightly different from the HC0 SEs
# # (e.g., SE for 'tothrs' is 0.00123 with HC3 vs 0.00121 with HC0). In this specific case,
# # the differences between HC0 and HC3 are minor and don't change the conclusions about
# # statistical significance compared to HC0. Using robust standard errors confirms that
# # sat, hsperc, tothrs, female, and black have statistically significant effects on cumgpa
# # in this sample, even if heteroskedasticity is present.
# ```
#
# Robust standard errors can also be used for hypothesis tests involving multiple restrictions, such as F-tests. We test the joint significance of the race dummies (`black` and `white`), comparing the standard F-test (assuming homoskedasticity) with robust F-tests.

# %%
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

# %% [markdown]
# ## 8.2 Heteroskedasticity Tests
#
# While robust inference provides a way to proceed despite heteroskedasticity, it's often useful to formally test for its presence. Tests can help understand the data better and decide whether alternative estimation methods like WLS might be beneficial (for efficiency).
#
# ### Breusch-Pagan Test
#
# The **Breusch-Pagan (BP) test** checks if the error variance is systematically related to the explanatory variables.
# *   $H_0$: Homoskedasticity ($Var(u|X) = \sigma^2$, constant variance)
# *   $H_1$: Heteroskedasticity ($Var(u|X)$ depends on $X$)
#
# The test involves:
# 1.  Estimate the original model by OLS and obtain the squared residuals, $\hat{u}^2$.
# 2.  Regress the squared residuals on the original explanatory variables: $\hat{u}^2 = \delta_0 + \delta_1 x_1 + ... + \delta_k x_k + \text{error}$.
# 3.  Test the joint significance of $\delta_1, ..., \delta_k$. If they are jointly significant (using an F-test or LM test), we reject $H_0$ and conclude heteroskedasticity is present.
#
# ### Example 8.4: Heteroskedasticity in a Housing Price Equation (Levels)
#
# We test for heteroskedasticity in a model explaining house prices (`price`) using lot size (`lotsize`), square footage (`sqrft`), and number of bedrooms (`bdrms`).

# %%
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

# %%
# --- Breusch-Pagan Test (LM version) using statsmodels function ---
# We need the residuals from the original model and the design matrix (X).
# patsy.dmatrices helps create the X matrix easily from the formula.
y, X = pt.dmatrices(
    "price ~ lotsize + sqrft + bdrms",  # Formula defines X
    data=hprice1,
    return_type="dataframe",
)
# Perform Breusch-Pagan test for heteroskedasticity
# H₀: Var(u|X) = σ² (homoskedasticity)
# H₁: Var(u|X) depends on X (heteroskedasticity)
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
            f"REJECT H₀ at {significance_level:.0%} level"
            if bp_lm_pvalue < significance_level
            else f"FAIL TO REJECT H₀ at {significance_level:.0%} level",
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

# %%
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

# %% [markdown]
# ### White Test
#
# The **White test** is a more general test for heteroskedasticity that doesn't assume a specific linear relationship between the variance and the predictors. It tests whether the variance depends on any combination of the levels, squares, and cross-products of the original regressors.
#
# A simplified version (often used in practice) regresses the squared residuals $\hat{u}^2$ on the OLS fitted values $\hat{y}$ and squared fitted values $\hat{y}^2$:
# $$ \hat{u}^2 = \delta_0 + \delta_1 \hat{y} + \delta_2 \hat{y}^2 + \text{error} $$
# An F-test (or LM test) for $H_0: \delta_1 = 0, \delta_2 = 0$ is performed.
#
# ### Example 8.5: BP and White test in the Log Housing Price Equation
#
# Often, taking the logarithm of the dependent variable can mitigate heteroskedasticity. We now test the log-log housing price model.

# %%
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

# %%
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

# %% [markdown]
# ## 8.3 Weighted Least Squares (WLS)
#
# If heteroskedasticity is detected and we have an idea about its form, we can use **Weighted Least Squares (WLS)**. WLS is a transformation of the original model that yields estimators that are BLUE (efficient) under heteroskedasticity, provided the variance structure is correctly specified.
#
# The idea is to divide the original equation by the standard deviation of the error term, $\sqrt{h(X)}$, where $h(X) = Var(u|X)$. This gives more weight to observations with smaller error variance and less weight to observations with larger error variance.
#
# In practice, $h(X)$ is unknown, so we use an estimate $\hat{h}(X)$. This leads to **Feasible Generalized Least Squares (FGLS)**. A common approach is to specify a model for the variance function, estimate it, and use the predicted variances to construct the weights $w = 1/\hat{h}(X)$.
#
# ### Example 8.6: Financial Wealth Equation (WLS with Assumed Weights)
#
# We model net total financial assets (`nettfa`) for single-person households (`fsize == 1`) as a function of income (`inc`), age (`age`), gender (`male`), and 401k eligibility (`e401k`). It's plausible that the variance of `nettfa` increases with income. We assume $Var(u|inc, age, ...) = \sigma^2 inc$. Thus, the standard deviation is $\sigma \sqrt{inc}$, and the appropriate weight for WLS is $w = 1/inc$.

# %%
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

# %%
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

# %% [markdown]
# What if our assumed variance function ($Var = \sigma^2 inc$) is wrong? The WLS estimator will still be consistent (under standard assumptions) but its standard errors might be incorrect, and it might not be efficient. We can compute robust standard errors *for the WLS estimator* to get valid inference even if the weights are misspecified.

# %%
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

# %%
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

# %% [markdown]
# ### Example 8.7: Demand for Cigarettes (FGLS with Estimated Weights)
#
# Here, we don't assume the form of heteroskedasticity beforehand. Instead, we estimate it using **Feasible GLS (FGLS)**.
# 1.  Estimate the original model (cigarette demand) by OLS.
# 2.  Obtain the residuals $\hat{u}$ and square them.
# 3.  Model the log of squared residuals as a function of the predictors: $\log(\hat{u}^2) = \delta_0 + \delta_1 x_1 + ...$. This models the variance function $h(X)$.
# 4.  Obtain the fitted values from this regression, $\widehat{\log(u^2)}$. Exponentiate to get estimates of the variance: $\hat{h} = \exp(\widehat{\log(u^2)})$.
# 5.  Use weights $w = 1/\hat{h}$ in a WLS estimation of the original model.

# %%
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

# %%
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

# %%
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

# %%
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

# %% [markdown]
# This notebook covered the detection of heteroskedasticity (Breusch-Pagan, White tests), inference robust to heteroskedasticity (White/HC standard errors), and estimation via Weighted Least Squares (WLS/FGLS) to potentially gain efficiency when heteroskedasticity is present. Choosing between OLS with robust SEs and WLS/FGLS often depends on whether efficiency gains are a primary concern and how confident one is in specifying the variance function for WLS/FGLS.
