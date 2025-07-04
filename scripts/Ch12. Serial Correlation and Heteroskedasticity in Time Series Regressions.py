# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,markdown//md,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: merino
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 12. Serial Correlation and Heteroskedasticity in Time Series Regressions
#
# This notebook explores two important complications that can arise when applying OLS to time series data: **serial correlation** (also known as autocorrelation) and **heteroskedasticity** in the error terms.
#
# *   **Serial Correlation:** Occurs when the error terms in different time periods are correlated ($Corr(u_t, u_s) \neq 0$ for $t \neq s$). This violates the OLS assumption of no serial correlation. While OLS coefficient estimates might still be unbiased and consistent under certain conditions (contemporaneous exogeneity), the standard errors calculated by OLS are incorrect (usually biased downwards), leading to invalid t-statistics, p-values, and confidence intervals.
# *   **Heteroskedasticity:** Occurs when the variance of the error term is not constant across time ($Var(u_t)$ depends on $t$). This also leads to incorrect OLS standard errors and invalid inference, although the coefficient estimates may remain unbiased and consistent.
#
# We will cover methods for testing for these issues and discuss strategies for obtaining valid inference, either by using Feasible Generalized Least Squares (FGLS) or by correcting the OLS standard errors.
#
# First, let's install and import the necessary libraries.

# %%
# # %pip install numpy pandas pandas_datareader patsy statsmodels wooldridge -q

# %%
import numpy as np  # noqa
import pandas as pd
import patsy as pt  # Used for creating design matrices easily from formulas
import statsmodels.api as sm  # Provides statistical models and tests
import statsmodels.formula.api as smf  # Convenient formula interface for statsmodels
import wooldridge as wool  # Access to Wooldridge textbook datasets

# %% [markdown]
# ## 12.1 Testing for Serial Correlation of the Error Term
#
# Serial correlation means that the error in one time period provides information about the error in subsequent periods. The simplest and most common form is **Autoregressive order 1 (AR(1))** serial correlation, where the error $u_t$ is related to the previous period's error $u_{t-1}$:
# $$ u_t = \rho u_{t-1} + e_t $$
# where $e_t$ is an uncorrelated error term (white noise), and $\rho$ is the autocorrelation coefficient ($|\rho| < 1$ for stationarity). If $\rho \neq 0$, we have serial correlation.
#
# A common way to test for AR(1) serial correlation is to:
# 1.  Estimate the original model by OLS and obtain the residuals, $\hat{u}_t$.
# 2.  Regress the residuals on their first lag: $\hat{u}_t$ on $\hat{u}_{t-1}$.
# 3.  Perform a t-test on the coefficient of $\hat{u}_{t-1}$. If it is statistically significant, we reject the null hypothesis of no serial correlation ($\rho = 0$).
#
# ### Example 12.2: Testing for AR(1) Serial Correlation in Phillips Curves
#
# We test for AR(1) serial correlation in the residuals of two Phillips curve models estimated using data up to 1996:
# 1.  **Static Phillips Curve:** Inflation (`inf`) regressed on unemployment (`unem`).
# 2.  **Expectations-Augmented Phillips Curve:** Change in inflation (`inf_diff1`) regressed on unemployment (`unem`).

# %%
# Load the Phillips curve data
phillips = wool.data("phillips")
T = len(phillips)

# Define a yearly time series index starting in 1948
date_range = pd.date_range(start="1948", periods=T, freq="YE")
phillips.index = date_range.year

# --- Test for Static Phillips Curve ---

# Define subset of data up to 1996
yt96 = phillips["year"] <= 1996

# 1. Estimate the static Phillips curve model
# Use Q() for 'inf' just in case, although not strictly needed here
reg_s = smf.ols(formula='Q("inf") ~ unem', data=phillips, subset=yt96)
results_s = reg_s.fit()

# 2. Obtain residuals and create lagged residuals
phillips["resid_s"] = results_s.resid
phillips["resid_s_lag1"] = phillips["resid_s"].shift(1)

# 3. Regress residuals on lagged residuals (using the same subset of data, note NaNs are handled)
# The intercept in this regression should be statistically indistinguishable from zero if the original model included one.
reg_test_s = smf.ols(formula="resid_s ~ resid_s_lag1", data=phillips, subset=yt96)
results_test_s = reg_test_s.fit()

# Print the results of the residual regression
print("--- Test for AR(1) Serial Correlation (Static Phillips Curve) ---")
table_s = pd.DataFrame(
    {
        "b": round(results_test_s.params, 4),
        "se": round(results_test_s.bse, 4),
        "t": round(results_test_s.tvalues, 4),
        "pval": round(results_test_s.pvalues, 4),
    },
)
print(f"Regression: resid_s ~ resid_s_lag1 \n{table_s}\n")

# Interpretation (Static Phillips Curve):
# The coefficient on the lagged residual (resid_s_lag1) is 0.5730, and it is highly
# statistically significant (p-value = 0.0000). This provides strong evidence of positive
# AR(1) serial correlation in the errors of the static Phillips curve model.
# OLS standard errors for the original regression are likely invalid.

# %%
# --- Test for Expectations-Augmented Phillips Curve ---
# Reload data or ensure previous modifications don't interfere if running cells independently
phillips = wool.data("phillips")
T = len(phillips)
date_range = pd.date_range(start="1948", periods=T, freq="YE")
phillips.index = date_range.year
yt96 = phillips["year"] <= 1996

# 1. Estimate the expectations-augmented Phillips curve model
# Calculate the change in inflation (first difference)
phillips["inf_diff1"] = phillips["inf"].diff()
reg_ea = smf.ols(formula="inf_diff1 ~ unem", data=phillips, subset=yt96)
results_ea = reg_ea.fit()

# 2. Obtain residuals and create lagged residuals
phillips["resid_ea"] = results_ea.resid
phillips["resid_ea_lag1"] = phillips["resid_ea"].shift(1)

# 3. Regress residuals on lagged residuals
reg_test_ea = smf.ols(formula="resid_ea ~ resid_ea_lag1", data=phillips, subset=yt96)
results_test_ea = reg_test_ea.fit()

# Print the results of the residual regression
print(
    "--- Test for AR(1) Serial Correlation (Expectations-Augmented Phillips Curve) ---",
)
table_ea = pd.DataFrame(
    {
        "b": round(results_test_ea.params, 4),
        "se": round(results_test_ea.bse, 4),
        "t": round(results_test_ea.tvalues, 4),
        "pval": round(results_test_ea.pvalues, 4),
    },
)
print(f"Regression: resid_ea ~ resid_ea_lag1 \n{table_ea}\n")

# Interpretation (Expectations-Augmented Phillips Curve):
# The coefficient on the lagged residual (resid_ea_lag1) is -0.0356, which is much smaller
# than in the static model and is not statistically significant at conventional levels (p-value = 0.7752).
# This suggests that the expectations-augmented model (using the change in inflation)
# has largely eliminated the AR(1) serial correlation found in the static model.

# %% [markdown]
# ### Testing for Higher-Order Serial Correlation: Breusch-Godfrey Test
#
# Serial correlation might extend beyond just one lag (e.g., AR(q) process). The **Breusch-Godfrey (BG) test** is a general test for AR(q) serial correlation.
# The null hypothesis is $H_0: \rho_1 = 0, \rho_2 = 0, ..., \rho_q = 0$.
# The test involves:
# 1.  Estimate the original model by OLS and get residuals $\hat{u}_t$.
# 2.  Regress $\hat{u}_t$ on the original regressors ($x_{t1}, ..., x_{tk}$) **and** the lagged residuals ($\hat{u}_{t-1}, ..., \hat{u}_{t-q}$).
# 3.  Compute the F-statistic for the joint significance of the coefficients on the lagged residuals. A significant F-statistic indicates rejection of the null hypothesis (i.e., presence of serial correlation up to order q).
#
# Including the original regressors in the auxiliary regression makes the test robust even if some regressors are lagged dependent variables.
#
# ### Example 12.4: Testing for AR(3) Serial Correlation in Barium Imports Model
#
# We test for serial correlation up to order 3 (AR(3)) in the errors of the Barium imports model from Chapter 10.

# %%
# Load Barium data
barium = wool.data("barium")
T = len(barium)

# Define monthly time series index
barium.index = pd.date_range(start="1978-02", periods=T, freq="ME")

# 1. Estimate the original model
reg = smf.ols(
    formula="np.log(chnimp) ~ np.log(chempi) + np.log(gas) +"
    "np.log(rtwex) + befile6 + affile6 + afdec6",
    data=barium,
)
results = reg.fit()

# --- Breusch-Godfrey Test using statsmodels built-in function ---
# This is the recommended way. It automatically performs steps 2 & 3.
# We test up to nlags=3.
print("--- Breusch-Godfrey Test (Automated) ---")
# Returns: LM statistic, LM p-value, F statistic, F p-value
bg_result = sm.stats.diagnostic.acorr_breusch_godfrey(results, nlags=3)
fstat_auto = bg_result[2]
fpval_auto = bg_result[3]
print(f"BG Test F-statistic (lags=3): {fstat_auto:.4f}")
print(f"BG Test F p-value: {fpval_auto:.4f}\n")

# Interpretation (Automated BG Test):
# The F-statistic is large (5.1247) and the p-value is very small (0.0023).
# We strongly reject the null hypothesis of no serial correlation up to order 3.
# There is significant evidence of serial correlation in the model's errors.

# %%
# --- Breusch-Godfrey Test (Manual / "Pedestrian" Calculation) ---
# This demonstrates the steps involved.

# 2. Get residuals and create lags
barium["resid"] = results.resid
barium["resid_lag1"] = barium["resid"].shift(1)
barium["resid_lag2"] = barium["resid"].shift(2)
barium["resid_lag3"] = barium["resid"].shift(3)

# 3. Run auxiliary regression: residuals on original regressors and lagged residuals
# Note: We must include ALL original regressors from the first stage.
reg_manual_bg = smf.ols(
    formula="resid ~ resid_lag1 + resid_lag2 + resid_lag3 +"
    "np.log(chempi) + np.log(gas) + np.log(rtwex) +"
    "befile6 + affile6 + afdec6",
    data=barium,  # statsmodels handles NaNs introduced by lags
)
results_manual_bg = reg_manual_bg.fit()

# 4. Perform F-test for joint significance of lagged residuals
hypotheses = ["resid_lag1 = 0", "resid_lag2 = 0", "resid_lag3 = 0"]
ftest_manual = results_manual_bg.f_test(hypotheses)
fstat_manual = ftest_manual.statistic  # Extract F-stat value
fpval_manual = ftest_manual.pvalue
print("--- Breusch-Godfrey Test (Manual) ---")
print(f"Manual BG F-statistic (lags=3): {fstat_manual:.4f}")
print(f"Manual BG F p-value: {fpval_manual:.4f}\n")

# Interpretation (Manual BG Test):
# The manually calculated F-statistic and p-value match the automated results,
# confirming significant serial correlation up to order 3.

# %% [markdown]
# ### Durbin-Watson Test
#
# The Durbin-Watson (DW) statistic is an older test primarily designed for AR(1) serial correlation ($u_t = \rho u_{t-1} + e_t$).
# $$ DW = \frac{\sum_{t=2}^T (\hat{u}_t - \hat{u}_{t-1})^2}{\sum_{t=1}^T \hat{u}_t^2} \approx 2(1 - \hat{\rho}) $$
# where $\hat{\rho}$ is the estimated AR(1) coefficient from the residual regression.
# *   If $\hat{\rho} \approx 0$ (no serial correlation), $DW \approx 2$.
# *   If $\hat{\rho} \approx 1$ (strong positive serial correlation), $DW \approx 0$.
# *   If $\hat{\rho} \approx -1$ (strong negative serial correlation), $DW \approx 4$.
#
# **Limitations:**
# *   Requires the assumption that regressors are strictly exogenous (stronger than needed for BG test). Not valid if lagged dependent variables are included.
# *   Has "inconclusive regions" in its critical value tables.
# *   Primarily tests for AR(1).
#
# It's generally recommended to use the Breusch-Godfrey test, but `statsmodels` provides the DW statistic easily.

# %%
# --- Durbin-Watson Test for Phillips Curve Models ---
# Reload data or ensure models are estimated if running cells independently
phillips = wool.data("phillips")
T = len(phillips)
date_range = pd.date_range(start="1948", periods=T, freq="YE")
phillips.index = date_range.year
yt96 = phillips["year"] <= 1996
phillips["inf_diff1"] = phillips["inf"].diff()
# Re-estimate models if necessary
reg_s = smf.ols(formula='Q("inf") ~ unem', data=phillips, subset=yt96)
reg_ea = smf.ols(formula="inf_diff1 ~ unem", data=phillips, subset=yt96)
results_s = reg_s.fit()
results_ea = reg_ea.fit()

# Calculate DW statistics from the residuals of the original models
DW_s = sm.stats.stattools.durbin_watson(results_s.resid)
DW_ea = sm.stats.stattools.durbin_watson(results_ea.resid)

print("--- Durbin-Watson Statistics ---")
print(f"DW statistic (Static Phillips Curve): {DW_s:.4f}")
print(f"DW statistic (Expectations-Augmented Phillips Curve): {DW_ea:.4f}\n")

# Interpretation:
# - Static model: DW = 0.8027. This is far below 2, indicating strong positive serial correlation, consistent with our earlier AR(1) test.
# - Expectations-Augmented model: DW = 1.7696. This is much closer to 2, suggesting little evidence of AR(1) serial correlation, also consistent with our earlier test.
# (Formal conclusion requires comparing these to critical values from DW tables, considering sample size and number of regressors).

# %% [markdown]
# ## 12.2 FGLS Estimation
#
# When serial correlation is present, OLS is inefficient (i.e., not the Best Linear Unbiased Estimator), and its standard errors are invalid. **Feasible Generalized Least Squares (FGLS)** is a method to obtain estimators that are asymptotically more efficient than OLS by transforming the model to eliminate the serial correlation.
#
# Common FGLS procedures for AR(1) errors include **Cochrane-Orcutt (C-O)** and **Prais-Winsten (P-W)**. They involve:
# 1.  Estimate the original model by OLS and get residuals $\hat{u}_t$.
# 2.  Estimate the AR(1) coefficient $\rho$ by regressing $\hat{u}_t$ on $\hat{u}_{t-1}$ (getting $\hat{\rho}$).
# 3.  Transform the variables: $y_t^* = y_t - \hat{\rho} y_{t-1}$, $x_{tj}^* = x_{tj} - \hat{\rho} x_{t,j-1}$. (Prais-Winsten applies a special transformation to the first observation, while C-O drops it).
# 4.  Estimate the transformed model $y_t^* = \beta_0(1-\hat{\rho}) + \beta_1 x_{t1}^* + ... + \beta_k x_{tk}^* + \text{error}$ by OLS.
# 5.  (Cochrane-Orcutt often iterates steps 1-4 until $\hat{\rho}$ converges).
#
# `statsmodels` provides `GLSAR` (Generalized Least Squares with Autoregressive error) which can implement these procedures.
#
# ### Example 12.5: Cochrane-Orcutt Estimation for Barium Imports
#
# We apply Cochrane-Orcutt FGLS estimation to the Barium imports model, where we previously found significant serial correlation.

# %%
# Load Barium data
barium = wool.data("barium")
T = len(barium)
barium.index = pd.date_range(start="1978-02", periods=T, freq="ME")

# --- Cochrane-Orcutt Estimation using GLSAR ---

# 1. Define the model using patsy to get the design matrices (y vector and X matrix)
# This is required for the GLSAR class interface.
y, X = pt.dmatrices(
    "np.log(chnimp) ~ np.log(chempi) + np.log(gas) +"
    "np.log(rtwex) + befile6 + affile6 + afdec6",
    data=barium,
    return_type="dataframe",  # Get pandas DataFrames
)

# 2. Initialize the GLSAR model assuming AR(p) errors. Default is AR(1).
# We don't specify rho initially; it will be estimated.
reg_glsar = sm.GLSAR(y, X)

# 3. Perform iterative Cochrane-Orcutt estimation.
# maxiter specifies the maximum number of iterations for rho to converge.
# GLSAR's iterative_fit implements Cochrane-Orcutt (dropping first obs).
# Use fit for Prais-Winsten (keeps first obs with transformation).
print("--- Cochrane-Orcutt Estimation Results ---")
CORC_results = reg_glsar.iterative_fit(maxiter=100)

# 4. Display results: estimated rho and FGLS coefficients/standard errors
print(
    f"Estimated AR(1) coefficient (rho): {reg_glsar.rho[0]:.4f}\n",
)  # rho is estimated during iteration
table_corc = pd.DataFrame(
    {
        "b_CORC": round(CORC_results.params, 4),
        "se_CORC": round(CORC_results.bse, 4),  # These are the FGLS standard errors
    },
)
# Add t-stats and p-values manually if needed:
table_corc["t_CORC"] = round(CORC_results.tvalues, 4)
table_corc["pval_CORC"] = round(CORC_results.pvalues, 4)
print(f"Cochrane-Orcutt FGLS Estimates:\n{table_corc}\n")

# Interpretation:
# - The estimated AR(1) coefficient rho is 0.2959, indicating positive serial correlation, consistent with the BG test.
# - The table shows the FGLS coefficient estimates (b_CORC) and their standard errors (se_CORC).
#   These standard errors are asymptotically valid, unlike the original OLS standard errors.
# - Comparing FGLS results to the original OLS results (not shown here) would reveal potentially different standard errors and significance levels for the coefficients.
#   The coefficient estimates themselves might also change slightly.

# %% [markdown]
# ## 12.3 Serial Correlation-Robust Inference with OLS
#
# An alternative to FGLS is to stick with the OLS coefficient estimates (which are consistent under weaker assumptions than needed for FGLS efficiency) but compute **robust standard errors** that account for serial correlation (and potentially heteroskedasticity).
#
# These are often called **HAC (Heteroskedasticity and Autocorrelation Consistent)** standard errors, with the **Newey-West** estimator being the most common. This approach corrects the standard errors, t-statistics, and p-values after OLS estimation.
#
# A key choice is the **maximum lag (`maxlags`)** to include when estimating the variance-covariance matrix of the OLS estimator. This determines how many lags of the autocorrelation structure are accounted for. Rules of thumb exist (e.g., related to $T^{1/4}$), or it can be chosen based on where the sample autocorrelation function seems to die out.
#
# ### Example 12.1: The Puerto Rican Minimum Wage (Revisited)
#
# We estimate the effect of the minimum wage coverage (`mincov`) on the employment rate (`prepop`) in Puerto Rico, controlling for GNP variables and a time trend. We compare standard OLS inference with HAC inference.

# %%
# Load Puerto Rican minimum wage data
prminwge = wool.data("prminwge")
T = len(prminwge)

# Create time trend and set yearly index
prminwge["time"] = prminwge["year"] - 1949  # time = 1 (1950), ..., T
prminwge.index = pd.date_range(start="1950", periods=T, freq="YE").year

# Define the OLS regression model
reg = smf.ols(
    formula="np.log(prepop) ~ np.log(mincov) + np.log(prgnp) + np.log(usgnp) + time",
    data=prminwge,
)

# --- OLS Results with Standard (Non-Robust) SEs ---
results_regu = reg.fit()
print("--- OLS Results with Standard Standard Errors ---")
table_regu = pd.DataFrame(
    {
        "b": round(results_regu.params, 4),
        "se": round(results_regu.bse, 4),
        "t": round(results_regu.tvalues, 4),
        "pval": round(results_regu.pvalues, 4),
    },
)
print(f"Standard OLS Estimates:\n{table_regu}\n")

# Interpretation (Standard OLS):
# The coefficient on log(mincov) is -0.2123 and highly significant (p = 0.0000),
# suggesting a negative impact of minimum wage coverage on employment.
# However, if serial correlation is present, these standard errors and p-values might be unreliable.

# %%
# --- OLS Results with HAC (Newey-West) Standard Errors ---
# Use the same fitted OLS model object but specify the covariance type.
# cov_type='HAC' requests Heteroskedasticity and Autocorrelation Consistent SEs.
# cov_kwds={'maxlags': 2} specifies the maximum lag for the Newey-West estimator.
# The choice of maxlags can influence the results. Here, 2 is used following the textbook example.
results_hac = reg.fit(cov_type="HAC", cov_kwds={"maxlags": 2})

print("--- OLS Results with HAC (Newey-West) Standard Errors (maxlags=2) ---")
# Note: The coefficients 'b' are identical to standard OLS. Only SEs, t-stats, p-values change.
table_hac = pd.DataFrame(
    {
        "b": round(results_hac.params, 4),
        "se": round(results_hac.bse, 4),  # HAC Standard Errors
        "t": round(results_hac.tvalues, 4),  # Robust t-statistics
        "pval": round(results_hac.pvalues, 4),  # Robust p-values
    },
)
print(f"OLS Estimates with HAC SEs:\n{table_hac}\n")

# Interpretation (HAC OLS):
# - The coefficient estimates are identical to standard OLS, as expected.
# - The HAC standard error for log(mincov) is 0.0426, which is slightly larger than the standard OLS SE (0.0402).
# - Consequently, the robust t-statistic (-4.9821) is smaller in magnitude, and the robust p-value (0.0000) remains very small.
# - The evidence for the minimum wage effect remains strong after accounting
#   for potential serial correlation and heteroskedasticity. This highlights the importance of robust inference.

# %% [markdown]
# ## 12.4 Autoregressive Conditional Heteroskedasticity (ARCH)
#
# **Autoregressive Conditional Heteroskedasticity (ARCH)** is a specific model for time-varying volatility often observed in financial time series. It assumes the variance of the error term at time $t$, *conditional* on past information, depends on the magnitude of past error terms.
#
# The **ARCH(1)** model specifies the conditional variance as:
# $$ Var(u_t | u_{t-1}, u_{t-2}, ...) = E(u_t^2 | u_{t-1}, u_{t-2}, ...) = \alpha_0 + \alpha_1 u_{t-1}^2 $$
# where $\alpha_0 > 0$ and $\alpha_1 \ge 0$. If $\alpha_1 > 0$, the variance is higher following periods with large (positive or negative) errors, capturing volatility clustering.
#
# **Testing for ARCH(1) effects:**
# 1.  Estimate the mean equation (the original model) by OLS and obtain residuals $\hat{u}_t$.
# 2.  Square the residuals: $\hat{u}_t^2$.
# 3.  Regress the squared residuals on their first lag: $\hat{u}_t^2$ on $\hat{u}_{t-1}^2$.
# 4.  Test the significance of the coefficient on $\hat{u}_{t-1}^2$. A significant positive coefficient suggests the presence of ARCH(1) effects. (This can be extended to test for ARCH(q) by including more lags).
#
# If ARCH is present, OLS standard errors are invalid (even if there's no serial correlation). HAC standard errors or specific ARCH/GARCH model estimation techniques should be used.
#
# ### Example 12.9: ARCH in Stock Returns
#
# We test for ARCH(1) effects in the daily NYSE stock returns data, using a simple AR(1) model for the mean return.

# %%
# Load NYSE daily returns data
nyse = wool.data("nyse")
nyse["ret"] = nyse["return"]  # Rename for convenience
nyse["ret_lag1"] = nyse["ret"].shift(1)

# 1. Estimate the mean equation (AR(1) for returns)
reg_mean = smf.ols(formula="ret ~ ret_lag1", data=nyse)
results_mean = reg_mean.fit()

# 2. Obtain residuals and square them
nyse["resid"] = results_mean.resid
nyse["resid_sq"] = nyse["resid"] ** 2

# Create lagged squared residuals
nyse["resid_sq_lag1"] = nyse["resid_sq"].shift(1)

# 3. Regress squared residuals on lagged squared residuals
# resid_sq_t = alpha_0 + alpha_1 * resid_sq_{t-1} + error_t
ARCH_test_reg = smf.ols(formula="resid_sq ~ resid_sq_lag1", data=nyse)
results_ARCH_test = ARCH_test_reg.fit()

# 4. Examine the coefficient on the lagged squared residual
print("--- Test for ARCH(1) Effects in NYSE Returns ---")
table_arch = pd.DataFrame(
    {
        "b": round(results_ARCH_test.params, 4),
        "se": round(results_ARCH_test.bse, 4),
        "t": round(results_ARCH_test.tvalues, 4),
        "pval": round(results_ARCH_test.pvalues, 4),
    },
)
print(f"Regression: resid_sq ~ resid_sq_lag1\n{table_arch}\n")

# Interpretation (ARCH Test):
# - The coefficient on the lagged squared residual (resid_sq_lag1) is 0.3371.
# - This coefficient is positive and highly statistically significant (p-value = 0.0).
# - This provides strong evidence for ARCH(1) effects in the daily NYSE returns.
#   The volatility (variance of the error) in one day is positively correlated with the squared error from the previous day.
# - Standard OLS inference for the mean equation (ret ~ ret_lag1) would be invalid due to this conditional heteroskedasticity.
#   HAC standard errors or estimation of a GARCH model would be more appropriate.

# %% [markdown]
# This notebook demonstrated how to test for serial correlation (AR(1), AR(q) using BG test, DW test) and ARCH effects in time series regressions. It also covered two approaches to handle serial correlation: FGLS (Cochrane-Orcutt) for efficiency and OLS with HAC (Newey-West) standard errors for robust inference. Understanding and addressing these issues is crucial for reliable time series analysis.
