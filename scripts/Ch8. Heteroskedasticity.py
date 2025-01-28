# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,markdown//md,scripts//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.6
#   kernelspec:
#     display_name: merino
#     language: python
#     name: python3
# ---

# # 8. Heteroskedasticity

# %pip install numpy pandas patsy statsmodels wooldridge -q

import numpy as np
import pandas as pd
import patsy as pt
import statsmodels.api as sm
import statsmodels.formula.api as smf
import wooldridge as wool

# ## 8.1 Heteroskedasticity-Robust Inference
#
# ### Example 8.2: Heteroskedasticity-Robust Inference

# +
gpa3 = wool.data("gpa3")

# define regression model:
reg = smf.ols(
    formula="cumgpa ~ sat + hsperc + tothrs + female + black + white",
    data=gpa3,
    subset=(gpa3["spring"] == 1),
)

# estimate default model (only for spring data):
results_default = reg.fit()

table_default = pd.DataFrame(
    {
        "b": round(results_default.params, 5),
        "se": round(results_default.bse, 5),
        "t": round(results_default.tvalues, 5),
        "pval": round(results_default.pvalues, 5),
    },
)
print(f"table_default: \n{table_default}\n")

# +
# estimate model with White SE (only for spring data):
results_white = reg.fit(cov_type="HC0")

table_white = pd.DataFrame(
    {
        "b": round(results_white.params, 5),
        "se": round(results_white.bse, 5),
        "t": round(results_white.tvalues, 5),
        "pval": round(results_white.pvalues, 5),
    },
)
print(f"table_white: \n{table_white}\n")

# +
# estimate model with refined White SE (only for spring data):
results_refined = reg.fit(cov_type="HC3")

table_refined = pd.DataFrame(
    {
        "b": round(results_refined.params, 5),
        "se": round(results_refined.bse, 5),
        "t": round(results_refined.tvalues, 5),
        "pval": round(results_refined.pvalues, 5),
    },
)
print(f"table_refined: \n{table_refined}\n")

# +
gpa3 = wool.data("gpa3")

# definition of model and hypotheses:
reg = smf.ols(
    formula="cumgpa ~ sat + hsperc + tothrs + female + black + white",
    data=gpa3,
    subset=(gpa3["spring"] == 1),
)
hypotheses = ["black = 0", "white = 0"]

# F-Tests using different variance-covariance formulas:
# ususal VCOV:
results_default = reg.fit()
ftest_default = results_default.f_test(hypotheses)
fstat_default = ftest_default.statistic
fpval_default = ftest_default.pvalue
print(f"fstat_default: {fstat_default}\n")
print(f"fpval_default: {fpval_default}\n")
# -

# refined White VCOV:
results_hc3 = reg.fit(cov_type="HC3")
ftest_hc3 = results_hc3.f_test(hypotheses)
fstat_hc3 = ftest_hc3.statistic
fpval_hc3 = ftest_hc3.pvalue
print(f"fstat_hc3: {fstat_hc3}\n")
print(f"fpval_hc3: {fpval_hc3}\n")

# classical White VCOV:
results_hc0 = reg.fit(cov_type="HC0")
ftest_hc0 = results_hc0.f_test(hypotheses)
fstat_hc0 = ftest_hc0.statistic
fpval_hc0 = ftest_hc0.pvalue
print(f"fstat_hc0: {fstat_hc0}\n")
print(f"fpval_hc0: {fpval_hc0}\n")

# ## 8.2 Heteroskedasticity Tests
#
# ### Example 8.4: Heteroskedasticity in a Housing Price Equation

# +
hprice1 = wool.data("hprice1")

# estimate model:
reg = smf.ols(formula="price ~ lotsize + sqrft + bdrms", data=hprice1)
results = reg.fit()
table_results = pd.DataFrame(
    {
        "b": round(results.params, 4),
        "se": round(results.bse, 4),
        "t": round(results.tvalues, 4),
        "pval": round(results.pvalues, 4),
    },
)
print(f"table_results: \n{table_results}\n")
# -

# automatic BP test (LM version):
y, X = pt.dmatrices(
    "price ~ lotsize + sqrft + bdrms",
    data=hprice1,
    return_type="dataframe",
)
result_bp_lm = sm.stats.diagnostic.het_breuschpagan(results.resid, X)
bp_lm_statistic = result_bp_lm[0]
bp_lm_pval = result_bp_lm[1]
print(f"bp_lm_statistic: {bp_lm_statistic}\n")
print(f"bp_lm_pval: {bp_lm_pval}\n")

# manual BP test (F version):
hprice1["resid_sq"] = results.resid**2
reg_resid = smf.ols(formula="resid_sq ~ lotsize + sqrft + bdrms", data=hprice1)
results_resid = reg_resid.fit()
bp_F_statistic = results_resid.fvalue
bp_F_pval = results_resid.f_pvalue
print(f"bp_F_statistic: {bp_F_statistic}\n")
print(f"bp_F_pval: {bp_F_pval}\n")

# ### Example 8.5: BP and White test in the Log Housing Price Equation

# +
hprice1 = wool.data("hprice1")

# estimate model:
reg = smf.ols(
    formula="np.log(price) ~ np.log(lotsize) + np.log(sqrft) + bdrms",
    data=hprice1,
)
results = reg.fit()

# BP test:
y, X_bp = pt.dmatrices(
    "np.log(price) ~ np.log(lotsize) + np.log(sqrft) + bdrms",
    data=hprice1,
    return_type="dataframe",
)
result_bp = sm.stats.diagnostic.het_breuschpagan(results.resid, X_bp)
bp_statistic = result_bp[0]
bp_pval = result_bp[1]
print(f"bp_statistic: {bp_statistic}\n")
print(f"bp_pval: {bp_pval}\n")
# -

# White test:
X_wh = pd.DataFrame(
    {
        "const": 1,
        "fitted_reg": results.fittedvalues,
        "fitted_reg_sq": results.fittedvalues**2,
    },
)
result_white = sm.stats.diagnostic.het_breuschpagan(results.resid, X_wh)
white_statistic = result_white[0]
white_pval = result_white[1]
print(f"white_statistic: {white_statistic}\n")
print(f"white_pval: {white_pval}\n")

# ## 8.3 Weighted Least Squares
#
# ### Example 8.6: Financial Wealth Equation

# +
k401ksubs = wool.data("401ksubs")

# subsetting data:
k401ksubs_sub = k401ksubs[k401ksubs["fsize"] == 1]

# OLS (only for singles, i.e. 'fsize'==1):
reg_ols = smf.ols(
    formula="nettfa ~ inc + I((age-25)**2) + male + e401k",
    data=k401ksubs_sub,
)
results_ols = reg_ols.fit(cov_type="HC0")

# print regression table:
table_ols = pd.DataFrame(
    {
        "b": round(results_ols.params, 4),
        "se": round(results_ols.bse, 4),
        "t": round(results_ols.tvalues, 4),
        "pval": round(results_ols.pvalues, 4),
    },
)
print(f"table_ols: \n{table_ols}\n")

# +
# WLS:
wls_weight = list(1 / k401ksubs_sub["inc"])
reg_wls = smf.wls(
    formula="nettfa ~ inc + I((age-25)**2) + male + e401k",
    weights=wls_weight,
    data=k401ksubs_sub,
)
results_wls = reg_wls.fit()

# print regression table:
table_wls = pd.DataFrame(
    {
        "b": round(results_wls.params, 4),
        "se": round(results_wls.bse, 4),
        "t": round(results_wls.tvalues, 4),
        "pval": round(results_wls.pvalues, 4),
    },
)
print(f"table_wls: \n{table_wls}\n")

# +
k401ksubs = wool.data("401ksubs")

# subsetting data:
k401ksubs_sub = k401ksubs[k401ksubs["fsize"] == 1]

# WLS:
wls_weight = list(1 / k401ksubs_sub["inc"])
reg_wls = smf.wls(
    formula="nettfa ~ inc + I((age-25)**2) + male + e401k",
    weights=wls_weight,
    data=k401ksubs_sub,
)

# non-robust (default) results:
results_wls = reg_wls.fit()
table_default = pd.DataFrame(
    {
        "b": round(results_wls.params, 4),
        "se": round(results_wls.bse, 4),
        "t": round(results_wls.tvalues, 4),
        "pval": round(results_wls.pvalues, 4),
    },
)
print(f"table_default: \n{table_default}\n")
# -

# robust results (Refined White SE):
results_white = reg_wls.fit(cov_type="HC3")
table_white = pd.DataFrame(
    {
        "b": round(results_white.params, 4),
        "se": round(results_white.bse, 4),
        "t": round(results_white.tvalues, 4),
        "pval": round(results_white.pvalues, 4),
    },
)
print(f"table_white: \n{table_white}\n")

# ### Example 8.7: Demand for Cigarettes

# +
smoke = wool.data("smoke")

# OLS:
reg_ols = smf.ols(
    formula="cigs ~ np.log(income) + np.log(cigpric) +"
    "educ + age + I(age**2) + restaurn",
    data=smoke,
)
results_ols = reg_ols.fit()
table_ols = pd.DataFrame(
    {
        "b": round(results_ols.params, 4),
        "se": round(results_ols.bse, 4),
        "t": round(results_ols.tvalues, 4),
        "pval": round(results_ols.pvalues, 4),
    },
)
print(f"table_ols: \n{table_ols}\n")
# -

# BP test:
y, X = pt.dmatrices(
    "cigs ~ np.log(income) + np.log(cigpric) + educ +age + I(age**2) + restaurn",
    data=smoke,
    return_type="dataframe",
)
result_bp = sm.stats.diagnostic.het_breuschpagan(results_ols.resid, X)
bp_statistic = result_bp[0]
bp_pval = result_bp[1]
print(f"bp_statistic: {bp_statistic}\n")
print(f"bp_pval: {bp_pval}\n")

# FGLS (estimation of the variance function):
smoke["logu2"] = np.log(results_ols.resid**2)
reg_fgls = smf.ols(
    formula="logu2 ~ np.log(income) + np.log(cigpric) +"
    "educ + age + I(age**2) + restaurn",
    data=smoke,
)
results_fgls = reg_fgls.fit()
table_fgls = pd.DataFrame(
    {
        "b": round(results_fgls.params, 4),
        "se": round(results_fgls.bse, 4),
        "t": round(results_fgls.tvalues, 4),
        "pval": round(results_fgls.pvalues, 4),
    },
)
print(f"table_fgls: \n{table_fgls}\n")

# FGLS (WLS):
wls_weight = list(1 / np.exp(results_fgls.fittedvalues))
reg_wls = smf.wls(
    formula="cigs ~ np.log(income) + np.log(cigpric) +"
    "educ + age + I(age**2) + restaurn",
    weights=wls_weight,
    data=smoke,
)
results_wls = reg_wls.fit()
table_wls = pd.DataFrame(
    {
        "b": round(results_wls.params, 4),
        "se": round(results_wls.bse, 4),
        "t": round(results_wls.tvalues, 4),
        "pval": round(results_wls.pvalues, 4),
    },
)
print(f"table_wls: \n{table_wls}\n")
