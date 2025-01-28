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

# # 9. Specification and Data Issues

# %pip install matplotlib numpy pandas statsmodels wooldridge scipy -q

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.outliers_influence as smo
import wooldridge as wool
from scipy import stats

# ## 9.1 Functional Form Misspecification
#
# ### Example 9.2: Housing Price Equation

# +
hprice1 = wool.data("hprice1")

# original OLS:
reg = smf.ols(formula="price ~ lotsize + sqrft + bdrms", data=hprice1)
results = reg.fit()

# regression for RESET test:
hprice1["fitted_sq"] = results.fittedvalues**2
hprice1["fitted_cub"] = results.fittedvalues**3
reg_reset = smf.ols(
    formula="price ~ lotsize + sqrft + bdrms + fitted_sq + fitted_cub",
    data=hprice1,
)
results_reset = reg_reset.fit()

# print regression table:
table = pd.DataFrame(
    {
        "b": round(results_reset.params, 4),
        "se": round(results_reset.bse, 4),
        "t": round(results_reset.tvalues, 4),
        "pval": round(results_reset.pvalues, 4),
    },
)
print(f"table: \n{table}\n")

# +
# RESET test (H0: all coeffs including "fitted" are=0):
hypotheses = ["fitted_sq = 0", "fitted_cub = 0"]
ftest_man = results_reset.f_test(hypotheses)
fstat_man = ftest_man.statistic
fpval_man = ftest_man.pvalue

print(f"fstat_man: {fstat_man}\n")
print(f"fpval_man: {fpval_man}\n")

# +
hprice1 = wool.data("hprice1")

# original linear regression:
reg = smf.ols(formula="price ~ lotsize + sqrft + bdrms", data=hprice1)
results = reg.fit()

# automated RESET test:
reset_output = smo.reset_ramsey(res=results, degree=3)
fstat_auto = reset_output.statistic
fpval_auto = reset_output.pvalue

print(f"fstat_auto: {fstat_auto}\n")
print(f"fpval_auto: {fpval_auto}\n")

# +
hprice1 = wool.data("hprice1")

# two alternative models:
reg1 = smf.ols(formula="price ~ lotsize + sqrft + bdrms", data=hprice1)
results1 = reg1.fit()

reg2 = smf.ols(
    formula="price ~ np.log(lotsize) +np.log(sqrft) + bdrms",
    data=hprice1,
)
results2 = reg2.fit()

# encompassing test of Davidson & MacKinnon:
# comprehensive model:
reg3 = smf.ols(
    formula="price ~ lotsize + sqrft + bdrms + np.log(lotsize) + np.log(sqrft)",
    data=hprice1,
)
results3 = reg3.fit()

# model 1 vs. comprehensive model:
anovaResults1 = sm.stats.anova_lm(results1, results3)
print(f"anovaResults1: \n{anovaResults1}\n")
# -

# model 2 vs. comprehensive model:
anovaResults2 = sm.stats.anova_lm(results2, results3)
print(f"anovaResults2: \n{anovaResults2}\n")

# ## 9.2 Measurement Error

# +
# set the random seed:
np.random.seed(1234567)

# set sample size and number of simulations:
n = 1000
r = 10000

# set true parameters (betas):
beta0 = 1
beta1 = 0.5

# initialize arrays to store results later (b1 without ME, b1_me with ME):
b1 = np.empty(r)
b1_me = np.empty(r)

# draw a sample of x, fixed over replications:
x = stats.norm.rvs(4, 1, size=n)

# repeat r times:
for i in range(r):
    # draw a sample of u:
    u = stats.norm.rvs(0, 1, size=n)

    # draw a sample of ystar:
    ystar = beta0 + beta1 * x + u

    # measurement error and mismeasured y:
    e0 = stats.norm.rvs(0, 1, size=n)
    y = ystar + e0
    df = pd.DataFrame({"ystar": ystar, "y": y, "x": x})

    # regress ystar on x and store slope estimate at position i:
    reg_star = smf.ols(formula="ystar ~ x", data=df)
    results_star = reg_star.fit()
    b1[i] = results_star.params["x"]

    # regress y on x and store slope estimate at position i:
    reg_me = smf.ols(formula="y ~ x", data=df)
    results_me = reg_me.fit()
    b1_me[i] = results_me.params["x"]

# mean with and without ME:
b1_mean = np.mean(b1)
b1_me_mean = np.mean(b1_me)
print(f"b1_mean: {b1_mean}\n")
print(f"b1_me_mean: {b1_me_mean}\n")
# -

# variance with and without ME:
b1_var = np.var(b1, ddof=1)
b1_me_var = np.var(b1_me, ddof=1)
print(f"b1_var: {b1_var}\n")
print(f"b1_me_var: {b1_me_var}\n")

# +
# set the random seed:
np.random.seed(1234567)

# set sample size and number of simulations:
n = 1000
r = 10000

# set true parameters (betas):
beta0 = 1
beta1 = 0.5

# initialize b1 arrays to store results later:
b1 = np.empty(r)
b1_me = np.empty(r)

# draw a sample of x, fixed over replications:
xstar = stats.norm.rvs(4, 1, size=n)

# repeat r times:
for i in range(r):
    # draw a sample of u:
    u = stats.norm.rvs(0, 1, size=n)

    # draw a sample of y:
    y = beta0 + beta1 * xstar + u

    # measurement error and mismeasured x:
    e1 = stats.norm.rvs(0, 1, size=n)
    x = xstar + e1
    df = pd.DataFrame({"y": y, "xstar": xstar, "x": x})

    # regress y on xstar and store slope estimate at position i:
    reg_star = smf.ols(formula="y ~ xstar", data=df)
    results_star = reg_star.fit()
    b1[i] = results_star.params["xstar"]

    # regress y on x and store slope estimate at position i:
    reg_me = smf.ols(formula="y ~ x", data=df)
    results_me = reg_me.fit()
    b1_me[i] = results_me.params["x"]

# mean with and without ME:
b1_mean = np.mean(b1)
b1_me_mean = np.mean(b1_me)
print(f"b1_mean: {b1_mean}\n")
print(f"b1_me_mean: {b1_me_mean}\n")
# -

# variance with and without ME:
b1_var = np.var(b1, ddof=1)
b1_me_var = np.var(b1_me, ddof=1)
print(f"b1_var: {b1_var}\n")
print(f"b1_me_var: {b1_me_var}\n")

# ## 9.3 Missing Data and Nonrandom Samples

# +
# nan and inf handling in numpy:
x = np.array([-1, 0, 1, np.nan, np.inf, -np.inf])
logx = np.log(x)
invx = np.array(1 / x)
ncdf = np.array(stats.norm.cdf(x))
isnanx = np.isnan(x)

results = pd.DataFrame(
    {"x": x, "logx": logx, "invx": invx, "logx": logx, "ncdf": ncdf, "isnanx": isnanx},
)
print(f"results: \n{results}\n")

# +
lawsch85 = wool.data("lawsch85")
lsat_pd = lawsch85["LSAT"]

# create boolean indicator for missings:
missLSAT = lsat_pd.isna()

# LSAT and indicator for Schools No. 120-129:
preview = pd.DataFrame({"lsat_pd": lsat_pd[119:129], "missLSAT": missLSAT[119:129]})
print(f"preview: \n{preview}\n")
# -

# frequencies of indicator:
freq_missLSAT = pd.crosstab(missLSAT, columns="count")
print(f"freq_missLSAT: \n{freq_missLSAT}\n")

# missings for all variables in data frame (counts):
miss_all = lawsch85.isna()
colsums = miss_all.sum(axis=0)
print(f"colsums: \n{colsums}\n")

# computing amount of complete cases:
complete_cases = miss_all.sum(axis=1) == 0
freq_complete_cases = pd.crosstab(complete_cases, columns="count")
print(f"freq_complete_cases: \n{freq_complete_cases}\n")

# +
lawsch85 = wool.data("lawsch85")

# missings in numpy:
x_np = np.array(lawsch85["LSAT"])
x_np_bar1 = np.mean(x_np)
x_np_bar2 = np.nanmean(x_np)
print(f"x_np_bar1: {x_np_bar1}\n")
print(f"x_np_bar2: {x_np_bar2}\n")
# -

# missings in pandas:
x_pd = lawsch85["LSAT"]
x_pd_bar1 = np.mean(x_pd)
x_pd_bar2 = np.nanmean(x_pd)
print(f"x_pd_bar1: {x_pd_bar1}\n")
print(f"x_pd_bar2: {x_pd_bar2}\n")

# observations and variables:
print(f"lawsch85.shape: {lawsch85.shape}\n")

# regression (missings are taken care of by default):
reg = smf.ols(formula="np.log(salary) ~ LSAT + cost + age", data=lawsch85)
results = reg.fit()
print(f"results.nobs: {results.nobs}\n")

# ## 9.4 Outlying Observations

# +
rdchem = wool.data("rdchem")

# OLS regression:
reg = smf.ols(formula="rdintens ~ sales + profmarg", data=rdchem)
results = reg.fit()

# studentized residuals for all observations:
studres = results.get_influence().resid_studentized_external

# display extreme values:
studres_max = np.max(studres)
studres_min = np.min(studres)
print(f"studres_max: {studres_max}\n")
print(f"studres_min: {studres_min}\n")

# +
# histogram (and overlayed density plot):
kde = sm.nonparametric.KDEUnivariate(studres)
kde.fit()

plt.hist(studres, color="grey", density=True)
plt.plot(kde.support, kde.density, color="black", linewidth=2)
plt.ylabel("density")
plt.xlabel("studres")
# -

# ## 9.5 Least Absolute Deviations (LAD) Estimation

# +
rdchem = wool.data("rdchem")

# OLS regression:
reg_ols = smf.ols(formula="rdintens ~ I(sales/1000) + profmarg", data=rdchem)
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

# +
# LAD regression:
reg_lad = smf.quantreg(formula="rdintens ~ I(sales/1000) + profmarg", data=rdchem)
results_lad = reg_lad.fit(q=0.5)

table_lad = pd.DataFrame(
    {
        "b": round(results_lad.params, 4),
        "se": round(results_lad.bse, 4),
        "t": round(results_lad.tvalues, 4),
        "pval": round(results_lad.pvalues, 4),
    },
)
print(f"table_lad: \n{table_lad}\n")
