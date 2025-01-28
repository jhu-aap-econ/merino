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

# # 10. Basic Regression Analysis with Time Series Data

# %pip install matplotlib numpy pandas statsmodels wooldridge -q

import matplotlib.pyplot as plt
import numpy as np  # noqa
import pandas as pd
import statsmodels.formula.api as smf
import wooldridge as wool

# ## 10.1 Static Time Series Models
#
# $$ y_t = \beta_0 + \beta_1 z_{1t} + \beta_2 z_{2t} + \cdots + \beta_k z_{kt} + u_t $$
#
# ### Example 10.2 Effects of Inflation and Deficits on Interest Rates

# +
intdef = wool.dataWoo("intdef")

# linear regression of static model (Q function avoids conflicts with keywords):
reg = smf.ols(formula='i3 ~ Q("inf") + Q("def")', data=intdef)
results = reg.fit()

# print regression table:
table = pd.DataFrame(
    {
        "b": round(results.params, 4),
        "se": round(results.bse, 4),
        "t": round(results.tvalues, 4),
        "pval": round(results.pvalues, 4),
    },
)
print(f"table: \n{table}\n")
# -

# ## 10.2 Time Series Data Types in Python
#
# ### 10.2.1 Equispaced Time Series in Python

# +
barium = wool.dataWoo("barium")
T = len(barium)

# monthly time series starting Feb. 1978:
barium.index = pd.date_range(start="1978-02", periods=T, freq="ME")
print(f'barium["chnimp"].head(): \n{barium["chnimp"].head()}\n')
# -

# plot chnimp (default: index on the x-axis):
plt.plot("chnimp", data=barium, color="black", linestyle="-")
plt.ylabel("chnimp")
plt.xlabel("time")

# ## 10.3 Other Time Series Models
#
# ### 10.3.1 Finite Distributed Lag Models
#
# $$ y_t = \alpha_0 + \delta_0 z_t + \delta_1 z_{t-1} + \cdots + \delta_k z_{t-k} + u_t $$
#
# ### Example 10.4 Effects of Personal Exemption on Fertility Rates

# +
fertil3 = wool.dataWoo("fertil3")
T = len(fertil3)

# define yearly time series beginning in 1913:
fertil3.index = pd.date_range(start="1913", periods=T, freq="YE").year

# add all lags of 'pe' up to order 2:
fertil3["pe_lag1"] = fertil3["pe"].shift(1)
fertil3["pe_lag2"] = fertil3["pe"].shift(2)

# linear regression of model with lags:
reg = smf.ols(formula="gfr ~ pe + pe_lag1 + pe_lag2 + ww2 + pill", data=fertil3)
results = reg.fit()

# print regression table:
table = pd.DataFrame(
    {
        "b": round(results.params, 4),
        "se": round(results.bse, 4),
        "t": round(results.tvalues, 4),
        "pval": round(results.pvalues, 4),
    },
)
print(f"table: \n{table}\n")
# -

# ### Eample 10.4 (continued)

# +
fertil3 = wool.dataWoo("fertil3")
T = len(fertil3)

# define yearly time series beginning in 1913:
fertil3.index = pd.date_range(start="1913", periods=T, freq="YE").year

# add all lags of 'pe' up to order 2:
fertil3["pe_lag1"] = fertil3["pe"].shift(1)
fertil3["pe_lag2"] = fertil3["pe"].shift(2)

# linear regression of model with lags:
reg = smf.ols(formula="gfr ~ pe + pe_lag1 + pe_lag2 + ww2 + pill", data=fertil3)
results = reg.fit()

# F test (H0: all pe coefficients are=0):
hypotheses1 = ["pe = 0", "pe_lag1 = 0", "pe_lag2 = 0"]
ftest1 = results.f_test(hypotheses1)
fstat1 = ftest1.statistic
fpval1 = ftest1.pvalue

print(f"fstat1: {fstat1}\n")
print(f"fpval1: {fpval1}\n")
# -

# calculating the LRP:
b = results.params
b_pe_tot = b["pe"] + b["pe_lag1"] + b["pe_lag2"]
print(f"b_pe_tot: {b_pe_tot}\n")

# +
# F test (H0: LRP=0):
hypotheses2 = ["pe + pe_lag1 + pe_lag2 = 0"]
ftest2 = results.f_test(hypotheses2)
fstat2 = ftest2.statistic
fpval2 = ftest2.pvalue

print(f"fstat2: {fstat2}\n")
print(f"fpval2: {fpval2}\n")
# -

# ### 10.3.2 Trends
#
# ### Example 10.7 Housing Investment and Prices

# +
hseinv = wool.dataWoo("hseinv")

# linear regression without time trend:
reg_wot = smf.ols(formula="np.log(invpc) ~ np.log(price)", data=hseinv)
results_wot = reg_wot.fit()

# print regression table:
table_wot = pd.DataFrame(
    {
        "b": round(results_wot.params, 4),
        "se": round(results_wot.bse, 4),
        "t": round(results_wot.tvalues, 4),
        "pval": round(results_wot.pvalues, 4),
    },
)
print(f"table_wot: \n{table_wot}\n")

# +
# linear regression with time trend (data set includes a time variable t):
reg_wt = smf.ols(formula="np.log(invpc) ~ np.log(price) + t", data=hseinv)
results_wt = reg_wt.fit()

# print regression table:
table_wt = pd.DataFrame(
    {
        "b": round(results_wt.params, 4),
        "se": round(results_wt.bse, 4),
        "t": round(results_wt.tvalues, 4),
        "pval": round(results_wt.pvalues, 4),
    },
)
print(f"table_wt: \n{table_wt}\n")
# -

# ### 10.3.3 Seasonality
#
# ### Example 10.11 Effects of Antidumping Filings

# +
barium = wool.dataWoo("barium")

# linear regression with seasonal effects:
reg = smf.ols(
    formula="np.log(chnimp) ~ np.log(chempi) + np.log(gas) +"
    "np.log(rtwex) + befile6 + affile6 + afdec6 +"
    "feb + mar + apr + may + jun + jul +"
    "aug + sep + oct + nov + dec",
    data=barium,
)
results = reg.fit()

# print regression table:
table = pd.DataFrame(
    {
        "b": round(results.params, 4),
        "se": round(results.bse, 4),
        "t": round(results.tvalues, 4),
        "pval": round(results.pvalues, 4),
    },
)
print(f"table: \n{table}\n")
