# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,markdown//md,scripts//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: merino
#     language: python
#     name: python3
# ---

# # 11. Further Issues in Using OLS with Time Series Data

# %pip install matplotlib numpy pandas statsmodels wooldridge scipy -q

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import wooldridge as wool
from scipy import stats

# ## 11.1 Asymptotics with Time Seires
#
# ### Example 11.4: Efficient Markets Hypothesis

# +
nyse = wool.data("nyse")
nyse["ret"] = nyse["return"]

# add all lags up to order 3:
nyse["ret_lag1"] = nyse["ret"].shift(1)
nyse["ret_lag2"] = nyse["ret"].shift(2)
nyse["ret_lag3"] = nyse["ret"].shift(3)

# linear regression of model with lags:
reg1 = smf.ols(formula="ret ~ ret_lag1", data=nyse)
reg2 = smf.ols(formula="ret ~ ret_lag1 + ret_lag2", data=nyse)
reg3 = smf.ols(formula="ret ~ ret_lag1 + ret_lag2 + ret_lag3", data=nyse)
results1 = reg1.fit()
results2 = reg2.fit()
results3 = reg3.fit()

# print regression tables:
table1 = pd.DataFrame(
    {
        "b": round(results1.params, 4),
        "se": round(results1.bse, 4),
        "t": round(results1.tvalues, 4),
        "pval": round(results1.pvalues, 4),
    },
)
print(f"table1: \n{table1}\n")
# -

table2 = pd.DataFrame(
    {
        "b": round(results2.params, 4),
        "se": round(results2.bse, 4),
        "t": round(results2.tvalues, 4),
        "pval": round(results2.pvalues, 4),
    },
)
print(f"table2: \n{table2}\n")

table3 = pd.DataFrame(
    {
        "b": round(results3.params, 4),
        "se": round(results3.bse, 4),
        "t": round(results3.tvalues, 4),
        "pval": round(results3.pvalues, 4),
    },
)
print(f"table3: \n{table3}\n")

# ## 11.2 The Nature of Highly Persistent Time Series

# +
# set the random seed:
np.random.seed(1234567)

# initialize plot:
x_range = np.linspace(0, 50, num=51)
plt.ylim([-18, 18])
plt.xlim([0, 50])

# loop over draws:
for r in range(30):
    # i.i.d. standard normal shock:
    e = stats.norm.rvs(0, 1, size=51)

    # set first entry to 0 (gives y_0 = 0):
    e[0] = 0

    # random walk as cumulative sum of shocks:
    y = np.cumsum(e)

    # add line to graph:
    plt.plot(x_range, y, color="lightgrey", linestyle="-")

plt.axhline(linewidth=2, linestyle="--", color="black")
plt.ylabel("y")
plt.xlabel("time")

# +
# set the random seed:
np.random.seed(1234567)

# initialize plot:
x_range = np.linspace(0, 50, num=51)
plt.ylim([0, 100])
plt.xlim([0, 50])

# loop over draws:
for r in range(30):
    # i.i.d. standard normal shock:
    e = stats.norm.rvs(0, 1, size=51)

    # set first entry to 0 (gives y_0 = 0):
    e[0] = 0

    # random walk as cumulative sum of shocks plus drift:
    y = np.cumsum(e) + 2 * x_range

    # add line to graph:
    plt.plot(x_range, y, color="lightgrey", linestyle="-")

plt.plot(x_range, 2 * x_range, linewidth=2, linestyle="--", color="black")
plt.ylabel("y")
plt.xlabel("time")
# -

# ## 11.3 Differences of Highly Persistent Time Series

# +
# set the random seed:
np.random.seed(1234567)

# initialize plot:
x_range = np.linspace(1, 50, num=50)
plt.ylim([-1, 5])
plt.xlim([0, 50])

# loop over draws:
for r in range(30):
    # i.i.d. standard normal shock and cumulative sum of shocks:
    e = stats.norm.rvs(0, 1, size=51)
    e[0] = 0
    y = np.cumsum(2 + e)

    # first difference:
    Dy = y[1:51] - y[0:50]

    # add line to graph:
    plt.plot(x_range, Dy, color="lightgrey", linestyle="-")

plt.axhline(y=2, linewidth=2, linestyle="--", color="black")
plt.ylabel("y")
plt.xlabel("time")
# -

# ## 11.4 Regression with First Differences
#
# ### Example 11.6: Fertility Equation

# +
fertil3 = wool.data("fertil3")
T = len(fertil3)

# define time series (years only) beginning in 1913:
fertil3.index = pd.date_range(start="1913", periods=T, freq="YE").year

# compute first differences:
fertil3["gfr_diff1"] = fertil3["gfr"].diff()
fertil3["pe_diff1"] = fertil3["pe"].diff()
print(f"fertil3.head(): \n{fertil3.head()}\n")

# +
# linear regression of model with first differences:
reg1 = smf.ols(formula="gfr_diff1 ~ pe_diff1", data=fertil3)
results1 = reg1.fit()

# print regression table:
table1 = pd.DataFrame(
    {
        "b": round(results1.params, 4),
        "se": round(results1.bse, 4),
        "t": round(results1.tvalues, 4),
        "pval": round(results1.pvalues, 4),
    },
)
print(f"table1: \n{table1}\n")

# +
# linear regression of model with lagged differences:
fertil3["pe_diff1_lag1"] = fertil3["pe_diff1"].shift(1)
fertil3["pe_diff1_lag2"] = fertil3["pe_diff1"].shift(2)

reg2 = smf.ols(
    formula="gfr_diff1 ~ pe_diff1 + pe_diff1_lag1 + pe_diff1_lag2",
    data=fertil3,
)
results2 = reg2.fit()

# print regression table:
table2 = pd.DataFrame(
    {
        "b": round(results2.params, 4),
        "se": round(results2.bse, 4),
        "t": round(results2.tvalues, 4),
        "pval": round(results2.pvalues, 4),
    },
)
print(f"table2: \n{table2}\n")
