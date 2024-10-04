# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,markdown//md,scripts//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: merino
#     language: python
#     name: python3
# ---

# # Ch6. Multiple Regression Analysis: Further Issues

# %pip install matplotlib numpy pandas statsmodels wooldridge -q

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import wooldridge as wool

# ## 6.1 Model Formulae
#
# ### 6.1.1 Data Scaling: Arithmetic Operations witin a Formula
#
# $$ \text{bwght} = \beta_0 + \beta_1 \cdot \text{cigs} + \beta_2 \cdot \text{faminc} + u$$

# +
bwght = wool.data("bwght")

# regress and report coefficients:
reg = smf.ols(formula="bwght ~ cigs + faminc", data=bwght)
results = reg.fit()

# weight in pounds, manual way:
bwght["bwght_lbs"] = bwght["bwght"] / 16
reg_lbs = smf.ols(formula="bwght_lbs ~ cigs + faminc", data=bwght)
results_lbs = reg_lbs.fit()

# weight in pounds, direct way:
reg_lbs2 = smf.ols(formula="I(bwght/16) ~ cigs + faminc", data=bwght)
results_lbs2 = reg_lbs2.fit()

# packs of cigarettes:
reg_packs = smf.ols(formula="bwght ~ I(cigs/20) + faminc", data=bwght)
results_packs = reg_packs.fit()

# compare results:
table = pd.DataFrame(
    {
        "b": round(results.params, 4),
        "b_lbs": round(results_lbs.params, 4),
        "b_lbs2": round(results_lbs2.params, 4),
        "b_packs": round(results_packs.params, 4),
    },
)
print(f"table: \n{table}\n")


# -

# ### 6.1.2 Standardization: Beta Coefficients
#
# $$z_y = \frac{y - \bar{y}}{\text{sd}(y)}  \qquad \text{and} \qquad z_{x_1} = \frac{x_1 - \bar{x}_1}{\text{sd}(x_1)}$$
#
# ### Example 6.1: Effects of Pollution on Housing Prices
#
# $$\text{price\_sc} = \beta_0 + \beta_1 \cdot \text{nox\_sc} + \beta_2 \cdot \text{crime\_sc} + \beta_3 \cdot \text{rooms\_sc} + \beta_4 \cdot \text{dist\_sc} + \beta_5 \cdot \text{stratio\_sc} + u$$

# +
# define a function for the standardization:
def scale(x):
    x_mean = np.mean(x)
    x_var = np.var(x, ddof=1)
    x_scaled = (x - x_mean) / np.sqrt(x_var)
    return x_scaled


# standardize and estimate:
hprice2 = wool.data("hprice2")
hprice2["price_sc"] = scale(hprice2["price"])
hprice2["nox_sc"] = scale(hprice2["nox"])
hprice2["crime_sc"] = scale(hprice2["crime"])
hprice2["rooms_sc"] = scale(hprice2["rooms"])
hprice2["dist_sc"] = scale(hprice2["dist"])
hprice2["stratio_sc"] = scale(hprice2["stratio"])

reg = smf.ols(
    formula="price_sc ~ 0 + nox_sc + crime_sc + rooms_sc + dist_sc + stratio_sc",
    data=hprice2,
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
# -

# ### 6.1.3 Logarithms
#
# $$\log(y) = \beta_0 + \beta_1 \log(x_1) + \beta_2 x_2 + u$$

# +
hprice2 = wool.data("hprice2")

reg = smf.ols(formula="np.log(price) ~ np.log(nox) + rooms", data=hprice2)
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

# ### 6.1.4 Quadratics and Polynomials
#
# $$ y = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + u$$
#
# ### Example 6.2: Effects of Pollution on Housing Prices
#
# $$\log(\text{price}) = \beta_0 + \beta_1 \log(\text{nox}) + \beta_2 \log(\text{dist}) + \beta_3 \text{rooms} + \beta_4 \text{rooms}^2 + \beta_5 \text{stratio} + u$$

# +
hprice2 = wool.data("hprice2")

reg = smf.ols(
    formula="np.log(price) ~ np.log(nox)+np.log(dist)+rooms+I(rooms**2)+stratio",
    data=hprice2,
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
# -

# ### 6.1.5 Hypothesis Testing

# +
hprice2 = wool.data("hprice2")
n = hprice2.shape[0]

reg = smf.ols(
    formula="np.log(price) ~ np.log(nox)+np.log(dist)+rooms+I(rooms**2)+stratio",
    data=hprice2,
)
results = reg.fit()

# implemented F test for rooms:
hypotheses = ["rooms = 0", "I(rooms ** 2) = 0"]
ftest = results.f_test(hypotheses)
fstat = ftest.statistic
fpval = ftest.pvalue

print(f"fstat: {fstat}\n")
print(f"fpval: {fpval}\n")
# -

# ### 6.1.6 Interaction Terms
#
# $$ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1 x_2 + u$$
#
# ### Example 6.3: Effects of Attendance on Final Exam Performance
#
# $$\text{stndfnl} = \beta_0 + \beta_1 \text{atndrte} + \beta_2 \text{priGPA} + \beta_3 \text{ACT} + \beta_4 \text{priGPA}^2 + \beta_5 \text{ACT}^2 + \beta_6 \text{atndrte} \cdot \text{priGPA} + u$$

# +
attend = wool.data("attend")
n = attend.shape[0]

reg = smf.ols(
    formula="stndfnl ~ atndrte*priGPA + ACT + I(priGPA**2) + I(ACT**2)",
    data=attend,
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
# -

# estimate for partial effect at priGPA=2.59:
b = results.params
partial_effect = b["atndrte"] + 2.59 * b["atndrte:priGPA"]
print(f"partial_effect: {partial_effect}\n")

# +
# F test for partial effect at priGPA=2.59:
hypotheses = "atndrte + 2.59 * atndrte:priGPA = 0"
ftest = results.f_test(hypotheses)
fstat = ftest.statistic
fpval = ftest.pvalue

print(f"fstat: {fstat}\n")
print(f"fpval: {fpval}\n")
# -

# ## 6.2 Prediction
#
# ### 6.2.1 Confidence and Prediction Intervals for Predictions

# +
gpa2 = wool.data("gpa2")

reg = smf.ols(formula="colgpa ~ sat + hsperc + hsize + I(hsize**2)", data=gpa2)
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

# generate data set containing the regressor values for predictions:
cvalues1 = pd.DataFrame(
    {"sat": [1200], "hsperc": [30], "hsize": [5]},
    index=["newPerson1"],
)
print(f"cvalues1: \n{cvalues1}\n")

# point estimate of prediction (cvalues1):
colgpa_pred1 = results.predict(cvalues1)
print(f"colgpa_pred1: \n{colgpa_pred1}\n")

# define three sets of regressor variables:
cvalues2 = pd.DataFrame(
    {"sat": [1200, 900, 1400], "hsperc": [30, 20, 5], "hsize": [5, 3, 1]},
    index=["newPerson1", "newPerson2", "newPerson3"],
)
print(f"cvalues2: \n{cvalues2}\n")

# point estimate of prediction (cvalues2):
colgpa_pred2 = results.predict(cvalues2)
print(f"colgpa_pred2: \n{colgpa_pred2}\n")

# ### Example 6.5: Confidence Interval for Predicted College GPA

# +
gpa2 = wool.data("gpa2")

reg = smf.ols(formula="colgpa ~ sat + hsperc + hsize + I(hsize**2)", data=gpa2)
results = reg.fit()

# define three sets of regressor variables:
cvalues2 = pd.DataFrame(
    {"sat": [1200, 900, 1400], "hsperc": [30, 20, 5], "hsize": [5, 3, 1]},
    index=["newPerson1", "newPerson2", "newPerson3"],
)

# point estimates and 95% confidence and prediction intervals:
colgpa_PICI_95 = results.get_prediction(cvalues2).summary_frame(alpha=0.05)
print(f"colgpa_PICI_95: \n{colgpa_PICI_95}\n")
# -

# point estimates and 99% confidence and prediction intervals:
colgpa_PICI_99 = results.get_prediction(cvalues2).summary_frame(alpha=0.01)
print(f"colgpa_PICI_99: \n{colgpa_PICI_99}\n")

# ### 6.2.2 Effect Plots for Nonlinear Specifications

# +
hprice2 = wool.data("hprice2")

# repeating the regression from Example 6.2:
reg = smf.ols(
    formula="np.log(price) ~ np.log(nox)+np.log(dist)+rooms+I(rooms**2)+stratio",
    data=hprice2,
)
results = reg.fit()

# predictions with rooms = 4-8, all others at the sample mean:
nox_mean = np.mean(hprice2["nox"])
dist_mean = np.mean(hprice2["dist"])
stratio_mean = np.mean(hprice2["stratio"])
X = pd.DataFrame(
    {
        "rooms": np.linspace(4, 8, num=5),
        "nox": nox_mean,
        "dist": dist_mean,
        "stratio": stratio_mean,
    },
)
print(f"X: \n{X}\n")
# -

# calculate 95% confidence interval:
lpr_PICI = results.get_prediction(X).summary_frame(alpha=0.05)
lpr_CI = lpr_PICI[["mean", "mean_ci_lower", "mean_ci_upper"]]
print(f"lpr_CI: \n{lpr_CI}\n")

# plot:
plt.plot(X["rooms"], lpr_CI["mean"], color="black", linestyle="-", label="")
plt.plot(
    X["rooms"],
    lpr_CI["mean_ci_upper"],
    color="lightgrey",
    linestyle="--",
    label="upper CI",
)
plt.plot(
    X["rooms"],
    lpr_CI["mean_ci_lower"],
    color="darkgrey",
    linestyle="--",
    label="lower CI",
)
plt.ylabel("lprice")
plt.xlabel("rooms")
plt.legend()
plt.show()
