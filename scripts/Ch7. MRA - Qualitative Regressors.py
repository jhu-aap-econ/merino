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

# # 7. Multiple Regression Analysis with Qualitative Regressors

# %pip install matplotlib numpy pandas statsmodels wooldridge -q

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import wooldridge as wool
import numpy as np

# ## 7.1 Linear Regression with Dummy Variables as Regressors
#
# ### Example 7.1: Hourly Wage Equation

# +
wage1 = wool.data("wage1")

reg = smf.ols(formula="wage ~ female + educ + exper + tenure", data=wage1)
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

# ### Example 7.6: Log Hourly Wage Equation

# +
wage1 = wool.data("wage1")

reg = smf.ols(
    formula="np.log(wage) ~ married*female + educ + exper +"
    "I(exper**2) + tenure + I(tenure**2)",
    data=wage1,
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

# ## 7.2 Boolean variables

# +
wage1 = wool.data("wage1")

# regression with boolean variable:
wage1["isfemale"] = wage1["female"] == 1
reg = smf.ols(formula="wage ~ isfemale + educ + exper + tenure", data=wage1)
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

# ## 7.3 Categorical Variables

# +
CPS1985 = pd.read_csv("../data/CPS1985.csv")
# rename variable to make outputs more compact:
CPS1985["oc"] = CPS1985["occupation"]

# table of categories and frequencies for two categorical variables:
freq_gender = pd.crosstab(CPS1985["gender"], columns="count")
print(f"freq_gender: \n{freq_gender}\n")

freq_occupation = pd.crosstab(CPS1985["oc"], columns="count")
print(f"freq_occupation: \n{freq_occupation}\n")

# +
# directly using categorical variables in regression formula:
reg = smf.ols(
    formula="np.log(wage) ~ education +experience + C(gender) + C(oc)",
    data=CPS1985,
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

# +
# rerun regression with different reference category:
reg_newref = smf.ols(
    formula="np.log(wage) ~ education + experience + "
    'C(gender, Treatment("male")) + '
    'C(oc, Treatment("technical"))',
    data=CPS1985,
)
results_newref = reg_newref.fit()

# print results:
table_newref = pd.DataFrame(
    {
        "b": round(results_newref.params, 4),
        "se": round(results_newref.bse, 4),
        "t": round(results_newref.tvalues, 4),
        "pval": round(results_newref.pvalues, 4),
    },
)
print(f"table_newref: \n{table_newref}\n")
# -

# ### 7.3.1 ANOVA Tables

# +
CPS1985 = pd.read_csv("../data/CPS1985.csv")

# run regression:
reg = smf.ols(
    formula="np.log(wage) ~ education + experience + gender + occupation",
    data=CPS1985,
)
results = reg.fit()

# print regression table:
table_reg = pd.DataFrame(
    {
        "b": round(results.params, 4),
        "se": round(results.bse, 4),
        "t": round(results.tvalues, 4),
        "pval": round(results.pvalues, 4),
    },
)
print(f"table_reg: \n{table_reg}\n")
# -

# ANOVA table:
table_anova = sm.stats.anova_lm(results, typ=2)
print(f"table_anova: \n{table_anova}\n")

# ## 7.4 Breaking a Numeric Variable Into Categories
#
# ### Example 7.8: Effects of Law School Rankings on Starting Salaries

# +
lawsch85 = wool.data("lawsch85")

# define cut points for the rank:
cutpts = [0, 10, 25, 40, 60, 100, 175]

# create categorical variable containing ranges for the rank:
lawsch85["rc"] = pd.cut(
    lawsch85["rank"],
    bins=cutpts,
    labels=["(0,10]", "(10,25]", "(25,40]", "(40,60]", "(60,100]", "(100,175]"],
)

# display frequencies:
freq = pd.crosstab(lawsch85["rc"], columns="count")
print(f"freq: \n{freq}\n")

# +
# run regression:
reg = smf.ols(
    formula='np.log(salary) ~ C(rc, Treatment("(100,175]")) +'
    "LSAT + GPA + np.log(libvol) + np.log(cost)",
    data=lawsch85,
)
results = reg.fit()

# print regression table:
table_reg = pd.DataFrame(
    {
        "b": round(results.params, 4),
        "se": round(results.bse, 4),
        "t": round(results.tvalues, 4),
        "pval": round(results.pvalues, 4),
    },
)
print(f"table_reg: \n{table_reg}\n")
# -

# ANOVA table:
table_anova = sm.stats.anova_lm(results, typ=2)
print(f"table_anova: \n{table_anova}\n")

# ## 7.5 Interactions and Differences in Regression Functions Across Groups

# +
gpa3 = wool.data("gpa3")

# model with full interactions with female dummy (only for spring data):
reg = smf.ols(
    formula="cumgpa ~ female * (sat + hsperc + tothrs)",
    data=gpa3,
    subset=(gpa3["spring"] == 1),
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

# +
# F-Test for H0 (the interaction coefficients of 'female' are zero):
hypotheses = ["female = 0", "female:sat = 0", "female:hsperc = 0", "female:tothrs = 0"]
ftest = results.f_test(hypotheses)
fstat = ftest.statistic
fpval = ftest.pvalue

print(f"fstat: {fstat}\n")
print(f"fpval: {fpval}\n")

# +
gpa3 = wool.data("gpa3")

# estimate model for males (& spring data):
reg_m = smf.ols(
    formula="cumgpa ~ sat + hsperc + tothrs",
    data=gpa3,
    subset=(gpa3["spring"] == 1) & (gpa3["female"] == 0),
)
results_m = reg_m.fit()

# print regression table:
table_m = pd.DataFrame(
    {
        "b": round(results_m.params, 4),
        "se": round(results_m.bse, 4),
        "t": round(results_m.tvalues, 4),
        "pval": round(results_m.pvalues, 4),
    },
)
print(f"table_m: \n{table_m}\n")

# +
# estimate model for females (& spring data):
reg_f = smf.ols(
    formula="cumgpa ~ sat + hsperc + tothrs",
    data=gpa3,
    subset=(gpa3["spring"] == 1) & (gpa3["female"] == 1),
)
results_f = reg_f.fit()

# print regression table:
table_f = pd.DataFrame(
    {
        "b": round(results_f.params, 4),
        "se": round(results_f.bse, 4),
        "t": round(results_f.tvalues, 4),
        "pval": round(results_f.pvalues, 4),
    },
)
print(f"table_f: \n{table_f}\n")
