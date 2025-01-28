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

# # 4. Multiple Regression Analysis: Inference

# %pip install numpy statsmodels wooldridge scipy -q

import numpy as np
import statsmodels.formula.api as smf
import wooldridge as wool
from scipy import stats

# ## 4.1 The $t$ Test
#
# ### 4.1.1 General Setup
#
# $$H_0: \beta_j = a_j$$
#
# $$H_1: \beta_j \neq a_j \quad \text{or} \quad H_1:\beta_j > a_j \quad \text{or} \quad H_1:\beta_j < a_j$$
#
# $$t = \frac{\hat{\beta}_j - a_j}{se(\hat{\beta}_j)}$$
#
# ### 4.1.2 Standard Case
#
# $$H_0: \beta_j = 0, \qquad H_1: \beta_j \neq 0$$
#
# $$t_{\hat{\beta}_j} = \frac{\hat{\beta}_j}{se(\hat{\beta}_j)}$$
#
# $$\text{reject } H_0 \text{ if } |t_{\hat{\beta}_j}| > c$$
#
# $$p_{\hat{\beta}_j} = 2 \cdot F_{t_{n-k-1}}(-|t_{\hat{\beta}_j}|)$$
#
# $$\text{reject } H_0 \text{ if } p_{\hat{\beta}_j} < \alpha$$
#
# ### Example 4.3: Determinants of College GPA
#
# $$\text{colGPA} = \beta_0 + \beta_1 \text{hsGPA} + \beta_2 \text{ACT} + \beta_3 \text{skipped} + u$$

# CV for alpha=5% and 1% using the t distribution with 137 d.f.:
alpha = np.array([0.05, 0.01])
cv_t = stats.t.ppf(1 - alpha / 2, 137)
print(f"cv_t: {cv_t}\n")

# CV for alpha=5% and 1% using the normal approximation:
cv_n = stats.norm.ppf(1 - alpha / 2)
print(f"cv_n: {cv_n}\n")

# +
gpa1 = wool.data("gpa1")

# store and display results:
reg = smf.ols(formula="colGPA ~ hsGPA + ACT + skipped", data=gpa1)
results = reg.fit()
print(f"results.summary(): \n{results.summary()}\n")

# +
# manually confirm the formulas, i.e. extract coefficients and SE:
b = results.params
se = results.bse

# reproduce t statistic:
tstat = b / se
print(f"tstat: \n{tstat}\n")

# reproduce p value:
pval = 2 * stats.t.cdf(-abs(tstat), 137)
print(f"pval: \n{pval}\n")
# -

# ### 4.1.3 Other Hypotheses
#
# ### Example 4.1: Hourly Wage Equation
#
# $$\log(\text{wage}) = \beta_0 + \beta_1 \text{educ} + \beta_2 \text{exper} + \beta_3 \text{tenure} + u$$

# CV for alpha=5% and 1% using the t distribution with 522 d.f.:
alpha = np.array([0.05, 0.01])
cv_t = stats.t.ppf(1 - alpha, 522)
print(f"cv_t: {cv_t}\n")

# CV for alpha=5% and 1% using the normal approximation:
cv_n = stats.norm.ppf(1 - alpha)
print(f"cv_n: {cv_n}\n")

# +
wage1 = wool.data("wage1")

reg = smf.ols(formula="np.log(wage) ~ educ + exper + tenure", data=wage1)
results = reg.fit()
print(f"results.summary(): \n{results.summary()}\n")
# -

# ## 4.2 Confidence Intervals
#
# $$\hat{\beta}_j \pm c \cdot se(\hat{\beta}_j)$$
#
# ### Example 4.8: Model of R&D Expenditures
#
# $$\log(\text{rd}) = \beta_0 + \beta_1 \log(\text{sales}) + \beta_2 \text{profmarg} + u$$

# +
rdchem = wool.data("rdchem")

# OLS regression:
reg = smf.ols(formula="np.log(rd) ~ np.log(sales) + profmarg", data=rdchem)
results = reg.fit()
print(f"results.summary(): \n{results.summary()}\n")
# -

# 95% CI:
CI95 = results.conf_int(0.05)
print(f"CI95: \n{CI95}\n")

# 99% CI:
CI99 = results.conf_int(0.01)
print(f"CI99: \n{CI99}\n")

# ## 4.3 Linear Restrictions: $F$ Tests
#
# $$\log(\text{salary}) = \beta_0 + \beta_1 \text{years} + \beta_2 \text{gamesyr} + \beta_3 \text{bavg} + \beta_4 \text{hrunsyr} + \beta_5 \text{rbisyr} + u$$
#
# $$F = \frac{SSR_r - SSR_{ur}}{SSR_{ur}} \cdot \frac{n - k - 1}{q} = \frac{R^2_{ur} - R^2_r}{1 - R^2_{ur}} \cdot \frac{n - k - 1}{q}$$

# +
mlb1 = wool.data("mlb1")
n = mlb1.shape[0]

# unrestricted OLS regression:
reg_ur = smf.ols(
    formula="np.log(salary) ~ years + gamesyr + bavg + hrunsyr + rbisyr",
    data=mlb1,
)
fit_ur = reg_ur.fit()
r2_ur = fit_ur.rsquared
print(f"r2_ur: {r2_ur}\n")
# -

# restricted OLS regression:
reg_r = smf.ols(formula="np.log(salary) ~ years + gamesyr", data=mlb1)
fit_r = reg_r.fit()
r2_r = fit_r.rsquared
print(f"r2_r: {r2_r}\n")

# F statistic:
fstat = (r2_ur - r2_r) / (1 - r2_ur) * (n - 6) / 3
print(f"fstat: {fstat}\n")

# CV for alpha=1% using the F distribution with 3 and 347 d.f.:
cv = stats.f.ppf(1 - 0.01, 3, 347)
print(f"cv: {cv}\n")

# p value = 1-cdf of the appropriate F distribution:
fpval = 1 - stats.f.cdf(fstat, 3, 347)
print(f"fpval: {fpval}\n")

# +
mlb1 = wool.data("mlb1")

# OLS regression:
reg = smf.ols(
    formula="np.log(salary) ~ years + gamesyr + bavg + hrunsyr + rbisyr",
    data=mlb1,
)
results = reg.fit()

# automated F test:
hypotheses = ["bavg = 0", "hrunsyr = 0", "rbisyr = 0"]
ftest = results.f_test(hypotheses)
fstat = ftest.statistic
fpval = ftest.pvalue

print(f"fstat: {fstat}\n")
print(f"fpval: {fpval}\n")

# +
mlb1 = wool.data("mlb1")

# OLS regression:
reg = smf.ols(
    formula="np.log(salary) ~ years + gamesyr + bavg + hrunsyr + rbisyr",
    data=mlb1,
)
results = reg.fit()

# automated F test:
hypotheses = ["bavg = 0", "hrunsyr = 2*rbisyr"]
ftest = results.f_test(hypotheses)
fstat = ftest.statistic
fpval = ftest.pvalue

print(f"fstat: {fstat}\n")
print(f"fpval: {fpval}\n")
