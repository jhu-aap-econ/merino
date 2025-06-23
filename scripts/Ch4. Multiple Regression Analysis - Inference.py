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
# # 4. Multiple Regression Analysis: Inference
#
# This notebook delves into the crucial aspect of **inference** in the context of multiple regression analysis. Building upon the concepts of estimation from previous chapters, we will now focus on drawing conclusions about the population parameters based on our sample data. This involves hypothesis testing and constructing confidence intervals, allowing us to assess the statistical significance and practical importance of our regression results.

# %%
import numpy as np
import statsmodels.formula.api as smf
import wooldridge as wool
from scipy import stats

# %% [markdown]
# ## 4.1 The $t$ Test
#
# The $t$ test is a fundamental tool for hypothesis testing about individual regression coefficients in multiple regression models. It allows us to formally examine whether a specific independent variable has a statistically significant effect on the dependent variable, holding other factors constant.
#
# ### 4.1.1 General Setup
#
# In a multiple regression model, we are often interested in testing hypotheses about a single population parameter, say $\beta_j$. We might want to test if $\beta_j$ is equal to some specific value, $a_j$.  The null hypothesis ($H_0$) typically represents a statement of no effect or a specific hypothesized value, while the alternative hypothesis ($H_1$) represents what we are trying to find evidence for.
#
# The general form of the null and alternative hypotheses for a $t$ test is:
#
# $$H_0: \beta_j = a_j$$
#
# $$H_1: \beta_j \neq a_j \quad \text{or} \quad H_1:\beta_j > a_j \quad \text{or} \quad H_1:\beta_j < a_j$$
#
# *   $H_0: \beta_j = a_j$: This is the **null hypothesis**, stating that the population coefficient $\beta_j$ is equal to a specific value $a_j$.  Often, $a_j = 0$, implying no effect of the $j^{th}$ independent variable on the dependent variable, *ceteris paribus*.
# *   $H_1: \beta_j \neq a_j$: This is a **two-sided alternative hypothesis**, stating that $\beta_j$ is different from $a_j$. We reject $H_0$ if $\beta_j$ is either significantly greater or significantly less than $a_j$.
# *   $H_1:\beta_j > a_j$: This is a **one-sided alternative hypothesis**, stating that $\beta_j$ is greater than $a_j$. We reject $H_0$ only if $\beta_j$ is significantly greater than $a_j$.
# *   $H_1:\beta_j < a_j$: This is another **one-sided alternative hypothesis**, stating that $\beta_j$ is less than $a_j$. We reject $H_0$ only if $\beta_j$ is significantly less than $a_j$.
#
# To test the null hypothesis, we use the **$t$ statistic**:
#
# $$t = \frac{\hat{\beta}_j - a_j}{se(\hat{\beta}_j)}$$
#
# *   $\hat{\beta}_j$: This is the estimated coefficient for the $j^{th}$ independent variable from our regression.
# *   $a_j$: This is the value of $\beta_j$ under the null hypothesis (from $H_0: \beta_j = a_j$).
# *   $se(\hat{\beta}_j)$: This is the standard error of the estimated coefficient $\hat{\beta}_j$, which measures the precision of our estimate.
#
# Under the null hypothesis and under the CLM assumptions, this $t$ statistic follows a $t$ distribution with $n-k-1$ degrees of freedom, where $n$ is the sample size and $k$ is the number of independent variables in the model.
#
# ### 4.1.2 Standard Case
#
# The most common hypothesis test is to check if a particular independent variable has no effect on the dependent variable in the population, which corresponds to testing if the coefficient is zero. In this standard case, we set $a_j = 0$.
#
# $$H_0: \beta_j = 0, \qquad H_1: \beta_j \neq 0$$
#
# The $t$ statistic simplifies to:
#
# $$t_{\hat{\beta}_j} = \frac{\hat{\beta}_j}{se(\hat{\beta}_j)}$$
#
# To decide whether to reject the null hypothesis, we compare the absolute value of the calculated $t$ statistic, $|t_{\hat{\beta}_j}|$, to a **critical value** ($c$) from the $t$ distribution, or we examine the **p-value** ($p_{\hat{\beta}_j}$).
#
# **Rejection Rule using Critical Value:**
#
# $$\text{reject } H_0 \text{ if } |t_{\hat{\beta}_j}| > c$$
#
# *   $c$:  The critical value is obtained from the $t$ distribution with $n-k-1$ degrees of freedom for a chosen significance level ($\alpha$). For a two-sided test at a significance level $\alpha$, we typically use $c = t_{n-k-1, 1-\alpha/2}$, which is the $(1-\alpha/2)$ quantile of the $t_{n-k-1}$ distribution. Common significance levels are $\alpha = 0.05$ (5%) and $\alpha = 0.01$ (1%).
#
# **Rejection Rule using p-value:**
#
# $$p_{\hat{\beta}_j} = 2 \cdot F_{t_{n-k-1}}(-|t_{\hat{\beta}_j}|)$$
#
# $$\text{reject } H_0 \text{ if } p_{\hat{\beta}_j} < \alpha$$
#
# *   $p_{\hat{\beta}_j}$: The p-value is the probability of observing a $t$ statistic as extreme as, or more extreme than, the one calculated from our sample, *assuming the null hypothesis is true*. It's a measure of the evidence against the null hypothesis. A small p-value indicates strong evidence against $H_0$.
# *   $F_{t_{n-k-1}}$: This denotes the cumulative distribution function (CDF) of the $t$ distribution with $n-k-1$ degrees of freedom. The formula calculates the area in both tails of the $t$ distribution beyond $|t_{\hat{\beta}_j}|$, hence the factor of 2 for a two-sided test.
#
# **In summary:** We reject the null hypothesis if the absolute value of the $t$ statistic is large enough (greater than the critical value) or if the p-value is small enough (less than the significance level $\alpha$). Both methods lead to the same conclusion.
#
# ### Example 4.3: Determinants of College GPA
#
# Let's consider an example investigating the factors influencing college GPA (`colGPA`). We hypothesize that high school GPA (`hsGPA`), ACT score (`ACT`), and number of skipped classes (`skipped`) are determinants of college GPA. The model is specified as:
#
# $$\text{colGPA} = \beta_0 + \beta_1 \text{hsGPA} + \beta_2 \text{ACT} + \beta_3 \text{skipped} + u$$
#
# We will perform hypothesis tests on the coefficients $\beta_1$, $\beta_2$, and $\beta_3$ to see which of these variables are statistically significant predictors of college GPA. We will use the standard null hypothesis $H_0: \beta_j = 0$ for each variable.

# %%
# CV for alpha=5% and 1% using the t distribution with 137 d.f.:
alpha = np.array([0.05, 0.01])
cv_t = stats.t.ppf(1 - alpha / 2, 137)  # Two-sided critical values
print(
    f"Critical values from t-distribution (df=137):\nFor alpha={alpha[0] * 100}%: +/-{cv_t[0]:.3f}\nFor alpha={alpha[1] * 100}%: +/-{cv_t[1]:.3f}\n",
)

# %% [markdown]
# This code calculates the critical values from the $t$ distribution for significance levels of 5% and 1% with 137 degrees of freedom (which we will see is approximately the degrees of freedom in our regression). These are the thresholds against which we'll compare our calculated $t$-statistics.

# %%
# CV for alpha=5% and 1% using the normal approximation:
cv_n = stats.norm.ppf(1 - alpha / 2)  # Two-sided critical values
print(
    f"Critical values from standard normal distribution:\nFor alpha={alpha[0] * 100}%: +/-{cv_n[0]:.3f}\nFor alpha={alpha[1] * 100}%: +/-{cv_n[1]:.3f}\n",
)

# %% [markdown]
# For large degrees of freedom, the $t$ distribution approaches the standard normal distribution. This code shows the critical values from the standard normal distribution for comparison. Notice that for these common significance levels, the critical values are quite similar for the $t$ and normal distributions when the degrees of freedom are reasonably large (like 137).

# %%
gpa1 = wool.data("gpa1")

# store and display results:
reg = smf.ols(formula="colGPA ~ hsGPA + ACT + skipped", data=gpa1)
results = reg.fit()
print(f"Regression summary:\n{results.summary()}\n")

# %% [markdown]
# This code runs the OLS regression of `colGPA` on `hsGPA`, `ACT`, and `skipped` using the `gpa1` dataset from the `wooldridge` package. The `results.summary()` provides a comprehensive output of the regression results, including estimated coefficients, standard errors, t-statistics, p-values, and other relevant statistics.

# %%
# manually confirm the formulas, i.e. extract coefficients and SE:
b = results.params
se = results.bse

# reproduce t statistic:
tstat = b / se
print(f"Calculated t-statistics:\n{tstat}\n")

# reproduce p value:
pval = 2 * stats.t.cdf(
    -abs(tstat),
    results.df_resid,
)  # df_resid is the degrees of freedom
print(f"Calculated p-values:\n{pval}\n")

# %% [markdown]
# This section manually calculates the $t$ statistics and p-values using the formulas we discussed. It extracts the estimated coefficients (`b`) and standard errors (`se`) from the regression results. Then, it calculates the $t$ statistic by dividing each coefficient by its standard error. Finally, it computes the two-sided p-value using the CDF of the $t$ distribution with the correct degrees of freedom (`results.df_resid`).  The calculated values should match those reported in the `results.summary()`, confirming our understanding of how these values are derived.
#
# **Interpreting the results from `results.summary()` and manual calculations for Example 4.3:**
#
# *   **`hsGPA` (High School GPA):** The estimated coefficient is 0.4118 and statistically significant (p-value < 0.01). The t-statistic is 4.396. We reject the null hypothesis that $\beta_{hsGPA} = 0$. This suggests that higher high school GPA is associated with a significantly higher college GPA, holding ACT score and skipped classes constant.
#
# *   **`ACT` (ACT Score):** The estimated coefficient is 0.0147 but not statistically significant at the 5% level (p-value is 0.166, which is > 0.05). The t-statistic is 1.393. We fail to reject the null hypothesis that $\beta_{ACT} = 0$ at the 5% significance level. This indicates that ACT score has a positive but weaker relationship with college GPA in this model compared to high school GPA.  More data might be needed to confidently conclude ACT score is a significant predictor, or perhaps its effect is less linear or captured by other variables.
#
# *   **`skipped` (Skipped Classes):** The estimated coefficient is -0.0831 and statistically significant (p-value = 0.002). The t-statistic is -3.197. We reject the null hypothesis that $\beta_{skipped} = 0$. This indicates that skipping more classes is associated with a significantly lower college GPA, holding high school GPA and ACT score constant.
#
# ### 4.1.3 Other Hypotheses
#
# While testing if $\beta_j = 0$ is the most common, we can also test other hypotheses of the form $H_0: \beta_j = a_j$ where $a_j \neq 0$. This might be relevant if we have a specific theoretical value in mind for $\beta_j$.
#
# We can also conduct **one-tailed tests** if we have a directional alternative hypothesis (either $H_1: \beta_j > a_j$ or $H_1: \beta_j < a_j$). In these cases, the rejection region is only in one tail of the $t$ distribution, and the p-value calculation is adjusted accordingly (we would not multiply by 2).  One-tailed tests should be used cautiously and only when there is a strong prior expectation about the direction of the effect.
#
# ### Example 4.1: Hourly Wage Equation
#
# Let's consider another example, examining the determinants of hourly wage. We model the logarithm of wage ($\log(\text{wage})$) as a function of education (`educ`), experience (`exper`), and tenure (`tenure`):
#
# $$\log(\text{wage}) = \beta_0 + \beta_1 \text{educ} + \beta_2 \text{exper} + \beta_3 \text{tenure} + u$$
#
# We will focus on testing hypotheses about the returns to education ($\beta_1$). We might want to test if the return to education is greater than some specific value, or simply if it is different from zero.

# %%
# CV for alpha=5% and 1% using the t distribution with 522 d.f.:
alpha = np.array([0.05, 0.01])
cv_t = stats.t.ppf(1 - alpha / 2, 522)  # Two-sided critical values
print(
    f"Critical values from t-distribution (df=522):\nFor alpha={alpha[0] * 100}%: +/-{cv_t[0]:.3f}\nFor alpha={alpha[1] * 100}%: +/-{cv_t[1]:.3f}\n",
)

# %% [markdown]
# Similar to the previous example, we calculate the critical values from the $t$ distribution for significance levels of 5% and 1%, but now with 522 degrees of freedom (approximately the degrees of freedom in this regression).

# %%
# CV for alpha=5% and 1% using the normal approximation:
cv_n = stats.norm.ppf(1 - alpha / 2)  # Two-sided critical values
print(
    f"Critical values from standard normal distribution:\nFor alpha={alpha[0] * 100}%: +/-{cv_n[0]:.3f}\nFor alpha={alpha[1] * 100}%: +/-{cv_n[1]:.3f}\n",
)

# %% [markdown]
# Again, we compare these to the critical values from the standard normal distribution.  With 522 degrees of freedom, the $t$ and normal critical values are almost identical.

# %%
wage1 = wool.data("wage1")

reg = smf.ols(formula="np.log(wage) ~ educ + exper + tenure", data=wage1)
results = reg.fit()
print(f"Regression summary:\n{results.summary()}\n")

# %% [markdown]
# This code runs the regression of $\log(\text{wage})$ on `educ`, `exper`, and `tenure` using the `wage1` dataset.
#
# **Interpreting the results from `results.summary()` for Example 4.1:**
#
# *   **`educ` (Education):** The estimated coefficient for `educ` is 0.0920. This means that, holding experience and tenure constant, an additional year of education is associated with an estimated 9.20% increase in hourly wage (since we are using the log of wage as the dependent variable, and for small changes, the coefficient multiplied by 100 gives the percentage change). The t-statistic for `educ` is 12.555 and the p-value is extremely small (< 0.001). We strongly reject the null hypothesis $H_0: \beta_{educ} = 0$. We conclude that education has a statistically significant positive effect on wages.
#
# *   **`exper` (Experience):** The coefficient for `exper` is 0.0041 and statistically significant (p-value = 0.017), indicating that more experience is associated with higher wages, holding education and tenure constant.
#
# *   **`tenure` (Tenure):** Similarly, the coefficient for `tenure` is 0.0221 and statistically significant (p-value < 0.001), suggesting that longer tenure with the current employer is associated with higher wages, controlling for education and overall experience.
#
# ## 4.2 Confidence Intervals
#
# Confidence intervals provide a range of plausible values for a population parameter, such as a regression coefficient. They give us a measure of the uncertainty associated with our point estimate ($\hat{\beta}_j$). A confidence interval is constructed around the estimated coefficient.
#
# The general formula for a $(1-\alpha) \cdot 100\%$ confidence interval for $\beta_j$ is:
#
# $$\hat{\beta}_j \pm c \cdot se(\hat{\beta}_j)$$
#
# *   $\hat{\beta}_j$: The estimated coefficient.
# *   $se(\hat{\beta}_j)$: The standard error of the estimated coefficient.
# *   $c$: The critical value from the $t$ distribution with $n-k-1$ degrees of freedom for a $(1-\alpha) \cdot 100\%$ confidence level. For a 95% confidence interval ($\alpha = 0.05$), we use $c = t_{n-k-1, 0.975}$. For a 99% confidence interval ($\alpha = 0.01$), we use $c = t_{n-k-1, 0.995}$.
#
# **Interpretation of a Confidence Interval:** We are $(1-\alpha) \cdot 100\%$ confident that the true population coefficient $\beta_j$ lies within the calculated interval. It's important to remember that the confidence interval is constructed from sample data, and it is the interval that varies from sample to sample, not the true population parameter $\beta_j$, which is fixed.
#
# ### Example 4.8: Model of R&D Expenditures
#
# Let's consider a model explaining research and development (R&D) expenditures (`rd`) as a function of sales (`sales`) and profit margin (`profmarg`):
#
# $$\log(\text{rd}) = \beta_0 + \beta_1 \log(\text{sales}) + \beta_2 \text{profmarg} + u$$
#
# We will construct 95% and 99% confidence intervals for the coefficients $\beta_1$ and $\beta_2$.

# %%
rdchem = wool.data("rdchem")

# OLS regression:
reg = smf.ols(formula="np.log(rd) ~ np.log(sales) + profmarg", data=rdchem)
results = reg.fit()
print(f"Regression summary:\n{results.summary()}\n")

# %% [markdown]
# This code runs the OLS regression of $\log(\text{rd})$ on $\log(\text{sales})$ and `profmarg` using the `rdchem` dataset.

# %%
# 95% CI:
CI95 = results.conf_int(0.05)  # alpha = 0.05 for 95% CI
print(f"95% Confidence Intervals:\n{CI95}\n")

# %% [markdown]
# This code uses the `conf_int()` method of the regression results object to calculate the 95% confidence intervals for all coefficients.

# %%
# 99% CI:
CI99 = results.conf_int(0.01)  # alpha = 0.01 for 99% CI
print(f"99% Confidence Intervals:\n{CI99}\n")

# %% [markdown]
# Similarly, this calculates the 99% confidence intervals.
#
# **Interpreting the Confidence Intervals from Example 4.8:**
#
# *   **`np.log(sales)` (Log of Sales):**
#     *   95% CI: [0.961, 1.207]. We are 95% confident that the true elasticity of R&D with respect to sales (percentage change in R&D for a 1% change in sales) lies between 0.961 and 1.207. Since 1 is within this interval, we cannot reject the hypothesis that the elasticity is exactly 1 at the 5% significance level.
#     *   99% CI: [0.918, 1.250]. The 99% confidence interval is wider than the 95% interval, reflecting the higher level of confidence.
#
# *   **`profmarg` (Profit Margin):**
#     *   95% CI: [-0.004, 0.048]. We are 95% confident that the true coefficient for profit margin is between -0.004 and 0.048. Since 0 is in this interval, we cannot reject the null hypothesis that $\beta_{profmarg} = 0$ at the 5% significance level.
#     *   99% CI: [-0.014, 0.057].  The 99% CI includes 0, confirming that profit margin is not statistically significant at the 1% level.
#
# As expected, the 99% confidence intervals are wider than the 95% confidence intervals. This is because to be more confident that we capture the true parameter, we need to consider a wider range of values.
#
# ## 4.3 Linear Restrictions: $F$ Tests
#
# The $t$ test is useful for testing hypotheses about a single coefficient. However, we often want to test hypotheses involving **multiple coefficients simultaneously**. For example, we might want to test if several independent variables are jointly insignificant, or if there is a specific linear relationship between multiple coefficients.  For these situations, we use the **$F$ test**.
#
# Consider the following model of baseball player salaries:
#
# $$\log(\text{salary}) = \beta_0 + \beta_1 \text{years} + \beta_2 \text{gamesyr} + \beta_3 \text{bavg} + \beta_4 \text{hrunsyr} + \beta_5 \text{rbisyr} + u$$
#
# Suppose we want to test if batting average (`bavg`), home runs per year (`hrunsyr`), and runs batted in per year (`rbisyr`) have no joint effect on salary, after controlling for years in the league (`years`) and games played per year (`gamesyr`). This translates to testing the following joint null hypothesis:
#
# $$H_0: \beta_3 = 0, \beta_4 = 0, \beta_5 = 0$$
#
# $$H_1: \text{at least one of } \beta_3, \beta_4, \beta_5 \neq 0$$
#
# To perform an $F$ test, we need to estimate two regressions:
#
# 1.  **Unrestricted Model:** The original, full model (with all variables included). In our example, this is the model above with `years`, `gamesyr`, `bavg`, `hrunsyr`, and `rbisyr`.  Let $SSR_{ur}$ be the sum of squared residuals from the unrestricted model.
#
# 2.  **Restricted Model:** The model obtained by imposing the restrictions specified in the null hypothesis. In our example, under $H_0$, $\beta_3 = \beta_4 = \beta_5 = 0$, so the restricted model is:
#
#     $$\log(\text{salary}) = \beta_0 + \beta_1 \text{years} + \beta_2 \text{gamesyr} + u$$
#
#     Let $SSR_r$ be the sum of squared residuals from the restricted model.
#
# The **$F$ statistic** is calculated as:
#
# $$F = \frac{SSR_r - SSR_{ur}}{SSR_{ur}} \cdot \frac{n - k - 1}{q} = \frac{R^2_{ur} - R^2_r}{1 - R^2_{ur}} \cdot \frac{n - k - 1}{q}$$
#
# *   $SSR_r$: Sum of squared residuals from the restricted model.
# *   $SSR_{ur}$: Sum of squared residuals from the unrestricted model.
# *   $R^2_r$: R-squared from the restricted model.
# *   $R^2_{ur}$: R-squared from the unrestricted model.
# *   $n$: Sample size.
# *   $k$: Number of independent variables in the unrestricted model.
# *   $q$: Number of restrictions being tested (in our example, $q=3$ because we are testing three restrictions: $\beta_3=0, \beta_4=0, \beta_5=0$).
#
# Under the null hypothesis and the CLM assumptions, the $F$ statistic follows an $F$ distribution with $(q, n-k-1)$ degrees of freedom. We reject the null hypothesis if the calculated $F$ statistic is large enough, or equivalently, if the p-value is small enough.
#
# **Rejection Rule:**
#
# *   Reject $H_0$ if $F > c$, where $c$ is the critical value from the $F_{q, n-k-1}$ distribution at the chosen significance level.
# *   Reject $H_0$ if $p \text{-value} < \alpha$, where $p \text{-value} = 1 - F_{F_{q, n-k-1}}(F)$ and $\alpha$ is the significance level.

# %%
mlb1 = wool.data("mlb1")
n = mlb1.shape[0]

# unrestricted OLS regression:
reg_ur = smf.ols(
    formula="np.log(salary) ~ years + gamesyr + bavg + hrunsyr + rbisyr",
    data=mlb1,
)
fit_ur = reg_ur.fit()
r2_ur = fit_ur.rsquared
print(f"R-squared of unrestricted model (r2_ur): {r2_ur:.4f}\n")

# %% [markdown]
# This code estimates the unrestricted model and extracts the R-squared value.

# %%
# restricted OLS regression:
reg_r = smf.ols(formula="np.log(salary) ~ years + gamesyr", data=mlb1)
fit_r = reg_r.fit()
r2_r = fit_r.rsquared
print(f"R-squared of restricted model (r2_r): {r2_r:.4f}\n")

# %% [markdown]
# This code estimates the restricted model (by omitting `bavg`, `hrunsyr`, and `rbisyr`) and extracts its R-squared. As expected, the R-squared of the restricted model is lower than that of the unrestricted model because we have removed variables.

# %%
# F statistic:
k = 5  # Number of independent variables in unrestricted model
q = 3  # Number of restrictions
fstat = (r2_ur - r2_r) / (1 - r2_ur) * (n - k - 1) / q
print(f"Calculated F statistic: {fstat:.3f}\n")

# %% [markdown]
# This code calculates the $F$ statistic using the formula based on R-squared values.

# %%
# CV for alpha=1% using the F distribution with 3 and 347 d.f.:
cv = stats.f.ppf(1 - 0.01, q, n - k - 1)  # Degrees of freedom (q, n-k-1)
print(
    f"Critical value from F-distribution (df=({q}, {n - k - 1})) for alpha=1%: {cv:.3f}\n",
)

# %% [markdown]
# This calculates the critical value from the $F$ distribution with 3 and $n-k-1$ degrees of freedom for a 1% significance level.

# %%
# p value = 1-cdf of the appropriate F distribution:
fpval = 1 - stats.f.cdf(fstat, q, n - k - 1)
print(f"Calculated p-value: {fpval:.4f}\n")

# %% [markdown]
# This calculates the p-value for the $F$ test using the CDF of the $F$ distribution.
#
# **Interpreting the results of the F-test for joint significance of batting stats:**
#
# The calculated $F$ statistic is around 9.25. The p-value is very small (approximately 0.00003).  Comparing the F-statistic to the critical value (or the p-value to the significance level), we strongly reject the null hypothesis $H_0: \beta_3 = 0, \beta_4 = 0, \beta_5 = 0$.
#
# **Conclusion:** We conclude that batting average, home runs per year, and runs batted in per year are jointly statistically significant determinants of baseball player salaries, even after controlling for years in the league and games played per year.  In other words, at least one of these batting statistics has a significant impact on salary.

# %%
mlb1 = wool.data("mlb1")

# OLS regression:
reg = smf.ols(
    formula="np.log(salary) ~ years + gamesyr + bavg + hrunsyr + rbisyr",
    data=mlb1,
)
results = reg.fit()

# automated F test:
hypotheses = ["bavg = 0", "hrunsyr = 0", "rbisyr = 0"]  # List of restrictions
ftest = results.f_test(hypotheses)  # Perform F-test using statsmodels
fstat = ftest.statistic
fpval = ftest.pvalue

print(f"F statistic from automated test: {fstat:.3f}\n")
print(f"P-value from automated test: {fpval:.4f}\n")

# %% [markdown]
# This code demonstrates how to perform the same $F$ test using the `f_test()` method in `statsmodels`, which provides a more convenient way to conduct $F$ tests for linear restrictions. The results should be identical to our manual calculation, which they are (within rounding).

# %%
mlb1 = wool.data("mlb1")

# OLS regression:
reg = smf.ols(
    formula="np.log(salary) ~ years + gamesyr + bavg + hrunsyr + rbisyr",
    data=mlb1,
)
results = reg.fit()

# automated F test with a more complex hypothesis:
hypotheses = [
    "bavg = 0",
    "hrunsyr = 2*rbisyr",
]  # Testing two restrictions: beta_bavg = 0 and beta_hrunsyr = 2*beta_rbisyr
ftest = results.f_test(hypotheses)  # Perform F-test
fstat = ftest.statistic
fpval = ftest.pvalue

print(f"F statistic for complex hypotheses: {fstat:.3f}\n")
print(f"P-value for complex hypotheses: {fpval:.4f}\n")

# %% [markdown]
# This final example shows the flexibility of the `f_test()` method. Here, we test a different joint hypothesis: $H_0: \beta_{bavg} = 0 \text{ and } \beta_{hrunsyr} = 2\beta_{rbisyr}$. This is a more complex linear restriction involving a relationship between two coefficients. The `f_test()` method easily handles such hypotheses, demonstrating its power and convenience for testing various linear restrictions in multiple regression models.
#
# **Interpreting the results of the second F-test:**
#
# In this case, the F-statistic is around 4.57 and the p-value is approximately 0.0109. If we use a 1% significance level, we would fail to reject the null hypothesis. However, at a 5% significance level, we would reject $H_0$.  Therefore, there is moderate evidence against the null hypothesis that batting average has no effect and that the effect of home runs per year is twice the effect of runs batted in per year.  Further analysis or a larger dataset might be needed to draw firmer conclusions.
#
# **Summary of F-tests:**
#
# *   $F$ tests are used to test joint hypotheses about multiple regression coefficients.
# *   They compare an unrestricted model to a restricted model that imposes the null hypothesis.
# *   The $F$ statistic measures the relative increase in the sum of squared residuals (or decrease in R-squared) when moving from the unrestricted to the restricted model.
# *   A large $F$ statistic (or small p-value) provides evidence against the null hypothesis.
# *   $F$ tests are essential for testing the joint significance of sets of variables and for testing more complex linear restrictions on regression coefficients.
#
# This notebook has provided a comprehensive overview of inference in multiple regression analysis, covering $t$ tests for individual coefficients, confidence intervals, and $F$ tests for joint hypotheses. These tools are fundamental for drawing meaningful conclusions from regression models and for rigorously testing economic and statistical hypotheses.
