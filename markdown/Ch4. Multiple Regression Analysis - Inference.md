---
jupyter:
  jupytext:
    formats: notebooks//ipynb,markdown//md,scripts//py
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: merino
    language: python
    name: python3
---

# Chapter 4: Multiple Regression Analysis - Inference

Statistical inference extends multiple regression analysis beyond point estimation to hypothesis testing and interval estimation. This chapter develops the theoretical and practical foundations for drawing conclusions about population parameters from sample data, enabling researchers to assess both statistical significance and practical importance of regression results.

The organization proceeds systematically through the essential components of inference. We begin with the sampling distribution of OLS estimators and construction of standard errors (Section 4.1), develop hypothesis testing procedures for single coefficients (Section 4.2), extend to joint hypothesis tests involving multiple restrictions (Section 4.3), and examine confidence intervals alongside practical considerations for specification and reporting (Section 4.4-4.7). Throughout, we emphasize both theoretical foundations and computational implementation in Python.

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import wooldridge as wool
from scipy import stats
```

## 4.1 Assumptions for Statistical Inference

Before discussing hypothesis testing, we must clarify which assumptions are required for valid statistical inference in multiple regression. From Chapter 3, we have the Gauss-Markov assumptions (MLR.1-MLR.5) which guarantee that OLS estimators are BLUE. For statistical inference (hypothesis tests and confidence intervals), we consider two cases:

**Case 1: Exact Inference in Finite Samples (Classical Linear Model)**

For exact inference in small or finite samples using the t-distribution and F-distribution, we need an additional assumption beyond MLR.1-MLR.5:

**MLR.6: Normality of Errors**
The population error $u$ is conditionally normally distributed:
$$u|x_1, x_2, \ldots, x_k \sim \text{Normal}(0, \sigma^2)$$

Under the full **Classical Linear Model (CLM)** assumptions **MLR.1-MLR.6**, the OLS estimators have exact normal sampling distributions, and the t-statistics and F-statistics follow exact t and F distributions in finite samples.

**Case 2: Asymptotic Inference in Large Samples**

For asymptotically valid inference in large samples, we need only the Gauss-Markov assumptions **MLR.1-MLR.5** (no normality required). By the Central Limit Theorem (Chapter 5), as $n \to \infty$:
- The OLS estimators are approximately normally distributed
- The t-statistics approximately follow the t-distribution (or standard normal for large $n$)
- The F-statistics approximately follow the F-distribution

**Practical Implication:** With moderate to large sample sizes (typically $n \geq 30$ or more), the t-tests and F-tests are robust to violations of normality, making them widely applicable even when errors are not normally distributed. Throughout this chapter, we assume sufficient conditions for valid inference--either MLR.1-MLR.6 for exact results or MLR.1-MLR.5 with large $n$ for asymptotic results.

**Summary Table: Assumptions Required for Different Properties**

| Property | Required Assumptions | Notes |
|----------|---------------------|-------|
| **Consistency** | MLR.1-MLR.4 | Weaker than unbiasedness; $n \to \infty$ |
| **Unbiasedness** | MLR.1-MLR.4 | Zero conditional mean (MLR.4) is crucial |
| **Efficiency (BLUE)** | MLR.1-MLR.5 | Homoscedasticity (MLR.5) required |
| **Exact t/F tests** | MLR.1-MLR.6 | Normality (MLR.6) for finite samples |
| **Asymptotic t/F tests** | MLR.1-MLR.5 | CLT applies as $n \to \infty$ |
| **Valid OLS standard errors** | MLR.1-MLR.5 | Homoscedasticity needed; otherwise use robust SE |

## 4.2 The $t$ Test

The $t$ test is a fundamental tool for hypothesis testing about individual regression coefficients in multiple regression models. It allows us to formally examine whether a specific independent variable has a statistically significant effect on the dependent variable, holding other factors constant.

### 4.2.1 General Setup

In a multiple regression model, we are often interested in testing hypotheses about a single population parameter, say $\beta_j$. We might want to test if $\beta_j$ is equal to some specific value, $a_j$.  The null hypothesis ($H_0$) typically represents a statement of no effect or a specific hypothesized value, while the alternative hypothesis ($H_1$) represents what we are trying to find evidence for.

The general form of the null and alternative hypotheses for a $t$ test is:

$$H_0: \beta_j = a_j$$

$$H_1: \beta_j \neq a_j \quad \text{or} \quad H_1:\beta_j > a_j \quad \text{or} \quad H_1:\beta_j < a_j$$

*   $H_0: \beta_j = a_j$: This is the **null hypothesis**, stating that the population coefficient $\beta_j$ is equal to a specific value $a_j$.  Often, $a_j = 0$, implying no effect of the $j^{th}$ independent variable on the dependent variable, *ceteris paribus*.
*   $H_1: \beta_j \neq a_j$: This is a **two-sided alternative hypothesis**, stating that $\beta_j$ is different from $a_j$. We reject $H_0$ if $\beta_j$ is either significantly greater or significantly less than $a_j$.
*   $H_1:\beta_j > a_j$: This is a **one-sided alternative hypothesis**, stating that $\beta_j$ is greater than $a_j$. We reject $H_0$ only if $\beta_j$ is significantly greater than $a_j$.
*   $H_1:\beta_j < a_j$: This is another **one-sided alternative hypothesis**, stating that $\beta_j$ is less than $a_j$. We reject $H_0$ only if $\beta_j$ is significantly less than $a_j$.

To test the null hypothesis, we use the **$t$ statistic**:

$$t = \frac{\hat{\beta}_j - a_j}{se(\hat{\beta}_j)}$$

*   $\hat{\beta}_j$: This is the estimated coefficient for the $j^{th}$ independent variable from our regression.
*   $a_j$: This is the value of $\beta_j$ under the null hypothesis (from $H_0: \beta_j = a_j$).
*   $se(\hat{\beta}_j)$: This is the standard error of the estimated coefficient $\hat{\beta}_j$, which measures the precision of our estimate.

Under the null hypothesis and under the CLM assumptions, this $t$ statistic follows a $t$ distribution with $n-k-1$ degrees of freedom, where $n$ is the sample size and $k$ is the number of independent variables in the model.

### 4.2.2 Standard Case

The most common hypothesis test is to check if a particular independent variable has no effect on the dependent variable in the population, which corresponds to testing if the coefficient is zero. In this standard case, we set $a_j = 0$.

$$H_0: \beta_j = 0, \qquad H_1: \beta_j \neq 0$$

The $t$ statistic simplifies to:

$$t_{\hat{\beta}_j} = \frac{\hat{\beta}_j}{se(\hat{\beta}_j)}$$

To decide whether to reject the null hypothesis, we compare the absolute value of the calculated $t$ statistic, $|t_{\hat{\beta}_j}|$, to a **critical value** ($c$) from the $t$ distribution, or we examine the **p-value** ($p_{\hat{\beta}_j}$).

**Rejection Rule using Critical Value:**

$$\text{reject } H_0 \text{ if } |t_{\hat{\beta}_j}| > c$$

*   $c$:  The critical value is obtained from the $t$ distribution with $n-k-1$ degrees of freedom for a chosen significance level ($\alpha$). For a two-sided test at a significance level $\alpha$, we typically use $c = t_{n-k-1, 1-\alpha/2}$, which is the $(1-\alpha/2)$ quantile of the $t_{n-k-1}$ distribution. Common significance levels are $\alpha = 0.05$ (5%) and $\alpha = 0.01$ (1%).

**Rejection Rule using p-value:**

$$p_{\hat{\beta}_j} = 2 \cdot F_{t_{n-k-1}}(-|t_{\hat{\beta}_j}|)$$

$$\text{reject } H_0 \text{ if } p_{\hat{\beta}_j} < \alpha$$

*   $p_{\hat{\beta}_j}$: The p-value is the probability of observing a $t$ statistic as extreme as, or more extreme than, the one calculated from our sample, *assuming the null hypothesis is true*. It's a measure of the evidence against the null hypothesis. A small p-value indicates strong evidence against $H_0$.
*   $F_{t_{n-k-1}}$: This denotes the cumulative distribution function (CDF) of the $t$ distribution with $n-k-1$ degrees of freedom. The formula calculates the area in both tails of the $t$ distribution beyond $|t_{\hat{\beta}_j}|$, hence the factor of 2 for a two-sided test.

**In summary:** We reject the null hypothesis if the absolute value of the $t$ statistic is large enough (greater than the critical value) or if the p-value is small enough (less than the significance level $\alpha$). Both methods lead to the same conclusion.

### Example 4.3: Determinants of College GPA

Let's consider an example investigating the factors influencing college GPA (`colGPA`). We hypothesize that high school GPA (`hsGPA`), ACT score (`ACT`), and number of skipped classes (`skipped`) are determinants of college GPA. The model is specified as:

$$\text{colGPA} = \beta_0 + \beta_1 \text{hsGPA} + \beta_2 \text{ACT} + \beta_3 \text{skipped} + u$$

We will perform hypothesis tests on the coefficients $\beta_1$, $\beta_2$, and $\beta_3$ to see which of these variables are statistically significant predictors of college GPA. We will use the standard null hypothesis $H_0: \beta_j = 0$ for each variable.

```python
# Calculate critical values for hypothesis testing
# These are the thresholds for rejecting H0 at different significance levels

significance_levels = np.array([0.05, 0.01])  # alpha = 5% and 1%
degrees_freedom = 137  # Will be n - k - 1 from our regression

# Two-sided critical values: P(|t| > c) = alpha
critical_values_t = stats.t.ppf(1 - significance_levels / 2, degrees_freedom)

# Display critical values
crit_val_df = pd.DataFrame(
    {
        "Significance Level": [
            f"{significance_levels[0]:.0%}",
            f"{significance_levels[1]:.0%}",
        ],
        "Critical Value": [
            f"+/-{critical_values_t[0]:.3f}",
            f"+/-{critical_values_t[1]:.3f}",
        ],
        "Reject H_0 if": [
            f"|t-stat| > {critical_values_t[0]:.3f}",
            f"|t-stat| > {critical_values_t[1]:.3f}",
        ],
    },
)
crit_val_df
```

This code calculates the critical values from the $t$ distribution for significance levels of 5% and 1% with 137 degrees of freedom (which we will see is approximately the degrees of freedom in our regression). These are the thresholds against which we'll compare our calculated $t$-statistics.

```python
# CV for alpha=5% and 1% using the normal approximation:
alpha = np.array([0.05, 0.01])
cv_n = stats.norm.ppf(1 - alpha / 2)  # Two-sided critical values
# Critical values from standard normal distribution
pd.DataFrame(
    {
        "Alpha": [f"{alpha[0] * 100}%", f"{alpha[1] * 100}%"],
        "Critical Values": [f"+/-{cv_n[0]:.3f}", f"+/-{cv_n[1]:.3f}"],
    },
)
```

For large degrees of freedom, the $t$ distribution approaches the standard normal distribution. This code shows the critical values from the standard normal distribution for comparison. Notice that for these common significance levels, the critical values are quite similar for the $t$ and normal distributions when the degrees of freedom are reasonably large (like 137).

```python
gpa1 = wool.data("gpa1")

# store and display results:
reg = smf.ols(formula="colGPA ~ hsGPA + ACT + skipped", data=gpa1)
results = reg.fit()
# Display regression results
pd.DataFrame(
    {
        "b": results.params.round(4),
        "se": results.bse.round(4),
        "t": results.tvalues.round(4),
        "pval": results.pvalues.round(4),
    },
)
```

This code runs the OLS regression of `colGPA` on `hsGPA`, `ACT`, and `skipped` using the `gpa1` dataset from the `wooldridge` package. The `results.summary()` provides a comprehensive output of the regression results, including estimated coefficients, standard errors, t-statistics, p-values, and other relevant statistics.

```python
# Manually verify hypothesis testing calculations
# This demonstrates how t-statistics and p-values are computed

# Extract estimated coefficients and their standard errors
coefficient_estimates = results.params  # beta_hat from OLS
standard_errors = results.bse  # SE(beta_hat)

# Calculate t-statistics for H_0: beta_j = 0
# Formula: t = (beta_hat_j - 0) / SE(beta_hat_j)
t_statistics = coefficient_estimates / standard_errors

# Manual Hypothesis Testing Calculations:
manual_test_results = pd.DataFrame(
    {
        "Variable": coefficient_estimates.index,
        "beta_hat": coefficient_estimates.values,
        "SE": standard_errors.values,
        "t-statistic": t_statistics.values,
    },
)
display(manual_test_results)

# Calculate two-sided p-values
# p-value = P(|T| > |t_obs|) = 2 * P(T < -|t_obs|) where T ~ t(df)
p_values = 2 * stats.t.cdf(
    -abs(t_statistics),  # Use negative absolute value for lower tail
    results.df_resid,  # degrees of freedom = n - k - 1
)

# Degrees of freedom
dof_info = pd.DataFrame(
    {
        "Metric": ["Degrees of freedom"],
        "Value": [int(results.df_resid)],
    },
)
display(dof_info)
# \nCalculated p-values:
# Create DataFrame with p-values and significance levels
p_val_results = pd.DataFrame(
    {
        "Variable": coefficient_estimates.index,
        "p-value": p_values,
        "Significance": [
            "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            for p in p_values
        ],
    },
)
display(p_val_results)
```

This section manually calculates the $t$ statistics and p-values using the formulas we discussed. It extracts the estimated coefficients (`b`) and standard errors (`se`) from the regression results. Then, it calculates the $t$ statistic by dividing each coefficient by its standard error. Finally, it computes the two-sided p-value using the CDF of the $t$ distribution with the correct degrees of freedom (`results.df_resid`).  The calculated values should match those reported in the `results.summary()`, confirming our understanding of how these values are derived.

**Interpreting the results from `results.summary()` and manual calculations for Example 4.3:**

*   **`hsGPA` (High School GPA):** The estimated coefficient is 0.4118 and statistically significant (p-value < 0.01). The t-statistic is 4.396. We reject the null hypothesis that $\beta_{hsGPA} = 0$. This suggests that higher high school GPA is associated with a significantly higher college GPA, holding ACT score and skipped classes constant.

*   **`ACT` (ACT Score):** The estimated coefficient is 0.0147 but not statistically significant at the 5% level (p-value is 0.166, which is > 0.05). The t-statistic is 1.393. We fail to reject the null hypothesis that $\beta_{ACT} = 0$ at the 5% significance level. This indicates that ACT score has a positive but weaker relationship with college GPA in this model compared to high school GPA.  More data might be needed to confidently conclude ACT score is a significant predictor, or perhaps its effect is less linear or captured by other variables.

*   **`skipped` (Skipped Classes):** The estimated coefficient is -0.0831 and statistically significant (p-value = 0.002). The t-statistic is -3.197. We reject the null hypothesis that $\beta_{skipped} = 0$. This indicates that skipping more classes is associated with a significantly lower college GPA, holding high school GPA and ACT score constant.

### 4.1.3 Other Hypotheses

While testing if $\beta_j = 0$ is the most common, we can also test other hypotheses of the form $H_0: \beta_j = a_j$ where $a_j \neq 0$. This might be relevant if we have a specific theoretical value in mind for $\beta_j$.

We can also conduct **one-tailed tests** if we have a directional alternative hypothesis (either $H_1: \beta_j > a_j$ or $H_1: \beta_j < a_j$). In these cases, the rejection region is only in one tail of the $t$ distribution, and the p-value calculation is adjusted accordingly (we would not multiply by 2).  One-tailed tests should be used cautiously and only when there is a strong prior expectation about the direction of the effect.

### Example 4.1: Hourly Wage Equation

Let's consider another example, examining the determinants of hourly wage. We model the logarithm of wage ($\log(\text{wage})$) as a function of education (`educ`), experience (`exper`), and tenure (`tenure`):

$$\log(\text{wage}) = \beta_0 + \beta_1 \text{educ} + \beta_2 \text{exper} + \beta_3 \text{tenure} + u$$

We will focus on testing hypotheses about the returns to education ($\beta_1$). We might want to test if the return to education is greater than some specific value, or simply if it is different from zero.

```python
# CV for alpha=5% and 1% using the t distribution with 522 d.f.:
alpha = np.array([0.05, 0.01])
cv_t = stats.t.ppf(1 - alpha / 2, 522)  # Two-sided critical values
# Critical values from t-distribution (df=522)
pd.DataFrame(
    {
        "Alpha": [f"{alpha[0] * 100}%", f"{alpha[1] * 100}%"],
        "Critical Values (t-dist, df=522)": [f"+/-{cv_t[0]:.3f}", f"+/-{cv_t[1]:.3f}"],
    },
)
```

Similar to the previous example, we calculate the critical values from the $t$ distribution for significance levels of 5% and 1%, but now with 522 degrees of freedom (approximately the degrees of freedom in this regression).

```python
# CV for alpha=5% and 1% using the normal approximation:
alpha = np.array([0.05, 0.01])
cv_n = stats.norm.ppf(1 - alpha / 2)  # Two-sided critical values
# Critical values from standard normal distribution
pd.DataFrame(
    {
        "Alpha": [f"{alpha[0] * 100}%", f"{alpha[1] * 100}%"],
        "Critical Values": [f"+/-{cv_n[0]:.3f}", f"+/-{cv_n[1]:.3f}"],
    },
)
```

Again, we compare these to the critical values from the standard normal distribution.  With 522 degrees of freedom, the $t$ and normal critical values are almost identical.

```python
wage1 = wool.data("wage1")

reg = smf.ols(formula="np.log(wage) ~ educ + exper + tenure", data=wage1)
results = reg.fit()
# Display regression results
pd.DataFrame(
    {
        "b": results.params.round(4),
        "se": results.bse.round(4),
        "t": results.tvalues.round(4),
        "pval": results.pvalues.round(4),
    },
)
```

This code runs the regression of $\log(\text{wage})$ on `educ`, `exper`, and `tenure` using the `wage1` dataset.

**Interpreting the results from `results.summary()` for Example 4.1:**

*   **`educ` (Education):** The estimated coefficient for `educ` is 0.0920. This means that, holding experience and tenure constant, an additional year of education is associated with an estimated 9.20% increase in hourly wage (since we are using the log of wage as the dependent variable, and for small changes, the coefficient multiplied by 100 gives the percentage change). The t-statistic for `educ` is 12.555 and the p-value is extremely small (< 0.001). We strongly reject the null hypothesis $H_0: \beta_{educ} = 0$. We conclude that education has a statistically significant positive effect on wages.

*   **`exper` (Experience):** The coefficient for `exper` is 0.0041 and statistically significant (p-value = 0.017), indicating that more experience is associated with higher wages, holding education and tenure constant.

*   **`tenure` (Tenure):** Similarly, the coefficient for `tenure` is 0.0221 and statistically significant (p-value < 0.001), suggesting that longer tenure with the current employer is associated with higher wages, controlling for education and overall experience.

## 4.3 Confidence Intervals

Confidence intervals provide a range of plausible values for a population parameter, such as a regression coefficient. They give us a measure of the uncertainty associated with our point estimate ($\hat{\beta}_j$). A confidence interval is constructed around the estimated coefficient.

The general formula for a $(1-\alpha) \cdot 100\%$ confidence interval for $\beta_j$ is:

$$\hat{\beta}_j \pm c \cdot se(\hat{\beta}_j)$$

*   $\hat{\beta}_j$: The estimated coefficient.
*   $se(\hat{\beta}_j)$: The standard error of the estimated coefficient.
*   $c$: The critical value from the $t$ distribution with $n-k-1$ degrees of freedom for a $(1-\alpha) \cdot 100\%$ confidence level. For a 95% confidence interval ($\alpha = 0.05$), we use $c = t_{n-k-1, 0.975}$. For a 99% confidence interval ($\alpha = 0.01$), we use $c = t_{n-k-1, 0.995}$.

**Interpretation of a Confidence Interval:** We are $(1-\alpha) \cdot 100\%$ confident that the true population coefficient $\beta_j$ lies within the calculated interval. It's important to remember that the confidence interval is constructed from sample data, and it is the interval that varies from sample to sample, not the true population parameter $\beta_j$, which is fixed.

### Example 4.8: Model of R&D Expenditures

Let's consider a model explaining research and development (R&D) expenditures (`rd`) as a function of sales (`sales`) and profit margin (`profmarg`):

$$\log(\text{rd}) = \beta_0 + \beta_1 \log(\text{sales}) + \beta_2 \text{profmarg} + u$$

We will construct 95% and 99% confidence intervals for the coefficients $\beta_1$ and $\beta_2$.

```python
rdchem = wool.data("rdchem")

# OLS regression:
reg = smf.ols(formula="np.log(rd) ~ np.log(sales) + profmarg", data=rdchem)
results = reg.fit()
# Display regression results
pd.DataFrame(
    {
        "b": results.params.round(4),
        "se": results.bse.round(4),
        "t": results.tvalues.round(4),
        "pval": results.pvalues.round(4),
    },
)
```

This code runs the OLS regression of $\log(\text{rd})$ on $\log(\text{sales})$ and `profmarg` using the `rdchem` dataset.

```python
# 95% CI:
CI95 = results.conf_int(0.05)  # alpha = 0.05 for 95% CI
# Display 95% Confidence Intervals
CI95
```

This code uses the `conf_int()` method of the regression results object to calculate the 95% confidence intervals for all coefficients.

```python
# 99% CI:
CI99 = results.conf_int(0.01)  # alpha = 0.01 for 99% CI
# Display 99% Confidence Intervals
CI99
```

Similarly, this calculates the 99% confidence intervals.

**Interpreting the Confidence Intervals from Example 4.8:**

*   **`np.log(sales)` (Log of Sales):**
    *   95% CI: [0.961, 1.207]. We are 95% confident that the true elasticity of R&D with respect to sales (percentage change in R&D for a 1% change in sales) lies between 0.961 and 1.207. Since 1 is within this interval, we cannot reject the hypothesis that the elasticity is exactly 1 at the 5% significance level.
    *   99% CI: [0.918, 1.250]. The 99% confidence interval is wider than the 95% interval, reflecting the higher level of confidence.

*   **`profmarg` (Profit Margin):**
    *   95% CI: [-0.004, 0.048]. We are 95% confident that the true coefficient for profit margin is between -0.004 and 0.048. Since 0 is in this interval, we cannot reject the null hypothesis that $\beta_{profmarg} = 0$ at the 5% significance level.
    *   99% CI: [-0.014, 0.057].  The 99% CI includes 0, confirming that profit margin is not statistically significant at the 1% level.

As expected, the 99% confidence intervals are wider than the 95% confidence intervals. This is because to be more confident that we capture the true parameter, we need to consider a wider range of values.

## 4.4 Linear Restrictions: $F$ Tests

The $t$ test is useful for testing hypotheses about a single coefficient. However, we often want to test hypotheses involving **multiple coefficients simultaneously**. For example, we might want to test if several independent variables are jointly insignificant, or if there is a specific linear relationship between multiple coefficients.  For these situations, we use the **$F$ test**.

Consider the following model of baseball player salaries:

$$\log(\text{salary}) = \beta_0 + \beta_1 \text{years} + \beta_2 \text{gamesyr} + \beta_3 \text{bavg} + \beta_4 \text{hrunsyr} + \beta_5 \text{rbisyr} + u$$

Suppose we want to test if batting average (`bavg`), home runs per year (`hrunsyr`), and runs batted in per year (`rbisyr`) have no joint effect on salary, after controlling for years in the league (`years`) and games played per year (`gamesyr`). This translates to testing the following joint null hypothesis:

$$H_0: \beta_3 = 0, \beta_4 = 0, \beta_5 = 0$$

$$H_1: \text{at least one of } \beta_3, \beta_4, \beta_5 \neq 0$$

To perform an $F$ test, we need to estimate two regressions:

1.  **Unrestricted Model:** The original, full model (with all variables included). In our example, this is the model above with `years`, `gamesyr`, `bavg`, `hrunsyr`, and `rbisyr`.  Let $SSR_{ur}$ be the sum of squared residuals from the unrestricted model.

2.  **Restricted Model:** The model obtained by imposing the restrictions specified in the null hypothesis. In our example, under $H_0$, $\beta_3 = \beta_4 = \beta_5 = 0$, so the restricted model is:

    $$\log(\text{salary}) = \beta_0 + \beta_1 \text{years} + \beta_2 \text{gamesyr} + u$$

    Let $SSR_r$ be the sum of squared residuals from the restricted model.

The **$F$ statistic** is calculated as:

$$F = \frac{SSR_r - SSR_{ur}}{SSR_{ur}} \cdot \frac{n - k - 1}{q} = \frac{R^2_{ur} - R^2_r}{1 - R^2_{ur}} \cdot \frac{n - k - 1}{q}$$

*   $SSR_r$: Sum of squared residuals from the restricted model.
*   $SSR_{ur}$: Sum of squared residuals from the unrestricted model.
*   $R^2_r$: R-squared from the restricted model.
*   $R^2_{ur}$: R-squared from the unrestricted model.
*   $n$: Sample size.
*   $k$: Number of independent variables in the unrestricted model.
*   $q$: Number of restrictions being tested (in our example, $q=3$ because we are testing three restrictions: $\beta_3=0, \beta_4=0, \beta_5=0$).

Under the null hypothesis and the CLM assumptions, the $F$ statistic follows an $F$ distribution with $(q, n-k-1)$ degrees of freedom. We reject the null hypothesis if the calculated $F$ statistic is large enough, or equivalently, if the p-value is small enough.

**Rejection Rule:**

*   Reject $H_0$ if $F > c$, where $c$ is the critical value from the $F_{q, n-k-1}$ distribution at the chosen significance level.
*   Reject $H_0$ if $p \text{-value} < \alpha$, where $p \text{-value} = 1 - F_{F_{q, n-k-1}}(F)$ and $\alpha$ is the significance level.

```python
mlb1 = wool.data("mlb1")
n = mlb1.shape[0]

# unrestricted OLS regression:
reg_ur = smf.ols(
    formula="np.log(salary) ~ years + gamesyr + bavg + hrunsyr + rbisyr",
    data=mlb1,
)
fit_ur = reg_ur.fit()
r2_ur = fit_ur.rsquared
# R-squared of unrestricted model
pd.DataFrame(
    {
        "Model": ["Unrestricted"],
        "R-squared": [f"{r2_ur:.4f}"],
    },
)
```

This code estimates the unrestricted model and extracts the R-squared value.

```python
# restricted OLS regression:
reg_r = smf.ols(formula="np.log(salary) ~ years + gamesyr", data=mlb1)
fit_r = reg_r.fit()
r2_r = fit_r.rsquared
# R-squared of restricted model
pd.DataFrame(
    {
        "Model": ["Restricted"],
        "R-squared": [f"{r2_r:.4f}"],
    },
)
```

This code estimates the restricted model (by omitting `bavg`, `hrunsyr`, and `rbisyr`) and extracts its R-squared. As expected, the R-squared of the restricted model is lower than that of the unrestricted model because we have removed variables.

```python
# F statistic:
k = 5  # Number of independent variables in unrestricted model
q = 3  # Number of restrictions
fstat = (r2_ur - r2_r) / (1 - r2_ur) * (n - k - 1) / q
# Calculated F statistic
pd.DataFrame(
    {
        "Statistic": ["F-statistic"],
        "Value": [f"{fstat:.3f}"],
    },
)
```

This code calculates the $F$ statistic using the formula based on R-squared values.

```python
# CV for alpha=1% using the F distribution with 3 and 347 d.f.:
cv = stats.f.ppf(1 - 0.01, q, n - k - 1)  # Degrees of freedom (q, n-k-1)
# Critical value from F-distribution
pd.DataFrame(
    {
        "Distribution": ["F-distribution"],
        "df": [f"({q}, {n - k - 1})"],
        "Alpha": ["1%"],
        "Critical Value": [f"{cv:.3f}"],
    },
)
```

This calculates the critical value from the $F$ distribution with 3 and $n-k-1$ degrees of freedom for a 1% significance level.

```python
# p value = 1-cdf of the appropriate F distribution:
fpval = 1 - stats.f.cdf(fstat, q, n - k - 1)
# Calculated p-value
pd.DataFrame(
    {
        "Statistic": ["p-value"],
        "Value": [f"{fpval:.4f}"],
    },
)
```

This calculates the p-value for the $F$ test using the CDF of the $F$ distribution.

**Interpreting the results of the F-test for joint significance of batting stats:**

The calculated $F$ statistic is around 9.25. The p-value is very small (approximately 0.00003).  Comparing the F-statistic to the critical value (or the p-value to the significance level), we strongly reject the null hypothesis $H_0: \beta_3 = 0, \beta_4 = 0, \beta_5 = 0$.

**Conclusion:** We conclude that batting average, home runs per year, and runs batted in per year are jointly statistically significant determinants of baseball player salaries, even after controlling for years in the league and games played per year.  In other words, at least one of these batting statistics has a significant impact on salary.

```python
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

# Automated test results
pd.DataFrame(
    {
        "Statistic": ["F-statistic", "p-value"],
        "Value": [f"{fstat:.3f}", f"{fpval:.4f}"],
    },
)
```

This code demonstrates how to perform the same $F$ test using the `f_test()` method in `statsmodels`, which provides a more convenient way to conduct $F$ tests for linear restrictions. The results should be identical to our manual calculation, which they are (within rounding).

```python
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

# Complex hypotheses test results
pd.DataFrame(
    {
        "Statistic": ["F-statistic", "p-value"],
        "Value": [f"{fstat:.3f}", f"{fpval:.4f}"],
    },
)
```

This final example shows the flexibility of the `f_test()` method. Here, we test a different joint hypothesis: $H_0: \beta_{bavg} = 0 \text{ and } \beta_{hrunsyr} = 2\beta_{rbisyr}$. This is a more complex linear restriction involving a relationship between two coefficients. The `f_test()` method easily handles such hypotheses, demonstrating its power and convenience for testing various linear restrictions in multiple regression models.

**Interpreting the results of the second F-test:**

In this case, the F-statistic is around 4.57 and the p-value is approximately 0.0109. If we use a 1% significance level, we would fail to reject the null hypothesis. However, at a 5% significance level, we would reject $H_0$.  Therefore, there is moderate evidence against the null hypothesis that batting average has no effect and that the effect of home runs per year is twice the effect of runs batted in per year.  Further analysis or a larger dataset might be needed to draw firmer conclusions.

**Summary of F-tests:**

*   $F$ tests are used to test joint hypotheses about multiple regression coefficients.
*   They compare an unrestricted model to a restricted model that imposes the null hypothesis.
*   The $F$ statistic measures the relative increase in the sum of squared residuals (or decrease in R-squared) when moving from the unrestricted to the restricted model.
*   A large $F$ statistic (or small p-value) provides evidence against the null hypothesis.
*   $F$ tests are essential for testing the joint significance of sets of variables and for testing more complex linear restrictions on regression coefficients.

## 4.5 Reporting Regression Results

When presenting regression results in academic papers, policy reports, or data science projects, clarity and completeness are essential. This section discusses best practices for reporting regression output in a way that allows readers to understand and evaluate your analysis.

### 4.5.1 Essential Components of Regression Tables

A well-constructed regression table should include:

1. **Coefficient estimates** with appropriate precision (typically 3-4 significant digits)
2. **Standard errors** in parentheses below each coefficient (or t-statistics, clearly labeled)
3. **Significance indicators** (stars: *** for p < 0.01, ** for p < 0.05, * for p < 0.10)
4. **Sample size** (n)
5. **R-squared** and adjusted R-squared
6. **F-statistic** for overall model significance
7. **Degrees of freedom** for the residuals

**Example: Properly Formatted Regression Table**

```python
# Create a formatted regression table for the wage equation
wage1 = wool.data("wage1")
reg_wage = smf.ols("np.log(wage) ~ educ + exper + tenure", data=wage1).fit()

# Extract key statistics
coef_table = pd.DataFrame(
    {
        "Variable": ["Intercept", "educ", "exper", "tenure"],
        "Coefficient": reg_wage.params.values,
        "Std Error": reg_wage.bse.values,
        "t-statistic": reg_wage.tvalues.values,
        "p-value": reg_wage.pvalues.values,
        "Significance": [
            "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            for p in reg_wage.pvalues.values
        ],
    },
)

display(coef_table.round(4))

# Model statistics
model_stats = pd.DataFrame(
    {
        "Statistic": ["Observations", "R-squared", "Adjusted R-squared", "F-statistic", "Prob(F)"],
        "Value": [
            reg_wage.nobs,
            reg_wage.rsquared,
            reg_wage.rsquared_adj,
            reg_wage.fvalue,
            reg_wage.f_pvalue,
        ],
    },
)

display(model_stats.round(4))
```

:::{note} Significance Stars Convention
:class: dropdown

The convention for significance stars:
- `***` indicates p < 0.01 (significant at the 1% level)
- `**` indicates p < 0.05 (significant at the 5% level)  
- `*` indicates p < 0.10 (significant at the 10% level)
- No star means not statistically significant at conventional levels

Always include a note at the bottom of your table explaining this convention!
:::

### 4.5.2 Standard Errors vs t-Statistics

There are two common ways to report uncertainty:

**Option 1: Standard Errors (more common)**
```
Coefficient:  0.092***
             (0.007)
```

**Option 2: t-Statistics**
```
Coefficient:  0.092***
             [13.14]
```

**Most econometric journals prefer standard errors** because they allow readers to:
- Construct confidence intervals at any confidence level
- Test hypotheses against any null value (not just zero)
- Better assess economic vs statistical significance

### 4.5.3 Comparing Multiple Specifications

Often you'll want to present several model specifications side-by-side to show robustness of results or the effect of adding controls:

```python
# Estimate three specifications with progressively more controls
gpa1 = wool.data("gpa1")

# Model 1: Simple regression
m1 = smf.ols("colGPA ~ hsGPA", data=gpa1).fit()

# Model 2: Add ACT score
m2 = smf.ols("colGPA ~ hsGPA + ACT", data=gpa1).fit()

# Model 3: Full model with skipped classes
m3 = smf.ols("colGPA ~ hsGPA + ACT + skipped", data=gpa1).fit()

# Create comparison table
comparison_table = pd.DataFrame(
    {
        "Variable": ["Intercept", "hsGPA", "ACT", "skipped"],
        "Model 1": [f"{m1.params[0]:.3f}", f"{m1.params[1]:.3f}***", "-", "-"],
        "Model 2": [
            f"{m2.params[0]:.3f}",
            f"{m2.params[1]:.3f}***",
            f"{m2.params[2]:.3f}",
            "-",
        ],
        "Model 3": [
            f"{m3.params[0]:.3f}",
            f"{m3.params[1]:.3f}***",
            f"{m3.params[2]:.3f}",
            f"{m3.params[3]:.3f}***",
        ],
    },
)

display(comparison_table)

# Add model statistics
stats_table = pd.DataFrame(
    {
        "Statistic": ["N", "R-squared", "Adj R-squared"],
        "Model 1": [m1.nobs, f"{m1.rsquared:.3f}", f"{m1.rsquared_adj:.3f}"],
        "Model 2": [m2.nobs, f"{m2.rsquared:.3f}", f"{m2.rsquared_adj:.3f}"],
        "Model 3": [m3.nobs, f"{m3.rsquared:.3f}", f"{m3.rsquared_adj:.3f}"],
    },
)

display(stats_table)
```

**Interpretation**: Comparing multiple models shows:
- How coefficient estimates change when adding controls
- Whether the variable of interest is robust to different specifications
- How R-squared improves with additional variables
- Whether new variables are jointly significant

### 4.5.4 What NOT to Report

**Avoid these common mistakes:**

1. **Don't report too many decimal places**: 0.0912847323 -> 0.092
2. **Don't omit standard errors**: Readers can't assess significance without them
3. **Don't forget to specify the dependent variable**: Always clearly state what you're modeling
4. **Don't report correlation matrices as regressions**: Show actual regression output
5. **Don't neglect to describe your sample**: State the dataset, time period, and any sample restrictions

:::{warning} Publication Standards
:class: dropdown

Most economics journals require:
- Robust standard errors (heteroskedasticity-consistent)
- Clear indication of which variables are controls vs variables of interest  
- Discussion of economic significance, not just statistical significance
- Sensitivity analysis or robustness checks
- Full replication data and code (increasingly common)

Always check journal-specific requirements!
:::

## 4.6 Revisiting Causal Effects and Policy Analysis

Throughout this chapter, we've focused on statistical inference - testing hypotheses and constructing confidence intervals. But the ultimate goal of regression analysis in economics and policy evaluation is often to estimate **causal effects**. This section revisits the conditions under which regression estimates can be interpreted causally and discusses the limitations.

### 4.6.1 From Association to Causation

**Key Distinction:**
- **Association**: $x$ and $y$ are correlated
- **Causation**: Changes in $x$ **cause** changes in $y$

**Statistical significance alone does NOT imply causation!**

A coefficient can be:
- Statistically significant (t-statistic > 2, p-value < 0.05)
- Precisely estimated (narrow confidence interval)
- **Yet still not represent a causal effect**

### 4.6.2 The Fundamental Problem: Omitted Variable Bias

Recall from Chapter 3 that for OLS to estimate a causal effect, we need:

$$E(u | x_1, x_2, \ldots, x_k) = 0$$

This means the error term must be **uncorrelated with all included variables**.

**This fails when:**
1. **Omitted variables** affect both treatment and outcome (confounding)
2. **Simultaneity**: $y$ affects $x$ while $x$ affects $y$ (reverse causality)
3. **Measurement error** in key variables

**Example: Returns to Education - Causal or Not?**

Consider the wage equation:
$$\log(\text{wage}) = \beta_0 + \beta_1 \text{educ} + \beta_2 \text{exper} + \beta_3 \text{tenure} + u$$

```python
# Estimate wage equation
wage1 = wool.data("wage1")
wage_reg = smf.ols("np.log(wage) ~ educ + exper + tenure", data=wage1).fit()

# Display results
results_df = pd.DataFrame(
    {
        "Variable": ["educ", "exper", "tenure"],
        "Coefficient": wage_reg.params[1:].values,
        "Std Error": wage_reg.bse[1:].values,
        "95% CI Lower": wage_reg.conf_int().iloc[1:, 0].values,
        "95% CI Upper": wage_reg.conf_int().iloc[1:, 1].values,
        "Interpretation": [
            "1 year educ -> 9.2% higher wage",
            "1 year exper -> 0.4% higher wage",
            "1 year tenure -> 1.7% higher wage",
        ],
    },
)

display(results_df.round(4))
```

**Question**: Does $\hat{\beta}_1 = 0.092$ represent the **causal effect** of education on wages?

**Potential Problems:**
1. **Ability bias**: Unobserved ability affects both education choices and wages
   - $\text{Cov}(\text{ability}, \text{educ}) > 0$ (smarter people get more education)
   - $\text{Cov}(\text{ability}, \text{wage}) > 0$ (smarter people earn higher wages)
   - Result: $\hat{\beta}_1$ **overestimates** the causal effect

2. **Family background**: Wealthy families provide both more education and job connections
   - Omitted variable creates upward bias

3. **Measurement error** in education: Survey responses may be inaccurate
   - Creates **attenuation bias** (underestimates true effect)

**Conclusion**: The coefficient 0.092 represents an **association**, not necessarily a causal effect. The true causal effect might be smaller if ability bias dominates.

### 4.6.3 When Can We Claim Causality?

To credibly claim a causal interpretation, you need one of the following:

**1. Randomized Controlled Trials (RCTs) - The Gold Standard**

- Treatment is **randomly assigned** to subjects
- Randomization ensures $E(u | \text{treatment}) = 0$
- Example: Job training program with random selection

**2. Natural Experiments**

- Some external event creates "as-if random" variation
- Example: Draft lottery, policy changes affecting some regions but not others
- Requires careful argument that assignment is exogenous

**3. Instrumental Variables (Chapter 15)**

- Find a variable (instrument) that affects treatment but not outcome directly
- Example: Distance to college as instrument for education
- Requires strong assumptions (instrument exogeneity, relevance)

**4. Regression Discontinuity Design**

- Treatment assignment based on threshold of a running variable
- Compare units just above vs just below the threshold
- Example: Scholarship eligibility based on test score cutoff

**5. Difference-in-Differences**

- Compare changes over time between treatment and control groups
- Example: Minimum wage increase in one state but not neighboring state
- Requires parallel trends assumption

### 4.6.4 Policy Analysis Example: Minimum Wage and Employment

```python
# Simulate a policy analysis scenario
np.random.seed(1234)
n_states = 50
n_years = 10

# Create panel data
states = np.repeat(np.arange(1, n_states + 1), n_years)
years = np.tile(np.arange(2010, 2020), n_states)

# State fixed effects (some states have higher employment naturally)
state_effects = np.repeat(stats.norm.rvs(50, 10, size=n_states), n_years)

# Treatment: Some states raise minimum wage in 2015
treatment = ((states <= 25) & (years >= 2015)).astype(int)

# Outcome: Employment rate
# True causal effect: min wage reduces employment by 2 percentage points
employment = state_effects + 2 * years - 2 * treatment + stats.norm.rvs(0, 5, size=n_states * n_years)

policy_data = pd.DataFrame(
    {"state": states, "year": years, "min_wage_increase": treatment, "employment": employment}
)

# Naive regression (WRONG - omits state fixed effects)
naive_reg = smf.ols("employment ~ min_wage_increase", data=policy_data).fit()

# Correct regression with state fixed effects
correct_reg = smf.ols("employment ~ min_wage_increase + C(state)", data=policy_data).fit()

# Compare results
comparison = pd.DataFrame(
    {
        "Model": ["Naive (No Controls)", "With State Fixed Effects", "True Causal Effect"],
        "Min Wage Coefficient": [
            naive_reg.params["min_wage_increase"],
            correct_reg.params["min_wage_increase"],
            -2.0,
        ],
        "Std Error": [
            naive_reg.bse["min_wage_increase"],
            correct_reg.bse["min_wage_increase"],
            "-",
        ],
        "Interpretation": [
            "BIASED (omitted state effects)",
            "Unbiased estimate",
            "True parameter",
        ],
    },
)

display(comparison.round(3))
```

**Lesson**: Without controlling for confounding factors (state fixed effects), the naive regression gives a **biased estimate** of the policy effect. Proper causal inference requires thinking carefully about what variables to include and what identification strategy to use.

### 4.6.5 Practical Guidelines for Causal Claims

**DO**:
- ✓ State your identification assumptions explicitly
- ✓ Discuss potential sources of bias
- ✓ Conduct robustness checks with alternative specifications
- ✓ Compare your estimates to previous research
- ✓ Be honest about limitations

**DON'T**:
- ✗ Claim causality based solely on statistical significance
- ✗ Ignore obvious omitted variables
- ✗ Overstate your findings
- ✗ Cherry-pick specifications that support your hypothesis
- ✗ Forget that "controlling for X" doesn't solve endogeneity

:::{important} The Gold Standard
:class: dropdown

**Causal inference is hard!** 

Even with all the right controls and a statistically significant coefficient, you may only have established an **association**. True causal identification requires:
1. Careful economic reasoning about potential confounders
2. Credible identification strategy (RCT, IV, RD, DID, etc.)
3. Transparent reporting of assumptions and limitations
4. Robustness checks to alternative specifications

When in doubt, use cautious language: "associated with" rather than "causes" or "affects."
:::

## Chapter Summary

This chapter has provided a comprehensive exploration of statistical inference in multiple regression analysis. We've covered the essential tools for testing hypotheses, constructing confidence intervals, and evaluating the significance of our regression estimates.

**Key Concepts Covered:**

**1. Classical Linear Model Assumptions**: For exact inference in finite samples, we require the Gauss-Markov assumptions (MLR.1-MLR.5) plus normality of the error term (MLR.6). Under these assumptions, the OLS estimators follow a t-distribution, enabling hypothesis testing and confidence interval construction.

**2. The t Test**: We use t tests to test hypotheses about individual regression coefficients. The t statistic is calculated as:
$$t = \frac{\hat{\beta}_j - a_j}{\text{se}(\hat{\beta}_j)}$$
where $a_j$ is the hypothesized value (usually zero). We reject the null hypothesis if $|t| > c$ where $c$ is the critical value from the t-distribution.

**3. p-values**: The p-value represents the probability of observing a test statistic as extreme as the one calculated, assuming the null hypothesis is true. Small p-values (typically < 0.05) provide evidence against the null hypothesis.

**4. Confidence Intervals**: A 95% confidence interval for $\beta_j$ is:
$$\hat{\beta}_j \pm t_{0.025} \cdot \text{se}(\hat{\beta}_j)$$
This interval contains the true parameter value in 95% of samples.

**5. F Tests**: F tests allow us to test joint hypotheses about multiple coefficients simultaneously. The F statistic compares the fit of an unrestricted model to a restricted model that imposes the null hypothesis constraints.

**6. Statistical vs Economic Significance**: A coefficient can be statistically significant (different from zero) yet economically insignificant (too small to matter in practice). Always interpret both the magnitude and the precision of estimates.

**7. Reporting Regression Results**: Proper regression tables should include coefficient estimates, standard errors, significance indicators, sample size, R-squared, and model fit statistics. Present multiple specifications to demonstrate robustness.

**8. Causal Inference Challenges**: Statistical significance does not imply causation. To make causal claims, we need credible identification strategies that address omitted variable bias, reverse causality, and measurement error. Randomized experiments provide the gold standard, but quasi-experimental methods (natural experiments, instrumental variables, regression discontinuity, difference-in-differences) can also establish causality under appropriate assumptions.

**Practical Applications:**

The tools developed in this chapter are fundamental for:
- Hypothesis testing in economic research
- Policy evaluation and impact assessment  
- Forecasting and prediction with quantified uncertainty
- Model comparison and specification testing
- Drawing causal inferences from observational data

**Looking Forward:**

In Chapter 5, we'll explore the **asymptotic properties** of OLS estimators, which allow us to make valid inferences even when the normality assumption fails or the sample is large. We'll also discuss **consistency**, **asymptotic normality**, and **large sample inference**, providing tools for situations where the classical linear model assumptions are violated.

**Critical Reminder**: Always think carefully about the **economic interpretation** and **causal validity** of your regression estimates. Statistical techniques are powerful tools, but they cannot substitute for sound economic reasoning and careful identification of causal relationships.
