# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,markdown//md,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: merino
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 3. Multiple Regression Analysis: Estimation
#
# This notebook delves into the estimation of multiple regression models. Building upon the foundation of simple linear regression, we will explore how to analyze the relationship between a dependent variable and *multiple* independent variables simultaneously. This is crucial in real-world scenarios where outcomes are rarely determined by a single factor. We'll use Python libraries like `statsmodels` and `pandas` along with datasets from the `wooldridge` package to illustrate these concepts with practical examples.
#
#

# %%
import numpy as np
import pandas as pd
import patsy as pt
import statsmodels.formula.api as smf
import statsmodels.stats.outliers_influence as smo
import wooldridge as wool

# %% [markdown]
# ## 3.1 Multiple Regression in Practice
#
# In multiple regression analysis, we extend the simple regression model to incorporate more than one explanatory variable. This allows us to control for various factors that might influence the dependent variable and isolate the effect of each independent variable, holding others constant.
#
# The general form of a multiple linear regression model is:
#
# $$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 +\beta_3 x_3 + \cdots + \beta_k x_k + u$$
#
# Where:
#
# *   $y$ is the dependent variable (the variable we want to explain).
# *   $x_1, x_2, \ldots, x_k$ are the independent variables (or regressors, explanatory variables) that we believe influence $y$.
# *   $\beta_0$ is the intercept, representing the expected value of $y$ when all independent variables are zero.
# *   $\beta_1, \beta_2, \ldots, \beta_k$ are the partial regression coefficients. Each $\beta_j$ represents the change in $y$ for a one-unit increase in $x_j$, *holding all other independent variables constant*. This is the crucial **ceteris paribus** interpretation in multiple regression.
# *   $u$ is the error term (or disturbance), representing unobserved factors that also affect $y$.
#
# Let's explore several examples from the textbook to understand how multiple regression is applied in practice.
#
# ### Example 3.1: Determinants of College GPA
#
# **Question:** What factors influence a student's college GPA?  We might hypothesize that a student's high school GPA (`hsGPA`) and their score on a standardized test like the ACT (`ACT`) are important predictors of college performance.
#
# **Model:** We can formulate a multiple regression model to investigate this:
#
# $$\text{colGPA} = \beta_0 + \beta_1 \text{hsGPA} + \beta_2 \text{ACT} + u$$
#
# *   `colGPA`: College Grade Point Average (dependent variable)
# *   `hsGPA`: High School Grade Point Average (independent variable)
# *   `ACT`: ACT score (independent variable)
#
# We expect $\beta_1 > 0$ and $\beta_2 > 0$, suggesting that higher high school GPA and ACT scores are associated with higher college GPA, holding the other factor constant.

# %%
gpa1 = wool.data("gpa1")

reg = smf.ols(formula="colGPA ~ hsGPA + ACT", data=gpa1)
results = reg.fit()
print(f"results.summary(): \n{results.summary()}\n")

# %% [markdown]
# **Interpreting the Results:**
#
# The `results.summary()` output from `statsmodels` provides a wealth of information. Let's focus on the estimated coefficients:
#
# *   **Intercept ($\beta_0$):**  The estimated intercept is 1.2863. This is the predicted college GPA when both high school GPA and ACT score are zero, which is not practically meaningful in this context but is a necessary part of the model.
# *   **hsGPA ($\beta_1$):** The estimated coefficient for `hsGPA` is 0.4535. This means that, holding ACT score constant, a one-point increase in high school GPA is associated with a 0.4535 point increase in college GPA.
# *   **ACT ($\beta_2$):** The estimated coefficient for `ACT` is 0.0094.  This means that, holding high school GPA constant, a one-point increase in ACT score is associated with a 0.0094 point increase in college GPA.
#
# **Key takeaway:** Multiple regression allows us to examine the effect of each variable while controlling for the others. For instance, the coefficient on `hsGPA` (0.447) is the estimated effect of `hsGPA` *after* accounting for the influence of `ACT`.
#
# ### Example 3.3 Hourly Wage Equation
#
# **Question:** What factors determine an individual's hourly wage?  Education, experience, and job tenure are commonly believed to be important determinants.
#
# **Model:** We can model the logarithm of wage as a function of education, experience, and tenure:
#
# $$ \log(\text{wage}) = \beta_0 + \beta_1 \text{educ} + \beta_2 \text{exper} + \beta_3 \text{tenure} + u$$
#
# *   $\log(\text{wage})$: Natural logarithm of hourly wage (dependent variable). Using the log of wage is common in economics as it often leads to a more linear relationship and allows for percentage change interpretations of coefficients.
# *   `educ`: Years of education (independent variable)
# *   `exper`: Years of work experience (independent variable)
# *   `tenure`: Years with current employer (independent variable)
#
# In this model, coefficients on `educ`, `exper`, and `tenure` will represent the approximate percentage change in wage for a one-unit increase in each respective variable, holding the others constant. For example, $\beta_1 \approx \% \Delta \text{wage} / \Delta \text{educ}$.

# %%
wage1 = wool.data("wage1")

reg = smf.ols(formula="np.log(wage) ~ educ + exper + tenure", data=wage1)
results = reg.fit()
print(f"results.summary(): \n{results.summary()}\n")

# %% [markdown]
# **Interpreting the Results:**
#
# *   **educ ($\beta_1$):** The coefficient for `educ` is 0.0920. This suggests that, holding experience and tenure constant, an additional year of education is associated with an approximate 9.20% increase in hourly wage.
# *   **exper ($\beta_2$):** The coefficient for `exper` is 0.0041.  Holding education and tenure constant, an additional year of experience is associated with an approximate 0.41% increase in hourly wage. The effect of experience seems to be smaller than education in this model.
# *   **tenure ($\beta_3$):** The coefficient for `tenure` is 0.0221. Holding education and experience constant, an additional year of tenure with the current employer is associated with an approximate 2.21% increase in hourly wage. Tenure appears to have a larger impact than general experience in this model, possibly reflecting firm-specific skills or returns to seniority.
#
# ### Example 3.4: Participation in 401(k) Pension Plans
#
# **Question:** What factors influence the participation rate in 401(k) pension plans among firms? Let's consider the firm's match rate and the age of the firm's employees.
#
# **Model:**
#
# $$ \text{prate} = \beta_0 + \beta_1 \text{mrate} + \beta_2 \text{age} + u$$
#
# *   `prate`: Participation rate in 401(k) plans (percentage of eligible workers participating) (dependent variable)
# *   `mrate`: Firm's match rate (e.g., if mrate=0.5, firm matches 50 cents for every dollar contributed by employee) (independent variable)
# *   `age`: Average age of firm's employees (independent variable)
#
# We expect $\beta_1 > 0$ because a higher match rate should encourage participation. The expected sign of $\beta_2$ is less clear. Older workforces might have had more time to enroll in 401(k)s, or they may be closer to retirement and thus more interested in pension plans. Conversely, younger firms might be more proactive in encouraging enrollment.

# %%
k401k = wool.data("401k")

reg = smf.ols(formula="prate ~ mrate + age", data=k401k)
results = reg.fit()
print(f"results.summary(): \n{results.summary()}\n")

# %% [markdown]
# **Interpreting the Results:**
#
# *   **mrate ($\beta_1$):** The coefficient for `mrate` is 5.5213. This indicates that for a one-unit increase in the match rate (e.g., increasing the match from 0.0 to 1.0, meaning the firm matches dollar for dollar), the participation rate is estimated to increase by about 5.52 percentage points, holding average employee age constant.
# *   **age ($\beta_2$):** The coefficient for `age` is 0.2431.  Holding the match rate constant, a one-year increase in the average age of employees is associated with a 0.2431 percentage point increase in the participation rate. This suggests a slightly positive relationship between average employee age and 401(k) participation.
#
# ### Example 3.5a: Explaining Arrest Records
#
# **Question:** Can we explain the number of arrests a young man has in 1986 based on his criminal history and employment status?
#
# **Model (without average sentence length):**
#
# $$\text{narr86} = \beta_0 + \beta_1 \text{pcnv} + \beta_2 \text{ptime86} + \beta_3 \text{qemp86} + u$$
#
# *   `narr86`: Number of arrests in 1986 (dependent variable)
# *   `pcnv`: Proportion of prior arrests that led to conviction (independent variable)
# *   `ptime86`: Months spent in prison in 1986 (independent variable)
# *   `qemp86`: Number of quarters employed in 1986 (independent variable)
#
# We expect $\beta_1 > 0$ because a higher conviction rate might deter future crime. We also expect $\beta_2 > 0$ as spending more time in prison in 1986 means more opportunity to be arrested in 1986 (although this might be complex).  We expect $\beta_3 < 0$ because employment should reduce the likelihood of arrests.

# %%
crime1 = wool.data("crime1")

# model without avgsen:
reg = smf.ols(formula="narr86 ~ pcnv + ptime86 + qemp86", data=crime1)
results = reg.fit()
print(f"results.summary(): \n{results.summary()}\n")

# %% [markdown]
# **Interpreting the Results (Model 3.5a):**
#
# *   **pcnv ($\beta_1$):** The coefficient for `pcnv` is -0.1499. A higher proportion of prior convictions is associated with a lower number of arrests in 1986, holding prison time and employment constant.
# *   **ptime86 ($\beta_2$):** The coefficient for `ptime86` is -0.0344. More months spent in prison in 1986 is associated with a lower number of arrests in 1986, controlling for conviction proportion and employment.
# *   **qemp86 ($\beta_3$):** The coefficient for `qemp86` is -0.1041.  For each additional quarter of employment in 1986, the number of arrests in 1986 is estimated to decrease by 0.1041, holding other factors constant. Employment appears to have a deterrent effect on arrests.
#
# ### Example 3.5b: Explaining Arrest Records
#
# **Model (with average sentence length):** Let's add another variable, `avgsen`, the average sentence served from prior convictions, to the model to see if it influences arrests in 1986.
#
# $$\text{narr86} = \beta_0 + \beta_1 \text{pcnv} + \beta_2 \text{avgsen} + \beta_3 \text{ptime86} + \beta_4 \text{qemp86} + u$$
#
# *   `avgsen`: Average sentence served from prior convictions (in months) (independent variable). We expect $\beta_2 < 0$ if longer average sentences deter crime.

# %%
crime1 = wool.data("crime1")

# model with avgsen:
reg = smf.ols(formula="narr86 ~ pcnv + avgsen + ptime86 + qemp86", data=crime1)
results = reg.fit()
print(f"results.summary(): \n{results.summary()}\n")

# %% [markdown]
# **Interpreting the Results (Model 3.5b):**
#
# Comparing this to Model 3.5a, we see:
#
# *   **avgsen ($\beta_2$):** The coefficient for `avgsen` is 0.0074.  While positive (longer average sentences associated with more arrests), the coefficient is very small and statistically insignificant (p-value = 0.116). This suggests that average sentence length, in this model and dataset, does not have a strong, statistically significant deterrent effect on arrests in 1986, once we control for other factors.
# *   **Changes in other coefficients:** Notice that the coefficients for `pcnv`, `ptime86`, and `qemp86` have slightly changed compared to Model 3.5a. This is a common occurrence in multiple regression when you add or remove regressors.  The coefficient on `qemp86` is still negative and statistically significant, suggesting that employment remains a relevant factor.
#
# **Comparison of 3.5a and 3.5b:** Adding `avgsen` did not substantially change the findings regarding `pcnv`, `ptime86`, and `qemp86`.  The variable `avgsen` itself was not found to be statistically significant. This highlights the importance of considering multiple potential determinants and testing their individual and collective effects within a multiple regression framework.
#
# ### Example 3.6: Hourly Wage Equation (Simple Regression)
#
# **Question:** What happens if we omit relevant variables from our wage equation? Let's revisit the wage equation but only include education as an explanatory variable.
#
# **Model (Simple Regression):**
#
# $$ \log(\text{wage}) = \beta_0 + \beta_1 \text{educ} + u$$

# %%
wage1 = wool.data("wage1")

reg = smf.ols(formula="np.log(wage) ~ educ", data=wage1)
results = reg.fit()
print(f"results.summary(): \n{results.summary()}\n")

# %% [markdown]
# **Interpreting the Results and Comparing to Example 3.3:**
#
# *   **educ ($\beta_1$):** In this simple regression model, the coefficient for `educ` is approximately 0.108. This is slightly larger than the coefficient for `educ` (0.092) in the multiple regression model (Example 3.3) that included `exper` and `tenure`.
#
# **Omitted Variable Bias (Preview):**  The difference in the coefficient for `educ` between the simple regression and the multiple regression illustrates the concept of **omitted variable bias**. By omitting `exper` and `tenure`, which are likely correlated with both `educ` and `log(wage)`, we might be incorrectly attributing some of the effect of experience and tenure to education in the simple regression.  We will formally explore omitted variable bias in Section 3.3.
#
# ## 3.2 OLS in Matrix Form
#
# While `statsmodels` handles the calculations for us, it's crucial to understand the underlying mechanics of Ordinary Least Squares (OLS) estimation, especially in the context of multiple regression.  Matrix algebra provides a concise and powerful way to represent and solve the OLS problem for multiple regression.
#
# In matrix form, the multiple regression model can be written as:
#
# $$y = X\beta + u$$
#
# Where:
#
# *   $y$ is an $(n \times 1)$ column vector of the dependent variable observations.
# *   $X$ is an $(n \times (k+1))$ matrix of independent variable observations, called the **design matrix**. The first column of $X$ is typically a column of ones (for the intercept), and the subsequent columns are the observations of the independent variables $x_1, x_2, \ldots, x_k$.
# *   $\beta$ is a $((k+1) \times 1)$ column vector of the unknown regression coefficients $(\beta_0, \beta_1, \ldots, \beta_k)'$.
# *   $u$ is an $(n \times 1)$ column vector of the error terms.
#
# The OLS estimator $\hat{\beta}$ that minimizes the sum of squared residuals can be expressed in matrix form as:
#
# $$\hat{\beta} = (X'X)^{-1}X'y$$
#
# Where:
#
# *   $X'$ is the transpose of the matrix $X$.
# *   $(X'X)^{-1}$ is the inverse of the matrix product $X'X$.
#
# Let's demonstrate this matrix calculation using the `gpa1` dataset from Example 3.1.

# %%
gpa1 = wool.data("gpa1")

# determine sample size & no. of regressors:
n = len(gpa1)
k = 2  # Number of independent variables (hsGPA and ACT)

# extract y:
y = gpa1["colGPA"]

# extract X & add a column of ones:
X = pd.DataFrame({"const": 1, "hsGPA": gpa1["hsGPA"], "ACT": gpa1["ACT"]})

# alternative with patsy (more streamlined for formula-based design matrices):
y2, X2 = pt.dmatrices("colGPA ~ hsGPA + ACT", data=gpa1, return_type="dataframe")

# display first few rows of X:
print(f"X.head(): \n{X.head()}\n")

# %% [markdown]
# The code above constructs the $X$ matrix and $y$ vector from the `gpa1` data.  The `patsy` library provides a convenient way to create design matrices directly from formulas, which is often more efficient for complex models.

# %%
# parameter estimates using matrix formula:
X = np.array(X)  # Convert pandas DataFrame to numpy array for matrix operations
y = np.array(y).reshape(n, 1)  # Reshape y to be a column vector (n x 1)
b = np.linalg.inv(X.T @ X) @ X.T @ y  # Matrix formula for OLS estimator
print(f"b (estimated coefficients):\n{b}\n")

# %% [markdown]
# This code performs the matrix operations to calculate $\hat{\beta}$. The result `b` should match the coefficients we obtained from `statsmodels` earlier.
#
# After estimating $\hat{\beta}$, we can calculate the residuals $\hat{u}$:
#
# $$\hat{u} = y - X\hat{\beta}$$
#
# And the estimator for the error variance $\sigma^2$:
#
# $$\hat{\sigma}^2 = \frac{1}{n-k-1} \hat{u}'\hat{u}$$
#
# The denominator $(n-k-1)$ represents the degrees of freedom in multiple regression, where $n$ is the sample size and $(k+1)$ is the number of parameters estimated (including the intercept).  The square root of $\hat{\sigma}^2$ is the Standard Error of the Regression (SER).

# %%
# residuals, estimated variance of u and SER:
u_hat = y - X @ b  # Calculate residuals
sigsq_hat = (u_hat.T @ u_hat) / (n - k - 1)  # Estimated error variance
SER = np.sqrt(sigsq_hat)  # Standard Error of Regression
print(f"SER: {SER}\n")

# %% [markdown]
# Finally, the estimated variance-covariance matrix of the OLS estimator $\hat{\beta}$ is given by:
#
# $$\widehat{\text{var}(\hat{\beta})} = \hat{\sigma}^2 (X'X)^{-1}$$
#
# The standard errors for each coefficient are the square roots of the diagonal elements of this variance-covariance matrix.

# %%
# estimated variance-covariance matrix of beta_hat and standard errors:
Vbeta_hat = sigsq_hat * np.linalg.inv(X.T @ X)  # Variance-covariance matrix
se = np.sqrt(np.diagonal(Vbeta_hat))  # Standard errors (diagonal elements' square root)
print(f"se (standard errors):\n{se}\n")

# %% [markdown]
# The manually calculated coefficients (`b`), SER, and standard errors (`se`) should be very close (or identical, considering potential rounding differences) to the values reported in the `results.summary()` output from `statsmodels` for Example 3.1. This demonstrates that `statsmodels` is using the matrix-based OLS formulas under the hood.
#
# ## 3.3 Ceteris Paribus Interpretation and Omitted Variable Bias
#
# A key advantage of multiple regression is its ability to provide **ceteris paribus** interpretations of the coefficients.  "Ceteris paribus" is Latin for "other things being equal" or "holding other factors constant." In the context of multiple regression, the coefficient on a particular independent variable represents the effect of that variable on the dependent variable *while holding all other included independent variables constant*.
#
# However, if we **omit** a relevant variable from our regression model, and this omitted variable is correlated with the included independent variables, we can encounter **omitted variable bias**. This means that the estimated coefficients on the included variables will be biased and inconsistent, and they will no longer have the desired ceteris paribus interpretation with respect to the omitted variable.
#
# Let's revisit the college GPA example to illustrate omitted variable bias. Suppose the "true" model is:
#
# $$\text{colGPA} = \beta_0 + \beta_1 \text{hsGPA} + \beta_2 \text{ACT} + u$$
#
# But we mistakenly estimate a simple regression model, omitting `hsGPA`:
#
# $$\text{colGPA} = \gamma_0 + \gamma_1 \text{ACT} + v$$
#
# If `hsGPA` is correlated with `ACT` (which is likely - students with higher high school GPAs tend to score higher on standardized tests), then the estimated coefficient $\hat{\gamma}_1$ in the simple regression model will be biased. It will capture not only the direct effect of `ACT` on `colGPA` but also the indirect effect of `hsGPA` on `colGPA` that is correlated with `ACT`.
#
# Let's see this empirically. First, we estimate the "full" model (including both `hsGPA` and `ACT`):

# %%
gpa1 = wool.data("gpa1")

# parameter estimates for full model:
reg = smf.ols(
    formula="colGPA ~ ACT + hsGPA",
    data=gpa1,
)  # Order of regressors doesn't matter in formula
results = reg.fit()
b = results.params  # Extract estimated coefficients
print(f"Coefficients from full model (b):\n{b}\n")

# %% [markdown]
# Now, let's consider the relationship between the included variable (`ACT`) and the omitted variable (`hsGPA`). We can regress the omitted variable (`hsGPA`) on the included variable (`ACT`):

# %%
# relation between regressors (hsGPA on ACT):
reg_delta = smf.ols(formula="hsGPA ~ ACT", data=gpa1)
results_delta = reg_delta.fit()
delta_tilde = results_delta.params  # Extract coefficient of ACT in this regression
print(f"Coefficients from regression of hsGPA on ACT (delta_tilde):\n{delta_tilde}\n")

# %% [markdown]
# $\delta_{\text{ACT}}$ from this regression represents how much `hsGPA` changes on average for a one-unit change in `ACT`.  It captures the correlation between `hsGPA` and `ACT`.
#
# The **omitted variable bias formula** provides an approximation for the bias in the simple regression coefficient $\hat{\gamma}_1$ when we omit `hsGPA`.  In this case, the bias in the coefficient of `ACT` (when `hsGPA` is omitted) is approximately:
#
# $$\text{Bias}(\hat{\gamma}_1) \approx \beta_1 \times \delta_{\text{ACT}}$$
#
# Where:
#
# *   $\beta_1$ is the coefficient of the omitted variable (`hsGPA`) in the full model.
# *   $\delta_{\text{ACT}}$ is the coefficient of the included variable (`ACT`) in the regression of the omitted variable (`hsGPA`) on the included variable (`ACT`).
#
# Let's calculate this approximate bias and see how it relates to the difference between the coefficient of `ACT` in the full model ($\beta_2$) and the coefficient of `ACT` in the simple model ($\gamma_1$).

# %%
# omitted variables formula for b1_tilde (approximate bias in ACT coefficient when hsGPA is omitted):
b1_tilde = b["ACT"] + b["hsGPA"] * delta_tilde["ACT"]  # Applying the bias formula
print(f"Approximate biased coefficient of ACT (b1_tilde):\n{b1_tilde}\n")

# %% [markdown]
# Finally, let's estimate the simple regression model (omitting `hsGPA`) and see the actual coefficient of `ACT`:

# %%
# actual regression with hsGPA omitted (simple regression):
reg_om = smf.ols(formula="colGPA ~ ACT", data=gpa1)
results_om = reg_om.fit()
b_om = results_om.params  # Extract coefficient of ACT from simple regression
print(f"Coefficient of ACT in simple regression (b_om):\n{b_om}\n")

# %% [markdown]
# Comparing `b_om["ACT"]` (the coefficient of ACT in the simple regression) with `b["ACT"]` (the coefficient of ACT in the full regression), we can see that they are different.  Furthermore, `b1_tilde` (the approximate biased coefficient calculated using the formula) is close to `b_om["ACT"]`.
#
# **Conclusion on Omitted Variable Bias:** Omitting a relevant variable like `hsGPA` that is correlated with an included variable like `ACT` can lead to biased estimates. In this case, the coefficient on `ACT` in the simple regression is larger than in the multiple regression, likely because it is picking up some of the positive effect of `hsGPA` on `colGPA`.  This highlights the importance of including all relevant variables in a regression model to obtain unbiased and consistent estimates and to achieve correct ceteris paribus interpretations.
#
# ## 3.4 Standard Errors, Multicollinearity, and VIF
#
# In multiple regression, we not only estimate coefficients but also need to assess their precision. **Standard errors (SEs)** of the estimated coefficients are crucial for conducting hypothesis tests and constructing confidence intervals.  Larger standard errors indicate less precise estimates.
#
# The standard error of a coefficient in multiple regression depends on several factors, including:
#
# *   **Sample size (n):** Larger sample sizes generally lead to smaller standard errors (more precision).
# *   **Error variance ($\sigma^2$):** Higher error variance (more noise in the data) leads to larger standard errors.
# *   **Total sample variation in the independent variable ($x_j$):** More variation in $x_j$ (holding other regressors constant) leads to smaller standard errors for $\beta_j$.
# *   **Correlation among independent variables (Multicollinearity):** Higher correlation among independent variables (multicollinearity) generally leads to larger standard errors.
#
# **Multicollinearity** arises when two or more independent variables in a regression model are highly linearly correlated.  While multicollinearity does not bias the OLS coefficient estimates, it increases their standard errors, making it harder to precisely estimate the individual effects of the correlated variables and potentially leading to statistically insignificant coefficients even if the variables are truly important.
#
# The **Variance Inflation Factor (VIF)** is a common measure of multicollinearity.  For each independent variable $x_j$, the VIF is calculated as:
#
# $$VIF_j = \frac{1}{1 - R_j^2}$$
#
# Where $R_j^2$ is the R-squared from regressing $x_j$ on all *other* independent variables in the model.
#
# *   A VIF of 1 indicates no multicollinearity.
# *   VIFs greater than 1 indicate increasing multicollinearity.
# *   As a rule of thumb, VIFs greater than 10 are often considered to indicate high multicollinearity, although there is no strict cutoff.
#
# Let's illustrate the calculation of standard errors and VIF using the `gpa1` example again.

# %%
gpa1 = wool.data("gpa1")

# full estimation results including automatic SE from statsmodels:
reg = smf.ols(formula="colGPA ~ hsGPA + ACT", data=gpa1)
results = reg.fit()

# Extract SER (Standard Error of Regression) directly from results object:
SER = np.sqrt(
    results.mse_resid,
)  # mse_resid is Mean Squared Error of Residuals (estimated sigma^2)

# Regress hsGPA on ACT to calculate R-squared for VIF of hsGPA:
reg_hsGPA = smf.ols(formula="hsGPA ~ ACT", data=gpa1)
results_hsGPA = reg_hsGPA.fit()
R2_hsGPA = results_hsGPA.rsquared  # R-squared from this auxiliary regression
VIF_hsGPA = 1 / (1 - R2_hsGPA)  # Calculate VIF for hsGPA
print(f"VIF for hsGPA: {VIF_hsGPA:.3f}\n")  # Format to 3 decimal places

# %% [markdown]
# The VIF for `hsGPA` (and similarly for `ACT`) will quantify the extent to which the variance of its estimated coefficient is inflated due to its correlation with the other independent variable (`ACT`).
#
# Now, let's manually calculate the standard error of the coefficient for `hsGPA` using a formula that incorporates the VIF.  A simplified formula for the standard error of $\hat{\beta}_j$ (coefficient of $x_j$) in multiple regression, highlighting the role of VIF, can be expressed as (under certain simplifying assumptions about variable scaling):
#
# $$SE(\hat{\beta}_j) \approx \frac{1}{\sqrt{n}} \cdot \frac{SER}{\text{sd}(x_j)} \cdot \sqrt{VIF_j}$$
#
# Where:
#
# *   $n$ is the sample size.
# *   $SER$ is the Standard Error of Regression.
# *   $\text{sd}(x_j)$ is the sample standard deviation of $x_j$.
# *   $VIF_j$ is the Variance Inflation Factor for $x_j$.
#
# This formula illustrates how the standard error is directly proportional to $\sqrt{VIF_j}$. Higher VIFs lead to larger standard errors.

# %%
# Manual calculation of SE of hsGPA coefficient using VIF:
n = results.nobs  # Sample size
sdx = np.std(gpa1["hsGPA"], ddof=1) * np.sqrt(
    (n - 1) / n,
)  # Sample standard deviation of hsGPA (using population std formula)
SE_hsGPA = 1 / np.sqrt(n) * SER / sdx * np.sqrt(VIF_hsGPA)  # Approximate SE calculation
print(
    f"Manually calculated SE for hsGPA coefficient: {SE_hsGPA:.4f}\n",
)  # Format to 4 decimal places
print(
    f"SE for hsGPA coefficient from statsmodels summary: {results.bse['hsGPA']:.4f}\n",
)  # Extract BSE from statsmodels and format

# %% [markdown]
# Compare the manually calculated `SE_hsGPA` with the standard error for `hsGPA` reported in the `results.summary()` output. They should be reasonably close.
#
# For models with more than two independent variables, we can use the `variance_inflation_factor` function from `statsmodels.stats.outliers_influence` to easily calculate VIFs for all regressors.

# %%
wage1 = wool.data("wage1")

# extract matrices using patsy for wage equation with educ, exper, tenure:
y, X = pt.dmatrices(
    "np.log(wage) ~ educ + exper + tenure",
    data=wage1,
    return_type="dataframe",
)

# get VIFs for all regressors (including intercept):
K = X.shape[1]  # Number of columns in X (including intercept)
VIF = np.empty(K)  # Initialize an array to store VIFs
for i in range(K):
    VIF[i] = smo.variance_inflation_factor(
        X.values,
        i,
    )  # Calculate VIF for each regressor
print(f"VIFs for each regressor (including intercept):\n{VIF}\n")

# VIFs for independent variables only (excluding intercept):
VIF_no_intercept = VIF[1:]  # Slice VIF array to exclude intercept's VIF (index 0)
variable_names = X.columns[1:]  # Get variable names excluding intercept
vif_df = pd.DataFrame(
    {"Variable": variable_names, "VIF": VIF_no_intercept},
)  # Create DataFrame for better presentation

print("\nVIFs for independent variables (excluding intercept):\n")
print(vif_df)

# %% [markdown]
# **Interpreting VIFs in Wage Equation:** Examine the VIF values for `educ`, `exper`, and `tenure` in the wage equation.  Relatively low VIF values (typically well below 10) would suggest that multicollinearity is not a severe problem in this model. If VIFs were high, it would indicate that some of these variables are highly correlated, potentially making it difficult to precisely estimate their individual effects on wage.
#
# **In summary,** standard errors quantify the uncertainty in our coefficient estimates. Multicollinearity, a condition of high correlation among independent variables, can inflate standard errors, reducing the precision of our estimates. VIF is a useful tool for detecting and assessing the severity of multicollinearity in multiple regression models.
