# ---
# jupyter:
#   jupytext:
#     formats: notebooks//ipynb,markdown//md,scripts//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: merino
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 3. Multiple Regression Analysis: Estimation
#
# :::{important} Learning Objectives
# :class: dropdown
#
# Upon completion of this chapter, readers will demonstrate proficiency in implementing multiple regression using Python by:
#
# **3.1** Estimating multiple regression models using statsmodels formula API with pandas DataFrames and wooldridge datasets.
#
# **3.2** Computing OLS estimators through matrix operations using NumPy linear algebra functions for $\hat{\beta} = (X'X)^{-1}X'y$.
#
# **3.3** Extracting and interpreting regression results including coefficients, standard errors, t-statistics, and p-values from statsmodels output.
#
# **3.4** Calculating fitted values, residuals, and R-squared measures programmatically to assess model goodness-of-fit.
#
# **3.5** Diagnosing omitted variable bias through auxiliary regressions and comparing restricted versus unrestricted model specifications.
#
# **3.6** Computing variance inflation factors (VIFs) using statsmodels to detect multicollinearity and assess precision of coefficient estimates.
#
# **3.7** Implementing robust covariance estimators (HC0, HC1, HC2, HC3) to obtain heteroskedasticity-robust standard errors.
#
# **3.8** Visualizing partial relationships through added variable plots and residual plots using seaborn and matplotlib.
#
# **3.9** Conducting F-tests for joint hypothesis restrictions using statsmodels wald_test and f_test methods.
#
# **3.10** Automating comprehensive regression workflows including data preparation, model estimation, diagnostics, and results tabulation with pandas.
# :::
#
# Multiple regression analysis extends simple linear regression to accommodate multiple explanatory variables, reflecting the reality that economic outcomes depend on numerous simultaneous factors rather than a single cause. This chapter develops the theoretical foundations and practical implementation of Ordinary Least Squares (OLS) estimation in the multiple regression framework, establishing the conditions under which OLS provides unbiased, consistent estimates with desirable statistical properties.
#
# The presentation proceeds hierarchically through the essential components of multiple regression. We begin with motivation and mathematical specification of the multiple regression model (Section 3.1), derive the OLS estimators through matrix algebra and geometric intuition (Section 3.2), develop the crucial concept of ceteris paribus interpretation holding other factors constant (Section 3.3), examine the Gauss-Markov assumptions required for unbiasedness and establish OLS as BLUE under these conditions (Section 3.4-3.5), and address practical issues including multicollinearity, variance estimation, and model specification (Section 3.6-3.10). Throughout, we implement methods using Python's scientific computing libraries and demonstrate applications with real econometric datasets from labor economics, education, and wage determination.

# %%
import numpy as np
import pandas as pd
import patsy as pt
import statsmodels.formula.api as smf
import statsmodels.stats.outliers_influence as smo
import wooldridge as wool
from IPython.display import display
from scipy import stats

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
# * $y$ is the dependent variable (the variable we want to explain).
# * $x_1, x_2, \ldots, x_k$ are the independent variables (or regressors, explanatory variables) that we believe influence $y$.
# * $\beta_0$ is the intercept, representing the expected value of $y$ when all independent variables are zero.
# * $\beta_1, \beta_2, \ldots, \beta_k$ are the partial regression coefficients (or slope coefficients). Each $\beta_j$ represents the change in $y$ for a one-unit increase in $x_j$, *holding all other independent variables constant*. This is the crucial **ceteris paribus** (other things equal) interpretation in multiple regression.
# * $u$ is the error term (or disturbance), representing unobserved factors that also affect $y$. As in simple regression (Chapter 2), we assume $E(u|x_1, \ldots, x_k) = 0$, which implies $E(u) = 0$.
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
# * `colGPA`: College Grade Point Average (dependent variable)
# * `hsGPA`: High School Grade Point Average (independent variable)
# * `ACT`: ACT score (independent variable)
#
# We expect $\beta_1 > 0$ and $\beta_2 > 0$, suggesting that higher high school GPA and ACT scores are associated with higher college GPA, holding the other factor constant.

# %%
gpa1 = wool.data("gpa1")

reg = smf.ols(formula="colGPA ~ hsGPA + ACT", data=gpa1)
results = reg.fit()
results.summary()

# %% [markdown]
# :::{note} Interpretation of Example 3.1: College GPA Determinants
# :class: dropdown
#
# Looking at the regression results, we can interpret the coefficients in our multiple regression model:
#
# **Estimated Model:** $\widehat{\text{colGPA}} = 1.29 + 0.453 \times \text{hsGPA} + 0.009 \times \text{ACT}$
#
# **Coefficient Interpretations:**
#
# * **Intercept ($\hat{\beta}_0 = 1.29$):** The predicted college GPA when both high school GPA and ACT score are zero. While not practically meaningful (no student has zero values for both), it's mathematically necessary for the model.
# * **hsGPA ($\hat{\beta}_1 = 0.453$):** Holding ACT score constant, a one-point increase in high school GPA is associated with a 0.453-point increase in college GPA. This is the ceteris paribus effect of high school GPA.
# * **ACT ($\hat{\beta}_2 = 0.009$):** Holding high school GPA constant, a one-point increase in ACT score is associated with a 0.009-point increase in college GPA.
#
# **Key Insight:** Multiple regression allows us to isolate the effect of each variable while controlling for others. The coefficient on `hsGPA` represents its effect *after* accounting for the influence of `ACT` scores, and vice versa. This is the fundamental advantage of multiple regression over simple regression.
# :::
#
# ### Example 3.2: Hourly Wage Equation (Simple Version)
#
# Before exploring more complex wage models, let's start with a baseline specification that examines how education, experience, and tenure affect wages.
#
# **Model:** We model the log of hourly wage as a function of three human capital variables:
#
# $$ \log(\text{wage}) = \beta_0 + \beta_1 \text{educ} + \beta_2 \text{exper} + \beta_3 \text{tenure} + u$$
#
# This is one of the most fundamental equations in labor economics. The log transformation has two key advantages:
# 1. **Percentage interpretation**: Coefficients represent approximate percentage changes
# 2. **Better fit**: Wage distributions are typically right-skewed; logs normalize them

# %%
# Load wage data
wage1 = wool.data("wage1")

# Estimate the log wage equation
wage_model = smf.ols(formula="np.log(wage) ~ educ + exper + tenure", data=wage1)
wage_results = wage_model.fit()

# Display results in a clean table
coef_table = pd.DataFrame(
    {
        "Variable": wage_results.params.index,
        "Coefficient": wage_results.params.values,
        "Std Error": wage_results.bse.values,
        "t-statistic": wage_results.tvalues.values,
        "P-value": wage_results.pvalues.values,
    }
)

# Display model statistics
model_stats = pd.DataFrame(
    {
        "Statistic": [
            "R-squared",
            "Adjusted R-squared",
            "F-statistic",
            "Observations",
        ],
        "Value": [
            f"{wage_results.rsquared:.4f}",
            f"{wage_results.rsquared_adj:.4f}",
            f"{wage_results.fvalue:.2f}",
            f"{int(wage_results.nobs)}",
        ],
    }
)

display(coef_table.round(4))
display(model_stats)

# %% [markdown]
# :::{note} Interpretation of Example 3.2
# :class: dropdown
#
# The coefficients in this log-linear model have a convenient interpretation:
# - **Education ($\beta_1$)**: A one-year increase in education is associated with approximately a $\beta_1 \times 100$% increase in wage, holding experience and tenure constant
# - **Experience ($\beta_2$)**: A one-year increase in experience is associated with approximately a $\beta_2 \times 100$% increase in wage, ceteris paribus  
# - **Tenure ($\beta_3$)**: A one-year increase in tenure is associated with approximately a $\beta_3 \times 100$% increase in wage, ceteris paribus
#
# For example, if $\hat{\beta}_1 = 0.09$, this means that each additional year of education is associated with approximately a 9% increase in hourly wage.
#
# The R-squared tells us what fraction of the variation in log(wage) is explained by these three variables. While useful, remember that a low R-squared doesn't necessarily mean the model is bad - individual wages are influenced by many unobserved factors!
# :::
#
# ### Example 3.3 Hourly Wage Equation
#
# **Question:** What factors determine an individual's hourly wage?  Education, experience, and job tenure are commonly believed to be important determinants.
#
# **Model:** We can model the logarithm of wage as a function of education, experience, and tenure:
#
# $$ \log(\text{wage}) = \beta_0 + \beta_1 \text{educ} + \beta_2 \text{exper} + \beta_3 \text{tenure} + u$$
#
# * $\log(\text{wage})$: Natural logarithm of hourly wage (dependent variable). Using the log of wage is common in economics as it often leads to a more linear relationship and allows for percentage change interpretations of coefficients.
# * `educ`: Years of education (independent variable)
# * `exper`: Years of work experience (independent variable)
# * `tenure`: Years with current employer (independent variable)
#
# In this model, coefficients on `educ`, `exper`, and `tenure` will represent the approximate percentage change in wage for a one-unit increase in each respective variable, holding the others constant. For example, $\beta_1 \approx \% \Delta \text{wage} / \Delta \text{educ}$.

# %%
wage1 = wool.data("wage1")

reg = smf.ols(formula="np.log(wage) ~ educ + exper + tenure", data=wage1)
results = reg.fit()
results.summary()

# %% [markdown]
# :::{note} Interpretation of Example 3.3: Wage Equation
# :class: dropdown
#
# In this log-level model, the coefficients have a percentage interpretation:
#
# **Estimated Model:** $\widehat{\log(\text{wage})} = 0.284 + 0.092 \times \text{educ} + 0.004 \times \text{exper} + 0.022 \times \text{tenure}$
#
# **Coefficient Interpretations:**
#
# * **educ ($\hat{\beta}_1 = 0.092$):** Holding experience and tenure constant, an additional year of education is associated with approximately a 9.2% increase in hourly wage. Education appears to have a substantial return in the labor market.
# * **exper ($\hat{\beta}_2 = 0.004$):** Holding education and tenure constant, an additional year of general work experience is associated with approximately a 0.4% increase in hourly wage. The return to general experience is modest.
# * **tenure ($\hat{\beta}_3 = 0.022$):** Holding education and experience constant, an additional year with the current employer is associated with approximately a 2.2% increase in wage. Tenure has a larger impact than general experience, possibly reflecting firm-specific human capital or seniority premiums.
#
# **Economic Insights:** The model suggests that formal education provides the highest returns, followed by firm-specific tenure, with general experience having the smallest effect on wages.
# :::
#
# ### Example 3.4: Participation in 401(k) Pension Plans
#
# **Question:** What factors influence the participation rate in 401(k) pension plans among firms? Let's consider the firm's match rate and the age of the firm's employees.
#
# **Model:**
#
# $$ \text{prate} = \beta_0 + \beta_1 \text{mrate} + \beta_2 \text{age} + u$$
#
# * `prate`: Participation rate in 401(k) plans (percentage of eligible workers participating) (dependent variable)
# * `mrate`: Firm's match rate (e.g., if mrate=0.5, firm matches 50 cents for every dollar contributed by employee) (independent variable)
# * `age`: Average age of firm's employees (independent variable)
#
# We expect $\beta_1 > 0$ because a higher match rate should encourage participation. The expected sign of $\beta_2$ is less clear. Older workforces might have had more time to enroll in 401(k)s, or they may be closer to retirement and thus more interested in pension plans. Conversely, younger firms might be more proactive in encouraging enrollment.

# %%
k401k = wool.data("401k")

reg = smf.ols(formula="prate ~ mrate + age", data=k401k)
results = reg.fit()
results.summary()

# %% [markdown]
# :::{note} Interpretation of Example 3.4: 401(k) Participation
# :class: dropdown
#
# This model examines factors affecting 401(k) participation rates:
#
# **Estimated Model:** $\widehat{\text{prate}} = 80.29 + 5.52 \times \text{mrate} + 0.24 \times \text{age}$
#
# **Coefficient Interpretations:**
#
# * **Intercept ($\hat{\beta}_0 = 80.29$):** The predicted participation rate when the match rate is zero and average employee age is zero. While the age component isn't meaningful, this suggests a base participation rate of about 80% without any employer matching.
# * **mrate ($\hat{\beta}_1 = 5.52$):** Holding average age constant, a one-unit increase in the match rate (e.g., from 0.0 to 1.0, or 0% to 100% matching) increases the participation rate by 5.52 percentage points. This shows that employer matching incentives are effective.
# * **age ($\hat{\beta}_2 = 0.24$):** Holding match rate constant, each additional year in average employee age is associated with a 0.24 percentage point increase in participation rate. Older employees may be more focused on retirement planning.
#
# **Policy Insight:** The strong positive effect of matching rates suggests that increasing employer contributions is an effective way to boost 401(k) participation.
# :::
#
# ### Example 3.5a: Explaining Arrest Records
#
# **Question:** Can we explain the number of arrests a young man has in 1986 based on his criminal history and employment status?
#
# **Model (without average sentence length):**
#
# $$\text{narr86} = \beta_0 + \beta_1 \text{pcnv} + \beta_2 \text{ptime86} + \beta_3 \text{qemp86} + u$$
#
# * `narr86`: Number of arrests in 1986 (dependent variable)
# * `pcnv`: Proportion of prior arrests that led to conviction (independent variable)
# * `ptime86`: Months spent in prison in 1986 (independent variable)
# * `qemp86`: Number of quarters employed in 1986 (independent variable)
#
# We expect $\beta_1 > 0$ because a higher conviction rate might deter future crime. We also expect $\beta_2 > 0$ as spending more time in prison in 1986 means more opportunity to be arrested in 1986 (although this might be complex).  We expect $\beta_3 < 0$ because employment should reduce the likelihood of arrests.

# %%
crime1 = wool.data("crime1")

# model without avgsen:
reg = smf.ols(formula="narr86 ~ pcnv + ptime86 + qemp86", data=crime1)
results = reg.fit()
results.summary()

# %% [markdown]
# :::{note} Interpretation of Example 3.5a: Arrest Records Analysis
# :class: dropdown
#
# This model examines factors affecting criminal behavior:
#
# **Estimated Model:** $\widehat{\text{narr86}} = 0.712 - 0.150 \times \text{pcnv} - 0.034 \times \text{ptime86} - 0.104 \times \text{qemp86}$
#
# **Coefficient Interpretations:**
#
# * **pcnv ($\hat{\beta}_1 = -0.150$):** Holding prison time and employment constant, a one-unit increase in the proportion of prior arrests leading to conviction is associated with 0.15 fewer arrests in 1986. This could reflect a deterrent effect of successful prosecutions.
# * **ptime86 ($\hat{\beta}_2 = -0.034$):** Holding conviction proportion and employment constant, an additional month in prison in 1986 is associated with 0.034 fewer arrests. This might seem counterintuitive but could reflect incapacitation - time in prison means less opportunity for arrests.
# * **qemp86 ($\hat{\beta}_3 = -0.104$):** Holding other factors constant, each additional quarter of employment in 1986 is associated with 0.104 fewer arrests. Employment provides legitimate income opportunities and structure, potentially reducing criminal activity.
#
# **Criminological Insight:** Employment appears to be the most important deterrent factor among the variables considered, supporting theories that legitimate economic opportunities reduce crime.
# :::
#
# ### Example 3.5b: Explaining Arrest Records
#
# **Model (with average sentence length):** Let's add another variable, `avgsen`, the average sentence served from prior convictions, to the model to see if it influences arrests in 1986.
#
# $$\text{narr86} = \beta_0 + \beta_1 \text{pcnv} + \beta_2 \text{avgsen} + \beta_3 \text{ptime86} + \beta_4 \text{qemp86} + u$$
#
# * `avgsen`: Average sentence served from prior convictions (in months) (independent variable). We expect $\beta_2 < 0$ if longer average sentences deter crime.

# %%
crime1 = wool.data("crime1")

# model with avgsen:
reg = smf.ols(formula="narr86 ~ pcnv + avgsen + ptime86 + qemp86", data=crime1)
results = reg.fit()
results.summary()

# %% [markdown]
# :::{note} Interpretation of Example 3.5b: Extended Arrest Records Model
# :class: dropdown
#
# This extended model adds average sentence length to examine additional deterrent effects:
#
# **Estimated Model:** $\widehat{\text{narr86}} = 0.707 - 0.151 \times \text{pcnv} + 0.007 \times \text{avgsen} - 0.037 \times \text{ptime86} - 0.104 \times \text{qemp86}$
#
# **Key Findings:**
#
# * **avgsen ($\hat{\beta}_2 = 0.007$):** The coefficient on average sentence length is small and positive, suggesting that longer sentences are associated with slightly more arrests. However, this effect is not statistically significant (p-value > 0.05), indicating we cannot conclude that sentence length has a meaningful deterrent effect.
# * **Coefficient Stability:** The coefficients on other variables remain similar to Model 3.5a, indicating robustness of the original findings.
# * **Employment Effect Persists:** The negative, significant coefficient on employment quarters remains, reinforcing the importance of legitimate work opportunities.
#
# **Model Comparison Insight:** Adding `avgsen` does not improve the model's explanatory power significantly. This demonstrates that not all theoretically relevant variables prove to be empirically important, and model building often involves testing multiple specifications to identify the most meaningful relationships.
#
# **Statistical Learning:** When adding variables to a model, it's important to assess both their statistical significance and their effect on existing coefficients to understand the overall model dynamics.
# :::
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
results.summary()

# %% [markdown]
# :::{note} Interpretation of Example 3.6: Simple vs Multiple Regression
# :class: dropdown
#
# This simple regression demonstrates the importance of including relevant variables:
#
# **Simple Regression Model:** $\widehat{\log(\text{wage})} = -0.185 + 0.108 \times \text{educ}$
# **Multiple Regression Model (from Example 3.3):** $\widehat{\log(\text{wage})} = 0.284 + 0.092 \times \text{educ} + 0.004 \times \text{exper} + 0.022 \times \text{tenure}$
#
# **Key Comparison:**
#
# * **Education Coefficient Difference:** The education coefficient is larger in the simple regression (0.108) than in the multiple regression (0.092). This difference of 0.016 represents potential omitted variable bias.
# * **Economic Interpretation:** The simple regression overestimates the return to education by about 1.6 percentage points because it fails to account for the fact that more educated workers also tend to have different levels of experience and tenure.
#
# **Omitted Variable Bias Preview:** When we omit relevant variables like experience and tenure that are correlated with education, the education coefficient "picks up" some of their effects, leading to biased estimates. This demonstrates why multiple regression is crucial for isolating the true effect of individual variables.
#
# **Methodological Insight:** Comparing simple and multiple regression results helps identify potential omitted variable bias and underscores the importance of including all relevant explanatory variables in our models.
# :::
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
# * $y$ is an $(n \times 1)$ column vector of the dependent variable observations.
# * $X$ is an $(n \times (k+1))$ matrix of independent variable observations, called the **design matrix**. The first column of $X$ is typically a column of ones (for the intercept), and the subsequent columns are the observations of the independent variables $x_1, x_2, \ldots, x_k$.
# * $\beta$ is a $((k+1) \times 1)$ column vector of the unknown regression coefficients $(\beta_0, \beta_1, \ldots, \beta_k)'$.
# * $u$ is an $(n \times 1)$ column vector of the error terms.
#
# The OLS estimator $\hat{\beta}$ that minimizes the sum of squared residuals can be expressed in matrix form as:
#
# $$\hat{\beta} = (X'X)^{-1}X'y$$
#
# Where:
#
# * $X'$ is the transpose of the matrix $X$.
# * $(X'X)^{-1}$ is the inverse of the matrix product $X'X$.
#
# Let's demonstrate this matrix calculation using the `gpa1` dataset from Example 3.1.

# %%
gpa1 = wool.data("gpa1")

# determine sample size & no. of regressors:
n = len(gpa1)
k = 2  # Number of independent variables (hsGPA and ACT)

# Extract dependent variable (college GPA)
y = gpa1["colGPA"]
# Show shape info
shape_info = pd.DataFrame(
    {
        "Variable": ["y (colGPA)"],
        "Shape": [str(y.shape)],
    },
)
shape_info

# Method 1: Manual construction of design matrix X
# Design matrix X = [1, hsGPA, ACT] for each observation
X = pd.DataFrame(
    {
        "const": 1,  # Intercept column (beta_0)
        "hsGPA": gpa1["hsGPA"],  # High school GPA (beta_1)
        "ACT": gpa1["ACT"],  # ACT test score (beta_2)
    },
)

# Method 2: Using patsy for automatic formula-based design matrix
# More convenient for complex models with interactions/polynomials
y2, X2 = pt.dmatrices(
    "colGPA ~ hsGPA + ACT",  # R-style formula
    data=gpa1,
    return_type="dataframe",
)

# Display design matrix structure
matrix_info = pd.DataFrame(
    {
        "Description": ["Design matrix dimensions", "Interpretation"],
        "Value": [f"{X.shape}", "(n observations x k+1 variables)"],
    },
)
display(matrix_info)
X.head()

# %% [markdown]
# The code above constructs the $X$ matrix and $y$ vector from the `gpa1` data.  The `patsy` library provides a convenient way to create design matrices directly from formulas, which is often more efficient for complex models.

# %%
# Calculate OLS estimates using the matrix formula: beta_hat = (X'X)^-^1X'y

# Step 1: Convert to numpy arrays for matrix operations
X_array = np.array(X)  # Design matrix as numpy array (n x k+1)
y_array = np.array(y).reshape(n, 1)  # Dependent variable as column vector (n x 1)

# Step 2: Calculate intermediate matrices for clarity
XtX = X_array.T @ X_array  # X'X matrix (k+1 x k+1)
Xty = X_array.T @ y_array  # X'y vector (k+1 x 1)

# Display matrix operation results
matrix_ops = pd.DataFrame(
    {
        "Operation": ["X'X matrix", "X'X symmetry check", "X'y vector"],
        "Shape": [str(XtX.shape), "-", str(Xty.shape)],
        "Result": ["(k+1 x k+1)", str(np.allclose(XtX, XtX.T)), "(k+1 x 1)"],
    },
)
matrix_ops

# Step 3: Apply OLS formula
XtX_inverse = np.linalg.inv(XtX)  # (X'X)^-^1
beta_estimates = XtX_inverse @ Xty  # beta_hat = (X'X)^-^1X'y

# Display results
coef_results = pd.DataFrame(
    {
        "Variable": ["Intercept", "hsGPA", "ACT"],
        "Coefficient (beta_hat)": [beta_estimates[i, 0] for i in range(3)],
    },
)
coef_results

# %% [markdown]
# This code performs the matrix operations to calculate $\hat{\beta}$. The result `b` should match the coefficients we obtained from `statsmodels` earlier.
#
# After estimating $\hat{\beta}$, we can calculate the residuals $\hat{u}$:
#
# $$\hat{u} = y - X\hat{\beta}$$
#
# And the estimator for the error variance $\sigma^2$:
#
# $$\hat{\sigma}^2 = \frac{1}{n-k-1} \hat{u}'\hat{u} = \frac{\text{SSR}}{n-k-1}$$
#
# The denominator $(n-k-1)$ represents the degrees of freedom in multiple regression, where $n$ is the sample size and $(k+1)$ is the number of parameters estimated (including the intercept).  The square root of $\hat{\sigma}^2$ is the Standard Error of the Regression (SER).
#
# **Key Insight:** The division by $n-k-1$ (not $n$) corrects for the degrees of freedom lost in estimating $k+1$ parameters, making $\hat{\sigma}^2$ an unbiased estimator of $\sigma^2$ under MLR.1-MLR.5.

# %%
# residuals, estimated variance of u and SER:
u_hat = (
    y.values.reshape(-1, 1) - X.values @ beta_estimates
)  # Calculate residuals as numpy array
sigsq_hat = float((u_hat.T @ u_hat) / (n - k - 1))  # Estimated error variance (scalar)
SER = np.sqrt(sigsq_hat)  # Standard Error of Regression
SER  # Display SER

# %% [markdown]
# Finally, the estimated variance-covariance matrix of the OLS estimator $\hat{\beta}$ is given by:
#
# $$\widehat{\text{var}(\hat{\beta})} = \hat{\sigma}^2 (X'X)^{-1}$$
#
# The standard errors for each coefficient are the square roots of the diagonal elements of this variance-covariance matrix.

# %%
# estimated variance-covariance matrix of beta_hat and standard errors:
Vbeta_hat = sigsq_hat * np.linalg.inv(
    X.values.T @ X.values,
)  # Variance-covariance matrix
se = np.sqrt(np.diagonal(Vbeta_hat))  # Standard errors (diagonal elements' square root)
se  # Display standard errors

# %% [markdown]
# The manually calculated coefficients (`b`), SER, and standard errors (`se`) should be very close (or identical, considering potential rounding differences) to the values reported in the `results.summary()` output from `statsmodels` for Example 3.1. This demonstrates that `statsmodels` is using the matrix-based OLS formulas under the hood.
#
# ## 3.4 Ceteris Paribus Interpretation and Omitted Variable Bias
#
# A key advantage of multiple regression is its ability to provide **ceteris paribus** interpretations of the coefficients.  "Ceteris paribus" is Latin for "other things being equal" or "holding other factors constant." In the context of multiple regression, the coefficient on a particular independent variable represents the effect of that variable on the dependent variable *while holding all other included independent variables constant*.
#
# However, if we **omit** a relevant variable from our regression model, and this omitted variable is correlated with the included independent variables, we can encounter **omitted variable bias** (OVB). This means that the estimated coefficients on the included variables will be biased and inconsistent, and they will no longer have the desired ceteris paribus interpretation with respect to the omitted variable.
#
# **Why Omitted Variable Bias Violates MLR.4:**
# When we omit a relevant variable $x_k$, it becomes part of the error term: $u' = u + \beta_k x_k$. If the omitted $x_k$ is correlated with any included $x_j$, then $E(u'|x_1, \ldots, x_{k-1}) \neq 0$, violating the zero conditional mean assumption. This makes OLS biased and inconsistent.
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
b  # Display coefficients from full model

# %% [markdown]
# Now, let's consider the relationship between the included variable (`ACT`) and the omitted variable (`hsGPA`). We can regress the omitted variable (`hsGPA`) on the included variable (`ACT`):

# %%
# relation between regressors (hsGPA on ACT):
reg_delta = smf.ols(formula="hsGPA ~ ACT", data=gpa1)
results_delta = reg_delta.fit()
delta_tilde = results_delta.params  # Extract coefficient of ACT in this regression
delta_tilde  # Display coefficients from regression of hsGPA on ACT

# %% [markdown]
# $\delta_{\text{ACT}}$ from this regression represents how much `hsGPA` changes on average for a one-unit change in `ACT`.  It captures the correlation between `hsGPA` and `ACT`.
#
# The **omitted variable bias formula** provides an approximation for the bias in the simple regression coefficient $\hat{\gamma}_1$ when we omit `hsGPA`.  In this case, the bias in the coefficient of `ACT` (when `hsGPA` is omitted) is approximately:
#
# $$\text{Bias}(\hat{\gamma}_1) \approx \beta_1 \times \delta_{\text{ACT}}$$
#
# Where:
#
# * $\beta_1$ is the coefficient of the omitted variable (`hsGPA`) in the full model.
# * $\delta_{\text{ACT}}$ is the coefficient of the included variable (`ACT`) in the regression of the omitted variable (`hsGPA`) on the included variable (`ACT`).
#
# Let's calculate this approximate bias and see how it relates to the difference between the coefficient of `ACT` in the full model ($\beta_2$) and the coefficient of `ACT` in the simple model ($\gamma_1$).

# %%
# omitted variables formula for b1_tilde (approximate bias in ACT coefficient when hsGPA is omitted):
b1_tilde = b["ACT"] + b["hsGPA"] * delta_tilde["ACT"]  # Applying the bias formula
b1_tilde  # Display approximate biased coefficient of ACT

# %% [markdown]
# Finally, let's estimate the simple regression model (omitting `hsGPA`) and see the actual coefficient of `ACT`:

# %%
# actual regression with hsGPA omitted (simple regression):
reg_om = smf.ols(formula="colGPA ~ ACT", data=gpa1)
results_om = reg_om.fit()
b_om = results_om.params  # Extract coefficient of ACT from simple regression
b_om  # Display coefficient of ACT in simple regression

# %% [markdown]
# Comparing `b_om["ACT"]` (the coefficient of ACT in the simple regression) with `b["ACT"]` (the coefficient of ACT in the full regression), we can see that they are different.  Furthermore, `b1_tilde` (the approximate biased coefficient calculated using the formula) is close to `b_om["ACT"]`.
#
# **Conclusion on Omitted Variable Bias:** Omitting a relevant variable like `hsGPA` that is correlated with an included variable like `ACT` can lead to biased estimates. In this case, the coefficient on `ACT` in the simple regression is larger than in the multiple regression, likely because it is picking up some of the positive effect of `hsGPA` on `colGPA`.  This highlights the importance of including all relevant variables in a regression model to obtain unbiased and consistent estimates and to achieve correct ceteris paribus interpretations.
#
# ## 3.3 The Gauss-Markov Assumptions
#
# To establish the statistical properties of the OLS estimators in multiple regression, we need to specify the assumptions under which these properties hold. The **Gauss-Markov assumptions** provide the foundation for understanding when OLS produces the Best Linear Unbiased Estimators (BLUE). Let's examine each assumption:
#
# ### The Five Gauss-Markov Assumptions (MLR.1 - MLR.5)
#
# These assumptions extend the Simple Linear Regression (SLR) assumptions from Chapter 2 to the multiple regression context.
#
# **MLR.1: Linear in Parameters** (extends SLR.1)
# The population model is linear in the parameters:
# $$y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \cdots + \beta_k x_k + u$$
#
# This assumption requires that the dependent variable is a linear function of the parameters $\beta_j$, though the relationship can be nonlinear in the variables themselves (e.g., $x_2 = x_1^2$, $x_3 = \log(x_1)$). The key is **linearity in parameters**, not necessarily in variables.
#
# **MLR.2: Random Sampling** (extends SLR.2)
# We have a random sample of size $n$: $\{(x_{i1}, x_{i2}, \ldots, x_{ik}, y_i): i = 1, 2, \ldots, n\}$ from the population model.
#
# This ensures our sample is representative of the population we want to study and that observations are independent across $i$.
#
# **MLR.3: No Perfect Collinearity** (extends SLR.3)
# In the sample (and therefore in the population), none of the independent variables is constant, and there are no exact linear relationships among the independent variables.
#
# More precisely: (i) each $x_j$ has sample variation ($\widehat{\text{Var}}(x_j) > 0$), and (ii) no $x_j$ can be written as an exact linear combination of the other independent variables. This assumption ensures that we can obtain unique OLS estimates. Perfect multicollinearity would make it impossible to isolate the individual effects of the explanatory variables. Note that **high** (but not perfect) correlation among regressors is allowed, though it increases standard errors.
#
# **MLR.4: Zero Conditional Mean** (extends SLR.4)
# The error term $u$ has an expected value of zero given any values of the independent variables:
# $$E(u|x_1, x_2, \ldots, x_k) = 0$$
#
# This is the most crucial assumption for **unbiasedness** of OLS estimators. It implies that all factors in $u$ are, on average, unrelated to $x_1, \ldots, x_k$. This assumption implies $E(u) = 0$ and $\text{Cov}(x_j, u) = 0$ for all $j = 1, \ldots, k$. If any $x_j$ is correlated with $u$, OLS estimators will be **biased** and **inconsistent**. Violations typically arise from omitted variables, measurement error in regressors, or simultaneity.
#
# **MLR.5: Homoscedasticity** (extends SLR.5)
# The error term $u$ has the same variance given any values of the independent variables:
# $$\text{Var}(u|x_1, x_2, \ldots, x_k) = \sigma^2$$
#
# This assumption ensures that the variance of the error term is constant across all combinations of explanatory variables. This assumption is required for **efficiency** (BLUE property) and for standard OLS standard errors to be valid. Under MLR.1-MLR.4 alone (without homoscedasticity), OLS estimators remain **unbiased** and **consistent**, but they are not BLUE, and standard errors must be corrected (e.g., using robust/heteroscedasticity-consistent standard errors, as discussed in Chapter 8).
#
# ### Properties Under the Gauss-Markov Assumptions
#
# **Theorem 3.1: Unbiasedness of OLS (MLR.1-MLR.4)**
# Under assumptions **MLR.1 through MLR.4** (linearity, random sampling, no perfect collinearity, and zero conditional mean), the OLS estimators are unbiased:
# $$E(\hat{\beta}_j) = \beta_j \text{ for } j = 0, 1, 2, \ldots, k$$
#
# This means that on average, across repeated random samples from the same population, the OLS estimates equal the true population parameters. Crucially, **homoscedasticity (MLR.5) is not required** for unbiasedness--only the first four assumptions are needed.
#
# **Theorem 3.2: Variance of OLS Estimators (MLR.1-MLR.5)**
# Under assumptions **MLR.1 through MLR.5** (including homoscedasticity), the variance-covariance matrix of the OLS estimators $\hat{\boldsymbol{\beta}} = (\hat{\beta}_0, \hat{\beta}_1, \ldots, \hat{\beta}_k)'$ conditional on the sample values of $\mathbf{X}$ is:
# $$\text{Var}(\hat{\boldsymbol{\beta}}|\mathbf{X}) = \sigma^2 (\mathbf{X}'\mathbf{X})^{-1}$$
#
# where $\sigma^2 = \text{Var}(u|x_1, \ldots, x_k)$ is the constant conditional variance of the error term. The diagonal elements of this matrix are the variances of individual OLS estimators, and the off-diagonal elements are the covariances. Standard errors are the square roots of the diagonal elements. This result allows us to conduct valid statistical inference (hypothesis tests, confidence intervals) when homoscedasticity holds.
#
# ### The Gauss-Markov Theorem
#
# **Theorem 3.3: Gauss-Markov Theorem (MLR.1-MLR.5)**
# Under the Gauss-Markov assumptions **MLR.1 through MLR.5**, the OLS estimators $\hat{\beta}_0, \hat{\beta}_1, \ldots, \hat{\beta}_k$ are the **Best Linear Unbiased Estimators (BLUE)** of $\beta_0, \beta_1, \ldots, \beta_k$.
#
# "Best" means that among all **linear** unbiased estimators (estimators that are linear functions of $y$), OLS has the smallest variance for each coefficient. More precisely, for any other linear unbiased estimator $\tilde{\beta}_j$ of $\beta_j$, we have $\text{Var}(\hat{\beta}_j) \leq \text{Var}(\tilde{\beta}_j)$. This theorem provides the fundamental justification for using OLS in linear regression analysis under the classical assumptions. 
#
# **Intuition Behind the Gauss-Markov Theorem:**
# The OLS estimator achieves minimum variance among linear unbiased estimators because:
# 1. **Orthogonality principle:** OLS residuals are orthogonal to all regressors, ensuring no systematic patterns remain
# 2. **Efficient use of information:** OLS optimally weights observations based on the variation in $X$
# 3. **Homoscedasticity crucial:** Equal error variances allow equal weighting; with heteroscedasticity, weighted least squares (WLS) would be more efficient
#
# **Important Notes:**
# - BLUE property requires all five assumptions, including homoscedasticity (MLR.5)
# - Unbiasedness requires only MLR.1-MLR.4
# - Consistency requires even weaker conditions than unbiasedness (see Chapter 5 on asymptotics)
#
# :::{note} Understanding the Gauss-Markov Assumptions
# :class: dropdown
#
# **Why These Assumptions Matter:**
#
# 1. **MLR.1 (Linearity):** Without this, OLS is not the appropriate estimation method.
# 2. **MLR.2 (Random Sampling):** Ensures our results can be generalized to the population.
# 3. **MLR.3 (No Perfect Collinearity):** Ensures unique, identifiable estimates.
# 4. **MLR.4 (Zero Conditional Mean):** Critical for unbiasedness - violations lead to biased estimates.
# 5. **MLR.5 (Homoscedasticity):** While not needed for unbiasedness, it's required for efficiency and valid standard errors.
#
# **Minimum Requirements for Unbiasedness:** Only MLR.1 through MLR.4 are needed for unbiasedness. MLR.5 is additionally required for the Gauss-Markov theorem to hold.
#
# **Practical Implications:** These assumptions guide model specification, data collection, and interpretation of results. Violations can lead to biased, inconsistent, or inefficient estimates.
# :::
#
# ## 3.5 Standard Errors, Multicollinearity, and VIF
#
# In multiple regression, we not only estimate coefficients but also need to assess their precision. **Standard errors (SEs)** of the estimated coefficients are crucial for conducting hypothesis tests and constructing confidence intervals.  Larger standard errors indicate less precise estimates.
#
# The standard error of a coefficient in multiple regression depends on several factors, including:
#
# * **Sample size (n):** Larger sample sizes generally lead to smaller standard errors (more precision).
# * **Error variance ($\sigma^2$):** Higher error variance (more noise in the data) leads to larger standard errors.
# * **Total sample variation in the independent variable ($x_j$):** More variation in $x_j$ (holding other regressors constant) leads to smaller standard errors for $\beta_j$.
# * **Correlation among independent variables (Multicollinearity):** Higher correlation among independent variables (multicollinearity) generally leads to larger standard errors.
#
# **Multicollinearity** arises when two or more independent variables in a regression model are highly linearly correlated.  While multicollinearity does not bias the OLS coefficient estimates, it increases their standard errors, making it harder to precisely estimate the individual effects of the correlated variables and potentially leading to statistically insignificant coefficients even if the variables are truly important.
#
# The **Variance Inflation Factor (VIF)** is a common measure of multicollinearity.  For each independent variable $x_j$, the VIF is calculated as:
#
# $$VIF_j = \frac{1}{1 - R_j^2}$$
#
# Where $R_j^2$ is the R-squared from regressing $x_j$ on all *other* independent variables in the model.
#
# * A VIF of 1 indicates no multicollinearity.
# * VIFs greater than 1 indicate increasing multicollinearity.
# * As a rule of thumb, VIFs greater than 10 are often considered to indicate high multicollinearity, although there is no strict cutoff.
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

# Calculate VIF for hsGPA to assess multicollinearity
# VIF measures how much the variance of beta_hat_hsGPA is inflated due to correlation with ACT

# Step 1: Auxiliary regression - regress hsGPA on other predictors (just ACT here)
auxiliary_regression = smf.ols(formula="hsGPA ~ ACT", data=gpa1)
auxiliary_results = auxiliary_regression.fit()
R2_auxiliary = auxiliary_results.rsquared  # R^2 from auxiliary regression

# Step 2: Calculate VIF using formula: VIF = 1 / (1 - R^2)
VIF_hsGPA = 1 / (1 - R2_auxiliary)

# VIF Calculation for hsGPA
vif_results = pd.DataFrame(
    {
        "Metric": [
            "R^2 from auxiliary regression",
            "VIF calculation",
            "VIF for hsGPA",
            "Interpretation",
        ],
        "Value": [
            f"{R2_auxiliary:.4f}",
            f"1/(1-{R2_auxiliary:.4f})",
            f"{VIF_hsGPA:.2f}",
            f"Variance inflated by {VIF_hsGPA:.2f}x",
        ],
    },
)
vif_results

# %% [markdown]
# The VIF for `hsGPA` (and similarly for `ACT`) will quantify the extent to which the variance of its estimated coefficient is inflated due to its correlation with the other independent variable (`ACT`).
#
# Now, let's manually calculate the standard error of the coefficient for `hsGPA` using a formula that incorporates the VIF.  A simplified formula for the standard error of $\hat{\beta}_j$ (coefficient of $x_j$) in multiple regression, highlighting the role of VIF, can be expressed as (under certain simplifying assumptions about variable scaling):
#
# $$SE(\hat{\beta}_j) \approx \frac{1}{\sqrt{n}} \cdot \frac{SER}{\text{sd}(x_j)} \cdot \sqrt{VIF_j}$$
#
# Where:
#
# * $n$ is the sample size.
# * $SER$ is the Standard Error of Regression.
# * $\text{sd}(x_j)$ is the sample standard deviation of $x_j$.
# * $VIF_j$ is the Variance Inflation Factor for $x_j$.
#
# This formula illustrates how the standard error is directly proportional to $\sqrt{VIF_j}$. Higher VIFs lead to larger standard errors.

# %%
# Manual calculation of SE of hsGPA coefficient using VIF:
n = results.nobs  # Sample size
sdx = np.std(gpa1["hsGPA"], ddof=1) * np.sqrt(
    (n - 1) / n,
)  # Sample standard deviation of hsGPA (using population std formula)
SE_hsGPA = 1 / np.sqrt(n) * SER / sdx * np.sqrt(VIF_hsGPA)  # Approximate SE calculation

# Display comparison of manual vs statsmodels standard errors
se_comparison = pd.DataFrame(
    {
        "Method": ["Manual Calculation", "Statsmodels"],
        "SE for hsGPA": [SE_hsGPA, results.bse["hsGPA"]],
    },
)
se_comparison

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

# Calculate VIFs for all regressors (vectorized approach)
K = X.shape[1]  # Number of columns in X (including intercept)
VIF = np.array(
    [smo.variance_inflation_factor(X.values, i) for i in range(K)]
)  # List comprehension (pythonic)

# VIFs for independent variables only (excluding intercept):
VIF_no_intercept = VIF[1:]  # Slice VIF array to exclude intercept's VIF (index 0)
variable_names = X.columns[1:]  # Get variable names excluding intercept
vif_df = pd.DataFrame(
    {"Variable": variable_names, "VIF": VIF_no_intercept},
)  # Create DataFrame for better presentation

vif_df  # Display VIFs for independent variables

# %% [markdown]
# :::{note} Interpreting VIFs in Wage Equation
# :class: dropdown
#
# Examine the VIF values for `educ`, `exper`, and `tenure` in the wage equation. Relatively low VIF values (typically well below 10) would suggest that multicollinearity is not a severe problem in this model. If VIFs were high, it would indicate that some of these variables are highly correlated, potentially making it difficult to precisely estimate their individual effects on wage.
# :::
#
# :::{note} Understanding Multicollinearity and VIF
# :class: dropdown
#
# **What the VIF Values Tell Us:**
#
# The Variance Inflation Factors calculated for the wage equation variables help us assess multicollinearity:
#
# **Interpretation Guidelines:**
#
# * **VIF = 1:** No correlation with other independent variables (ideal)
# * **1 < VIF < 5:** Moderate correlation (generally acceptable)
# * **5 <= VIF < 10:** High correlation (potential concern)
# * **VIF >= 10:** Very high correlation (serious multicollinearity problem)
#
# **Key Points About Multicollinearity:**
#
# 1. **Does Not Violate Gauss-Markov Assumptions:** Multicollinearity is a data problem, not an assumption violation
# 2. **Affects Efficiency, Not Bias:** OLS estimates remain unbiased, but standard errors increase
# 3. **Makes Individual Effects Hard to Estimate:** High correlation makes it difficult to isolate individual variable effects
# 4. **Can Lead to Insignificant Coefficients:** Even important variables may appear statistically insignificant due to inflated standard errors
#
# **Practical Consequences:**
#
# * Larger confidence intervals for coefficients
# * Reduced statistical power for hypothesis tests
# * Estimates become more sensitive to small data changes
# * Difficulty in interpreting individual variable effects
#
# **Solutions When VIF is High:**
#
# * Drop highly correlated variables (if theoretically justified)
# * Combine correlated variables into indices
# * Use principal components or factor analysis
# * Collect more data to improve precision
# :::
#
# **In summary,** standard errors quantify the uncertainty in our coefficient estimates. Multicollinearity, a condition of high correlation among independent variables, can inflate standard errors, reducing the precision of our estimates. VIF is a useful tool for detecting and assessing the severity of multicollinearity in multiple regression models.
#
# ## 3.6 The Language of Multiple Regression Analysis
#
# As you delve deeper into econometrics and read empirical research, you'll encounter various terms used to describe the variables and relationships in multiple regression models. Understanding this terminology is crucial for clear communication and proper interpretation of results.
#
# ### 3.6.1 Terminology for Variables
#
# Different disciplines and contexts use different terms for the same concepts:
#
# **Dependent Variable** (the outcome we're trying to explain):
# - Also called: response variable, regressand, explained variable, outcome variable, left-hand side (LHS) variable
# - Denoted: $y$
# - Example: wage, GPA, crime rate
#
# **Independent Variables** (the explanatory factors):
# - Also called: regressors, covariates, explanatory variables, predictors, right-hand side (RHS) variables, control variables
# - Denoted: $x_1, x_2, \ldots, x_k$
# - Example: education, experience, tenure
#
# **Key Distinction - Controls vs Variables of Interest:**
#
# Not all independent variables play the same role in your analysis:
#
# - **Variable of Interest (Treatment Variable)**: The main variable whose effect you want to measure
#   - Example: In a study of education returns, `educ` is the variable of interest
#   
# - **Control Variables**: Variables included to isolate the effect of the variable of interest
#   - Example: `exper` and `tenure` are controls when studying returns to education
#   - Purpose: Reduce omitted variable bias, improve precision

# %%
# Illustrate the distinction between variable of interest and controls
np.random.seed(42)
n = 1000

# Generate data where education is the treatment of interest
ability = stats.norm.rvs(0, 1, size=n)
education = 12 + 2 * ability + stats.norm.rvs(0, 2, size=n)
experience = 10 + stats.norm.rvs(0, 5, size=n)
wage = 5 + 1.5 * education + 0.5 * experience + 3 * ability + stats.norm.rvs(0, 5, size=n)

df_wage = pd.DataFrame({"wage": wage, "education": education, "experience": experience})

# Regression 1: Education only (omitting controls)
model_no_control = smf.ols("wage ~ education", data=df_wage).fit()

# Regression 2: Education with experience control
model_with_control = smf.ols("wage ~ education + experience", data=df_wage).fit()

# Compare the education coefficient (variable of interest)
comparison = pd.DataFrame(
    {
        "Specification": ["No Controls", "With Experience Control"],
        "Education Coef": [
            model_no_control.params["education"],
            model_with_control.params["education"],
        ],
        "Std Error": [model_no_control.bse["education"], model_with_control.bse["education"]],
        "Interpretation": [
            "Unconditional effect (biased)",
            "Conditional on experience (less biased)",
        ],
    }
)

display(comparison.round(4))

# %% [markdown]
# ### 3.6.2 Conditional vs Unconditional Effects
#
# This distinction is fundamental to understanding regression coefficients:
#
# **Unconditional Effect (Simple Regression)**:
# - The total association between $x$ and $y$
# - Includes both direct and indirect effects through other variables
# - Formula: $\frac{\text{Cov}(x,y)}{\text{Var}(x)}$
#
# **Conditional Effect (Multiple Regression)**:
# - The partial effect of $x$ on $y$, **holding other variables constant**
# - Attempts to isolate the direct effect
# - This is what the coefficient represents in multiple regression
#
# ### 3.6.3 Careful Language: Effect, Impact, Association
#
# Econometricians are increasingly careful about the language used to describe regression coefficients:
#
# **"Association" or "Relationship"**:
# - Always safe to use
# - Makes no causal claims
# - Example: "Education is associated with higher wages"
#
# **"Effect" or "Impact"**:
# - Implies causality
# - Should only be used when causal identification is credible
# - Example: "The effect of education on wages" (requires strong assumptions)
#
# **"Predictive Relationship"**:
# - Appropriate for forecasting models
# - Doesn't imply causality
# - Example: "Education helps predict wages"
#
# :::{warning} Causation vs Correlation
# :class: dropdown
#
# **Key Principle**: Regression estimates **associations**, not necessarily **causal effects**.
#
# To claim a causal interpretation, you need:
# 1. **Randomization** (experiments)
# 2. **Quasi-experimental variation** (natural experiments)
# 3. **Strong identification assumptions** (instrumental variables, etc.)
#
# Without these, stick to language like "associated with" or "correlated with" rather than "causes" or "affects."
# :::
#
# ## 3.7 Including "Bad Controls"
#
# Not all control variables improve your regression. Some variables, when included, can actually **introduce bias** rather than reduce it. Understanding which variables to include (and exclude) is crucial for credible empirical analysis.
#
# ### 3.7.1 What Makes a Control "Bad"?
#
# A control variable is "bad" when including it:
# 1. **Blocks the causal path** you're trying to estimate (post-treatment bias)
# 2. **Opens a non-causal path** (collider bias)
# 3. **Introduces measurement error** or other problems
#
# ### 3.7.2 Post-Treatment Bias (Controlling for Outcomes)
#
# **The Problem**: If you control for a variable that is itself **affected by your treatment**, you'll underestimate (or completely miss) the true effect.
#
# **Classic Example: Returns to Education**
#
# Suppose we want to estimate the effect of education on income. Consider this regression:
#
# $$ \text{income} = \beta_0 + \beta_1 \text{education} + \beta_2 \text{occupation} + u $$
#
# **Problem**: Occupation is **determined by education**! 
# - Education $\to$ Better occupation $\to$ Higher income
# - By controlling for occupation, we block part of education's effect
# - The coefficient $\beta_1$ now only captures the **direct** effect of education, not the total effect

# %%
# Demonstrate post-treatment bias
np.random.seed(123)
n = 1000

# Generate data with causal chain: education -> occupation -> income
education = stats.norm.rvs(12, 3, size=n)
occupation_quality = 0.5 * education + stats.norm.rvs(0, 2, size=n)  # Occupation CAUSED by education
income = (
    20 + 2 * education + 3 * occupation_quality + stats.norm.rvs(0, 5, size=n)
)  # Income from both

df_posttreat = pd.DataFrame(
    {"income": income, "education": education, "occupation": occupation_quality}
)

# Correct model: Don't control for post-treatment variable
model_correct = smf.ols("income ~ education", data=df_posttreat).fit()

# Bad model: Control for occupation (post-treatment variable)
model_bad = smf.ols("income ~ education + occupation", data=df_posttreat).fit()

# Compare
posttreat_comparison = pd.DataFrame(
    {
        "Model": ["Correct (No Occupation)", "Bad (With Occupation)", "True Total Effect"],
        "Education Coefficient": [
            model_correct.params["education"],
            model_bad.params["education"],
            2 + 3 * 0.5,
        ],  # True: direct + indirect
        "Interpretation": [
            "Total effect (direct + indirect)",
            "Only direct effect (BIASED LOW)",
            "True value = 2 + 3*0.5 = 3.5",
        ],
    }
)

display(posttreat_comparison.round(3))

# %% [markdown]
# :::{important} The Lesson
# :class: dropdown
#
# **Don't control for variables that are outcomes or consequences of your treatment!**
#
# This applies to:
# - Occupation (determined by education)
# - Current employment (determined by job training programs)
# - Health behaviors (determined by health insurance)
#
# You'll systematically **underestimate** the treatment effect by blocking its indirect pathways.
# :::
#
# ### 3.7.3 Collider Bias (Over-Controlling)
#
# **The Problem**: Controlling for a variable that is **caused by both** your treatment and outcome can create spurious associations.
#
# **Example: Discrimination in Hiring**
#
# Suppose we want to know if there's discrimination in promotion decisions based on race. Consider:
#
# $$ \text{promoted} = \beta_0 + \beta_1 \text{minority} + \beta_2 \text{performance} + u $$
#
# If we control for job performance, but:
# - Minority status affects hiring (discrimination in hiring)
# - Only high-performing minorities get hired (selection)
#
# Then performance is a **collider** - it's affected by both minority status and promotion potential. Controlling for it induces bias!
#
# ### 3.7.4 Proxy Controls and Measurement Error
#
# Sometimes we want to control for unobserved variables (like ability) and use **proxies** instead:
#
# **Example**: Using IQ test scores as a proxy for ability
#
# **Problems**:
# - If the proxy is **imperfect**, you won't fully eliminate omitted variable bias
# - If the proxy has **measurement error**, it can introduce new bias
# - You might introduce **post-treatment bias** if the proxy itself is affected by treatment
#
# ### 3.7.5 General Principles for Selecting Controls
#
# **✓ DO include controls that**:
# 1. Affect both treatment and outcome (confounders)
# 2. Reduce unexplained variation (improve precision)
# 3. Were determined **before** treatment
#
# **✗ DON'T include controls that**:
# 1. Are **outcomes** of the treatment (post-treatment variables)
# 2. Are **colliders** (affected by both treatment and outcome)
# 3. Have severe measurement error
# 4. Create perfect multicollinearity

# %%
# Demonstrate good vs bad control selection
np.random.seed(456)
n = 1000

# Causal structure:
# ability -> education & income (confounder - GOOD control)
# education -> occupation -> income (occupation is post-treatment - BAD control)

ability = stats.norm.rvs(0, 1, size=n)
education = 12 + 2 * ability + stats.norm.rvs(0, 2, size=n)
occupation = 0.5 * education + stats.norm.rvs(0, 1, size=n)  # POST-TREATMENT
income = 20 + 1.5 * education + 3 * ability + 2 * occupation + stats.norm.rvs(0, 5, size=n)

df_controls = pd.DataFrame(
    {"income": income, "education": education, "ability": ability, "occupation": occupation}
)

# Model 1: No controls (omitted variable bias)
m1 = smf.ols("income ~ education", data=df_controls).fit()

# Model 2: Good control (ability - confounder)
m2 = smf.ols("income ~ education + ability", data=df_controls).fit()

# Model 3: Bad control (occupation - post-treatment)
m3 = smf.ols("income ~ education + occupation", data=df_controls).fit()

# Model 4: Both controls (ability is good, occupation is bad)
m4 = smf.ols("income ~ education + ability + occupation", data=df_controls).fit()

# Compare education coefficients
control_comparison = pd.DataFrame(
    {
        "Model": [
            "No Controls",
            "Ability Only (GOOD)",
            "Occupation Only (BAD)",
            "Both Controls",
            "True Direct Effect",
        ],
        "Educ Coefficient": [
            m1.params["education"],
            m2.params["education"],
            m3.params["education"],
            m4.params["education"],
            1.5 + 2 * 0.5,
        ],  # True total
        "Assessment": [
            "Biased (confounding)",
            "Good estimate",
            "Biased (post-treatment)",
            "Biased (post-treatment dominates)",
            "True = 1.5 + 2*0.5 = 2.5",
        ],
    }
)

display(control_comparison.round(3))

# %% [markdown]
# :::{warning} The Central Lesson on Bad Controls
# :class: dropdown
#
# **Think causally about what determines what:**
#
# 1. **Draw a causal diagram** (DAG) showing relationships between variables
# 2. **Control for confounders** (variables that affect both treatment and outcome)
# 3. **Don't control for mediators** (variables on the causal path)
# 4. **Don't control for colliders** (variables affected by both treatment and outcome)
# 5. **When in doubt**, report results with and without the questionable control
#
# The goal is to isolate the causal effect, not to maximize R-squared by throwing in every available variable!
# :::
#
# ## Chapter Summary
#
# This chapter has provided a comprehensive introduction to multiple regression analysis, moving beyond the simple regression model to examine relationships between a dependent variable and multiple independent variables simultaneously. Through practical examples and theoretical foundations, we have covered the essential concepts and techniques that form the backbone of econometric analysis.
#
# ### Key Concepts Mastered
#
# **Multiple Regression Framework:** We learned why multiple regression is necessary for real-world analysis, where outcomes are typically influenced by several factors. The general multiple regression model allows us to examine ceteris paribus effects - the impact of each variable while holding others constant.
#
# **OLS Estimation and Interpretation:** We explored how Ordinary Least Squares extends to multiple regression, both conceptually and through matrix algebra. Each coefficient in multiple regression represents the partial effect of its corresponding variable, providing more nuanced and reliable insights than simple regression.
#
# **The Gauss-Markov Assumptions:** We examined the five key assumptions (MLR.1-MLR.5) that ensure OLS estimators are Best Linear Unbiased Estimators (BLUE). Understanding when these assumptions hold - and when they are violated - is crucial for proper model specification and interpretation.
#
# **Omitted Variable Bias:** Through examples like the wage equations, we demonstrated how excluding relevant variables can lead to biased estimates, emphasizing the importance of careful model specification.
#
# **Statistical Properties:** We covered the calculation and interpretation of standard errors, the role of multicollinearity, and tools like the Variance Inflation Factor (VIF) for assessing the precision of our estimates.
#
# ### Practical Applications
#
# Throughout this chapter, we applied multiple regression to diverse economic and social phenomena:
#
# * Educational outcomes (college GPA determinants)
# * Labor economics (wage determination)
# * Public policy (401k participation)
# * Criminology (factors affecting arrests)
# * Political economy (campaign spending and voting)
#
# These examples illustrate the versatility of multiple regression across different fields and research questions.
#
# ### Looking Forward
#
# The concepts and techniques mastered in this chapter form the foundation for more advanced topics in econometrics. Understanding multiple regression estimation, interpretation, and assumptions prepares you for hypothesis testing, model building, and addressing more complex econometric challenges.
#
# The ability to properly specify, estimate, and interpret multiple regression models is essential for empirical research in economics, business, and the social sciences. The tools and insights developed here will serve as building blocks for the more sophisticated econometric techniques explored in subsequent chapters.
#
# :::{note} Chapter Mastery Check
# :class: dropdown
#
# Before moving to the next chapter, ensure you can:
#
# * **Framework Understanding**: Articulate why multiple regression is essential, define all model components, and identify diverse applications across fields
# * **Parameter Interpretation**: Explain coefficients as ceteris paribus effects, handle nonlinear transformations, and distinguish from simple regression results
# * **Estimation Mechanics**: Apply OLS formulas, compute fitted values and residuals, and work with matrix representations
# * **Model Evaluation**: Interpret goodness of fit measures, compare models effectively, and understand coefficient relationships
# * **Assumption Analysis**: Master the five Gauss-Markov assumptions and their roles in statistical validity
# * **Theoretical Foundations**: Explain unbiasedness conditions and the significance of the Gauss-Markov theorem
# * **Bias Diagnosis**: Identify, calculate, and interpret the consequences of omitted variable bias
# * **Precision Analysis**: Derive variance formulas and compute standard errors for reliable inference
# * **Multicollinearity Assessment**: Use VIF measures to evaluate and address correlation among regressors
# * **Comprehensive Application**: Integrate all concepts to conduct complete empirical analyses with meaningful interpretations
#
# These consolidated competencies represent the essential skills for econometric analysis and provide the foundation for advanced topics in subsequent chapters.
# :::
