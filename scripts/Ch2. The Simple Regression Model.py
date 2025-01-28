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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # 2. The Simple Regression Model
#
# Welcome to this notebook, which delves into the fundamental concepts of the Simple Linear Regression model. This model is a cornerstone of econometrics and statistical analysis, used to understand the relationship between two variables. In this chapter, we will explore the mechanics of Ordinary Least Squares (OLS) regression, learn how to interpret its results, and understand the crucial assumptions that underpin its validity.
#
# We will use real-world datasets from the popular Wooldridge package to illustrate these concepts, making the theory come alive with practical examples. By the end of this notebook, you will have a solid grasp of simple regression, its properties, and how to apply it in Python.
#
# Let's start by importing the necessary libraries and setting up our environment.

# +
# Import required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf
import wooldridge as wool
from scipy import stats

# Set plotting style - Feel free to uncomment and customize these for your preferred look
# plt.style.use("seaborn")  # Consider 'ggplot', 'seaborn-v0_8-whitegrid', etc.
# sns.set_palette("deep")  # Explore 'viridis', 'magma', 'plasma', etc.
# plt.rcParams["figure.figsize"] = [10, 6]  # Adjust figure size as needed
# plt.rcParams["axes.grid"] = True  # Toggle gridlines on plots
# -

# ## 2.1 Simple OLS Regression
#
# The Simple Linear Regression model aims to explain the variation in a dependent variable, $y$, using a single independent variable, $x$. It assumes a linear relationship between $x$ and the expected value of $y$. The model is mathematically represented as:
#
# $$ y = \beta_0 + \beta_1 x + u $$
#
# Let's break down each component:
#
# - **$y$ (Dependent Variable)**: This is the variable we are trying to explain or predict. It's often called the explained variable, regressand, or outcome variable.
# - **$x$ (Independent Variable)**: This variable is used to explain the variations in $y$. It's also known as the explanatory variable, regressor, or control variable.
# - **$\beta_0$ (Intercept)**: This is the value of $y$ when $x$ is zero. It's the point where the regression line crosses the y-axis.
# - **$\beta_1$ (Slope Coefficient)**: This represents the change in $y$ for a one-unit increase in $x$. It quantifies the effect of $x$ on $y$.
# - **$u$ (Error Term)**: Also known as the disturbance term, it represents all other factors, besides $x$, that affect $y$. It captures the unexplained variation in $y$. We assume that the error term has an expected value of zero and is uncorrelated with $x$.
#
# Our goal in OLS regression is to estimate the unknown parameters $\beta_0$ and $\beta_1$. The Ordinary Least Squares (OLS) method achieves this by minimizing the sum of the squared residuals. The OLS estimators for $\beta_0$ and $\beta_1$ are given by the following formulas:
#
# $$\hat{\beta}_1 = \frac{\text{cov}(x,y)}{\text{var}(x)} = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^n (x_i - \bar{x})^2}$$
#
# This formula shows that $\hat{\beta}_1$ is the ratio of the sample covariance between $x$ and $y$ to the sample variance of $x$. It essentially captures the linear association between $x$ and $y$.
#
# $$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$
#
# Once we have $\hat{\beta}_1$, we can easily calculate $\hat{\beta}_0$ using this formula. It ensures that the regression line passes through the sample mean point $(\bar{x}, \bar{y})$.
#
# After estimating $\hat{\beta}_0$ and $\hat{\beta}_1$, we can compute the fitted values, which are the predicted values of $y$ for each observation based on our regression model:
#
# $$\hat{y} = \hat{\beta}_0 + \hat{\beta}_1 x$$
#
# These fitted values represent the points on the regression line.
#
# Let's put these formulas into practice with some real-world examples.
#
# ### Example 2.3: CEO Salary and Return on Equity
#
# In this example, we investigate the relationship between CEO salaries and the return on equity (ROE) of their firms. We want to see if firms with better performance (higher ROE) tend to pay their CEOs more. Our simple regression model is:
#
# $$ \text{salary} = \beta_0 + \beta_1 \text{roe} + u $$
#
# Here, `salary` is the dependent variable (CEO's annual salary in thousands of dollars), and `roe` is the independent variable (return on equity, in percentage). We hypothesize that $\beta_1 > 0$, meaning that a higher ROE is associated with a higher CEO salary. Let's calculate the OLS coefficients manually first to understand the underlying computations.

# +
# Load and prepare data
ceosal1 = wool.data("ceosal1")  # Load the ceosal1 dataset from wooldridge package
x = ceosal1["roe"]  # Extract 'roe' as the independent variable
y = ceosal1["salary"]  # Extract 'salary' as the dependent variable

# Calculate OLS coefficients manually
cov_xy = np.cov(x, y)[1, 0]  # Calculate the covariance between roe and salary
var_x = np.var(x, ddof=1)  # Calculate the variance of roe (ddof=1 for sample variance)
x_bar = np.mean(x)  # Calculate the mean of roe
y_bar = np.mean(y)  # Calculate the mean of salary

b1 = cov_xy / var_x  # Calculate beta_1_hat using the formula
b0 = y_bar - b1 * x_bar  # Calculate beta_0_hat using the formula

print("Manual calculation:")
print(f"Intercept (β₀): {b0:.2f}")  # Print the intercept, formatted to 2 decimal places
print(f"Slope (β₁): {b1:.2f}")  # Print the slope, formatted to 2 decimal places
# -

# The code first loads the `ceosal1` dataset and extracts the 'roe' and 'salary' columns as our $x$ and $y$ variables, respectively. Then, it calculates the covariance between `roe` and `salary`, the variance of `roe`, and the means of both variables. Finally, it applies the formulas to compute $\hat{\beta}_1$ and $\hat{\beta}_0$ and prints them.
#
# Now, let's use the `statsmodels` library, which provides a more convenient and comprehensive way to perform OLS regression. This will also serve as a verification of our manual calculations.

# +
# Fit regression model using statsmodels
reg = smf.ols(
    formula="salary ~ roe",
    data=ceosal1,
)  # Define the OLS regression model using formula notation
results = reg.fit()  # Fit the model to the data and store the results
b = results.params  # Extract the estimated coefficients

print("\nStatsmodels calculation:")
print(
    f"Intercept (β₀): {b.iloc[0]:.2f}",
)  # Print the intercept from statsmodels results, using iloc for position
print(
    f"Slope (β₁): {b.iloc[1]:.2f}",
)  # Print the slope from statsmodels results, using iloc for position


# -

# This code snippet uses `statsmodels.formula.api` to define and fit the same regression model. The `smf.ols` function takes a formula string (`"salary ~ roe"`) specifying the model and the dataframe (`ceosal1`) as input. `results.fit()` performs the OLS estimation, and `results.params` extracts the estimated coefficients. We can see that the coefficients obtained from `statsmodels` match our manual calculations, which is reassuring.
#
# To better visualize the regression results and the relationship between CEO salary and ROE, let's create an enhanced regression plot. We'll define a reusable function for this purpose, which includes the regression line, scatter plot of the data, confidence intervals, and annotations for the regression equation and R-squared.

def plot_regression(x, y, data, results, title, add_ci=True):
    """Create an enhanced regression plot with confidence intervals and annotations.

    Parameters
    ----------
    x : str
        Name of x variable in data
    y : str
        Name of y variable in data
    data : pandas.DataFrame
        Data containing x and y
    results : statsmodels results object
        Fitted regression results
    title : str
        Plot title
    add_ci : bool
        Whether to add confidence intervals

    """
    fig, ax = plt.subplots(
        figsize=(10, 6),
    )  # Create a figure and an axes object for the plot

    # Scatter plot of the actual data points
    sns.scatterplot(
        data=data,
        x=x,
        y=y,
        alpha=0.5,
        ax=ax,
    )  # Scatter plot of x vs y with some transparency

    # Regression line - Calculate predicted y values for a range of x values
    x_range = np.linspace(
        data[x].min(),
        data[x].max(),
        100,
    )  # Generate 100 evenly spaced x values
    y_pred = (
        results.params[0] + results.params[1] * x_range
    )  # Calculate predicted y values using the regression equation

    plt.plot(
        x_range,
        y_pred,
        color="red",
        label="Regression Line",
    )  # Plot the regression line in red

    if add_ci:
        # Add confidence intervals - Calculate confidence intervals for the predicted values
        y_hat = results.get_prediction(
            pd.DataFrame({x: x_range}),
        )  # Get prediction object for the x_range
        ci = y_hat.conf_int()  # Extract confidence interval bounds
        plt.fill_between(  # Fill the area between the confidence interval bounds
            x_range,
            ci[:, 0],  # Lower bound of confidence interval
            ci[:, 1],  # Upper bound of confidence interval
            color="red",
            alpha=0.1,  # Set transparency of the shaded area
            label="95% CI",  # Label for the confidence interval
        )

    # Add equation and R-squared as text annotations on the plot
    eq = f"y = {results.params[0]:.2f} + {results.params[1]:.2f}x"  # Format the regression equation
    r2 = f"R² = {results.rsquared:.3f}"  # Format the R-squared value
    plt.text(  # Add text to the plot
        0.05,
        0.95,
        eq + "\n" + r2,
        transform=ax.transAxes,  # Specify that coordinates are relative to the axes
        verticalalignment="top",  # Align text to the top
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.8,
        ),  # Add a white box around the text for better readability
    )

    plt.title(title)  # Set the title of the plot
    plt.xlabel(x)  # Set the label for the x-axis
    plt.ylabel(y)  # Set the label for the y-axis
    plt.legend()  # Display the legend
    plt.grid(True, alpha=0.3)  # Add grid lines with transparency
    plt.tight_layout()  # Adjust plot layout to prevent labels from overlapping


# This function, `plot_regression`, takes the variable names, data, regression results, and plot title as input. It generates a scatter plot of the data points, plots the regression line, and optionally adds 95% confidence intervals around the regression line. It also annotates the plot with the regression equation and the R-squared value. This function makes it easy to visualize and interpret simple regression results.

# Create enhanced regression plot for CEO Salary vs ROE
plot_regression("roe", "salary", ceosal1, results, "CEO Salary vs Return on Equity")

# Running this code will generate a scatter plot with 'roe' on the x-axis and 'salary' on the y-axis, along with the OLS regression line and its 95% confidence interval. The plot also displays the estimated regression equation and the R-squared value.
#
# **Interpretation of Example 2.3:**
#
# Looking at the output and the plot, we can interpret the results. The estimated regression equation (visible on the plot) will be something like:
#
# $$ \widehat{\text{salary}} = \hat{\beta}_0 + \hat{\beta}_1 \text{roe} $$
#
# Suppose we find $\hat{\beta}_1 \approx 18.50$. This means that, on average, for every one percentage point increase in ROE, CEO salary is predicted to increase by approximately \$18,500 (since salary is in thousands of dollars). The intercept, $\hat{\beta}_0$, represents the predicted salary when ROE is zero.  The R-squared value (also on the plot), let's say it is around 0.013, indicates that only about 1.3% of the variation in CEO salaries is explained by ROE in this simple linear model. This suggests that ROE alone is not a strong predictor of CEO salary, and other factors are likely more important.
#
# ### Example 2.4: Wage and Education
#
# Let's consider another example, examining the relationship between hourly wages and years of education. We use the `wage1` dataset and the following model:
#
# $$\text{wage} = \beta_0 + \beta_1 \text{educ} + u$$
#
# Here, `wage` is the hourly wage (in dollars), and `educ` is years of education. We expect a positive relationship, i.e., $\beta_1 > 0$, as more education is generally believed to lead to higher wages.

# +
# Load and analyze wage data
wage1 = wool.data("wage1")  # Load the wage1 dataset

# Fit regression model
reg = smf.ols(formula="wage ~ educ", data=wage1)  # Define and fit the OLS model
results = reg.fit()  # Fit the model

# Print regression results
print("Regression Results:")
print(
    f"Intercept (β₀): {results.params.iloc[0]:.2f}",
)  # Print the estimated intercept, using iloc for position
print(
    f"Education coefficient (β₁): {results.params.iloc[1]:.2f}",
)  # Print the estimated coefficient for education, using iloc for position
print(f"R-squared: {results.rsquared:.3f}")  # Print the R-squared value

# Create visualization
plot_regression(
    "educ",
    "wage",
    wage1,
    results,
    "Wage vs Years of Education",
)  # Generate regression plot
# -

# This code loads the `wage1` dataset, fits the regression model of `wage` on `educ` using `statsmodels`, and then prints the estimated coefficients and R-squared. Finally, it generates a regression plot using our `plot_regression` function.
#
# **Interpretation of Example 2.4:**
#
# Suppose we find $\hat{\beta}_1 \approx 0.54$. This implies that, on average, each additional year of education is associated with an increase in hourly wage of approximately \$0.54. The intercept, $\hat{\beta}_0$, represents the predicted wage for someone with zero years of education. If the R-squared is around 0.165, it means that about 16.5% of the variation in hourly wages is explained by years of education in this simple model. Education appears to be a somewhat more important factor in explaining wages than ROE was for CEO salaries, but still, a large portion of wage variation remains unexplained by education alone.
#
# ### Example 2.5: Voting Outcomes and Campaign Expenditures
#
# In this example, we explore the relationship between campaign spending and voting outcomes. We use the `vote1` dataset and the model:
#
# $$ \text{voteA} = \beta_0 + \beta_1 \text{shareA} + u $$
#
# Here, `voteA` is the percentage of votes received by candidate A, and `shareA` is the percentage of campaign spending by candidate A out of the total spending by both candidates. We expect that higher campaign spending share for candidate A will lead to a higher vote share, so we anticipate $\beta_1 > 0$.

# +
# Load and analyze voting data
vote1 = wool.data("vote1")  # Load the vote1 dataset

# Fit regression model
reg = smf.ols(formula="voteA ~ shareA", data=vote1)  # Define and fit the OLS model
results = reg.fit()  # Fit the model

# Print regression results
print("Regression Results:")
print(
    f"Intercept (β₀): {results.params.iloc[0]:.2f}",
)  # Print the estimated intercept, using iloc for position
print(
    f"Share coefficient (β₁): {results.params.iloc[1]:.2f}",
)  # Print the estimated coefficient for shareA, using iloc for position
print(f"R-squared: {results.rsquared:.3f}")  # Print the R-squared value

# Create visualization
plot_regression(
    "shareA",
    "voteA",
    vote1,
    results,
    "Vote Share vs Campaign Spending Share",
)  # Generate regression plot
# -

# This code follows the same pattern as the previous examples: load data, fit the regression model using `statsmodels`, print the coefficients and R-squared, and generate a regression plot.
#
# **Interpretation of Example 2.5:**
#
# Let's say we find $\hat{\beta}_1 \approx 0.30$. This suggests that for every one percentage point increase in candidate A's share of campaign spending, candidate A's vote share is predicted to increase by approximately 0.30 percentage points. The intercept, $\hat{\beta}_0$, represents the predicted vote share for candidate A if their campaign spending share is zero. If the R-squared is around 0.856, this is quite high! It indicates that about 85.6% of the variation in candidate A's vote share is explained by their share of campaign spending in this simple model. This suggests that campaign spending share is a very strong predictor of voting outcomes, at least in this dataset.
#
# ## 2.2. Coefficients, Fitted Values, and Residuals
#
# As we discussed earlier, after estimating the OLS regression, we obtain fitted values ($\hat{y}_i$) and residuals ($\hat{u}_i$). Let's formally define them again:
#
# **Fitted Values**: These are the predicted values of $y$ for each observation $i$, calculated using the estimated regression equation:
#
# $$\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i$$
#
# Fitted values lie on the OLS regression line. For each $x_i$, $\hat{y}_i$ is the point on the line directly above or below $x_i$.
#
# **Residuals**: These are the differences between the actual values of $y_i$ and the fitted values $\hat{y}_i$. They represent the unexplained part of $y_i$ for each observation:
#
# $$\hat{u}_i = y_i - \hat{y}_i$$
#
# Residuals are estimates of the unobservable error terms $u_i$. In OLS regression, we aim to minimize the sum of squared residuals.
#
# ### Example 2.6: CEO Salary and Return on Equity
#
# Let's go back to the CEO salary and ROE example and examine the fitted values and residuals. We will calculate these and present the first 15 observations in a table. We will also create a residual plot to visualize the residuals.

# +
# Prepare regression results - Re-run the regression for ceosal1 dataset
ceosal1 = wool.data("ceosal1")  # Load data again (if needed)
reg = smf.ols(formula="salary ~ roe", data=ceosal1)  # Define the regression model
results = reg.fit()  # Fit the model

# Calculate fitted values and residuals
salary_hat = results.fittedvalues  # Get fitted values from results object
u_hat = results.resid  # Get residuals from results object

# Create summary table
table = pd.DataFrame(  # Create a Pandas DataFrame
    {
        "ROE": ceosal1["roe"],  # Include ROE values
        "Actual Salary": ceosal1["salary"],  # Include actual salary values
        "Predicted Salary": salary_hat,  # Include fitted salary values
        "Residual": u_hat,  # Include residual values
    },
)

# Format and display the first 15 rows
pd.set_option(
    "display.float_format",
    lambda x: "%.2f" % x,
)  # Set float format for display
print("First 15 observations:")
print(table.head(15))  # Print the first 15 rows of the table

# Create residual plot - Scatter plot of fitted values against residuals
plt.figure(figsize=(10, 6))  # Create a figure for the plot
sns.scatterplot(
    x=salary_hat,
    y=u_hat,
    alpha=0.5,
)  # Scatter plot of fitted values vs residuals
plt.axhline(y=0, color="r", linestyle="--")  # Add a horizontal line at y=0
plt.title("Residual Plot")  # Set the title of the plot
plt.xlabel("Fitted Values")  # Set the x-axis label
plt.ylabel("Residuals")  # Set the y-axis label
plt.grid(True, alpha=0.3)  # Add grid lines
plt.tight_layout()  # Adjust layout
# -

# This code calculates the fitted values and residuals using the `results` object from our previous regression. It then creates a Pandas DataFrame to display ROE, actual salary, predicted salary, and residuals for each observation. We print the first 15 rows of this table. Finally, it generates a residual plot, which is a scatter plot of fitted values against residuals, with a horizontal line at zero.
#
# **Interpretation of Example 2.6:**
#
# By examining the table, you can see for each company the actual CEO salary, the salary predicted by the regression model based on ROE, and the residual, which is the difference between the actual and predicted salary. A positive residual means the actual salary is higher than predicted, and a negative residual means it's lower.
#
# The residual plot is useful for checking some of the assumptions of OLS regression, particularly the assumption of homoscedasticity (constant variance of errors). Ideally, in a good regression model, we would expect to see residuals randomly scattered around zero, with no systematic pattern. In this plot, we are looking for patterns such as the spread of residuals increasing or decreasing as fitted values change, which could indicate heteroscedasticity.
#
# ### Example 2.7: Wage and Education
#
# Let's verify some important properties of OLS residuals using the wage and education example. These properties are mathematical consequences of the OLS minimization process:
#
# 1. **The sum of residuals is zero**: $\sum_{i=1}^n \hat{u}_i = 0$. This implies that the mean of residuals is also zero: $\frac{1}{n}\sum_{i=1}^n \hat{u}_i = 0$.
# 2. **The sample covariance between regressors and residuals is zero**: $\sum_{i=1}^n x_i \hat{u}_i = 0$. This means that the residuals are uncorrelated with the independent variable $x$.
# 3. **The point $(\bar{x}, \bar{y})$ lies on the regression line**: This is ensured by the formula for $\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$.
#
# Let's check these properties using the `wage1` dataset.

# +
# Load and prepare data - Re-run the regression for wage1 dataset
wage1 = wool.data("wage1")  # Load wage1 data
reg = smf.ols(formula="wage ~ educ", data=wage1)  # Define regression model
results = reg.fit()  # Fit the model

# Get coefficients, fitted values and residuals
b = results.params  # Extract coefficients
wage_hat = results.fittedvalues  # Extract fitted values
u_hat = results.resid  # Extract residuals

# Property 1: Mean of residuals should be zero
u_hat_mean = np.mean(u_hat)  # Calculate the mean of residuals
print(f"Mean of residuals: {u_hat_mean:.10f}")  # Print the mean of residuals

# Property 2: Covariance between education and residuals should be zero
educ_u_cov = np.cov(wage1["educ"], u_hat)[
    1,
    0,
]  # Calculate covariance between educ and residuals
print(
    f"Covariance between education and residuals: {educ_u_cov:.10f}",
)  # Print the covariance

# Property 3: Point (x̄, ȳ) lies on regression line
educ_mean = np.mean(wage1["educ"])  # Calculate mean of education
wage_mean = np.mean(wage1["wage"])  # Calculate mean of wage
wage_pred = (
    b.iloc[0] + b.iloc[1] * educ_mean
)  # Predict wage at mean education using regression line

print("\nVerifying that (x̄, ȳ) lies on regression line:")
wage_mean = np.mean(
    wage1["wage"],
)  # Recalculate mean of wage (already calculated above, but for clarity)
print(f"wage_mean: {wage_mean}")  # Print the mean wage
print(f"Predicted wage at mean education: {wage_pred:.6f}")  # Print the predicted wage
# -

# This code calculates the mean of the residuals, the covariance between education and residuals, and verifies that the predicted wage at the mean level of education is equal to the mean wage.
#
# **Interpretation of Example 2.7:**
#
# The output should show that the mean of residuals is very close to zero (practically zero, given potential floating-point inaccuracies). Similarly, the covariance between education and residuals should be very close to zero. Finally, the predicted wage at the average level of education should be very close to the average wage. These results confirm the mathematical properties of OLS residuals. These properties are not assumptions, but rather outcomes of the OLS estimation procedure.
#
# ## 2.3. Goodness of Fit
#
# After fitting a regression model, it's important to assess how well the model fits the data. A key measure of goodness of fit in simple linear regression is the R-squared ($R^2$) statistic. R-squared measures the proportion of the total variation in the dependent variable ($y$) that is explained by the independent variable ($x$) in our model.
#
# To understand R-squared, we first need to define three sums of squares:
#
# **Total Sum of Squares (SST)**: This measures the total sample variation in $y$. It is the sum of squared deviations of $y_i$ from its mean $\bar{y}$:
#
# $$\text{SST} = \sum_{i=1}^n (y_i - \bar{y})^2 = (n-1) \text{var}(y)$$
#
# SST represents the total variability in the dependent variable that we want to explain.
#
# **Explained Sum of Squares (SSE)**: This measures the variation in $\hat{y}$ predicted by our model. It is the sum of squared deviations of the fitted values $\hat{y}_i$ from the mean of $y$, $\bar{y}$:
#
# $$\text{SSE} = \sum_{i=1}^n (\hat{y}_i - \bar{y})^2 = (n-1) \text{var}(\hat{y})$$
#
# SSE represents the variability in $y$ that is explained by our model.
#
# **Residual Sum of Squares (SSR)**: This measures the variation in the residuals $\hat{u}_i$, which is the unexplained variation in $y$. It is the sum of squared residuals:
#
# $$\text{SSR} = \sum_{i=1}^n (\hat{u}_i - 0)^2 = \sum_{i=1}^n \hat{u}_i^2 = (n-1) \text{var}(\hat{u})$$
#
# Note that the mean of residuals is zero, so we are summing squared deviations from zero here. SSR represents the variability in $y$ that is *not* explained by our model.
#
# These three sums of squares are related by the following identity:
#
# $$\text{SST} = \text{SSE} + \text{SSR}$$
#
# This equation states that the total variation in $y$ can be decomposed into the variation explained by the model (SSE) and the unexplained variation (SSR).
#
# Now we can define R-squared:
#
# $$R^2 = \frac{\text{SSE}}{\text{SST}} = 1 - \frac{\text{SSR}}{\text{SST}} = \frac{\text{var}(\hat{y})}{\text{var}(y)} = 1 - \frac{\text{var}(\hat{u})}{\text{var}(y)}$$
#
# R-squared is the ratio of the explained variation to the total variation. It ranges from 0 to 1 (or 0% to 100%).
#
# - $R^2 = 0$ means the model explains none of the variation in $y$. In this case, SSE = 0 and SSR = SST.
# - $R^2 = 1$ means the model explains all of the variation in $y$. In this case, SSR = 0 and SSE = SST.
#
# A higher R-squared generally indicates a better fit, but it's important to remember that a high R-squared does not necessarily mean that the model is good or that there is a causal relationship. R-squared only measures the strength of the linear relationship and the proportion of variance explained.
#
# ### Example 2.8: CEO Salary and Return on Equity
#
# Let's calculate and compare R-squared for the CEO salary and ROE example using different formulas. We will also create visualizations to understand the concept of goodness of fit.

# +
# Load and prepare data - Re-run regression for ceosal1
ceosal1 = wool.data("ceosal1")  # Load data
reg = smf.ols(formula="salary ~ roe", data=ceosal1)  # Define regression model
results = reg.fit()  # Fit the model

# Calculate predicted values & residuals
sal_hat = results.fittedvalues  # Get fitted values
u_hat = results.resid  # Get residuals
sal = ceosal1["salary"]  # Get actual salary values

# Calculate R² in three different ways - Using different formulas for R-squared
R2_a = np.var(sal_hat, ddof=1) / np.var(sal, ddof=1)  # R² = var(ŷ)/var(y)
R2_b = 1 - np.var(u_hat, ddof=1) / np.var(sal, ddof=1)  # R² = 1 - var(û)/var(y)
R2_c = np.corrcoef(sal, sal_hat)[1, 0] ** 2  # R² = correlation(y, ŷ)²

print("R-squared calculations:")
print(
    f"Using var(ŷ)/var(y): {R2_a:.4f}",
)  # Print R-squared calculated using variance of fitted values
print(
    f"Using 1 - var(û)/var(y): {R2_b:.4f}",
)  # Print R-squared calculated using variance of residuals
print(
    f"Using correlation coefficient: {R2_c:.4f}",
)  # Print R-squared calculated using correlation coefficient

# Create visualization of model fit - Two subplots: Actual vs Predicted and Residuals vs Fitted
plt.figure(figsize=(12, 5))

# Actual vs Predicted plot - Scatter plot of actual salary against predicted salary
plt.subplot(1, 2, 1)  # First subplot in a 1x2 grid
sns.scatterplot(x=sal, y=sal_hat, alpha=0.5)  # Scatter plot
plt.plot(
    [sal.min(), sal.max()],
    [sal.min(), sal.max()],
    "r--",
    label="Perfect Fit",
)  # 45-degree line for perfect fit
plt.title("Actual vs Predicted Salary")  # Set title
plt.xlabel("Actual Salary")  # Set x-axis label
plt.ylabel("Predicted Salary")  # Set y-axis label
plt.legend()  # Show legend

# Residuals vs Fitted plot - Scatter plot of fitted salary against residuals
plt.subplot(1, 2, 2)  # Second subplot in a 1x2 grid
sns.scatterplot(x=sal_hat, y=u_hat, alpha=0.5)  # Scatter plot
plt.axhline(y=0, color="r", linestyle="--", label="Zero Line")  # Horizontal line at y=0
plt.title("Residuals vs Fitted Values")  # Set title
plt.xlabel("Fitted Values")  # Set x-axis label
plt.ylabel("Residuals")  # Set y-axis label
plt.legend()  # Show legend

plt.tight_layout()  # Adjust layout
# -

# This code calculates R-squared in three different ways using the formulas we discussed. All three methods should yield the same R-squared value (within rounding errors). It then creates two plots side-by-side:
# 1. **Actual vs Predicted Plot**: This plot shows actual salary on the x-axis and predicted salary on the y-axis. If the model fit were perfect (R² = 1), all points would lie on the 45-degree dashed red line. The closer the points are to this line, the better the fit.
# 2. **Residuals vs Fitted Values Plot**: We already saw this plot in Example 2.6. In the context of goodness of fit, we can see how the residuals are distributed relative to the fitted values. Ideally, residuals should be randomly scattered around zero with no discernible pattern, indicating that the model has captured all systematic variation.
#
# **Interpretation of Example 2.8:**
#
# The R-squared value calculated (around 0.013 in our example) will be the same regardless of which formula is used, confirming their equivalence. The Actual vs Predicted plot will show how well the predicted salaries align with the actual salaries. Points clustered closer to the 45-degree line indicate a better fit. The Residuals vs Fitted Values plot helps to visually assess if the variance of residuals is constant across fitted values and if there are any patterns in residuals that might suggest model inadequacy.
#
# ### Example 2.9: Voting Outcomes and Campaign Expenditures
#
# Let's examine the complete regression summary for the voting outcomes and campaign expenditures example, including R-squared and other statistical measures. We will also create an enhanced visualization with a 95% confidence interval.

# +
# Load and analyze voting data - Re-run regression for vote1 dataset
vote1 = wool.data("vote1")  # Load data

# Fit regression model
reg = smf.ols(formula="voteA ~ shareA", data=vote1)  # Define model
results = reg.fit()  # Fit model

# Create a clean summary table - Extract key statistics from results object
summary_stats = pd.DataFrame(  # Create a DataFrame for summary statistics
    {
        "Coefficient": results.params,  # Estimated coefficients
        "Std. Error": results.bse,  # Standard errors of coefficients
        "t-value": results.tvalues,  # t-statistics for coefficients
        "p-value": results.pvalues,  # p-values for coefficients
    },
)

print("Regression Summary:")
print("==================")
print(f"R-squared: {results.rsquared:.4f}")  # Print R-squared
print(f"Adjusted R-squared: {results.rsquared_adj:.4f}")  # Print Adjusted R-squared
print(f"F-statistic: {results.fvalue:.2f}")  # Print F-statistic
print(f"Number of observations: {results.nobs}")  # Print number of observations
print("\nCoefficient Estimates:")
print("====================")
print(
    summary_stats.round(4),
)  # Print summary statistics table, rounded to 4 decimal places

# Create enhanced visualization - Regression plot with confidence interval
plt.figure(figsize=(10, 6))
plot_regression(
    "shareA",
    "voteA",
    vote1,
    results,
    "Vote Share vs Campaign Spending Share\nwith 95% Confidence Interval",  # Title with CI mention
)
# -

# This code fits the regression model for voting outcomes and campaign spending share. It then creates a summary table containing coefficient estimates, standard errors, t-values, and p-values. It prints this summary, along with R-squared, adjusted R-squared, F-statistic, and the number of observations. Finally, it generates an enhanced regression plot, including a 95% confidence interval, using our `plot_regression` function.
#
# **Interpretation of Example 2.9:**
#
# The output will provide a comprehensive summary of the regression results. You can see the estimated coefficients for the intercept and `shareA`, their standard errors, t-values, and p-values. The p-values are used for hypothesis testing (we will discuss this in detail in later chapters). The R-squared and Adjusted R-squared values indicate the goodness of fit. Adjusted R-squared is a modified version of R-squared that adjusts for the number of regressors in the model (it is more relevant in multiple regression). The F-statistic is used for testing the overall significance of the regression model.
#
# The enhanced regression plot visually represents the relationship between `shareA` and `voteA`, along with the 95% confidence interval around the regression line, providing a visual sense of the uncertainty in our predictions.
#
# ## 2.4 Nonlinearities
#
# So far, we have focused on linear relationships between variables. However, in many cases, the relationship between the dependent and independent variables might be nonlinear. We can still use linear regression techniques to model certain types of nonlinear relationships by transforming the variables. Common transformations include using logarithms of variables. Let's consider three common models involving logarithms:
#
# 1. **Log-Level Model**: In this model, the dependent variable is in logarithm form, while the independent variable is in level form:
#
#    $$\log(y) = \beta_0 + \beta_1 x + u$$
#
#    In this model, $\beta_1$ is interpreted as the approximate percentage change in $y$ for a one-unit change in $x$. Specifically, a one-unit increase in $x$ is associated with a $100 \cdot \beta_1$ percent change in $y$.
#
# 2. **Level-Log Model**: Here, the dependent variable is in level form, and the independent variable is in logarithm form:
#
#    $$y = \beta_0 + \beta_1 \log(x) + u$$
#
#    In this model, $\beta_1$ represents the change in $y$ for a one-percentage point change in $x$. Specifically, a one-percent increase in $x$ is associated with a change in $y$ of $\beta_1/100$. Or more simply, a 100 percent increase in x is associated with a change in y of $\beta_1$.
#
# 3. **Log-Log Model**: In this model, both the dependent and independent variables are in logarithm form:
#
#    $$\log(y) = \beta_0 + \beta_1 \log(x) + u$$
#
#    In the log-log model, $\beta_1$ is interpreted as the elasticity of $y$ with respect to $x$. That is, a one-percent increase in $x$ is associated with a $\beta_1$ percent change in $y$.
#
# ### Example 2.10: Wage and Education (Log-Level Model)
#
# Let's revisit the wage and education example and estimate a log-level model:
#
# $$\log(\text{wage}) = \beta_0 + \beta_1 \text{educ} + u$$
#
# In this model, we are interested in the percentage increase in wage for each additional year of education.

# +
# Load and prepare data - Re-use wage1 dataset
wage1 = wool.data("wage1")  # Load data (if needed)

# Estimate log-level model - Define and fit the model with log(wage) as dependent variable
reg = smf.ols(
    formula="np.log(wage) ~ educ",
    data=wage1,
)  # Use np.log() to take logarithm of wage
results = reg.fit()  # Fit the model

print("Log-Level Model Results:")
print(
    f"Intercept (β₀): {results.params.iloc[0]:.4f}",
)  # Print estimated intercept, using iloc for position
print(
    f"Education coefficient (β₁): {results.params.iloc[1]:.4f}",
)  # Print estimated education coefficient, using iloc for position
print(f"R-squared: {results.rsquared:.4f}")  # Print R-squared

# Create visualization - Scatter plot of educ vs log(wage) and regression line
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=wage1,
    x="educ",
    y=np.log(wage1["wage"]),
    alpha=0.5,
)  # Scatter plot of educ vs log(wage)
plt.plot(
    wage1["educ"],
    results.fittedvalues,
    color="red",
    label="Regression Line",
)  # Plot regression line
plt.title("Log(Wage) vs Years of Education")  # Set title
plt.xlabel("Years of Education")  # Set x-axis label
plt.ylabel("Log(Wage)")  # Set y-axis label
plt.legend()  # Show legend
plt.grid(True, alpha=0.3)  # Add grid lines
plt.tight_layout()  # Adjust layout
# -

# This code estimates the log-level model using `statsmodels`. We use `np.log(wage)` as the dependent variable in the formula. It then prints the estimated coefficients and R-squared and generates a scatter plot of education against log(wage) with the regression line.
#
# **Interpretation of Example 2.10:**
#
# Suppose we find $\hat{\beta}_1 \approx 0.092$. In the log-level model, this coefficient can be interpreted as the approximate percentage change in wage for a one-unit increase in education. So, approximately, each additional year of education is associated with a 9.2% increase in hourly wage. The intercept $\hat{\beta}_0$ represents the predicted log(wage) when education is zero. The R-squared value indicates the proportion of variation in $\log(\text{wage})$ explained by education.
#
# ### Example 2.11: CEO Salary and Firm Sales (Log-Log Model)
#
# Let's consider the CEO salary and firm sales relationship and estimate a log-log model:
#
# $$\log(\text{salary}) = \beta_0 + \beta_1 \log(\text{sales}) + u$$
#
# In this model, $\beta_1$ represents the elasticity of CEO salary with respect to firm sales.

# +
# Load and prepare data - Re-use ceosal1 dataset
ceosal1 = wool.data("ceosal1")  # Load data (if needed)

# Estimate log-log model - Define and fit model with log(salary) and log(sales)
reg = smf.ols(
    formula="np.log(salary) ~ np.log(sales)",
    data=ceosal1,
)  # Use np.log() for both salary and sales
results = reg.fit()  # Fit the model

print("Log-Log Model Results:")
print(
    f"Intercept (β₀): {results.params.iloc[0]:.4f}",
)  # Print estimated intercept, using iloc for position
print(
    f"Sales elasticity (β₁): {results.params.iloc[1]:.4f}",
)  # Print estimated elasticity (sales coefficient), using iloc for position
print(f"R-squared: {results.rsquared:.4f}")  # Print R-squared

# Create visualization - Scatter plot of log(sales) vs log(salary) and regression line
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=ceosal1,
    x=np.log(ceosal1["sales"]),
    y=np.log(ceosal1["salary"]),
    alpha=0.5,
)  # Scatter plot of log(sales) vs log(salary)
plt.plot(
    np.log(ceosal1["sales"]),
    results.fittedvalues,
    color="red",
    label="Regression Line",
)  # Plot regression line
plt.title("Log(Salary) vs Log(Sales)")  # Set title
plt.xlabel("Log(Sales)")  # Set x-axis label
plt.ylabel("Log(Salary)")  # Set y-axis label
plt.legend()  # Show legend
plt.grid(True, alpha=0.3)  # Add grid lines
plt.tight_layout()  # Adjust layout
# -

# This code estimates the log-log model using `statsmodels`. We use `np.log(salary)` as the dependent variable and `np.log(sales)` as the independent variable. It prints the estimated coefficients and R-squared and generates a scatter plot of log(sales) against log(salary) with the regression line.
#
# **Interpretation of Example 2.11:**
#
# Suppose we find $\hat{\beta}_1 \approx 0.257$. In the log-log model, this coefficient is the elasticity of salary with respect to sales. It means that a 1% increase in firm sales is associated with approximately a 0.257% increase in CEO salary. The intercept $\hat{\beta}_0$ does not have a direct practical interpretation in terms of original variables in this model, but it is needed for the regression equation. The R-squared value indicates the proportion of variation in $\log(\text{salary})$ explained by $\log(\text{sales})$.
#
# ## 2.5. Regression through the Origin and Regression on a Constant
#
# In standard simple linear regression, we include an intercept term ($\beta_0$). However, in some cases, it might be appropriate to omit the intercept, forcing the regression line to pass through the origin (0,0). This is called **regression through the origin**. The model becomes:
#
# $$y = \beta_1 x + u$$
#
# In `statsmodels`, you can perform regression through the origin by including `0 +` or `- 1 +` in the formula, like `"salary ~ 0 + roe"` or `"salary ~ roe - 1"`.
#
# Another special case is **regression on a constant only**, where we only estimate the intercept and do not include any independent variable. The model is:
#
# $$y = \beta_0 + u$$
#
# In this case, $\hat{\beta}_0$ is simply the sample mean of $y$, $\bar{y}$. This model essentially predicts the same value ($\bar{y}$) for all observations, regardless of any other factors. In `statsmodels`, you can specify this as `"salary ~ 1"` or `"salary ~ constant"`.
#
# Let's compare these three regression specifications using the CEO salary and ROE example.

# +
# Load and prepare data - Re-use ceosal1 dataset
ceosal1 = wool.data("ceosal1")  # Load data (if needed)

# 1. Regular OLS regression - Regression with intercept
reg1 = smf.ols(formula="salary ~ roe", data=ceosal1)  # Standard regression model
results1 = reg1.fit()  # Fit model 1

# 2. Regression through origin - Regression without intercept
reg2 = smf.ols(
    formula="salary ~ 0 + roe",
    data=ceosal1,
)  # Regression through origin (no intercept)
results2 = reg2.fit()  # Fit model 2

# 3. Regression on constant only - Regression with only intercept (mean model)
reg3 = smf.ols(formula="salary ~ 1", data=ceosal1)  # Regression on constant only
results3 = reg3.fit()  # Fit model 3

# Print results - Compare coefficients and R-squared for all three models
print("Regression Comparisons:")
print("======================")
print("\n1. Regular regression:")
print(
    f"β₀: {results1.params.iloc[0]:.2f}",
)  # Intercept from model 1, using iloc for position
print(
    f"β₁: {results1.params.iloc[1]:.2f}",
)  # Slope from model 1, using iloc for position
print(f"R²: {results1.rsquared:.4f}")  # R-squared from model 1

print("\n2. Regression through origin:")
print(
    f"β₁: {results2.params.iloc[0]:.2f}",
)  # Slope from model 2, using iloc for position
print(f"R²: {results2.rsquared:.4f}")  # R-squared from model 2

print("\n3. Regression on constant:")
print(
    f"β₀: {results3.params.iloc[0]:.2f}",
)  # Intercept (mean) from model 3, using iloc for position
print(f"R²: {results3.rsquared:.4f}")  # R-squared from model 3 (will be 0)

# Create visualization - Plot data and regression lines for all three models
plt.figure(figsize=(10, 6))
plt.scatter(
    ceosal1["roe"],
    ceosal1["salary"],
    alpha=0.5,
    label="Data",
)  # Scatter plot of data points
x_range = np.linspace(
    ceosal1["roe"].min(),
    ceosal1["roe"].max(),
    100,
)  # Generate x values for regression lines

# Plot all regression lines - Plot regression lines for each model
plt.plot(
    x_range,
    results1.params[0] + results1.params[1] * x_range,
    "r-",
    label="Regular regression",  # Regular OLS regression line
)
plt.plot(
    x_range,
    results2.params[0] * x_range,
    "g--",
    label="Through origin",
)  # Regression through origin line
plt.plot(
    [x_range.min(), x_range.max()],
    [results3.params[0], results3.params[0]],
    "b-.",
    label="Constant only",  # Regression on constant (horizontal line at mean)
)

plt.title("Comparison of Different Regression Specifications")  # Set title
plt.xlabel("Return on Equity (ROE)")  # Set x-axis label
plt.ylabel("Salary")  # Set y-axis label
plt.legend()  # Show legend
plt.grid(True, alpha=0.3)  # Add grid lines
plt.tight_layout()  # Adjust layout
# -

# This code estimates three regression models: regular OLS, regression through the origin, and regression on a constant only. It prints the estimated coefficients and R-squared for each model and then generates a scatter plot of the data with all three regression lines plotted.
#
# **Interpretation of Example 2.12:**
#
# By comparing the results, you can see how the estimated coefficients and R-squared values differ across these specifications. Regression through the origin forces the line to go through (0,0), which may or may not be appropriate depending on the context. Regression on a constant simply gives you the mean of the dependent variable as the prediction and will always have an R-squared of 0 (unless variance of y is also zero, which is trivial case).
#
# In the visualization, you can visually compare the fit of these different regression lines to the data. Notice that R-squared for regression through the origin can sometimes be higher or lower than regular OLS, and it should be interpreted cautiously as the total sum of squares is calculated differently in regression through the origin. Regression on a constant will be a horizontal line at the mean of salary.
#
# ## 2.6. Expected Values, Variances, and Standard Errors
#
# To understand the statistical properties of OLS estimators, we need to make certain assumptions about the population regression model and the data. These are known as the **Classical Linear Model (CLM) assumptions** for simple linear regression. The first five are often called **SLR assumptions**.
#
# Here are the five key assumptions for Simple Linear Regression:
#
# 1. **SLR.1: Linear Population Regression Function**: The relationship between $y$ and $x$ in the population is linear:
#    $$y = \beta_0 + \beta_1 x + u$$
#    This assumes that the model is correctly specified in terms of linearity.
#
# 2. **SLR.2: Random Sampling**: We have a random sample of size $n$, $\{(x_i, y_i): i=1, 2, ..., n\}$, from the population model.
#    This assumption ensures that our sample is representative of the population we want to study.
#
# 3. **SLR.3: Sample Variation in x**: There is sample variation in the independent variable $x$, i.e., $x_i$ are not all the same value.
#    If there is no variation in $x$, we cannot estimate the relationship between $x$ and $y$.
#
# 4. **SLR.4: Zero Conditional Mean**: The error term $u$ has an expected value of zero given any value of $x$:
#    $$\text{E}(u|x) = 0$$
#    This is a crucial assumption. It implies that the unobserved factors represented by $u$ are, on average, unrelated to $x$. If $x$ and $u$ are correlated, OLS estimators will be biased.
#
# 5. **SLR.5: Homoscedasticity**: The error term $u$ has the same variance given any value of $x$:
#    $$\text{var}(u|x) = \sigma^2$$
#    This assumption means that the spread of the errors is constant across all values of $x$. If this assumption is violated (heteroscedasticity), OLS estimators are still unbiased, but they are no longer the Best Linear Unbiased Estimators (BLUE), and standard errors will be incorrect.
#
# Under these assumptions, we have important theoretical results about the OLS estimators:
#
# - **Theorem 2.1: Unbiasedness of OLS Estimators**: Under assumptions SLR.1-SLR.4, the OLS estimators $\hat{\beta}_0$ and $\hat{\beta}_1$ are unbiased estimators of $\beta_0$ and $\beta_1$, respectively. That is,
#   $$\text{E}(\hat{\beta}_0) = \beta_0 \quad \text{and} \quad \text{E}(\hat{\beta}_1) = \beta_1$$
#   Unbiasedness means that on average, across many random samples, the OLS estimates will be equal to the true population parameters.
#
# - **Theorem 2.2: Variances of OLS Estimators**: Under assumptions SLR.1-SLR.5, the variances of the OLS estimators are given by:
#   $$\text{var}(\hat{\beta}_0) = \frac{\sigma^2 \sum_{i=1}^n x_i^2}{n \sum_{i=1}^n (x_i - \bar{x})^2} = \sigma^2 \frac{\frac{1}{n}\sum_{i=1}^n x_i^2}{\text{SST}_x}$$
#   $$\text{var}(\hat{\beta}_1) = \frac{\sigma^2}{\sum_{i=1}^n (x_i - \bar{x})^2} = \frac{\sigma^2}{\text{SST}_x}$$
#   where $\text{SST}_x = \sum_{i=1}^n (x_i - \bar{x})^2$ is the total sum of squares for $x$.
#
# To make these variance formulas practically useful, we need to estimate the error variance $\sigma^2$. An unbiased estimator of $\sigma^2$ is the **standard error of regression (SER)**, denoted as $\hat{\sigma}^2$:
#
# $$\hat{\sigma}^2 = \frac{1}{n-2} \cdot \sum_{i=1}^n \hat{u}_i^2 = \frac{\text{SSR}}{n-2}$$
#
# The denominator $n-2$ reflects the degrees of freedom, as we lose two degrees of freedom when estimating $\beta_0$ and $\beta_1$. Notice that $\hat{\sigma}^2 = \frac{n-1}{n-2} \cdot \text{var}(\hat{u}_i)$, which is a slight adjustment to the sample variance of residuals to get an unbiased estimator of $\sigma^2$.
#
# Using $\hat{\sigma}^2$, we can estimate the standard errors of $\hat{\beta}_0$ and $\hat{\beta}_1$:
#
# $$\text{se}(\hat{\beta}_0) = \sqrt{\widehat{\text{var}}(\hat{\beta}_0)} = \sqrt{\frac{\hat{\sigma}^2 \sum_{i=1}^n x_i^2}{n \sum_{i=1}^n (x_i - \bar{x})^2}} = \sqrt{\frac{\hat{\sigma}^2 [\frac{1}{n}\sum_{i=1}^n x_i^2]}{\text{SST}_x}}$$
#
# $$\text{se}(\hat{\beta}_1) = \sqrt{\widehat{\text{var}}(\hat{\beta}_1)} = \sqrt{\frac{\hat{\sigma}^2}{\sum_{i=1}^n (x_i - \bar{x})^2}} = \sqrt{\frac{\hat{\sigma}^2}{\text{SST}_x}}$$
#
# Standard errors measure the precision of our coefficient estimates. Smaller standard errors indicate more precise estimates. They are crucial for hypothesis testing and constructing confidence intervals (which we will cover in subsequent chapters).
#
# ### Example 2.12: Student Math Performance and the School Lunch Program
#
# Let's consider an example using the `meap93` dataset, examining the relationship between student math performance and the percentage of students eligible for the school lunch program. The model is:
#
# $$\text{math10} = \beta_0 + \beta_1 \text{lnchprg} + u$$
#
# where `math10` is the percentage of students passing a math test, and `lnchprg` is the percentage of students eligible for the lunch program. We expect a negative relationship, i.e., $\beta_1 < 0$, as higher lunch program participation (indicating more poverty) might be associated with lower math scores.

# +
# Load and analyze data - Use meap93 dataset
meap93 = wool.data("meap93")  # Load data

# Estimate model - Regular OLS regression
reg = smf.ols(formula="math10 ~ lnchprg", data=meap93)  # Define model
results = reg.fit()  # Fit model

# Calculate SER manually - Calculate Standard Error of Regression manually
n = results.nobs  # Number of observations
u_hat_var = np.var(results.resid, ddof=1)  # Sample variance of residuals
SER = np.sqrt(u_hat_var * (n - 1) / (n - 2))  # Calculate SER using formula

# Calculate standard errors manually - Calculate standard errors of beta_0 and beta_1 manually
lnchprg_var = np.var(meap93["lnchprg"], ddof=1)  # Sample variance of lnchprg
lnchprg_mean = np.mean(meap93["lnchprg"])  # Mean of lnchprg
lnchprg_sq_mean = np.mean(meap93["lnchprg"] ** 2)  # Mean of squared lnchprg

se_b1 = SER / (np.sqrt(lnchprg_var) * np.sqrt(n - 1))  # Standard error of beta_1
se_b0 = se_b1 * np.sqrt(lnchprg_sq_mean)  # Standard error of beta_0

print("Manual Calculations:")
print(f"SER: {SER:.4f}")  # Print manually calculated SER
print(f"SE(β₀): {se_b0:.4f}")  # Print manually calculated SE(beta_0)
print(f"SE(β₁): {se_b1:.4f}")  # Print manually calculated SE(beta_1)

print("\nStatsmodels Results:")
print(
    results.summary().tables[1],
)  # Print statsmodels summary table (coefficients, SEs, t-values, p-values)

# Create visualization with confidence intervals - Regression plot with CI
plt.figure(figsize=(10, 6))
plot_regression(
    "lnchprg",
    "math10",
    meap93,
    results,
    "Math Scores vs Lunch Program Participation\nwith 95% Confidence Interval",  # Title with CI mention
)
# -

# This code estimates the regression model, calculates the SER and standard errors of coefficients manually using the formulas, and then prints these manual calculations. It also prints the `statsmodels` regression summary table, which includes the standard errors calculated by `statsmodels` (which should match our manual calculations). Finally, it generates a regression plot with a 95% confidence interval.
#
# **Interpretation of Example 2.12:**
#
# By comparing the manually calculated SER and standard errors with those reported in the `statsmodels` summary table, you can verify that they are consistent. The standard errors provide a measure of the uncertainty associated with our coefficient estimates. Smaller standard errors mean our estimates are more precise. The regression plot with confidence intervals visually shows the range of plausible regression lines, given the uncertainty in our estimates.
#
# ## 2.7 Monte Carlo Simulations
#
# Monte Carlo simulations are powerful tools for understanding the statistical properties of estimators, like the OLS estimators. They involve repeatedly generating random samples from a known population model, estimating the parameters using OLS in each sample, and then examining the distribution of these estimates. This helps us to empirically verify properties like unbiasedness and understand the sampling variability of estimators.
#
# ### 2.7.1. One Sample
#
# Let's start by simulating a single sample from a population regression model. We will define a true population model, generate random data based on this model, estimate the model using OLS on this sample, and compare the estimated coefficients with the true population parameters.

# +
# Set random seed for reproducibility - Ensure consistent random number generation
np.random.seed(1234567)  # Set seed for random number generator

# Set parameters - Define true population parameters and sample size
n = 1000  # sample size
beta0 = 1  # true intercept
beta1 = 0.5  # true slope
sigma_u = 2  # standard deviation of error term

# Generate data - Simulate x, u, and y based on population model
x = stats.norm.rvs(4, 1, size=n)  # x ~ N(4, 1) - Generate x from normal distribution
u = stats.norm.rvs(
    0,
    sigma_u,
    size=n,
)  # u ~ N(0, 4) - Generate error term from normal distribution
y = (
    beta0 + beta1 * x + u
)  # population regression function - Generate y based on true model
df = pd.DataFrame({"y": y, "x": x})  # Create DataFrame

# Estimate model - Perform OLS regression on simulated data
reg = smf.ols(formula="y ~ x", data=df)  # Define OLS model
results = reg.fit()  # Fit model

# Print results - Compare true and estimated parameters
print("True vs. Estimated Parameters:")
print("=============================")
print(
    f"True β₀: {beta0:.4f} | Estimated β₀: {results.params.iloc[0]:.4f}",
)  # Compare true and estimated intercept, using iloc for position
print(
    f"True β₁: {beta1:.4f} | Estimated β₁: {results.params.iloc[1]:.4f}",
)  # Compare true and estimated slope, using iloc for position
print(f"\nR-squared: {results.rsquared:.4f}")  # Print R-squared

# Create visualization - Two subplots: Data with regression lines and Error distribution
plt.figure(figsize=(12, 5))

# Data and regression lines - Scatter plot of simulated data with true and estimated regression lines
plt.subplot(1, 2, 1)  # First subplot
plt.scatter(x, y, alpha=0.5, label="Sample data")  # Scatter plot of x vs y
x_range = np.linspace(x.min(), x.max(), 100)  # Generate x values for regression lines
plt.plot(
    x_range,
    beta0 + beta1 * x_range,
    "r-",
    label="True regression line",  # True population regression line (red solid)
    linewidth=2,
)
plt.plot(
    x_range,
    results.params[0] + results.params[1] * x_range,
    "g--",
    label="Estimated regression line",  # Estimated OLS regression line (green dashed)
)
plt.title("Sample Data and Regression Lines")  # Set title
plt.xlabel("x")  # Set x-axis label
plt.ylabel("y")  # Set y-axis label
plt.legend()  # Show legend

# Error distribution - Histogram of simulated error terms and true error distribution
plt.subplot(1, 2, 2)  # Second subplot
sns.histplot(
    u,
    bins=30,
    stat="density",
    alpha=0.5,
)  # Histogram of simulated error terms
x_range = np.linspace(u.min(), u.max(), 100)  # Generate x values for PDF
plt.plot(
    x_range,
    stats.norm.pdf(x_range, 0, sigma_u),
    "r-",
    label="True error distribution",  # True error distribution PDF (red solid)
)
plt.title("Distribution of Errors")  # Set title
plt.xlabel("Error term (u)")  # Set x-axis label
plt.ylabel("Density")  # Set y-axis label
plt.legend()  # Show legend

plt.tight_layout()  # Adjust layout
# -

# This code simulates a dataset from a simple linear regression model with known parameters. It then estimates the model using OLS and compares the estimated coefficients to the true parameters. It also visualizes the simulated data with both the true population regression line and the estimated regression line, as well as the distribution of the error terms compared to the true error distribution.
#
# **Interpretation of Example 2.13.1:**
#
# By running this code, you will see that the estimated coefficients $\hat{\beta}_0$ and $\hat{\beta}_1$ are close to, but not exactly equal to, the true values $\beta_0$ and $\beta_1$. This is because we are using a single random sample, and there is sampling variability. The regression plot shows the scatter of data points around the true population regression line, and the estimated regression line is an approximation based on this sample. The error distribution histogram should resemble the true normal distribution of errors.
#
# ### 2.7.2. Many Samples
#
# To better understand the sampling properties of OLS estimators, we need to repeat the simulation process many times. This will allow us to observe the distribution of the OLS estimates across different samples, which is known as the **sampling distribution**. We can then check if the estimators are unbiased by looking at the mean of these estimates and examine their variability.

# +
# Set parameters - Simulation parameters (number of replications increased)
np.random.seed(1234567)  # Set seed
n = 1000  # sample size
r = 10000  # number of replications (increased to 10000 for better results)
beta0 = 1  # true intercept
beta1 = 0.5  # true slope
sigma_u = 2  # standard deviation of error term

# Initialize arrays to store results - Arrays to store estimated beta_0 and beta_1 from each replication
b0 = np.empty(r)  # Array for intercept estimates
b1 = np.empty(r)  # Array for slope estimates

# Generate fixed x values - Keep x values constant across replications to isolate variability from error term
x = stats.norm.rvs(4, 1, size=n)  # Fixed x values from normal distribution

# Perform Monte Carlo simulation - Loop through replications, generate y, estimate model, store coefficients
for i in range(r):  # Loop for r replications
    # Generate new error terms and y values for each replication - Generate new u and y for each sample
    u = stats.norm.rvs(0, sigma_u, size=n)  # New error terms for each replication
    y = beta0 + beta1 * x + u  # Generate y based on fixed x and new u
    df = pd.DataFrame({"y": y, "x": x})  # Create DataFrame

    # Estimate model and store coefficients - Fit OLS and store estimated coefficients
    results = smf.ols(formula="y ~ x", data=df).fit()  # Fit OLS model
    b0[i] = results.params.iloc[0]  # Store estimated intercept, using iloc for position
    b1[i] = results.params.iloc[1]  # Store estimated slope, using iloc for position

# Calculate summary statistics - Calculate mean and standard deviation of estimated coefficients
print("Monte Carlo Results:")
print("===================")
print("\nIntercept (β₀):")
print(f"True value: {beta0:.4f}")  # True intercept
print(f"Mean estimate: {np.mean(b0):.4f}")  # Mean of estimated intercepts
print(
    f"Standard deviation: {np.std(b0, ddof=1):.4f}",
)  # Standard deviation of estimated intercepts

print("\nSlope (β₁):")
print(f"True value: {beta1:.4f}")  # True slope
print(f"Mean estimate: {np.mean(b1):.4f}")  # Mean of estimated slopes
print(
    f"Standard deviation: {np.std(b1, ddof=1):.4f}",
)  # Standard deviation of estimated slopes

# Create visualization - Two subplots: Sampling distribution of beta_0 and beta_1
plt.figure(figsize=(12, 5))

# Distribution of β₀ estimates - Histogram of estimated intercepts
plt.subplot(1, 2, 1)  # First subplot
sns.histplot(b0, bins=50, stat="density", alpha=0.5)  # Histogram of beta_0 estimates
plt.axvline(
    beta0,
    color="r",
    linestyle="--",
    label="True value",
)  # Vertical line at true beta_0
plt.axvline(
    np.mean(b0),
    color="g",
    linestyle="--",
    label="Mean estimate",
)  # Vertical line at mean of beta_0 estimates
plt.title("Sampling Distribution of β₀")  # Set title
plt.xlabel("β₀ estimates")  # x-axis label
plt.ylabel("Density")  # y-axis label
plt.legend()  # Show legend

# Distribution of β₁ estimates - Histogram of estimated slopes
plt.subplot(1, 2, 2)  # Second subplot
sns.histplot(b1, bins=50, stat="density", alpha=0.5)  # Histogram of beta_1 estimates
plt.axvline(
    beta1,
    color="r",
    linestyle="--",
    label="True value",
)  # Vertical line at true beta_1
plt.axvline(
    np.mean(b1),
    color="g",
    linestyle="--",
    label="Mean estimate",
)  # Vertical line at mean of beta_1 estimates
plt.title("Sampling Distribution of β₁")  # Set title
plt.xlabel("β₁ estimates")  # x-axis label
plt.ylabel("Density")  # y-axis label
plt.legend()  # Show legend

plt.tight_layout()  # Adjust layout
# -

# This code performs a Monte Carlo simulation with a large number of replications (e.g., 10000). In each replication, it generates new error terms and $y$ values while keeping the $x$ values fixed. It estimates the OLS regression and stores the estimated coefficients. After all replications, it calculates the mean and standard deviation of the estimated coefficients and plots histograms of their sampling distributions.
#
# **Interpretation of Example 2.13.2:**
#
# By running this code, you will observe the sampling distributions of $\hat{\beta}_0$ and $\hat{\beta}_1$. The histograms should be roughly bell-shaped (approximating normal distributions, due to the Central Limit Theorem). Importantly, you should see that the mean of the estimated coefficients (vertical green dashed line in the histograms) is very close to the true population parameters (vertical red dashed line). This empirically demonstrates the unbiasedness of the OLS estimators under the SLR assumptions. The standard deviations of the estimated coefficients provide a measure of their sampling variability.
#
# ### 2.7.3. Violation of SLR.4 (Zero Conditional Mean)
#
# Now, let's investigate what happens when one of the key assumptions is violated. Consider the violation of SLR.4, the zero conditional mean assumption, i.e., $\text{E}(u|x) \neq 0$. This means that the error term is correlated with $x$. In this case, we expect OLS estimators to be biased. Let's simulate a scenario where this assumption is violated and see the results.

# +
# Set parameters - Simulation parameters (same as before)
np.random.seed(1234567)  # Set seed
n = 1000
r = 10000
beta0 = 1
beta1 = 0.5
sigma_u = 2

# Initialize arrays - Arrays to store estimated coefficients
b0 = np.empty(r)
b1 = np.empty(r)

# Generate fixed x values - Fixed x values
x = stats.norm.rvs(4, 1, size=n)

# Perform Monte Carlo simulation with E(u|x) ≠ 0 - Simulation loop with violation of SLR.4
for i in range(r):  # Loop for replications
    # Generate errors with non-zero conditional mean - Error term depends on x, violating SLR.4
    u_mean = (x - 4) / 5  # E(u|x) = (x - 4)/5 - Conditional mean of error depends on x
    u = stats.norm.rvs(
        u_mean,
        sigma_u,
        size=n,
    )  # Error term with non-zero conditional mean
    y = beta0 + beta1 * x + u  # Generate y with biased error term
    df = pd.DataFrame({"y": y, "x": x})  # Create DataFrame

    # Estimate model - Fit OLS model
    results = smf.ols(formula="y ~ x", data=df).fit()  # Fit OLS model
    b0[i] = results.params.iloc[0]  # Store estimated intercept, using iloc for position
    b1[i] = results.params.iloc[1]  # Store estimated slope, using iloc for position

# Print results - Compare true values with mean estimates and bias
print("Monte Carlo Results with E(u|x) ≠ 0:")
print("===================================")
print("\nIntercept (β₀):")
print(f"True value: {beta0:.4f}")  # True intercept
print(f"Mean estimate: {np.mean(b0):.4f}")  # Mean of estimated intercepts
print(f"Bias: {np.mean(b0) - beta0:.4f}")  # Bias of intercept estimator

print("\nSlope (β₁):")
print(f"True value: {beta1:.4f}")  # True slope
print(f"Mean estimate: {np.mean(b1):.4f}")  # Mean of estimated slopes
print(f"Bias: {np.mean(b1) - beta1:.4f}")  # Bias of slope estimator

# Create visualization - Sampling distributions of beta_0 and beta_1 with violation of SLR.4
plt.figure(figsize=(12, 5))

# Distribution of β₀ estimates - Histogram of beta_0 estimates
plt.subplot(1, 2, 1)  # First subplot
sns.histplot(b0, bins=50, stat="density", alpha=0.5)  # Histogram
plt.axvline(beta0, color="r", linestyle="--", label="True value")  # True value line
plt.axvline(
    np.mean(b0),
    color="g",
    linestyle="--",
    label="Mean estimate",
)  # Mean estimate line
plt.title("Sampling Distribution of β₀\nwith E(u|x) ≠ 0")  # Title
plt.xlabel("β₀ estimates")  # x-axis label
plt.ylabel("Density")  # y-axis label
plt.legend()  # Legend

# Distribution of β₁ estimates - Histogram of beta_1 estimates
plt.subplot(1, 2, 2)  # Second subplot
sns.histplot(b1, bins=50, stat="density", alpha=0.5)  # Histogram
plt.axvline(beta1, color="r", linestyle="--", label="True value")  # True value line
plt.axvline(
    np.mean(b1),
    color="g",
    linestyle="--",
    label="Mean estimate",
)  # Mean estimate line
plt.title("Sampling Distribution of β₁\nwith E(u|x) ≠ 0")  # Title
plt.xlabel("β₁ estimates")  # x-axis label
plt.ylabel("Density")  # y-axis label
plt.legend()  # Legend

plt.tight_layout()  # Layout
# -

# In this code, we intentionally violate the zero conditional mean assumption by setting the mean of the error term to be dependent on $x$: $\text{E}(u|x) = (x - 4)/5$. We then perform the Monte Carlo simulation and examine the results.
#
# **Interpretation of Example 2.13.3:**
#
# By running this simulation, you will observe that the mean of the estimated coefficients $\hat{\beta}_0$ and $\hat{\beta}_1$ are no longer close to the true values $\beta_0$ and $\beta_1$. The bias, calculated as the difference between the mean estimate and the true value, will be noticeably different from zero. This empirically demonstrates that when the zero conditional mean assumption (SLR.4) is violated, the OLS estimators become biased. The histograms of the sampling distributions will be centered around the biased mean estimates, not the true values.
#
# ### 2.7.4. Violation of SLR.5 (Homoscedasticity)
#
# Finally, let's consider the violation of SLR.5, the homoscedasticity assumption, i.e., $\text{var}(u|x) \neq \sigma^2$. This means that the variance of the error term is not constant across values of $x$ (heteroscedasticity). While heteroscedasticity does not cause bias in OLS estimators, it affects their efficiency and the validity of standard errors and inference. Let's simulate a scenario with heteroscedasticity.

# +
# Set parameters - Simulation parameters (same as before)
np.random.seed(1234567)  # Set seed
n = 1000
r = 10000
beta0 = 1
beta1 = 0.5

# Initialize arrays - Arrays to store estimated coefficients
b0 = np.empty(r)
b1 = np.empty(r)

# Generate fixed x values - Fixed x values
x = stats.norm.rvs(4, 1, size=n)

# Perform Monte Carlo simulation with heteroscedasticity - Simulation loop with violation of SLR.5
for i in range(r):  # Loop for replications
    # Generate errors with variance depending on x - Error variance depends on x, violating SLR.5
    u_var = (
        4 / np.exp(4.5) * np.exp(x)
    )  # var(u|x) = 4e^(x-4.5) - Conditional variance depends on x
    u = stats.norm.rvs(
        0,
        np.sqrt(u_var),
        size=n,
    )  # Error term with heteroscedastic variance
    y = beta0 + beta1 * x + u  # Generate y with heteroscedastic errors
    df = pd.DataFrame({"y": y, "x": x})  # Create DataFrame

    # Estimate model - Fit OLS model
    results = smf.ols(formula="y ~ x", data=df).fit()  # Fit OLS model
    b0[i] = results.params.iloc[0]  # Store estimated intercept, using iloc for position
    b1[i] = results.params.iloc[1]  # Store estimated slope, using iloc for position

# Print results - Compare true values with mean estimates and standard deviations
print("Monte Carlo Results with Heteroscedasticity:")
print("==========================================")
print("\nIntercept (β₀):")
print(f"True value: {beta0:.4f}")  # True intercept
print(f"Mean estimate: {np.mean(b0):.4f}")  # Mean of estimated intercepts
print(
    f"Standard deviation: {np.std(b0, ddof=1):.4f}",
)  # Standard deviation of estimated intercepts

print("\nSlope (β₁):")
print(f"True value: {beta1:.4f}")  # True slope
print(f"Mean estimate: {np.mean(b1):.4f}")  # Mean of estimated slopes
print(
    f"Standard deviation: {np.std(b1, ddof=1):.4f}",
)  # Standard deviation of estimated slopes

# Create visualization - Scatter plot of data with heteroscedasticity and Error term vs x plot
plt.figure(figsize=(12, 5))

# Scatter plot of one sample with heteroscedasticity - Visualize data with heteroscedasticity
plt.subplot(1, 2, 1)  # First subplot
plt.scatter(x, y, alpha=0.5)  # Scatter plot of x vs y
plt.plot(
    x_range,
    beta0 + beta1 * x_range,
    "r--",
    label="True regression line",
)  # True regression line
plt.title("Sample Data with Heteroscedasticity")  # Title
plt.xlabel("x")  # x-axis label
plt.ylabel("y")  # y-axis label
plt.legend()  # Legend

# Distribution of errors conditional on x - Visualize heteroscedasticity directly
plt.subplot(1, 2, 2)  # Second subplot
plt.scatter(x, u, alpha=0.5)  # Scatter plot of x vs error term
plt.axhline(
    y=0,
    color="r",
    linestyle="--",
    label="E(u|x) = 0",
)  # Horizontal line at zero
plt.title("Error Terms vs. x")  # Title
plt.xlabel("x")  # x-axis label
plt.ylabel("Error term (u)")  # y-axis label
plt.legend()  # Legend

plt.tight_layout()  # Layout
# -

# In this code, we introduce heteroscedasticity by making the variance of the error term dependent on $x$: $\text{var}(u|x) = 4e^{(x-4.5)}$. We then perform the Monte Carlo simulation.
#
# **Interpretation of Example 2.13.4:**
#
# By running this simulation, you will observe that the mean of the estimated coefficients $\hat{\beta}_0$ and $\hat{\beta}_1$ are still close to the true values $\beta_0$ and $\beta_1$. This confirms that OLS estimators remain unbiased even under heteroscedasticity (as long as SLR.1-SLR.4 hold). However, you might notice that the standard deviations of the estimated coefficients (sampling variability) could be different compared to the homoscedastic case (Example 2.7.2), although unbiasedness is maintained. The scatter plot of data with heteroscedasticity will show that the spread of data points around the regression line is not constant across the range of $x$. The plot of error terms vs. $x$ directly visualizes the heteroscedasticity, as you'll see the spread of error terms changing with $x$.
#
# Through these Monte Carlo simulations, we have empirically explored the properties of OLS estimators and the consequences of violating some of the key assumptions of the simple linear regression model.
#
# This concludes our exploration of the Simple Regression Model. We have covered the basics of OLS estimation, interpretation of results, goodness of fit, nonlinear transformations, special regression cases, and the importance of underlying assumptions. We also used Monte Carlo simulations to understand the statistical properties of OLS estimators. This foundation is crucial for understanding more advanced econometric techniques and models in subsequent chapters.
