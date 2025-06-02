---
jupyter:
  jupytext:
    formats: notebooks//ipynb,markdown//md,scripts//py
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.17.2
  kernelspec:
    display_name: merino
    language: python
    name: python3
---

# 7. Multiple Regression Analysis with Qualitative Regressors

This notebook explores the use of qualitative regressors, also known as categorical variables, in multiple regression analysis. Qualitative regressors allow us to incorporate non-numeric factors like gender, marital status, occupation, or region into our regression models. We will use dummy variables (also called indicator variables) to represent these qualitative factors numerically, enabling us to analyze their impact on a dependent variable alongside quantitative regressors.

We will use the `statsmodels` library in Python to perform Ordinary Least Squares (OLS) regressions and the `wooldridge` library to access datasets from the textbook "Introductory Econometrics" by Jeffrey M. Wooldridge.

```python
import numpy as np  # noqa: F401
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import wooldridge as wool
```

## 7.1 Linear Regression with Dummy Variables as Regressors

Dummy variables are binary variables that take on the value 1 or 0 to indicate the presence or absence of a specific attribute or category. In regression analysis, they are used to include qualitative information in a quantitative framework.

### Example 7.1: Hourly Wage Equation

In this example, we will investigate how gender affects hourly wages, controlling for education, experience, and tenure. We will use the `wage1` dataset from the `wooldridge` package. The dataset includes information on wages, education, experience, tenure, and gender (female=1 if female, 0 if male).

```python
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
```

**Explanation:**

- `wage ~ female + educ + exper + tenure`: This formula specifies the regression model. We are regressing `wage` (hourly wage) on `female` (dummy variable for gender), `educ` (years of education), `exper` (years of experience), and `tenure` (years with current employer).
- `smf.ols(...)`: This function from `statsmodels.formula.api` performs Ordinary Least Squares (OLS) regression.
- `results.fit()`: This fits the regression model to the data.
- The printed table displays the regression coefficients (`b`), standard errors (`se`), t-statistics (`t`), and p-values (`pval`) for each regressor.

**Interpretation of Results:**

- **Intercept (const):** The estimated intercept is approximately 0.3223. This is the predicted wage for a male (female=0) with zero years of education, experience, and tenure. While not practically meaningful in this context (as education, experience, and tenure are rarely zero), it's a necessary part of the regression model.
- **female:** The coefficient for `female` is -1.8097. This suggests that, holding education, experience, and tenure constant, women earn, on average, \$1.81 less per hour than men in this dataset. The negative coefficient indicates a wage gap between women and men.
- **educ:** The coefficient for `educ` is 0.5640. This means that, holding other factors constant, each additional year of education is associated with an increase in hourly wage of approximately \$0.56.
- **exper:** The coefficient for `exper` is 0.0185. For each additional year of experience, hourly wage is predicted to increase by about \$0.0185, holding other factors constant.
- **tenure:** The coefficient for `tenure` is 0.0643. For each additional year of tenure with the current employer, hourly wage is predicted to increase by about \$0.0643, holding other factors constant.

- **P-values:** The p-values for `female`, `educ`, and `tenure` are very small (close to zero), indicating that these variables are statistically significant at conventional significance levels (e.g., 0.05 or 0.01). The p-value for `exper` is larger (0.2973), suggesting that experience is not statistically significant in this model at the 5% or 1% level, when other factors are controlled.

This simple example demonstrates how dummy variables can be used to quantify the effect of qualitative factors like gender on a dependent variable like wages in a multiple regression framework.

### Example 7.6: Log Hourly Wage Equation

Using the logarithm of the dependent variable, such as wage, is common in economics. When the dependent variable is in log form, the coefficients on the regressors (after multiplying by 100) can be interpreted as approximate percentage changes in the dependent variable for a one-unit change in the regressor.

In this example, we will use the natural logarithm of hourly wage as the dependent variable and explore the interaction effect between marital status and gender, along with education, experience, and tenure (including quadratic terms for experience and tenure to capture potential non-linear relationships).

```python
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
```

**Explanation:**

- `np.log(wage) ~ married*female + educ + exper + I(exper**2) + tenure + I(tenure**2)`:
  - `np.log(wage)`: The dependent variable is now the natural logarithm of `wage`.
  - `married*female`: This includes the main effects of `married` and `female` as well as their interaction term (`married:female`). The interaction term allows us to see if the effect of being female differs depending on marital status (or vice versa).
  - `I(exper**2)` and `I(tenure**2)`: We use `I(...)` to include squared terms of `exper` and `tenure` in the formula. This allows for a quadratic relationship between experience/tenure and log wage, capturing potential diminishing returns to experience and tenure.

**Interpretation of Results:**

- **Intercept (const):** The intercept is approximately 0.4254. This is the predicted log wage for a single (married=0), male (female=0) with zero education, experience, and tenure.
- **married:** The coefficient for `married` is 0.1885. This suggests that married men (the reference group for the interaction) earn approximately 18.85% more than single men, holding other factors constant. (For small coefficients, the percentage change is approximately 100 times the coefficient).
- **female:** The coefficient for `female` is -0.3531. This is the wage difference between single women (female=1, married=0) and single men (female=0, married=0). Single women earn approximately 35.31% less than single men, holding other factors constant.
- **married:female:** The interaction term `married:female` has a coefficient of 0.1450. This is the _additional_ effect of being female when married, compared to being female and single. To find the wage difference between married women and married men, we need to combine the coefficients for `female` and `married:female`: -0.3531 + 0.1450 = -0.2081. So, married women earn approximately 20.81% less than married men, holding other factors constant.
- **educ, exper, I(exper**2), tenure, I(tenure**2):** These coefficients represent the effects of education, experience, and tenure on log wage, controlling for gender and marital status. For example, the coefficient for `educ` (0.0789) suggests that each additional year of education is associated with approximately a 7.89% increase in hourly wage, holding other factors constant. The negative coefficient on `I(exper**2)` suggests diminishing returns to experience.

- **P-values:** Most variables, including the interaction term `married:female`, are statistically significant at conventional levels, indicating that marital status and gender, both individually and in interaction, play a significant role in explaining wage variations.

## 7.2 Boolean variables

In Python (and programming in general), boolean variables are often used to represent true/false or yes/no conditions. In the context of regression with dummy variables, a boolean variable that is `True` or `False` can directly correspond to 1 or 0, respectively.

In the `wage1` dataset, the `female` variable is already coded as 1 for female and 0 for male. We can explicitly create a boolean variable, although it is not strictly necessary in this case as `statsmodels` and regression formulas in general can interpret 1/0 variables as dummy variables.

```python
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
```

**Explanation:**

- `wage1["isfemale"] = wage1["female"] == 1`: This line creates a new column named `isfemale` in the `wage1` DataFrame. It assigns `True` to `isfemale` if the value in the `female` column is 1 (meaning female), and `False` otherwise. In essence, `isfemale` is a boolean representation of the `female` dummy variable.
- `reg = smf.ols(formula="wage ~ isfemale + educ + exper + tenure", data=wage1)`: We then run the same regression as in Example 7.1, but now using `isfemale` instead of `female` in the formula.

**Interpretation of Results:**

The regression table will be virtually identical to the one in Example 7.1. The coefficient, standard error, t-statistic, and p-value for `isfemale` will be the same as for `female` in the previous example.

**Key takeaway:**

Using a boolean variable (`isfemale`) that evaluates to `True` or `False` is functionally equivalent to using a dummy variable coded as 1 or 0 in regression analysis with `statsmodels`. `statsmodels` internally handles boolean `True` as 1 and `False` as 0 in regression formulas when they are used as regressors. Therefore, creating an explicit boolean variable like `isfemale` here doesn't change the regression outcome compared to directly using the 1/0 coded `female` variable.

## 7.3 Categorical Variables

Categorical variables represent groups or categories, and they can have more than two categories. Examples include occupation, industry, region, or education levels (e.g., high school, bachelor's, master's, PhD). To include categorical variables in a regression model, we typically use dummy variable encoding. If a categorical variable has _k_ categories, we include _k-1_ dummy variables in the regression to avoid perfect multicollinearity (the dummy variable trap). One category is chosen as the "reference" or "baseline" category, and the coefficients on the dummy variables represent the difference in the dependent variable between each category and the reference category, holding other regressors constant.

In this section, we will use the `CPS1985.csv` dataset, which contains data from the Current Population Survey in 1985. We will investigate how gender and occupation affect wages, controlling for education and experience.

```python
CPS1985 = pd.read_csv("../data/CPS1985.csv")
# rename variable to make outputs more compact:
CPS1985["oc"] = CPS1985["occupation"]

# table of categories and frequencies for two categorical variables:
freq_gender = pd.crosstab(CPS1985["gender"], columns="count")
print(f"freq_gender: \n{freq_gender}\n")

freq_occupation = pd.crosstab(CPS1985["oc"], columns="count")
print(f"freq_occupation: \n{freq_occupation}\n")
```

**Explanation:**

- `CPS1985 = pd.read_csv("../data/CPS1985.csv")`: Loads the dataset from a CSV file. You might need to adjust the path `../data/CPS1985.csv` depending on where you have saved the data file relative to your notebook.
- `CPS1985["oc"] = CPS1985["occupation"]`: Creates a shorter alias `oc` for the `occupation` variable for more compact output in regression tables.
- `pd.crosstab(...)`: This pandas function creates cross-tabulation tables, which are useful for displaying the frequency of each category in categorical variables.
  - `freq_gender = pd.crosstab(CPS1985["gender"], columns="count")`: Counts the occurrences of each gender category.
  - `freq_occupation = pd.crosstab(CPS1985["oc"], columns="count")`: Counts the occurrences of each occupation category.

**Output:**

The `freq_gender` output shows the counts for 'female' and 'male' in the dataset. The `freq_occupation` output shows the counts for each occupation category present in the dataset (e.g., 'admin', 'clerical', 'management', etc.). This gives us an overview of the distribution of these categorical variables.

```python
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
```

**Explanation:**

- `formula="np.log(wage) ~ education +experience + C(gender) + C(oc)"`:
  - `C(gender)` and `C(oc)`: The `C()` function in `statsmodels` formula language tells the regression function to treat `gender` and `oc` (occupation) as categorical variables. `statsmodels` will automatically create dummy variables for each category except for the reference category. By default, `statsmodels` will choose the first category in alphabetical order as the reference category. For `gender`, 'female' will likely be the reference (if categories are 'female', 'male'), and for `oc`, it will be the first occupation in alphabetical order.

**Interpretation of Results:**

- **Reference Categories:** Let's assume 'female' is the reference category for gender and 'admin' is the reference category for occupation (check the alphabetical order in your dataset to confirm).
- **C(gender)[T.male]:** The coefficient for `C(gender)[T.male]` represents the log wage difference between males and females, holding education, experience, and occupation constant. If the coefficient is positive and significant, it means males in the reference occupation category earn significantly more than females in the reference occupation category, controlling for education and experience.
- **C(oc)[T.category_name]:** For each occupation category (except the reference category 'admin'), there will be a coefficient like `C(oc)[T.clerical]`, `C(oc)[T.management]`, etc. `C(oc)[T.clerical]` represents the log wage difference between individuals in 'clerical' occupations and individuals in the reference occupation ('admin'), holding gender, education, and experience constant.

- **education and experience:** The coefficients for `education` and `experience` are interpreted as before, representing the effect of each additional year of education or experience on log wage, holding gender, occupation, and the other variable constant.

- **P-values:** P-values help assess the statistical significance of each coefficient. Small p-values indicate that the corresponding variable is a statistically significant predictor of log wage.

```python
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
```

**Explanation:**

- `C(gender, Treatment("male"))`: Here, we explicitly specify "male" as the reference category for the `gender` variable using `Treatment("male")` within the `C()` function. Now, the coefficient for `C(gender)[T.female]` will represent the log wage difference between females and males (with males as the baseline).
- `C(oc, Treatment("technical"))`: Similarly, we set "technical" as the reference category for the `oc` (occupation) variable. Coefficients for other occupation categories will now be interpreted relative to the 'technical' occupation group.

**Interpretation of Results:**

- **Reference Categories (Changed):** Now, 'male' is the reference category for gender, and 'technical' is the reference category for occupation.
- **C(gender)[T.female]:** The coefficient for `C(gender)[T.female]` now represents the log wage difference between females and males, with males being the reference group.
- **C(oc)[T.category_name]:** For each occupation category (except 'technical'), the coefficient `C(oc)[T.category_name]` represents the log wage difference between that occupation and the 'technical' occupation, holding gender, education, and experience constant. For example, `C(oc)[T.admin]` will now represent the log wage difference between 'admin' and 'technical' occupations.

**Key takeaway:**

By using `C()` and `Treatment()`, we can easily incorporate categorical variables into our regression models and control which category serves as the reference group. The interpretation of the coefficients for categorical variables is always in comparison to the chosen reference category.

### 7.3.1 ANOVA Tables

ANOVA (Analysis of Variance) tables in regression context help to assess the overall significance of groups of regressors. They decompose the total variance in the dependent variable into components attributable to different sets of regressors. In the context of categorical variables, ANOVA tables can be used to test the joint significance of all dummy variables representing a categorical variable.

```python
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
```

**Explanation:**

- `formula="np.log(wage) ~ education + experience + gender + occupation"`: In this formula, we are directly using `gender` and `occupation` as categorical variables without explicitly using `C()`. `statsmodels` can often automatically detect categorical variables (especially string or object type columns) and treat them as such in the regression formula, creating dummy variables behind the scenes. However, it is generally better to be explicit and use `C()` for clarity and control, especially when you want to specify reference categories.

```python
# ANOVA table:
table_anova = sm.stats.anova_lm(results, typ=2)
print(f"table_anova: \n{table_anova}\n")
```

**Explanation:**

- `table_anova = sm.stats.anova_lm(results, typ=2)`: This function from `statsmodels.stats` calculates the ANOVA table for the fitted regression model (`results`). `typ=2` specifies Type II ANOVA, which is generally recommended for regression models, especially when there might be some imbalance in the design (which is common in observational data).

**Interpretation of ANOVA Table:**

The ANOVA table output will typically include columns like:

- **df (degrees of freedom):** For each regressor (or group of regressors).
- **sum_sq (sum of squares):** The sum of squares explained by each regressor (or group of regressors).
- **mean_sq (mean square):** Mean square = sum_sq / df.
- **F (F-statistic):** F-statistic for testing the null hypothesis that all coefficients associated with a particular regressor (or group of regressors) are jointly zero.
- **PR(>F) (p-value):** The p-value associated with the F-statistic.

**For example, in the ANOVA table output:**

- **gender:** The row for `gender` will give you an F-statistic and a p-value. This tests the null hypothesis that _gender has no effect on wage, once education, experience, and occupation are controlled for_. A small p-value (e.g., < 0.05) would suggest that gender is a statistically significant factor in explaining wage variation, even after controlling for other variables.
- **occupation:** Similarly, the row for `occupation` will test the joint significance of all occupation dummy variables. It tests the null hypothesis that _occupation has no effect on wage, once education, experience, and gender are controlled for_. A small p-value would indicate that occupation is a significant factor.
- **education and experience:** ANOVA table also provides F-tests for `education` and `experience`, testing their significance in the model.
- **Residual:** This row represents the unexplained variance (residuals) in the model.

**Key takeaway:**

ANOVA tables provide F-tests for the joint significance of regressors, especially useful for categorical variables with multiple dummy variables. They help assess the overall contribution of each set of regressors to explaining the variation in the dependent variable.

## 7.4 Breaking a Numeric Variable Into Categories

Sometimes, it might be useful to convert a continuous numeric variable into a categorical variable. This can be done for several reasons:

- **Non-linear effects:** If you suspect that the effect of a numeric variable is not linear, categorizing it can capture non-linearities without explicitly modeling a complex functional form.
- **Easier interpretation:** Categorical variables can sometimes be easier to interpret, especially when dealing with ranges or levels of a variable.
- **Dealing with outliers or extreme values:** Grouping extreme values into categories can reduce the influence of outliers.

### Example 7.8: Effects of Law School Rankings on Starting Salaries

In this example, we will examine the effect of law school rankings on starting salaries of graduates. The `lawsch85` dataset from `wooldridge` contains information on law school rankings (`rank`), starting salaries (`salary`), LSAT scores (`LSAT`), GPA, library volumes (`libvol`), and cost (`cost`).

```python
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
```

**Explanation:**

- `lawsch85 = wool.data("lawsch85")`: Loads the `lawsch85` dataset.
- `cutpts = [0, 10, 25, 40, 60, 100, 175]`: Defines the cut points for creating categories based on law school rank. These cut points divide the rank variable into different rank ranges.
- `lawsch85["rc"] = pd.cut(...)`: The `pd.cut()` function is used to bin the `rank` variable into categories.
  - `lawsch85["rank"]`: The variable to be categorized.
  - `bins=cutpts`: The cut points that define the boundaries of the categories.
  - `labels=[...]`: Labels for each category. For example, ranks between 0 and 10 (exclusive of 0, inclusive of 10) will be labeled as "(0,10]".
- `freq = pd.crosstab(lawsch85["rc"], columns="count")`: Creates a frequency table to show the number of law schools in each rank category.

**Output:**

The `freq` output shows the number of law schools falling into each defined rank category (e.g., how many schools are ranked between 0 and 10, between 10 and 25, etc.).

```python
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
```

**Explanation:**

- `formula='np.log(salary) ~ C(rc, Treatment("(100,175]")) + LSAT + GPA + np.log(libvol) + np.log(cost)'`:
  - `C(rc, Treatment("(100,175]"))`: We use the categorized rank variable `rc` in the regression as a categorical variable using `C()`. We also specify `Treatment("(100,175]")` to set the category "(100,175]" (law schools ranked 100-175, the lowest rank category) as the reference category.
  - `LSAT + GPA + np.log(libvol) + np.log(cost)`: These are other control variables included in the regression: LSAT score, GPA, log of library volumes, and log of cost.

**Interpretation of Results:**

- **Reference Category:** The reference category is law schools ranked "(100,175]".
- **C(rc)[T.(0,10)] to C(rc)[T.(60,100)]:** The coefficients for `C(rc)[T.(0,10)]`, `C(rc)[T.(10,25)]`, `C(rc)[T.(25,40)]`, `C(rc)[T.(40,60)]`, and `C(rc)[T.(60,100)]` represent the log salary difference between law schools in each of these rank categories and law schools in the reference category "(100,175]", holding LSAT, GPA, library volumes, and cost constant. For example, `C(rc)[T.(0,10)]` coefficient would represent the approximate percentage salary premium for graduates of law schools ranked (0, 10] compared to those from schools ranked (100, 175]. We expect these coefficients to be positive and decreasing as the rank category range increases (i.e., better ranked schools should have higher starting salaries).
- **LSAT, GPA, np.log(libvol), np.log(cost):** These coefficients represent the effects of LSAT score, GPA, log of library volumes, and log of cost on log salary, controlling for law school rank category.

```python
# ANOVA table:
table_anova = sm.stats.anova_lm(results, typ=2)
print(f"table_anova: \n{table_anova}\n")
```

**Explanation & Interpretation:**

As before, `sm.stats.anova_lm(results, typ=2)` calculates the ANOVA table for this regression model. The row for `C(rc)` in the ANOVA table will provide an F-test and p-value for the joint significance of all the dummy variables representing the rank categories. This tests the null hypothesis that _law school rank (categorized as `rc`) has no effect on starting salary, once LSAT, GPA, library volumes, and cost are controlled for_. A small p-value would indicate that law school rank category is a statistically significant factor in predicting starting salaries.

## 7.5 Interactions and Differences in Regression Functions Across Groups

We can allow for regression functions to differ across groups by including interaction terms between group indicators (dummy variables for qualitative variables) and quantitative regressors. This allows the slope coefficients of the quantitative regressors to vary across different groups defined by the qualitative variable.

```python
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
```

**Explanation:**

- `gpa3 = wool.data("gpa3")`: Loads the `gpa3` dataset, which contains data on college GPA, SAT scores, high school percentile (`hsperc`), total hours studied (`tothrs`), and gender (`female`), among other variables.
- `subset=(gpa3["spring"] == 1)`: We are using only the data for the spring semester (`spring == 1`).
- `formula="cumgpa ~ female * (sat + hsperc + tothrs)"`:
  - `female * (sat + hsperc + tothrs)`: This specifies a full interaction model between the dummy variable `female` and the quantitative regressors `sat`, `hsperc`, and `tothrs`. It is equivalent to including: `female + sat + hsperc + tothrs + female:sat + female:hsperc + female:tothrs`.
  - This model allows both the intercept and the slopes of `sat`, `hsperc`, and `tothrs` to be different for females and males.

**Interpretation of Results:**

- **Intercept (const):** This is the intercept for the reference group, which is males (female=0). It represents the predicted cumulative GPA for a male with SAT=0, hsperc=0, and tothrs=0.
- **female:** This is the _difference in intercepts_ between females and males, for SAT=0, hsperc=0, and tothrs=0.
- **sat, hsperc, tothrs:** These are the slopes of SAT, hsperc, and tothrs, respectively, for the reference group (males). They represent the change in cumulative GPA for a one-unit increase in each of these variables for males.
- **female:sat, female:hsperc, female:tothrs:** These are the _interaction terms_.
  - `female:sat`: Represents the _difference in the slope of SAT_ between females and males. So, the slope of SAT for females is (coefficient of `sat` + coefficient of `female:sat`).
  - Similarly, `female:hsperc` and `female:tothrs` represent the differences in slopes for `hsperc` and `tothrs` between females and males.

To get the regression equation for each group:

- **For Males (female=0):**
  `cumgpa = b_const + b_sat * sat + b_hsperc * hsperc + b_tothrs * tothrs`
  (where `b_const`, `b_sat`, `b_hsperc`, `b_tothrs` are the coefficients for `const`, `sat`, `hsperc`, `tothrs` respectively).
- **For Females (female=1):**
  `cumgpa = (b_const + b_female) + (b_sat + b_female:sat) * sat + (b_hsperc + b_female:hsperc) * hsperc + (b_tothrs + b_female:tothrs) * tothrs`
  (where `b_female`, `b_female:sat`, `b_female:hsperc`, `b_female:tothrs` are the coefficients for `female`, `female:sat`, `female:hsperc`, `female:tothrs` respectively).

```python
# F-Test for H0 (the interaction coefficients of 'female' are zero):
hypotheses = ["female:sat = 0", "female:hsperc = 0", "female:tothrs = 0"]
ftest = results.f_test(hypotheses)
fstat = ftest.statistic
fpval = ftest.pvalue

print(f"fstat: {fstat}\n")
print(f"fpval: {fpval}\n")
```

**Explanation:**

- `hypotheses = ["female:sat = 0", "female:hsperc = 0", "female:tothrs = 0"]`: We define the null hypotheses we want to test. Here, we are testing if the coefficients of all interaction terms involving `female` are jointly zero. If these coefficients are zero, it means there is no difference in the slopes of `sat`, `hsperc`, and `tothrs` between females and males.
- `ftest = results.f_test(hypotheses)`: Performs an F-test to test these joint hypotheses.
- `fstat = ftest.statistic` and `fpval = ftest.pvalue`: Extracts the F-statistic and p-value from the test results.

**Interpretation:**

- **p-value (fpval):** If the p-value is small (e.g., < 0.05), we reject the null hypothesis. This would mean that at least one of the interaction coefficients is significantly different from zero, indicating that the effect of at least one of the quantitative regressors (`sat`, `hsperc`, `tothrs`) on `cumgpa` differs between females and males. If the p-value is large, we fail to reject the null hypothesis, suggesting that there is not enough evidence to conclude that the slopes are different across genders.

```python
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
```

**Explanation:**

- `subset=(gpa3["spring"] == 1) & (gpa3["female"] == 0)`: We are now subsetting the data to include only spring semester data (`spring == 1`) and only male students (`female == 0`).
- `formula="cumgpa ~ sat + hsperc + tothrs"`: We run a regression of `cumgpa` on `sat`, `hsperc`, and `tothrs` for this male-only subsample.

**Interpretation:**

The regression table `table_m` shows the estimated regression equation specifically for male students. The coefficients are the intercept and slopes of `sat`, `hsperc`, and `tothrs` for males only. These coefficients should be very similar to the coefficients for `const`, `sat`, `hsperc`, and `tothrs` in the full interaction model (from the first regression in this section), as those were also interpreted as the coefficients for the male group (reference group).

```python
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
```

**Explanation:**

- `subset=(gpa3["spring"] == 1) & (gpa3["female"] == 1)`: This time, we subset the data for spring semester data and only female students (`female == 1`).
- `formula="cumgpa ~ sat + hsperc + tothrs"`: We run the same regression model for this female-only subsample.

**Interpretation:**

The regression table `table_f` shows the estimated regression equation specifically for female students. The coefficients are the intercept and slopes of `sat`, `hsperc`, and `tothrs` for females only. These coefficients should correspond to the combined coefficients from the full interaction model. For example:

- Intercept for females (in `table_f`) should be approximately equal to (coefficient of `const` + coefficient of `female`) from the interaction model (`table`).
- Slope of `sat` for females (in `table_f`) should be approximately equal to (coefficient of `sat` + coefficient of `female:sat`) from the interaction model (`table`).
- And so on for `hsperc` and `tothrs`.

**Key takeaway:**

- Interaction terms allow regression functions to differ across groups. Full interaction models allow both intercepts and slopes to vary.
- Testing the joint significance of interaction terms helps determine if there are statistically significant differences in the slopes across groups.
- Estimating separate regressions for each group is an alternative way to explore and compare regression functions across groups, and the results should be consistent with those from a full interaction model.
