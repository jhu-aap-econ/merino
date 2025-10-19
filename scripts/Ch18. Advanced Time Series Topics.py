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
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 18. Advanced Time Series Topics
#
# :::{important} Learning Objectives
# :class: dropdown
# By the end of this chapter, you should be able to:
#
# **18.1** Estimate and interpret infinite distributed lag models (geometric and rational distributed lags).
#
# **18.2** Test for unit roots using the Augmented Dickey-Fuller (ADF) test.
#
# **18.3** Understand spurious regression and why non-stationary series can produce misleading results.
#
# **18.4** Estimate cointegration relationships and error correction models for long-run equilibria.
#
# **18.5** Forecast time series using various models and evaluate forecast accuracy.
#
# **18.6** Apply event study methods with control groups for policy evaluation.
# :::
#
# Welcome to Chapter 18, where we study **advanced time series methods** for handling:
#
# - **Long-run relationships**: Infinite distributed lags with geometric decay
# - **Non-stationarity**: Unit roots, random walks, and stochastic trends
# - **Spurious regression**: Detecting false relationships in trending data
# - **Cointegration**: Long-run equilibrium relationships between non-stationary variables
# - **Forecasting**: Prediction and forecast evaluation
# - **Event studies**: Difference-in-differences with time series data
#
# These topics are crucial for macroeconomic analysis, financial econometrics, and policy evaluation with time series data!
#
# **Key Challenge**: Many economic time series are **non-stationary** (trending over time). Standard regression assumptions break down, leading to:
# - Spurious correlations
# - Invalid hypothesis tests
# - Misleading inference
#
# **Solution**: Use specialized methods that account for non-stationarity (unit roots, cointegration, error correction).

# %%
# # %pip install matplotlib numpy pandas statsmodels wooldridge scipy -q

# %%
# Import required libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import wooldridge as wool
from IPython.display import display
from scipy import stats
from statsmodels.tsa.stattools import adfuller, coint

# Set plotting style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams["figure.figsize"] = [10, 6]
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["axes.labelsize"] = 12

# %% [markdown]
# ## 18.1 Infinite Distributed Lag Models
#
# An **infinite distributed lag (IDL)** model allows current $y$ to depend on **all past values** of $x$:
#
# $$ y_t = \alpha + \beta_0 x_t + \beta_1 x_{t-1} + \beta_2 x_{t-2} + \cdots + u_t $$
#
# **Problem**: Infinite parameters to estimate!
#
# **Solution**: Impose **structure** on how $\beta_j$ decays as $j$ increases.
#
# ### Geometric Distributed Lag (Koyck Transformation)
#
# Assume coefficients decay **geometrically**:
#
# $$ \beta_j = \beta_0 \rho^j, \quad 0 < \rho < 1 $$
#
# This gives:
#
# $$ y_t = \alpha + \beta_0(x_t + \rho x_{t-1} + \rho^2 x_{t-2} + \cdots) + u_t $$
#
# **Key insight**: Lag $y_t$ by one period and multiply by $\rho$:
#
# $$ \rho y_{t-1} = \rho \alpha + \beta_0 \rho(x_{t-1} + \rho x_{t-2} + \cdots) + \rho u_{t-1} $$
#
# Subtract from original equation:
#
# $$ y_t - \rho y_{t-1} = \alpha(1-\rho) + \beta_0 x_t + (u_t - \rho u_{t-1}) $$
#
# Rearrange to get the **Koyck model**:
#
# $$ y_t = \gamma_0 + \gamma_1 y_{t-1} + \beta_0 x_t + v_t $$
#
# where:
# - $\gamma_0 = \alpha(1-\rho)$
# - $\gamma_1 = \rho$
# - $v_t = u_t - \rho u_{t-1}$ (MA(1) error)
#
# **Advantages**:
# - Estimable with OLS (only 3 parameters!)
# - Captures long-run effects through lagged dependent variable
#
# **Disadvantages**:
# - $v_t$ is serially correlated (MA(1))
# - Restrictive assumption (geometric decay)
#
# ### Long-Run Propensity (LRP)
#
# The **long-run propensity** measures the total cumulative effect of a permanent change in $x$:
#
# $$ LRP = \beta_0 + \beta_1 + \beta_2 + \cdots = \frac{\beta_0}{1 - \rho} = \frac{\beta_0}{1 - \gamma_1} $$
#
# **Interpretation**: If $x$ increases by 1 unit permanently, $y$ increases by $LRP$ units in the long run.
#
# ### Rational Distributed Lag
#
# A more flexible alternative allows for **non-monotonic** lag patterns:
#
# $$ y_t = \alpha + \beta_0 x_t + \beta_1 x_{t-1} + \beta_2 x_{t-2} + \cdots + u_t $$
#
# With $\beta_j = (\beta_0 + \beta_1 \rho^{j-1}) \rho^j$ for $j \geq 1$.
#
# After Koyck-type transformation:
#
# $$ y_t = \gamma_0 + \gamma_1 y_{t-1} + \beta_0 x_t + \delta_1 x_{t-1} + v_t $$
#
# This allows the impact to initially increase or decrease before decaying geometrically.
#
# ### Example 18.1: Housing Investment and Prices

# %%
# Load housing investment data
hseinv = wool.data("hseinv")

print("18.1 INFINITE DISTRIBUTED LAG MODELS")
print("=" * 70)
print("\nEXAMPLE: Housing investment and real housing prices")

print(f"\nTotal observations: {len(hseinv)}")
print("Time period: 1947-1988 (annual data)")

print("\nVARIABLES:")
print("  invpc  = real per capita housing investment")
print("  price  = real housing price index")
print("  linvpc = log(invpc)")
print("  gprice = growth rate of real housing prices")

# Create detrended and lagged variables
hseinv["linvpc_det"] = sm.tsa.tsatools.detrend(hseinv["linvpc"])
hseinv["gprice_lag1"] = hseinv["gprice"].shift(1)
hseinv["linvpc_det_lag1"] = hseinv["linvpc_det"].shift(1)

print("\nQUESTION: How does housing price growth affect investment?")
print("  - Immediate effect (β₀)")
print("  - Long-run effect (LRP)")

# Summary statistics
key_vars = ["invpc", "price", "linvpc", "gprice"]
display(hseinv[key_vars].describe().round(4))

# %%
# Visualize the data
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Panel A: Housing investment (log)
axes[0, 0].plot(hseinv["year"], hseinv["linvpc"], "b-", linewidth=2)
axes[0, 0].set_xlabel("Year")
axes[0, 0].set_ylabel("Log Housing Investment per Capita")
axes[0, 0].set_title("Housing Investment (Log Scale)")
axes[0, 0].grid(True, alpha=0.3)

# Panel B: Housing prices
axes[0, 1].plot(hseinv["year"], hseinv["price"], "g-", linewidth=2)
axes[0, 1].set_xlabel("Year")
axes[0, 1].set_ylabel("Real Housing Price Index")
axes[0, 1].set_title("Real Housing Prices")
axes[0, 1].grid(True, alpha=0.3)

# Panel C: Detrended log investment
axes[1, 0].plot(hseinv["year"], hseinv["linvpc_det"], "r-", linewidth=2)
axes[1, 0].axhline(y=0, color="black", linestyle="--", linewidth=1)
axes[1, 0].set_xlabel("Year")
axes[1, 0].set_ylabel("Detrended Log Investment")
axes[1, 0].set_title("Detrended Housing Investment")
axes[1, 0].grid(True, alpha=0.3)

# Panel D: Price growth
axes[1, 1].plot(hseinv["year"], hseinv["gprice"], "orange", linewidth=2)
axes[1, 1].axhline(y=0, color="black", linestyle="--", linewidth=1)
axes[1, 1].set_xlabel("Year")
axes[1, 1].set_ylabel("Price Growth Rate (%)")
axes[1, 1].set_title("Housing Price Growth")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nOBSERVATIONS:")
print("✓ Both investment and prices trend upward over time")
print("✓ Detrending removes long-run growth pattern")
print("✓ Price growth is volatile with cyclical patterns")

# %%
# KOYCK GEOMETRIC DISTRIBUTED LAG
print("\nKOYCK MODEL (Geometric Distributed Lag)")
print("=" * 70)
print("Model: y_t = γ₀ + γ₁·y_{t-1} + β₀·x_t + v_t")
print("where y = detrended log investment, x = price growth")

koyck = smf.ols(
    formula="linvpc_det ~ linvpc_det_lag1 + gprice",
    data=hseinv,
).fit()

table_koyck = pd.DataFrame(
    {
        "Coefficient": koyck.params,
        "Std. Error": koyck.bse,
        "t-statistic": koyck.tvalues,
        "p-value": koyck.pvalues,
    },
)

display(table_koyck.round(4))

print(f"\nR-squared: {koyck.rsquared:.4f}")
print(f"Observations: {koyck.nobs:.0f}")

print("\nINTERPRETATION:")
beta_0 = koyck.params["gprice"]
rho = koyck.params["linvpc_det_lag1"]

print(f"  β₀ (immediate effect): {beta_0:.4f}")
print(
    f"    → 1 percentage point increase in price growth → {100 * beta_0:.2f}% immediate increase in investment"
)

print(f"\n  ρ (persistence): {rho:.4f}")
print(f"    → Each period, {100 * rho:.1f}% of previous period's effect carries over")

# Long-Run Propensity
lrp_koyck = beta_0 / (1 - rho)
print(f"\n  LRP (long-run effect): {lrp_koyck:.4f}")
print(
    f"    → 1 percentage point permanent increase in price growth → {100 * lrp_koyck:.2f}% long-run increase in investment"
)
print(f"    → LRP = β₀/(1-ρ) = {beta_0:.4f}/(1-{rho:.4f}) = {lrp_koyck:.4f}")

# %%
# RATIONAL DISTRIBUTED LAG
print("\n\nRATIONAL DISTRIBUTED LAG")
print("=" * 70)
print("Model: y_t = γ₀ + γ₁·y_{t-1} + β₀·x_t + δ₁·x_{t-1} + v_t")
print("More flexible: allows non-monotonic lag pattern")

rational = smf.ols(
    formula="linvpc_det ~ linvpc_det_lag1 + gprice + gprice_lag1",
    data=hseinv,
).fit()

table_rational = pd.DataFrame(
    {
        "Coefficient": rational.params,
        "Std. Error": rational.bse,
        "t-statistic": rational.tvalues,
        "p-value": rational.pvalues,
    },
)

display(table_rational.round(4))

print(f"\nR-squared: {rational.rsquared:.4f}")
print(f"Observations: {rational.nobs:.0f}")

print("\nINTERPRETATION:")
beta_0_rat = rational.params["gprice"]
delta_1 = rational.params["gprice_lag1"]
rho_rat = rational.params["linvpc_det_lag1"]

print(f"  β₀ (contemporaneous effect): {beta_0_rat:.4f}")
print(f"  δ₁ (one-period lag effect): {delta_1:.4f}")
print(f"  ρ (persistence): {rho_rat:.4f}")

# Long-Run Propensity for rational DL
lrp_rational = (beta_0_rat + delta_1) / (1 - rho_rat)
print(f"\n  LRP (long-run effect): {lrp_rational:.4f}")
print(
    f"    → LRP = (β₀ + δ₁)/(1-ρ) = ({beta_0_rat:.4f} + {delta_1:.4f})/(1-{rho_rat:.4f}) = {lrp_rational:.4f}"
)

if delta_1 < 0:
    print("\n  ✓ Negative lagged effect: price growth has smaller impact over time")
else:
    print("\n  ✓ Positive lagged effect: price growth has amplified impact initially")

# %%
# Compare the two models
print("\n\nCOMPARISON: KOYCK vs RATIONAL DISTRIBUTED LAG")
print("=" * 70)

comparison = pd.DataFrame(
    {
        "Koyck": [
            koyck.params["gprice"],
            np.nan,
            koyck.params["linvpc_det_lag1"],
            lrp_koyck,
            koyck.rsquared,
        ],
        "Rational": [
            rational.params["gprice"],
            rational.params["gprice_lag1"],
            rational.params["linvpc_det_lag1"],
            lrp_rational,
            rational.rsquared,
        ],
    },
    index=["β₀ (immediate)", "δ₁ (lagged x)", "ρ (lagged y)", "LRP", "R²"],
)

display(comparison.round(4))

print("\nKEY INSIGHTS:")
print("1. Both models show positive immediate effect of price growth on investment")
print("2. Rational DL has slightly better fit (higher R²)")
print(f"3. LRP is larger in Koyck ({lrp_koyck:.4f}) than Rational ({lrp_rational:.4f})")
print("4. Rational DL shows negative lagged effect → initial enthusiasm fades")

print("\nCHOICE OF MODEL:")
if rational.rsquared > koyck.rsquared + 0.01:
    print("✓ Rational DL preferred: better fit, more flexible")
else:
    print("✓ Koyck preferred: simpler, similar fit")

# %% [markdown]
# ## 18.2 Testing for Unit Roots
#
# A time series has a **unit root** if it follows a **random walk**:
#
# $$ y_t = y_{t-1} + u_t $$
#
# where $u_t$ is stationary. This means:
# - $y_t$ is **non-stationary** (variance grows over time)
# - Shocks have **permanent effects** (no mean reversion)
# - Standard inference breaks down (t-statistics don't follow t-distribution)
#
# ### AR(1) Model
#
# Consider the general AR(1):
#
# $$ y_t = \rho y_{t-1} + u_t $$
#
# Three cases:
# 1. $|\rho| < 1$: **Stationary** (mean-reverting)
# 2. $\rho = 1$: **Unit root** (random walk, non-stationary)
# 3. $|\rho| > 1$: **Explosive** (diverges)
#
# ### Dickey-Fuller Test
#
# To test for a unit root:
#
# **Null hypothesis**: $H_0: \rho = 1$ (unit root, non-stationary)
# **Alternative**: $H_1: \rho < 1$ (stationary)
#
# Subtract $y_{t-1}$ from both sides:
#
# $$ \Delta y_t = (\rho - 1) y_{t-1} + u_t = \theta y_{t-1} + u_t $$
#
# where $\theta = \rho - 1$.
#
# **Test**: $H_0: \theta = 0$ vs $H_1: \theta < 0$
#
# **Problem**: Under $H_0$, the t-statistic does NOT follow a t-distribution!
#
# **Solution**: Use **Dickey-Fuller critical values** (more negative than t-distribution).
#
# ### Augmented Dickey-Fuller (ADF) Test
#
# If $u_t$ is serially correlated, add lags of $\Delta y_t$:
#
# $$ \Delta y_t = \alpha + \theta y_{t-1} + \gamma_1 \Delta y_{t-1} + \cdots + \gamma_p \Delta y_{t-p} + u_t $$
#
# This is the **ADF test**.
#
# **Three specifications**:
# 1. **No constant, no trend**: $\Delta y_t = \theta y_{t-1} + \cdots$
# 2. **With constant**: $\Delta y_t = \alpha + \theta y_{t-1} + \cdots$
# 3. **With constant and trend**: $\Delta y_t = \alpha + \beta t + \theta y_{t-1} + \cdots$
#
# **Decision rule**:
# - If ADF statistic < critical value → **Reject $H_0$** (stationary)
# - If ADF statistic > critical value → **Fail to reject $H_0$** (unit root)
#
# ### Example 18.4: Unit Root Test for GDP

# %%
# Load inventory data
inven = wool.data("inven")
inven["lgdp"] = np.log(inven["gdp"])

print("\n18.2 TESTING FOR UNIT ROOTS")
print("=" * 70)
print("\nEXAMPLE: Is log GDP stationary or does it have a unit root?")

print(f"\nTotal observations: {len(inven)}")
print("Time period: 1959Q1-2000Q4 (quarterly data)")

print("\nVARIABLE:")
print("  gdp  = US real GDP")
print("  lgdp = log(GDP)")

print("\nQUESTION: Does log GDP have a unit root?")
print("  H₀: Unit root (non-stationary, random walk)")
print("  H₁: Stationary (mean-reverting)")

# Summary statistics
display(inven[["gdp", "lgdp"]].describe().round(4))

# %%
# Visualize log GDP
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Log GDP level
axes[0].plot(inven.index, inven["lgdp"], "b-", linewidth=2)
axes[0].set_xlabel("Quarter")
axes[0].set_ylabel("Log GDP")
axes[0].set_title("Log Real GDP (1959Q1-2000Q4)")
axes[0].grid(True, alpha=0.3)

# Panel B: First difference of log GDP (growth rate)
inven["dlgdp"] = inven["lgdp"].diff()
axes[1].plot(inven.index, inven["dlgdp"], "g-", linewidth=2)
axes[1].axhline(y=0, color="black", linestyle="--", linewidth=1)
axes[1].set_xlabel("Quarter")
axes[1].set_ylabel("Δ Log GDP (Growth Rate)")
axes[1].set_title("GDP Growth Rate")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nVISUAL INSPECTION:")
print("✓ Log GDP trends strongly upward → suggests non-stationarity")
print("✓ Growth rate fluctuates around constant mean → suggests stationarity")
print("→ Need formal test to confirm!")

# %%
# AUGMENTED DICKEY-FULLER TEST
print("\n\nAUGMENTED DICKEY-FULLER (ADF) TEST")
print("=" * 70)
print("Test: H₀: θ = 0 (unit root) vs H₁: θ < 0 (stationary)")
print("Model: Δy_t = α + β·t + θ·y_{t-1} + γ₁·Δy_{t-1} + u_t")
print("       (with constant and trend)")

# Perform ADF test with constant and trend
adf_results = adfuller(
    inven["lgdp"].dropna(),
    maxlag=1,
    regression="ct",  # constant and trend
    autolag=None,
)

# Extract results
adf_stat = adf_results[0]
adf_pval = adf_results[1]
adf_usedlag = adf_results[2]
adf_crit = adf_results[4]

print(f"\nADF statistic: {adf_stat:.4f}")
print(f"p-value: {adf_pval:.4f}")
print(f"Lags used: {adf_usedlag}")
print("Critical values:")
print(f"  1%: {adf_crit['1%']:.4f}")
print(f"  5%: {adf_crit['5%']:.4f}")
print(f"  10%: {adf_crit['10%']:.4f}")

print("\nADF REGRESSION OUTPUT:")
print(f"ADF test statistic: {adf_stat:.4f}")
print(f"p-value: {adf_pval:.4f}")
print(f"Lags used: {adf_usedlag}")

print("\nCRITICAL VALUES:")
for key, value in adf_crit.items():
    print(f"  {key}: {value:.4f}")

print("\nDECISION:")
if adf_stat < adf_crit["5%"]:
    print(f"✓ ADF statistic ({adf_stat:.4f}) < critical value ({adf_crit['5%']:.4f})")
    print("  → REJECT H₀: Log GDP is stationary")
else:
    print(f"✗ ADF statistic ({adf_stat:.4f}) > critical value ({adf_crit['5%']:.4f})")
    print("  → FAIL TO REJECT H₀: Log GDP has a unit root")
    print("  → Log GDP is NON-STATIONARY")

print("\nINTERPRETATION:")
print("  → Log GDP appears to have a unit root")
print("  → Shocks to GDP have permanent effects")
print("  → GDP follows a random walk with drift")

# %%
# Test first difference (should be stationary)
print("\n\nADF TEST ON FIRST DIFFERENCE (GDP Growth)")
print("=" * 70)
print("If level has unit root, first difference should be stationary")

adf_diff = adfuller(
    inven["dlgdp"].dropna(),
    maxlag=1,
    regression="c",  # constant only
    autolag=None,
)

adf_stat_diff = adf_diff[0]
adf_pval_diff = adf_diff[1]
adf_crit_diff = adf_diff[4]

print(f"\nADF statistic: {adf_stat_diff:.4f}")
print(f"p-value: {adf_pval_diff:.4f}")

print("\nCritical values:")
for key, value in adf_crit_diff.items():
    print(f"  {key}: {value:.4f}")

print("\nDECISION:")
if adf_stat_diff < adf_crit_diff["5%"]:
    print(
        f"✓ ADF statistic ({adf_stat_diff:.4f}) < critical value ({adf_crit_diff['5%']:.4f})"
    )
    print("  → REJECT H₀: GDP growth is stationary")
    print("  → Log GDP is I(1): integrated of order 1")
    print("  → First difference is stationary")
else:
    print(
        f"✗ ADF statistic ({adf_stat_diff:.4f}) > critical value ({adf_crit_diff['5%']:.4f})"
    )
    print("  → FAIL TO REJECT H₀")

print("\nCONCLUSION:")
print("✓ Log GDP has a unit root (non-stationary in levels)")
print("✓ First difference (growth rate) is stationary")
print("✓ Log GDP is I(1): need to difference once to achieve stationarity")
print("→ Use growth rates for regression, not levels!")

# %% [markdown]
# ## 18.3 Spurious Regression
#
# **Spurious regression** occurs when two unrelated non-stationary series appear to be significantly related just because they both trend over time.
#
# ### The Problem
#
# Consider two **independent random walks**:
#
# $$ y_t = y_{t-1} + u_t $$
# $$ x_t = x_{t-1} + v_t $$
#
# where $u_t$ and $v_t$ are independent white noise.
#
# **Key facts**:
# - $y_t$ and $x_t$ are **completely unrelated**
# - Both have **unit roots** (non-stationary)
# - Both **trend** randomly over time
#
# **What happens if we regress $y_t$ on $x_t$?**
#
# $$ y_t = \alpha + \beta x_t + e_t $$
#
# **Shocking result**:
# - $\hat{\beta}$ will often appear **highly significant** (large t-statistic)
# - $R^2$ can be high
# - But the relationship is **spurious** (not real)!
#
# ### Why Does This Happen?
#
# - Both $y_t$ and $x_t$ have **stochastic trends** (persistent movements)
# - OLS finds patterns in these trends
# - Standard errors are **too small** (classical assumptions violated)
# - t-statistics are **too large** (don't follow t-distribution)
# - Result: **false rejection** of $H_0: \beta = 0$
#
# ### Detection
#
# **Warning signs of spurious regression**:
# 1. High $R^2$ but very low Durbin-Watson statistic ($DW \approx 0$)
# 2. Non-stationary variables (unit roots confirmed by ADF test)
# 3. Residuals are non-stationary (have unit root)
#
# **Rule of thumb** (Granger-Newbold):
# - If $R^2 > DW$, be suspicious of spurious regression
#
# ### Solution
#
# 1. **Test for unit roots** in both $y_t$ and $x_t$
# 2. If both I(1), **test for cointegration** (next section)
# 3. If NOT cointegrated, use **first differences**:
#    $$ \Delta y_t = \beta \Delta x_t + u_t $$
#
# ### Simulation: Spurious Regression

# %%
print("\n18.3 SPURIOUS REGRESSION")
print("=" * 70)
print("\nDEMONSTRATION: Two independent random walks can appear related!")

# Set seed for reproducibility
np.random.seed(123456)

# Generate two INDEPENDENT random walks
n = 51
e = stats.norm.rvs(0, 1, size=n)
e[0] = 0
a = stats.norm.rvs(0, 1, size=n)
a[0] = 0

# Cumulative sums (random walks)
x = np.cumsum(a)
y = np.cumsum(e)

sim_data = pd.DataFrame({"y": y, "x": x})

print(f"\nGenerated {n} observations of two INDEPENDENT random walks")
print("  x_t = x_{t-1} + a_t, where a_t ~ N(0,1)")
print("  y_t = y_{t-1} + e_t, where e_t ~ N(0,1)")
print("  Cov(a_t, e_t) = 0  → x and y are UNRELATED")

# Visualize the two series
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Time series plot
axes[0].plot(x, "b-", linewidth=2, label="x (random walk)")
axes[0].plot(y, "r--", linewidth=2, label="y (random walk)")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Value")
axes[0].set_title("Two Independent Random Walks")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Panel B: Scatter plot
axes[1].scatter(x, y, alpha=0.6, s=50)
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_title("Scatter Plot: x vs y")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nOBSERVATION:")
print("✓ Both series wander randomly (unit roots)")
print("✓ They may appear to move together by chance")

# %%
# Spurious regression
print("\nSPURIOUS REGRESSION: y = α + β·x + e")
print("=" * 70)

spurious_reg = smf.ols(formula="y ~ x", data=sim_data).fit()

table_spurious = pd.DataFrame(
    {
        "Coefficient": spurious_reg.params,
        "Std. Error": spurious_reg.bse,
        "t-statistic": spurious_reg.tvalues,
        "p-value": spurious_reg.pvalues,
    },
)

display(table_spurious.round(4))

print(f"\nR-squared: {spurious_reg.rsquared:.4f}")
print(f"Durbin-Watson: {sm.stats.stattools.durbin_watson(spurious_reg.resid):.4f}")

print("\nSHOCKING RESULT:")
beta_hat = spurious_reg.params["x"]
t_stat = spurious_reg.tvalues["x"]
pval = spurious_reg.pvalues["x"]

if pval < 0.05:
    print(
        f"✗ β̂ = {beta_hat:.4f} appears HIGHLY SIGNIFICANT (t = {t_stat:.2f}, p = {pval:.4f})"
    )
    print("  → OLS says x and y are related")
    print("  → But we KNOW they are independent!")
    print("  → This is SPURIOUS REGRESSION")
else:
    print(f"  β̂ = {beta_hat:.4f} is not significant (t = {t_stat:.2f}, p = {pval:.4f})")
    print("  → This particular simulation avoided spurious regression")
    print("  → But it happens frequently with non-stationary data!")

# Durbin-Watson diagnostic
dw = sm.stats.stattools.durbin_watson(spurious_reg.resid)
if spurious_reg.rsquared > dw:
    print(f"\n✗ WARNING: R² ({spurious_reg.rsquared:.4f}) > DW ({dw:.4f})")
    print("  → Granger-Newbold rule suggests spurious regression")

# %%
# Test residuals for unit root
print("\n\nTESTING RESIDUALS FOR UNIT ROOT")
print("=" * 70)
print("If residuals have unit root → spurious regression")

adf_resid = adfuller(spurious_reg.resid.dropna(), maxlag=1, regression="c")

print(f"\nADF statistic on residuals: {adf_resid[0]:.4f}")
print(f"p-value: {adf_resid[1]:.4f}")
print(f"Critical value (5%): {adf_resid[4]['5%']:.4f}")

if adf_resid[0] > adf_resid[4]["5%"]:
    print("\n✗ Residuals have unit root (non-stationary)")
    print("  → Confirms spurious regression")
    print("  → Standard inference is invalid!")
else:
    print("\n✓ Residuals are stationary")
    print("  → Not spurious (or cointegrated - see next section)")

# %%
# Correct approach: use first differences
print("\n\nCORRECT APPROACH: First Differences")
print("=" * 70)
print("Model: Δy_t = β·Δx_t + u_t")

# Calculate first differences
sim_data["dy"] = sim_data["y"].diff()
sim_data["dx"] = sim_data["x"].diff()

correct_reg = smf.ols(formula="dy ~ dx", data=sim_data).fit()

table_correct = pd.DataFrame(
    {
        "Coefficient": correct_reg.params,
        "Std. Error": correct_reg.bse,
        "t-statistic": correct_reg.tvalues,
        "p-value": correct_reg.pvalues,
    },
)

display(table_correct.round(4))

print(f"\nR-squared: {correct_reg.rsquared:.4f}")
print(f"Durbin-Watson: {sm.stats.stattools.durbin_watson(correct_reg.resid):.4f}")

print("\nCORRECT INFERENCE:")
beta_diff = correct_reg.params["dx"]
t_diff = correct_reg.tvalues["dx"]
pval_diff = correct_reg.pvalues["dx"]

if pval_diff >= 0.05:
    print(
        f"✓ β̂ = {beta_diff:.4f} is NOT significant (t = {t_diff:.2f}, p = {pval_diff:.4f})"
    )
    print("  → Correctly finds NO relationship")
    print("  → First differences avoid spurious regression")
else:
    print(
        f"  β̂ = {beta_diff:.4f} is significant (t = {t_diff:.2f}, p = {pval_diff:.4f})"
    )
    print("  → Type I error (but much less likely than with levels)")

# %%
# Multiple simulations
print("\n\nMULTIPLE SIMULATIONS: Frequency of Spurious Regression")
print("=" * 70)

np.random.seed(42)
n_sims = 1000
significant_count = 0

for _ in range(n_sims):
    # Generate independent random walks
    e_sim = np.cumsum(stats.norm.rvs(0, 1, size=51))
    a_sim = np.cumsum(stats.norm.rvs(0, 1, size=51))

    # Regress
    sim_df = pd.DataFrame({"y": e_sim, "x": a_sim})
    reg_sim = smf.ols(formula="y ~ x", data=sim_df).fit()

    # Check significance
    if reg_sim.pvalues["x"] < 0.05:
        significant_count += 1

spurious_rate = significant_count / n_sims

print(f"\nSimulated {n_sims} pairs of independent random walks")
print("Regressed y on x in each simulation")
print("\nRESULTS:")
print(f"  Significant at 5% level: {significant_count} out of {n_sims}")
print(f"  Spurious regression rate: {100 * spurious_rate:.1f}%")
print("\nEXPECTED if variables were stationary: 5%")
print(f"ACTUAL with non-stationary variables: {100 * spurious_rate:.1f}%")

if spurious_rate > 0.10:
    print("\n✗ MUCH higher than 5%!")
    print("  → Non-stationary variables produce spurious relationships")
    print("  → ALWAYS test for unit roots before regressing levels")

# %% [markdown]
# ## 18.4 Cointegration and Error Correction Models
#
# **Cointegration** is a special case where two non-stationary I(1) variables have a **stationary linear combination**.
#
# ### The Concept
#
# Suppose $y_t$ and $x_t$ are both I(1) (have unit roots). Generally:
# - Any linear combination $y_t - \beta x_t$ is also I(1)
# - Regression of $y_t$ on $x_t$ is spurious
#
# **BUT** if there exists $\beta$ such that:
#
# $$ y_t - \beta x_t = u_t \quad \text{where } u_t \text{ is I(0) (stationary)} $$
#
# then $y_t$ and $x_t$ are **cointegrated** with cointegrating coefficient $\beta$.
#
# **Economic interpretation**:
# - $y_t$ and $x_t$ share a **common stochastic trend**
# - They move together in the **long run**
# - Deviations from equilibrium ($u_t$) are temporary
# - There's a genuine **long-run relationship**, not spurious!
#
# ### Testing for Cointegration
#
# **Engle-Granger two-step procedure**:
#
# 1. **Step 1**: Regress $y_t$ on $x_t$ (cointegrating regression)
#    $$ y_t = \alpha + \beta x_t + u_t $$
#    Save residuals $\hat{u}_t = y_t - \hat{\alpha} - \hat{\beta} x_t$
#
# 2. **Step 2**: Test if $\hat{u}_t$ is stationary using ADF test
#    - $H_0$: No cointegration ($\hat{u}_t$ has unit root)
#    - $H_1$: Cointegration ($\hat{u}_t$ is stationary)
#    - Use special **Engle-Granger critical values** (more negative than standard ADF)
#
# **Alternative**: Use `coint` function which performs both steps.
#
# ### Error Correction Model (ECM)
#
# If $y_t$ and $x_t$ are cointegrated, we can estimate an **error correction model**:
#
# $$ \Delta y_t = \gamma + \theta_1 \Delta x_t + \theta_2 (y_{t-1} - \beta x_{t-1}) + \text{error} $$
#
# or equivalently:
#
# $$ \Delta y_t = \gamma + \theta_1 \Delta x_t + \theta_2 \hat{u}_{t-1} + \text{error} $$
#
# where $\hat{u}_{t-1}$ is the lagged residual from the cointegrating regression.
#
# **Interpretation**:
# - $\theta_1$: **Short-run effect** of $\Delta x_t$ on $\Delta y_t$
# - $\theta_2$: **Error correction coefficient** (should be negative)
#   - If $y_{t-1} > \beta x_{t-1}$ (above equilibrium), then $\Delta y_t$ should decrease
#   - Speed of adjustment back to equilibrium
# - $\beta$: **Long-run effect** from cointegrating regression
#
# **Key advantage**: ECM is estimated in **first differences** (stationary) but includes **levels information** through error correction term.
#
# ### Example: Cointegration Test (Conceptual)

# %%
print("\n18.4 COINTEGRATION AND ERROR CORRECTION MODELS")
print("=" * 70)

print("\nCONCEPT: Cointegration")
print("Two I(1) variables can have a stationary linear combination")

print("\nEXAMPLE: Consumption and Income")
print("  - Both consumption (C) and income (Y) are I(1)")
print("  - But C_t - β·Y_t might be stationary")
print("  - Economic theory: long-run consumption proportional to income")
print("  - Deviations from this relationship are temporary")

print("\nKEY DIFFERENCE FROM SPURIOUS REGRESSION:")
print("  Spurious: y and x are unrelated, just both trending")
print("  Cointegration: y and x share a common trend (genuine relationship)")

print("\nTESTING PROCEDURE:")
print("  1. Confirm both y and x are I(1) (unit root tests)")
print("  2. Regress y_t on x_t → get residuals û_t")
print("  3. Test if û_t is stationary (ADF test)")
print("  4. If stationary → cointegrated (long-run relationship exists)")
print("  5. Estimate error correction model (ECM)")

# %%
# Simulation: Cointegrated variables
print("\n\nSIMULATION: Cointegrated Variables")
print("=" * 70)

np.random.seed(789)
n = 100

# Generate common stochastic trend
trend = np.cumsum(stats.norm.rvs(0, 1, size=n))

# Two variables sharing this trend, plus stationary deviations
x_coint = trend + stats.norm.rvs(0, 0.5, size=n)
y_coint = 2 + 1.5 * trend + stats.norm.rvs(0, 0.5, size=n)

coint_data = pd.DataFrame({"y": y_coint, "x": x_coint})

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: Time series
axes[0].plot(y_coint, "b-", linewidth=2, label="y")
axes[0].plot(x_coint, "r--", linewidth=2, label="x")
axes[0].set_xlabel("Time")
axes[0].set_ylabel("Value")
axes[0].set_title("Cointegrated Variables (Share Common Trend)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Panel B: Scatter
axes[1].scatter(x_coint, y_coint, alpha=0.6, s=50)
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_title("Strong Linear Relationship")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nOBSERVATION:")
print("✓ Both variables trend upward (unit roots)")
print("✓ They move together (share common trend)")
print("✓ Unlike spurious regression, this is a REAL relationship")

# %%
# Step 1: Cointegrating regression
print("\nSTEP 1: Cointegrating Regression")
print("=" * 70)
print("Regress: y_t = α + β·x_t + u_t")

coint_reg = smf.ols(formula="y ~ x", data=coint_data).fit()

table_coint = pd.DataFrame(
    {
        "Coefficient": coint_reg.params,
        "Std. Error": coint_reg.bse,
        "t-statistic": coint_reg.tvalues,
        "p-value": coint_reg.pvalues,
    },
)

display(table_coint.round(4))

print(f"\nR-squared: {coint_reg.rsquared:.4f}")

print("\nLONG-RUN RELATIONSHIP:")
beta_coint = coint_reg.params["x"]
print(f"  β̂ = {beta_coint:.4f}")
print(f"  → Long-run: y ≈ {coint_reg.params['Intercept']:.2f} + {beta_coint:.2f}·x")

# Save residuals
coint_data["resid"] = coint_reg.resid

# %%
# Step 2: Test residuals for stationarity
print("\nSTEP 2: Test Residuals for Stationarity (ADF Test)")
print("=" * 70)
print("H₀: No cointegration (residuals have unit root)")
print("H₁: Cointegration (residuals are stationary)")

adf_coint = adfuller(coint_data["resid"].dropna(), maxlag=1, regression="c")

print(f"\nADF statistic: {adf_coint[0]:.4f}")
print(f"p-value: {adf_coint[1]:.4f}")
print(f"Critical value (5%): {adf_coint[4]['5%']:.4f}")

# Note: Should use Engle-Granger critical values in practice
# Standard ADF critical values are too conservative for cointegration test

if adf_coint[0] < adf_coint[4]["5%"]:
    print("\n✓ Residuals are stationary")
    print("  → REJECT H₀: Variables are COINTEGRATED")
    print("  → Long-run relationship is genuine, not spurious")
else:
    print("\n✗ Residuals have unit root")
    print("  → FAIL TO REJECT H₀: Variables are NOT cointegrated")
    print("  → Relationship may be spurious")

# %%
# Alternative: Using coint function
print("\n\nALTERNATIVE: Using statsmodels coint() function")
print("=" * 70)

coint_test = coint(coint_data["y"], coint_data["x"])
coint_stat = coint_test[0]
coint_pval = coint_test[1]
coint_crit = coint_test[2]

print(f"Cointegration test statistic: {coint_stat:.4f}")
print(f"p-value: {coint_pval:.4f}")
print(f"Critical values: {coint_crit}")

if coint_pval < 0.05:
    print("\n✓ p-value < 0.05 → COINTEGRATED")
    print("  → Can estimate error correction model")
else:
    print("\n✗ p-value ≥ 0.05 → NOT cointegrated")
    print("  → Use first differences instead")

# %%
# Error Correction Model (ECM)
print("\n\nERROR CORRECTION MODEL (ECM)")
print("=" * 70)
print("Model: Δy_t = γ + θ₁·Δx_t + θ₂·û_{t-1} + error")
print("where û_{t-1} = y_{t-1} - β̂·x_{t-1} (lagged residual)")

# Create variables for ECM
coint_data["dy"] = coint_data["y"].diff()
coint_data["dx"] = coint_data["x"].diff()
coint_data["resid_lag1"] = coint_data["resid"].shift(1)

# Estimate ECM
ecm = smf.ols(formula="dy ~ dx + resid_lag1", data=coint_data).fit()

table_ecm = pd.DataFrame(
    {
        "Coefficient": ecm.params,
        "Std. Error": ecm.bse,
        "t-statistic": ecm.tvalues,
        "p-value": ecm.pvalues,
    },
)

display(table_ecm.round(4))

print(f"\nR-squared: {ecm.rsquared:.4f}")

print("\nINTERPRETATION:")
theta_1 = ecm.params["dx"]
theta_2 = ecm.params["resid_lag1"]

print(f"  θ₁ (short-run effect): {theta_1:.4f}")
print("    → Immediate impact of Δx on Δy")

print(f"\n  θ₂ (error correction): {theta_2:.4f}")
if theta_2 < 0:
    print("    ✓ Negative (as expected)")
    print("    → If y_{t-1} > β̂·x_{t-1} (above equilibrium), Δy_t decreases")
    print(
        f"    → Adjustment speed: {abs(theta_2):.1%} of deviation corrected per period"
    )
else:
    print("    ✗ Positive (unexpected)")
    print("    → May indicate misspecification")

print(f"\n  Long-run effect: β̂ = {beta_coint:.4f}")
print("    → From cointegrating regression")
print(f"    → Permanent 1-unit increase in x → {beta_coint:.2f}-unit increase in y")

# %% [markdown]
# ## 18.5 Forecasting
#
# **Forecasting** is a key application of time series regression. We use historical data to predict future values.
#
# ### Types of Forecasts
#
# **1. One-step-ahead forecast** ($h=1$):
# $$ \hat{y}_{T+1} = \hat{\alpha} + \hat{\beta}_1 y_T + \hat{\beta}_2 x_T $$
#
# **2. Multi-step-ahead forecast** ($h>1$):
# - Use predicted values to forecast further ahead
# - Example: $\hat{y}_{T+2} = \hat{\alpha} + \hat{\beta}_1 \hat{y}_{T+1} + \hat{\beta}_2 x_{T+1}$
#
# ### Forecast Errors and Evaluation
#
# **Forecast error**: $e_{T+h} = y_{T+h} - \hat{y}_{T+h}$
#
# **Evaluation metrics**:
# - **RMSE** (Root Mean Squared Error): $\text{RMSE} = \sqrt{\frac{1}{m} \sum_{h=1}^{m} e_{T+h}^2}$
#   - Penalizes large errors more heavily
#   - Same units as $y$
#
# - **MAE** (Mean Absolute Error): $\text{MAE} = \frac{1}{m} \sum_{h=1}^{m} |e_{T+h}|$
#   - Treats all errors equally
#   - Same units as $y$
#   - More robust to outliers
#
# **Lower is better** for both metrics.
#
# ### Forecast Intervals
#
# **Point forecast**: $\hat{y}_{T+h}$
#
# **95% forecast interval**:
# $$ \hat{y}_{T+h} \pm 1.96 \times SE(\hat{y}_{T+h}) $$
#
# where $SE(\hat{y}_{T+h})$ accounts for:
# - Parameter uncertainty (estimation error)
# - Innovation uncertainty (future shocks)
#
# **Interpretation**: We're 95% confident the actual value will fall in this range.
#
# ### Example 18-8: Forecasting Unemployment
#
# **Goal**: Forecast unemployment rate using past unemployment (AR model) and compare with model that includes inflation.
#
# **Data**: `phillips` - Annual US data 1948-2003
# - `unem`: Unemployment rate
# - `inf`: Inflation rate
#
# **Estimation period**: 1948-1996 (49 observations)
# **Forecast period**: 1997-2003 (7 years out-of-sample)
#
# **Two models**:
# 1. **AR(1)**: $\text{unem}_t = \alpha + \beta_1 \text{unem}_{t-1} + u_t$
# 2. **ARX**: $\text{unem}_t = \alpha + \beta_1 \text{unem}_{t-1} + \beta_2 \text{inf}_{t-1} + u_t$

# %%
print("\n18.5 FORECASTING")
print("=" * 70)
print("\nEXAMPLE 18-8: Forecasting Unemployment Rate")

# Load data
phillips = wool.data("phillips")

print(f"\nTotal observations: {len(phillips)}")
print("Time period: 1948-2003 (annual data)")
print("\nVARIABLES:")
print("  unem   = Unemployment rate (%)")
print("  inf    = Inflation rate (%)")
print("  unem_1 = Lagged unemployment")
print("  inf_1  = Lagged inflation")

# Create time index
phillips.index = pd.date_range(start="1948", periods=len(phillips), freq="YE").year

# Sample split
train_mask = phillips["year"] <= 1996
test_mask = phillips["year"] > 1996

print(f"\nESTIMATION PERIOD: 1948-1996 ({train_mask.sum()} observations)")
print(f"FORECAST PERIOD:   1997-2003 ({test_mask.sum()} observations)")

# Display summary statistics
print("\nSUMMARY STATISTICS (Full sample):")
display(phillips[["unem", "inf", "unem_1", "inf_1"]].describe().round(2))

# %%
# Model 1: AR(1) without inflation
print("\nMODEL 1: AR(1) - Unemployment Only")
print("=" * 70)
print("Specification: unem_t = α + β₁·unem_{t-1} + u_t")

model_1 = smf.ols(formula="unem ~ unem_1", data=phillips[train_mask]).fit()

table_1 = pd.DataFrame(
    {
        "Coefficient": model_1.params,
        "Std. Error": model_1.bse,
        "t-statistic": model_1.tvalues,
        "p-value": model_1.pvalues,
    },
)

display(table_1.round(4))

print(f"\nR-squared: {model_1.rsquared:.4f}")
print(f"Adj. R-squared: {model_1.rsquared_adj:.4f}")

print("\nINTERPRETATION:")
beta_1 = model_1.params["unem_1"]
print(f"  β̂₁ = {beta_1:.4f}")
if 0 < beta_1 < 1:
    print("  → Unemployment is persistent (0 < β₁ < 1)")
    print("  → High unemployment today → high unemployment tomorrow")
    print("  → But mean-reverting (β₁ < 1)")

# %%
# Model 2: ARX with inflation
print("\nMODEL 2: ARX - Unemployment and Inflation")
print("=" * 70)
print("Specification: unem_t = α + β₁·unem_{t-1} + β₂·inf_{t-1} + u_t")

model_2 = smf.ols(formula="unem ~ unem_1 + inf_1", data=phillips[train_mask]).fit()

table_2 = pd.DataFrame(
    {
        "Coefficient": model_2.params,
        "Std. Error": model_2.bse,
        "t-statistic": model_2.tvalues,
        "p-value": model_2.pvalues,
    },
)

display(table_2.round(4))

print(f"\nR-squared: {model_2.rsquared:.4f}")
print(f"Adj. R-squared: {model_2.rsquared_adj:.4f}")

print("\nINTERPRETATION:")
beta_inf = model_2.params["inf_1"]
pval_inf = model_2.pvalues["inf_1"]
print(f"  β̂₂ (inflation effect): {beta_inf:.4f}")
if pval_inf < 0.05:
    print(f"  ✓ Statistically significant (p = {pval_inf:.4f})")
    if beta_inf < 0:
        print("  → Higher inflation → lower unemployment (Phillips curve)")
    else:
        print("  → Higher inflation → higher unemployment")
else:
    print(f"  ✗ Not statistically significant (p = {pval_inf:.4f})")
    print("  → Inflation may not help forecast unemployment")

# Compare models
print("\nMODEL COMPARISON:")
print(f"  Model 1 (AR only):  R² = {model_1.rsquared:.4f}")
print(f"  Model 2 (with inf): R² = {model_2.rsquared:.4f}")
if model_2.rsquared > model_1.rsquared:
    print("  → Model 2 fits better in-sample")
else:
    print("  → Model 1 fits better in-sample")

# %%
# Forecasts with 95% intervals
print("\nOUT-OF-SAMPLE FORECASTS: 1997-2003")
print("=" * 70)

# Model 1 forecasts
pred_1 = model_1.get_prediction(phillips[test_mask])
pred_1_df = pred_1.summary_frame(alpha=0.05)[["mean", "obs_ci_lower", "obs_ci_upper"]]
pred_1_df.index = phillips[test_mask].index
pred_1_df.columns = ["Forecast", "Lower 95%", "Upper 95%"]

print("\nMODEL 1 FORECASTS (AR only):")
pred_1_display = pred_1_df.copy()
pred_1_display["Actual"] = phillips.loc[test_mask, "unem"].values
pred_1_display["Error"] = pred_1_display["Actual"] - pred_1_display["Forecast"]
display(pred_1_display.round(2))

# Model 2 forecasts
pred_2 = model_2.get_prediction(phillips[test_mask])
pred_2_df = pred_2.summary_frame(alpha=0.05)[["mean", "obs_ci_lower", "obs_ci_upper"]]
pred_2_df.index = phillips[test_mask].index
pred_2_df.columns = ["Forecast", "Lower 95%", "Upper 95%"]

print("\nMODEL 2 FORECASTS (with inflation):")
pred_2_display = pred_2_df.copy()
pred_2_display["Actual"] = phillips.loc[test_mask, "unem"].values
pred_2_display["Error"] = pred_2_display["Actual"] - pred_2_display["Forecast"]
display(pred_2_display.round(2))

# %%
# Forecast evaluation
print("\nFORECAST EVALUATION")
print("=" * 70)

# Calculate forecast errors
e1 = phillips.loc[test_mask, "unem"].values - pred_1_df["Forecast"].values
e2 = phillips.loc[test_mask, "unem"].values - pred_2_df["Forecast"].values

# RMSE
rmse_1 = np.sqrt(np.mean(e1**2))
rmse_2 = np.sqrt(np.mean(e2**2))

print("\nROOT MEAN SQUARED ERROR (RMSE):")
print(f"  Model 1 (AR only):  {rmse_1:.4f}")
print(f"  Model 2 (with inf): {rmse_2:.4f}")

if rmse_1 < rmse_2:
    improvement = ((rmse_2 - rmse_1) / rmse_2) * 100
    print(f"  ✓ Model 1 is better (RMSE {improvement:.1f}% lower)")
else:
    improvement = ((rmse_1 - rmse_2) / rmse_1) * 100
    print(f"  ✓ Model 2 is better (RMSE {improvement:.1f}% lower)")

# MAE
mae_1 = np.mean(np.abs(e1))
mae_2 = np.mean(np.abs(e2))

print("\nMEAN ABSOLUTE ERROR (MAE):")
print(f"  Model 1 (AR only):  {mae_1:.4f}")
print(f"  Model 2 (with inf): {mae_2:.4f}")

if mae_1 < mae_2:
    improvement = ((mae_2 - mae_1) / mae_2) * 100
    print(f"  ✓ Model 1 is better (MAE {improvement:.1f}% lower)")
else:
    improvement = ((mae_1 - mae_2) / mae_1) * 100
    print(f"  ✓ Model 2 is better (MAE {improvement:.1f}% lower)")

print("\nKEY INSIGHT:")
if rmse_1 < rmse_2 and mae_1 < mae_2:
    print("  → Simple AR(1) model forecasts better than model with inflation")
    print("  → Adding inflation does not improve out-of-sample forecasts")
    print("  → Simpler model preferred (parsimony principle)")
elif rmse_2 < rmse_1 and mae_2 < mae_1:
    print("  → Model with inflation forecasts better")
    print("  → Inflation provides useful information for forecasting unemployment")
else:
    print("  → Mixed results: one metric favors Model 1, another favors Model 2")
    print("  → Consider context and forecast horizon")

# %%
# Visualization
print("\nVISUALIZATION: Forecasts vs Actual")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Panel A: Model 1
actual = phillips.loc[test_mask, "unem"]
axes[0].plot(
    actual.index, actual.values, "ko-", linewidth=2, markersize=8, label="Actual"
)
axes[0].plot(
    pred_1_df.index,
    pred_1_df["Forecast"],
    "b--",
    linewidth=2,
    label="AR forecast",
)
axes[0].fill_between(
    pred_1_df.index,
    pred_1_df["Lower 95%"],
    pred_1_df["Upper 95%"],
    alpha=0.2,
    color="blue",
    label="95% interval",
)
axes[0].set_xlabel("Year")
axes[0].set_ylabel("Unemployment Rate (%)")
axes[0].set_title(f"Model 1: AR(1) Only\nRMSE = {rmse_1:.3f}, MAE = {mae_1:.3f}")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Panel B: Model 2
axes[1].plot(
    actual.index, actual.values, "ko-", linewidth=2, markersize=8, label="Actual"
)
axes[1].plot(
    pred_2_df.index,
    pred_2_df["Forecast"],
    "r-.",
    linewidth=2,
    label="ARX forecast",
)
axes[1].fill_between(
    pred_2_df.index,
    pred_2_df["Lower 95%"],
    pred_2_df["Upper 95%"],
    alpha=0.2,
    color="red",
    label="95% interval",
)
axes[1].set_xlabel("Year")
axes[1].set_ylabel("Unemployment Rate (%)")
axes[1].set_title(
    f"Model 2: ARX with Inflation\nRMSE = {rmse_2:.3f}, MAE = {mae_2:.3f}"
)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nOBSERVATIONS:")
print("✓ Forecast intervals widen for multi-step forecasts (cumulative uncertainty)")
print("✓ Both models track the general trend")
print("✓ Actual unemployment sometimes falls outside forecast intervals")
print("  → Unexpected shocks (recessions, policy changes)")

# %% [markdown]
# ## 18.6 Event Studies with Control Groups
#
# **Event studies** analyze the causal effect of a specific event (policy change, natural disaster, etc.) using time series data.
#
# ### Difference-in-Differences with Time Series
#
# **Setup**:
# - **Treatment group**: Affected by the event at time $T_0$
# - **Control group**: Not affected by the event
# - **Pre-period**: Before $T_0$
# - **Post-period**: After $T_0$
#
# **Model**:
# $$ y_{it} = \alpha + \delta_0 D_i + \delta_1 \text{Post}_t + \beta (D_i \times \text{Post}_t) + u_{it} $$
#
# where:
# - $D_i = 1$ for treatment group, 0 for control
# - $\text{Post}_t = 1$ after event, 0 before
# - $\beta$: **Treatment effect** (difference-in-differences estimator)
#
# **Interpretation of coefficients**:
# - $\delta_0$: Baseline difference between groups (pre-treatment)
# - $\delta_1$: Time trend (affects both groups)
# - $\beta$: **Causal effect** of treatment (DD estimator)
#
# **Key assumption**: **Parallel trends**
# - Without treatment, both groups would follow similar trends
# - Testable in pre-period data
#
# ### With Time Series Complications
#
# Time series event studies must account for:
# 1. **Serial correlation**: Observations over time are correlated
# 2. **Trends**: Both groups may have trends
# 3. **Dynamics**: Effects may build up over time
# 4. **Seasonality**: For high-frequency data
#
# **Extensions**:
# - Add lagged dependent variables (dynamic DiD)
# - Include group-specific trends
# - Use robust standard errors (HAC)
#
# ### Example: Conceptual Framework

# %%
print("\n18.6 EVENT STUDIES WITH CONTROL GROUPS")
print("=" * 70)

print("\nCONCEPT: Difference-in-Differences with Time Series")
print("\nSETUP:")
print("  - Treatment group: Affected by policy/event at time T₀")
print("  - Control group: Not affected")
print("  - Compare trends before and after T₀")

print("\nMODEL:")
print("  y_{it} = α + δ₀·Treat_i + δ₁·Post_t + β·(Treat_i × Post_t) + u_{it}")

print("\nINTERPRETATION:")
print("  α: Baseline level (control group, pre-period)")
print("  δ₀: Pre-treatment difference (treatment - control)")
print("  δ₁: Time trend (affects both groups)")
print("  β: TREATMENT EFFECT (difference-in-differences)")

print("\nKEY ASSUMPTION: Parallel Trends")
print("  → Without treatment, both groups would follow similar trends")
print("  → Testable: check if trends are parallel in pre-period")

print("\nEXAMPLE APPLICATIONS:")
print("  • Minimum wage increase in one state (control: neighboring state)")
print("  • Tax policy change in one region")
print("  • Environmental regulation in one industry")
print("  • COVID-19 policy (e.g., lockdowns in some regions)")

# %%
# Simulation: DiD with Time Series
print("\n\nSIMULATION: Difference-in-Differences Event Study")
print("=" * 70)

np.random.seed(2024)

# Generate data
n_periods = 40  # 20 pre, 20 post
treatment_time = 20

# Control group (never treated)
time = np.arange(1, n_periods + 1)
control = 5 + 0.1 * time + stats.norm.rvs(0, 0.5, size=n_periods)

# Treatment group (treated after period 20)
treatment = 6 + 0.1 * time + stats.norm.rvs(0, 0.5, size=n_periods)
treatment[treatment_time:] += 2  # Treatment effect = 2

# Create DataFrame
# Need to duplicate post indicator for both groups
post_indicator = np.array([0] * treatment_time + [1] * (n_periods - treatment_time))

did_data = pd.DataFrame(
    {
        "time": np.tile(time, 2),
        "group": ["Control"] * n_periods + ["Treatment"] * n_periods,
        "y": np.concatenate([control, treatment]),
        "treat": [0] * n_periods + [1] * n_periods,
        "post": np.tile(post_indicator, 2),
    },
)
did_data["treat_post"] = did_data["treat"] * did_data["post"]

print(f"\nGenerated {n_periods} time periods for 2 groups")
print(f"Treatment occurs at period {treatment_time}")
print("True treatment effect: 2.0")

print("\nDATA STRUCTURE:")
display(did_data.head(10))

# %%
# Visualize parallel trends
print("\nVISUALIZATION: Parallel Trends and Treatment Effect")

fig, ax = plt.subplots(figsize=(12, 6))

# Control group
control_data = did_data[did_data["group"] == "Control"]
ax.plot(
    control_data["time"],
    control_data["y"],
    "b-",
    linewidth=2,
    label="Control",
    alpha=0.7,
)

# Treatment group
treat_data = did_data[did_data["group"] == "Treatment"]
pre_treat = treat_data[treat_data["time"] <= treatment_time]
post_treat = treat_data[treat_data["time"] > treatment_time]

ax.plot(
    pre_treat["time"],
    pre_treat["y"],
    "r--",
    linewidth=2,
    label="Treatment (pre)",
    alpha=0.7,
)
ax.plot(
    post_treat["time"],
    post_treat["y"],
    "r-",
    linewidth=2,
    label="Treatment (post)",
    alpha=0.7,
)

# Mark treatment time
ax.axvline(
    x=treatment_time, color="gray", linestyle=":", linewidth=2, label="Treatment starts"
)

ax.set_xlabel("Time Period")
ax.set_ylabel("Outcome (y)")
ax.set_title("Event Study: Treatment Effect on Time Series")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nOBSERVATIONS:")
print("✓ Pre-treatment: Both groups follow parallel trends")
print("✓ Post-treatment: Treatment group jumps up (treatment effect)")
print("  → Difference-in-differences captures this jump")

# %%
# Estimate DiD model
print("\nDIFFERENCE-IN-DIFFERENCES ESTIMATION")
print("=" * 70)
print("Model: y = α + δ₀·Treat + δ₁·Post + β·(Treat × Post) + u")

did_model = smf.ols(formula="y ~ treat + post + treat_post", data=did_data).fit()

table_did = pd.DataFrame(
    {
        "Coefficient": did_model.params,
        "Std. Error": did_model.bse,
        "t-statistic": did_model.tvalues,
        "p-value": did_model.pvalues,
    },
)

display(table_did.round(4))

print(f"\nR-squared: {did_model.rsquared:.4f}")

print("\nINTERPRETATION:")
alpha = did_model.params["Intercept"]
delta_0 = did_model.params["treat"]
delta_1 = did_model.params["post"]
beta_dd = did_model.params["treat_post"]

print(f"  α̂ (baseline): {alpha:.4f}")
print("    → Control group level in pre-period")

print(f"\n  δ̂₀ (group difference): {delta_0:.4f}")
print(f"    → Treatment group {delta_0:.2f} units higher pre-treatment")

print(f"\n  δ̂₁ (time trend): {delta_1:.4f}")
print(f"    → Both groups increase {delta_1:.2f} units in post-period")

print(f"\n  β̂ (treatment effect): {beta_dd:.4f}")
print("    ✓ CAUSAL EFFECT of treatment")
print(f"    → Treatment causes {beta_dd:.2f}-unit increase in outcome")
print(f"    → True effect = 2.0, estimated = {beta_dd:.2f}")

# Compare to truth
if abs(beta_dd - 2.0) < 0.5:
    print("\n✓ Estimated effect close to true effect (2.0)")
else:
    print("\n⚠ Estimated effect differs from true effect (sampling variation)")

print("\nCONCLUSION:")
print("  → Difference-in-differences successfully identifies treatment effect")
print("  → Control group accounts for common time trends")
print("  → Critical assumption: parallel trends (testable in pre-period)")

# %% [markdown]
# ## Summary
#
# This chapter covered advanced time series topics essential for applied econometrics:
#
# 1. **Infinite Distributed Lag Models**:
#    - Geometric (Koyck) and rational distributed lags
#    - Long-run propensity (LRP) calculations
#    - Flexible modeling of dynamic effects
#
# 2. **Unit Root Testing**:
#    - Dickey-Fuller and Augmented Dickey-Fuller (ADF) tests
#    - Testing for stationarity vs random walk behavior
#    - Critical for avoiding spurious regression
#
# 3. **Spurious Regression**:
#    - False relationships between non-stationary series
#    - Detection: R² > DW, non-stationary residuals
#    - Solution: First differences or cointegration
#
# 4. **Cointegration**:
#    - Long-run equilibrium relationships
#    - Error correction models (ECM)
#    - Combines long-run (levels) and short-run (differences) information
#
# 5. **Forecasting**:
#    - One-step and multi-step forecasts
#    - Forecast evaluation: RMSE, MAE
#    - 95% forecast intervals
#    - Model comparison for out-of-sample performance
#
# 6. **Event Studies**:
#    - Difference-in-differences with time series
#    - Causal effect identification
#    - Parallel trends assumption
#
# ### Decision Tree: Time Series Regression
#
# ```
# Are your variables trending over time?
# ├─ NO → Standard OLS regression (Chapters 10-12)
# └─ YES
#    ├─ Test for unit roots (ADF test)
#    │  ├─ Both stationary (I(0))
#    │  │  └─ → Use levels in regression
#    │  └─ At least one non-stationary (I(1))
#    │     ├─ Test for cointegration
#    │     │  ├─ Cointegrated
#    │     │  │  └─ → Error correction model (ECM)
#    │     │  └─ Not cointegrated
#    │     │     └─ → Use first differences
#    │     └─ Forecasting?
#    │        ├─ One-step ahead → Direct forecast
#    │        ├─ Multi-step → Iterate forecasts
#    │        └─ Evaluate: RMSE, MAE, forecast intervals
# ```
#
# ### Key Takeaways
#
# 1. **Always test for unit roots** before regressing trending variables
# 2. **Spurious regression is a real danger** with non-stationary data
# 3. **Cointegration** allows using levels when a long-run relationship exists
# 4. **Error correction models** combine long-run and short-run dynamics
# 5. **Forecasting** requires out-of-sample evaluation (not just R²)
# 6. **Event studies** need parallel trends assumption for causal inference
#
# ### Best Practices
#
# - Test all time series for unit roots before modeling
# - Check residuals for stationarity (avoid spurious regression)
# - Use robust standard errors (HAC) for time series inference
# - Evaluate forecasts out-of-sample, not just in-sample fit
# - Plot data to visualize trends, breaks, and patterns
# - Consider economic theory when choosing between levels and differences
#
# **Next steps**: Panel data methods (Chapter 13-14) and limited dependent variables (Chapter 17).
#
