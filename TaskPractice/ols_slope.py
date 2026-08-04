"""
OLS (Ordinary Least Squares) Slope Calculation Script

Overview:
---------
This script demonstrates how to calculate the slope coefficient (beta_1) of a 
simple linear regression model using centered sum formulas in Python/NumPy.

What is OLS (Ordinary Least Squares)?
--------------------------------------
Ordinary Least Squares (OLS) is a fundamental statistical regression technique 
used to fit a linear line (y = beta_0 + beta_1 * x) through a set of data points. 
It works by MINIMIZING the Sum of Squared Residuals (SSR):
    SSR = sum((y_i - y_hat_i)^2)
where (y_i - y_hat_i) is the vertical error (residual) between the observed value 
and the predicted line.

What is Slope (beta_1)?
-----------------------
The slope (beta_1) represents the rate of change in the dependent variable (Y) 
for every 1-unit increase in the independent variable (X). Mathematically, it is 
the ratio of the sample covariance of X and Y to the sample variance of X:

    beta_1 = Cov(X, Y) / Var(X)
           = sum((x_i - mean(x)) * (y_i - mean(y))) / sum((x_i - mean(x))^2)

Script Functionality:
---------------------
1. `slope_from_sums(x, y)` computes beta_1 directly from data arrays using centered sums.
2. It evaluates the slope across two separate datasets ("accessible" and "remote").
3. It combines the datasets to illustrate how subgroup slopes compare to the overall 
   pooled slope (demonstrating phenomena like Simpson's Paradox).
"""


import numpy as np
import matplotlib
# Set backend for matplotlib rendering
matplotlib.use('TkAgg') # Use 'TkAgg' or 'QtAgg' based on available GUI backends
import matplotlib.pyplot as plt


def slope_from_sums(x, y):
    """
    Calculate the OLS slope coefficient (beta_1) using centered deviations.

    Formula:
        beta_1 = sum((x - mean(x)) * (y - mean(y))) / sum((x - mean(x))^2)

    Parameters:
        x (array-like): Independent variable values.
        y (array-like): Dependent variable values.

    Returns:
        float: Calculated slope (beta_1).
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Step 1: Center the variables by subtracting their respective sample means
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    
    # Step 2: Compute covariance numerator sum((x - x_bar) * (y - y_bar))
    numerator = np.sum(x_centered * y_centered)
    
    # Step 3: Compute variance denominator sum((x - x_bar)^2)
    denominator = np.sum(x_centered**2)
    
    # Step 4: Calculate slope (beta_1 = numerator / denominator)
    return numerator / denominator


# --- Dataset Definitions ---
# Subgroup 1: Accessible terrain data (Cable Length in km vs Cost in million PKR)
x_accessible = np.array([4, 5, 6, 7], dtype=float)
y_accessible = np.array([28, 30, 32, 34], dtype=float)

# Subgroup 2: Remote terrain data (Cable Length in km vs Cost in million PKR)
x_remote = np.array([1, 2, 3, 4], dtype=float)
y_remote = np.array([30, 33, 36, 39], dtype=float)

# Combined dataset combining accessible and remote observations
x_all = np.concatenate([x_accessible, x_remote])
y_all = np.concatenate([y_accessible, y_remote])

# --- Print Computed Slopes ---
# Demonstrates Simpson's Paradox: accessible slope is +2.0, remote slope is +3.0,
# but the naive combined pooled slope is -2.0 due to confounding group baseline differences.
print("accessible slope: ", slope_from_sums(x_accessible, y_accessible))
print("remote slope:     ", slope_from_sums(x_remote, y_remote))
print("combined slope:   ", slope_from_sums(x_all, y_all))


def line_for(x, y, grid):
    """
    Generate predicted Y values across a grid for an OLS regression line y = b0 + b1 * x.

    Parameters:
        x (array-like): Predictor data used to fit the line.
        y (array-like): Response data used to fit the line.
        grid (array-like): X values to generate predictions for.

    Returns:
        ndarray: Predicted Y values corresponding to `grid`.
    """
    # Calculate slope (b1) and intercept (b0 = mean(y) - b1 * mean(x))
    b1 = slope_from_sums(x, y)
    b0 = y.mean() - b1 * x.mean()
    
    # Predict y values on grid: y = b0 + b1 * grid
    return b0 + b1 * grid


# --- Visualization of Simpson's Paradox in OLS ---
grid = np.linspace(0.5, 7.5, 100)

plt.figure(figsize=(8, 5))

# Plot data points for each subgroup
plt.scatter(x_accessible, y_accessible, c='b', s=40, label='accessible terrain')
plt.scatter(x_remote, y_remote, c='r', s=70, label='remote terrain')

# Plot subgroup regression lines
plt.plot(grid, line_for(x_accessible, y_accessible, grid), linewidth=2, color='blue', label='Accessible trend (slope = +2.0)')
plt.plot(grid, line_for(x_remote, y_remote, grid), linewidth=2, color='red', label='Remote trend (slope = +3.0)')

# Plot overall pooled regression line (misleading due to Simpson's paradox)
plt.plot(grid, line_for(x_all, y_all, grid), linewidth=2, linestyle='--', color='black', label="Misleading combined line (slope = -2.0)")

# Label axis and plot details
plt.xlabel('Cable Length (km)')
plt.ylabel('Actual Cost (million PKR)')
plt.title("Simpson's Paradox in OLS Linear Regression")
plt.legend()
plt.tight_layout()
plt.show()