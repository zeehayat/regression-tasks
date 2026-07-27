import matplotlib.pyplot as plt
import numpy as np

# Set seed for reproducible synthetic data
np.random.seed(42)


# =====================================================================
# PART 1: INTUITIVE DERIVATIVE (Numerical "Nudge" Method)
# =====================================================================

# Define our profit function: P(x) = -2x^2 + 24x - 10
def profit_function(x):
    return -2 * (x ** 2) + 24 * x - 10


# A derivative measures: (Change in Output) / (Tiny Change in Input)
def calculate_derivative(func, x, h=0.0001):
    """Calculates the derivative numerically by giving input x a tiny 'nudge' h."""
    change_in_output = func(x + h) - func(x)
    change_in_input = h
    return change_in_output / change_in_input


print("--- PART 1: INTUITIVE UNDERSTANDING ---")
price_test = 3.0
sensitivity = calculate_derivative(profit_function, price_test)

print(f"At a price of ${price_test:.2f}:")
print(f"Current Profit: ${profit_function(price_test):.2f}")
print(f"Derivative (Sensitivity): {sensitivity:.2f}")
print(f"--> Layman Meaning: Increasing price by $1 will BOOST profit by ~${sensitivity:.2f}\n")

# =====================================================================
# PART 2: GENERATE DATA, FIT MODEL, & EXPLAIN INPUT IMPACT
# =====================================================================

# 1. Generate 100 noisy observation points (Simulating real-world customer data)
prices = np.random.uniform(1, 11, 100)
noise = np.random.normal(0, 4, 100)
observed_profit = profit_function(prices) + noise

# 2. Fit a quadratic model (Polynomial of degree 2) to the data
model_coefficients = np.polyfit(prices, observed_profit, 2)
fitted_model = np.poly1d(model_coefficients)

# 3. Create the derivative model (Calculus power rule on fitted coefficients)
derivative_model = np.polyder(fitted_model)

print("--- PART 2: MODEL DERIVATIVES AT DIFFERENT PRICES ---")

# Evaluate three distinct pricing scenarios
test_prices = [3.0, 6.0, 9.0]

for price in test_prices:
    rate_of_change = derivative_model(price)

    print(f"Scenario: Price = ${price:.2f}")
    print(f"  * Estimated Profit : ${fitted_model(price):.2f}")
    print(f"  * Derivative Value : {rate_of_change:.2f}")

    # Translate derivative value into layman terms
    if rate_of_change > 1.0:
        print(
            f"  * Layman Statement : 'Underpriced! A $1 increase in price INCREASES profit by ~${rate_of_change:.2f}.'")
    elif abs(rate_of_change) <= 1.0:
        print(
            f"  * Layman Statement : 'Optimal Price! Profit is at its peak. Price changes here have virtually ZERO impact.'")
    else:
        print(
            f"  * Layman Statement : 'Overpriced! A $1 increase in price DECREASES profit by ~${abs(rate_of_change):.2f}.'")
    print("-" * 65)

# =====================================================================
# VISUALIZATION
# =====================================================================

x_range = np.linspace(1, 11, 200)

plt.figure(figsize=(10, 6))

# Plot raw data points and fitted curve
plt.scatter(prices, observed_profit, color='gray', alpha=0.5, label='Observed Store Data')
plt.plot(x_range, fitted_model(x_range), color='blue', linewidth=2, label='Fitted Profit Model')

# Highlight the tangent line (derivative) at Price = $3.00
x_point = 3.0
y_point = fitted_model(x_point)
slope = derivative_model(x_point)

# Draw tangent line equation: y = m(x - x0) + y0
tangent_x = np.linspace(1.5, 4.5, 50)
tangent_y = slope * (tangent_x - x_point) + y_point
plt.plot(tangent_x, tangent_y, color='red', linestyle='--', linewidth=2,
         label=f'Tangent Line at $3 (Slope/Derivative = {slope:.1f})')

plt.axvline(x=6.0, color='green', linestyle=':', label='Peak Profit (Derivative = 0)')

plt.title('How Derivatives Measure Sensitivity (Price vs. Profit)', fontsize=14)
plt.xlabel('Product Price ($)', fontsize=12)
plt.ylabel('Daily Profit ($)', fontsize=12)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()