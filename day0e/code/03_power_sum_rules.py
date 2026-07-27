"""
Segment 3: Power Rule, Constant Rule & Sum Rule
Differentiates y = 3x^2 + 5x by rule and asserts dy_dx_by_rule(2.0) == 17.0
"""
import numpy as np

def y_func(x):
    return 3 * x**2 + 5 * x

def dy_dx_by_rule(x):
    # Rule: d/dx[3x^2] = 3*(2x) = 6x; d/dx[5x] = 5
    return 6 * x + 5

def dy_dx_numerical(x, h=1e-6):
    return (y_func(x + h) - y_func(x - h)) / (2 * h)

if __name__ == "__main__":
    x_test = 2.0
    analytic = dy_dx_by_rule(x_test)
    numeric = dy_dx_numerical(x_test)
    print(f"For y = 3x^2 + 5x at x = {x_test}:")
    print(f"Analytical dy/dx by rule: {analytic:.2f}")
    print(f"Numerical dy/dx estimate: {numeric:.6f}")
    assert np.isclose(analytic, 17.0), f"Expected 17.0, got {analytic}"
    assert np.isclose(analytic, numeric, atol=1e-5), f"Mismatch: {analytic} vs {numeric}"
    print("✅ ASSERTION PASSED: dy_dx_by_rule(2.0) == 17.0")
