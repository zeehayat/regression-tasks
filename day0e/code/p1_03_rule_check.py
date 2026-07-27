"""
p1_03_rule_check.py - Part 1 Segment 1.16
Compares rule vs numeric slope for y = 3x^2 + 5x at x = 2.0
Prints: 17.0 and approx 17.0 [VERIFIED]
"""
import numpy as np

def y_func(x):
    return 3 * x**2 + 5 * x

def dy_dx_rule(x):
    # d/dx[3x^2 + 5x] = 6x + 5
    return 6 * x + 5

def dy_dx_numeric(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)

if __name__ == "__main__":
    x_val = 2.0
    by_rule = dy_dx_rule(x_val)
    by_finite_difference = dy_dx_numeric(y_func, x_val)
    
    print(f"by rule:            {by_rule:.1f}")
    print(f"by finite difference: {by_finite_difference:.6f}")
    
    assert np.isclose(by_rule, 17.0), f"Expected 17.0 by rule, got {by_rule}"
    assert np.isclose(by_rule, by_finite_difference, atol=1e-5), "Mismatch between rule and numeric"
    print("✅ VERIFIED OUTPUT MATCH: 17.0 and 17.000000")
