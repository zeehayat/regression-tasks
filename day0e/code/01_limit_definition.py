"""
Segment 1 & 2: Limit Definition of Derivative & Numerical Approximations
Proves slope estimate of f(x) = x^2 at x = 3.0 equals 6.0
"""
import numpy as np

def f(x):
    return x**2

def numerical_slope(f, x, h):
    return (f(x + h) - f(x)) / h

def central_numerical_slope(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)

if __name__ == "__main__":
    x_val = 3.0
    h_ladder = [1.0, 0.1, 0.01, 0.0001, 0.000001]
    print("Evaluating slope of f(x) = x^2 at x = 3.0:")
    for h in h_ladder:
        slope_est = numerical_slope(f, x_val, h)
        print(f"h = {h:<8} -> Numerical Slope Estimate = {slope_est:.6f}")
    
    # Exact limit value
    final_slope = numerical_slope(f, x_val, 1e-6)
    print(f"\nFinal slope estimate at h=1e-6: {final_slope:.6f}")
    assert np.isclose(final_slope, 6.0, atol=1e-4), f"Expected 6.0, got {final_slope}"
    print("✅ ASSERTION PASSED: Slope of f(x) = x^2 at x = 3.0 equals 6.0")
