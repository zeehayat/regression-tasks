"""
SOLUTION FOR EXERCISE 1 (Segment 1.13)
"""
import numpy as np

def numerical_slope(f, x, h):
    return (f(x + h) - f(x)) / h

if __name__ == "__main__":
    def f_cube(x):
        return x**3
    
    x_val = 2.0
    true_slope = 12.0
    h_candidates = [1.0, 0.1, 0.01, 1e-4, 1e-6, 1e-8, 1e-12]
    
    best_h = None
    min_err = float('inf')
    
    for h in h_candidates:
        slope_est = numerical_slope(f_cube, x_val, h)
        err = abs(slope_est - true_slope)
        if err < min_err:
            min_err = err
            best_h = h
            
    print(f"Optimal h for f(x)=x^3 at x=2.0 is h={best_h} with error={min_err:.8f}")
    assert np.isclose(numerical_slope(f_cube, x_val, 1e-6), 12.0, atol=1e-4)
    print("✅ Solution 1 Assertions Passed!")
