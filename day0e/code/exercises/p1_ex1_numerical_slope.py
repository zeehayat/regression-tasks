"""
EXERCISE 1 (Segment 1.12) [core]
Write numerical_slope(f, x, h) from scratch without looking back at the lecture files.
Use it to find empirically the h that gives the most accurate slope for f(x) = x^3 at x = 2.0.
True answer is 12.0.
"""
import numpy as np

def numerical_slope(f, x, h):
    # TODO: Implement rise over run formula: (f(x+h) - f(x)) / h
    pass

if __name__ == "__main__":
    def f_cube(x):
        return x**3
    
    x_val = 2.0
    true_slope = 12.0
    
    # Assert stub raises until completed by user
    try:
        res = numerical_slope(f_cube, x_val, 1e-6)
        assert np.isclose(res, true_slope, atol=1e-4), f"Expected 12.0, got {res}"
        print("✅ Exercise 1 Passed!")
    except Exception as e:
        print("❌ Exercise 1 Stub Exception (Expected before completion):", e)
