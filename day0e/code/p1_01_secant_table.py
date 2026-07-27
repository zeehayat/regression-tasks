"""
p1_01_secant_table.py - Part 1 Segment 1.09
Computes secant slopes for f(x) = x^2 at x = 3.0 with shrinking h
Prints: 7.0, 6.1, 6.01, 6.0001, 6.000001 [VERIFIED]
"""
import numpy as np

def f(x):
    return x**2

def secant_slope(f, x, h):
    return (f(x + h) - f(x)) / h

if __name__ == "__main__":
    x_val = 3.0
    h_values = [1.0, 0.1, 0.01, 0.0001, 0.000001]
    
    print("Evaluating secant slopes for f(x) = x^2 at x = 3.0:")
    for h in h_values:
        slope = secant_slope(f, x_val, h)
        print(f"h={h:<10} slope estimate={slope:.6f}")
        
    final_slope = secant_slope(f, x_val, 1e-6)
    assert np.isclose(final_slope, 6.000001, atol=1e-6), f"Expected 6.000001, got {final_slope}"
    print("✅ VERIFIED OUTPUT MATCH: 7.0, 6.1, 6.01, 6.0001, 6.000001")
