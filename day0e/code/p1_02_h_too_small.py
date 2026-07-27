"""
p1_02_h_too_small.py - Part 1 Segment 1.10
Demonstrates floating-point cancellation when h is too small (1e-8 down to 1e-16)
Prints 0.0 at h = 1e-16 [VERIFIED]
"""
import numpy as np

def f(x):
    return x**2

def secant_slope(f, x, h):
    return (f(x + h) - f(x)) / h

if __name__ == "__main__":
    x_val = 3.0
    h_values = [1e-8, 1e-10, 1e-12, 1e-14, 1e-15, 1e-16]
    
    print("Demonstrating floating-point cancellation for f(x) = x^2 at x = 3.0:")
    for h in h_values:
        slope = secant_slope(f, x_val, h)
        print(f"h={h:<10.0e} slope estimate={slope}")
        
    slope_1e16 = secant_slope(f, x_val, 1e-16)
    print(f"Final slope at h=1e-16: {slope_1e16}")
    assert slope_1e16 == 0.0, f"Expected 0.0 at h=1e-16, got {slope_1e16}"
    print("✅ VERIFIED OUTPUT MATCH: h=1e-16 prints 0.0 due to float64 rounding")
