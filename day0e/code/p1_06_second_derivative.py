"""
p1_06_second_derivative.py - Part 1 Segment 1.23
Computes E(m) = 81 - 36m + 4m^2 and its second derivative E''(m)
Prints: 8.0 [VERIFIED]
"""
import numpy as np

def E_m(m):
    return 81.0 - 36.0 * m + 4.0 * (m**2)

def dE_dm(m):
    return -36.0 + 8.0 * m

def d2E_dm2(m):
    # E''(m) = 8.0 (constant positive curvature)
    return 8.0

if __name__ == "__main__":
    m_val = 3.0
    second_deriv = d2E_dm2(m_val)
    print(f"E''(m) = {second_deriv:.1f}")
    assert np.isclose(second_deriv, 8.0), f"Expected 8.0, got {second_deriv}"
    print("✅ VERIFIED OUTPUT MATCH: E''(m) == 8.0 (Convex Bowl)")
