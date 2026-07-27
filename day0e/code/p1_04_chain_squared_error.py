"""
p1_04_chain_squared_error.py - Part 1 Segment 1.20
Computes dE/dm for E = (y - mx - c)^2 at x=2, y=10, c=1, m=1
Prints: -28.0 three ways [VERIFIED]
"""
import numpy as np

def E_loss(m, x=2.0, y=10.0, c=1.0):
    return (y - m * x - c)**2

def dE_dm_chain(m, x=2.0, y=10.0, c=1.0):
    # dE/dm = -2x(y - mx - c)
    return -2.0 * x * (y - m * x - c)

def dE_dm_expanded(m, x=2.0, y=10.0, c=1.0):
    # Expanded: E = (y-c)^2 - 2(y-c)mx + m^2 x^2
    # dE/dm = -2(y-c)x + 2m x^2
    return -2.0 * (y - c) * x + 2.0 * m * (x**2)

def dE_dm_numeric(m, x=2.0, y=10.0, c=1.0, h=1e-6):
    f_m = lambda m_val: E_loss(m_val, x, y, c)
    return (f_m(m + h) - f_m(m - h)) / (2 * h)

if __name__ == "__main__":
    m_test = 1.0
    by_chain = dE_dm_chain(m_test)
    by_expanded = dE_dm_expanded(m_test)
    by_finite_difference = dE_dm_numeric(m_test)
    
    print(f"by chain rule:        {by_chain:.1f}")
    print(f"by expanding first:   {by_expanded:.1f}")
    print(f"by finite difference: {by_finite_difference:.6f}")
    
    assert np.isclose(by_chain, -28.0), f"Expected -28.0, got {by_chain}"
    assert np.isclose(by_expanded, -28.0), f"Expected -28.0, got {by_expanded}"
    assert np.isclose(by_chain, by_finite_difference, atol=1e-5), "Finite difference mismatch"
    print("✅ VERIFIED OUTPUT MATCH: -28.0 three ways")
