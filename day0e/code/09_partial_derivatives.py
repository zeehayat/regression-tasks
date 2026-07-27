"""
Segment 9: Partial Derivatives & Microhydro Power (MHP) Error Surface
Computes symbolic partial derivative of Error E(m, c) = (y - (m*x + c))^2 with respect to m
Asserts symbolic_gradient_m(2.0, 10.0, 1.0, 1.0) == -28.0
"""
import numpy as np

def mhp_error(x, y, m, c):
    """
    Microhydro Power (MHP) Error:
    x = Cable length in km
    y = Cost in Million PKR
    m = Slope (Cost per km)
    c = Fixed Base Intercept
    """
    y_hat = m * x + c
    return (y - y_hat)**2

def symbolic_gradient_m(x, y, m, c):
    # E(m, c) = (y - (m*x + c))^2
    # dE/dm = 2 * (y - (m*x + c)) * (-x) = -2 * x * (y - m*x - c)
    return -2.0 * x * (y - m * x - c)

def symbolic_gradient_c(x, y, m, c):
    # dE/dc = 2 * (y - (m*x + c)) * (-1) = -2 * (y - m*x - c)
    return -2.0 * (y - m * x - c)

if __name__ == "__main__":
    x_val, y_val = 2.0, 10.0
    m_val, c_val = 1.0, 1.0
    
    grad_m = symbolic_gradient_m(x_val, y_val, m_val, c_val)
    grad_c = symbolic_gradient_c(x_val, y_val, m_val, c_val)
    
    print(f"MHP Dataset Point: Cable x={x_val} km, Cost y={y_val} Million PKR")
    print(f"At m={m_val}, c={c_val}: dE/dm = {grad_m:.2f}, dE/dc = {grad_c:.2f}")
    
    assert np.isclose(grad_m, -28.0), f"Expected -28.0, got {grad_m}"
    print("✅ ASSERTION PASSED: symbolic_gradient_m(2, 10, 1, 1) == -28.0")
