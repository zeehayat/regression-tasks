"""
Gradient Checking Utility (Segment 19 - Reusable Tool)
Computes numerical gradient using central finite differences and verifies against analytical gradient.
Relative Error Formula: ||g_analytic - g_numeric|| / (||g_analytic|| + ||g_numeric||)
Tolerance: < 1e-7 is PASS, > 1e-4 is FAIL.
"""

import numpy as np

def compute_numerical_gradient(f, x, h=1e-6):
    """
    Computes numerical gradient of scalar function f at vector x using central finite differences.
    """
    grad = np.zeros_like(x, dtype=np.float64)
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        grad[i] = (f(x_plus) - f(x_minus)) / (2 * h)
    return grad

def gradient_check(f, grad_analytic_fn, x, h=1e-6, tol=1e-7):
    """
    Asserts that analytical gradient matches central finite difference numerical gradient.
    """
    g_num = compute_numerical_gradient(f, x, h)
    g_ana = grad_analytic_fn(x)
    
    diff = np.linalg.norm(g_ana - g_num)
    denom = np.linalg.norm(g_ana) + np.linalg.norm(g_num)
    if denom == 0:
        rel_error = 0.0
    else:
        rel_error = diff / denom
        
    return rel_error, rel_error < tol

if __name__ == "__main__":
    # Test on MHP cost loss function L(m, c) = (y - (m*x + c))^2
    x_val, y_val = 2.0, 10.0
    def loss_fn(params):
        m, c = params[0], params[1]
        return (y_val - (m * x_val + c))**2

    def grad_fn(params):
        m, c = params[0], params[1]
        pred = m * x_val + c
        err = pred - y_val # (m*x + c - y)
        # dL/dm = 2*(m*x+c-y)*x, dL/dc = 2*(m*x+c-y)
        return np.array([2 * err * x_val, 2 * err], dtype=np.float64)

    params_test = np.array([1.0, 1.0])
    rel_err, is_pass = gradient_check(loss_fn, grad_fn, params_test)
    print(f"Gradient Check Relative Error: {rel_err:.9e} | Pass: {is_pass}")
    assert is_pass, f"Gradient check failed with relative error {rel_err}"
