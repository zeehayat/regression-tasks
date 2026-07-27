"""
Segment 14: Matrix Calculus & OLS Normal Equations Derivation (Chapter 1 Bridge)
Derives beta_hat = (X^T X)^-1 X^T y on Microhydro Power (MHP) dataset
"""
import numpy as np

def solve_ols_normal_equations(X, y):
    """
    Computes closed-form OLS solution: beta_hat = (X^T X)^-1 X^T y
    """
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    Xty = X.T @ y
    beta_hat = XtX_inv @ Xty
    return beta_hat

def ols_loss_gradient(beta, X, y):
    """
    Gradient of ||y - X*beta||^2 with respect to beta vector: -2 * X^T * (y - X*beta)
    """
    residuals = y - X @ beta
    return -2.0 * (X.T @ residuals)

if __name__ == "__main__":
    # Microhydro Power (MHP) Dataset framing:
    # x1 = cable length (km), x2 = terrain index (1-5)
    # y = project cost in Million PKR
    X_mhp = np.array([
        [1.0, 2.0, 1.5],
        [1.0, 4.0, 2.5],
        [1.0, 6.0, 3.8],
        [1.0, 8.0, 4.2],
        [1.0, 10.0, 5.0]
    ], dtype=np.float64) # First column is 1.0 intercept
    
    y_mhp = np.array([12.5, 18.0, 25.2, 31.0, 38.5], dtype=np.float64)
    
    beta_hat = solve_ols_normal_equations(X_mhp, y_mhp)
    print("MHP Dataset OLS Closed-Form Solution (beta_hat):")
    print(f"Intercept c: {beta_hat[0]:.4f}")
    print(f"Cable Slope m1: {beta_hat[1]:.4f}")
    print(f"Terrain Slope m2: {beta_hat[2]:.4f}")
    
    # Gradient at optimal beta_hat must be ~ 0
    grad_at_opt = ols_loss_gradient(beta_hat, X_mhp, y_mhp)
    grad_norm = np.linalg.norm(grad_at_opt)
    print(f"Gradient norm at optimal beta_hat: {grad_norm:.6e}")
    assert grad_norm < 1e-10, f"Expected gradient norm near 0, got {grad_norm}"
    print("✅ ASSERTION PASSED: Closed-form OLS solution minimizes loss gradient to zero.")
