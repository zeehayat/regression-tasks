# mhp_estimator.py
"""
MHP Cost Estimator Application.
Builds a regression modeling framework from scratch for Khyber Pakhtunkhwa MHP projects.
"""
import numpy as np
import warnings

class MHPCostEstimator:
    """
    Continuous application to predict and analyze MHP project cost overruns.
    """
    def __init__(self):
        # Parameters (learned coefficients) - initialized to None
        self.parameters_ = None
        
    def _validate_inputs(self, X, y):
        """
        Ensures X and y have compatible dimensions.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"Dim mismatch! X has {X.shape[0]} rows, y has {y.shape[0]} samples.")
        return X, y

    def _add_intercept_column(self, X):
        """
        Prepends a column of ones to the design matrix X.
        """
        ones = np.ones((X.shape[0], 1))
        return np.hstack([ones, X])

    def compute_ssr(self, X, y, beta):
        """
        Calculates the Sum of Squared Residuals for a given parameter set beta.
        """
        X_design = self._add_intercept_column(X)
        y_hat = X_design @ beta
        residuals = y - y_hat
        return np.dot(residuals, residuals)

    def _check_condition_number(self, X):
        """
        Validates condition number to check for multicollinearity issues.
        """
        cond = np.linalg.cond(X)
        if cond > 1000:
            warnings.warn(f"Warning: High condition number detected ({cond:.2f}). Multicollinearity may destabilize OLS parameters.")

    def fit_ols(self, X, y):
        """
        Fits OLS parameters using closed-form matrix inversion: beta = (X^T X)^-1 X^T y
        """
        X_clean, y_clean = self._validate_inputs(X, y)
        self._check_condition_number(X_clean)
        X_design = self._add_intercept_column(X_clean)
        self.parameters_ = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y_clean
        return self

    def calculate_log_likelihood(self, X, y):
        """
        Computes the log-likelihood under the normal noise assumption.
        """
        if self.parameters_ is None:
            raise ValueError("Model must be fitted before computing log-likelihood.")
        X_clean, y_clean = self._validate_inputs(X, y)
        n = len(y_clean)
        ssr = self.compute_ssr(X_clean, y_clean, self.parameters_)
        sigma2 = ssr / n
        log_lik = -n / 2 * np.log(2 * np.pi * sigma2) - ssr / (2 * sigma2)
        return log_lik

    def score(self, X, y, loss_type='mse', delta=1.5):
        """
        Scores the model predictions using the specified loss metric.
        """
        if self.parameters_ is None:
            raise ValueError("Model must be fitted before scoring.")
        X_clean, y_clean = self._validate_inputs(X, y)
        X_design = self._add_intercept_column(X_clean)
        y_hat = X_design @ self.parameters_
        residuals = y_clean - y_hat
        
        if loss_type == 'mse':
            return np.mean(residuals ** 2)
        elif loss_type == 'mae':
            return np.mean(np.abs(residuals))
        elif loss_type == 'huber':
            abs_res = np.abs(residuals)
            huber_vals = np.where(abs_res <= delta, 0.5 * (residuals ** 2), delta * (abs_res - 0.5 * delta))
            return np.mean(huber_vals)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def fit_gradient_descent(self, X, y, learning_rate=0.01, epochs=1000):
        """
        Fits parameters numerically using Batch Gradient Descent.
        """
        X_clean, y_clean = self._validate_inputs(X, y)
        X_design = self._add_intercept_column(X_clean)
        n, p = X_design.shape
        
        # Initialize coefficients to zero
        self.parameters_ = np.zeros(p)
        
        for _ in range(epochs):
            y_hat = X_design @ self.parameters_
            gradient = (1.0 / n) * X_design.T @ (y_hat - y_clean)
            self.parameters_ -= learning_rate * gradient
            
        return self
