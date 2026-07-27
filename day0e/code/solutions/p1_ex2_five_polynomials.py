"""
SOLUTION FOR EXERCISE 2 (Segment 1.17)
"""
import numpy as np

def num_slope(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)

def derivative_rule_1(x): return 28.0 * (x**3)
def derivative_rule_2(x): return 5.0 * (x**4) + 4.0 * x
def derivative_rule_3(x): return 10.0
def derivative_rule_4(x): return x - 4.0
def derivative_rule_5(x): return 0.0

if __name__ == "__main__":
    x_test = 2.0
    
    # Numerical checks
    f1 = lambda x: 7 * x**4
    f2 = lambda x: x**5 + 2 * x**2
    f3 = lambda x: 10 * x + 3
    f4 = lambda x: 0.5 * x**2 - 4 * x
    f5 = lambda x: 6.0
    
    assert np.isclose(derivative_rule_1(x_test), num_slope(f1, x_test))
    assert np.isclose(derivative_rule_2(x_test), num_slope(f2, x_test))
    assert np.isclose(derivative_rule_3(x_test), num_slope(f3, x_test))
    assert np.isclose(derivative_rule_4(x_test), num_slope(f4, x_test))
    assert np.isclose(derivative_rule_5(x_test), num_slope(f5, x_test))
    
    print("Outputs at x = 2.0:")
    print("1. 7x^4       -> rule:", derivative_rule_1(2.0), "num:", num_slope(f1, 2.0))
    print("2. x^5 + 2x^2 -> rule:", derivative_rule_2(2.0), "num:", num_slope(f2, 2.0))
    print("3. 10x + 3    -> rule:", derivative_rule_3(2.0), "num:", num_slope(f3, 2.0))
    print("4. 0.5x^2-4x  -> rule:", derivative_rule_4(2.0), "num:", num_slope(f4, 2.0))
    print("5. 6          -> rule:", derivative_rule_5(2.0), "num:", num_slope(f5, 2.0))
    print("✅ Solution 2 Assertions Passed!")
