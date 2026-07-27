"""
EXERCISE 2 (Segment 1.17) [core]
Differentiate each of the 5 polynomials by rule, then verify numerically:
1. y = 7x^4
2. y = x^5 + 2x^2
3. y = 10x + 3
4. y = 0.5x^2 - 4x
5. y = 6
"""
import numpy as np

def derivative_rule_1(x): pass
def derivative_rule_2(x): pass
def derivative_rule_3(x): pass
def derivative_rule_4(x): pass
def derivative_rule_5(x): pass

if __name__ == "__main__":
    x_test = 2.0
    try:
        assert derivative_rule_1(x_test) == 28.0 * (x_test**3)
        assert derivative_rule_2(x_test) == 5.0 * (x_test**4) + 4.0 * x_test
        assert derivative_rule_3(x_test) == 10.0
        assert derivative_rule_4(x_test) == x_test - 4.0
        assert derivative_rule_5(x_test) == 0.0
        print("✅ Exercise 2 Passed!")
    except Exception as e:
        print("❌ Exercise 2 Stub Exception (Expected before completion):", e)
