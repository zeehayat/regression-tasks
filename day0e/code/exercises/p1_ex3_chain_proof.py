"""
EXERCISE 3 (Segment 1.22) [proof]
Differentiate each by chain rule, then verify numerically:
1. (3x + 1)^4
2. (5 - 2x)^3
3. (x^2 + 1)^2 (try chain rule & expanding first)
4. Given x=4, y=20, c=2, m=3: compute dE/dm for E = (y - mx - c)^2
"""
import numpy as np

def deriv_1(x): pass
def deriv_2(x): pass
def deriv_3(x): pass
def deriv_4(x=4.0, y=20.0, c=2.0, m=3.0): pass

if __name__ == "__main__":
    try:
        assert deriv_1(1.0) == 12 * (4.0**3) # 768.0
        assert deriv_2(1.0) == -6 * (3.0**2) # -54.0
        assert deriv_3(2.0) == 4 * 2.0 * (2.0**2 + 1) # 40.0
        assert deriv_4() == -48.0
        print("✅ Exercise 3 Passed!")
    except Exception as e:
        print("❌ Exercise 3 Stub Exception (Expected before completion):", e)
