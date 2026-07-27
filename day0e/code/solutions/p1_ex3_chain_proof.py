"""
SOLUTION FOR EXERCISE 3 (Segment 1.22)
"""
import numpy as np

def num_slope(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)

def deriv_1(x): return 12.0 * ((3.0 * x + 1.0)**3)
def deriv_2(x): return -6.0 * ((5.0 - 2.0 * x)**2)
def deriv_3(x): return 4.0 * x * (x**2 + 1.0)
def deriv_4(x=4.0, y=20.0, c=2.0, m=3.0):
    # dE/dm = -2x(y - mx - c) = -2(4)(20 - 12 - 2) = -8 * 6 = -48
    return -2.0 * x * (y - m * x - c)

if __name__ == "__main__":
    f1 = lambda x: (3*x + 1)**4
    f2 = lambda x: (5 - 2*x)**3
    f3 = lambda x: (x**2 + 1)**2
    
    assert np.isclose(deriv_1(1.0), num_slope(f1, 1.0))
    assert np.isclose(deriv_2(1.0), num_slope(f2, 1.0))
    assert np.isclose(deriv_3(2.0), num_slope(f3, 2.0))
    assert deriv_4() == -48.0
    
    print("1. (3x+1)^4 at x=1 ->", deriv_1(1.0), "(num:", num_slope(f1, 1.0), ")")
    print("2. (5-2x)^3 at x=1 ->", deriv_2(1.0), "(num:", num_slope(f2, 1.0), ")")
    print("3. (x^2+1)^2 at x=2 ->", deriv_3(2.0), "(num:", num_slope(f3, 2.0), ")")
    print("4. dE/dm at x=4, y=20, c=2, m=3 ->", deriv_4())
    print("✅ Solution 3 Assertions Passed!")
