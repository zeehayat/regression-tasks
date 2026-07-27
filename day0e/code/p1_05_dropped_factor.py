"""
p1_05_dropped_factor.py - Part 1 Segment 1.21
Demonstrates the breakage when the inner derivative (-x) is dropped from the chain rule.
Prints: 14.0 vs -28.0 [VERIFIED]
"""
import numpy as np

def dE_dm_broken(m=1.0, x=2.0, y=10.0, c=1.0):
    # BROKEN: Forgot inner derivative (-x), only used 2(y - mx - c)
    return 2.0 * (y - m * x - c)

def dE_dm_correct(m=1.0, x=2.0, y=10.0, c=1.0):
    return -2.0 * x * (y - m * x - c)

if __name__ == "__main__":
    broken_val = dE_dm_broken()
    correct_val = dE_dm_correct()
    
    print(f"dropped inner derivative: {broken_val:.1f}")
    print(f"correct:                 {correct_val:.1f}")
    
    assert np.isclose(broken_val, 14.0), f"Expected 14.0, got {broken_val}"
    assert np.isclose(correct_val, -28.0), f"Expected -28.0, got {correct_val}"
    print("✅ VERIFIED OUTPUT MATCH: dropped inner derivative=14.0, correct=-28.0")
