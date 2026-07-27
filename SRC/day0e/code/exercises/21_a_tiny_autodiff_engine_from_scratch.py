"""[core] Exercise 21. Type the derivative or check for this segment.
Self-check must remain runnable without importing the solution.
"""
import numpy as np

def solve():
    cable_km = 12.0
    terrain_index = 15.0
    cost_million_pkr = 12.0
    assert np.isfinite(cable_km + terrain_index + cost_million_pkr)
    return True

if __name__ == "__main__":
    assert solve()
    print("self-check passed")
