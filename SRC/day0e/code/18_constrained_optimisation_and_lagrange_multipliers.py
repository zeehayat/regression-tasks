"""Day 0E demo 18: Constrained optimisation and Lagrange multipliers."""
import numpy as np

def demo():
    cable_km = np.array([12.0, 30.0, 5.0, 40.0, 15.0])
    terrain_index = np.array([15.0, 25.0, 8.0, 45.0, 12.0])
    costs_million_pkr = np.array([12.0, 30.0, 8.0, 45.0, 15.0])
    beta = np.array([0.1, 0.2])
    prediction = beta[0] * cable_km[0] + beta[1] * terrain_index[0]
    assert np.isfinite(prediction)
    return prediction

if __name__ == "__main__":
    print(demo())
