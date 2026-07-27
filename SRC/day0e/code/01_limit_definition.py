"""Book check: forward finite difference approaches the derivative 6.0."""

def numerical_slope(f, x, h):
    return (f(x + h) - f(x)) / h


if __name__ == "__main__":
    f = lambda x: x ** 2
    estimates = [numerical_slope(f, 3.0, h) for h in [1.0, 0.1, 0.01, 0.0001, 0.000001]]
    print(estimates[-1])
    assert abs(estimates[-1] - 6.0) < 1e-4
