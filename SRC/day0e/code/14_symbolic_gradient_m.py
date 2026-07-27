"""Book check: symbolic_gradient_m(2, 10, 1, 1) is -28.0."""

def symbolic_gradient_m(x, y, m, c):
    return -2 * x * (y - m * x - c)


if __name__ == "__main__":
    print(symbolic_gradient_m(2, 10, 1, 1))
    assert symbolic_gradient_m(2, 10, 1, 1) == -28.0
