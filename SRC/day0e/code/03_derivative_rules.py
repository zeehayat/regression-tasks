"""Book check: d/dx (3x² + 5x) at x=2 is 17.0."""

def dy_dx_by_rule(x):
    return 6 * x + 5


if __name__ == "__main__":
    print(dy_dx_by_rule(2.0))
    assert dy_dx_by_rule(2.0) == 17.0
