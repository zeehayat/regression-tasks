"""Book check: a stable rate shrinks error; rate 5.0 diverges."""

def error(m, c):
    return (10 - (m * 2 + c)) ** 2

def partial_m(m, c, h=1e-6):
    return (error(m + h, c) - error(m, c)) / h

def partial_c(m, c, h=1e-6):
    return (error(m, c + h) - error(m, c)) / h


def run(rate, steps):
    m, c = 0.0, 0.0
    for _ in range(steps):
        m -= rate * partial_m(m, c)
        c -= rate * partial_c(m, c)
    return error(m, c)


if __name__ == "__main__":
    stable = run(0.01, 50)
    unstable = run(5.0, 10)
    print(stable, unstable)
    assert stable < error(0.0, 0.0)
    assert unstable > error(0.0, 0.0)
