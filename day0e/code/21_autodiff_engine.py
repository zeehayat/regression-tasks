"""
Segment 21: Autodiff Engine From Scratch (Value Class with backward())
Supports +, *, **, exp, log operations with reverse-mode automatic differentiation.
"""
import numpy as np

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-1.0 * other)

    def exp(self):
        x = self.data
        out = Value(np.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        x = self.data
        out = Value(np.log(x), (self,), 'log')

        def _backward():
            self.grad += (1.0 / x) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # Topological sort of all nodes in graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()

if __name__ == "__main__":
    # Solve toy MHP loss using Autodiff Engine: E = (y - (m*x + c))^2
    x_v = Value(2.0)
    y_v = Value(10.0)
    m_v = Value(1.0)
    c_v = Value(1.0)

    pred = m_v * x_v + c_v
    error = y_v - pred
    loss = error ** 2

    loss.backward()

    print(f"Autodiff Computed Loss: {loss.data:.4f}")
    print(f"Autodiff Grad dL/dm: {m_v.grad:.4f} (Expected -28.0)")
    print(f"Autodiff Grad dL/dc: {c_v.grad:.4f} (Expected -14.0)")

    assert np.isclose(m_v.grad, -28.0), f"Expected -28.0, got {m_v.grad}"
    assert np.isclose(c_v.grad, -14.0), f"Expected -14.0, got {c_v.grad}"
    print("✅ ASSERTION PASSED: Autodiff Engine matches analytical gradients exactly.")
