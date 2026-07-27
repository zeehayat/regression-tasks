# Video 7 — Second derivatives and curvature

The second derivative is the derivative of the derivative. It tells us how the slope itself is changing: this is curvature.

For f(x)=x², f′(x)=2x and f′′(x)=2, so the curve bends upward everywhere. A positive second derivative indicates local bowl-like curvature; a negative one indicates a cap-like shape.

Near a minimum, curvature helps determine how aggressive a step can be. Newton-style methods use both gradient and curvature, while ordinary gradient descent uses only the gradient.

In multiple dimensions, the matrix of second partial derivatives is the Hessian. You only need the core idea now: first derivatives give direction; second derivatives describe the shape around you.
