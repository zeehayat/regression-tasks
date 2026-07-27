# Video 6 — Gradient descent

Gradient descent updates parameters in the negative-gradient direction. The basic rule is w_new = w_old − η∇L(w), where η is the learning rate.

If the gradient is positive, subtracting it moves the parameter down. If the gradient is negative, subtracting it moves the parameter up. The learning rate controls the step size.

Too small means slow learning. Too large can overshoot or diverge. The gradient is local, so descent is an iterative process: recompute the gradient after each move.

For L(w)=(w−3)² starting at w=0, the gradient is −6. With learning rate 0.1, the next value is 0−0.1(−6)=0.6.
