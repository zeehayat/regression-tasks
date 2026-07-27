# Video 5 — Partial derivatives and gradients

A model has many parameters, so the loss depends on many inputs. A partial derivative changes one variable while holding the others fixed.

For L(w1,w2)=w1²+3w2², the partial derivatives are 2w1 and 6w2. Put them into a vector: the gradient ∇L = [2w1, 6w2].

The gradient points in the direction of steepest increase. Therefore the negative gradient points toward the steepest local decrease. Gradient descent uses that direction to reduce loss.

The gradient is not a single number; it is one sensitivity per parameter.
