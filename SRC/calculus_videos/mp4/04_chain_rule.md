# Video 4 — The chain rule

Machine-learning models are compositions: one function feeds another. The chain rule tells us how a small change travels through the chain.

If y=f(u) and u=g(x), then dy/dx = (dy/du)(du/dx). Read it as: sensitivity of the final output to the intermediate quantity, multiplied by sensitivity of the intermediate quantity to the original input.

For y=(3x+1)², let u=3x+1. Then dy/du=2u and du/dx=3, so dy/dx=6(3x+1).

Backpropagation is the chain rule applied repeatedly from the output of a neural network back toward its parameters.
