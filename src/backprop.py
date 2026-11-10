"""Explicit forward/backward propagation for a 2 → 2 → 1 sigmoid network."""

import numpy as np
from .probability import sigmoid


def fixed_example():
    """Return the assignment's fixed inputs; the missing target is set to 1."""
    return dict(x=np.array([1.0, 0.0]),
                W1=np.array([[0.1, 0.2], [0.3, 0.4]]),
                b1=np.zeros(2), W2=np.array([0.5, 0.6]),
                b2=np.array(0.0), y_true=1.0)


def forward(x, W1, b1, W2, b2, y_true):
    """Compute z1=W1@x+b1, a1=σ(z1), z2=W2@a1+b2, p=σ(z2).

    BCE uses logaddexp(0,z2)-y*z2, algebraically identical to the
    probability expression but stable for extreme logits.
    """
    x, W1, b1, W2 = [np.asarray(a, float) for a in (x, W1, b1, W2)]
    if x.shape != (2,) or W1.shape != (2, 2) or b1.shape != (2,) or W2.shape != (2,):
        raise ValueError("Expected x(2,), W1(2,2), b1(2,), W2(2,).")
    if np.asarray(b2).shape != () or not 0 <= y_true <= 1:
        raise ValueError("Expected scalar b2 and target in [0,1].")
    z1 = W1 @ x + b1
    a1 = sigmoid(z1)
    z2 = float(W2 @ a1 + b2)
    prediction = float(sigmoid(z2))
    loss = float(np.logaddexp(0.0, z2) - y_true * z2)
    return dict(z1=z1, a1=a1, z2=z2, y_pred=prediction, loss=loss)


def backward(x, W1, b1, W2, b2, y_true):
    """Apply the chain rule explicitly, including both bias gradients."""
    cache = forward(x, W1, b1, W2, b2, y_true)
    p, a1 = cache["y_pred"], cache["a1"]
    # dL/dp = -y/p + (1-y)/(1-p), for an interior probability.
    d_prediction = (-y_true / p + (1 - y_true) / (1 - p)) if 0 < p < 1 else float("nan")
    d_z2 = p - y_true  # Sigmoid + BCE cancels p(1-p).
    d_a1 = d_z2 * np.asarray(W2)
    d_z1 = d_a1 * a1 * (1 - a1)
    return dict(d_y_pred=d_prediction, d_z2=d_z2, W2=d_z2 * a1,
                b2=np.array(d_z2), d_a1=d_a1, d_z1=d_z1,
                W1=np.outer(d_z1, x), b1=d_z1)
