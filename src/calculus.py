"""Central differences and geometric visualization of gradients."""

import numpy as np
import matplotlib.pyplot as plt


def central_difference(function, x, h=1e-5):
    """Estimate scalar f'(x) = [f(x+h)-f(x-h)]/(2h), with O(h²) error."""
    if h <= 0:
        raise ValueError("h must be positive.")
    return float((function(x + h) - function(x - h)) / (2 * h))


def numerical_gradient(function, point, h=1e-5):
    """Return central differences of a scalar function over any array shape."""
    if h <= 0:
        raise ValueError("h must be positive.")
    point = np.asarray(point, float)
    gradient = np.empty_like(point)
    for index in np.ndindex(point.shape):
        plus, minus = point.copy(), point.copy()
        plus[index] += h
        minus[index] -= h
        gradient[index] = (function(plus) - function(minus)) / (2 * h)
    return gradient


def plot_gradient():
    """Show ∇(x²+y²) normal to level sets, with equal geometric axis scales."""
    line = np.linspace(-3, 3, 160)
    xx, yy = np.meshgrid(line, line)
    fig, ax = plt.subplots(figsize=(6.5, 6), constrained_layout=True)
    contour = ax.contour(xx, yy, xx ** 2 + yy ** 2, levels=[0.5, 1, 2, 4, 6, 9, 12])
    ax.clabel(contour, fontsize=8)
    grid = np.linspace(-2, 2, 5)
    gx, gy = np.meshgrid(grid, grid)
    ax.quiver(gx, gy, 2 * gx, 2 * gy, color="#dc2626",
              angles="xy", scale_units="xy", scale=7, label="Gradient (2x, 2y)")
    ax.set(xlabel="x", ylabel="y", title="Gradient is perpendicular to contours")
    ax.set_aspect("equal")
    ax.legend()
    return fig
