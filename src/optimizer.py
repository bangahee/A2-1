"""First/second-order optimization implemented using NumPy only."""

import numpy as np
import matplotlib.pyplot as plt


class VanillaGD:
    """Gradient descent: θ_(t+1) = θ_t - lr * ∇f(θ_t)."""

    def __init__(self, lr=0.1):
        """Configure a positive learning rate."""
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be finite and positive.")
        self.lr = float(lr)

    def step(self, parameters, gradient):
        """Return the next parameters without mutating the input."""
        return np.asarray(parameters, float) - self.lr * np.asarray(gradient, float)


class Momentum(VanillaGD):
    """Heavy-ball momentum: v_t=βv_(t-1)+g_t; θ_(t+1)=θ_t-lr*v_t."""

    def __init__(self, lr=0.1, beta=0.9):
        """Configure learning rate and 0≤beta<1; initialize velocity lazily."""
        super().__init__(lr)
        if not 0 <= beta < 1:
            raise ValueError("beta must be in [0,1).")
        self.beta, self.velocity = float(beta), None

    def step(self, parameters, gradient):
        """Accumulate velocity and return the heavy-ball update."""
        g = np.asarray(gradient, float)
        if self.velocity is None:
            self.velocity = np.zeros_like(g)
        if self.velocity.shape != g.shape:
            raise ValueError("Create a new optimizer for a different shape.")
        self.velocity = self.beta * self.velocity + g
        return np.asarray(parameters, float) - self.lr * self.velocity


class Adam(VanillaGD):
    """Adam with exponential moments and bias correction."""

    def __init__(self, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
        """Configure moment decay, positive epsilon, and zero initial state."""
        super().__init__(lr)
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1 or not np.isfinite(eps) or eps <= 0:
            raise ValueError("Use beta values in [0,1) and positive epsilon.")
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m, self.v, self.t = None, None, 0

    def step(self, parameters, gradient):
        """Update corrected moments m_hat/v_hat and return Adam's step."""
        g = np.asarray(gradient, float)
        if self.m is None:
            self.m, self.v = np.zeros_like(g), np.zeros_like(g)
        if self.m.shape != g.shape:
            raise ValueError("Create a new optimizer for a different shape.")
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * g
        self.v = self.beta2 * self.v + (1 - self.beta2) * g ** 2
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return np.asarray(parameters, float) - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def quadratic(point, y_weight=1.0):
    """Evaluate f(x,y)=x²+y_weight*y² (also accepts a grid in the last axis)."""
    point = np.asarray(point)
    return point[..., 0] ** 2 + y_weight * point[..., 1] ** 2


def quadratic_gradient(point, y_weight=1.0):
    """Return the analytic gradient (2x,2*y_weight*y)."""
    return 2 * np.asarray(point, float) * np.array([1.0, y_weight])


def optimize(gradient, initial, optimizer, steps=100):
    """Return the full path including initial position and every update."""
    if not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a nonnegative integer.")
    point = np.asarray(initial, float).copy()
    if not np.isfinite(point).all():
        raise ValueError("initial must be finite.")
    path = [point.copy()]
    for _ in range(steps):
        g = np.asarray(gradient(point), float)
        if g.shape != point.shape:
            raise ValueError("Gradient must match parameter shape.")
        point = optimizer.step(point, g)
        if not np.isfinite(point).all():
            raise FloatingPointError("Optimizer produced non-finite parameters.")
        path.append(point.copy())
    return np.array(path)


def newton_method(gradient, hessian, initial, steps=5, damping=1.0):
    """Return Newton path using solve(H,gradient), without matrix inversion.

    Requires an invertible Hessian at each step. A fixed damping factor is
    provided; this educational implementation has no global line search.
    """
    if not 0 < damping <= 1 or not isinstance(steps, int) or steps < 0:
        raise ValueError("Use 0<damping≤1 and nonnegative integer steps.")
    point = np.asarray(initial, float).copy()
    path = [point.copy()]
    for _ in range(steps):
        point = point - damping * np.linalg.solve(hessian(point), gradient(point))
        path.append(point.copy())
    return np.array(path)


def first_hit(path, radius=0.1):
    """Return first iteration inside radius, or None if never reached."""
    hits = np.flatnonzero(np.linalg.norm(path, axis=1) <= radius)
    return int(hits[0]) if len(hits) else None


def plot_paths(paths, y_weight=1.0, title="Optimization paths"):
    """Overlay named trajectories as dashed lines on quadratic contours."""
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    bound = max(1.0, max(float(np.max(np.abs(p))) for p in paths.values())) * 1.12
    line = np.linspace(-bound, bound, 250)
    xx, yy = np.meshgrid(line, line)
    levels = np.geomspace(max(0.01, bound ** 2 / 5000), (1 + y_weight) * bound ** 2, 18)
    ax.contour(xx, yy, xx ** 2 + y_weight * yy ** 2, levels=levels, alpha=0.4)
    for name, path in paths.items():
        ax.plot(path[:, 0], path[:, 1], "--o", markersize=2.5, linewidth=1, label=name)
    ax.scatter(0, 0, marker="*", s=120, color="black", label="Minimum")
    ax.set(xlabel="x", ylabel="y", title=title)
    ax.set_aspect("equal")
    ax.legend(fontsize=8)
    return fig


def plot_convergence(paths, y_weight=1.0, title="Convergence comparison"):
    """Compare objective values by iteration, with a positive log-scale floor."""
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for name, path in paths.items():
        ax.semilogy(np.maximum(quadratic(path, y_weight), 1e-30), label=name)
    ax.set(xlabel="Iteration", ylabel="Objective (log scale)", title=title)
    ax.legend()
    ax.grid(alpha=0.25)
    return fig
