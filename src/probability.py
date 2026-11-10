"""Probability distributions, stable activations, and information measures."""

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(value):
    """Compute sigmoid with sign-specific formulas to avoid overflow."""
    z = np.asarray(value, float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1 / (1 + np.exp(-z[positive]))
    ez = np.exp(z[~positive])
    out[~positive] = ez / (1 + ez)
    return out


def softmax(logits, axis=-1):
    """Normalize exponentials after subtracting their maximum for stability."""
    z = np.asarray(logits, float)
    if not z.size or not np.isfinite(z).all():
        raise ValueError("logits must be nonempty and finite.")
    shifted = z - np.max(z, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def normal_pdf(x, mean=0.0, variance=1.0):
    """Evaluate N(mean, variance); the second parameter is variance, not σ."""
    if variance <= 0:
        raise ValueError("variance must be positive.")
    return np.exp(-((np.asarray(x) - mean) ** 2) / (2 * variance)) / np.sqrt(2 * np.pi * variance)


def bernoulli_pmf(p):
    """Return [P(X=0),P(X=1)] = [1-p,p]."""
    if not 0 <= p <= 1:
        raise ValueError("p must be in [0,1].")
    return np.array([1 - p, p])


def probability_vector(values):
    """Validate a finite one-dimensional probability vector summing to one."""
    p = np.asarray(values, float)
    if p.ndim != 1 or not p.size or not np.isfinite(p).all() or (p < 0).any() or not np.isclose(p.sum(), 1, rtol=0, atol=1e-10):
        raise ValueError("Expected nonnegative probabilities summing to one.")
    return p


def entropy(p):
    """Return H(p) = -Σ p log p in nats with 0 log 0 = 0."""
    p = probability_vector(p)
    active = p > 0
    return float(-np.sum(p[active] * np.log(p[active])))


def cross_entropy(p, q):
    """Return H(p,q) = -Σ p log q; missing q support yields infinity."""
    p, q = probability_vector(p), probability_vector(q)
    if p.shape != q.shape:
        raise ValueError("p and q must have matching shapes.")
    active = p > 0
    if (q[active] == 0).any():
        return float("inf")
    return float(-np.sum(p[active] * np.log(q[active])))


def kl_divergence(p, q):
    """Return D_KL(p||q) = H(p,q)-H(p), using natural logarithms."""
    return cross_entropy(p, q) - entropy(p)


def plot_distributions():
    """Plot Normal PDFs at 600 points in [-4,5] and both Bernoulli PMFs.

    PMF bars have width .32 and offsets ±.17, separating the two distributions
    at each outcome without overlap. The figure is 11×4 inches, saved at 150 dpi.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    x = np.linspace(-4, 5, 600)
    for mean, variance in [(0, 1), (2, 0.5)]:
        axes[0].plot(x, normal_pdf(x, mean, variance),
                     label=f"N(μ={mean}, variance={variance})")
    axes[0].set(title="Normal probability densities", xlabel="x", ylabel="PDF")
    for p, offset in [(0.3, -0.17), (0.7, 0.17)]:
        axes[1].bar(np.array([0, 1]) + offset, bernoulli_pmf(p),
                    width=0.32, label=f"Bernoulli p={p}")
    axes[1].set(title="Bernoulli probability masses", xlabel="outcome", ylabel="PMF",
                xticks=[0, 1], ylim=(0, 1))
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    return fig
