"""Reproducible evidence for numerical accuracy and learning-rate experiments."""

import numpy as np
import matplotlib.pyplot as plt
from src.calculus import central_difference
from src.optimizer import VanillaGD, optimize, quadratic_gradient


def derivative_step_study():
    """Compare h sensitivity for an exact quadratic difference and sin(x).

    Central differences have zero truncation error on x² in exact arithmetic.
    sin(0.7) reveals the O(h²) truncation regime before roundoff dominates.
    """
    rows = []
    for h in np.logspace(-1, -12, 12):
        square = central_difference(lambda x: x*x, 3.0, h)
        sine = central_difference(np.sin, .7, h)
        rows.append(dict(h=float(h), square_derivative=square,
                         square_error=abs(square - 6.0),
                         sine_derivative=sine,
                         sine_error=abs(sine - float(np.cos(.7)))))
    return rows


def plot_derivative_study(rows):
    """Plot truncation/roundoff tradeoffs without implying O(h²) for x²."""
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    h = [row["h"] for row in rows]
    ax.loglog(h, np.maximum([r["square_error"] for r in rows], 1e-18),
              "o-", label="x² at x=3: zero truncation error")
    ax.loglog(h, np.maximum([r["sine_error"] for r in rows], 1e-18),
              "o-", label="sin(x) at x=0.7: O(h²) truncation")
    ax.axhline(1e-4, linestyle="--", color="gray", label="Required x² error bound")
    ax.set(xlabel="Central-difference step h", ylabel="Absolute error",
           title="Central differences: step-size sensitivity")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    return fig


def divergence_study(steps=30):
    """Verify divergent examples chosen within the lr≥0.5 interval.

    This is an existential demonstration, not the false universal assertion
    that all lr≥0.5 diverge for x²+y². The exact boundary is lr>1.
    """
    initial = np.array([5., 5.])
    rates = [.5, .75, 1., 1.1, 1.2, 1.5]
    paths, runs = {}, []
    for lr in rates:
        path = optimize(quadratic_gradient, initial, VanillaGD(lr), steps)
        factor = 1 - 2*lr
        exact = initial[None, :] * factor ** np.arange(steps + 1)[:, None]
        np.testing.assert_allclose(path, exact, rtol=1e-12, atol=1e-12)
        behavior = "divergent" if abs(factor) > 1 else (
            "neutral oscillation" if abs(factor) == 1 else "convergent")
        paths[lr] = path
        norms = np.linalg.norm(path, axis=1)
        runs.append(dict(lr=lr, update_factor=factor,
                         expected_behavior=behavior, final_radius=float(norms[-1]),
                         radius_growth_factor=float(norms[-1]/norms[0]),
                         every_step_grows=bool(np.all(np.diff(norms) > 0)),
                         closed_form_verified=True))
    return dict(function="f(x,y)=x²+y²", gradient="[2x,2y]",
                initial=initial.tolist(), seed=42, steps=steps,
                tested_interval="lr >= 0.5",
                demonstrated_divergent_rates=[1.1, 1.2, 1.5],
                exact_divergence_condition="lr > 1",
                scope="Divergent examples within the stated interval; not all rates in that range diverge.",
                figure="outputs/lr_ge_0_5_divergence.png", runs=runs), paths


def plot_divergence_evidence(evidence, paths):
    """Show growing oscillations and objective growth for all three examples."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for lr in evidence["demonstrated_divergent_rates"]:
        path = paths[lr]
        axes[0].plot(path[:, 0], "--o", markersize=2, label=f"lr={lr}")
        axes[1].semilogy(np.sum(path**2, axis=1), label=f"lr={lr}")
    axes[0].set_yscale("symlog", linthresh=10)
    axes[0].set(xlabel="Iteration", ylabel="x coordinate (symmetric log)",
                title="Growing oscillations from (5,5)")
    axes[1].set(xlabel="Iteration", ylabel="f(x,y)=x²+y² (log scale)",
                title="Objective increases at every step")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.25)
    fig.suptitle("Divergence examples in the lr >= 0.5 interval\n"
                 "Selected rates: 1.1, 1.2, 1.5; exact boundary for this function: lr > 1",
                 fontsize=11)
    return fig


def optimizer_conditions():
    """List every optimizer comparison's initial state and hyperparameters."""
    rows = []
    for experiment, weight, steps, rates in [
        ("circle", 1, 100, [("VanillaGD", .1), ("Momentum", .1)]),
        ("ellipse", 10, 200, [("VanillaGD", .09), ("Momentum", .09)]),
        ("bonus_optimizers", 10, 300,
         [("VanillaGD", .01), ("Momentum", .01), ("Adam", .1)]),
    ]:
        for name, lr in rates:
            rows.append(dict(experiment=experiment, function=f"x²+{weight}y²",
                             optimizer=name, initial=[5, 5], seed=42, steps=steps,
                             lr=lr, beta=.9 if name == "Momentum" else None,
                             beta1=.9 if name == "Adam" else None,
                             beta2=.999 if name == "Adam" else None,
                             eps=1e-8 if name == "Adam" else None))
    rows.append(dict(experiment="newton", function="x²+10y²",
                     optimizer="Newton", initial=[5, 5], seed=42, steps=3,
                     lr=None, beta=None, damping=1.0, hessian="diag(2,20)"))
    return rows
