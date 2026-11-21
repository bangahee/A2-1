"""Reproducible evidence for numerical accuracy and learning-rate experiments."""

import numpy as np
import matplotlib.pyplot as plt
from src.calculus import central_difference
from src.optimizer import VanillaGD, optimize, quadratic, quadratic_gradient


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


TESTED_RATES = [.5, .75, 1., 1.1, 1.2, 1.5]


def _divergence_runs(y_weight, initial, steps):
    """Run Vanilla GD for every tested lr and verify the closed-form path.

    For f(x,y)=x²+w·y² the update is separable:
        x_{t+1}=(1-2·lr)x_t,   y_{t+1}=(1-2w·lr)y_t
    so θ_t = diag((1-2lr)^t, (1-2w·lr)^t) θ_0 and GD diverges exactly when
    max(|1-2lr|, |1-2w·lr|) > 1.
    """
    paths, runs = {}, []
    factors_of = lambda lr: np.array([1 - 2*lr, 1 - 2*y_weight*lr])
    for lr in TESTED_RATES:
        path = optimize(lambda p: quadratic_gradient(p, y_weight), initial,
                        VanillaGD(lr), steps)
        factors = factors_of(lr)
        exact = initial[None, :] * factors[None, :] ** np.arange(steps + 1)[:, None]
        np.testing.assert_allclose(path, exact, rtol=1e-12, atol=1e-12)
        spectral = float(np.max(np.abs(factors)))
        behavior = "divergent" if spectral > 1 else (
            "neutral oscillation" if spectral == 1 else "convergent")
        norms = np.linalg.norm(path, axis=1)
        objective = quadratic(path, y_weight)
        paths[lr] = path
        runs.append(dict(lr=lr, update_factors=factors.tolist(),
                         max_abs_update_factor=spectral,
                         expected_behavior=behavior,
                         final_point=path[-1].tolist(),
                         final_radius=float(norms[-1]),
                         radius_growth_factor=float(norms[-1] / norms[0]),
                         final_objective=float(objective[-1]),
                         every_step_grows=bool(np.all(np.diff(norms) > 0)),
                         objective_every_step_grows=bool(np.all(np.diff(objective) > 0)),
                         closed_form_verified=True))
    return runs, paths


def divergence_study(steps=30):
    """Show with graphs that Vanilla GD paths diverge for lr ≥ 0.5.

    Two test functions from the assignment are used with initial point (5,5):

    * Ellipse f=x²+10y² (stiff y-direction): the y factor 1-20·lr satisfies
      |1-20·lr| ≥ 9 for every lr ≥ 0.5, so **every** lr ≥ 0.5 diverges.
      Its exact divergence threshold is lr > 0.1.
    * Circle f=x²+y²: the factor is 1-2·lr, so lr=0.5 lands on the origin in
      one step and lr=1 oscillates; divergence occurs for lr > 1
      (shown with lr=1.1, 1.2, 1.5, all inside the lr ≥ 0.5 range).
    """
    initial = np.array([5., 5.])
    ellipse_runs, ellipse_paths = _divergence_runs(10., initial, steps)
    circle_runs, circle_paths = _divergence_runs(1., initial, steps)
    ellipse_all = all(r["expected_behavior"] == "divergent" and r["every_step_grows"]
                      and r["objective_every_step_grows"] for r in ellipse_runs)
    circle_selected = [1.1, 1.2, 1.5]
    return dict(
        requirement="lr=0.5 이상에서 최적화 경로가 발산함을 그래프로 확인한다",
        conclusion=("Verified: on f=x²+10y² every tested lr ≥ 0.5 diverges "
                    "(|1-20·lr| ≥ 9); on f=x²+y² lr=1.1, 1.2, 1.5 diverge "
                    "and the exact boundary is lr > 1."),
        initial=initial.tolist(), seed=42, steps=steps, tested_rates=TESTED_RATES,
        figure="outputs/lr_ge_0_5_divergence.png",
        ellipse=dict(function="f(x,y)=x²+10y²", gradient="[2x,20y]",
                     update_rule="x←(1-2lr)x, y←(1-20lr)y",
                     exact_divergence_condition="lr > 0.1",
                     all_tested_rates_ge_0_5_diverge=ellipse_all,
                     runs=ellipse_runs),
        circle=dict(function="f(x,y)=x²+y²", gradient="[2x,2y]",
                    update_rule="θ←(1-2lr)θ",
                    exact_divergence_condition="lr > 1",
                    demonstrated_divergent_rates=circle_selected,
                    runs=circle_runs),
    ), dict(ellipse=ellipse_paths, circle=circle_paths)


def _symlog_grid(limit, linthresh):
    """Return coordinates dense near zero and log-spaced far away."""
    far = np.logspace(np.log10(linthresh), np.log10(limit), 120)
    near = np.linspace(-linthresh, linthresh, 81)
    return np.unique(np.concatenate([-far[::-1], near, far]))


def _plot_path_panel(ax, paths, rates, y_weight, steps_shown, title):
    """Draw dashed GD paths over contours on symmetric-log axes."""
    shown = {lr: paths[lr][:steps_shown + 1] for lr in rates}
    limit = max(np.abs(p).max() for p in shown.values()) * 1.5
    grid = _symlog_grid(limit, 10)
    gx, gy = np.meshgrid(grid, grid)
    levels = np.logspace(0, np.log10(quadratic(np.array([limit, limit]), y_weight)), 12)
    ax.contour(gx, gy, quadratic(np.stack([gx, gy], -1), y_weight), levels=levels,
               norm="log", cmap="Greys", linewidths=.7, alpha=.6)
    for lr, path in shown.items():
        ax.plot(path[:, 0], path[:, 1], "--o", markersize=4, label=f"lr={lr}")
    ax.plot(5, 5, "k*", markersize=12, label="start (5,5)")
    ax.plot(0, 0, "g+", markersize=12, mew=2, label="minimum")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_yscale("symlog", linthresh=10)
    ax.set(xlabel="x (symlog)", ylabel="y (symlog)", title=title)


def plot_divergence_evidence(evidence, paths):
    """Plot divergent GD paths for lr ≥ 0.5 on both assignment test functions.

    Row 1 (ellipse x²+10y²): every lr ≥ 0.5 diverges.
    Row 2 (circle x²+y²): lr=1.1,1.2,1.5 diverge; lr=0.5/0.75/1.0 shown as
    reference so the exact lr > 1 boundary is visible.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10.5), constrained_layout=True)
    rates = evidence["tested_rates"]
    ell, cir = paths["ellipse"], paths["circle"]
    _plot_path_panel(axes[0, 0], ell, [.5, .75, 1., 1.5], 10, 4,
                     "Ellipse x²+10y²: first 4 GD steps (dashed paths)")
    for lr in rates:
        axes[0, 1].semilogy(np.linalg.norm(ell[lr], axis=1), "--o", markersize=2,
                            label=f"lr={lr} (|1-20lr|={abs(1 - 20*lr):g})")
        axes[0, 2].semilogy(quadratic(ell[lr], 10), label=f"lr={lr}")
    axes[0, 1].set(xlabel="Iteration", ylabel="Distance to origin (log scale)",
                   title="Ellipse: distance grows at every step for lr ≥ 0.5")
    axes[0, 2].set(xlabel="Iteration", ylabel="f(x,y) (log scale)",
                   title="Ellipse: objective grows for EVERY lr ≥ 0.5")

    selected = evidence["circle"]["demonstrated_divergent_rates"]
    _plot_path_panel(axes[1, 0], cir, selected, 1, 8,
                     "Circle x²+y²: first 8 GD steps (dashed paths)")
    for lr in rates:
        style = "-" if lr in selected else ":"
        axes[1, 1].plot(cir[lr][:, 0], style, marker="o", markersize=2, label=f"lr={lr}")
        axes[1, 2].semilogy(np.maximum(quadratic(cir[lr], 1), 1e-30), style,
                            label=f"lr={lr}")
    axes[1, 1].set_yscale("symlog", linthresh=10)
    axes[1, 1].set(xlabel="Iteration", ylabel="x coordinate (symlog)",
                   title="Circle: |1-2·lr| > 1 only when lr > 1")
    axes[1, 2].set(xlabel="Iteration", ylabel="f(x,y) (log scale, floor 1e-30)",
                   title="Circle: lr=1.1/1.2/1.5 diverge; 0.5/0.75/1.0 do not")
    for ax in axes.ravel():
        ax.legend(fontsize=8)
        ax.grid(alpha=.25)
    fig.suptitle("Vanilla GD diverges for lr ≥ 0.5 — start (5,5), 30 steps, seed 42\n"
                 "Ellipse f=x²+10y²: all lr ≥ 0.5 diverge (threshold lr > 0.1)   |   "
                 "Circle f=x²+y²: threshold lr > 1", fontsize=13)
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
