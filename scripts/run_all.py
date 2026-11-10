"""Regenerate every PNG and machine-readable assignment acceptance result."""

import json
import os
from pathlib import Path
import platform
import struct

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.linear_algebra import (
    area_report, rotation, scaling, shear, plot_transformations,
    power_iteration, plot_svd,
)
from src.calculus import central_difference, numerical_gradient, plot_gradient
from src.backprop import fixed_example, forward, backward
from src.probability import softmax, plot_distributions, entropy, cross_entropy, kl_divergence
from src.optimizer import (
    VanillaGD, Momentum, Adam, optimize, quadratic_gradient,
    newton_method, first_hit, plot_paths, plot_convergence,
)
from scripts.experiment_evidence import (
    derivative_step_study, plot_derivative_study, divergence_study,
    plot_divergence_evidence, optimizer_conditions,
)

FIGURE_MANIFEST = {}


def save_figure(figure, filename):
    """Save a 150-dpi PNG and record its actual encoded pixel dimensions."""
    path = ROOT / "outputs" / filename
    figure.savefig(path, dpi=150)
    width, height = struct.unpack(">II", path.read_bytes()[16:24])
    FIGURE_MANIFEST[filename] = dict(
        path="outputs/" + filename, width_px=width, height_px=height, dpi=150)
    plt.close(figure)


def serializable(value):
    """Convert NumPy scalar/array values to JSON-native representations."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def gradient_checks():
    """Independently compare each parameter gradient with central differences."""
    parameters = fixed_example()
    gradients = backward(**parameters)
    errors = {}
    for key in ("W1", "b1", "W2", "b2"):
        def loss(candidate, name=key):
            """Evaluate BCE while perturbing one parameter tensor."""
            return forward(**dict(parameters, **{name: candidate}))["loss"]
        numerical = numerical_gradient(loss, parameters[key])
        errors[key] = float(np.max(np.abs(numerical - gradients[key])))
    return errors


def main():
    """Run all mandatory and bonus experiments with deterministic settings."""
    np.random.seed(42)
    FIGURE_MANIFEST.clear()
    (ROOT / "outputs").mkdir(exist_ok=True)
    results = {"seed": 42, "environment": {
        "python": platform.python_version(), "numpy": np.__version__,
        "matplotlib": matplotlib.__version__}}
    checks = {}

    save_figure(plot_transformations(), "matrix_transformations.png")
    areas = {name: area_report(a) for name, a in [
        ("rotation", rotation(np.pi / 4)), ("scaling", scaling()),
        ("shear", shear()), ("reflection", np.diag([-2.0, 0.5])),
        ("zero_matrix", np.zeros((2, 2))),
        ("near_singular", np.diag([1., 1e-16]))]}
    results["areas"] = areas
    checks["determinant_area_within_1_percent"] = all(r["relative_error"] <= .01 for r in areas.values())
    checks["singular_area_absolute_error_stable"] = all(
        areas[name]["error_kind"] == "absolute_near_zero_determinant"
        and areas[name]["absolute_error"] < 1e-12
        for name in ("zero_matrix", "near_singular"))

    matrix = np.array([[4., 1.], [1., 3.]])
    value, vector, iterations = power_iteration(matrix)
    # eig is used ONLY as the reference verification, never by the algorithm.
    reference_values, _ = np.linalg.eig(matrix)
    reference = float(np.real_if_close(reference_values[np.argmax(np.abs(reference_values))]))
    eigen_error = abs(value - reference) / abs(reference)
    results["power_iteration"] = dict(value=value, vector=vector, iterations=iterations,
                                      reference=reference, relative_error=eigen_error,
                                      verification_script="scripts/run_all.py",
                                      residual=float(np.linalg.norm(matrix @ vector - value * vector)))
    checks["eigenvalue_within_5_percent"] = eigen_error <= .05

    image = np.load(ROOT / "data/grace_hopper_64.npy", allow_pickle=False)
    figure, svd = plot_svd(image)
    save_figure(figure, "svd_comparison.png")
    for index, row in enumerate(svd, start=1):
        row.update(figure_path="outputs/svd_comparison.png",
                   panel_index_zero_based=index)
    results["svd"] = svd
    results["svd_image"] = dict(source_path="data/grace_hopper_64.npy",
                                shape=list(image.shape),
                                comparison_figure="outputs/svd_comparison.png",
                                panel_ranks=[10, 50, 100], effective_ranks=[10, 50, 64])
    checks["svd_requested_ranks_present"] = [r["requested_k"] for r in svd] == [10, 50, 100]
    checks["svd_rank_cap_and_full_reconstruction"] = svd[-1]["effective_k"] == 64 and svd[-1]["mse"] < 1e-24
    checks["svd_error_nonincreasing"] = all(svd[i]["mse"] >= svd[i+1]["mse"] for i in range(2))

    derivative = central_difference(lambda x: x * x, 3)
    results["derivative"] = dict(numerical=derivative, analytical=6.0, error=abs(derivative - 6))
    checks["derivative_error_within_1e_4"] = abs(derivative - 6) <= 1e-4
    study = derivative_step_study()
    results["derivative_step_study"] = study
    save_figure(plot_derivative_study(study), "derivative_step_study.png")
    checks["central_difference_sine_second_order"] = all(
        95 < study[i]["sine_error"] / study[i+1]["sine_error"] < 105
        for i in (0, 1))
    save_figure(plot_gradient(), "gradient_contours.png")
    point = np.array([1.5, -0.7])
    results["gradient_tangent_dot_product"] = float((2 * point) @ np.array([-point[1], point[0]]))
    checks["gradient_normal_to_contour_tangent"] = abs(results["gradient_tangent_dot_product"]) <= 1e-12

    results["backprop_forward"] = forward(**fixed_example())
    results["backprop_backward"] = backward(**fixed_example())
    results["gradient_check_max_abs_errors"] = gradient_checks()
    checks["all_parameter_gradients_match_finite_difference"] = max(results["gradient_check_max_abs_errors"].values()) < 1e-7
    checks["backprop_hand_checkpoint_4_decimals"] = bool(
        np.all(np.round(results["backprop_backward"]["d_z1"], 4) == [-.0440, -.0517])
        and round(results["backprop_forward"]["y_pred"], 4) == .6473)

    circle = {
        "GD lr=0.1": optimize(quadratic_gradient, [5, 5], VanillaGD(.1), 100),
        "Momentum lr=0.1 beta=0.9": optimize(quadratic_gradient, [5, 5], Momentum(.1, .9), 100)}
    save_figure(plot_paths(circle, title="Circle: GD and Momentum (same lr=0.1)"), "gd_momentum_circle.png")
    save_figure(plot_convergence(circle, title="Circle: Momentum is not always faster"), "circle_convergence.png")
    checks["gd_radius_after_100_within_0_1"] = np.linalg.norm(circle["GD lr=0.1"][-1]) <= .1
    results["circle"] = {name: dict(final=p[-1], final_radius=float(np.linalg.norm(p[-1])),
                                    first_radius_0_1=first_hit(p)) for name, p in circle.items()}

    learning_rates = [.1, .5, .75, 1., 1.1]
    learning_paths = {f"lr={lr}": optimize(quadratic_gradient, [5, 5], VanillaGD(lr), 30)
                      for lr in learning_rates}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for label, path in learning_paths.items():
        axes[0].plot(path[:, 0], path[:, 1], "--o", markersize=2, label=label)
        axes[1].semilogy(np.maximum(np.linalg.norm(path, axis=1), 1e-15), label=label)
    axes[0].set(xlabel="x", ylabel="y", title="Learning rate trajectories")
    axes[0].set_aspect("equal")
    axes[1].set(xlabel="Iteration", ylabel="Distance to origin", title="Threshold: 0 < lr < 1 converges")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=.2)
    save_figure(fig, "learning_rate_stability.png")
    results["learning_rates"] = {str(lr): dict(factor=1 - 2 * lr,
        final_radius=float(np.linalg.norm(learning_paths[f"lr={lr}"][-1]))) for lr in learning_rates}
    checks["lr_0_5_reaches_origin_in_one_step"] = bool(np.all(learning_paths["lr=0.5"][1] == 0))
    checks["lr_1_1_diverges"] = np.linalg.norm(learning_paths["lr=1.1"][-1]) > np.linalg.norm([5, 5]) * 100
    evidence, divergence_paths = divergence_study()
    results["learning_rate_divergence"] = evidence
    save_figure(plot_divergence_evidence(evidence, divergence_paths),
                "lr_ge_0_5_divergence.png")
    selected = [run for run in evidence["runs"]
                if run["lr"] in evidence["demonstrated_divergent_rates"]]
    checks["lr_ge_0_5_divergence_examples_verified"] = len(selected) == 3 and all(
        run["lr"] >= .5 and abs(run["update_factor"]) > 1
        and run["every_step_grows"] and run["radius_growth_factor"] > 100
        and run["closed_form_verified"] for run in selected)
    results["optimizer_experiments"] = optimizer_conditions()

    def elliptical_gradient(p):
        """Return gradient of x²+10y²."""
        return quadratic_gradient(p, 10)
    ellipse = {
        "GD lr=0.09": optimize(elliptical_gradient, [5, 5], VanillaGD(.09), 200),
        "Momentum lr=0.09 beta=0.9": optimize(elliptical_gradient, [5, 5], Momentum(.09, .9), 200)}
    save_figure(plot_paths(ellipse, 10, "Ellipse: same lr=0.09, beta=0.9"), "gd_momentum_ellipse.png")
    save_figure(plot_convergence(ellipse, 10, "Ellipse: fair comparison at the same lr"), "ellipse_convergence.png")
    # Extra tuned comparison keeps beta=0.9, showing an explicit useful setting.
    tuned = {
        "GD lr=0.01": optimize(elliptical_gradient, [5, 5], VanillaGD(.01), 300),
        "Momentum lr=0.01 beta=0.9": optimize(elliptical_gradient, [5, 5], Momentum(.01, .9), 300),
        "Adam lr=0.1": optimize(elliptical_gradient, [5, 5], Adam(.1), 300)}
    save_figure(plot_paths({k: v[:101] for k, v in tuned.items()}, 10,
                          "Ellipse: first 100 updates; explicit rates"), "bonus_optimizer_paths.png")
    save_figure(plot_convergence(tuned, 10, "Bonus: GD, Momentum, Adam (rates in legend)"), "bonus_optimizer_convergence.png")
    results["ellipse"] = {name: dict(final=p[-1], first_radius_0_1=first_hit(p))
                          for name, p in ellipse.items()}
    results["bonus_optimizers"] = {name: dict(final=p[-1], final_radius=float(np.linalg.norm(p[-1])),
                                             first_radius_0_1=first_hit(p)) for name, p in tuned.items()}
    checks["momentum_advantage_in_documented_setting"] = first_hit(tuned["Momentum lr=0.01 beta=0.9"]) < first_hit(tuned["GD lr=0.01"])

    newton = newton_method(elliptical_gradient, lambda p: np.diag([2., 20.]), [5, 5], 3)
    save_figure(plot_convergence({"GD lr=0.09": ellipse["GD lr=0.09"], "Newton": newton}, 10,
                                  "Newton: exact quadratic Hessian"), "bonus_newton.png")
    results["newton"] = dict(path=newton, first_radius_0_1=first_hit(newton))
    checks["newton_exact_quadratic_one_step"] = bool(np.allclose(newton[1], 0, atol=1e-12))

    save_figure(plot_distributions(), "probability_distributions.png")
    probabilities = softmax(np.array([1000., 1001., 1002.]))
    results["softmax"] = dict(probabilities=probabilities, sum=float(probabilities.sum()))
    checks["softmax_sum_within_1e_6"] = abs(probabilities.sum() - 1) <= 1e-6
    extreme_cases = []
    for logits in ([10000., 10001., 10002.], [-10000., -10001., -10002.],
                   [1000., -1000., 0.], [-1e6, -1e6, -1e6]):
        with np.errstate(over="raise", invalid="raise"):
            probability = softmax(logits)
        extreme_cases.append(dict(logits=logits, probabilities=probability,
                                  sum=float(probability.sum())))
    results["softmax_extreme_cases"] = extreme_cases
    checks["softmax_extreme_cases_stable"] = all(
        np.isfinite(r["probabilities"]).all()
        and abs(r["sum"] - 1) <= 1e-6 for r in extreme_cases)
    p, q = np.array([.3, .7]), np.array([.6, .4])
    results["information_theory"] = dict(entropy=entropy(p), cross_entropy=cross_entropy(p, q),
                                        kl_divergence=kl_divergence(p, q))
    checks["cross_entropy_equals_entropy_plus_kl"] = abs(cross_entropy(p, q) - entropy(p) - kl_divergence(p, q)) < 1e-12
    results["visualization_parameters"] = dict(
        circle_samples=100, gradient_grid_shape=[5, 5], gradient_scale=7,
        gradient_scale_units="xy", normal_pdf_samples=600,
        normal_pdf_domain=[-4, 5], bernoulli_bar_width=.32,
        bernoulli_bar_offsets=[-.17, .17],
        convergence_log_floor=1e-30, radius_log_floor=1e-15)
    results["figures"] = [FIGURE_MANIFEST[name] for name in sorted(FIGURE_MANIFEST)]
    manifest_lines = [
        "# 출력 PNG 경로와 해상도", "",
        "scripts/run_all.py가 실제 PNG 헤더에서 읽어 생성한다. 모든 그림은 150 dpi로 저장한다.",
        "입력 회색조 데이터는 64×64이며, 아래 Figure 해상도와 구분한다.", "",
        "| 출력 파일 | 실제 크기 (px) | dpi |",
        "|---|---|---|",
    ]
    for row in results["figures"]:
        manifest_lines.append(
            f'| [{row["path"]}](../{row["path"]}) | {row["width_px"]}×{row["height_px"]} | {row["dpi"]} |')
    (ROOT / "docs/OUTPUT_MANIFEST.md").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8")
    results["acceptance_checks"] = checks
    (ROOT / "outputs/metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=serializable) + "\n", encoding="utf-8")
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} {name}")
    if not all(checks.values()):
        raise SystemExit("Some assignment acceptance checks failed.")
    print(f"All {len(checks)} acceptance checks passed; PNGs and metrics saved.")
    return results


if __name__ == "__main__":
    main()
