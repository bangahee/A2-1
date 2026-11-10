"""Check the complete hand-in structure and retained execution evidence."""

import json
import struct
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
FIGURES = [
    "matrix_transformations.png", "svd_comparison.png", "gradient_contours.png",
    "gd_momentum_circle.png", "circle_convergence.png", "learning_rate_stability.png",
    "gd_momentum_ellipse.png", "ellipse_convergence.png", "probability_distributions.png",
    "bonus_optimizer_paths.png", "bonus_optimizer_convergence.png", "bonus_newton.png",
    "derivative_step_study.png", "lr_ge_0_5_divergence.png",
]


def main():
    """Fail if a required file, numerical check, or executed notebook is missing."""
    required = [
        "src/linear_algebra.py", "src/calculus.py", "src/optimizer.py",
        "README.md", "requirements.txt", "CONTRIBUTING.md", "DERIVATIONS.md",
        "docs/REQUIREMENTS_CHECKLIST.md", "docs/MATHEMATICAL_DECISIONS.md",
        "docs/OUTPUT_MANIFEST.md", "docs/NOTEBOOK_INDEX.md",
        "outputs/metrics.json", "data/grace_hopper_64.npy",
    ] + ["outputs/" + name for name in FIGURES]
    for filename in required:
        path = ROOT / filename
        assert path.is_file() and path.stat().st_size > 0, filename
    metrics = json.loads((ROOT/"outputs/metrics.json").read_text())
    checks = metrics["acceptance_checks"]
    assert len(checks) >= 22 and all(v is True for v in checks.values())
    assert checks["lr_ge_0_5_divergence_examples_verified"] is True
    assert metrics["power_iteration"]["reference_method"] == "np.linalg.eig"
    assert metrics["power_iteration"]["passed"] and metrics["power_iteration"]["eigenvector_passed"]
    assert {d["path"] for d in metrics["documents"]} == {
        "notebooks/backprop_derivation.ipynb", "notebooks/probability_loss.ipynb"}
    proof = (ROOT/"DERIVATIONS.md").read_text()
    for topic in ["순전파 수식과 손계산", "연쇄 법칙", "MSE 최소화",
                  "Binary Cross-Entropy", "Categorical Cross-Entropy"]:
        assert topic in proof, topic
    manifest = {row["path"]: row for row in metrics["figures"]}
    for filename in FIGURES:
        path = "outputs/" + filename
        width, height = struct.unpack(">II", (ROOT/path).read_bytes()[16:24])
        assert (width, height) == (manifest[path]["width_px"], manifest[path]["height_px"]), path
        assert manifest[path]["dpi"] == 150
    total_code = 0
    names = ["backprop_derivation.ipynb", "probability_loss.ipynb",
             "experiment_validation.ipynb"]
    for name in names:
        nb = nbformat.read(ROOT/"notebooks"/name, as_version=4)
        nbformat.validate(nb)
        code = [cell for cell in nb.cells if cell.cell_type == "code"]
        assert code and all(cell.execution_count is not None for cell in code), name
        assert all(output.output_type != "error" for cell in code for output in cell.outputs), name
        total_code += len(code)
    print(f"PASS submission: {len(FIGURES)} PNGs, {len(checks)} acceptance checks, "
          f"{len(names)} executed notebooks ({total_code} code cells).")


if __name__ == "__main__":
    main()
