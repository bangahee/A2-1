"""Check the complete hand-in structure and retained execution evidence."""

import json
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
FIGURES = [
    "matrix_transformations.png", "svd_comparison.png", "gradient_contours.png",
    "gd_momentum_circle.png", "circle_convergence.png", "learning_rate_stability.png",
    "gd_momentum_ellipse.png", "ellipse_convergence.png", "probability_distributions.png",
    "bonus_optimizer_paths.png", "bonus_optimizer_convergence.png", "bonus_newton.png",
]


def main():
    """Fail if a required file, numerical check, or executed notebook is missing."""
    required = [
        "src/linear_algebra.py", "src/calculus.py", "src/optimizer.py",
        "README.md", "requirements.txt", "CONTRIBUTING.md",
        "docs/REQUIREMENTS_CHECKLIST.md", "docs/MATHEMATICAL_DECISIONS.md",
        "outputs/metrics.json", "data/grace_hopper_64.npy",
    ] + ["outputs/" + name for name in FIGURES]
    for filename in required:
        path = ROOT / filename
        assert path.is_file() and path.stat().st_size > 0, filename
    metrics = json.loads((ROOT/"outputs/metrics.json").read_text())
    checks = metrics["acceptance_checks"]
    assert len(checks) >= 16 and all(v is True for v in checks.values())
    total_code = 0
    for name in ["backprop_derivation.ipynb", "probability_loss.ipynb"]:
        nb = nbformat.read(ROOT/"notebooks"/name, as_version=4)
        nbformat.validate(nb)
        code = [cell for cell in nb.cells if cell.cell_type == "code"]
        assert code and all(cell.execution_count is not None for cell in code), name
        assert all(output.output_type != "error" for cell in code for output in cell.outputs), name
        total_code += len(code)
    print(f"PASS submission: {len(FIGURES)} PNGs, {len(checks)} acceptance checks, "
          f"2 executed notebooks ({total_code} code cells).")


if __name__ == "__main__":
    main()
