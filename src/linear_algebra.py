"""Linear transformations, dominant eigenpairs, and SVD image compression."""

import numpy as np
import matplotlib.pyplot as plt


def unit_circle(n=100):
    """Return n sampled unit-circle points with shape (n, 2)."""
    if n < 4:
        raise ValueError("At least four samples are required.")
    theta = np.linspace(0, 2 * np.pi, n)
    return np.column_stack((np.cos(theta), np.sin(theta)))


def rotation(theta):
    """Return the 2D counterclockwise rotation matrix R(theta)."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def scaling(sx=2.0, sy=0.5):
    """Return the diagonal scaling matrix diag(sx, sy)."""
    return np.diag([float(sx), float(sy)])


def shear(k=0.8):
    """Return horizontal shear Sh(k) = [[1,k],[0,1]]."""
    return np.array([[1.0, k], [0.0, 1.0]])


def transform_points(points, matrix):
    """Apply column-vector transformation A to points stored as rows."""
    points, matrix = np.asarray(points, float), np.asarray(matrix, float)
    if points.ndim != 2 or points.shape[1] != 2 or matrix.shape != (2, 2):
        raise ValueError("Expected points (n,2) and matrix (2,2).")
    return points @ matrix.T


def polygon_area(points):
    """Compute unsigned polygon area with the shoelace formula."""
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 2 or len(p) < 3:
        raise ValueError("Expected at least three 2D vertices.")
    return float(abs(np.dot(p[:, 0], np.roll(p[:, 1], -1))
                     - np.dot(p[:, 1], np.roll(p[:, 0], -1))) / 2)


def area_report(matrix, n=100):
    """Compare transformed polygon area / original area with abs(det(A)).

    Above 1e-15, relative_error is |ratio-|det||/|det|. At or below that
    threshold it records absolute ratio error instead, avoiding division by
    zero/amplified roundoff for a collapsed or nearly collapsed polygon.
    error_kind explicitly distinguishes the two error definitions.
    """
    circle = unit_circle(n)
    before = polygon_area(circle)
    after = polygon_area(transform_points(circle, matrix))
    determinant = float(np.linalg.det(matrix))
    ratio = after / before
    expected = abs(determinant)
    relative = expected > 1e-15
    absolute_error = abs(ratio - expected)
    error = absolute_error / expected if relative else absolute_error
    return dict(determinant=determinant, area_before=before, area_after=after,
                area_ratio=ratio, relative_error=float(error),
                absolute_error=float(absolute_error),
                error_kind="relative" if relative else "absolute_near_zero_determinant")


def plot_transformations():
    """Overlay original/transformed circles and annotate area evidence."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    matrices = [rotation(np.pi / 4), scaling(), shear()]
    for ax, matrix, title in zip(axes, matrices, ["Rotation 45°", "Scaling (2, 0.5)", "Shear k=0.8"]):
        circle = unit_circle()
        moved = transform_points(circle, matrix)
        report = area_report(matrix)
        ax.plot(*circle.T, label="Original", color="#2563eb")
        ax.plot(*moved.T, label="Transformed", color="#ef4444")
        # Track one radius so rotation remains visible on a symmetric circle.
        start, end = np.array([1., 0.]), matrix @ np.array([1., 0.])
        ax.plot([0, start[0]], [0, start[1]], "--o", color="#2563eb", markersize=4)
        ax.plot([0, end[0]], [0, end[1]], "--o", color="#ef4444", markersize=4)
        ax.set(title=f"{title}\n det={report['determinant']:.3f}, area ratio={report['area_ratio']:.3f}",
               xlabel="x", ylabel="y")
        ax.set_aspect("equal")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
    return fig


def power_iteration(matrix, tol=1e-10, max_iter=10000, seed=42, initial=None):
    """Return (dominant-by-magnitude eigenvalue, unit vector, iterations).

    Uses normalized repeated multiplication and the Rayleigh quotient; no
    eigensolver. A unique dominant magnitude and nonzero initial projection
    are required. Residual convergence also handles negative eigenvalues.
    """
    a = np.asarray(matrix, float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not a.size:
        raise ValueError("matrix must be nonempty and square.")
    if not np.all(np.isfinite(a)) or tol <= 0 or max_iter < 1:
        raise ValueError("Use a finite matrix and positive tolerance/iterations.")
    v = np.random.default_rng(seed).normal(size=a.shape[0]) if initial is None else np.asarray(initial, float).copy()
    if v.shape != (a.shape[0],) or not np.isfinite(v).all() or np.linalg.norm(v) == 0:
        raise ValueError("initial must be a finite nonzero vector of matching size.")
    v /= np.linalg.norm(v)
    for iteration in range(1, max_iter + 1):
        product = a @ v
        norm = np.linalg.norm(product)
        if norm == 0:
            return 0.0, v, iteration
        v = product / norm
        value = float(v @ a @ v)
        if np.linalg.norm(a @ v - value * v) <= tol * max(1.0, abs(value)):
            return value, v, iteration
    raise RuntimeError("Power Iteration did not converge; inspect the spectral gap.")


def compare_power_iteration(matrix, relative_tolerance=0.05, seed=42):
    """Verify a real symmetric matrix's custom eigenpair against np.linalg.eig.

    This comparison function is validation code only. power_iteration itself
    uses repeated matrix multiplication, normalization, and Rayleigh quotients.
    Eigenvector error minimizes over ±v because eigenvector signs are arbitrary.
    """
    a = np.asarray(matrix, float)
    if (a.ndim != 2 or a.shape[0] != a.shape[1] or not np.allclose(a, a.T)
            or not np.isfinite(relative_tolerance) or relative_tolerance < 0):
        raise ValueError("Use a real symmetric matrix and nonnegative tolerance.")
    value, vector, iterations = power_iteration(a, seed=seed)
    # Reference eigensolver is called only AFTER the independent implementation.
    reference_values, reference_vectors = np.linalg.eig(a)
    index = int(np.argmax(np.abs(reference_values)))
    reference = float(np.real_if_close(reference_values[index]))
    reference_vector = np.asarray(np.real_if_close(reference_vectors[:, index]), float)
    reference_vector /= np.linalg.norm(reference_vector)
    error = abs(value-reference)/abs(reference) if reference != 0 else abs(value)
    direction_error = float(min(np.linalg.norm(vector-reference_vector),
                                np.linalg.norm(vector+reference_vector)))
    return dict(matrix=a, value=value, vector=vector, iterations=iterations,
                reference=reference, reference_vector=reference_vector,
                relative_error=float(error), relative_tolerance=relative_tolerance,
                passed=bool(error <= relative_tolerance),
                eigenvector_sign_invariant_error=direction_error,
                eigenvector_absolute_tolerance=1e-6,
                eigenvector_passed=bool(direction_error <= 1e-6),
                residual=float(np.linalg.norm(a @ vector-value*vector)),
                verification_function="src/linear_algebra.py:compare_power_iteration",
                verification_script="scripts/run_all.py",
                reference_method="np.linalg.eig")


def svd_compress(image, k):
    """Return truncated factors and effective rank, clamping k to min(shape).

    np.linalg.svd is permitted for the assignment's SVD exercise.
    Storage counts floating-point values, not encoded PNG/JPEG bytes.
    """
    a = np.asarray(image, float)
    if a.ndim != 2 or not a.size or not np.isfinite(a).all():
        raise ValueError("image must be a nonempty finite grayscale matrix.")
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
        raise ValueError("k must be a positive integer.")
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    effective = min(int(k), len(s))
    return {"u": u[:, :effective], "s": s[:effective], "vt": vt[:effective],
            "requested_k": int(k), "effective_k": effective}


def svd_restore(compressed):
    """Reconstruct A_k = U_k diag(s_k) V_k^T without clipping errors."""
    return (compressed["u"] * compressed["s"]) @ compressed["vt"]


def plot_svd(image, ranks=(10, 50, 100)):
    """Compare original image and reconstructions in a single figure."""
    fig, axes = plt.subplots(1, len(ranks) + 1, figsize=(14, 4), constrained_layout=True)
    axes[0].imshow(image, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"Original {image.shape[0]}×{image.shape[1]}")
    metrics = []
    for ax, k in zip(axes[1:], ranks):
        packed = svd_compress(image, k)
        restored = svd_restore(packed)
        stored = sum(packed[key].size for key in ("u", "s", "vt"))
        mse = float(np.mean((image - restored) ** 2))
        metrics.append(dict(requested_k=k, effective_k=packed["effective_k"],
                            mse=mse, stored_values=stored,
                            storage_ratio=stored / image.size))
        ax.imshow(restored, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"k={k}, effective={packed['effective_k']}\nMSE={mse:.2e}; storage={stored/image.size:.2f}×")
    for ax in axes:
        ax.axis("off")
    return fig, metrics
