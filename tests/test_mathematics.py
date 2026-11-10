"""Independent numerical oracles and edge cases for the mathematical modules."""

import ast
import math
import os
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
import matplotlib
matplotlib.use("Agg")
import numpy as np

from src.linear_algebra import (
    rotation, scaling, shear, unit_circle, transform_points,
    polygon_area, area_report, power_iteration, svd_compress, svd_restore,
)
from src.calculus import central_difference, numerical_gradient
from src.backprop import fixed_example, forward, backward
from src.optimizer import (
    VanillaGD, Momentum, Adam, optimize, quadratic_gradient, first_hit, newton_method,
)
from src.probability import (
    sigmoid, softmax, normal_pdf, bernoulli_pmf, entropy, cross_entropy, kl_divergence,
)


class LinearAlgebraTests(unittest.TestCase):
    """Test transformations against geometry and decompositions against oracles."""

    def test_quarter_turn(self):
        """A quarter turn sends the x basis vector onto the y basis vector."""
        np.testing.assert_allclose(rotation(math.pi/2) @ [1, 0], [0, 1], atol=1e-15)

    def test_scaling_and_shear(self):
        """Check separate geometric transformations on a known point."""
        np.testing.assert_allclose(transform_points([[1, 2]], scaling()), [[2, 1]])
        np.testing.assert_allclose(shear(.5) @ [1, 2], [2, 2])

    def test_area_nonunit_determinant(self):
        """A determinant of six changes area by six, including reflection."""
        for matrix in [np.diag([2., 3.]), np.diag([-2., 3.])]:
            self.assertAlmostEqual(area_report(matrix)["area_ratio"], 6)

    def test_area_singular_transform(self):
        """A rank-one transform collapses a planar shape to zero area."""
        self.assertLess(area_report(np.diag([1., 0.]))["area_ratio"], 1e-12)

    def test_circle_area(self):
        """Polygon area approximates π independently of transformation ratios."""
        self.assertLess(abs(polygon_area(unit_circle()) - math.pi) / math.pi, .01)

    def test_power_analytic_eigenvalue(self):
        """Use the quadratic-formula eigenvalue rather than another eigensolver."""
        a = np.array([[4., 1.], [1., 3.]])
        value, vector, _ = power_iteration(a)
        self.assertAlmostEqual(value, (7 + math.sqrt(5))/2, places=9)
        np.testing.assert_allclose(a @ vector, value * vector, atol=1e-8)

    def test_power_negative_dominant(self):
        """Sign-alternating vectors still converge by eigenpair residual."""
        value, vector, _ = power_iteration(np.diag([-5., 2.]))
        self.assertAlmostEqual(value, -5)
        self.assertAlmostEqual(abs(vector[0]), 1)

    def test_power_zero_matrix(self):
        """A zero matrix has eigenvalue zero and a normalized vector."""
        value, vector, _ = power_iteration(np.zeros((2, 2)))
        self.assertEqual(value, 0)
        self.assertAlmostEqual(np.linalg.norm(vector), 1)

    def test_power_invalid_inputs(self):
        """Reject non-square matrices and zero initialization."""
        with self.assertRaises(ValueError):
            power_iteration(np.ones((2, 3)))
        with self.assertRaises(ValueError):
            power_iteration(np.eye(2), initial=[0, 0])

    def test_power_nonconvergence(self):
        """A real rotation without a real eigenvector should not claim success."""
        with self.assertRaises(RuntimeError):
            power_iteration(np.array([[0., -1.], [1., 0.]]), max_iter=20)

    def test_svd_optimal_rank_error(self):
        """Known singular values give an independent rank-one residual norm."""
        image = np.diag([5., 3., 1.])
        restored = svd_restore(svd_compress(image, 1))
        self.assertAlmostEqual(float(np.sum((image-restored)**2)), 3**2 + 1**2)

    def test_svd_rank_cap(self):
        """Oversized rank requests reconstruct a rectangular matrix exactly."""
        image = np.arange(12.).reshape(3, 4)
        packed = svd_compress(image, 100)
        self.assertEqual(packed["effective_k"], 3)
        np.testing.assert_allclose(svd_restore(packed), image, atol=1e-12)

    def test_svd_reject_invalid(self):
        """Reject invalid rank and color-array input."""
        for k in [0, -1, 2.5, True]:
            with self.assertRaises(ValueError):
                svd_compress(np.eye(2), k)
        with self.assertRaises(ValueError):
            svd_compress(np.zeros((2, 2, 3)), 1)


class CalculusBackpropTests(unittest.TestCase):
    """Validate differentiation using analytic functions and finite differences."""

    def test_scalar_derivatives(self):
        """Check polynomial and trigonometric functions at nontrivial points."""
        self.assertLess(abs(central_difference(lambda x: x*x, 3) - 6), 1e-4)
        self.assertLess(abs(central_difference(math.sin, .7) - math.cos(.7)), 1e-8)

    def test_array_gradient(self):
        """Central differences preserve a matrix parameter's shape."""
        a = np.array([[.2, -.3], [1., 2.]])
        actual = numerical_gradient(lambda x: np.sum(x**2), a)
        np.testing.assert_allclose(actual, 2*a, atol=1e-9)

    def test_invalid_difference_step(self):
        """Zero or negative difference steps are undefined."""
        with self.assertRaises(ValueError):
            central_difference(math.sin, 1, h=0)
        with self.assertRaises(ValueError):
            numerical_gradient(lambda x: np.sum(x), [1], h=-1)

    def test_fixed_forward_checkpoint(self):
        """Check published fixed-example values as independent constants."""
        result = forward(**fixed_example())
        self.assertAlmostEqual(result["y_pred"], .6472915699946178, places=12)
        self.assertAlmostEqual(result["loss"], .4349584368514606, places=12)

    def test_fixed_backward_checkpoints(self):
        """Catch the incorrect reference value in the assignment's example."""
        gradient = backward(**fixed_example())
        np.testing.assert_allclose(gradient["d_z1"], [-.043978515808697, -.051733504390926], atol=1e-12)
        np.testing.assert_array_equal(gradient["W1"][:, 1], 0)

    def test_all_gradients_multiple_targets(self):
        """Nonzero input coordinates and both labels prevent masked mistakes."""
        for target in [0., 1.]:
            p = fixed_example()
            p.update(x=np.array([.7, -1.2]), b1=np.array([.2, -.1]), b2=np.array(.3), y_true=target)
            analytical = backward(**p)
            for name in ["W1", "b1", "W2", "b2"]:
                def loss(candidate):
                    """Substitute one complete parameter tensor into the loss."""
                    return forward(**dict(p, **{name: candidate}))["loss"]
                numerical = numerical_gradient(loss, p[name])
                np.testing.assert_allclose(analytical[name], numerical, atol=1e-8)


class OptimizerTests(unittest.TestCase):
    """Check optimizer state, closed-form GD, and exact quadratic Newton steps."""

    def test_gd_closed_form(self):
        """After n steps on x²+y², each coordinate is x0*(1-2lr)^n."""
        path = optimize(quadratic_gradient, [5, 5], VanillaGD(.1), 100)
        np.testing.assert_allclose(path[-1], [5*.8**100]*2, rtol=1e-12)
        self.assertLess(np.linalg.norm(path[-1]), .1)

    def test_learning_rate_boundary(self):
        """Demonstrate convergence, neutral oscillation, and true divergence."""
        instant = optimize(quadratic_gradient, [5, 5], VanillaGD(.5), 1)
        alternating = optimize(quadratic_gradient, [5, 5], VanillaGD(.75), 20)
        neutral = optimize(quadratic_gradient, [5, 5], VanillaGD(1), 20)
        divergent = optimize(quadratic_gradient, [5, 5], VanillaGD(1.1), 20)
        np.testing.assert_array_equal(instant[-1], [0, 0])
        self.assertLess(np.linalg.norm(alternating[-1]), .001)
        np.testing.assert_allclose(neutral[-1], [5, 5])
        self.assertGreater(np.linalg.norm(divergent[-1]), np.linalg.norm([5, 5]))

    def test_momentum_zero_beta(self):
        """With beta zero the entire path must equal Vanilla GD."""
        a = optimize(quadratic_gradient, [5, 3], VanillaGD(.03), 30)
        b = optimize(quadratic_gradient, [5, 3], Momentum(.03, 0), 30)
        np.testing.assert_allclose(a, b)

    def test_momentum_first_two_steps(self):
        """Check velocity accumulation independently with a constant gradient."""
        optimizer = Momentum(.1, .9)
        first = optimizer.step([1.], [2.])
        second = optimizer.step(first, [2.])
        np.testing.assert_allclose(first, [.8])
        np.testing.assert_allclose(second, [.42])

    def test_adam_bias_correction(self):
        """A constant gradient yields normalized unit steps after correction."""
        optimizer = Adam(.1)
        point = np.array([1., -1.])
        for _ in range(3):
            point = optimizer.step(point, [2., -2.])
        np.testing.assert_allclose(point, [.7, -.7], atol=1e-8)

    def test_adam_zero_gradient(self):
        """Zero moments plus epsilon produce a finite unchanged parameter."""
        np.testing.assert_array_equal(Adam().step([2., 3.], [0., 0.]), [2., 3.])

    def test_momentum_documented_advantage(self):
        """At the shared small rate Momentum first reaches the target sooner."""
        def gradient(p):
            """Return the elliptical function gradient."""
            return quadratic_gradient(p, 10)
        gd = optimize(gradient, [5, 5], VanillaGD(.01), 300)
        momentum = optimize(gradient, [5, 5], Momentum(.01, .9), 300)
        self.assertLess(first_hit(momentum), first_hit(gd))

    def test_newton_one_step(self):
        """Exact positive-definite quadratic curvature reaches zero at once."""
        path = newton_method(lambda x: quadratic_gradient(x, 10),
                             lambda x: np.diag([2., 20.]), [5, 5], 1)
        np.testing.assert_allclose(path[-1], 0, atol=1e-14)

    def test_input_not_mutated(self):
        """Optimizer execution preserves the caller's initial point."""
        initial = np.array([2., 3.])
        optimize(quadratic_gradient, initial, VanillaGD(.1), 5)
        np.testing.assert_array_equal(initial, [2., 3.])

    def test_invalid_optimizer_settings(self):
        """Reject invalid learning rates and momentum/moment parameters."""
        for lr in [0, -1, np.inf]:
            with self.assertRaises(ValueError):
                VanillaGD(lr)
        with self.assertRaises(ValueError):
            Momentum(beta=1)
        with self.assertRaises(ValueError):
            Adam(eps=0)


class ProbabilityTests(unittest.TestCase):
    """Test distributions, log-sum stability, and information-theory boundaries."""

    def test_extreme_sigmoid(self):
        """Extreme logits are finite without exponential overflow."""
        with np.errstate(over="raise", invalid="raise"):
            np.testing.assert_array_equal(sigmoid([-1e4, 0, 1e4]), [0, .5, 1])

    def test_softmax_invariance(self):
        """A common logit shift does not change normalized probabilities."""
        actual = softmax([1000., 1001., 1002.])
        np.testing.assert_allclose(actual, softmax([0., 1., 2.]))
        self.assertLess(abs(actual.sum() - 1), 1e-6)

    def test_batched_softmax(self):
        """Each row in a batch has independent normalization."""
        p = softmax([[1., 2.], [100., 101.]])
        np.testing.assert_allclose(p.sum(axis=1), 1)

    def test_normal_pdf_center(self):
        """Check the Gaussian normalization coefficient analytically."""
        self.assertAlmostEqual(float(normal_pdf(0)), 1/math.sqrt(2*math.pi))
        self.assertAlmostEqual(float(normal_pdf(2, 2, .5)), 1/math.sqrt(math.pi))

    def test_bernoulli_endpoints(self):
        """Degenerate Bernoulli distributions still sum to one."""
        np.testing.assert_array_equal(bernoulli_pmf(0), [1, 0])
        np.testing.assert_array_equal(bernoulli_pmf(1), [0, 1])

    def test_information_identity(self):
        """Cross-entropy equals entropy plus KL in natural-log units."""
        p, q = [.3, .7], [.6, .4]
        self.assertAlmostEqual(cross_entropy(p, q), entropy(p)+kl_divergence(p, q))
        self.assertAlmostEqual(kl_divergence(p, p), 0)
        self.assertGreater(kl_divergence(p, q), 0)

    def test_zero_probability_conventions(self):
        """Use zero-log-zero convention and infinite loss for missing support."""
        self.assertEqual(entropy([1., 0.]), 0)
        self.assertTrue(math.isinf(cross_entropy([1., 0.], [0., 1.])))

    def test_invalid_distributions(self):
        """Reject probability vectors with invalid mass and invalid variance."""
        for p in [[-.1, 1.1], [.2, .2], [np.nan, 0]]:
            with self.assertRaises(ValueError):
                entropy(p)
        with self.assertRaises(ValueError):
            normal_pdf(0, variance=0)
        with self.assertRaises(ValueError):
            bernoulli_pmf(2)


class RepositoryConstraintTests(unittest.TestCase):
    """Check pedagogical constraints without importing forbidden libraries."""

    def test_source_imports_and_docstrings(self):
        """All implementation math uses NumPy and every definition is documented."""
        for path in (ROOT/"src").glob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    self.assertTrue(ast.get_docstring(node), f"{path}:{node.lineno}")
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertIn(alias.name.split(".")[0], {"numpy", "matplotlib"})
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    self.assertIn(node.module.split(".")[0], {"numpy", "matplotlib"})
            self.assertNotIn("np.linalg.eig(", path.read_text())

    def test_image_constraint(self):
        """The committed dataset satisfies grayscale and maximum size requirements."""
        image = np.load(ROOT/"data/grace_hopper_64.npy", allow_pickle=False)
        self.assertEqual(image.ndim, 2)
        self.assertLessEqual(max(image.shape), 64)
        self.assertTrue(np.isfinite(image).all())
        self.assertGreaterEqual(image.min(), 0)
        self.assertLessEqual(image.max(), 1)


if __name__ == "__main__":
    unittest.main()
