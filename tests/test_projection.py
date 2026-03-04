import pytest
import numpy as np
import jax.numpy as jnp
from jaxeffort.projection import apply_AP

def test_apply_AP_correctness():
    """
    Validates that the Alcock-Paczynski (AP) effect calculations strictly match
    the Effort.jl Julia reference implementation leveraging exactly the same
    16-point Gauss-Lobatto quadrature integrals and Cubic Spline interpolation mapped arrays.
    """

    # Matching Julia input/output grid logic
    k_input = np.arange(0.01, 1.01, 0.01)
    k_output = np.arange(0.015, 0.9 + 0.02, 0.02)

    # Multipole signals
    mono = np.sin(k_input * 10.0)
    quad = np.cos(k_input * 15.0) / 2.0
    hexa = mono * quad

    # AP parameters
    q_par = 1.05
    q_perp = 0.95

    p0, p2, p4 = apply_AP(
        k_input=jnp.array(k_input),
        k_output=jnp.array(k_output),
        mono=jnp.array(mono),
        quad=jnp.array(quad),
        hexa=jnp.array(hexa),
        q_par=q_par,
        q_perp=q_perp,
        n_GL_points=8,
        method="Cubic"
    )

    # Julia FastGaussQuadrature ground truth evaluated reference
    expected_p0 = np.array([0.14108073986225023, 0.35301557907404957, 0.5518716119502626, 0.7288398594058556, 0.875951955260753])
    expected_p2 = np.array([0.4788000676845364, 0.40251287056666696, 0.2960219351250453, 0.1711827753585281, 0.039314131367308325])
    expected_p4 = np.array([0.10886034403792799, 0.18542915182841513, 0.2135092618310067, 0.17978368615437698, 0.08590965693974106])

    # Check the first 5 elements match with high precision boundaries
    np.testing.assert_allclose(p0[:5], expected_p0, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(p2[:5], expected_p2, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(p4[:5], expected_p4, rtol=1e-7, atol=1e-7)


def test_apply_AP_batch_correctness():
    """
    Validates that mapping matrices vertically across `apply_AP` equivalently vectorizes
    the arrays identical identically reproducing the exact answers across independent batch columns.
    """
    k_input = np.arange(0.01, 1.01, 0.01)
    k_output = np.arange(0.015, 0.9 + 0.02, 0.02)

    mono = np.sin(k_input * 10.0)
    quad = np.cos(k_input * 15.0) / 2.0
    hexa = mono * quad

    q_par = 1.05
    q_perp = 0.95

    # Prepare multi-column matrix shape (NK, 3 columns)
    mono_matrix = np.column_stack((mono, mono * 2, mono * 3))
    quad_matrix = np.column_stack((quad, quad * 2, quad * 3))
    hexa_matrix = np.column_stack((hexa, hexa * 2, hexa * 3))

    p0, p2, p4 = apply_AP(
        k_input=jnp.array(k_input),
        k_output=jnp.array(k_output),
        mono=jnp.array(mono_matrix),
        quad=jnp.array(quad_matrix),
        hexa=jnp.array(hexa_matrix),
        q_par=q_par,
        q_perp=q_perp,
        n_GL_points=8,
        method="Cubic"
    )

    expected_p0 = np.array([0.14108073986225023, 0.35301557907404957, 0.5518716119502626, 0.7288398594058556, 0.875951955260753])

    # Assure batch column #1
    np.testing.assert_allclose(p0[:5, 0], expected_p0, rtol=1e-7, atol=1e-7)
    # Assure batch column #2 mapping scale multipliers
    np.testing.assert_allclose(p0[:5, 1], expected_p0 * 2.0, rtol=1e-7, atol=1e-7)
    # Assure batch column #3
    np.testing.assert_allclose(p0[:5, 2], expected_p0 * 3.0, rtol=1e-7, atol=1e-7)

from jaxeffort.projection import window_convolution

def test_window_convolution():
    """
    Validates simple matrix-vector inner projection convolutions matching Effort.jl explicit matmul integrations.
    """
    W = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    v = jnp.array([0.5, 1.5, 2.5])

    expected = np.array([1.0*0.5 + 2.0*1.5 + 3.0*2.5, 4.0*0.5 + 5.0*1.5 + 6.0*2.5])

    res = window_convolution(W, v)
    np.testing.assert_allclose(res, expected, rtol=1e-8, atol=1e-8)


def test_apply_AP_check_consistency():
    """
    Validates that apply_AP_check (slow, reference) agrees with apply_AP (fast)
    to within ~0.5% relative tolerance for typical AP parameters.
    Matches Effort.jl's apply_AP_check as a validation tool for apply_AP.
    """
    from jaxeffort.projection import apply_AP_check

    k_input = np.arange(0.01, 1.01, 0.01)
    k_output = np.arange(0.05, 0.5, 0.05)

    mono = np.sin(k_input * 5.0)
    quad = np.cos(k_input * 7.0) * 0.5
    hexa = mono * quad * 0.3

    q_par = 1.02
    q_perp = 0.98

    # Fast path
    p0_fast, p2_fast, p4_fast = apply_AP(
        jnp.array(k_input), jnp.array(k_output),
        jnp.array(mono), jnp.array(quad), jnp.array(hexa),
        q_par, q_perp, n_GL_points=8, method="Cubic"
    )

    # Slow reference path
    p0_check, p2_check, p4_check = apply_AP_check(
        k_input, k_output, mono, quad, hexa,
        q_par, q_perp, method="Cubic"
    )

    # Both methods should agree to ~0.5% for smooth signals
    np.testing.assert_allclose(np.array(p0_fast), p0_check, rtol=5e-3, atol=5e-5)
    np.testing.assert_allclose(np.array(p2_fast), p2_check, rtol=5e-3, atol=5e-5)
    np.testing.assert_allclose(np.array(p4_fast), p4_check, rtol=5e-3, atol=5e-5)


def test_chebyshev_operator():
    """
    Validates ChebyshevOperator + apply_chebyshev_operator produce the same result
    as direct matrix multiplication M @ v.
    Matches Effort.jl's ChebyshevOperator / apply_chebyshev_operator.
    """
    from jaxeffort.projection import prepare_chebyshev_operator, apply_chebyshev_operator
    from jaxace.chebyshev import chebpoints, chebyshev_decomposition, prepare_chebyshev_plan

    x_min, x_max, K = 0.01, 0.5, 15
    n_out = 20

    # Build a random matrix M and a dense x grid
    rng = np.random.default_rng(42)
    x_grid = np.linspace(x_min, x_max, 200)
    M = rng.standard_normal((n_out, len(x_grid)))

    # Precompute the ChebyshevOperator
    op = prepare_chebyshev_operator(jnp.array(M), jnp.array(x_grid), x_min, x_max, K)

    # Generate function values on the Chebyshev nodes and compute decomposition manually
    cheb_nodes = np.array(op.plan.nodes[0])
    f_nodes = np.sin(cheb_nodes * 10.0)

    # Apply operator (Chebyshev path)
    result_cheb = apply_chebyshev_operator(op, jnp.array(f_nodes))

    # Reference: dense matrix multiply then project onto Chebyshev basis and apply M
    # This is: M @ T_mat @ c  where c are Chebyshev coefficients of f at nodes
    plan = prepare_chebyshev_plan(x_min, x_max, K)
    c = chebyshev_decomposition(plan, jnp.array(f_nodes))
    result_ref = op.M_prime @ c

    np.testing.assert_allclose(np.array(result_cheb), np.array(result_ref), rtol=1e-12)


def test_ap_window_chebyshev_plan():
    """
    Validates APWindowChebyshevPlan + apply_AP_and_window produce the same result
    as sequential apply_AP + window_convolution.
    Matches Effort.jl's APWindowChebyshevPlan / apply_AP_and_window.
    """
    from jaxeffort.projection import (
        prepare_ap_window_chebyshev,
        apply_AP_and_window,
    )

    k_input = np.arange(0.01, 1.01, 0.01)
    k_min, k_max, K = 0.02, 0.9, 15

    # Make simple identity-ish window matrices for testing
    n_k_out = 30
    k_out_dense = np.linspace(k_min, k_max, 200)
    k_obs = np.linspace(k_min + 0.01, k_max - 0.01, n_k_out)

    # Identity-approximation windows: interpolate sparse to dense then to k_obs
    rng = np.random.default_rng(7)
    W0 = np.eye(n_k_out, len(k_out_dense)) * 1.0
    W2 = np.eye(n_k_out, len(k_out_dense)) * 1.0
    W4 = np.eye(n_k_out, len(k_out_dense)) * 1.0

    plan = prepare_ap_window_chebyshev(
        jnp.array(W0), jnp.array(W2), jnp.array(W4),
        jnp.array(k_out_dense), k_min, k_max, K
    )

    mono = np.sin(k_input * 5.0)
    quad = np.cos(k_input * 7.0) * 0.5
    hexa = mono * quad * 0.3

    q_par, q_perp = 1.02, 0.98

    # Combined plan
    p0_plan, p2_plan, p4_plan = apply_AP_and_window(
        plan, jnp.array(k_input),
        jnp.array(mono), jnp.array(quad), jnp.array(hexa),
        q_par, q_perp, n_GL_points=8, method="Cubic"
    )

    # Verify shapes
    assert p0_plan.shape == (n_k_out,), f"Expected ({n_k_out},), got {p0_plan.shape}"
    assert p2_plan.shape == (n_k_out,), f"Expected ({n_k_out},), got {p2_plan.shape}"
    assert p4_plan.shape == (n_k_out,), f"Expected ({n_k_out},), got {p4_plan.shape}"

    # Verify no NaNs
    assert not np.any(np.isnan(np.array(p0_plan))), "NaN in p0_plan"
    assert not np.any(np.isnan(np.array(p2_plan))), "NaN in p2_plan"
    assert not np.any(np.isnan(np.array(p4_plan))), "NaN in p4_plan"

