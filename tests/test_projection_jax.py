"""Public projection contracts: independent references and JAX transformations."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxeffort import (
    apply_AP,
    apply_AP_check,
    apply_AP_and_window,
    apply_chebyshev_operator,
    prepare_ap_window_chebyshev,
    prepare_chebyshev_operator,
    q_par_perp,
    w0waCDMCosmology,
    window_convolution,
)
from jaxeffort.projection import gausslobatto


@pytest.mark.parametrize("n", [4, 8, 16, 32, 64])
def test_lobatto_polynomial_exactness(n):
    nodes, weights = gausslobatto(n)
    assert nodes[0] == -1 and nodes[-1] == 1
    assert np.all(np.diff(nodes) > 0)
    assert np.all(weights > 0)
    for degree in range(2*n - 2):
        expected = 0.0 if degree % 2 else 2.0/(degree + 1)
        np.testing.assert_allclose(weights @ nodes**degree, expected, rtol=0, atol=2e-13)


def signals():
    k = jnp.linspace(0.01, 1.0, 100)
    v = (jnp.sin(5*k), 0.5*jnp.cos(7*k), 0.15*jnp.sin(5*k)*jnp.cos(7*k))
    matrices = tuple(jnp.stack((x, jnp.cos((i+2)*k) + 0.3*x,
                               (i+1)*jnp.exp(-k)), axis=1) for i, x in enumerate(v))
    return k, jnp.linspace(0.03, 0.8, 31), v, matrices


@pytest.mark.parametrize("method", ["Cubic", "Akima"])
def test_full_julia_reference(method):
    k, ko, _, matrices = signals()
    reference = np.loadtxt(Path(__file__).parent / "reference_data" /
                           f"projection_{method.lower()}.txt")
    np.testing.assert_allclose(ko, reference[:, 0], atol=2e-16)
    qp, qt = jnp.array([1.02, 0.97, 1.04]), jnp.array([0.98, 1.03, 0.96])
    result = apply_AP(k, ko, *matrices, qp, qt, method=method)
    for column in range(3):
        scalar = apply_AP(k, ko, *(x[:, column] for x in matrices),
                          qp[column], qt[column], method=method)
        for ell in range(3):
            expected = reference[:, 1 + 3*column + ell]
            np.testing.assert_allclose(result[ell][:, column], expected, rtol=2e-10, atol=2e-11)
            np.testing.assert_allclose(scalar[ell], expected, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("method", ["Cubic", "Akima"])
@pytest.mark.parametrize("batch", [False, True])
def test_ap_identity_and_reverse_mode(method, batch):
    k, ko, vectors, matrices = signals()
    spectra = matrices if batch else vectors
    identity = apply_AP(k, k, *spectra, 1.0, 1.0, method=method)
    for actual, expected in zip(identity, spectra):
        np.testing.assert_allclose(actual, expected, atol=3e-12, rtol=3e-12)

    def loss(q, amplitude):
        out = apply_AP(k, ko, *(x * amplitude for x in spectra),
                       q[0], q[1], method=method)
        return sum(jnp.sum(x*x) for x in out)

    q = jnp.array([[1.02, 0.97, 1.04], [0.98, 1.03, 0.96]]) if batch else jnp.array([1.02, 0.98])
    reverse = jax.jit(jax.grad(loss, argnums=(0, 1)))(q, 1.1)
    forward = jax.jacfwd(loss, argnums=(0, 1))(q, 1.1)
    for a, b in zip(reverse, forward):
        assert np.all(np.isfinite(a))
        np.testing.assert_allclose(a, b, rtol=2e-10, atol=2e-10)
    # Independent central differences at an interior point, not a spline knot.
    direction = jnp.ones_like(q) * 0.3
    eps = 1e-5
    fd = (loss(q + eps*direction, 1.1) - loss(q - eps*direction, 1.1)) / (2*eps)
    np.testing.assert_allclose(jnp.sum(reverse[0]*direction), fd, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("which", ["parallel", "perpendicular"])
def test_mixed_scalar_and_column_ap(which):
    k, ko, _, matrices = signals()
    q = jnp.array([0.98, 1.01, 1.04])
    qp, qt = (q, 0.97) if which == "parallel" else (1.02, q)
    actual = apply_AP(k, ko, *matrices, qp, qt)
    expected = [apply_AP(k, ko, *(x[:, c] for x in matrices),
                         qp[c] if which == "parallel" else qp,
                         qt if which == "parallel" else qt[c]) for c in range(3)]
    for ell in range(3):
        np.testing.assert_allclose(actual[ell], jnp.stack([r[ell] for r in expected], axis=1), atol=1e-12)


@pytest.mark.parametrize("q", [jnp.ones(2), jnp.ones((3, 1))])
def test_invalid_ap_batch_shapes(q):
    k, ko, _, matrices = signals()
    with pytest.raises(ValueError, match="AP parameters"):
        apply_AP(k, ko, *matrices, q, 1.0)


def make_window_plan(scale=1.0):
    kd = jnp.linspace(0.03, 0.8, 200)
    rng = np.random.default_rng(42)
    windows = tuple(scale*jnp.array(rng.normal(size=(7, 200)))/200 for _ in range(3))
    return prepare_ap_window_chebyshev(*windows, kd, 0.03, 0.8, 31), kd, windows


@pytest.mark.parametrize("method", ["Cubic", "Akima"])
@pytest.mark.parametrize("batch", [False, True])
def test_plan_argument_jit_dense_parity_and_gradients(method, batch):
    k, _, vectors, matrices = signals()
    spectra = matrices if batch else vectors
    plan, kd, windows = make_window_plan()
    qp = jnp.array([1.02, 0.97, 1.04]) if batch else 1.02
    qt = jnp.array([0.98, 1.03, 0.96]) if batch else 0.98
    compiled = jax.jit(apply_AP_and_window, static_argnames=("method", "n_GL_points"))
    actual = compiled(plan, k, *spectra, qp, qt, method=method)
    dense = apply_AP(k, kd, *spectra, qp, qt, method=method)
    for a, w, b in zip(actual, windows, dense):
        np.testing.assert_allclose(a, w @ b, rtol=0, atol=3e-6)
    # A different same-shaped plan must change the answer, not reuse frozen data.
    other, _, _ = make_window_plan(1.7)
    changed = compiled(other, k, *spectra, qp, qt, method=method)
    for a, b in zip(actual, changed):
        np.testing.assert_allclose(b, 1.7*a, rtol=1e-12, atol=1e-12)

    def loss(p, q):
        return sum(jnp.sum(x*x) for x in apply_AP_and_window(p, k, *spectra, q, qt, method=method))
    reverse = jax.jit(jax.grad(loss, argnums=1))(plan, qp)
    forward = jax.jacfwd(loss, argnums=1)(plan, qp)
    np.testing.assert_allclose(reverse, forward, rtol=1e-10, atol=1e-10)


def test_chebyshev_operator_dynamic_plan_and_polynomial_reference():
    x = jnp.linspace(-0.7, 1.2, 83)
    M = jnp.array(np.random.default_rng(6).normal(size=(9, 83)))
    op = prepare_chebyshev_operator(M, x, -0.7, 1.2, 12)
    f = lambda t: jnp.stack((1 + 2*t - t**3, t**4 - 0.7*t), axis=1)
    nodes = op.plan.nodes[0]
    compiled = jax.jit(apply_chebyshev_operator)
    result = compiled(op, f(nodes))
    np.testing.assert_allclose(result, M @ f(x), rtol=1e-11, atol=1e-11)
    other = prepare_chebyshev_operator(2*M, x, -0.7, 1.2, 12)
    np.testing.assert_allclose(compiled(other, f(nodes)), 2*result, rtol=1e-12, atol=1e-12)
    for values in (f(nodes), f(nodes)[:, 0]):
        loss = lambda v: jnp.sum(apply_chebyshev_operator(op, v)**2)
        np.testing.assert_allclose(jax.jit(jax.grad(loss))(values), jax.jacfwd(loss)(values), atol=2e-10)


def test_external_vmap_over_models():
    k, ko, vectors, _ = signals()
    def evaluate(q):
        return apply_AP(k, ko, *vectors, q[0], q[1])
    q = jnp.array([[1.02, 0.98], [0.97, 1.03], [1.04, 0.96]])
    actual = jax.jit(jax.vmap(evaluate))(q)
    expected = [evaluate(row) for row in q]
    for ell in range(3):
        np.testing.assert_allclose(actual[ell], jnp.stack([x[ell] for x in expected]), atol=1e-12)


def test_cosmology_to_ap_jit_reverse_and_redshift_batch():
    ref = w0waCDMCosmology(3.044, 0.9649, 0.6736, 0.02237, 0.12, m_nu=0.06)
    np.testing.assert_allclose(q_par_perp(0.8, ref, ref), [1, 1], atol=1e-14)

    def parameters(x, z):
        cosmo = w0waCDMCosmology(3.044, 0.9649, x[0], 0.02237, x[1],
                                m_nu=0.06, w0=x[2], wa=x[3])
        return jnp.stack(q_par_perp(z, cosmo, ref))

    x = jnp.array([0.68, 0.122, -0.95, 0.05])
    actual = jax.jit(parameters)(x, 0.8)
    np.testing.assert_allclose(actual, parameters(x, 0.8), atol=1e-12)
    reverse = jax.jit(jax.jacrev(parameters))(x, 0.8)
    forward = jax.jacfwd(parameters)(x, 0.8)
    assert np.all(np.isfinite(reverse))
    np.testing.assert_allclose(reverse, forward, rtol=1e-9, atol=1e-10)
    z = jnp.array([0.5, 0.8, 1.2])
    batched = jax.jit(jax.vmap(parameters, in_axes=(None, 0)))(x, z)
    np.testing.assert_allclose(batched, jnp.stack([parameters(x, zi) for zi in z]), atol=1e-11)

    k, ko, spectra, _ = signals()
    def loss(x):
        qp, qt = parameters(x, 0.8)
        return sum(jnp.sum(p*p) for p in apply_AP(k, ko, *spectra, qp, qt))
    np.testing.assert_allclose(jax.jit(jax.grad(loss))(x), jax.jacfwd(loss)(x), rtol=1e-9, atol=1e-9)


@pytest.mark.parametrize("method", ["Cubic", "Akima"])
@pytest.mark.parametrize("batch", [False, True])
def test_zero_and_flat_spectrum_gradients(method, batch):
    k = jnp.linspace(0.01, 1.0, 30)
    ko = jnp.linspace(0.03, 0.8, 19)
    for value in (0.0, 1.0):
        v = jnp.full((30, 3) if batch else (30,), value)
        def loss(v):
            out = apply_AP(k, ko, v, jnp.zeros_like(v), jnp.zeros_like(v),
                           1.02, 0.98, method=method)
            return sum(jnp.sum(x*x) for x in out)
        reverse = jax.jit(jax.grad(loss))(v)
        forward = jax.jacfwd(loss)(v)
        assert np.all(np.isfinite(reverse))
        np.testing.assert_allclose(reverse, forward, atol=1e-11)


@pytest.mark.parametrize("method", ["Cubic", "Akima"])
def test_slow_reference_matches_boundaries_and_extrapolation(method):
    k = jnp.linspace(0.01, 1.0, 30)
    ko = jnp.array([0.01, 0.015, 0.03, 0.9, 1.0])
    spectra = (jnp.sin(5*k), 0.5*jnp.cos(7*k), 0.1*jnp.sin(5*k))
    actual = apply_AP(k, ko, *spectra, 1.02, 0.98, n_GL_points=32, method=method)
    expected = apply_AP_check(k, ko, *spectra, 1.02, 0.98, method=method)
    for a, b in zip(actual, expected):
        assert np.all(np.isfinite(b))
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)


def test_full_multipole_mixing_window_and_matrix_jacobian():
    k, ko, spectra, _ = signals()
    W = jnp.array(np.random.default_rng(3).normal(size=(7, 3*len(ko))))
    def predict(amplitudes):
        projected = apply_AP(k, ko, *(a*s for a, s in zip(amplitudes, spectra)), 1.02, 0.98)
        return window_convolution(W, jnp.concatenate(projected))
    amplitudes = jnp.array([1.2, 0.7, -0.3])
    actual = jax.jit(predict)(amplitudes)
    separate = apply_AP(k, ko, *(a*s for a, s in zip(amplitudes, spectra)), 1.02, 0.98)
    np.testing.assert_allclose(actual, W @ jnp.concatenate(separate), atol=1e-12)
    components = tuple(s[:, None]*jnp.eye(3)[i] for i, s in enumerate(spectra))
    projected = apply_AP(k, ko, *components, 1.02, 0.98)
    matrix = window_convolution(W, jnp.concatenate(projected, axis=0))
    np.testing.assert_allclose(matrix, jax.jit(jax.jacrev(predict))(amplitudes), atol=1e-12)
