"""Actual trained models through AP, windows and projected bias Jacobians."""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxeffort
from tests.fixtures.sample_cosmologies import (
    get_test_biases,
    get_test_biases_velocileptors,
    get_test_cosmology_array,
)


@pytest.mark.parametrize("model", ["pybird_mnuw0wacdm", "velocileptors_lpt_mnuw0wacdm",
                                  "velocileptors_rept_mnuw0wacdm"])
def test_trained_model_projection_and_bias_jacobian(model):
    if any(jaxeffort.trained_emulators.get(model, {}).get(ell) is None for ell in ("0", "2", "4")):
        jaxeffort.reload_emulators(model)
    emulators = tuple(jaxeffort.trained_emulators[model][ell] for ell in ("0", "2", "4"))
    assert all(emu is not None for emu in emulators), f"Missing required reference artifact: {model}"
    # Historical PyBird artifacts store their grid as an (n_k, 1) column.
    k = emulators[0].P11.k_grid.reshape(-1)
    for emu in emulators:
        np.testing.assert_array_equal(emu.P11.k_grid.reshape(-1), k)
    kd = jnp.linspace(0.03, 0.2, 60)
    rng = np.random.default_rng(12)
    windows = tuple(jnp.array(rng.normal(size=(8, 60)))/60 for _ in range(3))
    plan = jaxeffort.prepare_ap_window_chebyshev(*windows, kd, 0.03, 0.2, 80)
    cosmo = get_test_cosmology_array()
    biases = get_test_biases() if model.startswith("pybird") else get_test_biases_velocileptors()
    D, qp, qt = 0.8, 1.02, 0.98

    def predict(c, b, growth, parallel, perpendicular, p):
        spectra = tuple(emu.get_Pl(c, b, growth) for emu in emulators)
        return jnp.concatenate(jaxeffort.apply_AP_and_window(p, k, *spectra, parallel, perpendicular))

    compiled = jax.jit(predict)
    actual = compiled(cosmo, biases, D, qp, qt, plan)
    spectra = tuple(emu.get_Pl(cosmo, biases, D) for emu in emulators)
    dense = jaxeffort.apply_AP(k, kd, *spectra, qp, qt)
    expected = jnp.concatenate(tuple(w @ s for w, s in zip(windows, dense)))
    # Compression error is measured separately from the exact AD comparison.
    relative = float(jnp.max(jnp.abs(actual - expected)) / jnp.max(jnp.abs(expected)))
    assert relative < 1e-4, f"{model} Chebyshev/dense error: {relative}"

    jacobians = tuple(emu.get_Pl_jacobian(cosmo, biases, D)[1] for emu in emulators)
    projected = jnp.concatenate(jaxeffort.apply_AP_and_window(plan, k, *jacobians, qp, qt), axis=0)
    reverse_jac = jax.jit(jax.jacrev(predict, argnums=1))(cosmo, biases, D, qp, qt, plan)
    np.testing.assert_allclose(projected, reverse_jac, rtol=2e-10, atol=2e-8)

    def loss(c, b, growth, parallel, perpendicular):
        return jnp.sum(predict(c, b, growth, parallel, perpendicular, plan)**2) * 1e-8
    reverse = jax.jit(jax.grad(loss, argnums=(0, 1, 2, 3, 4)))(cosmo, biases, D, qp, qt)
    forward = jax.jacfwd(loss, argnums=(0, 1, 2, 3, 4))(cosmo, biases, D, qp, qt)
    for a, b in zip(reverse, forward):
        assert np.all(np.isfinite(a))
        np.testing.assert_allclose(a, b, rtol=2e-8, atol=2e-8)

    # Differentiating the projected bias Jacobian is needed by parameter-dependent
    # Fisher/Jeffreys terms; primal-only AD does not exercise this path.
    def jacobian_loss(c, b):
        jac = tuple(emu.get_Pl_jacobian(c, b, D)[1] for emu in emulators)
        out = jaxeffort.apply_AP_and_window(plan, k, *jac, qp, qt)
        return sum(jnp.sum(x*x) for x in out) * 1e-8
    reverse = jax.jit(jax.grad(jacobian_loss, argnums=(0, 1)))(cosmo, biases)
    forward = jax.jacfwd(jacobian_loss, argnums=(0, 1))(cosmo, biases)
    for a, b in zip(reverse, forward):
        assert np.all(np.isfinite(a))
        np.testing.assert_allclose(a, b, rtol=2e-8, atol=2e-8)


@pytest.mark.parametrize("model,reference_name", [
    ("pybird_mnuw0wacdm", "pybird"),
    ("velocileptors_lpt_mnuw0wacdm", "lpt"),
    ("velocileptors_rept_mnuw0wacdm", "rept"),
])
def test_canonical_artifact_spectra_and_jacobians_against_julia(model, reference_name):
    if any(jaxeffort.trained_emulators.get(model, {}).get(ell) is None for ell in ("0", "2", "4")):
        jaxeffort.reload_emulators(model)
    reference = np.loadtxt(Path(__file__).parent / "reference_data" /
                           f"canonical_spectra_{reference_name}.txt")
    biases = jnp.array([2.0, -0.7, 0.3, -0.2, 0.05, 0.02, -0.01, 0.8, 0.9, -0.4, 0.2])
    for index, ell in enumerate(("0", "2", "4")):
        emu = jaxeffort.trained_emulators[model][ell]
        assert emu is not None, f"Missing required artifact {model}, ell={ell}"
        block = reference[:, 13*index:13*(index+1)]
        np.testing.assert_allclose(emu.P11.k_grid.reshape(-1), block[:, 0], atol=1e-15)
        value, jac = emu.get_Pl_jacobian(get_test_cosmology_array(), biases, 0.8)
        np.testing.assert_allclose(value, block[:, 1], rtol=2e-9, atol=2e-8)
        np.testing.assert_allclose(jac, block[:, 2:], rtol=2e-9, atol=2e-8)
