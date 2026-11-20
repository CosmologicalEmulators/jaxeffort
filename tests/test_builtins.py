"""Canonical config loading and direct Julia/AD parity for artifact functions."""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxeffort.builtins import (
    BUILTIN_BIAS_COMBINATIONS,
    BUILTIN_COMPONENT_POSTPROCESSING,
    BUILTIN_JAC_BIAS_COMBINATIONS,
    BUILTIN_STOCHMODELS,
)
from jaxeffort.jaxeffort import _load_builtin_or_file, load_multipole_emulator
from tests.fixtures.mock_emulator_data import (
    create_mock_emulator_directory, create_mock_weights, create_mock_minmax,
)


REFERENCE = Path(__file__).parent / "reference_data"
BIASES = jnp.array([2.0, -0.7, 0.3, -0.2, 0.05, 0.02, -0.01, 0.8, 0.9, -0.4, 0.2])


@pytest.mark.parametrize("name", BUILTIN_BIAS_COMBINATIONS)
def test_bias_functions_julia_and_ad(name):
    f, jac = BUILTIN_BIAS_COMBINATIONS[name], BUILTIN_JAC_BIAS_COMBINATIONS[name]
    expected = np.loadtxt(REFERENCE / f"canonical_{name}.txt")
    np.testing.assert_allclose(jax.jit(f)(BIASES), expected[:, 0], atol=2e-15)
    np.testing.assert_allclose(jax.jit(jac)(BIASES), expected[:, 1:], atol=2e-15)
    samples = jnp.stack((BIASES, jnp.linspace(-0.9, 1.1, 11), jnp.zeros(11)))
    actual = jax.jit(jax.vmap(jac))(samples)
    for ad in (jax.jacfwd(f), jax.jacrev(f)):
        np.testing.assert_allclose(actual, jax.jit(jax.vmap(ad))(samples), atol=2e-15)


@pytest.mark.parametrize("name", BUILTIN_STOCHMODELS)
def test_stochastic_julia_and_ad(name):
    reference = np.loadtxt(REFERENCE / "canonical_stochastic.txt")
    index = list(BUILTIN_STOCHMODELS).index(name)
    expected = reference[:, 1 + 3*index:4 + 3*index]
    f = BUILTIN_STOCHMODELS[name]
    for k in (jnp.array(reference[:, 0]), jnp.array(reference[:, :1])):
        np.testing.assert_allclose(jax.jit(f)(k), expected, atol=2e-15)
        loss = lambda x: jnp.sum(f(x)**2)
        np.testing.assert_allclose(jax.grad(loss)(k), jax.jacfwd(loss)(k), atol=2e-15)


@pytest.mark.parametrize("name,power", [("ln10As_growth_linear", 1), ("ln10As_growth_quadratic", 2)])
def test_postprocessing(name, power):
    inputs, output, D = jnp.array([0.8, 3.044]), jnp.arange(6.0), 0.8
    f = BUILTIN_COMPONENT_POSTPROCESSING[name]
    expected = np.array(output) * (np.exp(3.044)*1e-10*D**2)**power
    np.testing.assert_allclose(jax.jit(f)(inputs, output, D), expected, rtol=1e-14)
    loss = lambda x: jnp.sum(f(x, output, D))
    np.testing.assert_allclose(jax.grad(loss)(inputs), jax.jacfwd(loss)(inputs), atol=1e-25)


@pytest.mark.parametrize("nested", [False, True])
def test_config_only_loading_without_legacy_python(tmp_path, nested):
    path = Path(create_mock_emulator_directory(str(tmp_path)))
    config = {"stochmodel_name": "pybird_ell0", "biascombination_name": "pybird_eftoflss",
              "jacbiascombination_name": "pybird_eftoflss"}
    (path / "multipole_setup.json").write_text(json.dumps({"emulator_description": config} if nested else config))
    for directory, n_components in (("11", 3), ("loop", 12), ("ct", 6)):
        setup = path / directory / "nn_setup.json"
        data = json.loads(setup.read_text())
        # The historical generic mock uses incompatible 4/22/7 counts.
        # A PyBird coefficient contract requires 3/12/6 plus 3 stochastic terms.
        data["n_output_features"] = n_components * 50
        np.save(path / directory / "weights.npy", np.full_like(create_mock_weights(data), 0.01))
        np.save(path / directory / "outminmax.npy", create_mock_minmax(data["n_output_features"]))
        target = data["emulator_description"] if nested else data
        target["postprocessing_name"] = "ln10As_growth_linear"
        setup.write_text(json.dumps(data))
    for file in path.rglob("*.py"):
        file.unlink()
    emu = load_multipole_emulator(str(path))
    assert emu.bias_combination is BUILTIN_BIAS_COMBINATIONS["pybird_eftoflss"]
    assert emu.jacobian_bias_combination is BUILTIN_JAC_BIAS_COMBINATIONS["pybird_eftoflss"]
    assert emu.stoch_model is BUILTIN_STOCHMODELS["pybird_ell0"]
    c = jnp.array([1, 3.044, 0.9649, 67.36, 0.02237, 0.12, 0.06, -1, 0.])
    value, jac = emu.get_Pl_jacobian(c, BIASES, 0.8)
    np.testing.assert_allclose(value, emu.get_Pl(c, BIASES, 0.8), atol=1e-14)
    np.testing.assert_allclose(jac, jax.jacrev(lambda b: emu.get_Pl(c, b, 0.8))(BIASES), atol=1e-14)


def test_unknown_name_does_not_fall_back_and_flat_name_wins():
    def forbidden_loader(path):
        pytest.fail("Configured names must never fall back to executable files")
    registry = BUILTIN_BIAS_COMBINATIONS
    for name in ("typo", 42):
        with pytest.raises(ValueError, match="Unknown biascombination_name"):
            _load_builtin_or_file("unused", {"biascombination_name": name},
                                  "biascombination_name", registry, forbidden_loader)
    config = {"biascombination_name": "pybird_eftoflss",
              "emulator_description": {"biascombination_name": "typo"}}
    assert _load_builtin_or_file("unused", config, "biascombination_name", registry,
                                 forbidden_loader) is registry["pybird_eftoflss"]
