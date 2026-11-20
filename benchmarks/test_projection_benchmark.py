"""Synchronized CPU/GPU benchmarks of emulator -> AP -> window and its gradient.

Run explicitly (not part of the correctness test suite):
  .venv/bin/python -m pip install pytest-benchmark
  JAX_PLATFORMS=cpu taskset -c 2 .venv/bin/python -m pytest benchmarks/ \
      --benchmark-save=projection

For matched before/after measurements, set JAXEFFORT_BENCH_BASELINE to a saved
pre-change projection.py. Both implementations are measured in the same process.
No compilation or plan preparation is included in steady-state measurements.
"""

import importlib.util
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxeffort
from jaxeffort import projection
from tests.fixtures.sample_cosmologies import get_test_biases, get_test_cosmology_array


MODULES = {"current": projection}
if os.environ.get("JAXEFFORT_BENCH_BASELINE"):
    spec = importlib.util.spec_from_file_location("projection_baseline", os.environ["JAXEFFORT_BENCH_BASELINE"])
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    # The old root solver has a known ~1e-9 quadrature error. Use the corrected
    # cached nodes in both modules so performance compares the same operator.
    # Node generation is host-side setup and is excluded from all hot timings.
    baseline.gausslobatto = projection.gausslobatto
    MODULES = {"baseline": baseline, **MODULES}


@pytest.fixture(scope="module")
def pipelines():
    emus = [jaxeffort.trained_emulators["pybird_mnuw0wacdm"][ell] for ell in ("0", "2", "4")]
    assert all(e is not None for e in emus)
    k = emus[0].P11.k_grid.reshape(-1)
    kd = jnp.linspace(0.03, 0.2, 200)
    W = tuple(jnp.array(np.random.default_rng(i).normal(size=(40, 200)))/200 for i in range(3))
    biases = get_test_biases()
    theta = jnp.concatenate((get_test_cosmology_array(), jnp.array([0.8, 1.02, 0.98])))
    result = {}
    for name, module in MODULES.items():
        plan = module.prepare_ap_window_chebyshev(*W, kd, 0.03, 0.2, 80)
        def forward(x, module=module, plan=plan):
            spectra = tuple(e.get_Pl(x[:9], biases, x[9]) for e in emus)
            return jnp.concatenate(module.apply_AP_and_window(plan, k, *spectra, x[10], x[11]))
        def loss(x, forward=forward):
            return jnp.sum(forward(x)**2) * 1e-8
        result[name] = (jax.jit(forward), jax.jit(jax.value_and_grad(loss)))
    # Compile both implementations before timing either; verify the same workload.
    for functions in result.values():
        for fn in functions:
            jax.block_until_ready(fn(theta))
    if "baseline" in result:
        for old, new in zip(result["baseline"], result["current"]):
            for a, b in zip(jax.tree_util.tree_leaves(old(theta)), jax.tree_util.tree_leaves(new(theta))):
                np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-9)
    # Compilation heats the host. Let it cool before matched steady-state runs.
    time.sleep(10)
    return result, theta


@pytest.mark.parametrize("mode", ["primal", "value_and_gradient"])
@pytest.mark.parametrize("implementation", MODULES)
def test_emulator_ap_window(benchmark, pipelines, mode, implementation):
    functions, theta = pipelines
    fn = functions[implementation][0 if mode == "primal" else 1]
    def run():
        return jax.block_until_ready(fn(theta))
    benchmark.extra_info.update({"device": str(jax.devices()[0]), "dtype": str(theta.dtype),
                                 "implementation": implementation, "mode": mode})
    benchmark.pedantic(run, rounds=100, iterations=1, warmup_rounds=10)
