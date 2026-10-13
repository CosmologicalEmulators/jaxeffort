"""Stochastic parameters must stay dynamic through emulator -> AP -> window."""

import jax
import jax.numpy as jnp
import numpy as np

from jaxeffort import MultipoleEmulators, apply_AP_and_window, prepare_ap_window_chebyshev


class Component:
    """Small analytic component exercising the real public multipole methods."""

    def __init__(self, k, scale):
        self.k_grid, self.scale = k, scale

    def get_component(self, cosmology, D):
        return (D**2 * self.scale * (1 + self.k_grid*cosmology[0]))[:, None]


def test_stochastic_kwargs_value_jacobian_jit_vmap_and_reverse():
    k = jnp.linspace(0.01, 1.0, 40)
    def stochastic(k, *, amplitude=1.0, slope=0.3):
        return (amplitude + slope*k*k)[:, None]
    emu = MultipoleEmulators(*(Component(k, s) for s in (1.0, 0.3, 0.2)),
                            bias_combination=lambda b: b,
                            stoch_model=stochastic,
                            jacobian_bias_combination=lambda b: jnp.eye(4))
    cosmo, biases, D = jnp.array([0.4]), jnp.array([2.0, 0.5, 0.1, 0.7]), 0.8
    # Compile the default call first to detect cached/frozen keyword values.
    default = emu.get_Pl(cosmo, biases, D)
    explicit = emu.get_Pl(cosmo, biases, D, amplitude=1.0, slope=0.3)
    np.testing.assert_allclose(default, explicit, atol=1e-14)
    evaluate = jax.jit(lambda a: emu.get_Pl_jacobian(cosmo, biases, D, amplitude=a, slope=0.6))
    for amplitude in (1.2, 2.1):
        value, jac = evaluate(amplitude)
        expected_jac = jnp.stack(tuple(D**2*s*(1+k*cosmo[0]) for s in (1, 0.3, 0.2))
                                 + (amplitude + 0.6*k*k,), axis=1)
        np.testing.assert_allclose(jac, expected_jac, atol=1e-14)
        np.testing.assert_allclose(value, expected_jac @ biases, atol=1e-14)
        autodiff = jax.jacrev(lambda b: emu.get_Pl(cosmo, b, D, amplitude=amplitude, slope=0.6))(biases)
        np.testing.assert_allclose(jac, autodiff, atol=1e-14)

    kd = jnp.linspace(0.03, 0.8, 50)
    W = jnp.eye(7, 50)
    plan = prepare_ap_window_chebyshev(W, 0.5*W, 0.2*W, kd, 0.03, 0.8, 15)
    def loss(theta, p):
        spectrum = emu.get_Pl(theta[:1], biases, D, amplitude=theta[1], slope=theta[2])
        out = apply_AP_and_window(p, k, spectrum, 0.3*spectrum, 0.1*spectrum, theta[3], theta[4])
        return sum(jnp.sum(x*x) for x in out)
    theta = jnp.array([0.4, 1.2, 0.6, 1.02, 0.98])
    reverse = jax.jit(jax.grad(loss))(theta, plan)
    forward = jax.jacfwd(loss)(theta, plan)
    assert np.all(np.isfinite(reverse))
    assert np.all(np.abs(reverse) > 1e-8)
    np.testing.assert_allclose(reverse, forward, rtol=2e-11, atol=2e-11)
    batch = jnp.stack((theta, theta.at[1].set(2.1)))
    actual = jax.jit(jax.vmap(loss, in_axes=(0, None)))(batch, plan)
    np.testing.assert_allclose(actual, jnp.array([loss(t, plan) for t in batch]), atol=1e-12)
