"""Canonical artifact functions matching Effort.jl reactant_dev (ea985a7)."""

import jax.numpy as jnp


def ln10As_growth_linear(inputs, output, D):
    amplitude = jnp.exp(inputs[1]) * 1e-10 * D**2
    return output * amplitude


def ln10As_growth_quadratic(inputs, output, D):
    amplitude = jnp.exp(inputs[1]) * 1e-10 * D**2
    return output * amplitude**2


def pybird_eftoflss_bias_combination(b):
    b1, b2, b3, b4, b5, b6, b7, f, ce0, ce1, ce2 = b
    return jnp.array([b1*b1, 2*b1*f, f*f, 1, b1, b2, b3, b4,
                      b1*b1, b1*b2, b1*b3, b1*b4, b2*b2, b2*b4, b4*b4,
                      2*b1*b5, 2*b1*b6, 2*b1*b7, 2*f*b5, 2*f*b6, 2*f*b7,
                      ce0, ce1, ce2*f])


def velocileptors_bias_combination(b):
    b1, b2, bs, b3, a0, a2, a4, a6, sn, sn2, sn4 = b
    return jnp.array([1, b1, b1*b1, b2, b1*b2, b2*b2, bs, b1*bs,
                      b2*bs, bs*bs, b3, b1*b3, a0, a2, a4, a6, sn, sn2, sn4])


def pybird_eftoflss_jacobian_bias_combination(b):
    b1, b2, b3, b4, b5, b6, b7, f, _, _, ce2 = b
    z = jnp.zeros_like(b1)
    o = jnp.ones_like(b1)
    return jnp.array([
        [2*b1, 2*f, z, z, o, z, z, z, 2*b1, b2, b3, b4, z, z, z, 2*b5, 2*b6, 2*b7, z, z, z, z, z, z],
        [z, z, z, z, z, o, z, z, z, b1, z, z, 2*b2, b4, z, z, z, z, z, z, z, z, z, z],
        [z, z, z, z, z, z, o, z, z, z, b1, z, z, z, z, z, z, z, z, z, z, z, z, z],
        [z, z, z, z, z, z, z, o, z, z, z, b1, z, b2, 2*b4, z, z, z, z, z, z, z, z, z],
        [z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, 2*b1, z, z, 2*f, z, z, z, z, z],
        [z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, 2*b1, z, z, 2*f, z, z, z, z],
        [z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, 2*b1, z, z, 2*f, z, z, z],
        [z, 2*b1, 2*f, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, 2*b5, 2*b6, 2*b7, z, z, ce2],
        [z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, o, z, z],
        [z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, o, z],
        [z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, z, f],
    ]).T


def velocileptors_jacobian_bias_combination(b):
    b1, b2, bs, b3 = b[:4]
    z, o = jnp.zeros_like(b1), jnp.ones_like(b1)
    nonlinear = jnp.array([
        [z, o, 2*b1, z, b2, z, z, bs, z, z, z, b3],
        [z, z, z, o, b1, 2*b2, z, z, bs, z, z, z],
        [z, z, z, z, z, z, o, b1, b2, 2*bs, z, z],
        [z, z, z, z, z, z, z, z, z, z, o, b1],
    ]).T
    return jnp.zeros((19, 11), dtype=nonlinear.dtype).at[:12, :4].set(nonlinear).at[12:, 4:].set(jnp.eye(7))


def pybird_ell0(k):
    k = jnp.asarray(k).reshape(-1)
    k2 = k*k / 0.49
    return jnp.stack((jnp.ones_like(k), k2, k2/3), axis=1)


def pybird_ell2(k):
    k = jnp.asarray(k).reshape(-1)
    z = jnp.zeros_like(k)
    return jnp.stack((z, z, k*k/0.49 * (2/3)), axis=1)


def pybird_ell4(k):
    k = jnp.asarray(k).reshape(-1)
    return jnp.zeros((k.size, 3), dtype=k.dtype)


def velocileptors_ell0(k):
    k = jnp.asarray(k).reshape(-1)
    return jnp.stack((jnp.ones_like(k), k*k/3, k**4/5), axis=1)


def velocileptors_ell2(k):
    k = jnp.asarray(k).reshape(-1)
    return jnp.stack((jnp.zeros_like(k), 2*k*k/3, 4*k**4/7), axis=1)


def velocileptors_ell4(k):
    k = jnp.asarray(k).reshape(-1)
    z = jnp.zeros_like(k)
    return jnp.stack((z, z, 8*k**4/35), axis=1)


BUILTIN_COMPONENT_POSTPROCESSING = {
    "ln10As_growth_linear": ln10As_growth_linear,
    "ln10As_growth_quadratic": ln10As_growth_quadratic,
}
BUILTIN_BIAS_COMBINATIONS = {
    "pybird_eftoflss": pybird_eftoflss_bias_combination,
    "velocileptors": velocileptors_bias_combination,
}
BUILTIN_JAC_BIAS_COMBINATIONS = {
    "pybird_eftoflss": pybird_eftoflss_jacobian_bias_combination,
    "velocileptors": velocileptors_jacobian_bias_combination,
}
BUILTIN_STOCHMODELS = {
    "pybird_ell0": pybird_ell0, "pybird_ell2": pybird_ell2, "pybird_ell4": pybird_ell4,
    "velocileptors_ell0": velocileptors_ell0,
    "velocileptors_ell2": velocileptors_ell2,
    "velocileptors_ell4": velocileptors_ell4,
}
