"""Sample cosmologies and parameters for testing."""

import jax.numpy as jnp
from jaxace import W0WaCDMCosmology


def get_test_cosmology():
    """Get a standard test cosmology."""
    return W0WaCDMCosmology(
        ln10As=3.044,
        ns=0.9649,
        h=0.6736,
        omega_b=0.02237,
        omega_c=0.1200,
        m_nu=0.06,
        w0=-1.0,
        wa=0.0
    )


def get_test_cosmology_array():
    """Get cosmology parameters as array for emulator input."""
    # Parameters: z, ln10^10 As, ns, H0, omega_b, omega_c, Mnu, w0, wa
    # This matches the expected input format for the emulators
    return jnp.array([
        1.0,     # z - redshift
        3.044,   # ln10As
        0.9649,  # ns
        67.36,   # H0 (h * 100)
        0.02237, # omega_b
        0.1200,  # omega_c
        0.06,    # m_nu
        -1.0,    # w0
        0.0      # wa
    ])


def get_test_biases():
    """Get standard galaxy bias parameters for testing."""
    # Standard bias parameters [b1, b2, bs2, b3nl]
    return jnp.array([1.0, 0.0, 0.0, 0.0])


def get_test_redshifts():
    """Get standard redshift array for testing."""
    return jnp.array([0.0, 0.5, 1.0, 1.5, 2.0])