import jax
import jax.numpy as jnp
from functools import partial
from scipy.integrate import quad as scipy_quad
from jaxace.utils import cubic_spline_interpolation, akima_interpolation
from jaxace.chebyshev import (
    ChebyshevPlan,
    chebyshev_polynomials,
    prepare_chebyshev_plan,
    chebyshev_decomposition,
)
from jaxace.background import E_z, dA_z, d̃A_z


import numpy as np


from functools import lru_cache
from math import gamma

def _innerjacobi_rec(n, x, alpha, beta):
    """
    Evaluate the Jacobi polynomial P^{(alpha,beta)}_n and its derivative.
    Direct port of innerjacobi_rec! from FastGaussQuadrature.jl.
    """
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    P  = np.empty(N)
    PP = np.empty(N)

    for j in range(N):
        xj = x[j]
        Pj   = (alpha - beta + (alpha + beta + 2.0) * xj) / 2.0
        Pm1  = 1.0
        PPj  = (alpha + beta + 2.0) / 2.0
        PPm1 = 0.0

        for k in range(1, n):
            k0 = 2.0 * k + alpha + beta
            k1 = k0 + 1.0
            k2 = k0 + 2.0
            A  = 2.0 * (k + 1.0) * (k + alpha + beta + 1.0) * k0
            B  = k1 * (alpha**2 - beta**2)
            C  = k0 * k1 * k2
            D  = 2.0 * (k + alpha) * (k + beta) * k2
            c1 = C * xj + B

            Pm1, Pj   = Pj,  (c1 * Pj  - D * Pm1) / A
            PPm1, PPj = PPj, (c1 * PPj - D * PPm1 + C * Pm1) / A

        P[j]  = Pj
        PP[j] = PPj

    return P, PP


def _halfrec(n, alpha, beta, flag):
    """
    Compute half of the Gauss-Jacobi nodes and their derivatives.
    Direct port of HalfRec from FastGaussQuadrature.jl.
    """
    if flag == 1:
        m = int(np.ceil(n / 2))
    else:
        m = int(np.floor(n / 2))

    if m == 0:
        return np.array([]), np.array([])

    r = np.arange(m, 0, -1, dtype=np.float64)

    c1   = 1.0 / (2.0 * n + alpha + beta + 1.0)
    a1   = 0.25 - alpha**2
    b1   = 0.25 - beta**2
    c1sq = c1**2

    C   = (2.0 * r + alpha - 0.5) * (np.pi * c1)
    C_2 = C / 2.0
    x   = np.cos(C + c1sq * (a1 / np.tan(C_2) - b1 * np.tan(C_2)))

    P  = np.empty(m)
    PP = np.empty(m)
    for _ in range(10):
        P, PP = _innerjacobi_rec(n, x, alpha, beta)
        dx = P / PP
        x -= dx
        if np.max(dx**2) <= np.finfo(np.float64).eps / 1e-6:
            break

    P, PP = _innerjacobi_rec(n, x, alpha, beta)
    return x, PP


def gaussjacobi(n, alpha, beta):
    """
    Compute Gauss-Jacobi nodes and weights on [-1, 1].
    Direct port of jacobi_rec from FastGaussQuadrature.jl.
    """
    if n == 0:
        return np.array([]), np.array([])
    if n == 1:
        x0 = (beta - alpha) / (alpha + beta + 2.0)
        w0 = (2.0**(alpha + beta + 1.0)
              * gamma(alpha + 2.0) * gamma(beta + 2.0)
              / gamma(alpha + beta + 2.0)
              / (alpha + 1.0) / (beta + 1.0))
        return np.array([x0]), np.array([w0])

    x11, x12 = _halfrec(n, alpha, beta, 1)
    x21, x22 = _halfrec(n, beta, alpha, 0)

    m1 = len(x11)
    m2 = len(x21)
    total = m1 + m2

    x_all = np.empty(total)
    w_all = np.empty(total)
    sum_w = 0.0

    for i in range(m2):
        idx = m2 - 1 - i
        xi  = -x21[i]
        der = x22[i]
        wi  = 1.0 / ((1.0 - xi**2) * der**2)
        x_all[idx] = xi
        w_all[idx] = wi
        sum_w += wi

    for i in range(m1):
        idx = m2 + i
        xi  = x11[i]
        der = x12[i]
        wi  = 1.0 / ((1.0 - xi**2) * der**2)
        x_all[idx] = xi
        w_all[idx] = wi
        sum_w += wi

    c = (2.0**(alpha + beta + 1.0)
         * gamma(2.0 + alpha) * gamma(2.0 + beta)
         / gamma(2.0 + alpha + beta)
         / (alpha + 1.0) / (beta + 1.0))

    w_all *= c / sum_w
    return x_all, w_all


@lru_cache(maxsize=None)
def gausslobatto(n: int):
    """
    Compute the n-point Gauss-Lobatto-Legendre nodes and weights on [-1, 1].

    Matches Julia's FastGaussQuadrature.gausslobatto(n).
    Results are cached (via lru_cache) because `apply_AP` calls this repeatedly
    with the same small constant (e.g., n=16).

    Args:
        n: Number of quadrature points (must satisfy n >= 2).

    Returns:
        (nodes, weights): Two numpy arrays of length n, both 64-bit floats.
                          The returned arrays are read-only to protect the cache.
    """
    if n < 2:
        raise ValueError("gausslobatto requires n >= 2")
    if n == 2:
        nodes   = np.array([-1.0, 1.0])
        weights = np.array([1.0, 1.0])
    elif n == 3:
        nodes   = np.array([-1.0, 0.0, 1.0])
        weights = np.array([1.0/3, 4.0/3, 1.0/3])
    else:
        x, w = gaussjacobi(n - 2, 1.0, 1.0)
        w = w / (1.0 - x**2)
        w_end = 2.0 / (n * (n - 1))
        nodes   = np.concatenate([[-1.0], x,  [1.0]])
        weights = np.concatenate([[w_end], w, [w_end]])

    # Protect the cached numpy arrays from accidental mutation by the caller
    nodes.setflags(write=False)
    weights.setflags(write=False)

    return nodes, weights




# =============================================================================
# Legendre Polynomials
# =============================================================================
def _Legendre_0(mu):
    return jnp.ones_like(mu)

def _Legendre_2(mu):
    return 0.5 * (3 * mu**2 - 1)

def _Legendre_4(mu):
    return 0.125 * (35 * mu**4 - 30 * mu**2 + 3)


# =============================================================================
# Coordinate Maps
# =============================================================================
def _k_true(k_o, mu_o, q_perp, F):
    """
    Calculates the true (physical) wavenumber `k` from the observed wavenumber `k_o`
    and observed cosine of the angle to the line-of-sight `μ_o`.
    """
    a = jnp.sqrt(1 + mu_o**2 * (1 / F**2 - 1))
    # In Julia: result = (k_o ./ q_perp) * a'
    # Julia flattens column-major (Fortran order). To preserve exact evaluation
    # alignment inside jaxace spline evaluations, we mimic the strict array structures.
    # We return the 2D grid [N_k, N_μ] directly.
    return jnp.outer(k_o / q_perp, a)


def _mu_true(mu_o, F):
    """
    Calculates the true (physical) cosine of the angle to the line-of-sight `μ` from the
    observed cosine of the angle to the line-of-sight `μ_o`.
    """
    a = 1 / jnp.sqrt(1 + mu_o**2 * (1 / F**2 - 1))
    return (mu_o / F) * a


# =============================================================================
# AP Parameters Calculation
# =============================================================================
def q_par_perp(z, cosmo_mcmc, cosmo_ref):
    """
    Calculates the parallel (q_par) and perpendicular (q_perp) Alcock-Paczynski (AP)
    parameters at a given redshift `z`.

    Matches Effort.jl's q_par_perp which uses the **conformal** angular diameter
    distance d̃A (i.e. the comoving angular diameter distance), not the physical dA.
    """
    E_ref = E_z(z, cosmo_ref)
    E_mcmc = E_z(z, cosmo_mcmc)

    # Use conformal (comoving) angular diameter distance to match Julia's d̃A_z
    dA_tilde_ref = d̃A_z(z, cosmo_ref)
    dA_tilde_mcmc = d̃A_z(z, cosmo_mcmc)

    q_perp = dA_tilde_mcmc / dA_tilde_ref
    q_par = E_ref / E_mcmc

    return q_par, q_perp


# =============================================================================
# Interpolates
# =============================================================================
def _interpolate_multipoles(method, k_input, k_t, mono, quad, hexa):
    if method == "Cubic":
        new_mono_flat = cubic_spline_interpolation(mono, k_input, k_t.flatten())
        new_quad_flat = cubic_spline_interpolation(quad, k_input, k_t.flatten())
        new_hexa_flat = cubic_spline_interpolation(hexa, k_input, k_t.flatten())
    elif method == "Akima":
        new_mono_flat = akima_interpolation(mono, k_input, k_t.flatten())
        new_quad_flat = akima_interpolation(quad, k_input, k_t.flatten())
        new_hexa_flat = akima_interpolation(hexa, k_input, k_t.flatten())
    else:
        raise ValueError(f"Unknown interpolation method: {method}")

    return new_mono_flat, new_quad_flat, new_hexa_flat


def _Pk_recon(mono, quad, hexa, l0, l2, l4):
    """Reconstructs the 2D power spectrum P(k, μ)."""
    # Equivalent to mono .* l0' .+ quad .* l2' .+ hexa .* l4' in Julia
    return mono * l0[None, :] + quad * l2[None, :] + hexa * l4[None, :]


# =============================================================================
# Alcock-Paczynski Projector
# =============================================================================
@partial(jax.jit, static_argnames=("n_GL_points", "method"))
def apply_AP(k_input, k_output, mono, quad, hexa, q_par, q_perp, n_GL_points=8, method="Cubic"):
    """
    Calculates the observed power spectrum multipole moments (monopole, quadrupole, hexadecapole)
    on a given observed wavenumber grid `k_output` leveraging Gauss-Lobatto quadrature integrals.
    """
    nk = len(k_output)

    # Compute nodes and weights for 2*n_GL_points Gauss-Lobatto quadrature
    # Since the integrand is symmetric we use only the first n_GL_points (negative half)
    nodes, weights = gausslobatto(n_GL_points * 2)
    mu_nodes = jnp.array(nodes[:n_GL_points])
    mu_weights = jnp.array(weights[:n_GL_points])
    F = q_par / q_perp

    # Compute true k and μ values
    k_t = _k_true(k_output, mu_nodes, q_perp, F)
    mu_t = _mu_true(mu_nodes, F)

    # Legendre polynomial maps
    Pl0_t = _Legendre_0(mu_t)
    Pl2_t = _Legendre_2(mu_t)
    Pl4_t = _Legendre_4(mu_t)

    # Gauss-Lobatto projection weights mapping ℓ multipoles correctly scaling
    # (2ℓ + 1) explicitly factored over the quadrature summation weights.
    Pl0 = _Legendre_0(mu_nodes) * mu_weights * (2 * 0 + 1)
    Pl2 = _Legendre_2(mu_nodes) * mu_weights * (2 * 2 + 1)
    Pl4 = _Legendre_4(mu_nodes) * mu_weights * (2 * 4 + 1)

    new_mono_flat, new_quad_flat, new_hexa_flat = _interpolate_multipoles(
        method, k_input, k_t.flatten("F"), mono, quad, hexa
    )

    # Verify if single component arrays or multiple component matrixes
    is_multi_column = jnp.ndim(mono) > 1

    if is_multi_column:
        n_cols = mono.shape[1]

        # Reshape matching Fortran-order flattening: (nk, n_GL_points, n_cols)
        new_mono = new_mono_flat.reshape((nk, n_GL_points, n_cols), order="F")
        new_quad = new_quad_flat.reshape((nk, n_GL_points, n_cols), order="F")
        new_hexa = new_hexa_flat.reshape((nk, n_GL_points, n_cols), order="F")

        # JAX mapping to perform the integral dot products across all columns correctly
        def process_column(col):
            Pkmu = _Pk_recon(new_mono[:, :, col], new_quad[:, :, col], new_hexa[:, :, col], Pl0_t, Pl2_t, Pl4_t) / (q_par * q_perp**2)
            return Pkmu @ Pl0, Pkmu @ Pl2, Pkmu @ Pl4

        mono_cols, quad_cols, hexa_cols = jax.vmap(process_column, out_axes=1)(jnp.arange(n_cols))
        return mono_cols, quad_cols, hexa_cols

    else:
        new_mono = new_mono_flat.reshape((nk, n_GL_points), order="F")
        new_quad = new_quad_flat.reshape((nk, n_GL_points), order="F")
        new_hexa = new_hexa_flat.reshape((nk, n_GL_points), order="F")

        Pkmu = _Pk_recon(new_mono, new_quad, new_hexa, Pl0_t, Pl2_t, Pl4_t) / (q_par * q_perp**2)

        # Dot product over quadrature weights mapping integral
        return Pkmu @ Pl0, Pkmu @ Pl2, Pkmu @ Pl4


# =============================================================================
# Window Convolution
# =============================================================================
@jax.jit
def window_convolution(W, v):
    """
    Performs matrix-vector multiplication, where the matrix `W` acts as a linear
    transformation or window applied to the vector input `v`.
    """
    return W @ v


# =============================================================================
# Reference AP Projection (slow, for validation)
# =============================================================================

def _P_obs_scalar(k_o, mu_o, q_par, q_perp, int_mono, int_quad, int_hexa):
    """
    Compute the observed power spectrum P_obs(k_o, mu_o) for scalar k_o/mu_o.
    Matches Julia's _P_obs function.
    """
    import numpy as np
    F = q_par / q_perp
    k_t = k_o / q_perp * np.sqrt(1 + mu_o**2 * (1 / F**2 - 1))
    mu_t = mu_o / F / np.sqrt(1 + mu_o**2 * (1 / F**2 - 1))
    P_kmu = (int_mono(k_t) + int_quad(k_t) * 0.5 * (3 * mu_t**2 - 1)
             + int_hexa(k_t) * 0.125 * (35 * mu_t**4 - 30 * mu_t**2 + 3))
    return P_kmu / (q_par * q_perp**2)


def apply_AP_check(
    k_input,
    k_output,
    mono,
    quad,
    hexa,
    q_par,
    q_perp,
    method="Cubic",
    reltol=1e-12,
):
    """
    Reference (slow) implementation of the Alcock-Paczynski projection.

    Numerically integrates P_obs(k_o, mu_o) * L_ℓ(mu_o) over mu_o ∈ [0, 1]
    using adaptive Gauss-Kronrod (scipy.integrate.quad), matching Julia's
    QuadGKJL with reltol=1e-12.

    This is equivalent to Effort.jl's apply_AP_check and is intended for
    validation of the fast Gauss-Lobatto apply_AP path.

    Args:
        k_input: Input wavenumber grid (numpy array).
        k_output: Output wavenumber grid (numpy array).
        mono: True monopole values on k_input.
        quad: True quadrupole values on k_input.
        hexa: True hexadecapole values on k_input.
        q_par: Parallel AP parameter.
        q_perp: Perpendicular AP parameter.
        method: Interpolation method, "Cubic" or "Akima".
        reltol: Relative tolerance for the quadrature. Default: 1e-12.

    Returns:
        Tuple (P0_obs, P2_obs, P4_obs) of numpy arrays on k_output.
    """
    import numpy as np
    from scipy.interpolate import CubicSpline, Akima1DInterpolator

    k_input = np.asarray(k_input)
    k_output = np.asarray(k_output)
    mono = np.asarray(mono)
    quad = np.asarray(quad)
    hexa = np.asarray(hexa)

    if method == "Cubic":
        int_mono = CubicSpline(k_input, mono, extrapolate=True)
        int_quad = CubicSpline(k_input, quad, extrapolate=True)
        int_hexa = CubicSpline(k_input, hexa, extrapolate=True)
    elif method == "Akima":
        int_mono = Akima1DInterpolator(k_input, mono)
        int_quad = Akima1DInterpolator(k_input, quad)
        int_hexa = Akima1DInterpolator(k_input, hexa)
    else:
        raise ValueError(f"Unknown method: {method}")

    nk = len(k_output)
    L = [(0, lambda mu: 1.0),
         (2, lambda mu: 0.5 * (3 * mu**2 - 1)),
         (4, lambda mu: 0.125 * (35 * mu**4 - 30 * mu**2 + 3))]

    result = np.zeros((3, nk))
    for i, k_o in enumerate(k_output):
        for l_idx, (ell, Pl) in enumerate(L):
            integrand = lambda mu_o: (
                Pl(mu_o) * _P_obs_scalar(k_o, mu_o, q_par, q_perp,
                                         int_mono, int_quad, int_hexa)
            )
            val, _ = scipy_quad(integrand, 0.0, 1.0, limit=200, epsrel=reltol)
            result[l_idx, i] = (2 * ell + 1) * val

    return result[0], result[1], result[2]


# =============================================================================
# Chebyshev-Optimized Operators (matching Effort.jl)
# =============================================================================

class ChebyshevOperator:
    """
    A linear operator M compressed using Chebyshev decomposition.

    Matches Effort.jl's ChebyshevOperator struct.

    The idea is to precompute M_prime = M @ T, where T is the Chebyshev
    polynomial matrix evaluated on the dense output grid. At evaluation time,
    only a fast matrix multiply against Chebyshev coefficients is needed.

    Attributes:
        M_prime: The transformed operator M @ T_mat (shape: n_out × (K+1)).
        plan: The ChebyshevPlan used for decomposition.
    """
    def __init__(self, M_prime, plan: ChebyshevPlan):
        self.M_prime = M_prime
        self.plan = plan


def prepare_chebyshev_operator(
    M: jnp.ndarray,
    x_grid: jnp.ndarray,
    x_min: float,
    x_max: float,
    K: int,
) -> ChebyshevOperator:
    """
    Precomputes a ChebyshevOperator by projecting the matrix M onto the Chebyshev basis.

    Matches Effort.jl's prepare_chebyshev_operator.

    Args:
        M: The linear operator matrix of shape (n_out, n_x).
        x_grid: Dense x grid of length n_x where M is evaluated.
        x_min: Domain minimum for the Chebyshev basis.
        x_max: Domain maximum for the Chebyshev basis.
        K: Polynomial degree (K+1 Chebyshev nodes).

    Returns:
        ChebyshevOperator with precomputed M_prime = M @ T_mat.
    """
    T_mat = chebyshev_polynomials(x_grid, x_min, x_max, K)  # (n_x, K+1)
    M_prime = M @ T_mat                                       # (n_out, K+1)
    plan = prepare_chebyshev_plan(x_min, x_max, K)
    return ChebyshevOperator(M_prime, plan)


def apply_chebyshev_operator(
    op: ChebyshevOperator,
    v_nodes: jnp.ndarray,
) -> jnp.ndarray:
    """
    Applies the compressed ChebyshevOperator to function values at Chebyshev nodes.

    Matches Effort.jl's apply_chebyshev_operator. Supports both vector (single
    evaluation) and matrix (batched) inputs.

    Args:
        op: A ChebyshevOperator returned by prepare_chebyshev_operator.
        v_nodes: Function values evaluated exactly at op.plan.nodes.
                 Shape: (K+1,) for a vector, or (K+1, n_cols) for batched.

    Returns:
        Result of M @ v, computed as M_prime @ chebyshev_decomposition(plan, v_nodes).
    """
    c = chebyshev_decomposition(op.plan, v_nodes)
    return op.M_prime @ c


class APWindowChebyshevPlan:
    """
    A plan for combining AP effect and window function convolution via Chebyshev.

    Matches Effort.jl's APWindowChebyshevPlan struct.

    Precomputes the AP+window operators offline on a sparse Chebyshev k-grid,
    so that at run-time only a AP evaluation on K+1 grid points and a single
    matrix multiply per multipole is needed.

    Attributes:
        M0, M2, M4: Precomputed window operators (shape: n_k_out × (K+1)) for
                    each multipole ℓ ∈ {0, 2, 4}.
        decomp_plan: ChebyshevPlan for the sparse Chebyshev k-grid.
        sparse_k_nodes: The K+1 Chebyshev k-node values.
        k_min, k_max: Chebyshev domain bounds.
        K: Polynomial degree.
    """
    def __init__(self, M0, M2, M4, decomp_plan, sparse_k_nodes, k_min, k_max, K):
        self.M0 = M0
        self.M2 = M2
        self.M4 = M4
        self.decomp_plan = decomp_plan
        self.sparse_k_nodes = sparse_k_nodes
        self.k_min = k_min
        self.k_max = k_max
        self.K = K


def prepare_ap_window_chebyshev(
    W0: jnp.ndarray,
    W2: jnp.ndarray,
    W4: jnp.ndarray,
    k_dense: jnp.ndarray,
    k_min: float,
    k_max: float,
    K: int,
) -> APWindowChebyshevPlan:
    """
    Precomputes the AP+window operators for combined application.

    Matches Effort.jl's prepare_ap_window_chebyshev.

    Args:
        W0, W2, W4: Window matrices for multipoles ℓ ∈ {0, 2, 4},
                    each of shape (n_k_out, n_k_dense).
        k_dense: Dense wavenumber grid on which W is evaluated.
        k_min: Minimum k for Chebyshev domain.
        k_max: Maximum k for Chebyshev domain.
        K: Polynomial degree.

    Returns:
        APWindowChebyshevPlan ready for use with apply_AP_and_window.
    """
    T_mat = chebyshev_polynomials(k_dense, k_min, k_max, K)  # (n_dense, K+1)
    M0 = W0 @ T_mat
    M2 = W2 @ T_mat
    M4 = W4 @ T_mat
    decomp_plan = prepare_chebyshev_plan(k_min, k_max, K)
    sparse_k_nodes = decomp_plan.nodes[0]  # The K+1 Chebyshev nodes
    return APWindowChebyshevPlan(
        M0=M0, M2=M2, M4=M4,
        decomp_plan=decomp_plan,
        sparse_k_nodes=sparse_k_nodes,
        k_min=k_min, k_max=k_max, K=K,
    )


def apply_AP_and_window(
    plan: APWindowChebyshevPlan,
    k_input,
    mono_in,
    quad_in,
    hexa_in,
    q_par,
    q_perp,
    n_GL_points: int = 8,
    method: str = "Cubic",
):
    """
    Combined application of Alcock-Paczynski effect and window function convolution.

    Matches Effort.jl's apply_AP_and_window. Supports both single-model
    (1D) and batched (2D) inputs.

    Steps:
      1. Apply AP correction on the sparse Chebyshev nodes.
      2. Decompose each multipole into Chebyshev coefficients.
      3. Apply the precomputed window operators via a matrix multiply.

    Args:
        plan: APWindowChebyshevPlan from prepare_ap_window_chebyshev.
        k_input: Input wavenumber grid for the multipole arrays.
        mono_in: Monopole values on k_input (vector or matrix).
        quad_in: Quadrupole values on k_input.
        hexa_in: Hexadecapole values on k_input.
        q_par: Parallel AP parameter (scalar or vector for batching).
        q_perp: Perpendicular AP parameter (scalar or vector for batching).
        n_GL_points: Number of Gauss-Lobatto points. Default: 8.
        method: Interpolation method, "Cubic" or "Akima".

    Returns:
        Tuple (p0_conv, p2_conv, p4_conv) of window-convolved multipoles.
    """
    # Step 1: Apply AP on the sparse Chebyshev nodes
    mono_AP, quad_AP, hexa_AP = apply_AP(
        k_input, plan.sparse_k_nodes,
        mono_in, quad_in, hexa_in,
        q_par, q_perp,
        n_GL_points=n_GL_points,
        method=method,
    )

    # Step 2: Chebyshev decomposition on the AP output (on the K+1 nodes)
    c0 = chebyshev_decomposition(plan.decomp_plan, mono_AP)
    c2 = chebyshev_decomposition(plan.decomp_plan, quad_AP)
    c4 = chebyshev_decomposition(plan.decomp_plan, hexa_AP)

    # Step 3: Apply precomputed window operators
    p0_conv = plan.M0 @ c0
    p2_conv = plan.M2 @ c2
    p4_conv = plan.M4 @ c4

    return p0_conv, p2_conv, p4_conv
