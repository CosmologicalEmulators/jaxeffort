import jax
import jax.numpy as np
from quadax import quadgk
from interpax import Interpolator1D


# ------------------------------
# 1) your core JAX‐compatible routines
# ------------------------------
@jax.jit
def get_y(m_nu, a, kB=8.617342e-5, T_nu=0.71611 * 2.7255):
    return m_nu * a / (kB * T_nu)

@jax.jit
def F(y):
    def integrand(x):
        return x**2 * np.sqrt(x**2 + y**2) / (1 + np.exp(x))
    result, _ = quadgk(integrand, [0.0, np.inf], epsrel=1e-12)
    return result

@jax.jit
def dFdy(y):
    def integrand(x):
        return x**2 / ((1 + np.exp(x)) * np.sqrt(x**2 + y**2))
    result, _ = quadgk(integrand, [0.0, np.inf], epsrel=1e-12)
    return result * y

_min_y = float(get_y(0.0, 0.0))    # lower bound
_max_y = float(get_y(1.0, 10.0))   # upper bound

# two panels in y, same as vcat(LinRange(...), LinRange(...))
_y1 = np.linspace(_min_y,   100.0,  100,  endpoint=True)
_y2 = np.linspace(100.1,     _max_y, 1000, endpoint=True)
_y_F   = np.concatenate([_y1, _y2])

_y3 = np.linspace(_min_y,   10.0,   10000, endpoint=True)
_y4 = np.linspace(10.1,      _max_y, 10000, endpoint=True)
_y_dF  = np.concatenate([_y3, _y4])

# vectorize the jitted routines over numpy arrays
_F_vec   = jax.vmap(F)
_dFdy_vec = jax.vmap(dFdy)

# move to host (numpy) and evaluate once
_F_grid    = np.array(_F_vec(_y_F))
_dFdy_grid = np.array(_dFdy_vec(_y_dF))

F_spline   = Interpolator1D(_y_F,  _F_grid,  method="cubic")
dF_spline  = Interpolator1D(_y_dF,  _dFdy_grid, method="cubic")

@jax.jit
def Omega_nu_E2(a,
                Omega_gamma0,
                m_nu,
                kB: float = 8.617342e-5,
                T_nu: float = 0.71611 * 2.7255,
                Neff: float = 3.044):
    # compute the neutrino temperature‐ratio factor
    Gamma_nu = (4.0 / 11.0)**(1.0/3.0) * (Neff / 3.0)**(1.0/4.0)
    # argument to your F_spline
    y = get_y(m_nu, a, kB, T_nu)
    # the dimensionless neutrino contribution to E(a)^2
    prefac = 15.0 / np.pi**4 * Gamma_nu**4 * Omega_gamma0
    return prefac * a**(-4) * F_spline(y)


@jax.jit
def dOmega_nu_E2_da(a,
                    Omega_gamma0,
                    m_nu,               # float or 1D array
                    kB: float = 8.617342e-5,
                    T_nu: float = 0.71611 * 2.7255,
                    Neff: float = 3.044):
    # neutrino temperature‐ratio factor
    Gamma_nu = (4.0/11.0)**(1.0/3.0) * (Neff/3.0)**(1.0/4.0)
    # compute y for each mass (broadcasts if m_nu is array)
    y = get_y(m_nu, a, kB, T_nu)
    # scalar contributions (shape matches m_nu)
    terms = (
        -4.0 * F_interp(y) / a**5
        + dFdy_interp(y) / a**4 * (m_nu / (kB * T_nu))
    )
    # sum over masses (if m_nu is scalar, sum has no effect)
    sum_terms = np.sum(terms)
    # overall prefactor
    prefac = 15.0 / np.pi**4 * Gamma_nu**4 * Omega_gamma0
    return prefac * sum_terms


# --- (1) scale factor from redshift ---
@jax.jit
def a_z(z):
    """
    Convert redshift z to scale factor a = 1/(1+z).
    Broadcasts over arrays.
    """
    return 1.0 / (1.0 + z)


# --- (2) dark‐energy density in a‐form and z‐form ---
@jax.jit
def rho_DE_a(a, w0, wa):
    """
    ρ_DE(a) = a^[-3(1+w0+wa)] * exp[3 wa (a−1)]
    """
    exponent1 = -3.0 * (1.0 + w0 + wa)
    exponent2 = 3.0 * wa * (a - 1.0)
    return a**exponent1 * np.exp(exponent2)


@jax.jit
def rho_DE_z(z, w0, wa):
    """
    ρ_DE(z) = (1+z)^[3(1+w0+wa)] * exp[−3 wa z/(1+z)]
    """
    exponent1 = 3.0 * (1.0 + w0 + wa)
    exponent2 = -3.0 * wa * z / (1.0 + z)
    return (1.0 + z)**exponent1 * np.exp(exponent2)


# --- (3) derivative dρ_DE/da ---
@jax.jit
def drho_DE_da(a, w0, wa):
    """
    dρ_DE/da = 3 [−(1+w0+wa)/a + wa] * ρ_DE(a)
    """
    factor = 3.0 * (-(1.0 + w0 + wa) / a + wa)
    return factor * rho_DE_a(a, w0, wa)


# --- (4) total expansion rate E(a) = H(a)/H0 ---
#    requires Omega_nu_E2 from earlier
@jax.jit
def E_a(a,
        Omega_cb0,
        h,
        m_nu=0.0,
        w0=-1.0,
        wa=0.0):
    """
    Dimensionless expansion rate E(a) = H(a)/H0 for a flat
    universe with radiation, matter+CDM+baryon, dark energy (w0,wa),
    and massive neutrinos m_nu (can be scalar or array).
    """
    # photon density today
    Omega_gamma0 = 2.469e-5 / h**2
    # neutrino density at a=1
    Omega_nu0 = Omega_nu_E2(1.0, Omega_gamma0, m_nu)
    # dark‐energy density today to enforce flatness
    Omega_Lambda0 = 1.0 - (Omega_gamma0 + Omega_cb0 + Omega_nu0)

    # components: gamma ∝ a⁻⁴, cb ∝ a⁻³, DE ∝ rho_DE_a(a), ν ∝ Omega_nu_E2(a)
    radiation = Omega_gamma0 * a**(-4)
    matter    = Omega_cb0     * a**(-3)
    de        = Omega_Lambda0 * rho_DE_a(a, w0, wa)
    neutrino  = Omega_nu_E2(a, Omega_gamma0, m_nu)

    return np.sqrt(radiation + matter + de + neutrino)

@jax.jit
def E_z(z,
        Omega_cb0,
        h,
        m_nu: float = 0.0,
        w0: float = -1.0,
        wa: float = 0.0):
    """
    Dimensionless expansion rate E(z) = H(z)/H0.
    """
    a = a_z(z)
    return E_a(a, Omega_cb0, h, m_nu=m_nu, w0=w0, wa=wa)


@jax.jit
def dlogEdloga(a,
               Omega_cb0,
               h,
               m_nu: float = 0.0,
               w0: float = -1.0,
               wa: float = 0.0):
    """
    d ln E / d ln a = a * (dE/da) / E
    for a flat universe with radiation, matter+CDM+baryon,
    dark energy (w0,wa), and massive neutrinos m_nu.
    """
    # critical‐density fractions today
    Omega_gamma0 = 2.469e-5 / h**2
    Omega_nu0    = Omega_nu_E2(1.0, Omega_gamma0, m_nu)
    Omega_Lambda0 = 1.0 - (Omega_gamma0 + Omega_cb0 + Omega_nu0)

    # E(a) and E(a)^2
    E_val   = E_a(a, Omega_cb0, h, m_nu=m_nu, w0=w0, wa=wa)
    E2      = E_val**2

    # derivatives of each component w.r.t a:
    #  matter+CDM+baryon:  d(a^-3)/da = -3 a^-4
    term_cb    = -3.0 * Omega_cb0 * a**(-4)
    #  radiation:         d(a^-4)/da = -4 a^-5
    term_rad   = -4.0 * Omega_gamma0 * a**(-5)
    #  dark energy:       ΩΛ0 * dρ_DE/da
    term_de    =  Omega_Lambda0 * drho_DE_da(a, w0, wa)
    #  neutrinos:         dΩνE2/da
    term_nu    =  dOmega_nu_E2_da(a, Omega_gamma0, m_nu)

    # sum and assemble d ln E / d ln a
    numerator = term_cb + term_rad + term_de + term_nu
    return a * 0.5 * numerator / E2


@jax.jit
def Omega_m(a,
            Omega_cb0,
            h,
            m_nu: float = 0.0,
            w0: float = -1.0,
            wa: float = 0.0):
    """
    Matter fraction Ω_m(a) = Ω_cb0 a^-3 / E(a)^2
    """
    E2 = E_a(a, Omega_cb0, h, m_nu=m_nu, w0=w0, wa=wa)**2
    return Omega_cb0 * a**(-3) / E2
