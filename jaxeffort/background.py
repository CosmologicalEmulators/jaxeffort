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
