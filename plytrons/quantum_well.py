import numpy as np
import numba as nb
from dataclasses import dataclass
from typing import List as PyList   # avoid clashing with numba.typed.List
from plytrons.math_utils import js_real, nb_meshgrid
from plytrons.math_utils import hbar, me
from numba import float64, complex128
from numba.experimental import jitclass

#Wave vector inside sphere -----------------------------------------------------------------------------------------
@nb.njit(['float64(float64)', 'float64[:](float64[:])', 'float64[:,:](float64[:,:])'])
def ke(x):
    '''
    Spherical quantum well wavevector inside the sphere

    Parameters
    ----------
    x : array_like
        Argument (float or complex).
    '''
    return np.sqrt(2 * me * x)/hbar

#Characteristic equation --------------------------------------------------------------------------------------------
    
@nb.vectorize('float64(float64, int64, float64)', target = 'cpu')
def F(x, l, a):
    '''
    Characteristic equation of spherical quantum well with finite potential
        *  Rearanged terms to get non-decaying behavior of F(x)
        
        ** Because of how F(x) has been defined, only the real part
        matters. Note that this is not always the case as different
        definitions of F(x) can have large imaginary parts too.

    Parameters
    ----------
    x : float array_like
        Energies (eV). 
    l : int
        spherical quantum index
    a : float
        sphere radius (nm)
    '''
    
    return np.real(js_real(l, ke(x)*a))

#Bound States --------------------------------------------------------------------------------------------------------

from numba import njit, prange, types
from numba.typed import List
import numpy as np

# ----------------------------------------------------------------------
# Vectorised characteristic equation (already in your file)
# ----------------------------------------------------------------------
@njit(fastmath=True)
def _bisection_root(f, l, a, E_low, E_high, max_iter, tol):
    """
    Simple bisection specialised for our scalar F(E,l,a).
    """
    f_low = f(E_low,  l, a)
    f_high= f(E_high, l, a)
    for _ in range(max_iter):
        E_mid = 0.5 * (E_low + E_high)
        f_mid = f(E_mid, l, a)
        if f_low * f_mid <= 0.0:
            E_high, f_high = E_mid, f_mid
        else:
            E_low,  f_low  = E_mid, f_mid
        if E_high - E_low < tol:
            break
    return 0.5 * (E_low + E_high)


@njit(parallel=True, fastmath=True)
def _bound_states_ragged(a_nm, lmax=50, V0=10.0,
                     nE_coarse=2000, max_iter=40, tol=1e-6):
    """
    Adaptive search for bound energies of a finite spherical quantum well.

    Parameters
    ----------
    a_nm : float
        Sphere radius in nm.
    lmax : int
        Highest angular-momentum index to consider (inclusive).
    V0 : float
        Well depth in eV (upper search bound).
    nE_coarse : int, optional
        Number of points in the *coarse* energy grid used only to bracket roots.
    max_iter, tol : int, float
        Bisection iteration cap and absolute tolerance on energy (eV).

    Returns
    -------
    bound_by_l : numba.typed.List[ndarray]
        bound_by_l[l] is a 1-D NumPy array with all E_{n,l} in ascending order.
    """
    # ----- 1. coarse grid used only for sign-change detection -------------
    E_grid = np.linspace(1e-4, V0 - 1e-4, nE_coarse)

    # ----- 2. pre-allocate ragged container in typed-list form ----------------
    bound_by_l = List.empty_list(types.float64[:])
    for _ in range(lmax + 1):
        bound_by_l.append(np.empty(0, dtype=np.float64))   # placeholder

    # ----- 3. loop over ℓ (parallel) --------------------------------------
    for l in prange(lmax + 1):
        l_i    = np.int64(l)                 # <<<<<< cast once
        F_vals = F(E_grid, l_i, a_nm)             # vectorised call
        roots_local = []                        # normal Python list

        # scan consecutive pairs for sign changes
        for i in range(nE_coarse - 1):
            if F_vals[i] * F_vals[i + 1] < 0.0:
                E_root = _bisection_root(F, l_i, a_nm,
                                         E_grid[i], E_grid[i + 1],
                                         max_iter, tol)
                roots_local.append(E_root)

        # convert typed list → 1-D ndarray (Numba allows this)
        nroots = len(roots_local)
        if nroots == 0:
            continue
        
        arr    = np.empty(nroots, dtype=np.float64)
        for j in range(nroots):
            arr[j] = roots_local[j]

        # store in the pre-sized list — SAFE in prange
        bound_by_l[l] = arr

    # remove empty values
    clean_bound_list = list(filter(len, bound_by_l))
    return clean_bound_list

def get_bound_states(*args, fill=np.nan, **kw):
    """
    Convenience wrapper that returns a 2-D padded array.

    Parameters are forwarded to `get_bound_states`. `fill` sets the padding
    value (default NaN).
    """
    E_ragged = _bound_states_ragged(*args, **kw)   # call the njit’d version
    max_n  = max(arr.size for arr in E_ragged)
    E_pad  = np.full((len(E_ragged), max_n), fill, dtype=np.float64)
    for l, arr in enumerate(E_ragged):
        E_pad[l, :arr.size] = arr
    return E_pad

# ---------------------------------------------------------------------
# Lightweight container for one ℓ “band” of bound states
# ---------------------------------------------------------------------
# • `@dataclass` auto-builds __init__, __repr__, __eq__, etc.
# • `slots=True` keeps the instances tiny (no __dict__), which matters
#   if you create thousands of these in post-processing sweeps.
# @dataclass(slots=True)
# class QWLevelSet:
#     """Bound-state data for one angular-momentum index ℓ."""
#     Eb: np.ndarray  # complex128 energies  shape = (n_levels,)
#     A : np.ndarray  # complex128 norms    shape = (n_levels,)

spec = [
    ('Eb', float64[:]),     # bound-state energies
    ('A',  complex128[:]),  # normalisation constants
]

@jitclass(spec)
class QWLevelSet:
    def __init__(self, Eb, A):
        self.Eb = Eb
        self.A  = A

# ---------------------------------------------------------------------
# Assemble energies + norms into a list[QWLevelSet]  ← pure Python
# ---------------------------------------------------------------------
# It accepts *either* the ragged list coming from get_bound_states
# *or* the padded 2-D array returned by get_bound_states_matrix.
def e_state_assembly(E_levels, A_levels) -> PyList[QWLevelSet]:
    """
    Convert raw bound-state arrays into a list of QWLevelSet objects.

    Parameters
    ----------
    E_levels : list[np.ndarray] | np.ndarray
        padded matrix of bound energies.
    A_levels : list[np.ndarray] | np.ndarray
        Matching normalisation factors (same layout as E_levels).

    Returns
    -------
    list[QWLevelSet]
        Index l → QWLevelSet holding *only* the valid (non-padding) levels.
    """

    # Type hint for editors / linters: list of QWLevelSet objects
    level_sets: PyList[QWLevelSet] = []

    # 1. Figure out whether we got the ragged list or padded matrix
    not_ragged = not isinstance(E_levels, (list, tuple))
    assert not_ragged, 'E_levels must be a 2D padded array'

    L = E_levels.shape[0]

    # 2. Loop over ℓ and slice out the real data
    for l in range(L):
        # Padded matrix: strip NaNs or zeros that mark “no level”
        mask = np.isfinite(E_levels[l]) & (E_levels[l] != 0.0)
        Eb   = E_levels[l, mask].astype(np.float64, copy=False)
        A    = A_levels[l, mask].astype(np.complex128, copy=False)

        # Package into dataclass and stash
        level_sets.append(QWLevelSet(Eb=Eb, A=A))
        
    return level_sets

@nb.njit(parallel = True)
def get_normalization(a, E_bound):
    '''
    Normalization constant of wave functions of spherical quantum well with finite potential

    Parameters
    ----------
    a : float
        sphere radius
    V0: float
        Quantum's well electric potential
    E_bound: ndarray
        Energy bound states
        
    Return
    --------
    E_bound : list of ndarray
        List of bound energy states
    '''
    A = np.zeros_like(E_bound)
    
    for l in nb.prange(E_bound.shape[0]):
        
        E_l = E_bound[l,E_bound[l,:]!=0] # get energy levels (ignore E = 0)
        
        # Perform integration inside the sphere (r < a)
        ri = np.linspace(0, a, 100)   # integration range
        rr, EE = nb_meshgrid(ri, E_l)  # meshgrid for column-wise integration
        int = np.trapz(np.abs(js_real(l, ke(EE) * rr))**2 * rr**2, ri, 1)
        
        # General fornulation for finite potential well
        A_l = 1 / np.sqrt(int)
        
        # Save normalization coefficients
        A[l,:len(E_l)] = A_l

    return A