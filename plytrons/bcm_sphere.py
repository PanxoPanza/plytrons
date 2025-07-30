"""
bcm_sphere.py
=============

Boundary-Charge Method (BCM) solver for *spherical* particles.
This standalone module is meant to be imported **from** projects that use
`plytrons.wigner3j` and `plytrons.math_utils`, but it is not itself a
sub-module of `plytrons`.  See the `EField`, `BCMObject`, and
`solve_BCM` docstrings for a quick-start guide.
"""

__all__ = [
    "EField", "BCMObject",
    "Ginternal", "Gexternal", "Efield_coupling", "solve_BCM",
    "EM_power",
]

import numpy as np
from functools import lru_cache
import numba as nb
from math import factorial
from plytrons.wigner3j import clebsch_gordan
from plytrons.math_utils import em_sph_harm
from dataclasses import dataclass, field
from functools   import cached_property
from typing import Callable, Optional, Union

# -------------------------------------------------------------------------
# Dataclasses that describe the light source and each sphere
# -------------------------------------------------------------------------
@dataclass
class EField:
    """
    Plane-wave electric field.

    Parameters
    ----------
    E0 : float
        Field amplitude in the same units expected downstream (e.g., V nm⁻¹).
    k_hat : np.ndarray
        3-component unit vector for the propagation direction.
    e_hat : np.ndarray
        3-component unit vector for the polarisation.
    """
    E0: float
    k_hat: np.ndarray
    e_hat: np.ndarray

    def __post_init__(self):
        # Normalise direction vectors for safety
        self.k_hat = self.k_hat / np.linalg.norm(self.k_hat)
        self.e_hat = self.e_hat / np.linalg.norm(self.e_hat)


@dataclass
class BCMObject:
    """
    One spherical nanoparticle for the Boundary-Charge Method.

    Attributes
    ----------
    label : str
        Identifier used in log messages and plots.
    diameter : float
        Sphere diameter in nanometres.
    lmax : int
        Highest multipole order retained in the BCM expansion.
    eps : Callable[[float], complex]
        Function that returns the complex permittivity at a given wavelength.
    position : np.ndarray
        Centre position **r**₀ in nanometres.
    BCM_coef   : 2-D np.ndarray, optional
        Expansion coefficients  shape = (n_modes, n_λ)  *or* None until solved.
    lam_um     : 1-D np.ndarray, optional
        Wavelength grid associated with `BCM_coef`  (same n_λ)  *or* None.
    """
    label    : str
    diameter : float
    lmax     : int
    eps      : Callable[[float], complex]
    position : np.ndarray

    # ---------- internal storage  (underscore → “private”) -----------
    _coef   : Optional[np.ndarray] = field(default=None,  repr=False, init=False)
    _lam_um : Optional[np.ndarray] = field(default=None,  repr=False, init=False)

    # ----------------------------------------------------------------
    # Post-init check for *construction* invariants (no coef yet)
    # ----------------------------------------------------------------
    def __post_init__(self):
        if self.position.shape != (3,):
            raise ValueError("position must be a length-3 Cartesian vector")

    # --- new helper ----
    @cached_property
    def n_coef(self) -> int:
        """Total (l,m) coefficients kept for this sphere = lmax (lmax + 2)."""
        return self.lmax * (self.lmax + 2)

    # --- replacement for the old free function ----
    @cached_property
    def index_range(self) -> np.ndarray:
        """
        Indices that this particle occupies inside the *global* coefficient
        vector/matrix.  Always 0…n_coef-1 **before** any offsets are added
        by `solve_BCM`.
        """
        return np.arange(self.n_coef)

    # ----------------------------------------------------------------
    # Read-only *properties*  (no setter → external code can’t modify)
    # ----------------------------------------------------------------
    @property
    def BCM_coef(self) -> Union[np.ndarray, None]:
        return self._coef

    @property
    def lam_um(self) -> Union[np.ndarray, None]:
        return self._lam_um

    # ----------------------------------------------------------------
    # Single public mutator that guarantees BOTH arrays are consistent
    # ----------------------------------------------------------------
    def set_coefficients(self, lam_um: np.ndarray, coef: np.ndarray) -> None:
        """
        Store solver output in a single, atomic operation.

        Parameters
        ----------
        coef   : complex ndarray, shape (n_modes, n_λ)
        lam_um : float   ndarray, shape (n_λ,)

        Raises
        ------
        ValueError  if shapes mismatch.
        """
        if lam_um.ndim != 1:
            raise ValueError("lam_um must be 1-D")

        if coef.shape[-1] != lam_um.size:
            raise ValueError("coef last axis length must equal lam_um size")

        # All good – assign to the *private* slots
        self._coef   = coef.astype(np.complex128, copy=False)
        self._lam_um = lam_um.astype(np.float64,      copy=False)

    # ------------------------------------------------------------------
    # Helper: get expansion coefficients at arbitrary λ by interpolation
    # ------------------------------------------------------------------
    def coef_at(self, lam_query, *, extrapolate: bool = False) -> np.ndarray:
        """
        Interpolated BCM coefficients at the requested wavelength(s).

        Parameters
        ----------
        lam_query : float | np.ndarray
            Target wavelength(s) in um.  Scalar or 1-D array.
        extrapolate : bool, default False
            • False → raise ValueError if any λ is outside the stored range.  
            • True  → allow linear extrapolation using numpy.interp.

        Returns
        -------
        np.ndarray
            Complex array of shape (n_modes, Nq) where Nq is the number of
            query points (scalar → Nq = 1).
        """
        if self.BCM_coef is None or self.lam_um is None:
            raise RuntimeError("BCM_coef and lam_um must be set before "
                               "calling coef_at().")

        # Ensure query is 1-D NumPy array for uniform handling
        lam_q = np.atleast_1d(np.asarray(lam_query, dtype=np.float64))

        # Range check unless user explicitly wants extrapolation
        if (not extrapolate) and (
            (lam_q.min() < self.lam_um.min()) or (lam_q.max() > self.lam_um.max())
        ):
            raise ValueError("Query wavelength(s) outside stored lam_um range. "
                             "Set extrapolate=True to override.")

        n_modes = self.BCM_coef.shape[0]
        n_query = lam_q.size
        coef_q  = np.empty((n_modes, n_query), dtype=np.complex128)

        # Loop over modes (fast, small) and interpolate real & imag separately
        for m in range(n_modes):
            real_part = np.interp(lam_q, self.lam_um, self.BCM_coef[m].real)
            imag_part = np.interp(lam_q, self.lam_um, self.BCM_coef[m].imag)
            coef_q[m] = real_part + 1j * imag_part

        # If user supplied a scalar λ, squeeze trailing dimension for ergonomics
        if np.isscalar(lam_query):
            coef_q = coef_q[:, 0]

        return coef_q

        

@nb.njit
def BCM_basis_sphere(R, l, m, theta, phi):
    """
    The basis for spherical coordinates from boundary charge method (BCM).
    
    Input:
        R       : Sphere radius (nm)
        l       : Zenith index from spherical harmonics
        m       : Azimuth index from spherical harmonics
        theta   : Zenith angle
        phi     : Azimuth angle

    Output:
        Beta    : Function basis output
    """
    # Compute the basis function
    Beta = np.sqrt((2 * l + 1) / R**3) * em_sph_harm(m, l, theta, phi)  # spherical harmonics

    return Beta

@nb.njit
def BCM_proj_sphere(R, l, m, theta, phi):
    """
    Projection operator of sphere basis under the formalism of the
    boundary charge method (BCM).
    
    Input:
        R       : Sphere radius (nm)
        l       : Zenith index from spherical harmonics
        m       : Azimuth index from spherical harmonics
        theta   : Zenith angle (in radians)
        phi     : Azimuth angle (in radians)

    Output:
        varphi  : Function basis projector output
    """
    # Compute the projection operator
    varphi = R / (2 * l + 1) * np.conj(BCM_basis_sphere(R, l, m, theta, phi))  # spherical harmonics

    return varphi

def v_normalize(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def get_axis(vector):
    """Determine which standard axis a vector corresponds to"""
    # Check if the input is a 3-dimensional vector
    if len(vector) != 3:
        raise ValueError('Input must be a 3-dimensional vector')
    
    # Normalize to get 0 or 1 only
    vector = np.array(vector) / np.array(vector)
    vector[np.isnan(vector)] = 0
    
    # Check each component to determine the axis
    if np.array_equal(vector, [1, 0, 0]):
        return 'x'
    elif np.array_equal(vector, [0, 1, 0]):
        return 'y'
    elif np.array_equal(vector, [0, 0, 1]):
        return 'z'
    else:
        raise ValueError('Input vector does not correspond to a standard axis')

def Ginternal(BCM_object):
    """
    Calcula la matriz de interacción interna de una esfera utilizando el método
    de carga de frontera (BCM):
    
    Parámetros:
    -----------
    BCM_object : objeto BCMObject
        Objeto que contiene las propiedades de la partícula
        .diameter : Diámetro de la partícula (nm)
        .lmax : Número máximo de índice angular
        .eps : Constante dieléctrica de la partícula
        .position : Posición de la partícula (nm)
    
    Retorna:
    --------
    Gin : ndarray
        Matriz de interacción interna
    
    Traducido de F. Ramirez 08/2019
    """
    # Asignar propiedades de la partícula
    lmax = BCM_object.lmax
    Ri = BCM_object.diameter / 2
    
    idx = 0
    idx_max = lmax * (lmax + 1) + (lmax + 1) - 1
    Gin = np.zeros((idx_max, idx_max), dtype = complex)
    
    for li in range(1, lmax + 1):
        for mi in range(-li, li + 1):
            Gin[idx, idx] = 1/2
            Gin[idx, idx] = Gin[idx, idx] * li / Ri**3  # factor adicional (conveniencia matemática)
            idx += 1
    
    return Gin

def Gexternal(BCM_object_i, BCM_object_j):
    """
    Calcula la matriz de interacción externa entre esferas utilizando el método
    de carga de frontera (BCM):
    
    Parámetros:
    -----------
    BCM_object_i : objeto BCMObject
        Objeto que contiene las propiedades de la primera partícula
        .label : Etiqueta de la partícula
        .diameter : Diámetro de la partícula (nm)
        .lmax : Número máximo de índice angular
        .eps : Constante dieléctrica de la partícula
        .position : Posición de la partícula (nm)
    
    BCM_object_j : objeto BCMObject
        Objeto que contiene las propiedades de la segunda partícula
    
    Retorna:
    --------
    Gext : ndarray
        Matriz de interacción externa
    
    Traducido de F. Ramirez 08/2019
    """
    # Asignar propiedades de las partículas
    Label_i = BCM_object_i.label
    lmax_i = BCM_object_i.lmax
    Ri = BCM_object_i.diameter / 2
    xi = BCM_object_i.position

    Label_j = BCM_object_j.label
    lmax_j = BCM_object_j.lmax
    Rj = BCM_object_j.diameter / 2
    xj = BCM_object_j.position

    # -------------------------------------------------------------------------
    # Auto-interacción en espacio libre 1/2Ri x <Phi_i,Gii * beta_i>
    # -------------------------------------------------------------------------
    if Label_j == Label_i:
        idx = 0
        idx_max = lmax_i * (lmax_i + 1) + (lmax_i + 1) - 1
        Gext = np.zeros((idx_max, idx_max), dtype = complex)
        
        for li in range(1, lmax_i + 1):
            for mi in range(-li, li + 1):
                Gext[idx, idx] = 1 / 2 / (2 * li + 1)
                Gext[idx, idx] = Gext[idx, idx] * li / Ri**3  # factor adicional (conveniencia matemática)
                idx += 1
    
    # -------------------------------------------------------------------------
    # Interacción con un segundo objeto
    # -------------------------------------------------------------------------
    else:
        idx_max = lmax_i * (lmax_i + 1) + (lmax_i + 1) - 1
        jdx_max = lmax_j * (lmax_j + 1) + (lmax_j + 1) - 1
        Gext = np.zeros((idx_max, jdx_max), dtype = complex)
        
        idx = 0
        for li in range(1, lmax_i + 1):
            for mi in range(-li, li + 1):
                jdx = 0
                for lj in range(1, lmax_j + 1):
                    for mj in range(-lj, lj + 1):
                        Gext[idx, jdx] = Gcoupling_ij(li, mi, Ri, xi,
                                                 lj, mj, Rj, xj)
                        jdx += 1
                idx += 1
    
    return Gext

def Gcoupling_ij(li, mi, Ri, xi, lj, mj, Rj, xj):
    """
    Calcula el coeficiente de acoplamiento entre dos esferas.
    
    Parámetros:
    -----------
    li, mi : int
        Índices angulares de la primera esfera
    Ri : float
        Radio de la primera esfera
    xi : list o ndarray
        Posición de la primera esfera
    lj, mj : int
        Índices angulares de la segunda esfera
    Rj : float
        Radio de la segunda esfera
    xj : list o ndarray
        Posición de la segunda esfera
    
    Retorna:
    --------
    Gpq_ij : float o complex
        Coeficiente de acoplamiento
    """
    # -------------------------------------------------------------------------
    # Coordenadas del vector entre las dos esferas
    # -------------------------------------------------------------------------
    d_ij_x = xj[0] - xi[0]
    d_ij_y = xj[1] - xi[1]
    d_ij_z = xj[2] - xi[2]

    # -------------------------------------------------------------------------
    # Obtener módulo y ángulos del vector entre las dos esferas
    # -------------------------------------------------------------------------
    d_ij_mod = np.sqrt(d_ij_x**2 + d_ij_y**2 + d_ij_z**2)  # módulo del vector
    tt_ij = np.arctan2(np.sqrt(d_ij_x**2 + d_ij_y**2), d_ij_z)  # ángulo cenital
    pp_ij = np.arctan2(d_ij_y, d_ij_x)  # ángulo azimutal

    # -------------------------------------------------------------------------
    # Calcular la integral:
    #       Y(li,mi)(Omega_i)*Y(lj,mj)(Omega_ij)/r_ij^(lj+1) dOmega_q
    # -------------------------------------------------------------------------
    int_ij = ((-1)**(lj-mi)) * ((2*lj + 1)/(2*(li+lj) + 1)) \
            * np.sqrt(4*np.pi*factorial(2*(li+lj) + 1) / \
             (factorial(2*li + 1)*factorial(2*lj + 1))) \
            * clebsch_gordan(li, -mi, lj, mj, li+lj, mj-mi) \
            * em_sph_harm(mj-mi, li+lj, tt_ij, pp_ij) \
            * Ri**li / d_ij_mod**(li+lj+1)

    # -------------------------------------------------------------------------
    # Terminar el cálculo agregando factores adicionales
    # -------------------------------------------------------------------------

    # Estos factores son independientes de la base
    Gpq_ij = -(Rj/Ri) * Rj**(lj + 1) / (2*lj + 1) * int_ij  # base: Ylm(Omega)
    Gpq_ij = np.sqrt(lj/Rj**3) * np.sqrt(li/Ri**3) * Gpq_ij  # base: sqrt(l/R^3)Ylm(Omega)

    # Agregar factores para base específica
    Gpq_ij = np.sqrt(lj*li) * (Ri/Rj)**(3/2) * Gpq_ij
    
    return Gpq_ij

def trapz2(Z, x=None, y=None):
    """
    Compute the double integral of a 2D array using the trapezoidal method.
    
    Parameters:
    -----------
    Z : ndarray
        The 2D array to integrate.
    x : ndarray, optional
        The x coordinates. If None, assumes uniform spacing.
    y : ndarray, optional
        The y coordinates. If None, assumes uniform spacing.
    
    Returns:
    --------
    float
        The result of the double integration.
    
    Notes:
    ------
    This function implements the equivalent of MATLAB's trapz2 function.
    It first integrates along the columns (axis=0) and then along the rows (axis=1).
    """
    # First integration along columns (axis=0)
    if y is None:
        col_integral = np.trapz(Z, axis=0)
    else:
        col_integral = np.trapz(Z, y, axis=0)
    
    # Second integration along rows (axis=1) of the result from the first integration
    if x is None:
        result = np.trapz(col_integral)
    else:
        result = np.trapz(col_integral, x)
    
    return result

@lru_cache(maxsize=None)
def _prepare_grid(n_theta: int = 150, n_phi: int = 30):
    """Build and cache the (θ,φ) mesh plus helpers."""
    θ = np.linspace(0, np.pi,  n_theta)
    φ = np.linspace(0, 2*np.pi, n_phi)
    tt, pp = np.meshgrid(θ, φ, indexing="ij")        # shape (nθ, nφ)

    sinθ   = np.sin(tt)
    n_hat  = np.stack((sinθ*np.cos(pp),               # x
                       sinθ*np.sin(pp),               # y
                       np.cos(tt)), axis=0)           # z  – shape (3,nθ,nφ)
    return θ, φ, tt, pp, sinθ, n_hat

def Ecoupling_matrix(lmax: int, Ri: float, e_hat: np.ndarray,
                     n_theta: int = 150, n_phi: int = 30) -> np.ndarray:
    """
    Vectorised replacement for Ecoupling_coef.
    Returns Xi for *all* (ℓ,m) with 1 ≤ ℓ ≤ lmax, -ℓ ≤ m ≤ ℓ.
    """
    θ, φ, tt, pp, sinθ, n_hat = _prepare_grid(n_theta, n_phi)

    # n·e  on the whole grid — broadcast e_hat over (nθ,nφ)
    ne_dot = (n_hat[0]*e_hat[0] +
              n_hat[1]*e_hat[1] +
              n_hat[2]*e_hat[2])                      # shape (nθ,nφ)

    # --- build flattened arrays of ℓ and m ---
    li, mi = [], []
    for ℓ in range(1, lmax+1):
        li.extend([ℓ]*(2*ℓ+1))
        mi.extend(range(-ℓ, ℓ+1))
    li = np.asarray(li)
    mi = np.asarray(mi)

    # Broadcast (ℓ,m) over the grid → Ylm shape (modes,nθ,nφ)
    Ylm = em_sph_harm(mi[:, None, None], li[:, None, None], tt, pp)

    # Integrand:  conj(Ylm) * (n·e)                       (broadcasted)
    integrand = np.conj(Ylm) * ne_dot[None, :, :]

    # Simple rectangular rule (same as trapz2 for uniform spacing)
    dθ  = θ[1] - θ[0]
    dφ  = φ[1] - φ[0]
    Xi  = np.tensordot(integrand * sinθ,               # shape (modes,nθ,nφ)
                       np.ones_like(ne_dot),           # integrate over grid
                       axes=([1,2],[0,1])) * dθ * dφ

    # Final radial prefactor √(ℓ / R³)
    Xi *= np.sqrt(li / Ri**3)
    return Xi

def Efield_coupling(obj_i: BCMObject, Efield: EField,
                    n_theta: int = 150, n_phi: int = 30) -> np.ndarray:
    """Vectorised – no inner (ℓ,m) loops."""
    eps0 = 0.055263493756           # vacuum permittivity (e / V·nm)
    Ri   = obj_i.diameter / 2
    Xi   = Ecoupling_matrix(obj_i.lmax, Ri, Efield.e_hat,
                            n_theta=n_theta, n_phi=n_phi)
    Si   = eps0 * Efield.E0 * Xi
    return Si

def solve_BCM(w, eps_h, BCM_objects, Efield, Gint, Gext, Sext):
    """
    Calcula los coeficientes de la expansión de densidad de carga superficial:
    
    Parámetros:
    -----------
    w : float
        Frecuencia (rad/s)
    eps_h : float
        Constante dieléctrica del medio
    BCM_objects : list
        Lista de objetos BCM con propiedades de las partículas
        .label : Etiqueta de la partícula
        .diameter : Diámetro de la partícula (nm)
        .lmax : Número máximo de índice angular
        .eps : Función de constante dieléctrica de la partícula
        .position : Posición de la partícula (nm)
    Efield : objeto ElectricField
        Propiedades de la fuente del campo E
        .E0 : Intensidad del campo E (W/m)
        .e_hat : Dirección del vector del campo E
        .k_hat : Dirección de la onda plana
    Gint : list
        Lista con matrices de interacción interna
    Gext : list of lists
        Lista de listas con matrices de interacción externa
    Sext : list
        Lista con coeficientes de acoplamiento de la fuente
    
    Retorna:
    --------
    X : list
        Lista de coeficientes de expansión para cada objeto
    Sw : list
        Lista de vectores de fuente para cada objeto
    
    Traducido de F. Ramirez 08/2019
    """
    # -------------------------------------------------------------------------
    # Definir constantes
    # -------------------------------------------------------------------------
    c0 = 299792458  # Velocidad de la luz (m/s)
    # Calcular vector de onda externo
    kh = w / c0 * np.sqrt(eps_h) * 1E-9  # vector de onda (1/nm)
    k_hat = Efield.k_hat
    E0 = Efield.E0
    lambda_value = 2 * np.pi * c0 / w * 1E6
    
    # -------------------------------------------------------------------------
    # Calcular sistema de matrices
    # -------------------------------------------------------------------------
    Np = len(BCM_objects)
    G_all_size = 0
    for in_idx in range(Np):
        # Calcular el tamaño total de la matriz de interacción
        G_all_size = G_all_size + Gint[in_idx].shape[0]
    
    G_all = np.zeros((G_all_size, G_all_size), dtype=complex)
    S_all = np.zeros(G_all_size, dtype=complex)
    
    idx_last = 0
    for in_idx in range(Np):
        # Calcular constante eta
        eps_i = BCM_objects[in_idx].eps(lambda_value)
        eta_i = (eps_i + eps_h) / (eps_i - eps_h)
        
        # Obtener el rango de índices de la matriz del objeto i
        i_range = idx_last + BCM_objects[in_idx].index_range
        
        # Almacenar matriz de interacción interna
        G_all[np.ix_(i_range, i_range)] = eta_i * Gint[in_idx]
        
        jdx_last = 0
        for jn_idx in range(Np):
            # Obtener el rango de índices de la matriz del objeto j
            j_range = jdx_last + BCM_objects[jn_idx].index_range
            
            # Almacenar matriz de interacción externa
            G_all[np.ix_(i_range, j_range)] = G_all[np.ix_(i_range, j_range)] - Gext[in_idx][jn_idx]
            
            # Actualizar índice
            jdx_last = j_range[-1]+1
        
        # Almacenar vector de interacción de campo E externo
        kh_ri = np.dot(k_hat, BCM_objects[in_idx].position)
        S_all[i_range] = Sext[in_idx] * np.exp(-1j * kh * kh_ri)
        
        # Actualizar índice
        idx_last = i_range[-1]+1
    
    # -------------------------------------------------------------------------
    # Resolver sistema lineal y almacenar resultados
    # -------------------------------------------------------------------------
    X_all_result = np.linalg.solve(G_all, S_all)
    X_all = X_all_result
    
    X = [None] * Np
    Sw = [None] * Np
    
    idx_last = 0
    for in_idx in range(Np):  # Using in_idx as 'in' is a reserved keyword in Python
        # Get the matrix index range from i-object
        i_range = idx_last + BCM_objects[in_idx].index_range
        
        # store results for each object - ensure i_range is properly used for indexing
        X[in_idx] = X_all[i_range]  # This should work since i_range is a numpy array
        Sw[in_idx] = S_all[i_range]
        
        # update index
        idx_last = i_range[-1]+1  # +1 to start at the next index
    
    return X, Sw

def EM_power(w, eps_h, Gint, Gext, BCM_objects):
    """
    Calcula la potencia EM dispersada y absorbida.
    
    Parámetros:
    -----------
    w : ndarray
        Frecuencia de cálculo (rad/s)
    eps_h : float
        Constante dieléctrica del medio
    Gint : list
        Lista con matrices de interacción interna
    Gext : list of lists
        Lista de listas con matrices de interacción externa
    BCM_objects : list
        Lista de objetos BCM con propiedades de las partículas
        .label      : Etiqueta de la partícula
        .diameter   : Diámetro de la partícula (nm)
        .lmax       : Número máximo de índice angular
        .eps        : Función de constante dieléctrica de la partícula
        .position   : Posición de la partícula (nm)
        .BCM_coef   : Coeficientes BCM calculados
    
    Retorna:
    --------
    Psca : list
        Lista de potencias dispersadas (W/umm^2)
    Pabs : list
        Lista de potencias absorbidas (W/umm^2)
    
    Traducido de F. Ramirez 08/2019
    """
    # -------------------------------------------------------------------------
    # Definir constantes
    # -------------------------------------------------------------------------
    eps0 = 0.055263493756                  # Permitividad del vacío (e/V-nm)
    c0 = 299792458                         # Velocidad de la luz (m/s)
    
    # -------------------------------------------------------------------------
    # Calcular potencia por objeto
    # -------------------------------------------------------------------------
    Np = len(BCM_objects)
    Pabs = [None] * Np
    Psca = [None] * Np
    
    for in_idx in range(Np):
        # Calcular constante eta
        R = BCM_objects[in_idx].diameter / 2
        
        Pabs[in_idx] = np.zeros(len(w))
        Psca[in_idx] = np.zeros(len(w))
        
        for iw in range(len(w)):
            kh = w[iw] / c0 * np.sqrt(eps_h) * 1E-9    # Vector de onda externo (1/nm)
            lambda_value = 2 * np.pi * c0 / w[iw] * 1E6  # Longitudes de onda
            
            eps_i = BCM_objects[in_idx].eps(lambda_value)
            eta_i = (eps_i + eps_h) / (eps_i - eps_h)
            
            # Calcular matriz de interacción interna
            G_i = eta_i * Gint[in_idx] - Gext[in_idx][in_idx]
            
            # Extraer coeficiente de expansión para la frecuencia requerida
            Xi = BCM_objects[in_idx].BCM_coef[:, iw]
            
            # Calcular potencia absorbida
            Pabs[in_idx][iw] = -np.real(w[iw] * R**3 / (2 * eps0) * 
                                       np.dot(np.conj(Xi).T, np.dot(asym(G_i), Xi)))
            
            # Calcular potencia dispersada
            Psca[in_idx][iw] = -np.real(w[iw] * (kh * R)**3 / (12 * np.pi * eps0) * 
                                       np.dot(np.conj(Xi).T, np.dot(asym(eta_i * G_i), Xi)))
    
    return Psca, Pabs

def asym(M):
    """
    Calcula la parte antisimétrica de una matriz.
    
    Parámetros:
    -----------
    M : ndarray
        Matriz de entrada
    
    Retorna:
    --------
    asymM : ndarray
        Parte antisimétrica de la matriz M
    """
    return 1/(2j) * (M - np.conj(M.T))