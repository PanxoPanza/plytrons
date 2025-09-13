"""
math_utils.py

Utility functions for spherical harmonics, Bessel functions, Legendre polynomials,
and related mathematical operations, optimized for semi-analytical modeling of 
plasmonic systems.

- Associated Legendre polynomials use the SHTOOLS normalization, which differs from
  scipy.special.lpmv by a known scaling factor and (optionally) the Condon–Shortley phase:

      scale = (-1)**m * sqrt((2l + 1) * scipy.special.poch(l + m + 1, -2*m))
      ours = Plm(l, m, z)
      ref = scale * scipy.special.lpmv(m, l, z)

- Spherical harmonics follow quantum mechanics conventions, but check for
  normalization and phase if comparing to SciPy or other references.

References:
-----------
- Holmes and Featherstone, J. Geodesy, 76, 279–299, 2002
- SHTOOLS: https://github.com/SHTOOLS/SHTOOLS
- scipy.special documentation

"""

import numpy as np
from scipy.special import jv, kv
from scipy.signal import find_peaks
import numba as nb
from numba.experimental import jitclass
import logging
import datetime as dt

hbar = 0.6582118      # Reduced planck constant (eV*fs)
me   = 5.686          # electron mass (eV*fs^2/nm^2)
eps0 = 0.055263493756 # vacuum permittivity (e/V nm)
e    = 1              # electron charge (eV) --- NOT SURE OF THIS

@nb.njit(parallel=False)
def nb_meshgrid(x, y):
    """
    Numba-compatible meshgrid for 1D arrays.

    Parameters
    ----------
    x : ndarray
        1D array representing x-coordinates of the grid.
    y : ndarray
        1D array representing y-coordinates of the grid.

    Returns
    -------
    xx : ndarray
        2D array where each row is a copy of x.
    yy : ndarray
        2D array where each column is a copy of y.
    """
    xx = np.empty(shape=(y.size, x.size), dtype=x.dtype)
    yy = np.empty(shape=(y.size, x.size), dtype=y.dtype)
    for j in nb.prange(y.size):
        for k in nb.prange(x.size):
            xx[j,k] = x[k]  # change to x[k] if indexing xy
            yy[j,k] = y[j]  # change to y[j] if indexing xy
    return xx, yy
    
# logging.info(f'{dt.datetime.now()} nb_meshgrid function defined')

#----------------------------------| Spherical Bessel First Kind and real argument |-------------------
@nb.vectorize('float64(int64, float64)', target = "cpu", nopython = True)
def _vec_js_real(l, x):
    '''
    Spherical Bessel function of the first kind of real order and real argument.

    Parameters
    ----------
    l : array_like
        Order (int).
    x : array_like
        Argument (float or complex). 
    '''
    return np.sqrt(np.pi / (2 * x + 1E-10)) * jv(l + 1/2, x)

@nb.njit(['float64(int64, float64)','float64[:](int64, float64[:])','float64[:,:](int64, float64[:,:])'])
def js_real(l: int, x: float) -> float:
    """
    Vectorized spherical Bessel function of the first kind (j_l) for real arguments.

    Computes the spherical Bessel function of the first kind for given order `l` and
    real argument `x`, using the relationship with the cylindrical Bessel function.

    Parameters
    ----------
    l : int or array_like of int
        Order of the spherical Bessel function.
    x : float or array_like of float
        Argument of the spherical Bessel function.

    Returns
    -------
    out : float or ndarray
        Value(s) of the spherical Bessel function j_l(x).
    """
    return _vec_js_real(l, x)

#----------------------------------| Spherical Bessel third Kind and imaginary argument |-------------------
@nb.vectorize('complex128(int64, float64)', target = 'cpu', nopython=True)
def _vec_hs_imag(l, x):
    '''
    Spherical Bessel function of the third kind for imaginary arugments.

    Parameters
    ----------
    l : array_like
        Order (int).
    x : array_like
        Imaginary part of "x" (float). 
    '''
    l = l+1/2
    hv_cplx = 2/np.pi*np.exp(-((l+1)%4)*np.pi*1j/2)*kv(l,x)
    return np.sqrt(np.pi / (2*x*1j)) *hv_cplx

@nb.njit(['complex128(int64, float64)','complex128[:](int64, float64[:])','complex128[:,:](int64, float64[:,:])'])
def hs_imag(l:int, x:float) -> float:
    """
    Spherical Hankel (Bessel) function of the third kind for imaginary arguments.

    Computes the spherical Bessel function of the third kind (Hankel function)
    for order `l` and purely imaginary argument `x`. Returns the function evaluated
    using the relationship with the modified Bessel function of the second kind.

    Parameters
    ----------
    l : int or array_like of int
        Order of the spherical Bessel function.
    x : float or array_like of float
        Imaginary part of the argument (i.e., for an input of the form x * 1j).

    Returns
    -------
    out : complex or ndarray of complex
        Value(s) of the spherical Hankel (Bessel) function for imaginary arguments.
    """
    return _vec_hs_imag(l, x)

#------------------------------------------| Legendre Polynomials |------------------------------------
spec = [
    ('lmax', nb.int64),
    ('sqr', nb.float64[:]),
    ('f1', nb.float64[:]),
    ('f2', nb.float64[:])
]

@jitclass(spec)
class Legendre_poly:
    def __init__(self):
        self.lmax = 0

    def _compute_prefactors(self):
        '''
        Compute multiplicative prefactors used in recursion relationships
           Plmbar(l,m) = x*f1(l,m)*Plmbar(l-1,m) - Plmbar(l-2,m)*f2(l,m)
           k = l*(l+1)/2 + m + 1
        Note that prefactors are not used for the case when m=l and m=l-1,
        as a different recursion is used for these two values. 
        '''
        dp = np.float64
        lmax = self.lmax
        
        f1 = np.empty((lmax+1)*(lmax+2)//2, dtype=dp)
        f2 = np.empty((lmax+1)*(lmax+2)//2, dtype=dp)

        # Precompute square roots of integers that are used several times.
        sqr = np.sqrt(np.arange(1, 2*lmax+2, dtype=dp))
        
        # Compute multiplicative factors
        k = 3
        for l in range(2, lmax+1):
            k += 1
            f1[k-1] = sqr[2*l-2] * sqr[2*l] / dp(l)
            f2[k-1] = dp(l-1) * sqr[2*l] / sqr[2*l-4] / dp(l)
    
            for m in range(1, l-1):
                k += 1
                f1[k-1] = sqr[2*l] * sqr[2*l-2] / sqr[l+m-1] / sqr[l-m-1]
                f2[k-1] = sqr[2*l] * sqr[l-m-2] * sqr[l+m-2] / \
                            sqr[2*l-4] / sqr[l+m-1] / sqr[l-m-1]
    
            k += 2
            
        self.f1, self.f2, self.sqr = f1, f2, sqr

    def Plm(self, lmax, mi, z, csphase=-1, cnorm=1):
        """
        Normalized associated Legendre functions up to degree lmax. The functions are 
        initially scaled by 10^280 sin^m in order to minimize the effects of underflow 
        at large m near the poles (1). The scaled portion of the algorithm will not overflow
        for degrees less than or equal to 2800.
    
        This code is a literal translation of PlmBar.f95 (2), part of the SHTools 
        project (3). 
        
        References:
         1. Holmes and Featherstone 2002, J. Geodesy, 76, 279-299
         2. https://github.com/SHTOOLS/SHTOOLS/blob/develop/src/PlmBar.f95
         3. Wieczorek, M. A., & Meschede, M. (2018). SHTools: Tools for working with
           spherical harmonics. Geochemistry, Geophysics, Geosystems,
        
        Parameters:
            lmax:       int
                        Maximum spherical harmonic degree to compute.
            
            z:          float, ndarray     
                        Polynomial argument.
                    
            csphase     int (optional)
                        Condon-Shortley phase factor of (-1)^m:
                             1: Do not include phase factor (default).
                            -1: Apply phase factor.
            
            cnorm       int (optional)
                        Defines the normalization value of int(-1,1) Plm
                            1: Use complex normalization (default). int(-1,1) Plm = 2
                            0: Use real normalization. int(-1,1) Plm = (2 - delta(0,m))
        
        Returns
            p:          float, ndarray           
                        A vector of all associated Legendre polynomials,
                        evaluated at z up to lmax.
        """
        # some value check before running the code
        assert abs(csphase)==1, "csphase must be 1 (exclude) or -1 (include)."
        assert lmax >= 0       , f'lmax must be greater than or equal to 0.\nInput value is {lmax}'
        assert abs(z) <= 1.0  , f"abs(z) must be less than or equal to 1."
        assert mi >= 0        ,'index m must be >= 0'
    
        # Constants
        phase = 1
        if csphase == -1:
            phase = -1
        
        scalef = 1.0e-280

        if lmax > self.lmax:
            self.lmax = lmax
            self._compute_prefactors()

        f1, f2, sqr = self.f1, self.f2, self.sqr
        
        # Allocate memory
        p = np.empty(((lmax+1)*(lmax+2)//2), dtype=np.float64)
        
        # Calculate P(l,0). These are not scaled.
        u = np.sqrt((1.0 - z) * (1.0 + z))  # sin(theta)
        p[0] = 1.0
    
        if lmax > 0:
            p[1] = sqr[2] * z
            
            k = 2
            for l in range(2, lmax+1):
                k += l
                p[k-1] = f1[k-1] * z * p[k-l-1] - f2[k-1] * p[k-2*l]
        
            # Calculate P(m,m), P(m+1,m), and P(l,m)
            if cnorm == 1:
                pmm = scalef
            else:
                pmm = sqr[1] * scalef
                
            rescalem = 1.0 / scalef
            kstart = 1
            
            for m in range(1, lmax):
                rescalem *= u
        
                # Calculate P(m,m)
                kstart += m + 1
                pmm = phase * pmm * sqr[2*m] / sqr[2*m-1]
                p[kstart-1] = pmm
        
                # Calculate P(m+1,m)
                k = kstart + m + 1
                p[k-1] = z * sqr[2*m+2] * pmm
        
                # Calculate P(l,m)
                for l in range(m+2, lmax+1):
                    k += l
                    p[k-1] = z * f1[k-1] * p[k-l-1] - f2[k-1] * p[k-2*l]
                    p[k-2*l] *= rescalem
        
                p[k-1] *= rescalem
                p[k-lmax-1] *= rescalem
        
            # Calculate P(lmax,lmax)
            rescalem *= u
            
            kstart += m + 1
            p[kstart] = phase * pmm * sqr[2*lmax] / sqr[2*lmax-1] * rescalem
    
        return p[(lmax*(lmax+1))//2+mi]

@nb.vectorize(['float64(int64, int64, float64)'], cache=False)
def nb_lpmv(l, m, z):
    '''
    Associated Legendre polynomials with Condon–Shortley phase factor (-1)**m.
    The polynomials are normalized by:
         sqrt((2l + 1)(l - m)!/(l + m)!)
    
    Parameters
    ------------
        - l: int
             degree of the polynomial
        - m: int
             order of the polynomial
        - z: float(ndarray)
             argument
    Returns
    ------------
        Evaluated polynomial Plm(z)
    '''
    poly = Legendre_poly() # set a Legendre Polynomial class 
    return poly.Plm(l, m, z, csphase = -1)

@nb.vectorize('complex128(float64, float64, float64, float64)', target='cpu', nopython=True, cache = False)
def qm_sph_harm(m, l, theta, phi):
    """
    Quantum spherical harmonics. Condon–Shortley phase factor (-1)**m is implicit in 
    Associated Legendre polynomials "nb_lpmv".
    
    * Note: this function is equivalent to scipy.special.sph_harm
                sph_harm(m, l, phi, theta)
    
    Parameters
    ----------
    m     : int
            Order of the harmonic (int); must have |m| <= l.
    l     : int
            Degree of the harmonic (int); must have l >= 0. 
    phi   : array_like
            Azimuthal (longitudinal) coordinate; must be in [0, 2*pi]
    theta : array_like
            Polar (colatitudinal) coordinate; must be in [0, pi].
        
    Returns
    -------
    y_ml  : complex array
            complex scalar or ndarray
    """
    
    mp = abs(m)
    assert mp <= l , 'm cannot not be greater than l'
    assert l >= 0,   'l cannot not be negative'
    
    x = np.cos(theta)

    #  Associated legendre polynomial (with Condon-Shortley phase factor)
    y_ml = nb_lpmv(nb.int64(l), nb.int64(mp), x)
    
    y_ml *= np.sqrt(1/ (4.0 * np.pi)) # 4pi normalization
    y_ml *= np.exp(1j * mp * phi)      # azimuthal component

    if m < 0:
        # y_ml = - y_ml
        y_ml = (-1)**mp*np.conj(y_ml)
    return y_ml

@nb.vectorize('complex128(float64, float64, float64, float64)', target='cpu', nopython=True, cache = False)
def em_sph_harm(m, l, theta, phi):
    """
    Electromagnetic spherical harmonics without Condon–Shortley phase factor (-1)**m
    
    * Note: this function is equivalent to scipy.special.sph_harm
                sph_harm(m, l, phi, theta)
    
    Parameters
    ----------
    m     : int
            Order of the harmonic (int); must have |m| <= l.
    l     : int
            Degree of the harmonic (int); must have l >= 0. 
    phi   : array_like
            Azimuthal (longitudinal) coordinate; must be in [0, 2*pi]
    theta : array_like
            Polar (colatitudinal) coordinate; must be in [0, pi].
        
    Returns
    -------
    y_ml  : complex array
            complex scalar or ndarray
    """
    
    mp = abs(m)
    assert mp <= l , 'm cannot not be greater than l'
    assert l >= 0,   'l cannot not be negative'
    
    return (-1)**(-m)*qm_sph_harm(m, l, theta, phi)

def detect_peaks(x_data, y_data, prominence=0.1, width=None, height=None,print_data=True):
    """
    Detect peaks in the data and return their x and y values.
    
    Parameters:
    -----------
    x_data : array-like
        The x-coordinates of the data
    y_data : array-like
        The y-coordinates of the data
    prominence : float, optional
        The prominence threshold for peak detection
    width : float, optional
        The width threshold for peak detection
    height : float, optional
        The height threshold for peak detection
    plot_result : bool, optional
        Whether to plot the result with the detected peaks
        
    Returns:
    --------
    peak_x : array
        The x-coordinates of the detected peaks
    peak_y : array
        The y-coordinates of the detected peaks
    """
    # Find peaks using scipy's find_peaks function
    peaks, properties = find_peaks(y_data, prominence=prominence, width=width, height=height)
    
    # Get x and y values of the peaks
    peak_x = x_data[peaks]
    peak_y = y_data[peaks]

    if print_data:    
        # Print the detected peaks
        print("Detected peaks:")
        for i in range(len(peak_x)):
            print(f"Peak {i+1}: x = {peak_x[i]:.3f} eV, y = {peak_y[i]:.3f} (10^-2)/eV/(ps·nm²)")
    
    return peak_x, peak_y
