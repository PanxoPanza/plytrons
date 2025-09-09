from __future__ import annotations

from typing import List, Tuple

import numpy as np
import numba as nb
from numba import prange
from numba.typed import List as _NList   # ← NEW

# ── Project‑specific helpers ───────────────────────────────────────────────
from plytrons.math_utils import eps0, hbar, nb_meshgrid
from plytrons.wigner3j import Wigner3j, gaunt_coeff
from plytrons.quantum_well import js_real, ke, QWLevelSet

__all__ = ["hot_e_dist"]

# =============================================================================
# 1. Low‑level utilities
# =============================================================================

@nb.njit(cache=True, fastmath=True)
def _fermi_dirac(E: np.ndarray, E_F: float, T: float = 300.0) -> np.ndarray:
    """Vectorised Fermi–Dirac occupation *f(E)* (Numba‑accelerated)."""
    k_B = 8.617333262e-5  # eV K⁻¹
    return 1.0 / (np.exp((E - E_F) / (k_B * T)) + 1.0)

@nb.njit(cache=True)
def idx_to_lm(k: int) -> tuple[int, int]:
    """
    Constant-time conversion from array position `k` → (l, m).

    Parameters
    ----------
    k : int
        Position in the list (Python 0-based by default).
    one_based : bool, optional
        Set to True if your “cardinal index” starts at 1 instead of 0.

    Returns
    -------
    l, m : tuple[int, int]
    """
    if k < 0:
        raise IndexError("index must be ≥ 0")

    # ---- 1. find l ---------------------------------------------------------
    # total items up to and including l is Nl = l(l + 2)
    # minimal l with Nl > k  ⇒  l = ⌊√(k + 1) − 1⌋ + 1
    l = np.floor(np.sqrt(k + 1) - 1) + 1         # integer math, O(1)

    # ---- 2. local offset → m ----------------------------------------------
    offset = k - ((l - 1) * (l + 1))   # k minus items in previous blocks
    m = offset - l                      # map 0…2l  →  −l…l
    return l, m

@nb.njit(cache=True)
def lm_to_idx(l: int, m: int) -> int:
    """
    Constant-time conversion from (l, m) → array position k (0-based).

    Parameters
    ----------
    l : int
        Degree, must be >= 1.
    m : int
        Order, must satisfy -l <= m <= l.

    Returns
    -------
    k : int
        Position in the flattened (l,m) list (0-based).

    Notes
    -----
    Items are grouped by l, with sizes 2l+1. The number of items before
    block l is (l-1)(l+1). Inside the block, m maps to offset m + l.
    """
    if l < 1:
        raise IndexError("l must be ≥ 1")
    if m < -l or m > l:
        raise IndexError("m must satisfy −l ≤ m ≤ l")

    # items before this l-block + offset within block
    k = (l - 1) * (l + 1) + (m + l)
    return k

# -----------------------------------------------------------------------------
# 2. Single‑multipole transition matrix M_fi  (serial inside)
# -----------------------------------------------------------------------------

@nb.njit(fastmath=True, parallel = False)
def _M_transition_squared(
    lf: int,
    li: int,
    a_nm: float,
    X_lm: np.ndarray,     # complex128[:]
    state_f: QWLevelSet,
    state_i: QWLevelSet,
) -> np.ndarray:
    """Compute *Mᶠᵢ* for a given pair of quantum numbers (l_f, m_f) ← (l_i, m_i)."""
    
    # get parameters of final state
    Ef, Af = state_f.Eb.real.astype(np.float64), state_f.A
    n_f = Ef.size

    # get parameters of initial state
    Ei, Ai = state_i.Eb.real.astype(np.float64), state_i.A
    n_i = Ei.size

    # |Af*Ai|^2 as an outer product -> (n_f, n_i)
    Af2 = (Af * Af.conj()).real
    Ai2 = (Ai * Ai.conj()).real
    AA_abs2 = Af2[:, None] * Ai2[None, :]

    # ----- radial grid & Bessels -------------------------------------------
    Nr = 128
    r = np.linspace(0.0, a_nm, Nr)      # (Nr,)
    rr = r[:, None]                          # (Nr, 1) for broadcasting

    # Bessel columns: A=(Nr,n_f), B=(Nr,n_i)
    j_lf = js_real(lf, ke(Ef[None, :]) * rr)       # j_lf(k_f r)
    j_li = js_real(li, ke(Ei[None, :]) * rr)       # j_li(k_i r)

    # trapezoid weights along r (Numba-safe)
    dr = 0.0 if Nr < 2 else (r[1] - r[0])
    w  = np.full(Nr, dr, dtype=np.float64)
    if Nr >= 1:
        w[0] *= 0.5
        w[-1] *= 0.5

    # will accumulate the real, positive squared amplitudes
    Mfi_2 = np.zeros((n_f, n_i), dtype=np.float64)

    # max l present in X_lm
    le_max = idx_to_lm(X_lm.size - 1)[0]

    for le in range(1, int(le_max) + 1):

        # triangle rule: |lf - li| <= le <= lf + li
        if le < abs(li - lf) or le > li + lf:
            continue

        # even-sum rule: lf + le + li must be even
        if ((lf + le + li) & 1) == 1:
            continue

       # ---------- Integration along solid angle ---------------------
        # power in field multipole le: P_le = sum_m |X_{le m}|^2
        idx0 = lm_to_idx(le, -le)
        idx1 = lm_to_idx(le,  le) + 1
        Xl = X_lm[idx0:idx1]

        # Numba-friendly real power sum
        X_lm_sum = 0.0
        for x_lm in Xl:
            X_lm_sum += x_lm.real * x_lm.real + x_lm.imag * x_lm.imag

        if X_lm_sum <= 1e-18:
            continue

        # angular factor from 3j orthogonality (sum over m_f, m_i)
        W = Wigner3j(lf, le, li, 0, 0, 0)
        Mfi_ang2 = ((2.0*lf + 1.0)*(2.0*li + 1.0)*(W*W)*X_lm_sum  # real\ 
                     / (4.0*np.pi))
                    
        # amplitude prefactor, squared (same physics as in _transition_M)
        # note: this is distinct from the orthogonality 1/(2le+1) that cancelled
        pref = (1.0/eps0) * np.sqrt(le / (a_nm**3)) / (2*le + 1)
        scale2 = (pref / (a_nm**(le - 1)))**2  # real

        # radial integrals for each Ef row
        # build weights r^(le+2)*w once per le
        rw = np.empty(Nr, dtype=np.float64)
        for ii in range(Nr):
            rw[ii] = w[ii] * (r[ii] ** (le + 2))

        # compute weighted B, then a single GEMM for all (f,i):
        # I = ∫ j_lf(k_f r) j_li(k_i r) r^(le+2) dr → (n_f, n_i)
        j_li_w = (rw[:, None]) * j_li                  # (Nr, n_i)
        I   = j_lf.T @ j_li_w                          # (n_f, n_i)

        # accumulate squared integral
        Mfi_2 += scale2 * Mfi_ang2 * (I * I)        # all real

    # include |Af*Ai|^2
    return Mfi_2 * AA_abs2

@nb.njit(fastmath=False, parallel = False)
def _M_transition_squared(
    lf: int,
    li: int,
    a_nm: float,
    X_lm: np.ndarray,     # complex128[:]
    state_f: QWLevelSet,
    state_i: QWLevelSet,
) -> np.ndarray:
    """Compute *Mᶠᵢ* for a given pair of quantum numbers (l_f, m_f) ← (l_i, m_i)."""
    
    # get parameters of final state
    Ef, Af = state_f.Eb.real.astype(np.float64), state_f.A
    n_f = Ef.size

    # get parameters of initial state
    Ei, Ai = state_i.Eb.real.astype(np.float64), state_i.A
    n_i = Ei.size

    # |Af*Ai|^2 as an outer product -> (n_f, n_i)
    AAi, AAf = nb_meshgrid(Ai, Af)          # shapes (n_i, n_f) & (n_f, n_i)

    AA_abs2 = np.abs(AAf.conj() * AAi)**2    # (n_f, n_i), real

    # ----- radial grid & Bessels -------------------------------------------
    Nr = 128
    r = np.linspace(0.001, a_nm, Nr)      # (Nr,)
    rr = r[:, None]                          # (Nr, 1) for broadcasting

    # Bessel columns: A=(Nr,n_f), B=(Nr,n_i)
    j_lf = js_real(lf, ke(Ef[None, :]) * rr)       # j_lf(k_f r)
    j_li = js_real(li, ke(Ei[None, :]) * rr)       # j_li(k_i r)

    # trapezoid weights along r (Numba-safe)
    dr = 0.0 if Nr < 2 else (r[1] - r[0])
    w  = np.full(Nr, dr, dtype=np.float64)
    if Nr >= 1:
        w[0] *= 0.5
        w[-1] *= 0.5

    # will accumulate the real, positive squared amplitudes
    Mfi_2 = np.zeros((n_f, n_i), dtype=np.float64)

    # max l present in X_lm
    le_max = idx_to_lm(X_lm.size - 1)[0]

    for le in range(1, int(le_max) + 1):

        # triangle rule: |lf - li| <= le <= lf + li
        if le < abs(li - lf) or le > li + lf:
            continue

        # even-sum rule: lf + le + li must be even
        if ((lf + le + li) & 1) == 1:
            continue

       # ---------- Integration along solid angle ---------------------
        # power in field multipole le: P_le = sum_m |X_{le m}|^2
        idx0 = lm_to_idx(le, -le)
        idx1 = lm_to_idx(le,  le) + 1
        Xl = X_lm[idx0:idx1]

        # Numba-friendly real power sum
        X_lm_sum = 0.0
        for x_lm in Xl:
            X_lm_sum += x_lm.real * x_lm.real + x_lm.imag * x_lm.imag

        if X_lm_sum <= 1e-5:
            continue

        # angular factor from m_i, m_f sums (orthogonality of 3j’s)
        W = Wigner3j(lf, le, li, 0, 0, 0)     # order is harmless once squared
        Mfi_ang2 = ((2.0*lf + 1.0) * (2.0*li + 1.0) / (4.0*np.pi*(2*le + 1))) * (W * W) * X_lm_sum


                    
        # amplitude prefactor, squared (same physics as in _transition_M)
        # note: this is distinct from the orthogonality 1/(2le+1) that cancelled
        pref = (1.0/eps0) * np.sqrt(le / (a_nm**3))/np.sqrt(2*le + 1)
        scale2 = (pref / (a_nm**(le - 1)))**2  # real

        # radial integrals for each Ef row
        # build weights r^(le+2)*w once per le
        rw = np.empty(Nr, dtype=np.float64)
        for ii in range(Nr):
            rw[ii] = w[ii] * (r[ii] ** (le + 2))

        # compute weighted B, then a single GEMM for all (f,i):
        # I = ∫ j_lf(k_f r) j_li(k_i r) r^(le+2) dr → (n_f, n_i)
        j_li_w = (rw[:, None]) * j_li                  # (Nr, n_i)

        j_lf_T  = np.ascontiguousarray(j_lf.T)   # (n_f, Nr)
        j_li_wc = np.ascontiguousarray(j_li_w)   # (Nr, n_i)
        I = j_lf_T @ j_li_wc                        # (n_f, n_i)

        # accumulate squared integral
        Mfi_2 += scale2 * Mfi_ang2 * (I * I)        # all real

        # print(AA_abs2, Mfi_ang2, (I * I))

    # include |Af*Ai|^2
    return Mfi_2 * AA_abs2

# =============================================================================
# 3. Parallel driver with full (l,m) summation
# =============================================================================

@nb.njit(fastmath=False, parallel=True)
def _hot_e_dist_parallel(
    a_nm: float,
    hv_eV: float,
    E_F: float,
    tau_e: np.ndarray,
    e_state: List[QWLevelSet],
    X_lm: np.ndarray,
    Pabs: float
) -> Tuple[np.ndarray, np.ndarray]:

    # --- flatten bound levels ---------------------------------------------
    lmax = len(e_state)
    l_range = np.zeros(lmax + 1, dtype=np.int64)
    for l in range(lmax):
        l_range[l + 1] = l_range[l] + e_state[l].Eb.size
    N = l_range[-1]

    E_all = np.empty(N, dtype=np.float64)
    for l in range(lmax):
        E_all[l_range[l]: l_range[l + 1]] = e_state[l].Eb.real

    # global transition matrices
    Mfi_2_all = np.zeros((N, N), dtype=np.float64)
    # Mif_2_all = np.zeros_like(Mfi_2_all)

    # outer parallelism over final l index ----------------------------------
    for lf in prange(lmax):
        state_lf = e_state[lf]
        lf_s, lf_e = l_range[lf], l_range[lf + 1]
        for li in range(lmax):
            state_li = e_state[li]
            li_s, li_e = l_range[li], l_range[li + 1]

            # Compute transition matrix for pair (lf, li)
            Mfi_2_block = _M_transition_squared(lf, li, a_nm, X_lm, state_lf, state_li)
            # Mif_2_block = _M_transition_squared(li, lf, a_nm, X_lm, state_li, state_lf)

            # place blocks into global matrices
            Mfi_2_all[lf_s:lf_e, li_s:li_e] = Mfi_2_block
            # Mif_2_all[li_s:li_e, lf_s:lf_e] = Mif_2_block

# --- Golden‑rule probability matrices ----------------------------------
    EE_i, EE_f = nb_meshgrid(E_all, E_all)
    fd_i = _fermi_dirac(EE_i, E_F)
    fd_f = _fermi_dirac(EE_f, E_F)

    Te = np.zeros((len(tau_e), N), dtype=np.float64)
    Th = np.zeros((len(tau_e), N), dtype=np.float64)

    for i in range(len(tau_e)):

        tau_dx = tau_e[i]

        gamma_e = hbar / tau_dx  # eV    

        denom_e = (hv_eV - EE_f + EE_i)**2 + gamma_e**2
        denom_h = (hv_eV - EE_i + EE_f)**2 + gamma_e**2

        TTe = 4.0/tau_dx * fd_i * (1.0 - fd_f) * (
            Mfi_2_all/denom_e + Mfi_2_all.T/denom_h
            )
        
        TTh = 4.0/tau_dx * fd_f * (1.0 - fd_i) * (
            Mfi_2_all/denom_h + Mfi_2_all.T/denom_e
            )

        Te_raw = TTe.sum(axis=1)
        Th_raw = TTh.sum(axis=1)

        # --- Normalisation -----------------------------------------------------
        omega = hv_eV / hbar                     # fs^-1   (hv is energy in eV)
        Vol = 4/3*np.pi*a_nm**3            # Volume of sphere (nm^3)
        Gamma_e_total = np.sum(Te_raw)*Vol # total electron gen. rate = sum over f (per fs)
        S = Pabs / (hbar * omega * Gamma_e_total)                 # dimensionless scaling
        
        Te[i] = S * Te_raw/Vol
        Th[i] = S * Th_raw/Vol

    return Te, Th


# =============================================================================
# 4. Thin wrapper
# =============================================================================

def hot_e_dist(
    a_nm: float,
    hv_eV: float,
    E_F: float,
    tau_e_fs: np.ndarray,
    e_state,                # plain list OR numba.typed.List
    X_lm: np.ndarray,
    Pabs: float
) -> Tuple[np.ndarray, np.ndarray]:
    
    """Parallel hot‑carrier generation with full (l,m) summation."""
    if not isinstance(e_state, _NList):
        tmp = _NList()
        for s in e_state:
            tmp.append(s)
        e_state = tmp
        
    return _hot_e_dist_parallel(a_nm, hv_eV, E_F, tau_e_fs, e_state, X_lm, Pabs)

