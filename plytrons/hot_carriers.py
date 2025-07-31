from __future__ import annotations

from typing import List, Tuple

import numpy as np
import numba as nb
from numba import prange
from numba.typed import List as _NList   # ← NEW

# ── Project‑specific helpers ───────────────────────────────────────────────
from plytrons.math_utils import eps0, hbar, nb_meshgrid
from plytrons.wigner3j import gaunt_coeff
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


# -----------------------------------------------------------------------------
# 2. Single‑multipole transition matrix M_fi  (serial inside)
# -----------------------------------------------------------------------------

@nb.njit(fastmath=True, parallel = False)
def _transition_M(
    sf: Tuple[int, int],
    si: Tuple[int, int],
    a_nm: float,
    X_lm: np.ndarray,     # complex128[:]
    state_f: QWLevelSet,
    state_i: QWLevelSet,
) -> np.ndarray:
    """Compute *Mᶠᵢ* for a given pair of quantum numbers (l_f, m_f) ← (l_i, m_i)."""
    
    # get parameters of final state
    Ef, Af = state_f.Eb.real.astype(np.float64), state_f.A
    lf, mf = sf
    n_f = Ef.size

    # get parameters of initial state
    Ei, Ai = state_i.Eb.real.astype(np.float64), state_i.A
    li, mi = si
    n_i = Ei.size

    # Pre‑compute meshgrids that depend only on radial indices
    AAi, AAf = nb_meshgrid(Ai, Af)          # shapes (n_i, n_f) & (n_f, n_i)
    Mfi = np.zeros((n_f, n_i), dtype=np.complex128)

    # Radial grid constant for every multipole
    r = np.linspace(0.0, a_nm, 128)
    rr, EEi = nb_meshgrid(r, Ei)
    js_li = js_real(li, ke(EEi) * rr)       # shape (N_r, n_i)

    # Scan electromagnetic multipoles
    for idx in range(X_lm.size):
        le, mel = idx_to_lm(idx)

        # Wigner‑3j selection rules
        if mi + (-mf) + mel != 0:
            continue
        if abs(lf - le) > li or li > lf + le:
            continue

        X_abs = abs(X_lm[idx])
        if X_abs <= 1e-05:
            continue

        pref = (1.0/eps0)*np.sqrt(le/a_nm**3)*X_lm[idx]/(2*le + 1)
        scale = pref / a_nm**(le - 1)
        
        # Integration along solid angle
        Mfi_ang = gaunt_coeff(lf, le, li, -mf, mel, mi) * ((-1) ** (-mf))

        Mfi_rad = np.zeros_like(Mfi)
        # Radial integral for every Ef row (serial loop)
        for k in range(n_f):
            
            # Perform integration inside the sphere (r < a)
            rho = np.conj(js_real(lf, ke(Ef[k]) * rr)) * js_li   # (N_r, n_i)
            Mfi_rad[k, :] = np.trapz(rho*rr**(le + 2), r, 1)  # length n_i

        Mfi += scale*AAf.conj() * AAi * Mfi_ang*Mfi_rad
    return Mfi

# =============================================================================
# 3. Parallel driver with full (l,m) summation
# =============================================================================

@nb.njit(fastmath=True, parallel=True)
def _hot_e_dist_parallel(
    a_nm: float,
    hv_eV: float,
    E_F: float,
    tau_e_fs: float,
    e_state: List[QWLevelSet],
    X_lm: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:

    volume_nm3 = (4.0 / 3.0) * np.pi * a_nm ** 3
    gamma_e = hbar / (tau_e_fs * 1e3)  # eV

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
    Mfi_all = np.zeros((N, N), dtype=np.complex128)
    Mif_all = np.zeros_like(Mfi_all)

    # outer parallelism over final l index ----------------------------------
    for lf in prange(lmax):
        state_lf = e_state[lf]
        Ef_len = state_lf.Eb.size
        for li in range(lmax):
            state_li = e_state[li]
            Ei_len = state_li.Eb.size

            Mfi_block = np.zeros((Ef_len, Ei_len), dtype=np.complex128)
            Mif_block = np.zeros((Ei_len, Ef_len), dtype=np.complex128)

            # full m‑sum (serial)
            for mf in range(-lf, lf + 1):
                for mi in range(-li, li + 1):
                    sf = (lf, mf)
                    si = (li, mi)
                    Mfi_block += _transition_M(sf, si, a_nm, X_lm, state_lf, state_li)
                    Mif_block += _transition_M(si, sf, a_nm, X_lm, state_li, state_lf)

            # place blocks into global matrices
            li_s, li_e = l_range[li], l_range[li + 1]
            lf_s, lf_e = l_range[lf], l_range[lf + 1]
            Mfi_all[lf_s:lf_e, li_s:li_e] = Mfi_block
            Mif_all[li_s:li_e, lf_s:lf_e] = Mif_block

    # --- Golden‑rule probability matrices ----------------------------------
    EE_i, EE_f = nb_meshgrid(E_all, E_all)
    fd_i = _fermi_dirac(EE_i, E_F)
    fd_f = _fermi_dirac(EE_f, E_F)

    denom_e = (hv_eV - EE_f + EE_i) ** 2 + gamma_e ** 2
    denom_h = (hv_eV - EE_i + EE_f) ** 2 + gamma_e ** 2

    TTe = 4.0 / tau_e_fs * fd_i * (1.0 - fd_f) * (
        np.abs(Mfi_all) ** 2 / denom_e + np.abs(Mif_all.T.conj()) ** 2 / denom_h
    )
    TTh = 4.0 / tau_e_fs * fd_f * (1.0 - fd_i) * (
        np.abs(Mif_all) ** 2 / denom_h + np.abs(Mfi_all.T.conj()) ** 2 / denom_e
    )

    Te = TTe.sum(axis=1) / volume_nm3
    Th = TTh.sum(axis=1) / volume_nm3
    return Te, Th

# =============================================================================
# 4. Thin wrapper
# =============================================================================

def hot_e_dist(
    a_nm: float,
    hv_eV: float,
    E_F: float,
    tau_e_fs: float,
    e_state,                # plain list OR numba.typed.List
    X_lm: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    
    """Parallel hot‑carrier generation with full (l,m) summation."""
    if not isinstance(e_state, _NList):
        tmp = _NList()
        for s in e_state:
            tmp.append(s)
        e_state = tmp
        
    return _hot_e_dist_parallel(a_nm, hv_eV, E_F, tau_e_fs, e_state, X_lm)