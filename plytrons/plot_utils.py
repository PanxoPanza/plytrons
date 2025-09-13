# plytrons/plot_utils.py

from __future__ import annotations
from pathlib import Path
from itertools import combinations
from typing import Optional, Union, Sequence
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from plytrons.bcm_sphere import get_axis as _bcm_get_axis
import plytrons.bcm_sphere as bcm
from scipy.constants import hbar, eV

__all__ = ["make_results_folder"]

def make_results_folder(
    bcms: Sequence,                 # sequence of BCMObject (must have .diameter, .position)
    efield,                         # EField (must have .e_hat and .k_hat)
    *,
    lmax: Optional[int] = None,
    eps_h: Optional[float] = None,
    prefix: Union[str, Path] = "results",
    rtol: float = 1e-3,
    atol: float = 1e-3,
    include_timestamp: bool = False,
    extra_tag: Optional[str] = None,
) -> Path:
    """
    Create (and return) a results folder whose name encodes the geometry.

    Naming rules:
      - Symmetric dimer/trimer with equal diameters → "{dimer|trimer}_D{:.1f}nm_gap{:.1f}nm_{Exkz}"
      - Otherwise → "{<N>-mer}-{shape}_{Dmix|D{:.1f}nm}_gap{:.1f}nm_{Exkz}"
        where shape ∈ {single, line, tri-equil, tri-iso, tri-scal, planar, ring, 3D}

    Parameters
    ----------
    bcms : sequence of BCMObject
        Each object must expose `.diameter: float` (nm) and `.position: array-like (nm)`.
    efield : EField
        Must expose `.e_hat` and `.k_hat` for orientation labeling via Exkz, Eykx, etc.
    lmax : int, optional
        If provided, appended as `_lmax{lmax}`.
    eps_h : float, optional
        If provided, appended as `_epsh{eps_h}` (compact).
    prefix : str | Path
        Parent directory for the results folder. Created if missing.
    rtol, atol : float
        Tolerances for geometric comparisons (equal diameters, equal sides, etc.).
    include_timestamp : bool
        If True, append a UTC timestamp suffix `_YYYYmmdd-HHMMSS`.
    extra_tag : str, optional
        Any extra descriptor to append at the end (e.g., "run42").

    Returns
    -------
    Path
        The created directory path.

    Examples
    --------
    >>> outdir = make_results_folder(BCM_objects, efield)
    >>> fig.savefig(outdir / "absorption_spectrum.png", dpi=200)
    """
    bcms = list(bcms)
    Np = len(bcms)

    # ----- base name by particle count -----
    nname_map = {1: "monomer", 2: "dimer", 3: "trimer", 4: "tetramer"}
    nname = nname_map.get(Np, f"{Np}-mer")

    # ----- diameters -----
    diam = np.array([float(o.diameter) for o in bcms], dtype=float)
    equal_D = np.allclose(diam, diam[0], rtol=rtol, atol=atol)
    Dpart = f"D{diam[0]:.1f}nm" if equal_D else "Dmix"

    # ----- minimum gap (nm) -----
    if Np >= 2:
        centers = np.vstack([np.asarray(o.position, dtype=float) for o in bcms])
        gap_min = np.inf
        for i, j in combinations(range(Np), 2):
            dc = float(np.linalg.norm(centers[i] - centers[j]))
            gap = dc - 0.5 * (diam[i] + diam[j])
            if gap < gap_min:
                gap_min = gap
        gap_min = max(0.0, float(gap_min))
        Gpart = f"gap{gap_min:.1f}nm"
    else:
        Gpart = "isolated"

    # ----- simple shape descriptor -----
    if Np == 1:
        shape = "single"
    elif Np == 2:
        # two points are always collinear
        shape = "line"
    elif Np == 3:
        coords = np.vstack([np.asarray(o.position, dtype=float) for o in bcms])
        centered = coords - coords.mean(axis=0, keepdims=True)
        rank = np.linalg.matrix_rank(centered, tol=1e-9)
        if rank <= 1:
            shape = "line"
        else:
            # classify triangle
            sides = np.array([np.linalg.norm(coords[i] - coords[j])
                              for i, j in combinations(range(3), 2)])
            if np.allclose(sides, sides[0], rtol=rtol, atol=atol):
                shape = "tri-equil"
            elif (np.isclose(sides[0], sides[1], rtol=rtol, atol=atol) or
                  np.isclose(sides[0], sides[2], rtol=rtol, atol=atol) or
                  np.isclose(sides[1], sides[2], rtol=rtol, atol=atol)):
                shape = "tri-iso"
            else:
                shape = "tri-scal"
    else:
        coords = np.vstack([np.asarray(o.position, dtype=float) for o in bcms])
        centered = coords - coords.mean(axis=0, keepdims=True)
        rank = np.linalg.matrix_rank(centered, tol=1e-9)
        if rank <= 1:
            shape = "line"
        elif rank == 2:
            radii = np.linalg.norm(centered, axis=1)
            # ring if all radii identical within loose tol
            shape = "ring" if np.allclose(radii, radii[0], rtol=1e-2, atol=1e-2) else "planar"
        else:
            shape = "3D"

    # ----- field orientation (e.g., Exkz) -----
    try:
        orient = f"E{_bcm_get_axis(efield.e_hat)}k{_bcm_get_axis(efield.k_hat)}"
    except Exception:
        orient = "E?k?"  # fallback if efield lacks expected attrs

    # ----- assemble name -----
    if (Np in (2, 3)) and equal_D and ((Np == 2 and shape == "line") or (Np == 3 and shape == "tri-equil")):
        # Your preferred symmetric naming
        base = f"{nname}_{Dpart}_{Gpart}_{orient}"
    else:
        base = f"{nname}-{shape}_{Dpart}_{Gpart}_{orient}"

    if lmax is not None:
        base += f"_lmax{int(lmax)}"
    if eps_h is not None:
        # compact float repr without trailing zeros if possible
        eps_str = f"{eps_h:g}"
        base += f"_epsh{eps_str}"
    if extra_tag:
        base += f"_{extra_tag}"

    if include_timestamp:
        from datetime import datetime, timezone
        base += "_" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")

    outdir = Path(prefix) / base
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir

def flip_coefs_along_wavelength(lam, coef, flip_lam_too=False):
    c = np.asarray(coef)
    if c.ndim != 2:
        raise ValueError("coef must be 2-D")
    if c.shape[1] == lam.size:      # (modes, λ)
        c = c[:, ::-1]
    elif c.shape[0] == lam.size:    # (λ, modes)
        c = c[::-1, :]
    else:
        raise ValueError("coef shape doesn't match lam length")
    lam_out = lam[::-1] if flip_lam_too else lam
    return lam_out, c

def coefficients_plot(Np, lam, coef, flip=True, flip_lam_too=False):
    # ensure shape (λ, modes) for consistent plotting
    Yre, Yim = np.asarray(coef.real), np.asarray(coef.imag)
    if Yre.shape[0] != lam.size:
        Yre, Yim = Yre.T, Yim.T

    if flip:
        lam, Yre = flip_coefs_along_wavelength(lam, Yre, flip_lam_too)
        _,   Yim = flip_coefs_along_wavelength(lam, Yim, flip_lam_too=False)

    # highlight first mode; plot all others without legend
    plt.plot(lam, Yre[:, 0], 'k.-', label='real part of EM coefficients')
    plt.plot(lam, Yim[:, 0], 'k-',  label='imaginary part of EM coefficients')
    plt.plot(lam, Yre, '.-', alpha=1, label='_nolegend_')
    plt.plot(lam, Yim, '-',  alpha=1, label='_nolegend_')

    plt.title(f"Nanoparticle {Np+1} — EM coefficients vs. wavelength")
    plt.legend(); plt.tight_layout(); plt.show()


def print_active_lm(BCM, thr=1.0):
    # |coef| as (n_modes, n_lambda)
    mag = np.nan_to_num(np.abs(BCM.BCM_coef).astype(float))
    if mag.shape[1] != BCM.lam_um.size:  # fix orientation if needed
        mag = mag.T

    k = np.flatnonzero(mag.max(axis=1) > thr)   # active mode indices
    if k.size == 0:
        print("No active modes."); return

    # vectorized k -> (l, m)
    l = (np.floor(np.sqrt(k + 1) - 1) + 1).astype(int)
    m = k - ((l - 1) * (l + 1)) - l

    print(*[f"Active l,m -> l={ll}, m={mm}" for ll, mm in zip(l, m)], sep="\n")

def convert_raw_hot(Te, Th, E, dE_factor):
    idx = np.argsort(E)
    E = E[idx]; Te = Te[idx]; Th = Th[idx]
    E_plot = np.linspace(E.min(), E.max(), 1000)
    dE = (E_plot[1] - E_plot[0]) * dE_factor
    Phi = lambda x: (1/np.pi) * (dE/2) / (x**2 + (dE/2)**2)  # Lorentzian (unit area)

    EE, EEp = np.meshgrid(E, E_plot)
    TTe = np.meshgrid(Te, E_plot)[0]
    TTh = np.meshgrid(Th, E_plot)[0]

    Te_plot = np.sum(TTe * Phi(EEp - EE), axis=1)
    Th_plot = np.sum(TTh * Phi(EEp - EE), axis=1)
    return E_plot, Te_plot, Th_plot

def hot_carriers_plot(Te, Th, Te_raw, Th_raw, 
                      e_states, Np, peak, tau_e, D, hv, EF, dE_factor, delta):

    Te = Te/1000
    Th = Th/1000
    Te_raw = Te_raw
    Th_raw = Th_raw

    # energies in SAME order as Te/Th
    E_all = np.concatenate([es.Eb[es.Eb != 0] for es in e_states]).real
    assert len(E_all) == len(Te) == len(Th), "Length mismatch between E_all and Te/Th."

    # smooth to fine grid
    x, Te_x, Th_x = convert_raw_hot(Te, Th, E_all, dE_factor)

    # fs^-1 → ps^-1, then per eV
    to_ps = 1000.0
    scale = to_ps / hv
    Te_x *= scale; Th_x *= scale

    # masks
    mask_e = (x >= EF) & (x <= EF + delta)
    mask_h = (x <= EF) & (x >= EF - delta)

    # plot
    fig, ax = plt.subplots(figsize=(20, 4.5))
    ax.fill_between((x - EF)[mask_e], Te_x[mask_e], color='r', alpha=0.38)
    ax.fill_between((x - EF)[mask_h], Th_x[mask_h], color='b', alpha=0.38)

    # SECOND Y-AXIS for bars
    ax2 = ax.twinx()
    bar_width = 2.0e-2
    ax2.bar(E_all - EF, Te_raw * to_ps, width=bar_width, color='firebrick', alpha=0.9, label='Electrons ')
    ax2.bar(E_all - EF, Th_raw * to_ps, width=bar_width, color='royalblue', alpha=0.9, label='Holes')

    # guides
    ax.axvline(0.0, ls='--', lw=1, color='k', alpha=0.5)
    ax.axvline(+hv, ls='--', lw=1, color='gray', alpha=0.6)
    ax.axvline(-hv, ls='--', lw=1, color='gray', alpha=0.6)

    ax.set_xlim(-delta, delta)
    ax.set_xlabel('Hot carrier energy relative to Fermi level (eV)')
    ax.set_ylabel(r'Hot carrier generation rate density $[{10}^{-3}\mathrm{eV}^{-1}\,\mathrm{ps}^{-1}\,\mathrm{nm}^{-3}]$')
    ax2.set_ylabel('hot carrier generation rate per particle $[\mathrm{ps}^{-1}]$')  # ← label for the bars’ axis

    # optional: fix independent y-limits
    ax.set_ylim(0, 1.05 * max(Te_x[mask_e].max(initial=0), Th_x[mask_h].max(initial=0)))
    ax2.set_ylim(0, 3.0 * to_ps * max(Te_raw.max(initial=0), Th_raw.max(initial=0)))

    # one combined legend
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right')

    ax.set_title(rf'Nanoparticle N°{Np}, Resonance peak N°{peak}, D = {D} nm, $\tau = ${np.atleast_1d(tau_e)[0]/1000:.2f} ps, $h\nu$ = {hv:.2f} eV')
    ax.grid(True, ls=':')
    plt.tight_layout()
    plt.show()

def hot_carrier_dynamics_plot(Te, Th, Te_raw, Th_raw, 
                              e_states, Np, peak, tau_e, D, hv, EF, dE_factor, delta, fps, out_path, bar_width=2.0e-2):

    # energies in SAME order as Te/Th
    E_all = np.concatenate([es.Eb[es.Eb != 0] for es in e_states]).real

    # arrays & sanity
    Te_arr   = np.asarray(Te,     float)/1000  # (n_tau, N)
    Th_arr   = np.asarray(Th,     float)/1000
    Te_raw_a = np.asarray(Te_raw, float)
    Th_raw_a = np.asarray(Th_raw, float)
    tau_arr  = np.asarray(tau_e,  float)

    n_tau, N = Te_arr.shape
    assert Th_arr.shape   == (n_tau, N)
    assert Te_raw_a.shape == (n_tau, N)
    assert Th_raw_a.shape == (n_tau, N)
    assert E_all.size     == N

    # fixed y-limits
    to_ps = 1000.0  # fs -> ps
    def frame_max_lines(i):
        x, Te_x, Th_x = convert_raw_hot(Te_arr[i], Th_arr[i], E_all, dE_factor)
        s = to_ps / hv
        Te_x *= s; Th_x *= s
        mask_e = (x >= EF) & (x <= EF + delta)
        mask_h = (x <= EF) & (x >= EF - delta)
        m_e = Te_x[mask_e].max() if np.any(mask_e) else 0.0
        m_h = Th_x[mask_h].max() if np.any(mask_h) else 0.0
        return max(m_e, m_h)

    YMAX  = 1.05 * max(frame_max_lines(i) for i in range(n_tau)) if n_tau else 1.0
    YMAX2 = 1.20 * to_ps * max(np.max(Te_raw_a), np.max(Th_raw_a)) if n_tau else 1.0

    # figure & artists
    fig, ax = plt.subplots(figsize=(20, 4.5))
    line_e, = ax.plot([], [], lw=1.8, color='r', alpha = 0.5)
    line_h, = ax.plot([], [], lw=1.8, color='b', alpha = 0.5)

    # will hold the fill_between polygons; start empty
    fill_e = None
    fill_h = None

    # secondary axis for bars
    ax2 = ax.twinx()
    x_bar = E_all - EF
    bars_e = ax2.bar(x_bar, np.zeros_like(x_bar), width=bar_width,
                     color='firebrick', alpha=1, label='Electrons')
    bars_h = ax2.bar(x_bar, np.zeros_like(x_bar), width=bar_width,
                     color='royalblue',  alpha=1, label='Holes')

    # guides
    ax.axvline(0.0, ls='--', lw=1, alpha=0.5)
    ax.axvline(+hv, ls='--', lw=1, alpha=0.6)
    ax.axvline(-hv, ls='--', lw=1, alpha=0.6)

    ax.set_xlim(-delta, delta)
    ax.set_ylim(0, YMAX)
    ax2.set_ylim(0, YMAX2)

    ax.set_xlabel('Hot carrier energy relative to $E_F$ (eV)')
    ax.set_ylabel(r'Hot carrier generation rate density $[{10}^{-3}\mathrm{eV}^{-1}\,\mathrm{ps}^{-1}\,\mathrm{nm}^{-3}]$')
    ax2.set_ylabel('hot carrier generation rate per particle $[\mathrm{ps}^{-1}]$')  # ← label for the bars’ axis
    ax.grid(True, ls=':')

    # merged legend (lines + bars; fills get no legend via '_nolegend_')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc='upper right')

    # frame builder
    def make_frame_data(i):
        x, Te_x, Th_x = convert_raw_hot(Te_arr[i], Th_arr[i], E_all, dE_factor)
        s = to_ps / hv
        Te_x *= s; Th_x *= s
        mask_e = (x >= EF) & (x <= EF + delta)
        mask_h = (x <= EF) & (x >= EF - delta)
        return (x - EF)[mask_e], Te_x[mask_e], (x - EF)[mask_h], Th_x[mask_h]

    def init():
        nonlocal fill_e, fill_h
        line_e.set_data([], [])
        line_h.set_data([], [])
        if fill_e: fill_e.remove(); fill_e = None
        if fill_h: fill_h.remove(); fill_h = None
        for r in list(bars_e) + list(bars_h):
            r.set_height(0.0)
        ax.set_title('')
        return [line_e, line_h, *bars_e.patches, *bars_h.patches]

    def update(i):
        nonlocal fill_e, fill_h
        xe, ye, xh, yh = make_frame_data(i)

        # lines (for legend/edges)
        line_e.set_data(xe, ye)
        line_h.set_data(xh, yh)

        # refresh fills each frame
        if fill_e: fill_e.remove()
        fill_e = ax.fill_between(xe, 0.0, ye, color='r', alpha=0.3, label='_nolegend_')
        if fill_h: fill_h.remove()
        fill_h = ax.fill_between(xh, 0.0, yh, color='b', alpha=0.3, label='_nolegend_')

        # bars (secondary axis)
        he = to_ps * Te_raw_a[i]
        hh = to_ps * Th_raw_a[i]
        for rect, h in zip(bars_e, he): rect.set_height(h)
        for rect, h in zip(bars_h, hh): rect.set_height(h)

        tau_ps = tau_arr[i] / 1000.0
    
        ax.set_title(rf'Nanoparticle N°{Np}, Resonance peak N°{peak}, D = {D} nm, $\tau$ = {tau_ps:.2f} ps, $h\nu$ = {hv:.2f} eV   (frame {i+1}/{n_tau})')
        return [line_e, line_h, *bars_e.patches, *bars_h.patches]

    anim = FuncAnimation(fig, update, frames=n_tau, init_func=init,
                         blit=False, interval=int(1000 / fps))
    writer = PillowWriter(fps=fps, metadata={'artist': 'Simulation'})
    anim.save(out_path, writer=writer, dpi=120)
    plt.close(fig)

        # (Optional) if running in a Jupyter notebook, show it inline:
    try:
        from IPython.display import Image, display, HTML
        display(HTML(f"<b>Saved GIF:</b> {out_path}"))
        display(Image(filename=out_path))
    except Exception:
        pass
