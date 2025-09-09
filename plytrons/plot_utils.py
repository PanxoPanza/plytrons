import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

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

def hot_carriers_plot(Te_raw, Th_raw, e_states, tau_e, D, hv, EF, dE_factor, delta, DO_CHECKS = True):
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

    Te = Te_raw[0]
    Th = Th_raw[0]

    # ---- load energies in the SAME order as generation arrays ----
    e_state = []
    for e in range(len(e_states)):
        e_state.append(e_states[e].Eb)
    E_bound = [row[row != 0] for row in e_state]
    E_all = np.concatenate(E_bound).real


    # sanity check
    assert len(E_all) == len(Te) == len(Th), "Length mismatch between E_all and Te/Th."

    # ---- smooth onto a fine energy grid ----
    x, Te_x, Th_x = convert_raw_hot(Te, Th, E_all, dE_factor)

    # ---- convert units: fs^-1 → ps^-1, and divide by hv to get per eV ----
    to_ps = 1000
    scale = to_ps / hv
    Te_x *= scale
    Th_x *= scale

    # ---- masks for plotting: electrons (E>=EF), holes (E<=EF), within ±delta ----
    mask_e = (x >= EF) & (x <= EF + delta)
    mask_h = (x <= EF) & (x >= EF - delta)

    # ---- (optional) quick checks: e/h balance and leakage beyond ±hv ----
    if DO_CHECKS:
        from numpy import trapz
        e_win = (x >= EF) & (x <= EF + hv)
        h_win = (x >= EF - hv) & (x <= EF)
        N_e = trapz(Te_x[e_win], x[e_win])
        N_h = trapz(Th_x[h_win], x[h_win])
        imb = 100.0 * abs(N_e - N_h) / max(N_e, N_h) if max(N_e, N_h) > 0 else 0.0

        tot_e = trapz(Te_x, x); tot_h = trapz(Th_x, x)
        leak_e = max(tot_e - trapz(Te_x[e_win], x[e_win]), 0.0)
        leak_h = max(tot_h - trapz(Th_x[h_win], x[h_win]), 0.0)
        leak_e_pct = 100.0 * leak_e / (tot_e + 1e-20)
        leak_h_pct = 100.0 * leak_h / (tot_h + 1e-20)

        print(f"[check] N_e = {N_e:.4g}, N_h = {N_h:.4g}, imbalance = {imb:.2f}%")
        print(f"[check] leakage e (outside [EF, EF+hv]): {leak_e_pct:.2f}%")
        print(f"[check] leakage h (outside [EF-hv, EF]): {leak_h_pct:.2f}%")

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(20, 4.5))
    ax.plot((x - EF)[mask_e], Te_x[mask_e], color='r', alpha=1, label='Electrons')
    ax.plot((x - EF)[mask_h], Th_x[mask_h], color='b', alpha=1, label='Holes')

    # guides
    ax.axvline(0.0, ls='--', lw=1, color='k', alpha=0.5)
    ax.axvline(+hv, ls='--', lw=1, color='gray', alpha=0.6)
    ax.axvline(-hv, ls='--', lw=1, color='gray', alpha=0.6)

    ax.set_xlim(-delta, delta)
    ax.set_xlabel('Hot carrier energy relative to $E_F$ (eV)')
    ax.set_ylabel(r'Hot carriers per energy $\;[\mathrm{eV}^{-1}\,\mathrm{ps}^{-1}\,\mathrm{nm}^{-3}]$')
    ax.set_title(rf'D = {D} nm, $\tau = ${tau_e/1000:.2f} ps, $h\nu$ = {hv:.2f} eV')
    ax.grid(True, ls=':')
    ax.legend()
    plt.tight_layout()
    plt.show()

def hot_carrier_dynamics_plot(Te, Th, e_states, tau_e, D, hv, EF, dE_factor, delta, fps, out_path, use_raw):


    # ---- 1) Build the bound-state energies in the SAME order as the generation arrays ----
    # e_states must be an iterable of objects each having .Eb (array of bound energies)
    E_bound = [es.Eb for es in e_states]                    # list of arrays
    E_all = np.concatenate([row[row != 0] for row in E_bound]).real

    Te_arr = Te
    Th_arr = Th 
    tau_arr = np.asarray(tau_e, dtype=float)  # fs

    # Basic sanity
    n_tau, N = Te_arr.shape[0], Te_arr.shape[1]
    assert Th_arr.shape == (n_tau, N), "Th array shape mismatch with Te."
    assert E_all.size == N, "E_all size must match the number of states N."

    # ---- 3) Pre-pass to compute a global YMAX so y-axis is fixed ----
    to_ps = 1000.0  # fs -> ps
    def frame_max(i):
        Te_i = Te_arr[i]
        Th_i = Th_arr[i]
        x, Te_x, Th_x = convert_raw_hot(Te_i, Th_i, E_all, dE_factor)

        # Convert units: fs^-1 → ps^-1 and per eV (divide by hv)
        scale = to_ps / hv
        Te_x = Te_x * scale
        Th_x = Th_x * scale

        # Masks: electrons (E >= EF), holes (E <= EF) within ±delta window
        mask_e = (x >= EF) & (x <= EF + delta)
        mask_h = (x <= EF) & (x >= EF - delta)

        m_e = np.max(Te_x[mask_e]) if np.any(mask_e) and Te_x[mask_e].size else 0.0
        m_h = np.max(Th_x[mask_h]) if np.any(mask_h) and Th_x[mask_h].size else 0.0
        return max(m_e, m_h)

    YMAX = 0.0
    for i in range(n_tau):
        YMAX = max(YMAX, frame_max(i))
    YMAX = 1.05 * YMAX if YMAX > 0 else 1.0  # small headroom

    # ---- 4) Prepare figure, fixed axes, and artists ----
    fig, ax = plt.subplots(figsize=(20, 4.5))
    (line_e,) = ax.plot([], [], lw=1.8, label="Electrons")  # E >= EF
    (line_h,) = ax.plot([], [], lw=1.8, label="Holes")      # E <= EF

    # Guides (static)
    ax.axvline(0.0, ls='--', lw=1, alpha=0.5)
    ax.axvline(+hv, ls='--', lw=1, alpha=0.6)
    ax.axvline(-hv, ls='--', lw=1, alpha=0.6)

    ax.set_xlim(-delta, delta)
    ax.set_ylim(0, YMAX)  # ← fixed y-axis
    ax.set_xlabel('Hot carrier energy relative to $E_F$ (eV)')
    ax.set_ylabel(r'Hot carriers per energy $\;[\mathrm{eV}^{-1}\,\mathrm{ps}^{-1}\,\mathrm{nm}^{-3}]$')
    ax.grid(True, ls=':')
    ax.legend()

    # ---- 5) Helpers for each frame ----
    def make_frame_data(i):
        Te_i = Te_arr[i]
        Th_i = Th_arr[i]

        x, Te_x, Th_x = convert_raw_hot(Te_i, Th_i, E_all, dE_factor)

        # Units: fs^-1 → ps^-1 and divide by hv to get per eV
        scale = to_ps / hv
        Te_x = Te_x * scale
        Th_x = Th_x * scale

        mask_e = (x >= EF) & (x <= EF + delta)
        mask_h = (x <= EF) & (x >= EF - delta)

        return (x - EF)[mask_e], Te_x[mask_e], (x - EF)[mask_h], Th_x[mask_h]

    def init():
        line_e.set_data([], [])
        line_h.set_data([], [])
        ax.set_title('')
        return (line_e, line_h)

    def update(i):
        xe, ye, xh, yh = make_frame_data(i)
        line_e.set_data(xe, ye)
        line_h.set_data(xh, yh)
        tau_ps = tau_arr[i] / 1000.0
        ax.set_title(rf'D = {D} nm, $\tau$ = {tau_ps:.2f} ps, $h\nu$ = {hv:.2f} eV   (frame {i+1}/{n_tau})')
        return (line_e, line_h)

    # ---- 6) Build & save animation ----
    anim = FuncAnimation(fig, update, frames=n_tau, init_func=init, blit=True, interval=int(1000 / fps))
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
