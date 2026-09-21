"""Animation 5. The critical thickness as the charge draws near.

Closed-form first-reflection tension sigma_T = P_s(|R_b| - exp(-2 delta))
over the tensile strength, in the plane of void fraction and plate
thickness, as the slant range from a 1 kg charge sweeps from 20 m to
0.7 m.  P_s = 2 Z_s P/(Z_s + Z_w) assumes a water-contact front face, so
the field is an upper bound on front loading.  The white curve is the
spall-onset thickness d_c = c_s theta artanh(Z_c/Z_s); the dashed line is
its gas-free value.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from reefblast.anim import dark, fig_to_rgb, write_gif
from reefblast.canopy import canopy_impedance
from reefblast.core import decay_time, peak_pressure
from reefblast.scenario import PRM, W_REF

al = np.logspace(-5, -1.5, 240)
dd = np.linspace(0.005, 0.25, 240)
A, DD = np.meshgrid(al, dd)
RS = np.r_[np.logspace(np.log10(20), np.log10(0.7), 84), [0.7] * 10]
dark()
frames = []
for R in RS:
    P = float(peak_pressure(PRM, W_REF, R))
    th = float(decay_time(PRM, W_REF, R))
    Zc = canopy_impedance(PRM, A, P)
    Rb = (PRM.Z_s - Zc) / (PRM.Z_s + Zc)
    Ps = 2 * PRM.Z_s * P / (PRM.Z_s + PRM.Z_w)
    sig = Ps * np.clip(Rb - np.exp(-2 * DD / (PRM.c_s * th)), 0, None)
    fig = plt.figure(figsize=(5.6, 4.4), dpi=100)
    ax = fig.add_axes([0.13, 0.13, 0.7, 0.8])
    im = ax.pcolormesh(al, 100 * dd, np.where(sig > 0, sig / PRM.sigma_t,
                                              np.nan),
                       cmap="magma", norm=LogNorm(1e-2, 30), shading="auto")
    ax.contour(al, 100 * dd, sig, levels=[PRM.sigma_t], colors="#56B4E9",
               linewidths=1.2)
    zc1 = canopy_impedance(PRM, al, P)
    ax.plot(al, 100 * PRM.c_s * th * np.arctanh(zc1 / PRM.Z_s),
            color="white", lw=1.2)
    ax.axhline(100 * PRM.c_s * th * np.arctanh(PRM.Z_w / PRM.Z_s),
               color="white", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_ylim(0.5, 25)
    ax.set_xlabel(r"void fraction $\alpha$")
    ax.set_ylabel("plate thickness (cm)")
    ax.text(0.03, 0.96, f"R = {R:5.2f} m   P = {P / 1e6:5.1f} MPa",
            transform=ax.transAxes, va="top", fontsize=9)
    cax = fig.add_axes([0.85, 0.13, 0.03, 0.8])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(r"$\sigma_T/\sigma_t$")
    frames.append(fig_to_rgb(fig))
    plt.close(fig)
print(write_gif("anim05_thickness", frames, fps=14))
