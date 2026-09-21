"""Figure 3. When a plate spalls.

(a) First-reflection tensile peak in a plate over the transmitted peak,
sigma_T/P_s = |R_b| - exp(-2 delta), against delta = d/(c_s theta) and
the impedance ratio Z_c/Z_s of the medium behind the plate.  Tension
exists only below the curve Z_c/Z_s = tanh(delta) (black).  The dashed
line is a water-backed plate.
(b) Critical plate thickness d_c = c_s theta artanh(Z_c/Z_s) against void
fraction, for a 1 kg charge at three slant ranges.  Plates thinner than
d_c carry no first-reflection tension.  Dashed lines are the gas-free
limit at each range.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from reefblast.canopy import canopy_impedance
from reefblast.core import decay_time, peak_pressure
from reefblast.io_utils import write_csv
from reefblast.plotting import (setup, OKABE_ITO as C, panel_label,
                                outside_legend, handles, save)
from reefblast.scenario import PRM, W_REF

setup()
fig, (ax, axb) = plt.subplots(1, 2, figsize=(6.8, 2.9),
                              gridspec_kw={"width_ratios": [1.15, 1.0]})

delta = np.logspace(-2, 0.3, 300)
zr = np.logspace(-2.3, 0, 300)
D, ZR = np.meshgrid(delta, zr)
Rb = (1 - ZR) / (1 + ZR)
s = np.clip(Rb - np.exp(-2 * D), 0, None)
im = ax.pcolormesh(D, ZR, np.where(s > 0, s, np.nan), shading="auto",
                   cmap="magma_r", norm=LogNorm(1e-3, 1.0), rasterized=True)
ax.plot(delta, np.tanh(delta), color="k", lw=1.3)
ax.axhline(PRM.Z_w / PRM.Z_s, color=C["grey"], lw=1.0, ls="--")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(delta[0], delta[-1])
ax.set_ylim(zr[0], zr[-1])
ax.set_xlabel(r"$\delta=d/(c_s\theta)$")
ax.set_ylabel(r"$Z_c/Z_s$")
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label(r"$\sigma_T/P_s$")
panel_label(ax, "(a)")
write_csv("fig03a_onset_curve", {"delta": delta, "tanh_delta":
                                 np.tanh(delta)})

al = np.logspace(-5, -1.5, 300)
RS = [2.0, 5.0, 10.0]
cols = [C["vermil"], C["blue"], C["black"]]
tab = {"alpha": al}
for R, col in zip(RS, cols):
    P = float(peak_pressure(PRM, W_REF, R))
    th = float(decay_time(PRM, W_REF, R))
    Zc = canopy_impedance(PRM, al, P)
    dc = PRM.c_s * th * np.arctanh(Zc / PRM.Z_s)
    dw = PRM.c_s * th * np.arctanh(PRM.Z_w / PRM.Z_s)
    tab[f"d_c_R{R:g}"] = dc
    axb.semilogx(al, 100 * dc, color=col)
    axb.axhline(100 * dw, color=col, lw=0.7, ls="--")
axb.set_xlim(al[0], al[-1])
axb.set_ylim(0, 20)
axb.set_xlabel(r"void fraction $\alpha$")
axb.set_ylabel(r"critical thickness $d_c$ (cm)")
panel_label(axb, "(b)")
write_csv("fig03b_critical_thickness", tab)

fig.tight_layout(w_pad=1.5)
h1, l1 = handles([r"$Z_c/Z_s=\tanh\delta$", "water backing"],
                 ["k", C["grey"]], ["-", "--"])
h2, l2 = handles([rf"$R={R:g}$ m" for R in RS], cols)
outside_legend(fig, h1 + h2, l1 + l2, ncol=5, y=0.02)
print(save(fig, "fig03_spall"))
