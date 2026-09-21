"""Figure 1. The canopy under shock loading.

(a) Rayleigh-line shock speed of the bubbly canopy against overpressure,
for void fractions alpha = 1e-5 ... 3e-2.  Each curve starts at the Wood
speed and rises toward the liquid sound speed; the open circle marks the
crossover overpressure p* = alpha K_l/(1 - alpha), beyond which liquid
compliance dominates and the canopy becomes nearly transparent.
(b) Crossover slant range R*, at which the similitude peak pressure equals
p*, for charges W = 0.25, 1 and 4 kg TNT equivalent.  Inside R* the
canopy is effectively transparent to the shock.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from reefblast.canopy import shock_speed
from reefblast.core import crossover_pressure, crossover_range
from reefblast.io_utils import write_csv
from reefblast.plotting import (setup, OKABE_ITO as C, SEQ, panel_label,
                                outside_legend, handles, save, sci)
from reefblast.scenario import PRM, ALPHAS

setup()
fig, (ax, axb) = plt.subplots(1, 2, figsize=(6.6, 2.9))

dp = np.logspace(2, 8.5, 600)
tab = {"dp_Pa": dp}
for a, col in zip(ALPHAS, SEQ):
    U = shock_speed(PRM, a, dp)
    tab[f"U_alpha_{a:g}"] = U
    ax.semilogx(dp, U / PRM.c_l, color=col)
    ps = float(crossover_pressure(PRM, a))
    ax.plot(ps, shock_speed(PRM, a, ps) / PRM.c_l, "o", ms=4, mfc="white",
            mec=col, mew=1.0)
ax.axhline(1.0, color=C["grey"], lw=0.6, ls=":")
ax.set_xlim(dp[0], dp[-1])
ax.set_ylim(0, 1.05)
ax.set_xlabel(r"overpressure $\Delta p$ (Pa)")
ax.set_ylabel(r"shock speed $U/c_l$")
panel_label(ax, "(a)")
write_csv("fig01a_shock_speed", tab)

al = np.logspace(-5, -1.5, 300)
tab = {"alpha": al, "p_star_Pa": crossover_pressure(PRM, al)}
Ws = [0.25, 1.0, 4.0]
wcol = ["#b0b0b0", "#6a6a6a", "#000000"]
wls = [":", "--", "-"]
for W, col, ls in zip(Ws, wcol, wls):
    Rs = crossover_range(PRM, W, al)
    tab[f"R_star_W{W:g}"] = Rs
    axb.loglog(al, Rs, color=col, ls=ls)
axb.axhspan(1.0, 20.0, color=C["yellow"], alpha=0.18, lw=0)
axb.set_xlim(al[0], al[-1])
axb.set_ylim(0.3, 3e3)
axb.set_xlabel(r"void fraction $\alpha$")
axb.set_ylabel(r"crossover range $R^*$ (m)")
panel_label(axb, "(b)")
write_csv("fig01b_crossover_range", tab)

fig.tight_layout(w_pad=2.0)
h1, l1 = handles([rf"$\alpha={sci(a)}$" for a in ALPHAS], SEQ)
h2, l2 = handles([rf"$W={W:g}$ kg" for W in Ws], wcol, wls)
outside_legend(fig, h1, l1, ncol=6, y=0.02)
outside_legend(fig, h2, l2, ncol=3, y=-0.06)
print(save(fig, "fig01_canopy"))
