"""Figure 4. Can a canopy bubble keep up with the shock.

(a) Keller-Miksis radius of a 0.5 mm bubble, over its equilibrium
radius, under a step-exponential overpressure of 20 p0, for pulse
lengths theta f_M = 0.2, 1 and 5 (f_M the surface-tension-corrected
Minnaert frequency).  The dotted line is the static radius at the peak
pressure, which is what the Hugoniot of Figure 1 assumes.
(b) Minimum volume over the static volume at peak pressure against
theta f_M for three overpressures, the minimum located by an event on
R' = 0.  Values below one mean inertial overshoot: the step front drives
the bubble past equilibrium even when theta f_M >> 1, so the equilibrium
Hugoniot does not bound the instantaneous compaction.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from reefblast.bubble import Bubble, equilibrium_radius, minnaert_hz, \
    solve_km
from reefblast.io_utils import write_csv
from reefblast.plotting import (setup, OKABE_ITO as C, SEQ, panel_label,
                                outside_legend, handles, save)
from reefblast.scenario import PRM

BUB = Bubble()
M, WE, RE, TC = BUB.groups(PRM)
FM = minnaert_hz(PRM, BUB)


def run(P, tf, n_out=20000):
    """Radius history for overpressure P (units of p0) and theta f_M."""
    th = tf / FM / TC
    s_end = max(6.0 * th, 40.0 / (FM * TC))
    return solve_km(PRM, BUB, P, th, s_end, n_out=n_out), th


setup()
fig, (ax, axb) = plt.subplots(1, 2, figsize=(6.6, 2.8))

P_A = 20.0
req = equilibrium_radius(PRM.kappa, WE, 1.0 + P_A)
TFS = [0.2, 1.0, 5.0]
cols = [C["blue"], C["vermil"], C["black"]]
for tf, col in zip(TFS, cols):
    (s, r, v, _), th = run(P_A, tf)
    ax.plot(s / th, r, color=col, lw=1.0)
    write_csv(f"fig04a_radius_tf{tf:g}", {"t_over_theta": s / th, "r": r})
ax.axhline(req, color=C["grey"], lw=0.7, ls=":")
ax.set_xlim(0, 6)
ax.set_ylim(0, 1.25)
ax.set_xlabel(r"$t/\theta$")
ax.set_ylabel(r"$R/R_0$")
panel_label(ax, "(a)")

tfs = np.logspace(-1.3, 1.3, 22)
PS = [3.0, 10.0, 30.0]
pcol = SEQ[1:4]
tab = {"theta_fM": tfs}
for P, col in zip(PS, pcol):
    veq = equilibrium_radius(PRM.kappa, WE, 1.0 + P) ** 3
    ratio = np.array([run(P, tf, 200)[0][3] ** 3 / veq for tf in tfs])
    tab[f"Vmin_over_Veq_P{P:g}"] = ratio
    axb.semilogx(tfs, ratio, "o-", color=col, ms=3.0, mfc="none", lw=1.0)
axb.axhline(1.0, color=C["grey"], lw=0.7, ls=":")
axb.set_xlim(tfs[0], tfs[-1])
axb.set_yscale("log")
axb.set_xlabel(r"$\theta f_M$")
axb.set_ylabel(r"$V_{\min}/V_{\mathrm{eq}}$")
panel_label(axb, "(b)")
write_csv("fig04b_compaction", tab)

fig.tight_layout(w_pad=2.0)
h1, l1 = handles([rf"$\theta f_M={t:g}$" for t in TFS] + ["static"],
                 cols + [C["grey"]], ["-"] * 3 + [":"])
h2, l2 = handles([rf"$P={p:g}\,p_0$" for p in PS], pcol,
                 markers=["o"] * 3)
outside_legend(fig, h1 + h2, l1 + l2, ncol=7, y=0.02)
print(f"f_M = {FM:.1f} Hz")
print(save(fig, "fig04_bubble"))
