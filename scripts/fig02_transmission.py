"""Figure 2. Transmission through the canopy into a skeleton half-space.

(a) Pressure transmitted into the skeleton, over the incident peak P,
against time since first arrival, for canopy one-way travel times
tau/theta = 0, 0.25, 1 and 4 (alpha = 1e-2, secant impedance at 5 MPa).
The pulse is split into a train of reverberations of ratio
q = R_cs R_cw.
(b) Cumulative transmitted impulse over incident impulse.  Every curve
approaches the same limit 2 Z_s/(Z_s + Z_w): the canopy redistributes the
impulse in time but cannot change it.
(c) Peak transmitted pressure against tau/theta for three void fractions.
Lines are the closed form max_N T1 T2 (r^(N+1) - q^(N+1))/(r - q);
markers are the Goupillaud solver.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from reefblast.canopy import canopy_impedance
from reefblast.core import incident
from reefblast.io_utils import write_csv
from reefblast.plotting import (setup, OKABE_ITO as C, SEQ, panel_label,
                                outside_legend, handles, save, sci)
from reefblast.scenario import PRM
from reefblast.stack import (goupillaud, impulse_invariant,
                             reverberation_peak)

DP = 5e6
SUB = 200
NW = 4


def transmitted(alpha, tau_over_theta, n_steps):
    """Pressure in the first skeleton cell, incident P = 1, theta = 1."""
    Zc = float(canopy_impedance(PRM, alpha, DP))
    n_c = int(round(tau_over_theta * SUB))
    Z = np.r_[np.full(NW, PRM.Z_w), np.full(n_c, Zc), np.full(4, PRM.Z_s)]
    inc = incident(np.arange(n_steps) / SUB, 1.0, 1.0)
    out = goupillaud(Z[None], inc, n_steps, probes=np.array([NW + n_c]))
    return out["down"][0][NW + n_c:], n_c


setup()
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6))
TAUS = [0.0, 0.25, 1.0, 4.0]
cols = SEQ[:4]
n_steps = 40 * SUB
tt = np.arange(n_steps) / SUB
tab = {"t_over_theta": tt}
inv = impulse_invariant(PRM.Z_w, PRM.Z_s)
for tau, col in zip(TAUS, cols):
    ps, n_c = transmitted(1e-2, tau, n_steps + NW + 4 * SUB * 4)
    ps = ps[:n_steps]
    tab[f"p_tau{tau:g}"] = ps
    cum = np.cumsum(ps) / SUB
    tab[f"impulse_tau{tau:g}"] = cum
    axes[0].plot(tt, ps, color=col, lw=1.1)
    axes[1].plot(tt, cum, color=col, lw=1.2)
axes[0].set_xlim(0, 12)
axes[0].set_ylim(0, 1.6)
axes[0].set_xlabel(r"$(t-t_a)/\theta$")
axes[0].set_ylabel(r"$p_s/P$")
panel_label(axes[0], "(a)")
axes[1].axhline(inv, color=C["grey"], lw=0.7, ls=":")
axes[1].set_xlim(0, 40)
axes[1].set_ylim(0, 1.7)
axes[1].set_xlabel(r"$(t-t_a)/\theta$")
axes[1].set_ylabel(r"$\int p_s\,dt\,/\,P\theta$")
panel_label(axes[1], "(b)")
write_csv("fig02ab_histories", tab)

ax = axes[2]
tq = np.logspace(-2, 1.3, 300)
AL = [1e-3, 1e-2, 3e-2]
acol = [C["skyblue"], C["vermil"], C["purple"]]
tab = {"tau_over_theta": tq}
mk = {"tau_over_theta": np.array([0.02, 0.1, 0.3, 1.0, 3.0, 10.0])}
for a, col in zip(AL, acol):
    Zc = float(canopy_impedance(PRM, a, DP))
    pk = reverberation_peak(PRM.Z_w, Zc, PRM.Z_s, tq)
    tab[f"peak_alpha_{a:g}"] = pk
    ax.semilogx(tq, pk, color=col)
    num = []
    for tau in mk["tau_over_theta"]:
        ps, _ = transmitted(a, tau, int((2 * tau + 6) * SUB) + NW + 10)
        num.append(ps.max())
    mk[f"goupillaud_alpha_{a:g}"] = np.array(num)
    ax.plot(mk["tau_over_theta"], num, "o", ms=3.6, mfc="none", mec=col,
            mew=1.0)
ax.axhline(inv, color=C["grey"], lw=0.7, ls=":")
ax.set_xlim(tq[0], tq[-1])
ax.set_ylim(0, 1.6)
ax.set_xlabel(r"$\tau/\theta$")
ax.set_ylabel(r"$\max\,p_s/P$")
panel_label(ax, "(c)")
write_csv("fig02c_peak_closed_form", tab)
write_csv("fig02c_peak_goupillaud", mk)

fig.tight_layout(w_pad=1.2)
h1, l1 = handles([rf"$\tau/\theta={t:g}$" for t in TAUS], cols)
h2, l2 = handles([rf"$\alpha={sci(a)}$" for a in AL], acol)
h3, l3 = handles([r"$2Z_s/(Z_s+Z_w)$"], [C["grey"]], [":"])
outside_legend(fig, h1 + h3, l1 + l3, ncol=5, y=0.02)
outside_legend(fig, h2, l2, ncol=3, y=-0.07)
print(save(fig, "fig02_transmission"))
