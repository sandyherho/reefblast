"""Animation 3. Canopy bubbles caught by the shock.

Keller-Miksis phase portraits of forty bubbles with equilibrium radii
0.1 to 1 mm (colour, small to large from violet to yellow) under the same
step-exponential overpressure of 10 p0 with theta = 0.12 ms, drawn in the
plane of R/R0 against the dimensionless wall speed R'/sqrt(p0/rho).  All
bubbles share physical time; small bubbles ring many times per pulse and
large ones barely complete a cycle.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from reefblast.anim import dark, fig_to_rgb, write_gif
from reefblast.bubble import Bubble, solve_km
from reefblast.scenario import PRM

THETA, P = 1.2e-4, 10.0
R0S = np.logspace(-4, -3, 40)
NT = 110
t_phys = np.linspace(0, 5 * THETA, 1200)
traj = []
for R0 in R0S:
    b = Bubble(R0=R0)
    tc = b.groups(PRM)[3]
    s, r, v, _ = solve_km(PRM, b, P, THETA / tc, t_phys[-1] / tc,
                          n_out=t_phys.size, rtol=1e-9)
    traj.append((r, v))
cmap = plt.get_cmap("plasma")
dark()
frames = []
idx = np.linspace(8, t_phys.size - 1, NT).astype(int)
for k in idx:
    fig = plt.figure(figsize=(5.4, 5.4), dpi=100)
    ax = fig.add_axes([0.1, 0.1, 0.86, 0.86])
    lo = max(0, k - 160)
    for i, (r, v) in enumerate(traj):
        col = cmap(i / (len(traj) - 1))
        ax.plot(r[lo:k], v[lo:k], color=col, lw=0.7, alpha=0.8)
        ax.plot(r[k], v[k], "o", color=col, ms=2.5)
    ax.set_xlim(0.2, 1.15)
    ax.set_ylim(-3.2, 2.2)
    ax.set_xlabel(r"$R/R_0$")
    ax.set_ylabel(r"$\dot R/\sqrt{p_0/\rho}$")
    ax.text(0.97, 0.95, f"{1e3 * t_phys[k]:.3f} ms", ha="right",
            va="top", transform=ax.transAxes)
    frames.append(fig_to_rgb(fig))
    plt.close(fig)
print(write_gif("anim03_bubbles", frames, fps=18))
