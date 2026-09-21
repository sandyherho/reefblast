"""Figure 0. The reef column and its impedance profile.

(a) Cartoon of the configuration: a charge in the water column, the
incident front, a gas-laden canopy of thickness h over a skeletal plate of
thickness d, and canopy below the plate.
(b) Impedance against depth for a gas-free canopy and for void fraction
alpha = 1e-2, evaluated with the secant (shock) impedance at 5 MPa.  The
plate is the only layer stiffer than water, so its back face reflects with
the sign of a free surface once the canopy impedance falls far enough.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Wedge

from reefblast.canopy import canopy_impedance
from reefblast.io_utils import write_csv
from reefblast.plotting import (setup, OKABE_ITO as C, panel_label,
                                outside_legend, handles, save)
from reefblast.scenario import PRM, H_CANOPY, D_REF

setup()
fig, (ax, axb) = plt.subplots(1, 2, figsize=(6.6, 3.2),
                              gridspec_kw={"width_ratios": [1.25, 1.0]})

# ----------------------------------------------------------------- (a) cartoon
rng = np.random.default_rng(3)
ax.set_xlim(0, 1)
ax.set_ylim(1, 0)
ax.axhline(0.02, color=C["blue"], lw=1.0)
ax.add_patch(Rectangle((0, 0.55), 1, 0.20, color=C["skyblue"], alpha=0.25,
                       lw=0))
ax.add_patch(Rectangle((0, 0.63), 1, 0.07, color=C["grey"], lw=0))
ax.add_patch(Rectangle((0, 0.75), 1, 0.25, color=C["skyblue"], alpha=0.25,
                       lw=0))
for _ in range(260):
    x, z = rng.uniform(0, 1), rng.choice([rng.uniform(0.56, 0.62),
                                          rng.uniform(0.71, 0.99)])
    ax.add_patch(Circle((x, z), 0.004 + 0.004 * rng.random(), fc="white",
                        ec=C["blue"], lw=0.3))
ax.plot(0.32, 0.18, marker="*", ms=11, color=C["vermil"], mec="k", mew=0.4)
for rr, a in ((0.16, 0.9), (0.26, 0.6), (0.36, 0.35)):
    ax.add_patch(Wedge((0.32, 0.18), rr, 20, 160, width=0.006,
                       color=C["vermil"], alpha=a))
ax.annotate("", xy=(0.93, 0.55), xytext=(0.93, 0.63),
            arrowprops=dict(arrowstyle="<->", lw=0.7))
ax.annotate("", xy=(0.93, 0.63), xytext=(0.93, 0.70),
            arrowprops=dict(arrowstyle="<->", lw=0.7))
ax.text(0.955, 0.59, r"$h$", va="center", fontsize=8)
ax.text(0.955, 0.665, r"$d$", va="center", fontsize=8)
ax.text(0.03, 0.44, "water", fontsize=8)
ax.text(0.03, 0.59, r"canopy, $\alpha$", fontsize=8)
ax.text(0.03, 0.665, "plate", fontsize=8, color="white")
ax.text(0.03, 0.87, r"canopy, $\alpha$", fontsize=8)
ax.text(0.40, 0.19, r"$W$", fontsize=8)
ax.set_xticks([])
ax.set_yticks([])
panel_label(ax, "(a)")

# ------------------------------------------------------------- (b) impedance
z = np.linspace(-0.10, 0.45, 1101)
zp0, zp1 = H_CANOPY, H_CANOPY + D_REF


def profile(alpha):
    """Impedance profile for a canopy of void fraction alpha."""
    zc = float(canopy_impedance(PRM, alpha, 5e6))
    Z = np.full(z.shape, zc)
    Z[z < 0] = PRM.Z_w
    Z[(z >= zp0) & (z < zp1)] = PRM.Z_s
    return Z


tab = {"z_m": z}
cols = [C["blue"], C["vermil"]]
for a, col, ls in ((1e-5, cols[0], "-"), (1e-2, cols[1], "--")):
    Z = profile(a)
    tab[f"Z_alpha_{a:g}"] = Z
    axb.plot(Z / 1e6, z, color=col, ls=ls, lw=1.3, drawstyle="steps-post")
axb.axvline(PRM.Z_w / 1e6, color=C["grey"], lw=0.6, ls=":")
axb.set_ylim(z[-1], z[0])
axb.set_xlim(0, 5.5)
axb.set_xlabel(r"impedance $Z$ (MPa s m$^{-1}$)")
axb.set_ylabel(r"depth below canopy top (m)")
panel_label(axb, "(b)")
write_csv("fig00_impedance", tab)

h, lab = handles([r"$\alpha=10^{-5}$", r"$\alpha=10^{-2}$", r"$Z_w$"],
                 cols + [C["grey"]], ["-", "--", ":"])
fig.tight_layout(w_pad=1.5)
outside_legend(fig, h, lab, y=0.0)
print(save(fig, "fig00_schematic"))
