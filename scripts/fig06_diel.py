"""Figure 6. A diel cycle in the reach of one charge.

A 1 kg charge detonates at 3.5 m depth, 1.5 m above plates at 5 m depth,
as for a thrown bottle charge that sinks before its fuse fires.  The
canopy void fraction follows the prescribed diel forcing of
:mod:`reefblast.diel` with peak values alpha_max = 1e-3, 1e-2 and 3e-2.
(a) Void fraction against local solar time.
(b) Horizontal spall radius of a 12 cm plate (solid) and horizontal crush
radius (dashed, alpha_max = 3e-2, the scenario in which it varies most).
Both are read from a table of first-pass plate stresses
computed on a grid of range, void fraction and plate thickness; the same
table drives animation 2.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from reefblast.damage import damage_radius, plate_extremes
from reefblast.diel import void_fraction
from reefblast.io_utils import save_cache, write_csv
from reefblast.plotting import (setup, OKABE_ITO as C, panel_label,
                                outside_legend, handles, save, sci)
from reefblast.scenario import (PRM, W_REF, H_CANOPY, D_REF, Z_CHARGE,
                                Z_REEF, ALPHA_MAX, SUB)

HV = Z_REEF - Z_CHARGE
X = np.linspace(0.0, 15.0, 61)
R = np.sqrt(X ** 2 + HV ** 2)
A_TAB = np.logspace(-5, -1.5, 36)
D_TAB = np.array([0.06, 0.08, 0.10, 0.12, 0.14, 0.16])
T_H = np.linspace(0.0, 24.0, 97)

tens = np.empty((D_TAB.size, A_TAB.size, R.size))
comp = np.empty_like(tens)
RR, AA = np.meshgrid(R, A_TAB)
for k, d in enumerate(D_TAB):
    comp[k], tens[k], _ = plate_extremes(PRM, W_REF, RR, AA, d, H_CANOPY,
                                         sub=SUB)
save_cache("diel_table", x=X, R=R, alpha=A_TAB, d=D_TAB, tension=tens,
           compression=comp, hv=HV)


def at_alpha(field, alpha):
    """Interpolate a (n_alpha, n_R) field to one alpha, linear in log."""
    la = np.log(A_TAB)
    j = np.clip(np.searchsorted(la, np.log(alpha)) - 1, 0, la.size - 2)
    f = (np.log(alpha) - la[j]) / (la[j + 1] - la[j])
    return (1 - f) * field[j] + f * field[j + 1]


def horizontal(r):
    """Horizontal radius from a slant radius."""
    return np.sqrt(max(r ** 2 - HV ** 2, 0.0)) if r > 0 else 0.0


setup()
fig, (ax, axb) = plt.subplots(2, 1, figsize=(3.6, 4.6), sharex=True)
k12 = int(np.argmin(np.abs(D_TAB - D_REF)))
cols = [C["green"], C["vermil"], C["purple"]]
tab = {"t_hours": T_H}
for am, col in zip(ALPHA_MAX, cols):
    a = void_fraction(T_H, am)
    xs = np.array([horizontal(damage_radius(R, at_alpha(tens[k12], ai),
                                            PRM.sigma_t)) for ai in a])
    xc = np.array([horizontal(damage_radius(R, at_alpha(comp[k12], ai),
                                            PRM.sigma_c)) for ai in a])
    tab[f"alpha_max{am:g}"] = a
    tab[f"x_spall_max{am:g}"] = xs
    tab[f"x_crush_max{am:g}"] = xc
    ax.semilogy(T_H, a, color=col)
    axb.plot(T_H, xs, color=col)
    print(f"alpha_max = {am:g}: spall radius {xs.min():.2f} .. "
          f"{xs.max():.2f} m, crush {xc.min():.2f} .. {xc.max():.2f} m")
axb.plot(T_H, xc, color=C["black"], ls="--", lw=1.0)
ax.set_ylabel(r"void fraction $\alpha$")
ax.set_ylim(5e-6, 6e-2)
panel_label(ax, "(a)")
axb.set_xlim(0, 24)
axb.set_xticks([0, 6, 12, 18, 24])
axb.set_ylim(0, None)
axb.set_xlabel("local solar time (h)")
axb.set_ylabel("horizontal radius (m)")
panel_label(axb, "(b)")
write_csv("fig06_diel", tab)

fig.tight_layout(h_pad=1.2)
h, lab = handles([rf"$\alpha_{{\max}}={sci(a)}$" for a in ALPHA_MAX]
                 + ["crush"], cols + [C["black"]], ["-"] * 3 + ["--"])
outside_legend(fig, h, lab, ncol=2, y=0.0)
print(save(fig, "fig06_diel"))
