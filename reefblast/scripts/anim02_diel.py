"""Animation 2. The reach of one charge through a day.

A dial in which radius is vertical standoff between a 1 kg charge and a
plate directly beneath it (0.7 to 15 m, logarithmic) and angle is plate
thickness (4 cm at the top, increasing clockwise to 16 cm).  At each local
solar time the canopy void fraction follows the diel forcing with
alpha_max = 1e-2, and each cell is coloured from the stress table of
Figure 6: amber where compression exceeds sigma_c (crushed), cyan with
brightness sigma_T/sigma_t where tension exceeds sigma_t (spalled), dark
otherwise.  The dial is a parameter map, not a plan view of a reef.  The
inset traces alpha(t).  Requires the cache written by fig06_diel.py.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from reefblast.anim import dark, fig_to_rgb, write_gif
from reefblast.diel import void_fraction
from reefblast.io_utils import load_cache
from reefblast.scenario import PRM

c = load_cache("diel_table")
if c is None:
    raise SystemExit("run fig06_diel.py first")
AM = 1e-2
R, la, D = c["R"], np.log(c["alpha"]), c["d"]
DF = np.linspace(D[0], D[-1], 180)
RAD = np.log(R / R[0]) / np.log(R[-1] / R[0])


def state(alpha):
    """RGB field over (thickness, standoff) for one void fraction."""
    ia = np.clip(np.searchsorted(la, np.log(alpha)) - 1, 0, la.size - 2)
    fa = (np.log(alpha) - la[ia]) / (la[ia + 1] - la[ia])
    out = []
    for fld in (c["tension"], c["compression"]):
        f = (1 - fa) * fld[:, ia] + fa * fld[:, ia + 1]
        out.append(np.array([np.interp(DF, D, f[:, j])
                             for j in range(R.size)]).T)
    ten, com = out
    img = np.zeros(ten.shape + (3,))
    img[:] = (0.07, 0.09, 0.14)
    sp = ten >= PRM.sigma_t
    g = np.clip(ten / PRM.sigma_t / 3, 0.45, 1.0)
    img[sp] = np.c_[0.2 * g[sp], 0.75 * g[sp] + 0.1, g[sp]]
    img[com >= PRM.sigma_c] = (1.0, 0.6, 0.18)
    return img


NP = 520
yy, xx = np.mgrid[-1:1:NP * 1j, -1:1:NP * 1j]
RHO = np.hypot(xx, yy)
ANG = np.mod(np.arctan2(xx, yy), 2 * np.pi)          # 0 at top, clockwise
IR = np.clip(np.interp(RHO, RAD, np.arange(R.size)), 0, R.size - 1)
ID = ANG / (2 * np.pi) * (DF.size - 1)
IN = RHO <= 1.0


def to_disc(img):
    """Map a (thickness, standoff) RGB field onto the dial."""
    out = np.zeros((NP, NP, 3))
    out[:] = (0.027, 0.035, 0.059)
    i0 = np.clip(ID.astype(int), 0, DF.size - 1)
    j0 = np.clip(np.rint(IR).astype(int), 0, R.size - 1)
    out[IN] = img[i0[IN], j0[IN]]
    return out


T = np.linspace(0, 24, 97)[:-1]
AT = void_fraction(T, AM)
dark()
frames = []
for tk, ak in zip(T, AT):
    disc = to_disc(state(ak))
    fig = plt.figure(figsize=(5.4, 5.4), dpi=100)
    ax = fig.add_axes([0.04, 0.04, 0.92, 0.92])
    ax.imshow(disc, extent=[-1, 1, -1, 1], origin="lower",
              interpolation="bilinear")
    phi = np.linspace(0, 2 * np.pi, 300)
    for rr in (1.0, 2.0, 5.0, 10.0):
        x = np.log(rr / R[0]) / np.log(R[-1] / R[0])
        ax.plot(x * np.sin(phi), x * np.cos(phi), color="#3a4152", lw=0.5)
        ax.text(0.02, x + 0.01, f"{rr:g} m", fontsize=7, color="#8793a8")
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.axis("off")
    hh, mm = int(tk), int(round(60 * (tk - int(tk))))
    fig.text(0.04, 0.95, f"{hh:02d}:{mm:02d}", fontsize=13, va="top")
    fig.text(0.96, 0.95, "angle: plate 4 to 16 cm\nradius: standoff",
             fontsize=7, ha="right", va="top", color="#8793a8")
    ins = fig.add_axes([0.68, 0.04, 0.28, 0.14])
    ins.semilogy(T, AT, color="#56B4E9", lw=1.0)
    ins.plot(tk, ak, "o", color="#fff4d6", ms=4)
    ins.set_xlim(0, 24)
    ins.set_xticks([0, 12, 24])
    ins.set_yticks([1e-5, 1e-3])
    ins.tick_params(labelsize=7)
    ins.set_ylabel(r"$\alpha$", fontsize=8)
    ins.patch.set_alpha(0.0)
    frames.append(fig_to_rgb(fig))
    plt.close(fig)
print(write_gif("anim02_diel", frames, fps=12))
