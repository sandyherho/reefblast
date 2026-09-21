"""Animation 2. The reach of one charge through a day.

Plan view of a reef patch with colonies of random plate thickness (4 to
16 cm) around a 1 kg charge detonating 1.5 m above the plates.  At each
local solar time the canopy void fraction follows the diel forcing with
alpha_max = 1e-2, and each colony is coloured by the first-pass stresses
from the table of Figure 6: amber where compression exceeds sigma_c
(crushed), cyan with brightness sigma_T/sigma_t where tension exceeds
sigma_t (spalled), dim otherwise.  The inset traces alpha(t).  Requires the
cache written by fig06_diel.py.
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
rng = np.random.default_rng(11)
n = 900
r = 7.5 * np.sqrt(rng.random(n))
phi = 2 * np.pi * rng.random(n)
X, Y = r * np.cos(phi), r * np.sin(phi)
D = rng.uniform(0.04, 0.16, n)
size = 6 + 26 * (D - 0.04) / 0.12
la, lx, ld = np.log(c["alpha"]), c["x"], c["d"]


def stresses(alpha):
    """Tension and compression at every colony for one void fraction."""
    ia = np.clip(np.searchsorted(la, np.log(alpha)) - 1, 0, la.size - 2)
    fa = (np.log(alpha) - la[ia]) / (la[ia + 1] - la[ia])
    out = []
    for fld in (c["tension"], c["compression"]):
        f_a = (1 - fa) * fld[:, ia] + fa * fld[:, ia + 1]
        by_d = np.array([np.interp(r, lx, f_a[k]) for k in range(ld.size)])
        out.append(np.array([np.interp(D[i], ld, by_d[:, i])
                             for i in range(n)]))
    return out


T = np.linspace(0, 24, 97)[:-1]
AT = void_fraction(T, AM)
dark()
frames = []
for k, (tk, ak) in enumerate(zip(T, AT)):
    ten, com = stresses(ak)
    crushed = com >= PRM.sigma_c
    spalled = (ten >= PRM.sigma_t) & ~crushed
    fig = plt.figure(figsize=(5.4, 5.4), dpi=100)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.scatter(X, Y, s=size, c="#1d2433", lw=0)
    ax.scatter(X[crushed], Y[crushed], s=size[crushed], c="#ff9a2e", lw=0)
    glow = np.clip(ten[spalled] / PRM.sigma_t / 3, 0.45, 1)
    cols = np.c_[0.2 * glow, 0.75 * glow + 0.1, glow, np.ones_like(glow)]
    ax.scatter(X[spalled], Y[spalled], s=size[spalled] * 1.6,
               c=cols * [1, 1, 1, 0.35], lw=0)
    ax.scatter(X[spalled], Y[spalled], s=size[spalled], c=cols, lw=0)
    ax.plot(0, 0, marker="*", ms=12, color="#fff4d6", mec="none")
    ax.set_xlim(-7.9, 7.9)
    ax.set_ylim(-7.9, 7.9)
    ax.set_aspect("equal")
    ax.axis("off")
    hh, mm = int(tk), int(round(60 * (tk - int(tk))))
    ax.text(0.04, 0.95, f"{hh:02d}:{mm:02d}", transform=ax.transAxes,
            fontsize=13, va="top")
    ins = fig.add_axes([0.66, 0.05, 0.3, 0.16])
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
