"""Animation 1. One charge, two canopies.

Pressure from the two-dimensional runs of Figure 7, gas-poor canopy on the
left (alpha = 1e-5) and gas-rich canopy on the right (alpha = 1e-2),
shown as sign(p)|p/p_ref|^(1/2) with the same p_ref in both panels.
Compression is amber, tension blue; the skeleton is outlined.  The broad
blue lobe that spreads from the charge position is the late-time near
field of a two-dimensional line source, whose pressure grows like
log r; it is common to both panels and is not a property of the reef.
Requires the cache written by fig07_fdtd.py.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from reefblast.anim import SIGNED, dark, fig_to_rgb, signed_power, write_gif
from reefblast.fdtd import SKELETON, WATER
from reefblast.io_utils import load_cache

DX, DECIM, FSCALE = 4e-3, 2, 1e5
c = load_cache("fdtd")
if c is None:
    raise SystemExit("run fig07_fdtd.py first")
lab = c["lab"]
skel = (lab == SKELETON)[::DECIM, ::DECIM]
p_ref = float(np.percentile(c["pmax_lo"][lab != WATER], 99.5))
nz, nx = skel.shape
ext = [0, nx * DX * DECIM, nz * DX * DECIM, 0]
xs = np.linspace(0, ext[1], nx)
zs = np.linspace(0, ext[2], nz)

dark()
frames = []
for k, tk in enumerate(c["times"]):
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.3), dpi=88)
    for ax, key, name in zip(axes, ("lo", "hi"),
                             (r"$\alpha=10^{-5}$", r"$\alpha=10^{-2}$")):
        f = c[f"frames_{key}"][k].astype(float) * FSCALE
        ax.imshow(signed_power(f, p_ref), cmap=SIGNED, vmin=-1, vmax=1,
                  extent=ext, interpolation="bilinear")
        ax.contour(xs, zs, skel.astype(float), levels=[0.5],
                   colors="#8793a8", linewidths=0.5)
        ax.set_xlim(0.15, 3.05)
        ax.set_ylim(2.25, 0.25)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.02, 0.96, name, transform=ax.transAxes, va="top",
                fontsize=10)
    axes[1].text(0.98, 0.96, f"{1e3 * tk:4.2f} ms", transform=axes[1]
                 .transAxes, va="top", ha="right", fontsize=9)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01,
                        wspace=0.02)
    frames.append(fig_to_rgb(fig))
    plt.close(fig)
frames += [frames[-1]] * 8
print(write_gif("anim01_diptych", frames, fps=16))
