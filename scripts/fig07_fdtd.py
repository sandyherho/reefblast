"""Figure 7. The same charge over a gas-poor and a gas-rich canopy.

Two-dimensional linear acoustics (line source, plane strain) over a
branching thicket and a 10 cm tabular plate on a stalk, with the canopy at
alpha = 1e-5 (top row) and alpha = 1e-2 (bottom row), secant impedance at
5 MPa.  The upper boundary is absorbing, as in the one-dimensional model:
the surface-reflected rarefaction is omitted because bulk cavitation caps
it near -p0, which a linear solver cannot represent.  The skeleton is
outlined in grey.
(a, b, d, e) Pressure at two instants, shown as sign(p)|p/p_ref|^(1/2)
with p_ref the 99.5th percentile of the peak compression in the reef of
the gas-poor run; red is compression, blue tension.
Blue patches above the reef belong to the late-time near field of the
line source, identical in both runs.
(c, f) Tension-to-compression ratio in the skeleton, the largest tension
reached at each point over the largest compression reached there.  The
ratio does not depend on the arbitrary source amplitude.  Percentiles over
the emergent skeleton (above the reef base) are tabulated in
outputs/data/fig07_ratio_percentiles.csv.  The runs also produce
animation 1.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import time

import numpy as np
import matplotlib.pyplot as plt

from reefblast.core import incident
from reefblast.fdtd import SKELETON, WATER, line_source_rate, reef_scene, run
from reefblast.io_utils import load_cache, save_cache, write_csv
from reefblast.plotting import setup, DIVERGING, panel_label, save
from reefblast.scenario import PRM

DX = 4e-3
T_END = 2.4e-3
SRC = (1.1, 0.6)
THETA, RISE = 1.2e-4, 1.5e-5
ALPHA_LO, ALPHA_HI, DP = 1e-5, 1e-2, 5e6
N_FRAMES = 96
T_SNAP = (0.85e-3, 1.30e-3)
FSCALE = 1e5        # frames are cached as float16 in units of FSCALE


_TQ = np.linspace(0.0, T_END, 6001)
_QQ = line_source_rate(_TQ, incident(_TQ, 1.0, THETA, t_rise=RISE))


def volume_rate(t):
    """Line-source volume rate radiating the similitude-like pulse.

    Units are arbitrary; every plotted quantity is normalized by p_ref.
    """
    return np.interp(t, _TQ, _QQ)


lab = reef_scene(dx=DX)
cache = load_cache("fdtd")
if cache is None:
    res = {}
    for key, a in (("lo", ALPHA_LO), ("hi", ALPHA_HI)):
        t0 = time.time()
        r = run(PRM, lab, a, DP, DX, T_END, SRC, THETA, RISE,
                n_frames=N_FRAMES, free_surface=False, waveform=volume_rate)
        print(f"alpha = {a:g}: {r['nt']} steps, dt = {r['dt']:.3e} s, "
              f"{time.time() - t0:.1f} s")
        res[key] = r
    cache = {"lab": lab, "times": res["lo"]["times"]}
    for key in ("lo", "hi"):
        cache[f"frames_{key}"] = (res[key]["frames"] / FSCALE).astype(
            np.float16)
        cache[f"pmin_{key}"] = res[key]["pmin"].astype(np.float32)
        cache[f"pmax_{key}"] = res[key]["pmax"].astype(np.float32)
    save_cache("fdtd", **cache)

reef = lab != WATER
skel = lab == SKELETON
zc = (np.arange(lab.shape[0]) + 0.5) * DX
emergent = skel & (zc[:, None] < 1.97)
PCT = np.array([50.0, 90.0, 99.0])
p_ref = float(np.percentile(cache["pmax_lo"][reef], 99.5))
nz, nx = lab.shape
ext = [0, nx * DX, nz * DX, 0]
times = cache["times"]

setup()
fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.6), sharex=True, sharey=True)
labels = iter("abcdef")
tab = {"percentile": PCT}
for row, key in enumerate(("lo", "hi")):
    fr = cache[f"frames_{key}"].astype(float) * FSCALE
    for col, ts in enumerate(T_SNAP):
        k = int(np.argmin(np.abs(times - ts)))
        v = np.sign(fr[k]) * np.sqrt(np.abs(fr[k]) / p_ref)
        ax = axes[row, col]
        im = ax.imshow(v, cmap=DIVERGING, vmin=-1, vmax=1, extent=ext,
                       interpolation="bilinear", rasterized=True)
        ax.contour(np.linspace(0, nx * DX, nx), np.linspace(0, nz * DX, nz),
                   skel.astype(float), levels=[0.5], colors="0.45",
                   linewidths=0.4)
        panel_label(ax, f"({next(labels)})")
    ten = np.clip(-cache[f"pmin_{key}"].astype(float), 0, None)
    cmp_ = np.clip(cache[f"pmax_{key}"].astype(float), 1e-30, None)
    ratio = np.where(skel, ten / cmp_, np.nan)
    ax = axes[row, 2]
    im2 = ax.imshow(ratio, cmap="magma", vmin=0, vmax=0.5, extent=ext,
                    interpolation="nearest", rasterized=True)
    panel_label(ax, f"({next(labels)})")
    tab[f"ratio_{key}"] = np.percentile(ratio[emergent], PCT)
    tab[f"compression_over_pref_{key}"] = np.percentile(
        cmp_[emergent] / p_ref, PCT)
for ax in axes.ravel():
    ax.set_ylim(2.3, 1.2)
    ax.set_xlim(0.1, 3.1)
for ax in axes[1]:
    ax.set_xlabel(r"$x$ (m)")
for ax in axes[:, 0]:
    ax.set_ylabel(r"depth (m)")
fig.tight_layout(h_pad=0.8, w_pad=0.6)
cb = fig.colorbar(im, ax=axes[:, :2], location="bottom", shrink=0.6,
                  pad=0.16, aspect=40)
cb.set_label(r"$\mathrm{sign}(p)\,|p/p_{\mathrm{ref}}|^{1/2}$")
cb2 = fig.colorbar(im2, ax=axes[:, 2], location="bottom", shrink=0.9,
                   pad=0.16, aspect=20)
cb2.set_label(r"$\max(-p)\,/\max(p)$")
write_csv("fig07_ratio_percentiles", tab)
for k, v in tab.items():
    print(k, np.array2string(v, precision=3))
print(f"p_ref = {p_ref:.4e} (arbitrary source units)")
print(save(fig, "fig07_fdtd"))
