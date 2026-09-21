"""Animation 4. Reverberation in the reef column, and the invariant.

Goupillaud solution for water | canopy (15 cm) | plate (12 cm) | canopy,
alpha = 1e-2, incident similitude pulse of a 1 kg charge at 3 m.  Left:
the depth-time diagram revealed up to the current time.  Right: the
current pressure profile over the incident peak, and, as a bar, the
cumulative impulse carried into the lower canopy over 2 Z_c/(Z_c + Z_w),
the value any lossless stack must deliver into that half-space whatever
lies between.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt

from reefblast.anim import SIGNED, dark, fig_to_rgb, signed_power, write_gif
from reefblast.canopy import canopy_impedance, shock_speed
from reefblast.core import decay_time, peak_pressure
from reefblast.scenario import PRM, W_REF
from reefblast.stack import goupillaud, impulse_invariant

R, ALPHA = 3.0, 1e-2
P = float(peak_pressure(PRM, W_REF, R))
TH = float(decay_time(PRM, W_REF, R))
dt = TH / 60
U = float(shock_speed(PRM, ALPHA, P))
Zc = float(canopy_impedance(PRM, ALPHA, P))
layers = [(PRM.Z_w, PRM.c_l, 0.25), (Zc, U, 0.15),
          (PRM.Z_s, PRM.c_s, 0.12), (Zc, U, 0.30)]
Z, zc = [], []
z0 = 0.0
for Zl, cl, hl in layers:
    n = int(round(hl / (cl * dt)))
    Z += [Zl] * n
    zc += list(z0 + (np.arange(n) + 0.5) * cl * dt)
    z0 += n * cl * dt
Z, zc = np.array(Z), np.array(zc)
ibelow = int(np.searchsorted(zc, 0.25 + 0.15 + 0.12)) + 1
n_steps = int(40 * TH / dt) + len(Z)
tt = np.arange(n_steps) * dt
inc = P * np.exp(-tt / TH)
out = goupillaud(Z[None], inc, n_steps, probes=np.array([ibelow]),
                 field_every=1)
F = out["field"] / P
imp = np.cumsum(out["down"][0]) / inc.sum()
inv = impulse_invariant(PRM.Z_w, Zc)
zg = np.linspace(0, zc[-1], 400)
Fz = np.array([np.interp(zg, zc, f) for f in F])

dark()
frames = []
ks = np.unique(np.geomspace(4, n_steps - 1, 110).astype(int))
bounds = np.cumsum([h for _, _, h in layers])[:-1]
for k in ks:
    fig = plt.figure(figsize=(8.0, 4.0), dpi=100)
    a1 = fig.add_axes([0.07, 0.12, 0.5, 0.83])
    shown = np.full_like(Fz, np.nan)
    shown[:k] = Fz[:k]
    a1.imshow(signed_power(shown.T, 1.0), cmap=SIGNED, vmin=-1, vmax=1,
              aspect="auto", extent=[0, 1e3 * tt[-1], zg[-1], 0],
              interpolation="bilinear")
    for b in bounds:
        a1.axhline(b, color="#8793a8", lw=0.5)
    a1.set_xlim(0, 1e3 * 12 * TH)
    a1.set_xlabel("time (ms)")
    a1.set_ylabel("depth (m)")
    a2 = fig.add_axes([0.63, 0.12, 0.22, 0.83])
    a2.fill_betweenx(zg, 0, Fz[k], where=Fz[k] >= 0, color="#ff8a1f",
                     alpha=0.8, lw=0)
    a2.fill_betweenx(zg, 0, Fz[k], where=Fz[k] < 0, color="#35c3ff",
                     alpha=0.8, lw=0)
    for b in bounds:
        a2.axhline(b, color="#8793a8", lw=0.5)
    a2.set_ylim(zg[-1], 0)
    a2.set_xlim(-0.4, 1.6)
    a2.set_yticks([])
    a2.set_xlabel(r"$p/P$")
    a3 = fig.add_axes([0.9, 0.12, 0.04, 0.83])
    a3.bar([0], [imp[k] / inv], color="#56B4E9", width=0.8)
    a3.axhline(1.0, color="#fff4d6", lw=0.8, ls=":")
    a3.set_ylim(0, 1.15)
    a3.set_xticks([])
    a3.set_ylabel("impulse / invariant", fontsize=8)
    a3.yaxis.set_label_position("right")
    a3.yaxis.tick_right()
    frames.append(fig_to_rgb(fig))
    plt.close(fig)
frames += [frames[-1]] * 10
print(f"final impulse over invariant {imp[-1] / inv:.6f}")
print(write_gif("anim04_reverberation", frames, fps=16))
