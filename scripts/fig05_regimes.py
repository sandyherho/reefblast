"""Figure 5. Damage regimes around a 1 kg charge.

(a) Largest first-pass tensile stress in a 12 cm plate over its tensile
strength, against slant range and void fraction, from the batched
Goupillaud solution of water | canopy | plate | canopy.  Solid, dashed
and dotted black contours are sigma_T = sigma_t for sigma_t = 2, 1 and
4 MPa.  The light-blue contour encloses compressive failure at
sigma_c = 20 MPa.  The grey line is the crossover range R*(alpha).
(b) Spall radius, the largest range at which sigma_T >= 2 MPa, against
void fraction for four plate thicknesses, and the compressive (crush)
radius of the 12 cm plate (dashed).  The shaded band lies inside the
smallest range computed, 0.7 m, where the similitude source is not used;
a curve that starts at the band edge begins inside it.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from reefblast.core import crossover_range
from reefblast.damage import damage_radius, plate_extremes
from reefblast.io_utils import save_cache, write_csv
from reefblast.plotting import (setup, OKABE_ITO as C, SEQ, panel_label,
                                outside_legend, handles, save)
from reefblast.scenario import (PRM, W_REF, H_CANOPY, D_REF, D_FAMILY,
                                ALPHA_GRID, R_GRID, SUB)

setup()
fig, (ax, axb) = plt.subplots(1, 2, figsize=(6.8, 3.0),
                              gridspec_kw={"width_ratios": [1.2, 1.0]})

RR, AA = np.meshgrid(R_GRID, ALPHA_GRID)
comp, tens, info = plate_extremes(PRM, W_REF, RR, AA, D_REF, H_CANOPY,
                                  sub=SUB)
print(f"map: dt = {info['dt']:.3e} s, d = {info['d_real']:.4f} m, "
      f"h in [{info['h_real_min']:.4f}, {info['h_real_max']:.4f}] m")
ratio = tens / PRM.sigma_t
im = ax.pcolormesh(RR, AA, np.where(ratio > 1e-3, ratio, np.nan),
                   shading="auto", cmap="magma_r", norm=LogNorm(1e-2, 10),
                   rasterized=True)
for st, ls in ((2e6, "-"), (1e6, "--"), (4e6, ":")):
    ax.contour(RR, AA, tens, levels=[st], colors="k", linestyles=ls,
               linewidths=1.0)
ax.contour(RR, AA, comp, levels=[PRM.sigma_c], colors=C["skyblue"],
           linewidths=1.4)
ax.plot(crossover_range(PRM, W_REF, ALPHA_GRID), ALPHA_GRID,
        color=C["grey"], lw=1.0)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(R_GRID[0], R_GRID[-1])
ax.set_ylim(ALPHA_GRID[0], ALPHA_GRID[-1])
ax.set_xlabel(r"slant range $R$ (m)")
ax.set_ylabel(r"void fraction $\alpha$")
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label(r"$\sigma_T/\sigma_t$")
panel_label(ax, "(a)")
write_csv("fig05a_tension_map", {"R_m": RR.ravel(), "alpha": AA.ravel(),
                                 "tension_Pa": tens.ravel(),
                                 "compression_Pa": comp.ravel()})

al = ALPHA_GRID[::2]
tab = {"alpha": al}
radii = {}
for d, col in zip(D_FAMILY, SEQ[1:]):
    RR2, AA2 = np.meshgrid(R_GRID, al)
    _, t2, inf2 = plate_extremes(PRM, W_REF, RR2, AA2, d, H_CANOPY,
                                 sub=SUB)
    rs = np.array([damage_radius(R_GRID, t2[i], PRM.sigma_t)
                   for i in range(al.size)])
    radii[d] = rs
    tab[f"spall_radius_d{d:g}"] = rs
    axb.semilogx(al, np.where(rs > 0, rs, np.nan), color=col)
    print(f"d = {d:.2f} m (realised {inf2['d_real']:.4f} m): "
          f"R_spall {rs[0]:.2f} -> {rs[-1]:.2f} m")
crush = np.array([damage_radius(R_GRID, comp[i], PRM.sigma_c)
                  for i in range(ALPHA_GRID.size)])
axb.semilogx(ALPHA_GRID, crush, color=C["skyblue"], ls="--", lw=1.4)
axb.axhspan(0, R_GRID[0], color=C["grey"], alpha=0.18, lw=0)
axb.set_xlim(al[0], al[-1])
axb.set_ylim(0, None)
axb.set_xlabel(r"void fraction $\alpha$")
axb.set_ylabel("damage radius (m)")
panel_label(axb, "(b)")
write_csv("fig05b_spall_radius", tab)
write_csv("fig05b_crush_radius", {"alpha": ALPHA_GRID, "R_crush": crush})
save_cache("fig05", R=R_GRID, alpha=ALPHA_GRID, tension=tens, comp=comp,
           crush=crush, al_b=al, **{f"rs_{d:g}": radii[d] for d in radii})

fig.tight_layout(w_pad=1.4)
h1, l1 = handles([rf"$d={100 * d:g}$ cm" for d in D_FAMILY]
                 + [r"crush, $d=12$ cm", r"$R^*$"],
                 SEQ[1:5] + [C["skyblue"], C["grey"]],
                 ["-"] * 4 + ["--", "-"])
outside_legend(fig, h1, l1, ncol=6, y=0.02)
print(save(fig, "fig05_regimes"))
