"""Figure 8. Verification.

(a) Transmitted pressure in a skeleton half-space below a canopy: absolute
difference between the Goupillaud solver and the closed-form ray sum, and
between the Goupillaud solver and the transfer-matrix FFT solution, for a
smooth two-exponential pulse of unit peak.  The dotted line is machine
epsilon; the series residual is exactly zero at most samples.
(b) Relative departure of the Rayleigh-line shock speed from the Wood
speed as the overpressure vanishes, for three void fractions.  The grey
line has slope one.
(c) Self-convergence of the two-dimensional solver in a quasi-one-
dimensional column: maximum difference between successive resolutions
(circles), with reference slopes one and two, and error of the
transmitted-impulse invariant (squares), which is independent of
resolution and set by the finite integration window.
(d) Relative departure of the Keller-Miksis oscillation frequency
(inviscid, incompressible limit) from the linear frequency about the
equilibrium displaced by a pressure step of size epsilon.  The grey line
has slope two, the order of the leading nonlinear correction.
Every number is also written to the verification report.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from reefblast.bubble import (Bubble, equilibrium_radius, km_rhs,
                              linear_frequency, solve_km)
from reefblast.canopy import (canopy_impedance, reflection, shock_speed,
                              wood_speed)
from reefblast.core import incident
from reefblast.fdtd import plane_column, run
from reefblast.io_utils import load_cache, save_cache, write_csv
from reefblast.plotting import (setup, OKABE_ITO as C, panel_label,
                                outside_legend, handles, save, sci)
from reefblast.scenario import PRM
from reefblast.stack import (goupillaud, impulse_invariant, plate_tension,
                             reverberation_peak, reverberation_series,
                             transfer_matrix)

V = {}
INV = impulse_invariant(PRM.Z_w, PRM.Z_s)

# ------------------------------------------------ (a) three layered solvers
THETA, SUB, NW, NC = 1.0, 200, 4, 150
dt = THETA / SUB
a_v, dp_v = 3e-3, 5e6
U = float(shock_speed(PRM, a_v, dp_v))
Zc = float(canopy_impedance(PRM, a_v, dp_v))
n_steps = 8000
t = np.arange(n_steps) * dt
inc = incident(t, 1.0, THETA, t_rise=THETA / 10)
Z = np.r_[np.full(NW, PRM.Z_w), np.full(NC, Zc), np.full(50, PRM.Z_s)]
g = goupillaud(Z[None], inc, n_steps, probes=np.array([NW + NC]))["down"][0]
ser = reverberation_series(inc, PRM.Z_w, Zc, PRM.Z_s, NW + NC, 2 * NC)
tm = transfer_matrix(inc, dt, [(Zc, U, NC * U * dt)], PRM.Z_w, PRM.Z_s)
e_ser = np.abs(g - ser)
e_tm = np.abs(g[NW:] - tm[:n_steps - NW])
V["goupillaud_vs_series_max"] = e_ser.max()
V["goupillaud_vs_transfer_max"] = e_tm.max()
V["impulse_goupillaud_rel"] = abs(g.sum() / inc.sum() / INV - 1)
V["impulse_transfer_rel"] = abs(tm.sum() / inc.sum() / INV - 1)
V["impulse_series_rel"] = abs(ser.sum() / inc.sum() / INV - 1)
step = incident(t, 1.0, THETA)
gs = goupillaud(Z[None], step, n_steps, probes=np.array([NW + NC]))
V["peak_closed_vs_goupillaud"] = abs(
    gs["down"][0].max()
    - reverberation_peak(PRM.Z_w, Zc, PRM.Z_s, NC * dt / THETA)[0])
n_p = 40
Zb = float(canopy_impedance(PRM, 1e-2, dp_v))
Zp = np.r_[np.full(NW, PRM.Z_w), np.full(n_p, PRM.Z_s), np.full(20, Zb)]
res = goupillaud(np.tile(Zp, (n_p, 1)), step, 600,
                 probes=np.arange(NW, NW + n_p))
j = np.arange(n_p)
L = 2 * (n_p - 1 - j) + 1
Ps = 1 + reflection(PRM.Z_w, PRM.Z_s)
pred = plate_tension(Ps, reflection(PRM.Z_s, Zb), L * dt / (2 * THETA))
V["plate_first_reflection_max"] = np.max(np.abs(res["p"][j, NW + j + L]
                                                - pred))
write_csv("fig08a_three_solvers", {"t_over_theta": t, "goupillaud": g,
                                   "series": ser,
                                   "transfer_shifted": np.r_[
                                       np.full(NW, np.nan),
                                       tm[:n_steps - NW]]})

# ---------------------------------------------------- (b) Hugoniot to Wood
dps = np.logspace(-4, 1, 60) * PRM.p0
AW = [1e-4, 1e-3, 1e-2]
tab = {"dp_over_p0": dps / PRM.p0}
for a in AW:
    tab[f"rel_alpha_{a:g}"] = np.abs(shock_speed(PRM, a, dps)
                                     / wood_speed(PRM, a) - 1)
sl = np.polyfit(np.log(dps[:15]), np.log(tab["rel_alpha_0.001"][:15]), 1)[0]
V["wood_limit_slope"] = sl
V["wood_limit_rel_at_1e-4_p0"] = tab["rel_alpha_0.001"][0]
write_csv("fig08b_wood_limit", tab)

# ----------------------------------------------- (c) FDTD self-convergence
DXS = [4e-3, 2e-3, 1e-3, 5e-4]
Z0, LZ, TEND = 3.0, 10.0, 3.6e-3
tg = np.linspace(0, TEND, 4001)


def wf(tt):
    """Smooth source for the convergence study."""
    return np.exp(-((tt - 1e-4) / 3e-5) ** 2)


hist, imp = {}, {}
for dx in DXS:
    stem = f"fdtd_conv_{dx:.1e}"
    got = load_cache(stem)
    if got is not None:
        hist[dx], imp[dx] = got["hist"], float(got["imp"][0])
        continue
    w = int(round(0.16 / dx))
    kw = dict(free_surface=False, sponge_w=w, periodic_x=True,
              plane_source_row=int(round(Z0 / dx)),
              sponge_strength=0.015 * 40 / w, waveform=wf)
    imps = {}
    for name, lab in (("w", plane_column(LZ, dx, Z0 + 0.3, 0.0, None)),
                      ("l", plane_column(LZ, dx, Z0 + 0.3, 0.1,
                                         Z0 + 0.45))):
        r = run(PRM, lab, 1e-2, 5e6, dx, TEND, None, 1e-4, 1e-5,
                probes=[(int(round((Z0 + 0.7) / dx)), 0)], **kw)
        tt = np.arange(r["nt"]) * r["dt"]
        imps[name] = np.trapezoid(r["probes"][0], tt)
        if name == "l":
            hist[dx] = np.interp(tg, tt, r["probes"][0])
    imp[dx] = abs(imps["l"] / imps["w"] / INV - 1)
    save_cache(stem, hist=hist[dx], imp=np.array([imp[dx]]))
    print(f"fdtd dx = {dx:.2e}: impulse error {imp[dx]:.3e}", flush=True)
diffs = np.array([np.max(np.abs(hist[DXS[i]] - hist[DXS[i + 1]]))
                  for i in range(len(DXS) - 1)])
scale = np.max(np.abs(hist[DXS[-1]]))
orders = np.log2(diffs[:-1] / diffs[1:])
V["fdtd_selfconv_rel"] = diffs / scale
V["fdtd_orders"] = orders
V["fdtd_impulse_err"] = np.array([imp[d] for d in DXS])
write_csv("fig08c_fdtd_convergence",
          {"dx_m": np.array(DXS[:-1]), "succ_diff_rel": diffs / scale,
           "impulse_err": np.array([imp[d] for d in DXS[:-1]])})

# ---------------------------------------------- (d) Keller-Miksis frequency
BUB = Bubble()
M, WE, RE, TC = BUB.groups(PRM)
w0 = linear_frequency(PRM.kappa, WE)
EPS = np.array([3e-3, 1e-2, 3e-2, 1e-1])


def km_period(eps, method):
    """Period from successive minima under a small step of size eps."""
    def turn(s, y, *a):
        return y[1]
    turn.direction = 1.0
    sol = solve_ivp(km_rhs, (0, 120.0), [1.0, 0.0], method=method,
                    rtol=1e-12, atol=1e-14, events=turn,
                    args=(PRM.kappa, 0.0, WE, np.inf,
                          lambda s: 1.0 + eps))
    te = sol.t_events[0]
    return np.mean(np.diff(te)), sol.y_events[0][:, 0]


def omega_about(eps):
    """Linear frequency about the equilibrium displaced by the step."""
    re = equilibrium_radius(PRM.kappa, WE, 1.0 + eps)
    dpb = (-3.0 * PRM.kappa * (1.0 + WE) * re ** (-3.0 * PRM.kappa - 1.0)
           + WE / re ** 2)
    return np.sqrt(-dpb / re)


shift = []
for e in EPS:
    per, _ = km_period(e, "DOP853")
    shift.append(abs(2 * np.pi / per / omega_about(e) - 1))
shift = np.array(shift)
V["km_shift"] = shift
V["km_shift_slope"] = np.polyfit(np.log(EPS), np.log(shift), 1)[0]
p1, m1 = km_period(1e-2, "DOP853")
p2, m2 = km_period(1e-2, "Radau")
V["km_integrators_period_rel"] = abs(p1 / p2 - 1)
_, _, _, rmin_a = solve_km(PRM, BUB, 20.0, 5.0, 60.0, n_out=10,
                           method="DOP853", rtol=1e-11)
_, _, _, rmin_b = solve_km(PRM, BUB, 20.0, 5.0, 60.0, n_out=10,
                           method="Radau", rtol=1e-11)
V["km_rmin_integrators_rel"] = abs(rmin_a / rmin_b - 1)
write_csv("fig08d_km_frequency", {"eps": EPS, "rel_shift": shift})

save_cache("verification", **{k: np.atleast_1d(v) for k, v in V.items()})
for k, v in V.items():
    print(f"{k}: {np.array2string(np.atleast_1d(v), precision=4)}")

# ------------------------------------------------------------------- figure
setup()
fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.0))
ax = axes[0, 0]
ax.semilogy(t[NW:], np.maximum(e_tm, 1e-19), color=C["vermil"], lw=0.7)
ax.semilogy(t, np.maximum(e_ser, 1e-19), color=C["purple"], lw=0.9)
ax.axhline(np.finfo(float).eps, color=C["grey"], lw=0.7, ls=":")
ax.set_xlim(0, 30)
ax.set_ylim(1e-19, 1e-12)
ax.set_xlabel(r"$t/\theta$")
ax.set_ylabel("absolute difference")
panel_label(ax, "(a)")

ax = axes[0, 1]
wc = [C["skyblue"], C["blue"], C["black"]]
for a, col in zip(AW, wc):
    ax.loglog(dps / PRM.p0, tab[f"rel_alpha_{a:g}"], color=col)
ref = dps[:20] / PRM.p0
ax.loglog(ref, 0.3 * ref, color=C["grey"], lw=0.7, ls=":")
ax.set_xlabel(r"$\Delta p/p_0$")
ax.set_ylabel(r"$|U/c_W-1|$")
panel_label(ax, "(b)")

ax = axes[1, 0]
dxa = np.array(DXS[:-1])
ax.loglog(dxa * 1e3, diffs / scale, "o-", color=C["black"], mfc="none",
          ms=4)
ax.loglog(np.array(DXS) * 1e3, [imp[d] for d in DXS], "s-",
          color=C["green"], mfc="none", ms=4)
xr = np.array([0.5, 4.0])
ax.loglog(xr, 0.02 * xr, color=C["grey"], lw=0.7, ls=":")
ax.loglog(xr, 0.002 * xr ** 2, color=C["grey"], lw=0.7, ls="--")
ax.set_xticks([0.5, 1, 2, 4])
ax.set_xticklabels(["0.5", "1", "2", "4"])
ax.minorticks_off()
ax.set_xlabel(r"$\Delta x$ (mm)")
ax.set_ylabel("relative error")
panel_label(ax, "(c)")

ax = axes[1, 1]
ax.loglog(EPS, shift, "o-", color=C["vermil"], mfc="none", ms=4)
ax.loglog(EPS, shift[1] * (EPS / EPS[1]) ** 2, color=C["grey"], lw=0.7,
          ls=":")
ax.set_xlabel(r"step amplitude $\varepsilon$")
ax.set_ylabel(r"$|\omega/\omega_{\mathrm{lin}}-1|$")
panel_label(ax, "(d)")

fig.tight_layout(h_pad=1.4, w_pad=1.6)
h, lab = handles(["Goupillaud - series", "Goupillaud - transfer",
                  rf"$\alpha={sci(1e-4)}$", rf"$\alpha={sci(1e-3)}$",
                  rf"$\alpha={sci(1e-2)}$", "successive", "impulse",
                  "Keller-Miksis", "reference slope"],
                 [C["purple"], C["vermil"]] + wc
                 + [C["black"], C["green"], C["vermil"], C["grey"]],
                 ["-"] * 5 + ["-", "-", "-", ":"],
                 [None] * 5 + ["o", "s", "o", None])
outside_legend(fig, h, lab, ncol=3, y=0.0)
print(save(fig, "fig08_verification"))
