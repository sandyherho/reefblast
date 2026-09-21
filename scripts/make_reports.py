"""Generate the plain-text reports.

Numbers are recomputed here from the package or read back from the CSV
files and caches written by the figure scripts, so every value quoted in a
report can be traced to a file under outputs/.
"""
import _bootstrap  # noqa: F401  (puts the repository root on sys.path)

import os

import numpy as np

from reefblast.bubble import Bubble, minnaert_hz
from reefblast.canopy import canopy_impedance, shock_speed, wood_speed
from reefblast.core import (crossover_pressure, crossover_range, decay_time,
                            peak_pressure)
from reefblast.io_utils import DATADIR, load_cache, write_report
from reefblast.scenario import (PRM, W_REF, H_CANOPY, D_REF, D_FAMILY,
                                Z_CHARGE, Z_REEF, ALPHA_MAX, SUB)
from reefblast.stack import impulse_invariant


def csv(stem):
    """Read a CSV written by io_utils.write_csv as a structured array."""
    return np.genfromtxt(os.path.join(DATADIR, f"{stem}.csv"),
                         delimiter=",", names=True)


def wrap(text, width=68):
    """Wrap prose to report lines indented two spaces."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append("  " + cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append("  " + cur)
    return lines


# ------------------------------------------------------------ closed forms
inv = impulse_invariant(PRM.Z_w, PRM.Z_s)
rows = ["  alpha    |  p* (MPa) | U(p*)/c_l | c_W/c_l  | R*(1 kg) m",
        "  " + "-" * 58]
for a in (1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2):
    ps = float(crossover_pressure(PRM, a))
    rows.append(f"  {a:8.1e} | {ps / 1e6:9.4f} |"
                f" {float(shock_speed(PRM, a, ps)) / PRM.c_l:9.5f} |"
                f" {float(wood_speed(PRM, a)) / PRM.c_l:8.5f} |"
                f" {float(crossover_range(PRM, W_REF, a)):9.3f}")
dc_rows = ["  R (m) | theta (ms) | P (MPa) | d_c water (cm) |"
           " d_c at alpha = 1e-3, 1e-2, 3e-2 (cm)", "  " + "-" * 76]
for R in (2.0, 5.0, 10.0):
    P = float(peak_pressure(PRM, W_REF, R))
    th = float(decay_time(PRM, W_REF, R))
    dw = PRM.c_s * th * np.arctanh(PRM.Z_w / PRM.Z_s)
    ds = [PRM.c_s * th * np.arctanh(float(canopy_impedance(PRM, a, P))
                                    / PRM.Z_s) for a in (1e-3, 1e-2, 3e-2)]
    dc_rows.append(f"  {R:5.1f} | {1e3 * th:10.4f} | {P / 1e6:7.3f} |"
                   f" {100 * dw:14.3f} | "
                   + ", ".join(f"{100 * d:6.3f}" for d in ds))
write_report("closed_forms", "Closed forms", [
    ("bubbly Hugoniot", wrap(
        "Per unit mass, v0 - v1 = v_l0 dp/K_l + v_g0 [1 - (p0/(p0 + dp))"
        "^(1/kappa)] with v_l0 = 1/rho_l and v_g0 = alpha/((1 - alpha)"
        " rho_l).  The Rayleigh line gives U^2 = v0^2 dp/(v0 - v1) and the"
        " secant impedance Z_c = U/v0.  As dp -> 0, U tends to the Wood"
        " speed; the departure is linear in dp (measured slope in the"
        " verification report).  The gas and liquid compliances are equal"
        " at p* = alpha K_l/(1 - alpha).  To leading order in p0/p* the"
        " shock speed at p* is c_l/(sqrt(2)(1 - alpha)), so the curves for"
        " alpha >= 1e-3 all cross p* near U/c_l = 0.72.")),
    ("crossover table (p0 at 5 m depth)", rows),
    ("transmission through a canopy", wrap(
        "For water | canopy | skeleton half-space and a step-exponential"
        " incident pulse, the transmitted pressure is T1 T2 sum_n q^n"
        " p_i(t - tau - 2 n tau), q = R_cs R_cw.  Just after the N-th"
        " arrival it equals T1 T2 P a_N with a_N = (r^(N+1) - q^(N+1))/"
        "(r - q), r = exp(-2 tau/theta).  The transmitted impulse is"
        " 2 Z_bot/(Z_bot + Z_w) times the incident impulse for any lossless"
        " stack between water and a half-space of impedance Z_bot; for a"
        f" skeleton half-space this is {inv:.12f}.")),
    ("spall onset", wrap(
        "Behind a plate of thickness d backed by impedance Z_c, the"
        " first-reflection stress at distance x from the back face is"
        " P_s (R_b + exp(-2 x/(c_s theta))), R_b = (Z_c - Z_s)/(Z_c + Z_s)."
        " Its tensile maximum over the plate is P_s (|R_b| - exp(-2 delta)),"
        " delta = d/(c_s theta), which is positive if and only if"
        " Z_c/Z_s < tanh(delta).  The critical thickness is therefore"
        " d_c = c_s theta artanh(Z_c/Z_s).")
     + [""] + dc_rows),
    ("bubble", wrap(
        "Linear angular frequency, dimensionless: omega_0^2 = 3 kappa"
        " (1 + We) - We.  For R0 = 0.5 mm at 5 m depth, f_M = "
        f"{minnaert_hz(PRM, Bubble()):.2f} Hz.")),
])

# ------------------------------------------------------------ verification
V = load_cache("verification")
if V is not None:
    o = V["fdtd_orders"]
    sc = V["fdtd_selfconv_rel"]
    ie = V["fdtd_impulse_err"]
    ks = V["km_shift"]
    lines = [
        f"  Goupillaud vs closed-form ray sum, max abs ........"
        f" {V['goupillaud_vs_series_max'][0]:.3e}",
        f"  Goupillaud vs transfer-matrix FFT, max abs ......."
        f" {V['goupillaud_vs_transfer_max'][0]:.3e}",
        f"  impulse invariant, Goupillaud, relative .........."
        f" {V['impulse_goupillaud_rel'][0]:.3e}",
        f"  impulse invariant, ray sum, relative ............."
        f" {V['impulse_series_rel'][0]:.3e}",
        f"  impulse invariant, transfer matrix, relative ....."
        f" {V['impulse_transfer_rel'][0]:.3e}",
        f"  peak a_N closed form vs Goupillaud, abs .........."
        f" {V['peak_closed_vs_goupillaud'][0]:.3e}",
        f"  plate first-reflection profile, max abs .........."
        f" {V['plate_first_reflection_max'][0]:.3e}",
        f"  Hugoniot to Wood, slope in dp ...................."
        f" {V['wood_limit_slope'][0]:.4f}",
        f"  Hugoniot to Wood, relative at dp = 1e-4 p0 ......."
        f" {V['wood_limit_rel_at_1e-4_p0'][0]:.3e}",
        "  FDTD successive differences, relative, dx = 4, 2, 1, 0.5 mm:",
        "      " + ", ".join(f"{v:.3e}" for v in sc),
        "  FDTD observed orders ............................."
        " " + ", ".join(f"{v:.3f}" for v in o),
        "  FDTD impulse invariant error, dx = 4, 2, 1, 0.5 mm:",
        "      " + ", ".join(f"{v:.3e}" for v in ie),
        "  Keller-Miksis frequency shift, eps = 3e-3 .. 1e-1:",
        "      " + ", ".join(f"{v:.3e}" for v in ks),
        f"  Keller-Miksis shift slope (expected 2) ..........."
        f" {V['km_shift_slope'][0]:.4f}",
        f"  period, DOP853 vs Radau, relative ................"
        f" {V['km_integrators_period_rel'][0]:.3e}",
        f"  minimum radius, DOP853 vs Radau, relative ........"
        f" {V['km_rmin_integrators_rel'][0]:.3e}",
    ]
    write_report("verification", "Numerical verification", [
        ("independent algorithms", wrap(
            "Three layered solvers share no code path: the Goupillaud"
            " equal-travel-time scattering scheme, the closed-form ray sum,"
            " and frequency-domain propagator matrices inverted by FFT."
            "  Keller-Miksis is integrated with DOP853 and cross-checked"
            " with Radau.  The two-dimensional solver is checked by"
            " self-convergence in a quasi-one-dimensional column.")),
        ("measured residuals", lines),
        ("reading the FDTD numbers", wrap(
            "The first observed order exceeds two because the coarsest grid"
            " is pre-asymptotic.  The second is close to one, as expected"
            " for arithmetic averaging of buoyancy across discontinuous"
            " material interfaces.  The impulse error does not fall with"
            " dx: it is set by truncating the reverberation tail at the"
            " end of the integration window, not by the discretization.")),
    ])

# -------------------------------------------------------------- parameters
write_report("parameters", "Symbols, units, and reference values", [
    ("units", wrap(
        "All quantities are SI.  Figures 2 and 8(a) use P and theta as"
        " units of pressure and time; Figure 4 and animation 3 scale by"
        " R0, p0 and t_c = R0 sqrt(rho/p0).  Figure 7 and animation 1 use"
        " an arbitrary source amplitude and show only ratios.")),
    ("reference values (illustrative, not calibrated)", [
        f"  rho_l = {PRM.rho_l} kg m^-3, c_l = {PRM.c_l} m s^-1,"
        f" K_l = {PRM.K_l:.4e} Pa",
        f"  p0 = {PRM.p0:.2f} Pa (canopy at {PRM.depth} m), kappa ="
        f" {PRM.kappa}",
        f"  rho_s = {PRM.rho_s} kg m^-3, c_s = {PRM.c_s} m s^-1,"
        f" Z_s = {PRM.Z_s:.4e} Pa s m^-1",
        f"  sigma_t = {PRM.sigma_t / 1e6} MPa, sigma_c ="
        f" {PRM.sigma_c / 1e6} MPa",
        f"  similitude: K_P = {PRM.K_P / 1e6} MPa, A_P = {PRM.A_P},"
        f" K_T = {1e6 * PRM.K_T} us kg^-1/3, A_T = {PRM.A_T}",
        f"  W = {W_REF} kg TNT equivalent, canopy above plate h ="
        f" {H_CANOPY} m, reference plate d = {D_REF} m",
        f"  plate family d = {D_FAMILY} m",
        f"  diel geometry: charge at {Z_CHARGE} m, plates at {Z_REEF} m;"
        f" alpha_max = {ALPHA_MAX}",
        f"  time steps per smallest decay constant: {SUB}",
        "  bubble: R0 = 0.5 mm, sigma = 0.072 N m^-1, mu = 1e-3 Pa s"]),
    ("provenance", wrap(
        "The similitude constants are the TNT values widely tabulated after"
        " Cole (1948); improvised ammonium-nitrate charges enter only"
        " through an equivalent W.  Skeleton density, sound speed and"
        " strengths are round values chosen to sit in a plausible range for"
        " porous aragonite; they are not taken from a specific measurement"
        " and every threshold result scales with them.  The canopy void"
        " fraction is a free parameter.")),
    ("what the results do not depend on", wrap(
        "The impulse invariant and the onset criterion Z_c/Z_s ="
        " tanh(delta) hold for any values of the material constants.  The"
        " crossover p* depends only on alpha and K_l.")),
])

# ------------------------------------------------------------ figure notes
notes = []
f5 = load_cache("fig05")
if f5 is not None:
    r12 = f5["rs_0.12"]
    notes += [("figure 5, damage regimes", wrap(
        f"Spall radius of the 12 cm plate rises from {r12[0]:.2f} m at"
        f" alpha = {f5['al_b'][0]:.0e} to {r12[-1]:.2f} m at alpha ="
        f" {f5['al_b'][-1]:.2e}.  Crush radius of the same plate falls from"
        f" {f5['crush'][0]:.2f} m to {f5['crush'][-1]:.2f} m.  Plate"
        " thicknesses are realised to within one skeleton cell; see the"
        " log printed by fig05_regimes.py.") + [
        "  d (m) | R_spall at alpha min | at alpha max"] + [
        f"  {d:5.2f} | {f5[f'rs_{d:g}'][0]:20.3f} |"
        f" {f5[f'rs_{d:g}'][-1]:12.3f}" for d in D_FAMILY])]
d6 = csv("fig06_diel")
lines6 = []
spall_cols = [n for n in d6.dtype.names if n.startswith("x_spall")]
crush_cols = [n for n in d6.dtype.names if n.startswith("x_crush")]
for am, cs, cc in zip(ALPHA_MAX, spall_cols, crush_cols):
    lines6.append(f"  alpha_max = {am:g}: horizontal spall radius"
                  f" {np.min(d6[cs]):.2f} to {np.max(d6[cs]):.2f} m,"
                  f" crush {np.min(d6[cc]):.2f} to {np.max(d6[cc]):.2f} m")
notes += [("figure 6, diel cycle", lines6)]
f7 = csv("fig07_ratio_percentiles")
notes += [("figure 7, two-dimensional runs", [
    "  percentile | ratio lo | ratio hi | compression/p_ref lo | hi"] + [
    f"  {r['percentile']:10.0f} | {r['ratio_lo']:8.3f} |"
    f" {r['ratio_hi']:8.3f} | {r['compression_over_pref_lo']:20.3f} |"
    f" {r['compression_over_pref_hi']:5.3f}" for r in np.atleast_1d(f7)])]
f4 = csv("fig04b_compaction")
notes += [("figure 4, bubble compaction", [
    f"  {n}: {np.min(f4[n]):.4f} to {np.max(f4[n]):.4f}"
    for n in f4.dtype.names[1:]])]
write_report("figure_notes", "Figure notes and tabulated values", [
    ("purpose", wrap(
        "Figures carry no in-panel annotation.  Values that would otherwise"
        " be printed inside an axes are recorded here.  Every panel is also"
        " available as CSV under outputs/data."))] + notes)

# -------------------------------------------------------------- open items
write_report("open_items", "Open items and negative results", [
    ("a claim that was tested and had to be corrected", wrap(
        "The project began from the expectation that the water-canopy"
        " reflection coefficient changes sign at a critical void fraction."
        "  It does not: any free gas lowers the canopy impedance below"
        " water, so that coefficient is negative for every alpha > 0.  The"
        " mechanism that survives is at the back face of the skeleton,"
        " where a low-impedance canopy makes the reflection closer to a"
        " free surface; tension appears only when Z_c/Z_s < tanh(delta).")),
    ("results that depend on an unmeasured quantity", wrap(
        "Every canopy effect requires void fractions of order 1e-3 or more."
        "  Diel oxygen bubbles are documented acoustically in seagrass"
        " meadows, but no void-fraction measurement in a coral canopy is"
        " known to the authors.  The diel results are scenarios, not"
        " predictions for a site.")),
    ("assumptions carried by the whole construction", wrap(
        "Normal incidence at every range.  Canopy impedance frozen at the"
        " secant value for the incident peak, for both loading and"
        " unloading, and for the canopy above and below the plate.  No"
        " cavitation cutoff: the surface-reflected rarefaction is omitted"
        " because bulk cavitation caps it near -p0, well below the assumed"
        " tensile strength.  Skeleton treated as an acoustic medium with no"
        " shear.  Damage judged by a single peak stress, not a cumulative"
        " criterion.")),
    ("the two-dimensional runs", wrap(
        "Line source in plane strain; the volume rate is the half-integral"
        " of the target pulse, which reproduces the pulse only in the far"
        " field and leaves a slowly decaying negative near field around the"
        " charge.  The runs are used for patterns and for the tension-to-"
        "compression ratio, which is independent of source amplitude, and"
        " never for thresholds.")),
    ("not attempted", wrap(
        "Fish mortality, the explosion gas bubble and its migration,"
        " fragmentation and rubble mobility, repeated blasting, and any"
        " calibration against field observations.")),
])
print("reports written")
