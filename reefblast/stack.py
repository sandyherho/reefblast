"""Normal-incidence wave transmission through a layered reef column.

Three algorithms with no shared code path are provided.

``goupillaud``
    Exact discrete scattering in a medium of equal-travel-time cells.  Every
    delay is an integer number of steps, so for layer thicknesses that are
    integer numbers of cells the sampled solution carries no discretization
    error.  The solver is batched: row ``b`` of the impedance array is an
    independent column, so a whole parameter sweep advances in one loop.

``reverberation_series``
    The closed-form ray sum for water | canopy | skeleton half-space,

        p_s(t) = T1 T2 sum_n q^n p_i(t - tau - 2 n tau),
        q = R_cs R_cw,

    with R_cs = (Z_s - Z_c)/(Z_s + Z_c) and R_cw = (Z_w - Z_c)/(Z_w + Z_c).

``transfer_matrix``
    Frequency-domain propagator matrices and radiation conditions, solved as
    a 2 x 2 linear system at each frequency and inverted by FFT.

Closed forms for the peak and impulse of the transmitted pulse and for the
first-reflection tension in a plate are also given here.
"""

import numpy as np

from .canopy import canopy_impedance, reflection, shock_speed

__all__ = ["goupillaud", "reverberation_series", "transfer_matrix",
           "reverberation_peak", "impulse_invariant", "plate_tension",
           "build_column", "column_extremes"]


# --------------------------------------------------------------- goupillaud
def goupillaud(Z, inc, n_steps, probes=None, mask=None, field_every=0):
    """Advance down- and up-going pressure waves through a cell column.

    Parameters
    ----------
    Z : array (B, N)
        Impedance of each cell.  Cell 0 is the incidence medium; waves
        leaving the top of cell 0 or the bottom of cell N-1 are absorbed.
    inc : array (n_steps,) or (B, n_steps)
        Pressure injected as the down-going wave into cell 0.
    n_steps : int
        Number of time steps.
    probes : array (B,) of int, optional
        Cell index per row whose down-going, up-going and total pressure
        histories are returned.
    mask : bool array (B, N), optional
        Cells over which the running minimum and maximum of the total
        pressure are accumulated.
    field_every : int, optional
        If positive, the total pressure of row 0 in every cell is stored
        every ``field_every`` steps under key ``field``.

    Returns
    -------
    dict with keys ``down``, ``up``, ``p`` (each (B, n_steps), present when
    ``probes`` is given), ``pmin`` and ``pmax`` ((B,), present when ``mask``
    is given), ``reflected`` ((B, n_steps), wave leaving the top).
    """
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    B, N = Z.shape
    inc = np.asarray(inc, dtype=float)
    inc = np.broadcast_to(inc, (B, inc.shape[-1]))
    R = (Z[:, 1:] - Z[:, :-1]) / (Z[:, 1:] + Z[:, :-1])
    Rp, Rm = 1.0 + R, 1.0 - R
    d = np.zeros((B, N))
    u = np.zeros((B, N))
    rows = np.arange(B)
    out = {"reflected": np.zeros((B, n_steps))}
    if probes is not None:
        probes = np.asarray(probes, dtype=int)
        for k in ("down", "up", "p"):
            out[k] = np.zeros((B, n_steps))
    if mask is not None:
        pmin = np.full(B, np.inf)
        pmax = np.full(B, -np.inf)
        big = np.where(mask, 0.0, np.inf)
    field = []
    for n in range(n_steps):
        out["reflected"][:, n] = u[:, 0]
        dn = np.empty_like(d)
        un = np.empty_like(u)
        dn[:, 0] = inc[:, n]
        dn[:, 1:] = Rp * d[:, :-1] - R * u[:, 1:]
        un[:, :-1] = R * d[:, :-1] + Rm * u[:, 1:]
        un[:, -1] = 0.0
        d, u = dn, un
        if probes is not None:
            out["down"][:, n] = d[rows, probes]
            out["up"][:, n] = u[rows, probes]
            out["p"][:, n] = d[rows, probes] + u[rows, probes]
        if mask is not None:
            p = d + u
            pmin = np.minimum(pmin, np.min(p + big, axis=1))
            pmax = np.maximum(pmax, np.max(p - big, axis=1))
        if field_every and n % field_every == 0:
            field.append(d[0] + u[0])
    if field_every:
        out["field"] = np.array(field)
    if mask is not None:
        out["pmin"], out["pmax"] = pmin, pmax
    return out


# ----------------------------------------------------- closed-form ray sum
def reverberation_series(inc, Z_w, Z_c, Z_s, n_delay, n_round, n_terms=None):
    """Return the pressure transmitted into a skeleton half-space.

    ``inc`` is the incident waveform sampled on the step grid, ``n_delay``
    the one-way arrival delay in steps and ``n_round`` the canopy round-trip
    time in steps.  Terms are summed until q^n falls below machine epsilon.
    """
    inc = np.asarray(inc, dtype=float)
    T1 = 2.0 * Z_c / (Z_w + Z_c)
    T2 = 2.0 * Z_s / (Z_c + Z_s)
    q = reflection(Z_c, Z_s) * reflection(Z_c, Z_w)
    n = inc.size
    out = np.zeros(n)
    if n_terms is None:
        n_terms = n if q <= 0 else int(np.log(1e-18) / np.log(q)) + 2
    for m in range(n_terms):
        lag = n_delay + m * n_round
        if lag >= n:
            break
        out[lag:] += q ** m * inc[:n - lag]
    return T1 * T2 * out


def reverberation_peak(Z_w, Z_c, Z_s, tau_over_theta):
    """Peak transmitted pressure over P for a step-exponential pulse.

    The value just after the N-th arrival is T1 T2 a_N with
    a_N = (r^(N+1) - q^(N+1)) / (r - q) and r = exp(-2 tau/theta).  The
    sequence is unimodal in N; the maximum is returned.
    """
    T1 = 2.0 * Z_c / (Z_w + Z_c)
    T2 = 2.0 * Z_s / (Z_c + Z_s)
    q = reflection(Z_c, Z_s) * reflection(Z_c, Z_w)
    r = np.exp(-2.0 * np.asarray(tau_over_theta, dtype=float))
    N = np.arange(0, 20000)[:, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(np.isclose(r, q), (N + 1) * r ** N,
                     (r ** (N + 1) - q ** (N + 1)) / (r - q))
    return T1 * T2 * a.max(axis=0)


def impulse_invariant(Z_w, Z_s):
    """Canopy-independent ratio of transmitted to incident impulse."""
    return 2.0 * Z_s / (Z_w + Z_s)


# ------------------------------------------------------- transfer matrices
def transfer_matrix(inc, dt, layers, Z_top, Z_bot, pad=8):
    """Return the pressure transmitted below a stack of layers, by FFT.

    ``layers`` is a list of (Z, c, thickness).  The incident wave is
    referenced to the top of the first layer; the returned signal is the
    pressure at the top of the bottom half-space.
    """
    inc = np.asarray(inc, dtype=float)
    n = inc.size
    nfft = int(2 ** np.ceil(np.log2(pad * n)))
    w = 2.0 * np.pi * np.fft.rfftfreq(nfft, dt)
    spec = np.fft.rfft(inc, nfft)
    Tw = np.empty(w.size, dtype=complex)
    for i, om in enumerate(w):
        M = np.eye(2, dtype=complex)
        for Z, c, h in layers:
            kh = om * h / c
            L = np.array([[np.cos(kh), -1j * Z * np.sin(kh)],
                          [-1j * np.sin(kh) / Z, np.cos(kh)]])
            M = L @ M
        # unknowns x = (Rr, T):  [T, T/Z_bot] = M [1 + Rr, (1 - Rr)/Z_top]
        A = np.array([[M[0, 0] - M[0, 1] / Z_top, -1.0],
                      [M[1, 0] - M[1, 1] / Z_top, -1.0 / Z_bot]])
        b = -np.array([M[0, 0] + M[0, 1] / Z_top,
                       M[1, 0] + M[1, 1] / Z_top])
        Tw[i] = np.linalg.solve(A, b)[1]
    return np.fft.irfft(spec * Tw, nfft)[:n]


# ------------------------------------------------------ plate first return
def plate_tension(P_s, R_b, x_over_ctheta):
    """First-reflection stress at distance x behind the plate back face.

    Returns P_s (R_b + exp(-2 x / (c_s theta))); negative values are
    tension.  The tensile peak across a plate of thickness d is
    P_s (|R_b| - exp(-2 delta)) with delta = d / (c_s theta).
    """
    return P_s * (R_b + np.exp(-2.0 * np.asarray(x_over_ctheta)))


# ------------------------------------------------ canopy-plate reef column
def build_column(prm, alpha, dp, h, d, dt, n_water=4, n_tail=None):
    """Batched column water | canopy(h) | plate(d) | canopy half-space.

    ``alpha`` and ``dp`` are arrays of equal length B.  The canopy
    impedance is the secant value at ``dp``.  Layer thicknesses are
    rounded to whole cells; the rounded thicknesses are returned so that
    their relative error can be reported.

    Returns ``Z`` (B, N), the plate mask, the index of the first plate
    cell, and the realised canopy and plate thicknesses.
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    dp = np.broadcast_to(np.asarray(dp, dtype=float), alpha.shape)
    B = alpha.size
    U = shock_speed(prm, alpha, dp)
    Zc = canopy_impedance(prm, alpha, dp)
    n_c = np.maximum(np.rint(h / (U * dt)).astype(int), 0)
    n_p = max(int(np.rint(d / (prm.c_s * dt))), 1)
    n_tail = n_tail or 8
    N = n_water + n_c.max() + n_p + n_tail
    Z = np.empty((B, N))
    mask = np.zeros((B, N), dtype=bool)
    first = n_water + n_c
    for b in range(B):
        Z[b, :n_water] = prm.Z_w
        Z[b, n_water:first[b]] = Zc[b]
        Z[b, first[b]:first[b] + n_p] = prm.Z_s
        Z[b, first[b] + n_p:] = Zc[b]
        mask[b, first[b]:first[b] + n_p] = True
    return {"Z": Z, "mask": mask, "first": first, "n_c": n_c, "n_p": n_p,
            "h_real": n_c * U * dt, "d_real": n_p * prm.c_s * dt,
            "Z_c": Zc, "U": U}


def column_extremes(prm, alpha, dp, inc, h, d, dt, n_steps, chunk=400):
    """Largest compression and tension in the plate for each column.

    ``inc`` is (n_steps,) or (B, n_steps).  Returns (compression, tension)
    in Pa, both non-negative.
    """
    alpha = np.atleast_1d(np.asarray(alpha, dtype=float))
    dp = np.broadcast_to(np.asarray(dp, dtype=float), alpha.shape)
    inc = np.asarray(inc, dtype=float)
    comp = np.empty(alpha.size)
    tens = np.empty(alpha.size)
    for s in range(0, alpha.size, chunk):
        sl = slice(s, s + chunk)
        col = build_column(prm, alpha[sl], dp[sl], h, d, dt)
        ii = inc if inc.ndim == 1 else inc[sl]
        res = goupillaud(col["Z"], ii, n_steps, mask=col["mask"])
        comp[sl] = np.maximum(res["pmax"], 0.0)
        tens[sl] = np.maximum(-res["pmin"], 0.0)
    return comp, tens
