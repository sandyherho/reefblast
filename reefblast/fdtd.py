"""Two-dimensional linear acoustics over a canopy-covered reef.

The solver advances pressure p and particle velocity (v_x, v_z) on a
staggered Yee-type grid,

    rho dv/dt = -grad p,        dp/dt = -K div v + K q(t) g(x, z),

with depth z positive downward, a pressure-release surface at z = 0, and
Cerjan-type absorbing sponges on the two sides and the bottom.  Buoyancy
1/rho is averaged arithmetically onto the velocity faces.  The scheme is
second order in space and time for smooth coefficients and first order
across material interfaces.

Materials are frozen: the canopy carries the secant impedance of
:mod:`reefblast.canopy` evaluated at one representative overpressure, and
the skeleton is treated as an acoustic medium with no shear rigidity.  The
geometry is a line source in plane strain, so the radiated waveform is not
the spherical similitude pulse.  The solver is used for qualitative wave
patterns, not for damage thresholds, which come from :mod:`reefblast.stack`.
"""

import numpy as np

from .canopy import mixture_density, shock_speed

__all__ = ["reef_scene", "plane_column", "run", "sponge",
           "line_source_rate"]

WATER, CANOPY, SKELETON = 0, 1, 2


def _smooth_noise(rng, n, scale, amp):
    """One-dimensional band-limited random profile of rms ``amp``."""
    k = np.fft.rfftfreq(n)
    spec = rng.normal(size=k.size) + 1j * rng.normal(size=k.size)
    spec *= np.exp(-(k * scale) ** 2)
    spec[0] = 0.0
    y = np.fft.irfft(spec, n)
    return amp * y / (np.std(y) + 1e-30)


def line_source_rate(t, pulse):
    """Volume rate of a line source that radiates ``pulse`` in the far field.

    In two dimensions a monopole of volume rate q(t) radiates, far from the
    line (t - r/c << r/c), a pressure proportional to the half-order
    Riemann-Liouville derivative of q.  Choosing q = I^(1/2)[pulse]
    therefore returns ``pulse`` itself up to a constant factor.  The
    half-integral is evaluated by midpoint product integration on the
    uniform grid ``t``, exact for piecewise-constant integrands.
    """
    t = np.asarray(t, dtype=float)
    g = np.asarray(pulse, dtype=float)
    Gm = 0.5 * (g[1:] + g[:-1])
    q = np.zeros_like(t)
    for n in range(1, t.size):
        w = np.sqrt(t[n] - t[:n]) - np.sqrt(t[n] - t[1:n + 1])
        q[n] = 2.0 / np.sqrt(np.pi) * np.dot(Gm[:n], w)
    return q


def reef_scene(Lx=3.2, Lz=2.3, dx=4e-3, seed=7):
    """Label grid for a branching thicket and a tabular plate.

    Returns an integer array (nz, nx) with WATER, CANOPY and SKELETON.
    """
    rng = np.random.default_rng(seed)
    nx, nz = int(round(Lx / dx)), int(round(Lz / dx))
    x = (np.arange(nx) + 0.5) * dx
    z = (np.arange(nz) + 0.5) * dx
    X, Zg = np.meshgrid(x, z)
    zb = Lz - 0.25 + _smooth_noise(rng, nx, 12.0, 0.02)
    ze = zb - 0.36 + _smooth_noise(rng, nx, 8.0, 0.03)
    lab = np.full((nz, nx), WATER, dtype=np.int8)
    lab[(Zg >= ze[None, :]) & (Zg < zb[None, :])] = CANOPY
    lab[Zg >= zb[None, :]] = SKELETON
    # branching thicket: fingers of width 3 cm at 9 cm spacing
    xc = 0.25
    while xc < 1.85:
        w = 0.03 + 0.006 * rng.normal()
        top = np.interp(xc, x, ze) + abs(0.05 * rng.normal())
        lean = 0.12 * rng.normal()
        zz = np.clip((Zg - top) / 0.4, 0, 1)
        inside = (np.abs(X - xc - lean * (1 - zz) * 0.1) < w / 2)
        lab[inside & (Zg >= top) & (lab == CANOPY)] = SKELETON
        xc += 0.09 + 0.015 * rng.normal()
    # tabular plate 10 cm thick on a 5 cm stalk
    x0, x1 = 2.05, 2.85
    ztop = np.interp(0.5 * (x0 + x1), x, ze) + 0.03
    plate = (X > x0) & (X < x1) & (Zg >= ztop) & (Zg < ztop + 0.10)
    stalk = (np.abs(X - 0.5 * (x0 + x1)) < 0.025) & (Zg >= ztop)
    lab[plate | stalk] = SKELETON
    return lab


def plane_column(Lz, dx, z_canopy, h, z_skel, nx=2):
    """Quasi-one-dimensional column for convergence and invariant tests."""
    nz = int(round(Lz / dx))
    z = (np.arange(nz) + 0.5) * dx
    lab = np.full((nz, nx), WATER, dtype=np.int8)
    if h > 0:
        lab[(z >= z_canopy) & (z < z_canopy + h)] = CANOPY
    if z_skel is not None:
        lab[z >= z_skel] = SKELETON
    return lab


def materials(prm, lab, alpha, dp):
    """Density and bulk modulus fields from a label grid."""
    rho = np.full(lab.shape, prm.rho_l)
    K = np.full(lab.shape, prm.K_l)
    rc = float(mixture_density(prm, alpha))
    Uc = float(shock_speed(prm, alpha, dp))
    rho[lab == CANOPY] = rc
    K[lab == CANOPY] = rc * Uc ** 2
    rho[lab == SKELETON] = prm.rho_s
    K[lab == SKELETON] = prm.rho_s * prm.c_s ** 2
    return rho, K


def sponge(nz, nx, width, strength=0.015, top=False, sides=True):
    """Multiplicative Cerjan damping profile."""
    def ramp(n, w):
        r = np.ones(n)
        i = np.arange(w)
        r[:w] = np.exp(-(strength * (w - i)) ** 2)
        return r
    fx = np.ones(nx)
    if sides:
        fx = ramp(nx, width) * ramp(nx, width)[::-1]
    fz = ramp(nz, width)[::-1]
    if top:
        fz = fz * ramp(nz, width)
    return fz[:, None] * fx[None, :]


def run(prm, lab, alpha, dp, dx, t_end, src_xz, theta, rise,
        cfl=0.5, free_surface=True, sponge_w=40, periodic_x=False,
        n_frames=0, decim=2, probes=(), plane_source_row=None,
        sponge_strength=0.015, waveform=None):
    """Integrate the acoustic system and collect diagnostics.

    ``src_xz`` is the source position in metres (ignored when
    ``plane_source_row`` is given, in which case a whole row is driven).
    The source rate is the normalized two-exponential with constants
    ``theta`` and ``rise`` unless a callable ``waveform`` is supplied.
    Returns a dict with frames (decimated float32), time stamps, the
    running minimum and maximum of p, probe histories, and dt.
    """
    rho, K = materials(prm, lab, alpha, dp)
    nz, nx = lab.shape
    cmax = np.sqrt(K / rho).max()
    dt = cfl * dx / cmax
    nt = int(np.ceil(t_end / dt))
    b = 1.0 / rho
    bx = np.empty((nz, nx + 1))
    bx[:, 1:-1] = 0.5 * (b[:, 1:] + b[:, :-1])
    bx[:, 0], bx[:, -1] = b[:, 0], b[:, -1]
    bz = np.empty((nz + 1, nx))
    bz[1:-1] = 0.5 * (b[1:] + b[:-1])
    bz[0], bz[-1] = b[0], b[-1]
    p = np.zeros((nz, nx))
    vx = np.zeros((nz, nx + 1))
    vz = np.zeros((nz + 1, nx))
    damp = sponge(nz, nx, sponge_w, strength=sponge_strength,
                  top=not free_surface, sides=not periodic_x)
    if plane_source_row is None:
        xs = (np.arange(nx) + 0.5) * dx
        zs = (np.arange(nz) + 0.5) * dx
        g = np.exp(-((xs[None, :] - src_xz[0]) ** 2
                     + (zs[:, None] - src_xz[1]) ** 2) / (2 * (1.5 * dx) ** 2))
        g /= g.sum() * dx * dx
    else:
        g = np.zeros((nz, nx))
        g[plane_source_row] = 1.0 / dx
    Kg = K * g
    tm = theta * rise / (theta - rise) * np.log(theta / rise)
    norm = np.exp(-tm / theta) - np.exp(-tm / rise)
    every = max(nt // n_frames, 1) if n_frames else 0
    frames, times = [], []
    pmin = np.zeros((nz, nx))
    pmax = np.zeros((nz, nx))
    hist = {k: np.zeros(nt) for k in range(len(probes))}
    for n in range(nt):
        t = n * dt
        vx[:, 1:-1] -= dt * bx[:, 1:-1] * (p[:, 1:] - p[:, :-1]) / dx
        vz[1:-1] -= dt * bz[1:-1] * (p[1:] - p[:-1]) / dx
        if free_surface:
            vz[0] -= dt * bz[0] * (2.0 * p[0]) / dx
        tt = t + 0.5 * dt
        if waveform is None:
            q = (np.exp(-tt / theta) - np.exp(-tt / rise)) / norm
        else:
            q = waveform(tt)
        p -= dt * K * ((vx[:, 1:] - vx[:, :-1]) / dx
                       + (vz[1:] - vz[:-1]) / dx)
        p += dt * q * Kg
        p *= damp
        vx[:, :-1] *= damp
        vz[:-1] *= damp
        np.minimum(pmin, p, out=pmin)
        np.maximum(pmax, p, out=pmax)
        for k, (iz, ix) in enumerate(probes):
            hist[k][n] = p[iz, ix]
        if every and n % every == 0:
            frames.append(p[::decim, ::decim].astype(np.float32))
            times.append(t)
    return {"frames": np.array(frames), "times": np.array(times),
            "pmin": pmin, "pmax": pmax, "dt": dt, "nt": nt,
            "probes": np.array([hist[k] for k in range(len(probes))]),
            "rho": rho, "K": K}
