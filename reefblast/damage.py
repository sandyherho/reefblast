"""Peak plate stresses over parameter sweeps, and damage radii.

Every column is water | canopy(h) | plate(d) | canopy half-space, loaded
by the similitude pulse for charge W at slant range R and solved with the
batched Goupillaud scheme.  Normal incidence is assumed at every range.
"""

import numpy as np

from .core import decay_time, incident, peak_pressure
from .canopy import shock_speed
from .stack import column_extremes

__all__ = ["plate_extremes", "damage_radius"]


def plate_extremes(prm, W, R, alpha, d, h, sub=100):
    """Largest compression and tension (Pa) in the plate.

    ``R``, ``alpha`` and ``d`` broadcast together; ``d`` may vary only if
    it is a scalar, because the plate cell count is shared by the batch.
    Returns (compression, tension, info) with ``info`` holding dt and the
    realised plate thickness.
    """
    R, alpha = np.broadcast_arrays(np.asarray(R, float),
                                   np.asarray(alpha, float))
    shape = R.shape
    R, alpha = R.ravel(), alpha.ravel()
    P = peak_pressure(prm, W, R)
    th = decay_time(prm, W, R)
    dt = th.min() / sub
    U = shock_speed(prm, alpha, P)
    n_c = np.rint(h / (U * dt)).astype(int)
    n_p = max(int(np.rint(d / (prm.c_s * dt))), 1)
    n_steps = int(4 + n_c.max() + 2 * n_p + np.ceil(10 * th.max() / dt))
    t = np.arange(n_steps) * dt
    inc = P[:, None] * np.exp(-t[None, :] / th[:, None])
    comp, tens = column_extremes(prm, alpha, P, inc, h, d, dt, n_steps)
    info = {"dt": dt, "d_real": n_p * prm.c_s * dt, "n_steps": n_steps,
            "h_real_min": (n_c * U * dt).min(),
            "h_real_max": (n_c * U * dt).max()}
    return comp.reshape(shape), tens.reshape(shape), info


def damage_radius(r, stress, threshold):
    """Largest r at which ``stress`` still reaches ``threshold``.

    ``stress`` is sampled on increasing ``r``; the crossing is located by
    linear interpolation in log r.  Returns 0 when the threshold is never
    reached and r[-1] when it is exceeded everywhere.
    """
    r = np.asarray(r, float)
    s = np.asarray(stress, float) - threshold
    above = np.where(s >= 0)[0]
    if above.size == 0:
        return 0.0
    k = above.max()
    if k == r.size - 1:
        return r[-1]
    lr = np.log(r)
    f = s[k] / (s[k] - s[k + 1])
    return float(np.exp(lr[k] + f * (lr[k + 1] - lr[k])))
