"""Prescribed diel cycle of canopy void fraction.

The void fraction is a forcing, not a model output:

    alpha(t) = alpha_night + (alpha_max - alpha_night) s(t)^2,
    s(t) = max(0, sin(pi (t - t_rise) / (t_set - t_rise))),

with t in local solar hours.  The squared sine keeps alpha at its night
value until well after sunrise, as expected when free gas appears only once
dissolved oxygen passes saturation.  Peak values are scenario inputs; no
reef-canopy measurement is assumed.
"""

import numpy as np

__all__ = ["void_fraction"]


def void_fraction(t_hours, alpha_max, alpha_night=1e-5, t_rise=6.0,
                  t_set=18.0):
    """Void fraction at local solar time ``t_hours``."""
    t = np.asarray(t_hours, dtype=float) % 24.0
    s = np.sin(np.pi * (t - t_rise) / (t_set - t_rise))
    s = np.where((t > t_rise) & (t < t_set), np.clip(s, 0.0, None), 0.0)
    return alpha_night + (alpha_max - alpha_night) * s ** 2
