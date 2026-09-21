"""Reference configuration shared by every figure and animation.

Values are chosen to sit inside the range of improvised charges and reef
depths described for blast fishing on shallow Indo-Pacific reefs.  They are
illustrative inputs, not estimates for a particular site.

Every range in the damage calculations is a vertical standoff: the charge is
directly above the plate and the shock arrives at normal incidence, which is
the only geometry the one-dimensional column represents without error.  At
oblique incidence beyond arcsin(c_l/c_s) = 30 degrees no propagating
compressional wave enters the skeleton, so horizontal damage radii are not
computed.
"""

import numpy as np

from .core import Params

__all__ = ["PRM", "W_REF", "H_CANOPY", "D_REF", "D_FAMILY", "ALPHAS",
           "ALPHA_GRID", "R_GRID", "Z_REEF", "ALPHA_MAX",
           "SUB"]

PRM = Params()

W_REF = 1.0                    # TNT-equivalent charge, kg
H_CANOPY = 0.15                # canopy thickness above a plate, m
D_REF = 0.12                   # reference plate thickness, m
D_FAMILY = [0.06, 0.09, 0.12, 0.15]
ALPHAS = [1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2]
ALPHA_GRID = np.logspace(-5, -1.5, 56)
R_GRID = np.logspace(np.log10(0.7), np.log10(20.0), 64)
Z_REEF = PRM.depth             # depth of the plate, m
ALPHA_MAX = [1e-3, 1e-2, 3e-2]  # diel peak scenarios
SUB = 100                      # time steps per smallest decay constant
