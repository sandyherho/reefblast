"""Material parameters, the similitude source, and the crossover pressure.

All quantities are SI.  Reference values are illustrative and are not
calibrated against measurements on any particular reef; see
``outputs/reports/parameters.txt`` for their provenance and for what the
results do and do not depend on.

The incident loading is represented by the similitude form

    p_i(t) = P_m exp(-t / theta) H(t),
    P_m = K_P (W^(1/3) / R)^A_P,
    theta = K_T W^(1/3) (W^(1/3) / R)^(-A_T),

with W the TNT-equivalent charge mass in kilograms and R the slant range in
metres.  The constants are the widely tabulated TNT values attributed to
Cole (1948).  Improvised ammonium-nitrate charges are represented only
through an equivalent W.
"""

from dataclasses import dataclass, replace

import numpy as np

__all__ = ["Params", "peak_pressure", "decay_time", "crossover_pressure",
           "crossover_range", "incident", "hydrostatic"]

G = 9.81


@dataclass(frozen=True)
class Params:
    """Physical parameters of water, canopy gas, skeleton, and source."""

    rho_l: float = 1025.0        # seawater density, kg m^-3
    c_l: float = 1500.0          # seawater sound speed, m s^-1
    kappa: float = 1.4           # polytropic exponent of canopy gas
    depth: float = 5.0           # depth of the canopy, m
    p_atm: float = 101325.0      # atmospheric pressure, Pa
    rho_s: float = 1600.0        # bulk density of porous skeleton, kg m^-3
    c_s: float = 3000.0          # longitudinal speed in skeleton, m s^-1
    sigma_t: float = 2.0e6       # tensile strength of skeleton, Pa
    sigma_c: float = 20.0e6      # compressive strength of skeleton, Pa
    K_P: float = 52.16e6         # similitude peak-pressure constant, Pa
    A_P: float = 1.13            # similitude peak-pressure exponent
    K_T: float = 92.5e-6         # similitude decay-time constant, s kg^-1/3
    A_T: float = 0.22            # similitude decay-time exponent

    @property
    def K_l(self):
        """Bulk modulus of seawater, Pa."""
        return self.rho_l * self.c_l ** 2

    @property
    def Z_w(self):
        """Acoustic impedance of seawater, Pa s m^-1."""
        return self.rho_l * self.c_l

    @property
    def Z_s(self):
        """Acoustic impedance of the skeleton, Pa s m^-1."""
        return self.rho_s * self.c_s

    @property
    def p0(self):
        """Absolute ambient pressure at the canopy, Pa."""
        return hydrostatic(self, self.depth)

    def with_(self, **kw):
        """Return a copy with the given fields replaced."""
        return replace(self, **kw)


def hydrostatic(prm, z):
    """Absolute pressure at depth ``z`` metres."""
    return prm.p_atm + prm.rho_l * G * np.asarray(z, dtype=float)


def peak_pressure(prm, W, R):
    """Similitude peak overpressure P_m(W, R), Pa."""
    s = np.cbrt(W) / np.asarray(R, dtype=float)
    return prm.K_P * s ** prm.A_P


def decay_time(prm, W, R):
    """Similitude exponential decay constant theta(W, R), s."""
    s = np.cbrt(W) / np.asarray(R, dtype=float)
    return prm.K_T * np.cbrt(W) * s ** (-prm.A_T)


def crossover_pressure(prm, alpha):
    """Overpressure p* = alpha K_l / (1 - alpha) at which gas and liquid.

    compliances of the canopy are equal under strong compression.
    """
    alpha = np.asarray(alpha, dtype=float)
    return alpha * prm.K_l / (1.0 - alpha)


def crossover_range(prm, W, alpha):
    """Slant range R* at which the incident peak equals p*, m."""
    ps = crossover_pressure(prm, alpha)
    return np.cbrt(W) * (prm.K_P / ps) ** (1.0 / prm.A_P)


def incident(t, P, theta, t_rise=0.0):
    """Incident waveform with an optional finite rise.

    ``t_rise = 0`` gives the similitude step-exponential.  A positive
    ``t_rise`` gives P (e^{-t/theta} - e^{-t/t_rise}) / n, normalized so
    that the maximum equals P; this smooth form is used where a
    discontinuous front would pollute a convergence measurement.
    """
    t = np.asarray(t, dtype=float)
    tp = np.clip(t, 0.0, None)
    if t_rise <= 0.0:
        return np.where(t >= 0.0, P * np.exp(-tp / theta), 0.0)
    tm = theta * t_rise / (theta - t_rise) * np.log(theta / t_rise)
    norm = np.exp(-tm / theta) - np.exp(-tm / t_rise)
    val = (np.exp(-tp / theta) - np.exp(-tp / t_rise)) / norm
    return np.where(t >= 0.0, P * val, 0.0)
