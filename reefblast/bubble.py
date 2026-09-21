"""Keller-Miksis dynamics of a single canopy bubble under a blast pulse.

Lengths are scaled by the equilibrium radius R0, pressures by the ambient
pressure p0 and times by t_c = R0 sqrt(rho / p0), so that the
dimensionless equation reads

    (1 - M r') r r'' + (3/2)(1 - M r'/3) r'^2
        = (1 + M r')(p_B - p_inf(s)) + M r dp_B/ds,

    p_B = (1 + We) r^(-3 kappa) - We / r - (4 / Re) r' / r,

with M = sqrt(p0/rho)/c, We = 2 sigma/(R0 p0) and Re = R0 sqrt(rho p0)/mu.
Because dp_B/ds contains r'' through the viscous term, the equation is
solved for r'' explicitly.  Linearization about r = 1 with M = 0 and
Re -> infinity gives the angular frequency
omega_0^2 = 3 kappa (1 + We) - We, the Minnaert value corrected for
surface tension.
"""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

__all__ = ["Bubble", "km_rhs", "solve_km", "linear_frequency",
           "equilibrium_radius", "minnaert_hz"]


@dataclass(frozen=True)
class Bubble:
    """Dimensionless groups of a gas bubble in the canopy."""

    R0: float = 0.5e-3       # equilibrium radius, m
    sigma: float = 0.072     # surface tension, N m^-1
    mu: float = 1.0e-3       # dynamic viscosity, Pa s

    def groups(self, prm):
        """Return (M, We, Re, t_c) for the ambient state in ``prm``."""
        p0, rho = prm.p0, prm.rho_l
        uc = np.sqrt(p0 / rho)
        M = uc / prm.c_l
        We = 2.0 * self.sigma / (self.R0 * p0)
        Re = self.R0 * np.sqrt(rho * p0) / self.mu if self.mu > 0 else np.inf
        return M, We, Re, self.R0 / uc


def linear_frequency(kappa, We):
    """Dimensionless small-amplitude angular frequency."""
    return np.sqrt(3.0 * kappa * (1.0 + We) - We)


def minnaert_hz(prm, bub):
    """Natural frequency in hertz, surface-tension corrected."""
    M, We, Re, tc = bub.groups(prm)
    return linear_frequency(prm.kappa, We) / (2.0 * np.pi * tc)


def km_rhs(s, y, kappa, M, We, Re, p_inf, dp_inf=None):
    """Right-hand side of the dimensionless Keller-Miksis system."""
    r, v = y
    inv_re = 0.0 if np.isinf(Re) else 1.0 / Re
    rk = r ** (-3.0 * kappa)
    pB = (1.0 + We) * rk - We / r - 4.0 * inv_re * v / r
    # dpB/ds without the r'' part
    dpB0 = (-3.0 * kappa * (1.0 + We) * rk / r * v + We * v / r ** 2
            + 4.0 * inv_re * v ** 2 / r ** 2)
    A = (1.0 - M * v) * r + 4.0 * M * inv_re
    rhs = (-1.5 * (1.0 - M * v / 3.0) * v ** 2
           + (1.0 + M * v) * (pB - p_inf(s)) + M * r * dpB0)
    return [v, rhs / A]


def solve_km(prm, bub, P_over_p0, theta_s, s_end, n_out=4000, rise=None,
             M=None, Re=None, rtol=1e-10, method="DOP853"):
    """Integrate Keller-Miksis for a step-exponential overpressure.

    ``theta_s`` is the decay constant in units of t_c.  ``rise`` (in
    units of t_c) smooths the front with a factor 1 - exp(-s/rise).
    ``M`` and ``Re`` override the physical groups, which is how the
    inviscid incompressible verification limit is reached.  Returns the
    sampled time, radius and wall speed, and the smallest radius, located
    exactly by an event on r' = 0 rather than from the samples.
    """
    M0, We, Re0, tc = bub.groups(prm)
    M = M0 if M is None else M
    Re = Re0 if Re is None else Re

    def p_inf(s):
        if s <= 0.0:
            return 1.0
        g = np.exp(-s / theta_s)
        if rise:
            g *= -np.expm1(-s / rise)
        return 1.0 + P_over_p0 * g

    def turn(s, y, *args):
        return y[1]
    turn.direction = 1.0

    s_eval = np.linspace(0.0, s_end, n_out)
    sol = solve_ivp(km_rhs, (0.0, s_end), [1.0, 0.0], method=method,
                    t_eval=s_eval, rtol=rtol, atol=rtol * 1e-2,
                    args=(prm.kappa, M, We, Re, p_inf), events=turn,
                    max_step=min(theta_s / 20, 1.0))
    minima = sol.y_events[0][:, 0] if sol.y_events[0].size else sol.y[0]
    return sol.t, sol.y[0], sol.y[1], float(np.min(minima))


def equilibrium_radius(kappa, We, p_over_p0):
    """Return the static radius at absolute pressure p (units of p0)."""
    def f(r):
        return (1.0 + We) * r ** (-3.0 * kappa) - We / r - p_over_p0
    return brentq(f, 1e-6, 10.0, xtol=1e-15)
