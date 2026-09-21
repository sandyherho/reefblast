"""Acoustic and shock properties of a gas-laden canopy layer.

The canopy is seawater carrying a volume fraction ``alpha`` of free gas at
ambient pressure p0.  Gas mass is neglected against liquid mass.  Per unit
mass of mixture the specific volumes of liquid and gas are

    v_l0 = 1 / rho_l,        v_g0 = alpha / ((1 - alpha) rho_l).

A jump of overpressure dp compresses the liquid linearly with bulk modulus
K_l and the gas along the polytrope p v^kappa = const, so

    v0 - v1 = v_l0 dp / K_l + v_g0 [1 - (p0 / (p0 + dp))^(1/kappa)].

The Rayleigh line gives the shock speed U^2 = v0^2 dp / (v0 - v1) and the
secant (shock) impedance Z_c = U / v0.  As dp -> 0 this reduces exactly to
Wood's equation; for dp >> p0 the gas term saturates at v_g0 and the layer
becomes transparent once dp exceeds p* = alpha K_l / (1 - alpha).
"""

import numpy as np

__all__ = ["wood_speed", "shock_speed", "canopy_impedance",
           "reflection", "mixture_density"]


def mixture_density(prm, alpha):
    """Density of the bubbly mixture, gas mass neglected."""
    return (1.0 - np.asarray(alpha, dtype=float)) * prm.rho_l


def wood_speed(prm, alpha, p0=None):
    """Low-frequency sound speed of the bubbly mixture (Wood)."""
    alpha = np.asarray(alpha, dtype=float)
    p0 = prm.p0 if p0 is None else p0
    comp = (1.0 - alpha) / prm.K_l + alpha / (prm.kappa * p0)
    return 1.0 / np.sqrt(mixture_density(prm, alpha) * comp)


def shock_speed(prm, alpha, dp, p0=None):
    """Rayleigh-line shock speed of the bubbly mixture for overpressure dp.

    Returns the Wood speed where ``dp`` is zero.
    """
    alpha = np.asarray(alpha, dtype=float)
    dp = np.asarray(dp, dtype=float)
    p0 = prm.p0 if p0 is None else p0
    v_l0 = 1.0 / prm.rho_l
    v_g0 = alpha / ((1.0 - alpha) * prm.rho_l)
    v0 = v_l0 + v_g0
    safe = np.where(dp > 0.0, dp, 1.0)
    # 1 - (p0/(p0+dp))^(1/kappa), written with expm1/log1p for small dp
    gas = -np.expm1(-np.log1p(safe / p0) / prm.kappa)
    dv = v_l0 * safe / prm.K_l + v_g0 * gas
    U = v0 * np.sqrt(safe / dv)
    return np.where(dp > 0.0, U, wood_speed(prm, alpha, p0))


def canopy_impedance(prm, alpha, dp=0.0, p0=None):
    """Secant impedance Z_c = rho_c U of the canopy at overpressure dp."""
    return mixture_density(prm, alpha) * shock_speed(prm, alpha, dp, p0)


def reflection(Z1, Z2):
    """Pressure reflection coefficient for a wave in medium 1 hitting 2."""
    return (Z2 - Z1) / (Z2 + Z1)
