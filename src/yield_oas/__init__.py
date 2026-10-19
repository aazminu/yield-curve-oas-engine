"""
yield_oas: Institutional Fixed-Income Yield-Curve Bootstrapper,
Nelson-Siegel-Svensson (NSS) Parametric Fitter, Key-Rate DV01 Risk Engine,
and Short-Rate Recombining Lattice Option-Adjusted Spread (OAS) Pricer.

Author: Abdullah Azmi (m4u5t4@u.northwestern.edu)
"""

from .curve import (
    ParInstrument,
    ZeroCurve,
    NSSParameters,
    bootstrap_par_curve,
    fit_nss_curve,
)
from .risk import (
    BondSpec,
    RiskMetrics,
    price_bond_from_curve,
    bond_yield_to_maturity,
    analytical_duration_convexity,
    compute_dv01_cs01,
    compute_key_rate_dv01,
    full_risk_report,
)
from .oas_lattice import (
    EmbeddedOptionType,
    OptionScheduleBond,
    ShortRateLattice,
    calibrate_short_rate_lattice,
    price_bond_on_lattice,
    solve_oas,
    effective_duration_convexity,
)

__version__ = "0.2.0"
__author__ = "Abdullah Azmi"

__all__ = [
    "ParInstrument",
    "ZeroCurve",
    "NSSParameters",
    "bootstrap_par_curve",
    "fit_nss_curve",
    "BondSpec",
    "RiskMetrics",
    "price_bond_from_curve",
    "bond_yield_to_maturity",
    "analytical_duration_convexity",
    "compute_dv01_cs01",
    "compute_key_rate_dv01",
    "full_risk_report",
    "EmbeddedOptionType",
    "OptionScheduleBond",
    "ShortRateLattice",
    "calibrate_short_rate_lattice",
    "price_bond_on_lattice",
    "solve_oas",
    "effective_duration_convexity",
]
