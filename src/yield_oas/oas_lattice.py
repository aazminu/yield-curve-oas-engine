"""
Short-Rate Recombining Binomial Lattice & Option-Adjusted Spread (OAS) Engine.

Implements:
- Exact arbitrage-free calibration of a Ho-Lee / Hull-White recombining binomial
  short-rate tree to any input `ZeroCurve` via forward induction (Arrow-Debreu state prices)
- Backward-induction valuation of straight, callable, and puttable corporate bonds
- Option-Adjusted Spread (OAS) solver using Brent's method (`scipy.optimize.brentq`)
- OAS-constant Effective Duration ($D_{\text{eff}}$) and Effective Convexity ($C_{\text{eff}}$)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .curve import ZeroCurve
from .risk import brent_root


class EmbeddedOptionType(str, Enum):
    NONE = "NONE"
    CALLABLE = "CALLABLE"
    PUTTABLE = "PUTTABLE"


@dataclass(frozen=True)
class OptionScheduleBond:
    """
    Specification of a fixed-coupon bond with an embedded Bermudan/American call or put option.
    """

    maturity: float
    coupon_rate: float
    option_type: EmbeddedOptionType = EmbeddedOptionType.NONE
    lockout_years: float = 1.0  # Option exercisable for t >= lockout_years
    strike_price: float = 100.0  # Clean call/put strike price
    face_value: float = 100.0
    frequency: int = 2


@dataclass
class ShortRateLattice:
    """
    Calibrated recombining binomial short-rate tree over `n_steps` intervals of length `dt`.
    At step `i` (time `t_i = i * dt`) and state `j in {0..i}`, the continuously compounded
    one-step short rate is:
        r_{i, j} = theta_i + (2 * j - i) * sigma * sqrt(dt)
    """

    n_steps: int
    dt: float
    sigma: float
    thetas: np.ndarray  # Shape: (n_steps,)
    state_prices: List[np.ndarray]  # Arrow-Debreu prices Q_{i, j}

    def short_rates_at_step(self, step: int) -> np.ndarray:
        j = np.arange(step + 1, dtype=np.float64)
        return self.thetas[step] + (2.0 * j - step) * self.sigma * math.sqrt(self.dt)


def calibrate_short_rate_lattice(
    curve: ZeroCurve,
    horizon_years: float,
    steps_per_year: int = 20,
    sigma: float = 0.01,
) -> ShortRateLattice:
    """
    Calibrate the drift vector `theta_i` of a recombining short-rate binomial tree
    via Arrow-Debreu forward induction so that the tree reproduces the exact
    risk-free discount factors `P(0, (i+1)*dt)` of `curve` at every time step.

    Closed-form forward-induction update at step `i`:
        P(0, t_{i+1}) = sum_{j=0}^{i} Q_{i, j} * exp(-(theta_i + (2j - i)*sigma*sqrt(dt)) * dt)
      => exp(-theta_i * dt) = P(0, t_{i+1}) / sum_{j=0}^{i} Q_{i, j} * exp(-(2j - i)*sigma*dt^{3/2})
    """
    n_steps = max(2, int(round(horizon_years * steps_per_year)))
    dt = horizon_years / n_steps
    sqrt_dt = math.sqrt(dt)

    thetas = np.zeros(n_steps, dtype=np.float64)
    state_prices: List[np.ndarray] = [np.array([1.0], dtype=np.float64)]

    for i in range(n_steps):
        q_i = state_prices[i]
        j = np.arange(i + 1, dtype=np.float64)
        perturbation = (2.0 * j - i) * sigma * sqrt_dt
        target_df = curve.discount_factor((i + 1) * dt)

        unshifted_sum = float(np.sum(q_i * np.exp(-perturbation * dt)))
        theta_i = -math.log(target_df / unshifted_sum) / dt
        thetas[i] = theta_i

        # Propagate Arrow-Debreu state prices to step i + 1
        r_ij = theta_i + perturbation
        disc_q = 0.5 * q_i * np.exp(-r_ij * dt)
        q_next = np.zeros(i + 2, dtype=np.float64)
        q_next[:-1] += disc_q
        q_next[1:] += disc_q
        state_prices.append(q_next)

    return ShortRateLattice(
        n_steps=n_steps,
        dt=dt,
        sigma=sigma,
        thetas=thetas,
        state_prices=state_prices,
    )


def price_bond_on_lattice(
    bond: OptionScheduleBond,
    lattice: ShortRateLattice,
    oas_bps: float = 0.0,
) -> float:
    """
    Value a straight, callable, or puttable bond on `lattice` via backward induction
    with constant Option-Adjusted Spread `s = oas_bps / 10_000`.
    """
    n = lattice.n_steps
    dt = lattice.dt
    oas = oas_bps / 10_000.0
    steps_per_coupon = max(1, int(round(1.0 / (bond.frequency * dt))))
    coupon_amt = bond.face_value * (bond.coupon_rate / bond.frequency)

    # Terminal payoff at maturity (step n): Face Value + final coupon
    values = np.full(n + 1, bond.face_value + coupon_amt, dtype=np.float64)

    for i in range(n - 1, -1, -1):
        t_i = i * dt
        r_ij = lattice.short_rates_at_step(i) + oas
        df_step = np.exp(-r_ij * dt)

        # Risk-neutral expected continuation value (clean before step-i coupon)
        cont = df_step * 0.5 * (values[:-1] + values[1:])

        # Apply embedded option exercise constraint on clean continuation value
        if i > 0 and t_i >= bond.lockout_years - 1e-9:
            if bond.option_type == EmbeddedOptionType.CALLABLE:
                cont = np.minimum(cont, bond.strike_price)
            elif bond.option_type == EmbeddedOptionType.PUTTABLE:
                cont = np.maximum(cont, bond.strike_price)

        # Add coupon if step i is a scheduled coupon payment date (excluding t=0)
        if i > 0 and (n - i) % steps_per_coupon == 0:
            cont = cont + coupon_amt

        values = cont

    return float(values[0])


def solve_oas(
    bond: OptionScheduleBond,
    curve: ZeroCurve,
    market_price: float,
    steps_per_year: int = 20,
    sigma: float = 0.01,
) -> float:
    """
    Solve for the Option-Adjusted Spread (OAS, in basis points) that equates the
    lattice model price of `bond` to `market_price` using Brent's root-finding method.
    """
    lattice = calibrate_short_rate_lattice(
        curve=curve,
        horizon_years=bond.maturity,
        steps_per_year=steps_per_year,
        sigma=sigma,
    )

    def _objective(oas_bps: float) -> float:
        return price_bond_on_lattice(bond, lattice, oas_bps=oas_bps) - market_price

    return float(brent_root(_objective, -1000.0, 5000.0, xtol=1e-8))


def effective_duration_convexity(
    bond: OptionScheduleBond,
    curve: ZeroCurve,
    oas_bps: float = 0.0,
    bump_bps: float = 10.0,
    steps_per_year: int = 20,
    sigma: float = 0.01,
) -> Tuple[float, float, float]:
    """
    Compute OAS-constant model price $P_0$, Effective Duration ($D_{\\text{eff}}$),
    and Effective Convexity ($C_{\\text{eff}}$) by shifting the underlying `ZeroCurve`
    by $\\pm \\Delta y$, recalibrating the short-rate tree, and repricing with fixed OAS:
        D_eff = (P_- - P_+) / (2 * P_0 * Delta_y)
        C_eff = (P_- + P_+ - 2 * P_0) / (P_0 * Delta_y^2)
    """
    dy = bump_bps / 10_000.0
    lat_0 = calibrate_short_rate_lattice(curve, bond.maturity, steps_per_year, sigma)
    lat_up = calibrate_short_rate_lattice(
        curve.parallel_shift(+bump_bps), bond.maturity, steps_per_year, sigma
    )
    lat_dn = calibrate_short_rate_lattice(
        curve.parallel_shift(-bump_bps), bond.maturity, steps_per_year, sigma
    )

    p_0 = price_bond_on_lattice(bond, lat_0, oas_bps=oas_bps)
    p_up = price_bond_on_lattice(bond, lat_up, oas_bps=oas_bps)
    p_dn = price_bond_on_lattice(bond, lat_dn, oas_bps=oas_bps)

    d_eff = (p_dn - p_up) / (2.0 * p_0 * dy)
    c_eff = (p_dn + p_up - 2.0 * p_0) / (p_0 * (dy ** 2))
    return float(p_0), float(d_eff), float(c_eff)
