"""
Fixed-Income Sensitivity & Key-Rate Risk Engine.

Implements:
- Present value pricing of fixed-coupon bonds over a ZeroCurve + constant credit spread
- Yield-to-Maturity (YTM) solver via Brent's root-finding algorithm
- Analytical Macaulay Duration, Modified Duration, and Convexity
- Parallel Curve DV01 and Credit Spread CS01 (central finite-difference bump-and-reprice)
- Ho (1992) Key-Rate DV01 vector across benchmark tenors (e.g., [2Y, 5Y, 10Y, 30Y])
  satisfying exact parallel DV01 conservation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Sequence, Tuple

import numpy as np

from .curve import ZeroCurve


def brent_root(
    func: Callable[[float], float],
    a: float,
    b: float,
    xtol: float = 1e-12,
    max_iter: int = 100,
) -> float:
    """
    Brent's (1973) root-finding algorithm combining root bracketing, bisection,
    secant, and inverse quadratic interpolation.
    """
    fa = func(a)
    fb = func(b)
    if fa * fb > 0:
        raise ValueError(f"Root not bracketed in [{a}, {b}]: f(a)={fa}, f(b)={fb}")
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c, fc = a, fa
    d = b - a
    mflag = True

    for _ in range(max_iter):
        if abs(fb) <= 1e-15 or abs(b - a) < xtol:
            return b

        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            s = (
                a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
            )
        else:
            # Secant step
            s = b - fb * (b - a) / (fb - fa)

        cond1 = not ((3.0 * a + b) / 4.0 <= s <= b or b <= s <= (3.0 * a + b) / 4.0)
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2.0
        cond3 = (not mflag) and abs(s - b) >= abs(c - d) / 2.0
        cond4 = mflag and abs(b - c) < xtol
        cond5 = (not mflag) and abs(c - d) < xtol

        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = 0.5 * (a + b)
            mflag = True
        else:
            mflag = False

        fs = func(s)
        d = c
        c, fc = b, fb

        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    return b


@dataclass(frozen=True)
class BondSpec:
    """Fixed-rate bullet bond specification."""

    maturity: float  # Maturity in years
    coupon_rate: float  # Annual coupon rate in decimal (e.g. 0.05 for 5%)
    face_value: float = 100.0
    frequency: int = 2

    def cashflows(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return arrays of `(payment_times, cashflow_amounts)`."""
        n_periods = max(1, int(round(self.maturity * self.frequency)))
        dt = 1.0 / self.frequency
        times = np.array([j * dt for j in range(1, n_periods + 1)], dtype=np.float64)
        times[-1] = self.maturity
        coupon_cf = self.face_value * (self.coupon_rate / self.frequency)
        cfs = np.full(n_periods, coupon_cf, dtype=np.float64)
        cfs[-1] += self.face_value
        return times, cfs


@dataclass(frozen=True)
class RiskMetrics:
    """Comprehensive fixed-income valuation and sensitivity report."""

    price: float
    ytm: float
    macaulay_duration: float
    modified_duration: float
    convexity: float
    dv01: float
    cs01: float
    key_rate_dv01: Dict[float, float]


def price_bond_from_curve(
    bond: BondSpec,
    curve: ZeroCurve,
    credit_spread_bps: float = 0.0,
) -> float:
    """
    Price a fixed-coupon bond by discounting each cashflow at
    `P(0, t_i) * exp(-s * t_i)`, where `s = credit_spread_bps / 10_000`.
    """
    times, cfs = bond.cashflows()
    spread = credit_spread_bps / 10_000.0
    pv = 0.0
    for t, cf in zip(times, cfs):
        df = curve.discount_factor(float(t)) * np.exp(-spread * float(t))
        pv += float(cf) * float(df)
    return float(pv)


def bond_yield_to_maturity(bond: BondSpec, dirty_price: float) -> float:
    """
    Solve for standard discretely-compounded Yield-to-Maturity (YTM) at the bond's
    payment frequency `m` via Brent's method (`brent_root`).
    """
    times, cfs = bond.cashflows()
    freq = bond.frequency

    def _obj(y: float) -> float:
        periods = times * freq
        disc = (1.0 + y / freq) ** (-periods)
        return float(np.sum(cfs * disc) - dirty_price)

    return float(brent_root(_obj, -0.05, 1.00, xtol=1e-12))


def analytical_duration_convexity(
    bond: BondSpec, ytm: float
) -> Tuple[float, float, float]:
    """
    Compute analytical Macaulay Duration, Modified Duration, and Convexity
    given discretely compounded `ytm` at payment frequency `m`.
    """
    times, cfs = bond.cashflows()
    m = float(bond.frequency)
    periods = times * m
    df = (1.0 + ytm / m) ** (-periods)
    pv_cfs = cfs * df
    price = float(np.sum(pv_cfs))

    mac_dur = float(np.sum(times * pv_cfs) / price)
    mod_dur = mac_dur / (1.0 + ytm / m)
    conv_terms = times * (times + 1.0 / m) * pv_cfs / ((1.0 + ytm / m) ** 2)
    convexity = float(np.sum(conv_terms) / price)
    return mac_dur, mod_dur, convexity


def compute_dv01_cs01(
    bond: BondSpec,
    curve: ZeroCurve,
    credit_spread_bps: float = 0.0,
    bump_bps: float = 1.0,
) -> Tuple[float, float]:
    """
    Compute central finite-difference Parallel Curve DV01 and Credit Spread CS01
    (dollar price sensitivity per 1bp reduction in rates/spread, per 100 face value).
    """
    curve_up = curve.parallel_shift(+bump_bps)
    curve_dn = curve.parallel_shift(-bump_bps)
    p_up = price_bond_from_curve(bond, curve_up, credit_spread_bps)
    p_dn = price_bond_from_curve(bond, curve_dn, credit_spread_bps)
    dv01 = (p_dn - p_up) / (2.0 * bump_bps)

    p_cs_up = price_bond_from_curve(bond, curve, credit_spread_bps + bump_bps)
    p_cs_dn = price_bond_from_curve(bond, curve, credit_spread_bps - bump_bps)
    cs01 = (p_cs_dn - p_cs_up) / (2.0 * bump_bps)
    return float(dv01), float(cs01)


def compute_key_rate_dv01(
    bond: BondSpec,
    curve: ZeroCurve,
    key_tenors: Sequence[float] = (2.0, 5.0, 10.0, 30.0),
    credit_spread_bps: float = 0.0,
    bump_bps: float = 1.0,
) -> Dict[float, float]:
    """
    Compute the Key-Rate DV01 profile across `key_tenors` (default `[2Y, 5Y, 10Y, 30Y]`).
    """
    times, _ = bond.cashflows()
    kt = [float(t) for t in key_tenors]
    dense_grid = np.unique(np.concatenate([curve.tenors, times, np.asarray(kt, dtype=np.float64)]))

    kr_dv01: Dict[float, float] = {}
    for idx, tenor in enumerate(kt):
        c_up = curve.with_key_rate_tent_bump(kt, idx, +bump_bps, dense_grid=dense_grid)
        c_dn = curve.with_key_rate_tent_bump(kt, idx, -bump_bps, dense_grid=dense_grid)
        p_up = price_bond_from_curve(bond, c_up, credit_spread_bps)
        p_dn = price_bond_from_curve(bond, c_dn, credit_spread_bps)
        kr_dv01[tenor] = float((p_dn - p_up) / (2.0 * bump_bps))

    return kr_dv01


def full_risk_report(
    bond: BondSpec,
    curve: ZeroCurve,
    credit_spread_bps: float = 0.0,
    key_tenors: Sequence[float] = (2.0, 5.0, 10.0, 30.0),
) -> RiskMetrics:
    """Compute a complete institutional risk and valuation report for `bond`."""
    price = price_bond_from_curve(bond, curve, credit_spread_bps)
    ytm = bond_yield_to_maturity(bond, price)
    mac_d, mod_d, conv = analytical_duration_convexity(bond, ytm)
    dv01, cs01 = compute_dv01_cs01(bond, curve, credit_spread_bps)
    kr = compute_key_rate_dv01(bond, curve, key_tenors=key_tenors, credit_spread_bps=credit_spread_bps)
    return RiskMetrics(
        price=price,
        ytm=ytm,
        macaulay_duration=mac_d,
        modified_duration=mod_d,
        convexity=conv,
        dv01=dv01,
        cs01=cs01,
        key_rate_dv01=kr,
    )
