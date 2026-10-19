"""
Fixed-Income Yield Curve Bootstrapping & Nelson-Siegel-Svensson (NSS) Fitting.

Implements:
- Exact Sequential Par-Bond Discount Factor & Continuously Compounded Zero-Rate Bootstrapping
- Log-Linear Discount Factor (Piecewise-Constant Forward) & Fritsch-Carlson (1980)
  Monotone Cubic Hermite (PCHIP) Zero-Rate Interpolation
- Instantaneous & Discrete Forward Rate Extraction
- Nelson-Siegel-Svensson (NSS) 6-Parameter Yield Curve Calibration via separable
  nonlinear least squares (concentrated OLS + Gauss-Newton / golden-section refinement)
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ParInstrument:
    """Benchmark par coupon instrument (e.g., US Treasury on-the-run or par swap)."""

    maturity: float  # In years (e.g., 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
    par_yield: float  # Annualized par coupon rate in decimal (e.g., 0.045 for 4.50%)
    frequency: int = 2  # Coupon payments per year (default semi-annual = 2)


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute Fritsch-Carlson (1980) shape-preserving monotone cubic Hermite derivatives.
    """
    n = x.size
    if n == 2:
        sec = (y[1] - y[0]) / (x[1] - x[0])
        return np.array([sec, sec], dtype=np.float64)

    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros(n, dtype=np.float64)

    # Interior points: weighted harmonic mean when consecutive secants share the same sign
    for k in range(1, n - 1):
        if delta[k - 1] * delta[k] > 0.0:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
        else:
            d[k] = 0.0

    # Endpoints: one-sided shape-preserving three-point formula
    d0 = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    if d0 * delta[0] <= 0.0:
        d0 = 0.0
    elif (delta[0] * delta[1] <= 0.0) and (abs(d0) > abs(3.0 * delta[0])):
        d0 = 3.0 * delta[0]
    d[0] = d0

    dn = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if dn * delta[-1] <= 0.0:
        dn = 0.0
    elif (delta[-1] * delta[-2] <= 0.0) and (abs(dn) > abs(3.0 * delta[-1])):
        dn = 3.0 * delta[-1]
    d[-1] = dn
    return d


def _pchip_eval(x: np.ndarray, y: np.ndarray, slopes: np.ndarray, t: float) -> float:
    if t <= x[0]:
        return float(y[0])
    if t >= x[-1]:
        return float(y[-1])
    idx = int(np.searchsorted(x, t, side="right")) - 1
    idx = max(0, min(idx, x.size - 2))
    h = x[idx + 1] - x[idx]
    s = (t - x[idx]) / h
    h00 = (1.0 + 2.0 * s) * ((1.0 - s) ** 2)
    h10 = s * ((1.0 - s) ** 2)
    h01 = (s ** 2) * (3.0 - 2.0 * s)
    h11 = (s ** 2) * (s - 1.0)
    return float(
        h00 * y[idx] + h10 * h * slopes[idx] + h01 * y[idx + 1] + h11 * h * slopes[idx + 1]
    )


class ZeroCurve:
    """
    Term structure of zero-coupon rates and discount factors.

    Supports two institutional interpolation modes:
    - 'log_linear_df': Piecewise-linear in log discount factors (equivalent to
      piecewise-constant instantaneous forward rates, guaranteeing positive forwards
      and zero oscillation).
    - 'monotone_cubic': Shape-preserving Monotone Piecewise Cubic Hermite Interpolating
      Polynomial (Fritsch-Carlson PCHIP) on continuously compounded zero rates.
    """

    def __init__(
        self,
        tenors: Sequence[float],
        zero_rates: Sequence[float],
        interp_mode: str = "log_linear_df",
    ) -> None:
        t_arr = np.asarray(tenors, dtype=np.float64)
        z_arr = np.asarray(zero_rates, dtype=np.float64)
        if t_arr.size != z_arr.size or t_arr.size < 1:
            raise ValueError("tenors and zero_rates must be non-empty and of equal length")
        if np.any(t_arr <= 0) or np.any(np.diff(t_arr) <= 0):
            raise ValueError("tenors must be strictly positive and strictly increasing")
        if interp_mode not in ("log_linear_df", "monotone_cubic"):
            raise ValueError(f"Unsupported interp_mode: {interp_mode}")

        self.tenors = t_arr
        self.zero_rates = z_arr
        self.interp_mode = interp_mode
        self._log_dfs = -t_arr * z_arr
        self._pchip_d = _pchip_slopes(t_arr, z_arr) if t_arr.size >= 2 else None

    def zero_rate(self, t: float) -> float:
        """Return continuously compounded zero rate r(t) at maturity t (years)."""
        if t <= 0 or self.tenors.size == 1 or t <= self.tenors[0]:
            return float(self.zero_rates[0])
        if t >= self.tenors[-1]:
            return float(self.zero_rates[-1])

        if self.interp_mode == "monotone_cubic" and self._pchip_d is not None:
            return _pchip_eval(self.tenors, self.zero_rates, self._pchip_d, t)

        # log_linear_df: linear interpolation on log P(0, t) = -t * r(t)
        log_df = float(np.interp(t, self.tenors, self._log_dfs))
        return -log_df / t

    def discount_factor(self, t: float) -> float:
        """Return discount factor P(0, t) = exp(-r(t) * t)."""
        if t <= 0:
            return 1.0
        r = self.zero_rate(t)
        return math.exp(-r * t)

    def forward_rate(self, t1: float, t2: float) -> float:
        """
        Compute continuously compounded forward rate f(0; t1, t2) over [t1, t2]:
            f(0; t1, t2) = -[ln P(0, t2) - ln P(0, t1)] / (t2 - t1)
        """
        if t2 <= t1 or t1 < 0:
            raise ValueError(f"Require 0 <= t1 < t2, got t1={t1}, t2={t2}")
        df1 = self.discount_factor(t1)
        df2 = self.discount_factor(t2)
        return -math.log(df2 / df1) / (t2 - t1)

    def parallel_shift(self, bump_bps: float) -> ZeroCurve:
        """Return a new ZeroCurve with all continuously compounded zero rates shifted by `bump_bps`."""
        shift = bump_bps / 10_000.0
        return ZeroCurve(
            tenors=self.tenors.copy(),
            zero_rates=self.zero_rates + shift,
            interp_mode=self.interp_mode,
        )

    def with_key_rate_tent_bump(
        self,
        key_tenors: Sequence[float],
        active_index: int,
        bump_bps: float,
        dense_grid: Sequence[float] | None = None,
    ) -> ZeroCurve:
        """
        Apply a Ho (1992) piecewise-linear tent perturbation centered at `key_tenors[active_index]`
        such that the sum of all key-rate tent weights equals a flat parallel shift of `bump_bps`
        across the entire maturity spectrum.
        """
        kt = np.asarray(key_tenors, dtype=np.float64)
        n_k = kt.size
        if not (0 <= active_index < n_k):
            raise IndexError(f"active_index {active_index} out of bounds for {n_k} key tenors")

        if dense_grid is None:
            grid = np.unique(
                np.concatenate([self.tenors, kt, np.linspace(0.25, max(self.tenors[-1], kt[-1]), 120)])
            )
        else:
            grid = np.asarray(dense_grid, dtype=np.float64)

        shift = bump_bps / 10_000.0
        weights = np.zeros_like(grid)

        for idx, t in enumerate(grid):
            if active_index == 0 and t <= kt[0]:
                weights[idx] = 1.0
            elif active_index == n_k - 1 and t >= kt[-1]:
                weights[idx] = 1.0
            elif active_index > 0 and kt[active_index - 1] <= t <= kt[active_index]:
                weights[idx] = (t - kt[active_index - 1]) / (kt[active_index] - kt[active_index - 1])
            elif active_index < n_k - 1 and kt[active_index] <= t <= kt[active_index + 1]:
                weights[idx] = (kt[active_index + 1] - t) / (kt[active_index + 1] - kt[active_index])
            else:
                weights[idx] = 0.0

        base_zeros = np.array([self.zero_rate(float(t)) for t in grid], dtype=np.float64)
        bumped_zeros = base_zeros + shift * weights
        return ZeroCurve(tenors=grid, zero_rates=bumped_zeros, interp_mode=self.interp_mode)


def bootstrap_par_curve(
    instruments: Sequence[ParInstrument],
    interp_mode: str = "log_linear_df",
) -> ZeroCurve:
    """
    Bootstrap an exact zero-coupon curve from par coupon instruments so that every
    input benchmark instrument prices to exactly Par (100.000000).
    """
    sorted_inst = sorted(instruments, key=lambda x: x.maturity)
    if not sorted_inst:
        raise ValueError("Must provide at least one ParInstrument")

    known_t: List[float] = []
    known_df: List[float] = []

    def _interp_df(t: float, t_new: float, df_new: float) -> float:
        if t <= 0:
            return 1.0
        if not known_t:
            return math.exp(math.log(df_new) * (t / t_new))
        if t <= known_t[-1]:
            if t <= known_t[0]:
                return math.exp(math.log(known_df[0]) * (t / known_t[0]))
            log_dfs = [math.log(d) for d in known_df]
            log_d = float(np.interp(t, known_t, log_dfs))
            return math.exp(log_d)
        t_prev = known_t[-1]
        df_prev = known_df[-1]
        w = (t - t_prev) / (t_new - t_prev)
        log_d = (1.0 - w) * math.log(df_prev) + w * math.log(df_new)
        return math.exp(log_d)

    for inst in sorted_inst:
        T = inst.maturity
        freq = inst.frequency
        dt = 1.0 / freq
        n_steps = int(round(T * freq))
        coupon = inst.par_yield / freq

        if n_steps <= 1:
            df_T = 1.0 / (1.0 + coupon)
        else:
            pay_times = [j * dt for j in range(1, n_steps)]
            t_prev = known_t[-1] if known_t else 0.0
            intermediate = [t for t in pay_times if t > t_prev + 1e-9]

            if not intermediate:
                pv_earlier = sum(coupon * _interp_df(t, T, 0.5) for t in pay_times)
                df_T = (1.0 - pv_earlier) / (1.0 + coupon)
            else:
                lo, hi = 1e-6, 1.5
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    pv = sum(coupon * _interp_df(t, T, mid) for t in pay_times) + (1.0 + coupon) * mid
                    if pv > 1.0:
                        hi = mid
                    else:
                        lo = mid
                df_T = 0.5 * (lo + hi)

        known_t.append(T)
        known_df.append(df_T)

    zeros = [-math.log(df) / t for t, df in zip(known_t, known_df)]
    return ZeroCurve(tenors=known_t, zero_rates=zeros, interp_mode=interp_mode)


@dataclass(frozen=True)
class NSSParameters:
    """
    Nelson-Siegel-Svensson (1994) 6-parameter yield curve model:
      y(tau) = beta0
             + beta1 * [(1 - exp(-tau/tau1)) / (tau/tau1)]
             + beta2 * [(1 - exp(-tau/tau1)) / (tau/tau1) - exp(-tau/tau1)]
             + beta3 * [(1 - exp(-tau/tau2)) / (tau/tau2) - exp(-tau/tau2)]
    """

    beta0: float
    beta1: float
    beta2: float
    beta3: float
    tau1: float
    tau2: float
    rmse_bps: float = 0.0

    def evaluate(self, tau: float | np.ndarray) -> float | np.ndarray:
        """Evaluate the NSS zero yield at maturity `tau` (in years)."""
        t = np.maximum(np.asarray(tau, dtype=np.float64), 1e-8)
        x1 = t / self.tau1
        x2 = t / self.tau2
        f1 = (1.0 - np.exp(-x1)) / x1
        f2 = f1 - np.exp(-x1)
        f3 = (1.0 - np.exp(-x2)) / x2 - np.exp(-x2)
        y = self.beta0 + self.beta1 * f1 + self.beta2 * f2 + self.beta3 * f3
        if np.ndim(tau) == 0:
            return float(y)
        return y

    def to_zero_curve(self, tenors: Sequence[float]) -> ZeroCurve:
        t_arr = np.asarray(tenors, dtype=np.float64)
        z_arr = np.asarray(self.evaluate(t_arr), dtype=np.float64)
        return ZeroCurve(tenors=t_arr, zero_rates=z_arr, interp_mode="monotone_cubic")


def fit_nss_curve(
    tenors: Sequence[float],
    observed_zero_rates: Sequence[float],
) -> NSSParameters:
    """
    Calibrate Nelson-Siegel-Svensson (NSS) parameters (beta0, beta1, beta2, beta3, tau1, tau2)
    via Separable Nonlinear Least Squares (Golub-Pereyra variable projection):
    For any given decay pair `(tau1, tau2)`, the optimal `(beta0, beta1, beta2, beta3)`
    is solved in closed form via QR/SVD Ordinary Least Squares (`np.linalg.lstsq`),
    and `(tau1, tau2)` is refined via multi-stage coarse-to-fine coordinate pattern search.
    """
    t_arr = np.asarray(tenors, dtype=np.float64)
    y_arr = np.asarray(observed_zero_rates, dtype=np.float64)
    if t_arr.size != y_arr.size or t_arr.size < 4:
        raise ValueError("Require at least 4 (tenor, rate) observations to fit NSS curve")

    def _solve_betas(tau1: float, tau2: float) -> Tuple[np.ndarray, float]:
        x1 = np.maximum(t_arr / tau1, 1e-8)
        x2 = np.maximum(t_arr / tau2, 1e-8)
        f1 = (1.0 - np.exp(-x1)) / x1
        f2 = f1 - np.exp(-x1)
        f3 = (1.0 - np.exp(-x2)) / x2 - np.exp(-x2)
        X = np.column_stack([np.ones_like(t_arr), f1, f2, f3])
        betas, *_ = np.linalg.lstsq(X, y_arr, rcond=None)
        resid = X @ betas - y_arr
        ssr = float(np.sum(resid ** 2))
        return betas, ssr

    best_ssr = float("inf")
    best_t1, best_t2 = 1.5, 6.0
    best_betas = np.zeros(4, dtype=np.float64)

    # Stage 1: Global grid search over (tau1, tau2)
    for t1 in np.linspace(0.3, 4.0, 20):
        for t2 in np.linspace(2.0, 15.0, 25):
            if abs(t1 - t2) < 0.35:
                continue
            betas, ssr = _solve_betas(float(t1), float(t2))
            if ssr < best_ssr:
                best_ssr = ssr
                best_t1, best_t2 = float(t1), float(t2)
                best_betas = betas

    # Stage 2: Hooke-Jeeves / coordinate pattern refinement on (tau1, tau2)
    step = 0.25
    while step > 1e-7:
        improved = False
        for dt1 in (-step, 0.0, step):
            for dt2 in (-step, 0.0, step):
                if dt1 == 0.0 and dt2 == 0.0:
                    continue
                cand_t1 = max(0.10, min(15.0, best_t1 + dt1))
                cand_t2 = max(0.40, min(30.0, best_t2 + dt2))
                if abs(cand_t1 - cand_t2) < 0.20:
                    continue
                betas, ssr = _solve_betas(cand_t1, cand_t2)
                if ssr < best_ssr:
                    best_ssr = ssr
                    best_t1, best_t2 = cand_t1, cand_t2
                    best_betas = betas
                    improved = True
        if not improved:
            step *= 0.5

    rmse_bps = math.sqrt(best_ssr / t_arr.size) * 10_000.0
    return NSSParameters(
        beta0=float(best_betas[0]),
        beta1=float(best_betas[1]),
        beta2=float(best_betas[2]),
        beta3=float(best_betas[3]),
        tau1=best_t1,
        tau2=best_t2,
        rmse_bps=rmse_bps,
    )
