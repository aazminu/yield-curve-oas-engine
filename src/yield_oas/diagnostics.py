"""Deterministic runtime diagnostics and invariant verification for `yield_oas`."""

from __future__ import annotations

import time
from typing import Any

from .curve import ParInstrument, bootstrap_par_curve, fit_nss_curve

def _domain_probe(seed: int) -> dict:
    instruments = [
        ParInstrument(0.5, 0.0400, frequency=2),
        ParInstrument(1.0, 0.0415, frequency=2),
        ParInstrument(2.0, 0.0430, frequency=2),
        ParInstrument(5.0, 0.0450, frequency=2),
        ParInstrument(10.0, 0.0470, frequency=2),
        ParInstrument(30.0, 0.0490, frequency=2),
    ]
    zc = bootstrap_par_curve(instruments)
    nss = fit_nss_curve(zc.tenors, zc.zero_rates)
    df10 = zc.discount_factor(10.0)
    return {
        "tenors_bootstrapped": len(zc.tenors),
        "ten_year_zero_rate": round(zc.zero_rate(10.0), 6),
        "ten_year_df": round(df10, 6),
        "nss_beta0": round(nss.beta0, 6),
        "invariants_ok": (len(zc.tenors) == 6 and 0.5 < df10 < 0.7 and nss.beta0 > 0),
    }


def run_diagnostic_suite(seed: int = 42) -> dict[str, Any]:
    """Execute deterministic domain probe for `yield-curve-oas-engine` and return structured metrics."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    t0 = time.perf_counter()
    metrics = _domain_probe(seed)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return {
        "repository": "yield-curve-oas-engine",
        "package": "yield_oas",
        "seed": seed,
        "elapsed_ms": elapsed_ms,
        "metrics": metrics,
        "status": "ok" if metrics.get("invariants_ok", False) else "failed",
    }


def validate_runtime_invariants(seed: int = 42) -> dict[str, Any]:
    """Validate deterministic reproducibility and domain invariants for `yield_oas`."""
    run_a = run_diagnostic_suite(seed=seed)
    run_b = run_diagnostic_suite(seed=seed)
    reproducible = run_a["metrics"] == run_b["metrics"]
    passed = bool(run_a["metrics"].get("invariants_ok", False)) and reproducible
    return {
        "repository": "yield-curve-oas-engine",
        "package": "yield_oas",
        "seed": seed,
        "reproducible": reproducible,
        "passed": passed,
        "metrics": run_a["metrics"],
    }
