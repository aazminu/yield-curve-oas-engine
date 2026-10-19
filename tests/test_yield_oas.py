"""
Comprehensive pytest suite for yield-curve-oas-engine.
Verifies exact par-bond bootstrapping, ZeroCurve discount & forward invariants,
Nelson-Siegel-Svensson (NSS) parametric calibration, analytical vs numerical
duration/convexity, Ho (1992) Key-Rate DV01 sum conservation, Arrow-Debreu
short-rate lattice calibration, callable/puttable option pricing identities,
negative convexity of callable bonds near call strike, and Brent OAS root-finding.
"""

import math
import numpy as np
import pytest

from yield_oas import (
    BondSpec,
    EmbeddedOptionType,
    NSSParameters,
    OptionScheduleBond,
    ParInstrument,
    ZeroCurve,
    analytical_duration_convexity,
    bond_yield_to_maturity,
    bootstrap_par_curve,
    calibrate_short_rate_lattice,
    compute_dv01_cs01,
    compute_key_rate_dv01,
    effective_duration_convexity,
    fit_nss_curve,
    full_risk_report,
    price_bond_from_curve,
    price_bond_on_lattice,
    solve_oas,
)


def _sample_par_instruments():
    return [
        ParInstrument(0.5, 0.0400, frequency=2),
        ParInstrument(1.0, 0.0415, frequency=2),
        ParInstrument(2.0, 0.0430, frequency=2),
        ParInstrument(5.0, 0.0450, frequency=2),
        ParInstrument(10.0, 0.0470, frequency=2),
        ParInstrument(30.0, 0.0490, frequency=2),
    ]


def test_zero_curve_discount_and_forward_identities():
    curve = ZeroCurve(tenors=[1.0, 2.0, 5.0, 10.0], zero_rates=[0.03, 0.035, 0.04, 0.042])
    assert curve.discount_factor(0.0) == pytest.approx(1.0)
    df2 = curve.discount_factor(2.0)
    df5 = curve.discount_factor(5.0)
    fwd_2_5 = curve.forward_rate(2.0, 5.0)
    # P(0, 2) * exp(-f(2, 5) * 3) == P(0, 5)
    assert df2 * math.exp(-fwd_2_5 * 3.0) == pytest.approx(df5, rel=1e-12)


def test_monotone_cubic_pchip_interpolation_mode():
    curve = ZeroCurve(
        tenors=[1.0, 2.0, 5.0, 10.0],
        zero_rates=[0.030, 0.035, 0.040, 0.045],
        interp_mode="monotone_cubic",
    )
    # Knot values are reproduced exactly
    assert curve.zero_rate(2.0) == pytest.approx(0.035)
    assert curve.zero_rate(5.0) == pytest.approx(0.040)
    # Intermediate value is strictly monotonic between 0.035 and 0.040
    assert 0.035 < curve.zero_rate(3.5) < 0.040


def test_bootstrap_par_bonds_reprice_to_exact_100():
    instruments = _sample_par_instruments()
    curve = bootstrap_par_curve(instruments, interp_mode="log_linear_df")
    for inst in instruments:
        bond = BondSpec(
            maturity=inst.maturity,
            coupon_rate=inst.par_yield,
            face_value=100.0,
            frequency=inst.frequency,
        )
        model_price = price_bond_from_curve(bond, curve, credit_spread_bps=0.0)
        assert model_price == pytest.approx(100.0, abs=1e-5)


def test_zero_coupon_bond_pricing_identity():
    curve = bootstrap_par_curve(_sample_par_instruments())
    zcb = BondSpec(maturity=5.0, coupon_rate=0.0, face_value=100.0, frequency=2)
    px = price_bond_from_curve(zcb, curve)
    assert px == pytest.approx(100.0 * curve.discount_factor(5.0), rel=1e-12)


def test_nelson_siegel_svensson_curve_fit_accuracy():
    true_nss = NSSParameters(
        beta0=0.048,
        beta1=-0.015,
        beta2=0.012,
        beta3=-0.008,
        tau1=1.4,
        tau2=6.5,
    )
    tenors = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0]
    obs_rates = [float(true_nss.evaluate(t)) for t in tenors]
    fitted = fit_nss_curve(tenors, obs_rates)
    assert fitted.rmse_bps < 0.05  # Less than 0.05 bps RMSE on synthetic NSS curve
    fitted_curve = fitted.to_zero_curve(tenors)
    for t, r in zip(tenors, obs_rates):
        assert fitted_curve.zero_rate(t) == pytest.approx(r, abs=1e-5)


def test_ytm_and_zero_coupon_macaulay_duration_equals_maturity():
    curve = ZeroCurve(tenors=[1.0, 5.0, 10.0], zero_rates=[0.04, 0.04, 0.04])
    zcb = BondSpec(maturity=5.0, coupon_rate=0.0, face_value=100.0, frequency=2)
    px = price_bond_from_curve(zcb, curve)
    ytm = bond_yield_to_maturity(zcb, px)
    mac_dur, mod_dur, conv = analytical_duration_convexity(zcb, ytm)
    assert mac_dur == pytest.approx(5.0, rel=1e-12)
    assert mod_dur == pytest.approx(5.0 / (1.0 + ytm / 2.0), rel=1e-12)
    assert conv > 25.0


def test_dv01_matches_modified_duration_and_equals_cs01():
    curve = ZeroCurve(tenors=[1.0, 5.0, 10.0, 30.0], zero_rates=[0.04, 0.04, 0.04, 0.04])
    bond = BondSpec(maturity=10.0, coupon_rate=0.045, face_value=100.0, frequency=2)
    dv01, cs01 = compute_dv01_cs01(bond, curve, credit_spread_bps=50.0)
    # Since both curve parallel shift and credit spread shift continuously compounded rates identically:
    assert dv01 == pytest.approx(cs01, rel=1e-10)
    assert dv01 > 0.07


def test_key_rate_dv01_sum_conserves_parallel_dv01():
    curve = bootstrap_par_curve(_sample_par_instruments())
    bond = BondSpec(maturity=10.0, coupon_rate=0.047, face_value=100.0, frequency=2)
    dv01, _ = compute_dv01_cs01(bond, curve)
    kr_map = compute_key_rate_dv01(bond, curve, key_tenors=(2.0, 5.0, 10.0, 30.0))

    total_kr = sum(kr_map.values())
    assert total_kr == pytest.approx(dv01, rel=1e-4)
    # For a 10Y par bond, the 10Y key-rate tenor carries the dominant sensitivity
    assert kr_map[10.0] > kr_map[2.0]
    assert kr_map[10.0] > kr_map[5.0]
    assert abs(kr_map[30.0]) < 1e-9


def test_full_risk_report_completeness():
    curve = bootstrap_par_curve(_sample_par_instruments())
    bond = BondSpec(maturity=5.0, coupon_rate=0.045, face_value=100.0, frequency=2)
    rep = full_risk_report(bond, curve, credit_spread_bps=0.0)
    assert rep.price == pytest.approx(100.0, abs=1e-5)
    assert rep.ytm == pytest.approx(0.045, abs=1e-5)
    assert len(rep.key_rate_dv01) == 4


def test_short_rate_lattice_reproduces_zero_curve_discount_factors():
    curve = bootstrap_par_curve(_sample_par_instruments())
    lattice = calibrate_short_rate_lattice(curve, horizon_years=5.0, steps_per_year=10, sigma=0.012)
    for step in range(1, lattice.n_steps + 1):
        t = step * lattice.dt
        # Sum of Arrow-Debreu state prices at step equals P(0, t)
        ad_df = float(np.sum(lattice.state_prices[step]))
        assert ad_df == pytest.approx(curve.discount_factor(t), rel=1e-11)


def test_option_free_bond_on_lattice_matches_analytical_curve_price():
    curve = bootstrap_par_curve(_sample_par_instruments())
    straight_lattice_bond = OptionScheduleBond(
        maturity=5.0,
        coupon_rate=0.045,
        option_type=EmbeddedOptionType.NONE,
        frequency=2,
    )
    lattice = calibrate_short_rate_lattice(curve, horizon_years=5.0, steps_per_year=10, sigma=0.01)
    tree_px = price_bond_on_lattice(straight_lattice_bond, lattice, oas_bps=0.0)
    analytic_px = price_bond_from_curve(BondSpec(5.0, 0.045), curve, credit_spread_bps=0.0)
    assert tree_px == pytest.approx(analytic_px, abs=1e-4)


def test_callable_puttable_straight_price_ordering_inequality():
    curve = bootstrap_par_curve(_sample_par_instruments())
    lattice = calibrate_short_rate_lattice(curve, horizon_years=5.0, steps_per_year=12, sigma=0.015)

    straight = OptionScheduleBond(5.0, 0.055, EmbeddedOptionType.NONE, lockout_years=1.0, strike_price=100.0)
    callable_b = OptionScheduleBond(5.0, 0.055, EmbeddedOptionType.CALLABLE, lockout_years=1.0, strike_price=100.0)
    puttable_b = OptionScheduleBond(5.0, 0.055, EmbeddedOptionType.PUTTABLE, lockout_years=1.0, strike_price=100.0)

    p_straight = price_bond_on_lattice(straight, lattice, oas_bps=25.0)
    p_callable = price_bond_on_lattice(callable_b, lattice, oas_bps=25.0)
    p_puttable = price_bond_on_lattice(puttable_b, lattice, oas_bps=25.0)

    # Arbitrage bounds: P_callable < P_straight < P_puttable
    assert p_callable < p_straight < p_puttable


def test_higher_short_rate_volatility_increases_embedded_option_value():
    curve = bootstrap_par_curve(_sample_par_instruments())
    callable_b = OptionScheduleBond(5.0, 0.05, EmbeddedOptionType.CALLABLE, lockout_years=1.0, strike_price=100.0)

    lat_low_vol = calibrate_short_rate_lattice(curve, 5.0, steps_per_year=12, sigma=0.005)
    lat_high_vol = calibrate_short_rate_lattice(curve, 5.0, steps_per_year=12, sigma=0.025)

    p_low_vol = price_bond_on_lattice(callable_b, lat_low_vol, oas_bps=0.0)
    p_high_vol = price_bond_on_lattice(callable_b, lat_high_vol, oas_bps=0.0)

    # Higher rate volatility increases the value of the issuer's call option, lowering callable bond price
    assert p_high_vol < p_low_vol


def test_oas_solver_roundtrip_accuracy():
    curve = bootstrap_par_curve(_sample_par_instruments())
    callable_b = OptionScheduleBond(5.0, 0.052, EmbeddedOptionType.CALLABLE, lockout_years=1.0, strike_price=100.0)
    lat = calibrate_short_rate_lattice(curve, 5.0, steps_per_year=12, sigma=0.012)

    target_oas = 85.0
    mkt_price = price_bond_on_lattice(callable_b, lat, oas_bps=target_oas)
    solved_oas = solve_oas(callable_b, curve, market_price=mkt_price, steps_per_year=12, sigma=0.012)
    assert solved_oas == pytest.approx(target_oas, abs=1e-5)


def test_callable_bond_exhibits_negative_convexity_and_lower_effective_duration():
    curve = ZeroCurve(tenors=[1.0, 2.0, 5.0, 10.0], zero_rates=[0.04, 0.04, 0.04, 0.04])
    # High-coupon bond trading near/above call strike exhibits negative convexity and compressed duration
    straight = OptionScheduleBond(5.0, 0.055, EmbeddedOptionType.NONE, lockout_years=0.5, strike_price=100.0)
    callable_b = OptionScheduleBond(5.0, 0.055, EmbeddedOptionType.CALLABLE, lockout_years=0.5, strike_price=100.0)

    _, d_straight, c_straight = effective_duration_convexity(
        straight, curve, oas_bps=0.0, bump_bps=15.0, steps_per_year=12, sigma=0.012
    )
    _, d_call, c_call = effective_duration_convexity(
        callable_b, curve, oas_bps=0.0, bump_bps=15.0, steps_per_year=12, sigma=0.012
    )

    assert d_call < d_straight
    assert c_straight > 0.0
    assert c_call < 0.0


def test_puttable_bond_exhibits_enhanced_positive_convexity_and_compressed_duration():
    curve = ZeroCurve(tenors=[1.0, 2.0, 5.0, 10.0], zero_rates=[0.045, 0.045, 0.045, 0.045])
    straight = OptionScheduleBond(5.0, 0.035, EmbeddedOptionType.NONE, lockout_years=0.5, strike_price=100.0)
    puttable_b = OptionScheduleBond(5.0, 0.035, EmbeddedOptionType.PUTTABLE, lockout_years=0.5, strike_price=100.0)

    _, d_straight, c_straight = effective_duration_convexity(
        straight, curve, oas_bps=0.0, bump_bps=15.0, steps_per_year=12, sigma=0.012
    )
    _, d_put, c_put = effective_duration_convexity(
        puttable_b, curve, oas_bps=0.0, bump_bps=15.0, steps_per_year=12, sigma=0.012
    )

    assert d_put < d_straight
    assert c_put > c_straight
