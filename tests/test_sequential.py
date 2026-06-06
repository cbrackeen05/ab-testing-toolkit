"""Tests for ``abtesting.sequential``."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from abtesting.sequential import (
    SPRTResult,
    _wald_boundaries,
    always_valid_pvalue,
    sprt,
)


class TestSPRTBasics:
    def test_reject_on_large_effect_and_large_n(self) -> None:
        # Treatment converts at 20% vs a 12% alternative / 10% null, with lots of data.
        res = sprt(1000, 10_000, 2000, 10_000, h0_rate=0.10, h1_rate=0.12)
        assert isinstance(res, SPRTResult)
        assert res.decision == "reject_null"
        assert res.llr >= res.upper_boundary

    def test_continue_on_tiny_n(self) -> None:
        # Suggestive but tiny: the LLR sits between the boundaries.
        res = sprt(1, 10, 2, 10)
        assert res.decision == "continue"
        assert res.lower_boundary < res.llr < res.upper_boundary

    def test_accept_null_when_treatment_underperforms(self) -> None:
        # Treatment well below both the null and the alternative -> stop, accept null.
        res = sprt(1000, 10_000, 100, 10_000, h0_rate=0.10, h1_rate=0.12)
        assert res.decision == "accept_null"
        assert res.llr <= res.lower_boundary

    def test_boundaries_match_wald_formula(self) -> None:
        alpha, beta = 0.05, 0.20
        res = sprt(50, 500, 70, 500, alpha=alpha, beta=beta)
        assert res.upper_boundary == pytest.approx(math.log((1 - beta) / alpha))
        assert res.lower_boundary == pytest.approx(math.log(beta / (1 - alpha)))

    @pytest.mark.parametrize("alpha,beta", [(0.05, 0.20), (0.01, 0.10), (0.10, 0.30)])
    def test_boundary_helper(self, alpha: float, beta: float) -> None:
        lower, upper = _wald_boundaries(alpha, beta)
        assert lower < 0 < upper
        assert upper == pytest.approx(math.log((1 - beta) / alpha))

    @pytest.mark.parametrize("kwargs", [
        {"control_successes": 5, "control_n": 0, "treatment_successes": 1, "treatment_n": 10},
        {"control_successes": 20, "control_n": 10, "treatment_successes": 1, "treatment_n": 10},
        {"control_successes": 5, "control_n": 10, "treatment_successes": -1, "treatment_n": 10},
    ])
    def test_invalid_inputs_raise(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            sprt(**kwargs)

    def test_invalid_error_rates_raise(self) -> None:
        with pytest.raises(ValueError):
            sprt(5, 100, 8, 100, alpha=1.5)


class TestSPRTErrorControl:
    def test_aa_false_positive_rate_near_alpha(self) -> None:
        # 1000 A/A experiments with FIXED hypotheses (H0=0.10 vs H1=0.12) on true-0.10
        # data, peeking every 50 users. Wald's SPRT should keep the reject rate ~alpha,
        # in stark contrast to naive peeking which would inflate it badly.
        rng = np.random.default_rng(0)
        alpha, beta, n_max = 0.05, 0.20, 4_000
        rejects = 0
        trials = 1000
        for _ in range(trials):
            stream = rng.binomial(1, 0.10, n_max)
            cumulative = np.cumsum(stream)
            for n in range(50, n_max, 50):
                decision = sprt(
                    0, 1, int(cumulative[n - 1]), n,
                    alpha=alpha, beta=beta, h0_rate=0.10, h1_rate=0.12,
                ).decision
                if decision == "reject_null":
                    rejects += 1
                    break
                if decision == "accept_null":
                    break
        fpr = rejects / trials
        # Calibrated empirical value ~0.046; allow generous slack but require control.
        assert fpr < 0.08


class TestAlwaysValidPValue:
    def test_in_unit_interval(self) -> None:
        rng = np.random.default_rng(1)
        p = always_valid_pvalue(rng.normal(0, 1, 500), rng.normal(0, 1, 500))
        assert 0.0 < p <= 1.0

    def test_small_for_clear_effect(self) -> None:
        rng = np.random.default_rng(2)
        c = rng.normal(0.0, 1.0, 5_000)
        t = rng.normal(0.3, 1.0, 5_000)
        assert always_valid_pvalue(c, t) < 0.01

    def test_not_significant_for_no_effect(self) -> None:
        rng = np.random.default_rng(3)
        c = rng.normal(0.0, 1.0, 5_000)
        t = rng.normal(0.0, 1.0, 5_000)
        assert always_valid_pvalue(c, t) > 0.05

    def test_peeking_does_not_inflate_false_positives(self) -> None:
        # The headline property: peek 50 times at each of many A/A experiments and the
        # always-valid p-value should *rarely* ever dip below alpha (<= alpha in the
        # limit) -- whereas a naive t-test does so constantly.
        rng = np.random.default_rng(4)
        trials, n_max, step, alpha = 400, 2_000, 40, 0.05
        ever_below_avp = 0
        ever_below_naive = 0
        for _ in range(trials):
            c = rng.normal(0, 1, n_max)
            t = rng.normal(0, 1, n_max)
            avp_hit = naive_hit = False
            for n in range(20, n_max + 1, step):
                if not avp_hit and always_valid_pvalue(c[:n], t[:n]) < alpha:
                    avp_hit = True
                if not naive_hit:
                    _, p = stats.ttest_ind(t[:n], c[:n], equal_var=False)
                    if p < alpha:
                        naive_hit = True
                if avp_hit and naive_hit:
                    break
            ever_below_avp += avp_hit
            ever_below_naive += naive_hit
        avp_rate = ever_below_avp / trials
        naive_rate = ever_below_naive / trials
        # Always-valid stays controlled near alpha; naive peeking blows well past it.
        assert avp_rate < 0.10
        assert naive_rate > 0.20
        assert avp_rate < naive_rate

    def test_too_few_observations_raise(self) -> None:
        with pytest.raises(ValueError):
            always_valid_pvalue([1.0], [1.0, 2.0])

    def test_invalid_alpha_raises(self) -> None:
        with pytest.raises(ValueError):
            always_valid_pvalue([1.0, 2.0], [3.0, 4.0], alpha=0.0)
