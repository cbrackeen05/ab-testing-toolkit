"""Tests for ``abtesting.power``."""

from __future__ import annotations

import numpy as np
import pytest

from abtesting.power import (
    experiment_runtime_days,
    minimum_detectable_effect,
    minimum_sample_size,
    observed_power,
)


class TestMinimumSampleSize:
    def test_increases_as_mde_decreases(self) -> None:
        # Smaller effects are harder to detect -> need more samples.
        big = minimum_sample_size(0.10, 0.05)
        small = minimum_sample_size(0.10, 0.01)
        assert small > big

    def test_increases_as_power_increases(self) -> None:
        low = minimum_sample_size(0.10, 0.02, power=0.80)
        high = minimum_sample_size(0.10, 0.02, power=0.95)
        assert high > low

    def test_increases_as_alpha_decreases(self) -> None:
        lenient = minimum_sample_size(0.10, 0.02, alpha=0.10)
        strict = minimum_sample_size(0.10, 0.02, alpha=0.01)
        assert strict > lenient

    def test_continuous_matches_closed_form(self) -> None:
        from scipy import stats

        z_a = stats.norm.ppf(0.975)
        z_b = stats.norm.ppf(0.80)
        expected = int(np.ceil(2 * (z_a + z_b) ** 2 / 0.2**2))
        assert minimum_sample_size(0.0, 0.2, metric_type="continuous") == expected

    def test_returns_int(self) -> None:
        assert isinstance(minimum_sample_size(0.2, 0.03), int)

    @pytest.mark.parametrize("kwargs", [
        {"baseline_rate": 0.0, "minimum_detectable_effect": 0.02},
        {"baseline_rate": 1.0, "minimum_detectable_effect": 0.02},
        {"baseline_rate": 0.99, "minimum_detectable_effect": 0.02},  # p2 > 1
        {"baseline_rate": 0.10, "minimum_detectable_effect": 0.0},
    ])
    def test_invalid_inputs_raise(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            minimum_sample_size(**kwargs)


class TestMinimumDetectableEffect:
    def test_decreases_as_n_increases(self) -> None:
        small_n = minimum_detectable_effect(1_000, 0.10)
        large_n = minimum_detectable_effect(10_000, 0.10)
        assert large_n < small_n

    def test_roundtrips_with_sample_size(self) -> None:
        # Plug a sample size in, get an MDE out; it should be close to the original
        # MDE (within the binary baseline-variance approximation).
        n = minimum_sample_size(0.10, 0.02)
        recovered = minimum_detectable_effect(n, 0.10)
        assert recovered == pytest.approx(0.02, abs=2e-3)

    def test_continuous_roundtrips_exactly(self) -> None:
        n = minimum_sample_size(0.0, 0.3, metric_type="continuous")
        recovered = minimum_detectable_effect(n, 0.0, metric_type="continuous")
        assert recovered == pytest.approx(0.3, abs=2e-3)

    def test_invalid_n_raises(self) -> None:
        with pytest.raises(ValueError):
            minimum_detectable_effect(1, 0.10)


class TestObservedPower:
    def test_in_unit_interval(self) -> None:
        p = observed_power(500, 0.1, 1.0)
        assert 0.0 <= p <= 1.0

    def test_increases_with_sample_size(self) -> None:
        assert observed_power(2000, 0.1, 1.0) > observed_power(200, 0.1, 1.0)

    def test_increases_with_effect_size(self) -> None:
        assert observed_power(500, 0.3, 1.0) > observed_power(500, 0.1, 1.0)

    def test_matches_simulation(self) -> None:
        # The analytic power should match the empirical rejection rate of a simulated
        # two-sample z-test with the same parameters.
        rng = np.random.default_rng(0)
        n, effect, sigma, alpha = 200, 0.3, 1.0, 0.05
        from scipy import stats

        z_crit = stats.norm.ppf(1 - alpha / 2)
        rejections = 0
        trials = 4000
        for _ in range(trials):
            c = rng.normal(0.0, sigma, n)
            t = rng.normal(effect, sigma, n)
            se = np.sqrt(c.var(ddof=1) / n + t.var(ddof=1) / n)
            z = (t.mean() - c.mean()) / se
            if abs(z) > z_crit:
                rejections += 1
        empirical = rejections / trials
        analytic = observed_power(n, effect, sigma, alpha)
        assert analytic == pytest.approx(empirical, abs=0.04)

    @pytest.mark.parametrize("kwargs", [
        {"n": 1, "effect_size": 0.1, "baseline_std": 1.0},
        {"n": 100, "effect_size": 0.1, "baseline_std": 0.0},
    ])
    def test_invalid_inputs_raise(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            observed_power(**kwargs)


class TestExperimentRuntimeDays:
    def test_known_balanced_case(self) -> None:
        # n per group ~ 3839; balanced split -> 5000/day into each arm -> 1 day.
        assert experiment_runtime_days(10_000, 0.10, 0.02) == 1

    def test_more_traffic_means_fewer_days(self) -> None:
        slow = experiment_runtime_days(1_000, 0.10, 0.01)
        fast = experiment_runtime_days(5_000, 0.10, 0.01)
        assert fast < slow

    def test_uneven_split_takes_longer(self) -> None:
        balanced = experiment_runtime_days(2_000, 0.10, 0.01, traffic_split=0.5)
        skewed = experiment_runtime_days(2_000, 0.10, 0.01, traffic_split=0.1)
        assert skewed > balanced

    def test_matches_manual_computation(self) -> None:
        n = minimum_sample_size(0.10, 0.01)
        # Balanced split: each arm gets half the daily traffic.
        expected = int(np.ceil(n / (3_000 * 0.5)))
        assert experiment_runtime_days(3_000, 0.10, 0.01) == expected

    @pytest.mark.parametrize("kwargs", [
        {"daily_traffic": 0, "baseline_rate": 0.1, "mde": 0.01},
        {"daily_traffic": 1000, "baseline_rate": 0.1, "mde": 0.01, "traffic_split": 0.0},
        {"daily_traffic": 1000, "baseline_rate": 0.1, "mde": 0.01, "traffic_split": 1.0},
    ])
    def test_invalid_inputs_raise(self, kwargs: dict) -> None:
        with pytest.raises(ValueError):
            experiment_runtime_days(**kwargs)
