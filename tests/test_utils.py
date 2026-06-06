"""Tests for ``abtesting.utils``."""

from __future__ import annotations

import numpy as np
import pytest

from abtesting.utils import (
    SRMResult,
    check_sample_ratio_mismatch,
    cohens_d,
    log_transform,
    winsorize,
)


class TestSampleRatioMismatch:
    def test_balanced_split_is_not_a_mismatch(self) -> None:
        res = check_sample_ratio_mismatch(10_000, 10_000)
        assert isinstance(res, SRMResult)
        assert res.is_mismatch is False
        assert res.p_value > 0.05

    def test_clear_imbalance_is_flagged(self) -> None:
        # 10000 vs 8500 is a gross deviation from 50/50 and should trip the check.
        res = check_sample_ratio_mismatch(10_000, 8_500)
        assert res.is_mismatch is True
        assert res.p_value < 0.05

    def test_respects_non_even_expected_split(self) -> None:
        # A 90/10 holdout that actually landed ~90/10 is fine...
        ok = check_sample_ratio_mismatch(9_000, 1_000, expected_split=0.9)
        assert ok.is_mismatch is False
        # ...but the same counts judged against a 50/50 expectation is a mismatch.
        bad = check_sample_ratio_mismatch(9_000, 1_000, expected_split=0.5)
        assert bad.is_mismatch is True

    @pytest.mark.parametrize("split", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_split_raises(self, split: float) -> None:
        with pytest.raises(ValueError):
            check_sample_ratio_mismatch(100, 100, expected_split=split)

    def test_zero_total_raises(self) -> None:
        with pytest.raises(ValueError):
            check_sample_ratio_mismatch(0, 0)


class TestWinsorize:
    def test_clips_extreme_high_value(self) -> None:
        x = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
        out = winsorize(x, lower=0.0, upper=0.8)
        # 80th percentile of the data is 3.2, so 100 is pulled down to it.
        assert out.max() < 100.0
        assert out.min() == 0.0

    def test_does_not_mutate_input(self) -> None:
        x = np.array([0.0, 1.0, 2.0, 3.0, 100.0])
        original = x.copy()
        _ = winsorize(x)
        assert np.array_equal(x, original)

    def test_empty_input_returns_empty(self) -> None:
        out = winsorize(np.array([]))
        assert out.size == 0

    @pytest.mark.parametrize("lower,upper", [(0.5, 0.5), (0.9, 0.1), (-0.1, 0.9)])
    def test_invalid_bounds_raise(self, lower: float, upper: float) -> None:
        with pytest.raises(ValueError):
            winsorize(np.array([1.0, 2.0, 3.0]), lower=lower, upper=upper)


class TestLogTransform:
    def test_matches_log1p(self) -> None:
        x = np.array([0.0, 1.0, 9.0, 99.0])
        np.testing.assert_allclose(log_transform(x), np.log1p(x))

    def test_zero_maps_to_zero(self) -> None:
        assert log_transform(np.array([0.0]))[0] == 0.0

    def test_undefined_values_raise(self) -> None:
        with pytest.raises(ValueError):
            log_transform(np.array([0.0, -1.0]))


class TestCohensD:
    def test_zero_when_groups_identical(self) -> None:
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert cohens_d(x, x) == 0.0

    def test_sign_follows_treatment_minus_control(self) -> None:
        c = np.array([0.0, 0.0, 1.0, 1.0])
        t = np.array([2.0, 2.0, 3.0, 3.0])
        assert cohens_d(c, t) > 0
        assert cohens_d(t, c) < 0

    def test_recovers_known_effect_size(self) -> None:
        rng = np.random.default_rng(0)
        c = rng.normal(0.0, 1.0, 5_000)
        t = rng.normal(0.5, 1.0, 5_000)
        # True standardized effect is 0.5; large samples should land close.
        assert cohens_d(c, t) == pytest.approx(0.5, abs=0.05)

    def test_zero_pooled_sd_returns_zero(self) -> None:
        c = np.array([5.0, 5.0, 5.0])
        t = np.array([5.0, 5.0, 5.0])
        assert cohens_d(c, t) == 0.0

    @pytest.mark.parametrize("control,treatment", [([1.0], [1.0, 2.0]), ([], [1.0, 2.0])])
    def test_too_few_observations_raises(
        self, control: list[float], treatment: list[float]
    ) -> None:
        with pytest.raises(ValueError):
            cohens_d(np.array(control), np.array(treatment))
