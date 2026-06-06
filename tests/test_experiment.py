"""Tests for ``abtesting.experiment``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from abtesting import Experiment, ExperimentResult


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


class TestConstruction:
    def test_invalid_metric_type_raises(self) -> None:
        with pytest.raises(ValueError):
            Experiment([1, 2, 3], [1, 2, 3], metric_type="categorical")

    def test_empty_group_raises(self) -> None:
        with pytest.raises(ValueError):
            Experiment([], [1, 2, 3])
        with pytest.raises(ValueError):
            Experiment([1, 2, 3], [])

    def test_stores_sample_sizes(self) -> None:
        exp = Experiment([1, 2, 3], [4, 5])
        assert exp.n_control == 3
        assert exp.n_treatment == 2


class TestTtest:
    def test_matches_scipy_welch(self, rng: np.random.Generator) -> None:
        control = rng.normal(100, 15, 500)
        treatment = rng.normal(105, 15, 500)
        exp = Experiment(control, treatment)
        result = exp.ttest()

        expected = stats.ttest_ind(treatment, control, equal_var=False)
        assert isinstance(result, ExperimentResult)
        assert result.statistic == pytest.approx(expected.statistic)
        assert result.p_value == pytest.approx(expected.pvalue)
        assert result.test_name == "Welch's t-test"

    def test_detects_real_effect(self, rng: np.random.Generator) -> None:
        control = rng.normal(100, 10, 2000)
        treatment = rng.normal(106, 10, 2000)
        result = Experiment(control, treatment).ttest()
        assert result.is_significant is True
        assert result.confidence_interval[0] > 0  # CI on difference excludes zero

    def test_no_effect_not_significant(self, rng: np.random.Generator) -> None:
        control = rng.normal(50, 5, 1000)
        treatment = rng.normal(50, 5, 1000)
        result = Experiment(control, treatment).ttest()
        assert result.is_significant is False
        lo, hi = result.confidence_interval
        assert lo < 0 < hi

    def test_is_significant_iff_p_below_alpha(self, rng: np.random.Generator) -> None:
        control = rng.normal(0.0, 1.0, 300)
        treatment = rng.normal(0.25, 1.0, 300)
        result = Experiment(control, treatment).ttest(alpha=0.05)
        assert result.is_significant == (result.p_value < 0.05)
        # Same data, tiny alpha -> the relationship still holds.
        strict = Experiment(control, treatment).ttest(alpha=1e-9)
        assert strict.is_significant == (strict.p_value < 1e-9)

    def test_one_sided_alternative_has_infinite_bound(self, rng: np.random.Generator) -> None:
        control = rng.normal(0, 1, 200)
        treatment = rng.normal(0.5, 1, 200)
        result = Experiment(control, treatment).ttest(alternative="greater")
        assert np.isinf(result.confidence_interval[1])

    def test_invalid_alternative_raises(self) -> None:
        with pytest.raises(ValueError):
            Experiment([1, 2, 3], [4, 5, 6]).ttest(alternative="bigger")

    def test_single_element_group_raises(self) -> None:
        with pytest.raises(ValueError):
            Experiment([1.0], [2.0, 3.0, 4.0]).ttest()

    @pytest.mark.filterwarnings("ignore:Precision loss occurred")
    def test_identical_constant_groups_not_significant(self) -> None:
        # Zero variance in both arms -> nan p-value -> never "significant".
        exp = Experiment([5.0, 5.0, 5.0], [5.0, 5.0, 5.0])
        result = exp.ttest()
        assert result.is_significant is False


class TestChiSquared:
    def test_detects_conversion_difference(self, rng: np.random.Generator) -> None:
        control = rng.binomial(1, 0.10, 5000)
        treatment = rng.binomial(1, 0.13, 5000)
        result = Experiment(control, treatment, metric_type="binary").chi_squared()
        assert result.is_significant is True
        assert result.test_name == "Chi-squared test"
        assert result.confidence_interval[0] > 0  # treatment converts more

    def test_matches_scipy_contingency(self, rng: np.random.Generator) -> None:
        control = rng.binomial(1, 0.2, 800)
        treatment = rng.binomial(1, 0.25, 800)
        exp = Experiment(control, treatment, metric_type="binary")
        result = exp.chi_squared()

        c1 = int(control.sum())
        t1 = int(treatment.sum())
        table = [[c1, len(control) - c1], [t1, len(treatment) - t1]]
        chi2, p, _, _ = stats.chi2_contingency(table, correction=False)
        assert result.statistic == pytest.approx(chi2)
        assert result.p_value == pytest.approx(p)

    def test_non_binary_data_raises(self, rng: np.random.Generator) -> None:
        exp = Experiment(rng.normal(0, 1, 50), rng.normal(0, 1, 50))
        with pytest.raises(ValueError):
            exp.chi_squared()


class TestBootstrap:
    def test_ci_brackets_true_mean_difference(self, rng: np.random.Generator) -> None:
        # True difference in means is 5.
        control = rng.normal(20, 4, 1500)
        treatment = rng.normal(25, 4, 1500)
        exp = Experiment(control, treatment)
        lo, hi = exp.bootstrap_ci(n_bootstrap=2000, random_state=1)
        assert lo < 5.0 < hi

    def test_reproducible_with_seed(self, rng: np.random.Generator) -> None:
        exp = Experiment(rng.normal(0, 1, 200), rng.normal(1, 1, 200))
        a = exp.bootstrap_ci(n_bootstrap=500, random_state=42)
        b = exp.bootstrap_ci(n_bootstrap=500, random_state=42)
        assert a == b

    def test_invalid_ci_raises(self) -> None:
        with pytest.raises(ValueError):
            Experiment([1, 2, 3], [4, 5, 6]).bootstrap_ci(ci=1.5)


class TestMannWhitney:
    def test_detects_shift_on_non_normal_data(self, rng: np.random.Generator) -> None:
        # Exponential (heavily skewed) with a clear shift.
        control = rng.exponential(1.0, 1000)
        treatment = rng.exponential(1.5, 1000)
        result = Experiment(control, treatment).mann_whitney()
        assert result.is_significant is True
        assert result.test_name == "Mann-Whitney U"
        assert result.effect_size > 0  # treatment stochastically larger

    def test_no_shift_not_significant(self, rng: np.random.Generator) -> None:
        control = rng.exponential(1.0, 800)
        treatment = rng.exponential(1.0, 800)
        result = Experiment(control, treatment).mann_whitney()
        assert result.is_significant is False


class TestSummaryAndExport:
    def test_summary_fields(self, rng: np.random.Generator) -> None:
        control = rng.normal(10, 2, 100)
        treatment = rng.normal(12, 2, 100)
        summary = Experiment(control, treatment).summary()
        assert summary["n_control"] == 100
        assert summary["control_mean"] == pytest.approx(control.mean())
        assert summary["absolute_effect"] == pytest.approx(
            treatment.mean() - control.mean()
        )

    def test_relative_lift(self) -> None:
        exp = Experiment([10.0, 10.0, 10.0], [11.0, 11.0, 11.0])
        assert exp.relative_lift() == pytest.approx(0.1)

    def test_relative_lift_nan_when_control_mean_zero(self) -> None:
        exp = Experiment([-1.0, 0.0, 1.0], [1.0, 2.0, 3.0])
        assert np.isnan(exp.relative_lift())

    def test_to_dataframe_continuous(self, rng: np.random.Generator) -> None:
        df = Experiment(rng.normal(0, 1, 100), rng.normal(0.5, 1, 100)).to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert df.iloc[0]["test_name"] == "Welch's t-test"

    def test_to_dataframe_binary(self, rng: np.random.Generator) -> None:
        c = rng.binomial(1, 0.1, 500)
        t = rng.binomial(1, 0.2, 500)
        df = Experiment(c, t, metric_type="binary").to_dataframe()
        assert df.iloc[0]["test_name"] == "Chi-squared test"
