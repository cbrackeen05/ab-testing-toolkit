"""Core experiment analysis: the :class:`Experiment` class and :class:`ExperimentResult`.

This module is the heart of the toolkit. An :class:`Experiment` wraps the control and
treatment samples of a single A/B test and exposes the standard battery of analyses:

- :meth:`Experiment.ttest` — Welch's t-test (the correct default for continuous A/B metrics)
- :meth:`Experiment.mann_whitney` — non-parametric alternative when normality is doubtful
- :meth:`Experiment.chi_squared` — proportion test for binary / conversion metrics
- :meth:`Experiment.bootstrap_ci` — assumption-light confidence interval on the mean difference

Every hypothesis test returns a uniform :class:`ExperimentResult` rather than raw SciPy
output, so results from different tests are directly comparable and serializable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .utils import cohens_d

_VALID_METRIC_TYPES = ("continuous", "binary", "ratio")
_VALID_ALTERNATIVES = ("two-sided", "less", "greater")


@dataclass
class ExperimentResult:
    """The outcome of a single statistical test, in a uniform shape.

    Attributes
    ----------
    test_name : str
        Human-readable name of the test that produced this result.
    statistic : float
        The test statistic (t, U, or chi-squared depending on the test).
    p_value : float
        The p-value under the null hypothesis of no difference.
    confidence_interval : tuple[float, float]
        Confidence interval on the effect (the treatment-minus-control difference in
        means, or in proportions for ``chi_squared``). One-sided alternatives produce
        an infinite bound on the unconstrained side.
    effect_size : float
        A standardized effect-size measure appropriate to the test (Cohen's *d* for
        the t-test, rank-biserial correlation for Mann-Whitney, Cohen's *h* for the
        proportion test).
    relative_lift : float
        ``(treatment_mean - control_mean) / control_mean`` — the headline business lift.
    is_significant : bool
        ``True`` iff ``p_value < alpha``.
    alpha : float
        Significance level used for the decision.
    n_control : int
        Control sample size.
    n_treatment : int
        Treatment sample size.
    """

    test_name: str
    statistic: float
    p_value: float
    confidence_interval: tuple[float, float]
    effect_size: float
    relative_lift: float
    is_significant: bool
    alpha: float
    n_control: int
    n_treatment: int


class Experiment:
    """A single A/B experiment: one control sample and one treatment sample.

    Parameters
    ----------
    control : array-like
        Metric values for the control group.
    treatment : array-like
        Metric values for the treatment group.
    metric_type : {"continuous", "binary", "ratio"}, default "continuous"
        The kind of metric. ``"binary"`` indicates 0/1 conversion data and selects
        the proportion test as the canonical analysis; ``"continuous"`` and
        ``"ratio"`` default to Welch's t-test.

    Raises
    ------
    ValueError
        If either group is empty or ``metric_type`` is not recognized.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> control = rng.normal(100, 15, 1000)
    >>> treatment = rng.normal(103, 15, 1000)
    >>> exp = Experiment(control, treatment)
    >>> result = exp.ttest()
    >>> result.is_significant
    True
    """

    def __init__(
        self,
        control: np.ndarray,
        treatment: np.ndarray,
        metric_type: str = "continuous",
    ) -> None:
        if metric_type not in _VALID_METRIC_TYPES:
            raise ValueError(
                f"metric_type must be one of {_VALID_METRIC_TYPES}, got {metric_type!r}"
            )

        self.control = np.asarray(control, dtype=float)
        self.treatment = np.asarray(treatment, dtype=float)

        if self.control.ndim != 1 or self.treatment.ndim != 1:
            raise ValueError("control and treatment must be 1-dimensional")
        if self.control.size == 0 or self.treatment.size == 0:
            raise ValueError("control and treatment must be non-empty")

        self.metric_type = metric_type
        self.n_control = int(self.control.size)
        self.n_treatment = int(self.treatment.size)

    # ------------------------------------------------------------------ summaries

    def summary(self) -> dict:
        """Return group-level descriptive statistics as a dictionary.

        Returns
        -------
        dict
            Means, standard deviations, sample sizes, the absolute effect
            (treatment minus control mean), and the relative lift.
        """
        control_mean = float(self.control.mean())
        treatment_mean = float(self.treatment.mean())
        return {
            "metric_type": self.metric_type,
            "n_control": self.n_control,
            "n_treatment": self.n_treatment,
            "control_mean": control_mean,
            "treatment_mean": treatment_mean,
            "control_std": float(self.control.std(ddof=1)) if self.n_control > 1 else 0.0,
            "treatment_std": float(self.treatment.std(ddof=1))
            if self.n_treatment > 1
            else 0.0,
            "absolute_effect": treatment_mean - control_mean,
            "relative_lift": self.relative_lift(),
        }

    def relative_lift(self) -> float:
        """Return the relative lift ``(treatment_mean - control_mean) / control_mean``.

        Returns
        -------
        float
            The fractional change in the mean from control to treatment. ``nan`` if
            the control mean is zero (relative lift is undefined).
        """
        control_mean = self.control.mean()
        if control_mean == 0:
            return float("nan")
        return float((self.treatment.mean() - control_mean) / control_mean)

    # --------------------------------------------------------------------- tests

    def ttest(self, alpha: float = 0.05, alternative: str = "two-sided") -> ExperimentResult:
        """Welch's two-sample t-test (unequal variances).

        Welch's t-test — not Student's — is the correct default for A/B tests: the two
        arms routinely have different variances, and Welch's is robust to that while
        costing almost nothing when variances happen to be equal.

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level.
        alternative : {"two-sided", "less", "greater"}, default "two-sided"
            The alternative hypothesis about ``treatment - control``.

        Returns
        -------
        ExperimentResult
            Test statistic, p-value, confidence interval on the mean difference,
            Cohen's *d* effect size, and significance decision.

        Raises
        ------
        ValueError
            If either group has fewer than two observations, or ``alternative`` is
            invalid.
        """
        self._check_alternative(alternative)
        if self.n_control < 2 or self.n_treatment < 2:
            raise ValueError("Welch's t-test requires at least two observations per group")

        res = stats.ttest_ind(
            self.treatment, self.control, equal_var=False, alternative=alternative
        )
        ci = res.confidence_interval(confidence_level=1.0 - alpha)
        try:
            effect_size = cohens_d(self.control, self.treatment)
        except ValueError:
            effect_size = float("nan")

        p_value = float(res.pvalue)
        return ExperimentResult(
            test_name="Welch's t-test",
            statistic=float(res.statistic),
            p_value=p_value,
            confidence_interval=(float(ci.low), float(ci.high)),
            effect_size=effect_size,
            relative_lift=self.relative_lift(),
            is_significant=bool(p_value < alpha),
            alpha=float(alpha),
            n_control=self.n_control,
            n_treatment=self.n_treatment,
        )

    def mann_whitney(
        self,
        alpha: float = 0.05,
        alternative: str = "two-sided",
        n_bootstrap: int = 5000,
        random_state: int | None = 0,
    ) -> ExperimentResult:
        """Mann-Whitney U test — a non-parametric rank-based test.

        Use this when the metric is ordinal, heavily skewed, or you are unwilling to
        assume approximate normality of the sample means. It tests whether one group
        tends to produce larger values than the other (stochastic dominance).

        The effect size is the *rank-biserial correlation*; the confidence interval is
        a bootstrap interval on the difference in medians (treatment minus control).

        Parameters
        ----------
        alpha : float, default 0.05
            Significance level.
        alternative : {"two-sided", "less", "greater"}, default "two-sided"
            The alternative hypothesis.
        n_bootstrap : int, default 5000
            Number of bootstrap resamples for the median-difference interval.
        random_state : int or None, default 0
            Seed for the bootstrap (fixed by default for reproducibility).

        Returns
        -------
        ExperimentResult
        """
        self._check_alternative(alternative)
        res = stats.mannwhitneyu(self.treatment, self.control, alternative=alternative)
        u_statistic = float(res.statistic)
        p_value = float(res.pvalue)

        # Rank-biserial correlation: 0 = no effect, +1 = treatment dominates.
        # scipy returns U for the first argument (treatment), which is large when
        # treatment tends to exceed control, so r = 2U/(n_t * n_c) - 1.
        rank_biserial = (2.0 * u_statistic) / (self.n_treatment * self.n_control) - 1.0

        ci_low, ci_high = self._bootstrap_statistic_ci(
            statistic=lambda c, t: np.median(t) - np.median(c),
            n_bootstrap=n_bootstrap,
            ci=1.0 - alpha,
            random_state=random_state,
        )

        return ExperimentResult(
            test_name="Mann-Whitney U",
            statistic=u_statistic,
            p_value=p_value,
            confidence_interval=(ci_low, ci_high),
            effect_size=float(rank_biserial),
            relative_lift=self.relative_lift(),
            is_significant=bool(p_value < alpha),
            alpha=float(alpha),
            n_control=self.n_control,
            n_treatment=self.n_treatment,
        )

    def chi_squared(self, alpha: float = 0.05) -> ExperimentResult:
        """Chi-squared test of independence for a binary / conversion metric.

        Builds the 2x2 table of (converted, not-converted) by arm and tests whether
        the conversion rate depends on the arm. The effect size is Cohen's *h* and the
        confidence interval is the Wald interval on the difference in proportions
        (treatment minus control).

        Returns
        -------
        ExperimentResult

        Raises
        ------
        ValueError
            If the data are not binary (values other than 0 and 1).
        """
        self._require_binary()

        c_success = int(self.control.sum())
        t_success = int(self.treatment.sum())
        table = np.array(
            [
                [c_success, self.n_control - c_success],
                [t_success, self.n_treatment - t_success],
            ],
            dtype=float,
        )

        chi2, p_value, _, _ = stats.chi2_contingency(table, correction=False)

        p_c = c_success / self.n_control
        p_t = t_success / self.n_treatment

        # Cohen's h for two proportions.
        effect_size = 2.0 * np.arcsin(np.sqrt(p_t)) - 2.0 * np.arcsin(np.sqrt(p_c))

        # Wald CI on the difference in proportions.
        diff = p_t - p_c
        se = np.sqrt(p_c * (1 - p_c) / self.n_control + p_t * (1 - p_t) / self.n_treatment)
        z = stats.norm.ppf(1.0 - alpha / 2.0)
        ci = (float(diff - z * se), float(diff + z * se))

        return ExperimentResult(
            test_name="Chi-squared test",
            statistic=float(chi2),
            p_value=float(p_value),
            confidence_interval=ci,
            effect_size=float(effect_size),
            relative_lift=self.relative_lift(),
            is_significant=bool(p_value < alpha),
            alpha=float(alpha),
            n_control=self.n_control,
            n_treatment=self.n_treatment,
        )

    def bootstrap_ci(
        self,
        n_bootstrap: int = 10_000,
        ci: float = 0.95,
        random_state: int | None = None,
    ) -> tuple[float, float]:
        """Bootstrap percentile confidence interval on the difference in means.

        Resamples each arm with replacement ``n_bootstrap`` times and returns the
        percentile interval of ``mean(treatment) - mean(control)``. This makes no
        distributional assumption and is a good cross-check on the t-test interval,
        especially for skewed metrics.

        Parameters
        ----------
        n_bootstrap : int, default 10000
            Number of bootstrap resamples.
        ci : float, default 0.95
            Confidence level (e.g. ``0.95`` for a 95% interval).
        random_state : int or None, default None
            Seed for reproducibility.

        Returns
        -------
        tuple[float, float]
            The ``(low, high)`` bounds of the confidence interval.
        """
        return self._bootstrap_statistic_ci(
            statistic=lambda c, t: t.mean() - c.mean(),
            n_bootstrap=n_bootstrap,
            ci=ci,
            random_state=random_state,
        )

    # -------------------------------------------------------------------- export

    def to_dataframe(self) -> pd.DataFrame:
        """Return the canonical test result as a tidy one-row dataframe.

        The canonical test is the proportion test for ``binary`` metrics and Welch's
        t-test otherwise. Useful for stacking many experiments into a single table.

        Returns
        -------
        pd.DataFrame
            A single-row dataframe with the :class:`ExperimentResult` fields.
        """
        result = self._default_result()
        return pd.DataFrame([asdict(result)])

    # ------------------------------------------------------------------ internals

    def _default_result(self) -> ExperimentResult:
        if self.metric_type == "binary":
            return self.chi_squared()
        return self.ttest()

    def _bootstrap_statistic_ci(
        self,
        statistic,
        n_bootstrap: int,
        ci: float,
        random_state: int | None,
    ) -> tuple[float, float]:
        if not 0.0 < ci < 1.0:
            raise ValueError("ci must be strictly between 0 and 1")
        rng = np.random.default_rng(random_state)
        estimates = np.empty(n_bootstrap, dtype=float)
        for i in range(n_bootstrap):
            c_sample = rng.choice(self.control, size=self.n_control, replace=True)
            t_sample = rng.choice(self.treatment, size=self.n_treatment, replace=True)
            estimates[i] = statistic(c_sample, t_sample)
        tail = (1.0 - ci) / 2.0
        low, high = np.quantile(estimates, [tail, 1.0 - tail])
        return float(low), float(high)

    def _require_binary(self) -> None:
        combined = np.concatenate([self.control, self.treatment])
        if not np.all(np.isin(combined, (0.0, 1.0))):
            raise ValueError(
                "chi_squared requires binary (0/1) data; got non-binary values"
            )

    @staticmethod
    def _check_alternative(alternative: str) -> None:
        if alternative not in _VALID_ALTERNATIVES:
            raise ValueError(
                f"alternative must be one of {_VALID_ALTERNATIVES}, got {alternative!r}"
            )
