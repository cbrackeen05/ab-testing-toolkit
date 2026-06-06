"""Internal helpers shared across modules.

These are not part of the public experiment-analysis API but are used across the
package (and are useful on their own for data cleaning and pre-experiment checks).

Functions
---------
check_sample_ratio_mismatch
    Chi-squared test that the realized control/treatment split matches the intended
    split. Run this *before* looking at results — a significant mismatch usually
    signals a randomization or logging bug, not a real effect.
winsorize
    Clip extreme values at given percentiles. Useful for heavy-tailed metrics such
    as watch time or revenue, where a few outliers dominate the variance.
log_transform
    ``log1p`` transform for right-skewed continuous metrics.
cohens_d
    Standardized (pooled-SD) effect size for two independent samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class SRMResult:
    """Result of a sample-ratio-mismatch (SRM) check.

    Attributes
    ----------
    statistic : float
        Chi-squared test statistic.
    p_value : float
        P-value of the chi-squared goodness-of-fit test.
    n_control : int
        Observed number of units in control.
    n_treatment : int
        Observed number of units in treatment.
    expected_split : float
        Intended fraction of units in control (e.g. ``0.5`` for a 50/50 test).
    is_mismatch : bool
        ``True`` if the realized split differs from ``expected_split`` at the
        given ``alpha`` (i.e. ``p_value < alpha``). A ``True`` here should block
        analysis until the cause is found.
    alpha : float
        Significance level used for the decision.
    """

    statistic: float
    p_value: float
    n_control: int
    n_treatment: int
    expected_split: float
    is_mismatch: bool
    alpha: float


def check_sample_ratio_mismatch(
    n_control: int,
    n_treatment: int,
    expected_split: float = 0.5,
    alpha: float = 0.05,
) -> SRMResult:
    """Test whether the realized assignment split matches the intended split.

    A *sample ratio mismatch* (SRM) is a discrepancy between the ratio of units
    you intended to assign to each arm and the ratio you actually observed. Even a
    small but statistically significant mismatch invalidates the experiment,
    because it implies the randomization or logging pipeline is biased — and that
    bias is very likely correlated with the metric you care about.

    This runs a chi-squared goodness-of-fit test of the observed counts against the
    expected counts implied by ``expected_split``.

    Parameters
    ----------
    n_control : int
        Number of units assigned to control.
    n_treatment : int
        Number of units assigned to treatment.
    expected_split : float, default 0.5
        Intended fraction of *total* units assigned to control. ``0.5`` is a 50/50
        test; ``0.9`` would be a 90% control / 10% treatment holdout.
    alpha : float, default 0.05
        Significance level for declaring a mismatch.

    Returns
    -------
    SRMResult
        Test statistic, p-value, and an ``is_mismatch`` flag.

    Examples
    --------
    >>> res = check_sample_ratio_mismatch(10000, 10000)
    >>> res.is_mismatch
    False
    >>> bad = check_sample_ratio_mismatch(10000, 8500)
    >>> bad.is_mismatch
    True
    """
    if not 0.0 < expected_split < 1.0:
        raise ValueError("expected_split must be strictly between 0 and 1")
    if n_control < 0 or n_treatment < 0:
        raise ValueError("counts must be non-negative")

    total = n_control + n_treatment
    if total == 0:
        raise ValueError("at least one unit is required")

    expected_control = total * expected_split
    expected_treatment = total * (1.0 - expected_split)

    observed = np.array([n_control, n_treatment], dtype=float)
    expected = np.array([expected_control, expected_treatment], dtype=float)

    statistic, p_value = stats.chisquare(f_obs=observed, f_exp=expected)

    return SRMResult(
        statistic=float(statistic),
        p_value=float(p_value),
        n_control=int(n_control),
        n_treatment=int(n_treatment),
        expected_split=float(expected_split),
        is_mismatch=bool(p_value < alpha),
        alpha=float(alpha),
    )


def winsorize(data: np.ndarray, lower: float = 0.01, upper: float = 0.99) -> np.ndarray:
    """Clip values to the ``[lower, upper]`` percentile range.

    Winsorizing replaces extreme values with the nearest percentile boundary rather
    than dropping them, preserving sample size while taming the influence of
    outliers on means and variances. This is appropriate for heavy-tailed metrics
    (watch time, revenue, session length) where a handful of extreme users would
    otherwise dominate the estimate.

    Parameters
    ----------
    data : array-like
        Input values.
    lower : float, default 0.01
        Lower percentile (as a fraction in ``[0, 1]``) at which to clip.
    upper : float, default 0.99
        Upper percentile (as a fraction in ``[0, 1]``) at which to clip.

    Returns
    -------
    np.ndarray
        A new array with values clipped to the ``[lower, upper]`` percentile range.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([0, 1, 2, 3, 100])
    >>> winsorize(x, 0.0, 0.8).max()  # doctest: +SKIP
    3.0
    """
    if not 0.0 <= lower < upper <= 1.0:
        raise ValueError("require 0 <= lower < upper <= 1")

    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return arr.copy()

    lo, hi = np.quantile(arr, [lower, upper])
    return np.clip(arr, lo, hi)


def log_transform(data: np.ndarray) -> np.ndarray:
    """Apply a ``log1p`` (``log(1 + x)``) transform.

    Right-skewed metrics (counts, revenue, durations) are often better behaved on a
    log scale, where multiplicative effects become additive and the distribution is
    closer to normal — improving the validity of t-tests. ``log1p`` is used instead
    of ``log`` so that exact zeros are handled (``log1p(0) == 0``).

    Parameters
    ----------
    data : array-like
        Non-negative input values.

    Returns
    -------
    np.ndarray
        ``log(1 + data)``.

    Raises
    ------
    ValueError
        If any value is less than or equal to ``-1`` (where ``log1p`` is undefined).

    Examples
    --------
    >>> import numpy as np
    >>> log_transform(np.array([0.0, np.e - 1]))
    array([0., 1.])
    """
    arr = np.asarray(data, dtype=float)
    if arr.size and np.min(arr) <= -1.0:
        raise ValueError("log1p is undefined for values <= -1")
    return np.log1p(arr)


def cohens_d(control: np.ndarray, treatment: np.ndarray) -> float:
    """Cohen's *d*: the standardized difference in means of two samples.

    Cohen's *d* expresses the treatment effect in units of pooled standard
    deviation, making it comparable across metrics on different scales. By
    convention |d| ≈ 0.2 is small, 0.5 medium, and 0.8 large.

    Uses the pooled standard deviation with the unbiased (``ddof=1``) sample
    variance of each group. The sign follows ``treatment - control``.

    Parameters
    ----------
    control : array-like
        Control-group values.
    treatment : array-like
        Treatment-group values.

    Returns
    -------
    float
        The standardized effect size ``(mean_treatment - mean_control) / pooled_sd``.

    Raises
    ------
    ValueError
        If either group has fewer than two observations.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> c = rng.normal(0, 1, 1000)
    >>> t = rng.normal(0.5, 1, 1000)
    >>> round(cohens_d(c, t), 1)
    0.5
    """
    c = np.asarray(control, dtype=float)
    t = np.asarray(treatment, dtype=float)
    n_c, n_t = c.size, t.size
    if n_c < 2 or n_t < 2:
        raise ValueError("each group needs at least two observations")

    var_c = c.var(ddof=1)
    var_t = t.var(ddof=1)
    pooled_sd = np.sqrt(((n_c - 1) * var_c + (n_t - 1) * var_t) / (n_c + n_t - 2))
    if pooled_sd == 0:
        return 0.0
    return float((t.mean() - c.mean()) / pooled_sd)
