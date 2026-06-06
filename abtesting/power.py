"""Sample-size and statistical power calculations for experiment planning.

Power analysis answers three linked questions:

- *Before* an experiment: "How many users per arm do I need to reliably detect an
  effect of a given size?" (:func:`minimum_sample_size`)
- *Before* an experiment: "Given the traffic I can allocate, what is the smallest
  effect I could detect?" (:func:`minimum_detectable_effect`)
- *After* an experiment: "Was my non-significant result actually informative, or was
  the test simply underpowered?" (:func:`observed_power`)

Plus a planning convenience, :func:`experiment_runtime_days`, that converts a required
sample size into a calendar duration given daily traffic.

All formulas use the normal approximation to the two-sample test, which is standard for
A/B test planning and accurate for the sample sizes experiments actually run at.

Conventions
-----------
- ``metric_type="binary"``: ``baseline_rate`` is the control conversion rate and the MDE
  is an **absolute** change in rate (e.g. ``0.02`` = a 2-percentage-point lift).
- ``metric_type="continuous"`` (or ``"ratio"``): the MDE is a **standardized** effect
  size (Cohen's *d*, i.e. the mean difference in units of standard deviation), and
  ``baseline_rate`` is ignored.
"""

from __future__ import annotations

import math

from scipy import stats

_BINARY = "binary"
_CONTINUOUS = ("continuous", "ratio")


def _critical_values(alpha: float, power: float) -> tuple[float, float]:
    """Return ``(z_alpha, z_beta)`` for a two-sided test at the given alpha/power."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < power < 1.0:
        raise ValueError("power must be in (0, 1)")
    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_beta = stats.norm.ppf(power)
    return float(z_alpha), float(z_beta)


def minimum_sample_size(
    baseline_rate: float,
    minimum_detectable_effect: float,
    alpha: float = 0.05,
    power: float = 0.80,
    metric_type: str = "binary",
) -> int:
    """Minimum sample size **per group** to detect the MDE at the given alpha and power.

    Parameters
    ----------
    baseline_rate : float
        Control conversion rate for binary metrics (in ``(0, 1)``); ignored for
        continuous metrics.
    minimum_detectable_effect : float
        Smallest effect worth detecting. Absolute rate change for binary metrics;
        standardized effect size (Cohen's *d*) for continuous metrics. Must be nonzero.
    alpha : float, default 0.05
        Two-sided significance level.
    power : float, default 0.80
        Desired power (``1 - beta``).
    metric_type : {"binary", "continuous", "ratio"}, default "binary"
        See module docstring for how the MDE is interpreted.

    Returns
    -------
    int
        Required sample size per group (rounded up).

    Examples
    --------
    >>> minimum_sample_size(0.10, 0.02)  # 10% baseline, +2pp lift
    3839
    """
    z_alpha, z_beta = _critical_values(alpha, power)
    mde = float(minimum_detectable_effect)
    if mde == 0:
        raise ValueError("minimum_detectable_effect must be nonzero")

    if metric_type == _BINARY:
        p1 = float(baseline_rate)
        p2 = p1 + mde
        if not 0.0 < p1 < 1.0:
            raise ValueError("baseline_rate must be in (0, 1) for binary metrics")
        if not 0.0 < p2 < 1.0:
            raise ValueError("baseline_rate + MDE must stay in (0, 1)")
        numerator = (z_alpha + z_beta) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2))
        n = numerator / mde**2
    elif metric_type in _CONTINUOUS:
        # MDE is a standardized effect size; both arms share unit variance.
        n = 2.0 * (z_alpha + z_beta) ** 2 / mde**2
    else:
        raise ValueError(f"unknown metric_type {metric_type!r}")

    return int(math.ceil(n))


def minimum_detectable_effect(
    n: int,
    baseline_rate: float,
    alpha: float = 0.05,
    power: float = 0.80,
    metric_type: str = "binary",
) -> float:
    """Smallest effect detectable with ``n`` samples per group at the given alpha/power.

    The inverse of :func:`minimum_sample_size`. For binary metrics this uses the
    baseline-variance approximation (``Var ≈ p(1 - p)`` for both arms), which is the
    conventional planning estimate.

    Parameters
    ----------
    n : int
        Sample size per group.
    baseline_rate : float
        Control rate for binary metrics; ignored for continuous metrics.
    alpha : float, default 0.05
        Two-sided significance level.
    power : float, default 0.80
        Desired power.
    metric_type : {"binary", "continuous", "ratio"}, default "binary"

    Returns
    -------
    float
        The MDE: an absolute rate change for binary metrics, or a standardized effect
        size for continuous metrics.

    Examples
    --------
    >>> mde = minimum_detectable_effect(3839, 0.10)
    >>> round(mde, 3)
    0.019
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    z_alpha, z_beta = _critical_values(alpha, power)

    if metric_type == _BINARY:
        p1 = float(baseline_rate)
        if not 0.0 < p1 < 1.0:
            raise ValueError("baseline_rate must be in (0, 1) for binary metrics")
        return float((z_alpha + z_beta) * math.sqrt(2.0 * p1 * (1 - p1) / n))
    if metric_type in _CONTINUOUS:
        return float((z_alpha + z_beta) * math.sqrt(2.0 / n))
    raise ValueError(f"unknown metric_type {metric_type!r}")


def observed_power(
    n: int,
    effect_size: float,
    baseline_std: float,
    alpha: float = 0.05,
) -> float:
    """Post-hoc power of a completed two-sample experiment.

    Given the observed absolute mean difference and the metric's standard deviation,
    returns the probability that a test at level ``alpha`` would reject the null. Use
    this to diagnose a non-significant result: low observed power means the experiment
    was too small to detect the effect it saw, so "not significant" is uninformative
    rather than evidence of no effect.

    Parameters
    ----------
    n : int
        Sample size per group.
    effect_size : float
        Observed **absolute** difference in means between the arms.
    baseline_std : float
        Standard deviation of the metric (assumed common to both arms).
    alpha : float, default 0.05
        Two-sided significance level.

    Returns
    -------
    float
        Statistical power in ``[0, 1]``.

    Examples
    --------
    >>> 0.0 <= observed_power(1000, 0.2, 1.0) <= 1.0
    True
    """
    if n < 2:
        raise ValueError("n must be at least 2")
    if baseline_std <= 0:
        raise ValueError("baseline_std must be positive")
    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)

    # Standardized effect and noncentrality for equal-n two-sample test.
    d = abs(effect_size) / baseline_std
    ncp = d * math.sqrt(n / 2.0)
    power = stats.norm.sf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp)
    return float(power)


def experiment_runtime_days(
    daily_traffic: int,
    baseline_rate: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.80,
    traffic_split: float = 0.5,
) -> int:
    """Number of days to reach sufficient power given daily traffic into the test.

    Computes the required per-group sample size, then divides by the number of users
    the *smaller* arm receives per day. With an uneven ``traffic_split`` the smaller
    arm fills more slowly and therefore sets the runtime.

    Parameters
    ----------
    daily_traffic : int
        Total users entering the experiment per day (across both arms).
    baseline_rate : float
        Control conversion rate (binary metric).
    mde : float
        Absolute minimum detectable effect.
    alpha : float, default 0.05
        Two-sided significance level.
    power : float, default 0.80
        Desired power.
    traffic_split : float, default 0.5
        Fraction of traffic allocated to the treatment arm.

    Returns
    -------
    int
        Number of days required (rounded up to a whole day).

    Examples
    --------
    >>> experiment_runtime_days(10_000, 0.10, 0.02)
    1
    """
    if daily_traffic <= 0:
        raise ValueError("daily_traffic must be positive")
    if not 0.0 < traffic_split < 1.0:
        raise ValueError("traffic_split must be in (0, 1)")

    n_per_group = minimum_sample_size(
        baseline_rate=baseline_rate,
        minimum_detectable_effect=mde,
        alpha=alpha,
        power=power,
        metric_type=_BINARY,
    )
    smaller_arm_fraction = min(traffic_split, 1.0 - traffic_split)
    per_day_into_smaller_arm = daily_traffic * smaller_arm_fraction
    return int(math.ceil(n_per_group / per_day_into_smaller_arm))
