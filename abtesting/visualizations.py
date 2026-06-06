"""Plotting helpers for experiment analysis (matplotlib / seaborn).

Every function returns a :class:`matplotlib.figure.Figure` and never calls
``plt.show()`` — rendering is left to the caller, so the same code works in notebooks,
scripts, and tests. A shared :func:`_set_style` keeps a consistent, minimal look.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .power import minimum_sample_size

if TYPE_CHECKING:  # avoid importing heavy types at runtime / circular imports
    from .experiment import Experiment, ExperimentResult
    from .sequential import SPRTResult

_CONTROL_COLOR = "#4C72B0"
_TREATMENT_COLOR = "#DD8452"


def _set_style() -> None:
    """Apply a consistent, minimal seaborn/matplotlib style."""
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False


def plot_distributions(experiment: "Experiment", title: str | None = None) -> plt.Figure:
    """Overlaid distribution plots (histogram + KDE) for control and treatment.

    Marks each group's mean with a vertical dashed line in the matching colour.

    Parameters
    ----------
    experiment : Experiment
        The experiment whose two arms to plot.
    title : str, optional
        Figure title. A sensible default is used if omitted.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    for data, label, color in (
        (experiment.control, "Control", _CONTROL_COLOR),
        (experiment.treatment, "Treatment", _TREATMENT_COLOR),
    ):
        sns.histplot(
            data, kde=True, stat="density", element="step", alpha=0.35,
            color=color, label=label, ax=ax,
        )
        ax.axvline(float(np.mean(data)), color=color, linestyle="--", linewidth=1.5)

    ax.set_xlabel("Metric value")
    ax.set_ylabel("Density")
    ax.set_title(title or "Control vs. Treatment distributions")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_confidence_interval(result: "ExperimentResult", title: str | None = None) -> plt.Figure:
    """Forest-plot style view of the effect estimate and its confidence interval.

    Draws the confidence interval as a horizontal bar with the point estimate (the CI
    midpoint) marked, plus a vertical reference line at zero. If the interval does not
    cross zero, the effect is significant at the result's ``alpha``.

    Parameters
    ----------
    result : ExperimentResult
        A result from one of the :class:`Experiment` tests.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _set_style()
    fig, ax = plt.subplots(figsize=(8, 2.6))

    low, high = result.confidence_interval
    finite = np.isfinite([low, high])
    point = (low + high) / 2.0 if finite.all() else result.effect_size

    crosses_zero = low <= 0.0 <= high
    color = "#C44E52" if crosses_zero else "#55A868"

    ax.axvline(0.0, color="grey", linestyle="--", linewidth=1.2, zorder=1)
    if finite.all():
        ax.plot([low, high], [0, 0], color=color, linewidth=3, zorder=2)
        ax.plot([low, high], [0, 0], "|", color=color, markersize=14, zorder=3)
    ax.plot(point, 0, "o", color=color, markersize=10, zorder=4)

    ax.set_yticks([])
    ax.set_xlabel("Effect (treatment - control)")
    conf = int(round((1.0 - result.alpha) * 100))
    ax.set_title(
        title or f"{result.test_name}: effect with {conf}% CI "
        f"({'significant' if not crosses_zero else 'not significant'})"
    )
    fig.tight_layout()
    return fig


def plot_power_curve(
    baseline_rate: float,
    mde_range: Sequence[float],
    alpha: float = 0.05,
    power_target: float = 0.80,
) -> plt.Figure:
    """Plot required sample size per group against the minimum detectable effect.

    A planning aid: it makes the steep cost of detecting small effects visible.

    Parameters
    ----------
    baseline_rate : float
        Control conversion rate (binary metric).
    mde_range : sequence of float
        Absolute minimum detectable effects to evaluate.
    alpha : float, default 0.05
        Significance level.
    power_target : float, default 0.80
        Desired power.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    mdes = list(mde_range)
    sizes = [
        minimum_sample_size(baseline_rate, mde, alpha=alpha, power=power_target)
        for mde in mdes
    ]
    ax.plot(mdes, sizes, marker="o", color=_CONTROL_COLOR, label="Required n per group")

    ax.set_xlabel("Minimum detectable effect (absolute)")
    ax.set_ylabel("Sample size per group")
    ax.set_title(
        f"Power curve (baseline={baseline_rate:.0%}, "
        f"power={power_target:.0%}, α={alpha})"
    )
    ax.legend()
    fig.tight_layout()
    return fig


def plot_pvalue_distribution(p_values: Sequence[float], title: str | None = None) -> plt.Figure:
    """Histogram of p-values, with the uniform null expectation marked.

    Under the null hypothesis, p-values are uniform on ``[0, 1]``. Running many A/A
    tests and plotting their p-values is a core platform-trustworthiness check: a
    non-uniform distribution signals a bug in randomization, logging, or the test
    itself.

    Parameters
    ----------
    p_values : sequence of float
        The p-values to plot.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    _set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    values = np.asarray(p_values, dtype=float)
    n_bins = 20
    ax.hist(values, bins=n_bins, range=(0, 1), color=_CONTROL_COLOR, alpha=0.8,
            edgecolor="white")

    expected = len(values) / n_bins
    ax.axhline(expected, color="#C44E52", linestyle="--", linewidth=1.5,
               label="Uniform (null) expectation")

    ax.set_xlabel("p-value")
    ax.set_ylabel("Count")
    ax.set_title(title or "p-value distribution (should be uniform under the null)")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_sequential_boundaries(
    sprt_results: Sequence["SPRTResult"],
    observations: Sequence[int],
) -> plt.Figure:
    """Plot the SPRT log-likelihood ratio over time against its decision boundaries.

    Shows visually when the accumulating evidence would have crossed a boundary and
    triggered an early stop.

    Parameters
    ----------
    sprt_results : sequence of SPRTResult
        SPRT results at successive sample sizes (boundaries are taken from the first).
    observations : sequence of int
        The sample size corresponding to each result; same length as ``sprt_results``.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If the two sequences differ in length or are empty.
    """
    if len(sprt_results) != len(observations):
        raise ValueError("sprt_results and observations must have the same length")
    if len(sprt_results) == 0:
        raise ValueError("need at least one SPRT result to plot")

    _set_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    llrs = [r.llr for r in sprt_results]
    upper = sprt_results[0].upper_boundary
    lower = sprt_results[0].lower_boundary

    ax.plot(observations, llrs, marker="o", color=_CONTROL_COLOR, label="Log-likelihood ratio")
    ax.axhline(upper, color="#55A868", linestyle="--", linewidth=1.5, label="Reject null")
    ax.axhline(lower, color="#C44E52", linestyle="--", linewidth=1.5, label="Accept null")
    ax.axhspan(lower, upper, color="grey", alpha=0.08)

    # Mark the first boundary crossing, if any.
    for obs, llr in zip(observations, llrs):
        if llr >= upper or llr <= lower:
            ax.axvline(obs, color="black", linestyle=":", linewidth=1.2)
            ax.annotate("early stop", xy=(obs, llr), xytext=(5, 5),
                        textcoords="offset points")
            break

    ax.set_xlabel("Observations")
    ax.set_ylabel("Log-likelihood ratio")
    ax.set_title("SPRT decision boundaries over time")
    ax.legend()
    fig.tight_layout()
    return fig
