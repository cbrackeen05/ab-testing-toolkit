"""Multiple-comparison corrections for running many tests at once.

When you test many hypotheses simultaneously — many metrics, many variants, many
segments — the chance that *at least one* clears ``p < alpha`` purely by luck grows
quickly. With 20 independent true-null tests at ``alpha = 0.05`` you expect one false
positive on average. These procedures adjust for that, and they differ in *what* error
rate they control and *how aggressively*:

- :func:`bonferroni` controls the **family-wise error rate** (FWER, the probability of
  *any* false positive) and is the most conservative. Best when even a single false
  positive is costly and the number of tests is small.
- :func:`holm_bonferroni` also controls the FWER but via a step-down procedure that is
  uniformly **more powerful** than Bonferroni — it should essentially always be
  preferred over plain Bonferroni when FWER control is the goal.
- :func:`benjamini_hochberg` controls the **false discovery rate** (FDR, the expected
  *proportion* of false positives among rejections). It is far less conservative and is
  the right choice when you are screening many experiments and can tolerate a known
  fraction of false discoveries.

Each function returns a :class:`CorrectionResult` with adjusted p-values (directly
comparable to the original ``alpha``) and the reject/accept decision per hypothesis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CorrectionResult:
    """Outcome of a multiple-comparison correction.

    Attributes
    ----------
    method : str
        Name of the correction procedure.
    original_p_values : list[float]
        The input p-values, in their original order.
    adjusted_p_values : list[float]
        Adjusted p-values in the same order; compare these directly against ``alpha``.
    rejected : list[bool]
        ``True`` where the (adjusted) hypothesis is rejected.
    n_rejected : int
        Total number of rejected hypotheses.
    alpha : float
        Significance level used.
    """

    method: str
    original_p_values: list[float]
    adjusted_p_values: list[float]
    rejected: list[bool]
    n_rejected: int
    alpha: float


def _validate(p_values: list[float], alpha: float) -> np.ndarray:
    arr = np.asarray(p_values, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("p_values must be a non-empty 1-D sequence")
    if np.any((arr < 0.0) | (arr > 1.0)) or np.any(np.isnan(arr)):
        raise ValueError("p_values must all lie in [0, 1]")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return arr


def _build(method: str, original: np.ndarray, adjusted: np.ndarray, alpha: float) -> CorrectionResult:
    rejected = adjusted <= alpha
    return CorrectionResult(
        method=method,
        original_p_values=[float(x) for x in original],
        adjusted_p_values=[float(x) for x in adjusted],
        rejected=[bool(x) for x in rejected],
        n_rejected=int(rejected.sum()),
        alpha=float(alpha),
    )


def bonferroni(p_values: list[float], alpha: float = 0.05) -> CorrectionResult:
    """Bonferroni correction — controls the family-wise error rate (FWER).

    Multiplies every p-value by the number of tests ``m`` (equivalently, tests each
    against ``alpha / m``). This guarantees the probability of *any* false positive is
    at most ``alpha``, regardless of dependence between tests. It is simple and very
    conservative: with many tests it sacrifices a lot of power.

    Use it when the number of comparisons is small and any single false positive is
    unacceptable. Prefer :func:`holm_bonferroni` if you want FWER control with more
    power.

    Parameters
    ----------
    p_values : list[float]
        Raw p-values from the individual tests.
    alpha : float, default 0.05
        Family-wise significance level.

    Returns
    -------
    CorrectionResult

    Examples
    --------
    >>> bonferroni([0.01, 0.02, 0.03, 0.04]).n_rejected
    1
    """
    arr = _validate(p_values, alpha)
    m = arr.size
    adjusted = np.clip(arr * m, 0.0, 1.0)
    return _build("bonferroni", arr, adjusted, alpha)


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> CorrectionResult:
    """Holm-Bonferroni correction — step-down FWER control, more powerful than Bonferroni.

    Sorts the p-values ascending and tests them sequentially against an increasingly
    lenient threshold (``alpha / m``, ``alpha / (m-1)``, ...), stopping at the first
    failure. It controls the FWER under any dependence structure, yet rejects at least
    as many hypotheses as Bonferroni — so it dominates plain Bonferroni and is the
    recommended default for FWER control.

    Parameters
    ----------
    p_values : list[float]
        Raw p-values.
    alpha : float, default 0.05
        Family-wise significance level.

    Returns
    -------
    CorrectionResult

    Examples
    --------
    >>> holm_bonferroni([0.01, 0.04, 0.03, 0.005]).n_rejected
    2
    """
    arr = _validate(p_values, alpha)
    m = arr.size
    order = np.argsort(arr)
    ranked = arr[order]

    # Step-down multipliers m, m-1, ..., 1, with enforced monotone non-decreasing
    # adjusted values so the reject set is a prefix of the sorted p-values.
    factors = m - np.arange(m)
    adj_sorted = np.clip(np.maximum.accumulate(ranked * factors), 0.0, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adj_sorted
    return _build("holm-bonferroni", arr, adjusted, alpha)


def benjamini_hochberg(p_values: list[float], alpha: float = 0.05) -> CorrectionResult:
    """Benjamini-Hochberg correction — controls the false discovery rate (FDR).

    Instead of guarding against *any* false positive, BH controls the expected
    *proportion* of false positives among the rejected hypotheses. This is much less
    conservative than FWER methods and is the standard choice when screening large
    numbers of experiments, where a small, known fraction of false discoveries is an
    acceptable price for greatly increased power.

    The returned adjusted p-values are BH "q-values": the smallest FDR at which each
    hypothesis would be rejected.

    Parameters
    ----------
    p_values : list[float]
        Raw p-values.
    alpha : float, default 0.05
        Target false discovery rate.

    Returns
    -------
    CorrectionResult

    References
    ----------
    Benjamini, Y. and Hochberg, Y. (1995). "Controlling the False Discovery Rate."
    *Journal of the Royal Statistical Society, Series B*, 57(1), 289-300.

    Examples
    --------
    >>> benjamini_hochberg([0.005, 0.015, 0.025, 0.039, 0.5]).n_rejected
    4
    """
    arr = _validate(p_values, alpha)
    m = arr.size
    order = np.argsort(arr)
    ranked = arr[order]

    ranks = np.arange(1, m + 1)
    raw = ranked * m / ranks
    # Enforce monotonicity by taking the running minimum from the largest p downward.
    adj_sorted = np.clip(np.minimum.accumulate(raw[::-1])[::-1], 0.0, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adj_sorted
    return _build("benjamini-hochberg", arr, adjusted, alpha)
