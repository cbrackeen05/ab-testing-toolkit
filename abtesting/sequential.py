"""Sequential testing for valid early stopping.

Why standard p-values are invalid under peeking
------------------------------------------------
A fixed-sample test controls the false-positive rate at ``alpha`` only if you look at
the data **once**, at a pre-committed sample size. If instead you monitor an experiment
and stop the moment ``p < 0.05``, you give yourself many chances to cross the threshold
by luck. Each peek is another draw, so the probability that *at least one* peek shows a
spurious "significant" result is far larger than ``alpha`` — with continuous monitoring
of a true null it approaches 1. This is the single most common way A/B platforms ship
false wins.

How sequential testing fixes it
-------------------------------
Sequential methods are designed to be looked at repeatedly while still controlling
error:

- :func:`sprt` — Wald's Sequential Probability Ratio Test. It accumulates the
  log-likelihood ratio between two hypotheses and stops as soon as that statistic
  crosses an upper boundary (reject the null) or a lower boundary (accept the null),
  with the boundaries chosen so that the type I and type II error rates are bounded by
  ``alpha`` and ``beta``. It is the theoretically optimal way to do early stopping for
  a simple-vs-simple hypothesis.
- :func:`always_valid_pvalue` — a p-value from the mixture SPRT (mSPRT) that is valid
  at *every* sample size simultaneously. You may peek as often as you like; the
  probability it ever drops below ``alpha`` under the null is still at most ``alpha``.

References
----------
- Wald, A. (1945). "Sequential Tests of Statistical Hypotheses." *Annals of
  Mathematical Statistics*, 16(2), 117-186.
- Johari, R., Pekelis, L., Walsh, D. (2017). "Always Valid Inference: Bringing
  Sequential Analysis to A/B Testing." (mSPRT)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


@dataclass
class SPRTResult:
    """Outcome of a Sequential Probability Ratio Test.

    Attributes
    ----------
    decision : str
        One of ``"reject_null"``, ``"accept_null"``, or ``"continue"``.
    llr : float
        The accumulated log-likelihood ratio (H1 vs H0).
    lower_boundary : float
        Cross at or below this -> accept the null and stop.
    upper_boundary : float
        Cross at or above this -> reject the null and stop.
    alpha : float
        Target type I error rate.
    beta : float
        Target type II error rate.
    """

    decision: str
    llr: float
    lower_boundary: float
    upper_boundary: float
    alpha: float
    beta: float


def _wald_boundaries(alpha: float, beta: float) -> tuple[float, float]:
    """Return Wald's ``(lower, upper)`` log-boundaries for the given error rates.

    Upper = ``log((1 - beta) / alpha)``; lower = ``log(beta / (1 - alpha))``.
    """
    if not 0.0 < alpha < 1.0 or not 0.0 < beta < 1.0:
        raise ValueError("alpha and beta must be in (0, 1)")
    upper = math.log((1.0 - beta) / alpha)
    lower = math.log(beta / (1.0 - alpha))
    return lower, upper


def sprt(
    control_successes: int,
    control_n: int,
    treatment_successes: int,
    treatment_n: int,
    alpha: float = 0.05,
    beta: float = 0.20,
    h0_rate: float | None = None,
    h1_rate: float | None = None,
) -> SPRTResult:
    """Wald's SPRT for a binary metric, testing the treatment arm against a baseline.

    Tests ``H0: treatment rate = h0_rate`` against ``H1: treatment rate = h1_rate``
    (a one-sided "is the variant better?" test). The accumulated log-likelihood ratio
    is compared against Wald's boundaries and a decision is returned: stop and reject
    the null, stop and accept it, or keep collecting data.

    For Wald's type I / type II error guarantees to hold exactly, **both hypotheses
    must be fixed in advance** — pass an explicit ``h1_rate`` (e.g. baseline plus your
    minimum detectable effect). If ``h1_rate`` is left as ``None`` the alternative is
    estimated from the data as a convenience; this is a common practical relaxation but
    mildly inflates the type I rate, so prefer a fixed ``h1_rate`` when error control
    matters.

    Parameters
    ----------
    control_successes, control_n : int
        Successes and sample size in the control arm (used for the default baseline).
    treatment_successes, treatment_n : int
        Successes and sample size in the treatment arm (the monitored arm).
    alpha : float, default 0.05
        Target type I error rate (controls the upper boundary).
    beta : float, default 0.20
        Target type II error rate (controls the lower boundary).
    h0_rate : float or None, default None
        Null conversion rate. If ``None``, defaults to the observed control rate.
    h1_rate : float or None, default None
        Alternative conversion rate. If ``None``, defaults to the observed treatment
        rate (see the note above about error control).

    Returns
    -------
    SPRTResult

    Examples
    --------
    >>> # A large, clear lift with plenty of data -> reject the null and stop.
    >>> sprt(1000, 10000, 1500, 10000, h0_rate=0.10, h1_rate=0.12).decision
    'reject_null'
    """
    if control_n <= 0 or treatment_n <= 0:
        raise ValueError("sample sizes must be positive")
    if not 0 <= control_successes <= control_n or not 0 <= treatment_successes <= treatment_n:
        raise ValueError("successes must be between 0 and the sample size")

    lower, upper = _wald_boundaries(alpha, beta)

    p0 = control_successes / control_n if h0_rate is None else float(h0_rate)
    p1 = treatment_successes / treatment_n if h1_rate is None else float(h1_rate)

    # Clamp both rates away from 0 and 1 so the log-likelihood ratio stays finite.
    p0 = min(max(p0, _EPS), 1.0 - _EPS)
    p1 = min(max(p1, _EPS), 1.0 - _EPS)

    llr = _binomial_llr(treatment_successes, treatment_n, p1, p0)

    if llr >= upper:
        decision = "reject_null"
    elif llr <= lower:
        decision = "accept_null"
    else:
        decision = "continue"

    return SPRTResult(
        decision=decision,
        llr=float(llr),
        lower_boundary=float(lower),
        upper_boundary=float(upper),
        alpha=float(alpha),
        beta=float(beta),
    )


def _binomial_llr(successes: int, n: int, p1: float, p0: float) -> float:
    """Log-likelihood ratio of ``n`` Bernoulli trials under rate ``p1`` vs ``p0``."""
    failures = n - successes
    return successes * math.log(p1 / p0) + failures * math.log((1.0 - p1) / (1.0 - p0))


def always_valid_pvalue(
    control: np.ndarray,
    treatment: np.ndarray,
    alpha: float = 0.05,
    tau_squared: float | None = None,
) -> float:
    """Always-valid p-value for the difference in means via the mixture SPRT (mSPRT).

    Unlike a fixed-sample p-value, this quantity is valid at *every* sample size at
    once: under the null, the probability that it *ever* falls below ``alpha`` — no
    matter how often you peek — is at most ``alpha``. That makes continuous monitoring
    and optional early stopping safe.

    It is computed from the mSPRT likelihood ratio for a normal mixture prior on the
    effect: ``p = min(1, 1 / Lambda)`` where ``Lambda`` mixes the alternative over
    ``N(0, tau_squared)``.

    Parameters
    ----------
    control, treatment : array-like
        Metric values for the two arms.
    alpha : float, default 0.05
        Significance level (only used to validate inputs / for the caller's threshold).
    tau_squared : float or None, default None
        Mixture variance hyperparameter. If ``None``, defaults to the pooled
        per-observation variance, a sensible scale for plausible effects.

    Returns
    -------
    float
        An always-valid p-value in ``(0, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> c = rng.normal(0.0, 1.0, 5000)
    >>> t = rng.normal(0.2, 1.0, 5000)
    >>> always_valid_pvalue(c, t) < 0.05   # clear effect -> small p
    True
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    c = np.asarray(control, dtype=float)
    t = np.asarray(treatment, dtype=float)
    n_c, n_t = c.size, t.size
    if n_c < 2 or n_t < 2:
        raise ValueError("each arm needs at least two observations")

    mean_diff = t.mean() - c.mean()
    var_c = c.var(ddof=1)
    var_t = t.var(ddof=1)

    # V = variance of the estimated mean difference.
    v = var_c / n_c + var_t / n_t
    if v <= 0:
        return 1.0

    if tau_squared is None:
        pooled_var = ((n_c - 1) * var_c + (n_t - 1) * var_t) / (n_c + n_t - 2)
        tau_squared = float(pooled_var)
    if tau_squared <= 0:
        return 1.0

    # mSPRT mixture likelihood ratio (normal mixing distribution centred at 0).
    log_lambda = 0.5 * math.log(v / (v + tau_squared)) + (
        tau_squared * mean_diff**2 / (2.0 * v * (v + tau_squared))
    )
    lambda_ = math.exp(log_lambda)
    return float(min(1.0, 1.0 / lambda_))
