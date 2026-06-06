# ab-testing-toolkit

[![CI](https://github.com/cbrackeen05/ab-testing-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/cbrackeen05/ab-testing-toolkit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-quality Python library for **end-to-end A/B experiment analysis**. It packages
the statistical machinery a data scientist actually needs to run a trustworthy experiment —
hypothesis testing, power and sample-size planning, multiple-comparison corrections,
sequentially valid early stopping, and validation tooling — behind a small, well-tested,
fully type-hinted API. It exists because the hard part of A/B testing isn't computing a
p-value; it's avoiding the dozen statistical traps (peeking, underpowering, multiplicity,
broken randomization) that quietly produce false wins.

## Installation

```bash
pip install -e ".[dev]"   # from a clone, with test/dev tooling
```

Requires Python ≥ 3.10. Runtime dependencies: `numpy`, `scipy`, `pandas`, `matplotlib`,
`seaborn`.

## Quickstart

```python
import numpy as np
from abtesting import Experiment

rng = np.random.default_rng(0)
control   = rng.normal(100, 15, 2_000)
treatment = rng.normal(103, 15, 2_000)

exp = Experiment(control, treatment, metric_type="continuous")
result = exp.ttest()                      # Welch's t-test by default

print(result.p_value)                     # 1.0e-12
print(result.is_significant)              # True
print(result.confidence_interval)         # (2.46, 4.32) on the mean difference
print(result.relative_lift)               # +0.034
```

Every test returns a uniform `ExperimentResult`, so results from different tests are
directly comparable and serializable (`exp.to_dataframe()`).

See [`notebooks/worked_example.ipynb`](notebooks/worked_example.ipynb) for a full narrative
walkthrough from planning to platform validation.

## Features

| Module | What it does |
| --- | --- |
| `experiment` | `Experiment` class: Welch's t-test, Mann-Whitney U, chi-squared, bootstrap CIs, relative lift |
| `power` | Minimum sample size, minimum detectable effect, post-hoc power, runtime estimation |
| `corrections` | Bonferroni, Holm-Bonferroni (FWER), Benjamini-Hochberg (FDR) |
| `sequential` | Wald's SPRT and the always-valid (mSPRT) p-value for safe early stopping |
| `visualizations` | Distribution, confidence-interval, power-curve, p-value, and SPRT-boundary plots |
| `utils` | Sample-ratio-mismatch check, winsorizing, log transforms, Cohen's *d* |

### Public API

**`experiment`**
- `Experiment(control, treatment, metric_type="continuous")` — core analysis object.
  - `.summary()` → dict of group means, SDs, sample sizes, absolute effect, relative lift
  - `.ttest(alpha=0.05, alternative="two-sided")` → Welch's t-test (continuous)
  - `.mann_whitney(alpha=0.05, alternative="two-sided")` → non-parametric rank test
  - `.chi_squared(alpha=0.05)` → proportion test (binary metrics)
  - `.bootstrap_ci(n_bootstrap=10000, ci=0.95)` → bootstrap CI on the mean difference
  - `.relative_lift()` → `(treatment_mean - control_mean) / control_mean`
  - `.to_dataframe()` → tidy one-row dataframe of the canonical test result
- `ExperimentResult` — dataclass holding `test_name`, `statistic`, `p_value`,
  `confidence_interval`, `effect_size`, `relative_lift`, `is_significant`, `alpha`,
  `n_control`, `n_treatment`.

**`power`**
- `minimum_sample_size(baseline_rate, minimum_detectable_effect, alpha=0.05, power=0.80, metric_type="binary")`
- `minimum_detectable_effect(n, baseline_rate, alpha=0.05, power=0.80, metric_type="binary")`
- `observed_power(n, effect_size, baseline_std, alpha=0.05)`
- `experiment_runtime_days(daily_traffic, baseline_rate, mde, alpha=0.05, power=0.80, traffic_split=0.5)`

**`corrections`** (each returns a `CorrectionResult`)
- `bonferroni(p_values, alpha=0.05)` — controls FWER
- `holm_bonferroni(p_values, alpha=0.05)` — controls FWER, more powerful
- `benjamini_hochberg(p_values, alpha=0.05)` — controls FDR

**`sequential`**
- `sprt(control_successes, control_n, treatment_successes, treatment_n, alpha=0.05, beta=0.20, h0_rate=None, h1_rate=None)` → `SPRTResult`
- `always_valid_pvalue(control, treatment, alpha=0.05)` → always-valid p-value (mSPRT)

**`visualizations`** (all return a `matplotlib.figure.Figure`)
- `plot_distributions`, `plot_confidence_interval`, `plot_power_curve`,
  `plot_pvalue_distribution`, `plot_sequential_boundaries`

**`utils`**
- `check_sample_ratio_mismatch`, `winsorize`, `log_transform`, `cohens_d`

## Statistical methodology

- **Welch's t-test** for comparing means under unequal variances —
  Welch, B. L. (1947), *Biometrika* 34(1/2), 28–35.
- **Mann-Whitney U** rank test for non-normal/ordinal metrics —
  Mann & Whitney (1947), *Annals of Mathematical Statistics* 18(1), 50–60.
- **Power analysis** via the normal approximation to the two-sample test —
  Cohen, J. (1988), *Statistical Power Analysis for the Behavioral Sciences*.
- **Benjamini-Hochberg FDR control** —
  Benjamini & Hochberg (1995), *JRSS B* 57(1), 289–300.
- **Holm step-down FWER control** —
  Holm, S. (1979), *Scandinavian Journal of Statistics* 6(2), 65–70.
- **Sequential Probability Ratio Test** —
  Wald, A. (1945), *Annals of Mathematical Statistics* 16(2), 117–186.
- **Always-valid inference (mSPRT)** —
  Johari, Pekelis & Walsh (2017), *Always Valid Inference: Bringing Sequential Analysis to
  A/B Testing*.

## Design decisions

**Welch's t-test, not Student's.** A/B arms routinely have different variances (a treatment
can change spread as well as level). Welch's test does not assume equal variances, is barely
less powerful when they happen to be equal, and is therefore the correct default. Student's
pooled-variance test is deliberately not offered as the default.

**FWER vs. FDR is a deliberate choice, not a default.** Bonferroni/Holm control the
probability of *any* false positive (family-wise error rate) — appropriate for a small
number of decisions where one false win is costly. Benjamini-Hochberg controls the *expected
proportion* of false positives among rejections (false discovery rate) — appropriate when
screening many experiments, trading a known false-discovery fraction for substantially more
power. The library exposes all three and documents when to reach for each, rather than
hard-coding one.

**Sequential testing because peeking is the norm.** Fixed-sample p-values are only valid at
a single, pre-committed sample size; teams peek constantly, which inflates false positives
toward 100% under continuous monitoring. The `sequential` module provides Wald's SPRT
(optimal stop/continue decisions for binary metrics) and the mSPRT always-valid p-value
(valid at every sample size, so you may peek freely). For SPRT's error guarantees to hold
exactly, fix both hypotheses in advance via `h1_rate`.

**Uniform results and no hidden plotting.** Every test returns the same `ExperimentResult`
shape; every plot returns a `Figure` and never calls `plt.show()`, so the code composes
cleanly in notebooks, scripts, dashboards, and tests.

## Development

```bash
pytest tests/ --cov=abtesting --cov-report=term-missing   # ~120 tests, 97% coverage
mypy abtesting/                                            # fully type-checked
```

CI runs the suite and type checks across Python 3.10–3.12 on every push and pull request.

## License

MIT — see [LICENSE](LICENSE).
