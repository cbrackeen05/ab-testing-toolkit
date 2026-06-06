"""Smoke tests for ``abtesting.visualizations``.

These assert that each plotting function returns a Matplotlib ``Figure`` with the
expected basic structure, without rendering to a screen. The Agg backend is forced so
the suite runs headless in CI.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
import pytest

from abtesting import Experiment, sprt
from abtesting.visualizations import (
    plot_confidence_interval,
    plot_distributions,
    plot_power_curve,
    plot_pvalue_distribution,
    plot_sequential_boundaries,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def experiment() -> Experiment:
    rng = np.random.default_rng(0)
    return Experiment(rng.normal(100, 15, 500), rng.normal(104, 15, 500))


def test_plot_distributions_returns_figure(experiment: Experiment) -> None:
    fig = plot_distributions(experiment)
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    assert ax.get_xlabel() == "Metric value"
    # Two means -> at least two dashed vertical lines plus the histograms.
    assert ax.get_legend() is not None


def test_plot_distributions_accepts_title(experiment: Experiment) -> None:
    fig = plot_distributions(experiment, title="Custom")
    assert fig.axes[0].get_title() == "Custom"


def test_plot_confidence_interval_returns_figure(experiment: Experiment) -> None:
    result = experiment.ttest()
    fig = plot_confidence_interval(result)
    assert isinstance(fig, plt.Figure)
    assert fig.axes[0].get_xlabel().startswith("Effect")


def test_plot_confidence_interval_handles_one_sided(experiment: Experiment) -> None:
    # One-sided CI has an infinite bound; the function should still produce a figure.
    result = experiment.ttest(alternative="greater")
    fig = plot_confidence_interval(result)
    assert isinstance(fig, plt.Figure)


def test_plot_power_curve_returns_figure() -> None:
    fig = plot_power_curve(0.10, [0.01, 0.02, 0.03, 0.05])
    assert isinstance(fig, plt.Figure)
    ax = fig.axes[0]
    assert ax.get_ylabel() == "Sample size per group"
    # Sample size should be monotonically decreasing across increasing MDE.
    line = ax.lines[0]
    ydata = line.get_ydata()
    assert np.all(np.diff(ydata) <= 0)


def test_plot_pvalue_distribution_returns_figure() -> None:
    rng = np.random.default_rng(1)
    p_values = rng.uniform(0, 1, 500).tolist()
    fig = plot_pvalue_distribution(p_values)
    assert isinstance(fig, plt.Figure)
    assert fig.axes[0].get_xlabel() == "p-value"


def test_plot_sequential_boundaries_returns_figure() -> None:
    rng = np.random.default_rng(2)
    stream = rng.binomial(1, 0.13, 3000)
    cum = np.cumsum(stream)
    observations = list(range(200, 3001, 200))
    results = [
        sprt(0, 1, int(cum[n - 1]), n, h0_rate=0.10, h1_rate=0.12)
        for n in observations
    ]
    fig = plot_sequential_boundaries(results, observations)
    assert isinstance(fig, plt.Figure)
    assert fig.axes[0].get_ylabel() == "Log-likelihood ratio"


def test_plot_sequential_boundaries_validates_lengths() -> None:
    rng = np.random.default_rng(3)
    results = [sprt(5, 100, 8, 100)]
    with pytest.raises(ValueError):
        plot_sequential_boundaries(results, [100, 200])
    with pytest.raises(ValueError):
        plot_sequential_boundaries([], [])
