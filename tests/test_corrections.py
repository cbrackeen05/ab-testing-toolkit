"""Tests for ``abtesting.corrections``."""

from __future__ import annotations

import numpy as np
import pytest

from abtesting.corrections import (
    CorrectionResult,
    benjamini_hochberg,
    bonferroni,
    holm_bonferroni,
)

ALL_METHODS = [bonferroni, holm_bonferroni, benjamini_hochberg]


class TestBonferroni:
    def test_known_adjusted_values(self) -> None:
        res = bonferroni([0.01, 0.02, 0.03, 0.04], alpha=0.05)
        assert isinstance(res, CorrectionResult)
        # adjusted = p * m, clipped to 1.
        assert res.adjusted_p_values == pytest.approx([0.04, 0.08, 0.12, 0.16])
        assert res.rejected == [True, False, False, False]
        assert res.n_rejected == 1

    def test_adjusted_clipped_at_one(self) -> None:
        res = bonferroni([0.5, 0.6], alpha=0.05)
        assert max(res.adjusted_p_values) <= 1.0

    def test_preserves_input_order(self) -> None:
        res = bonferroni([0.04, 0.01, 0.03], alpha=0.05)
        # Smallest p is in position 1; only it should be rejected.
        assert res.rejected == [False, True, False]


class TestBenjaminiHochberg:
    def test_classic_bh1995_example(self) -> None:
        # The 15 p-values from Benjamini & Hochberg (1995), Table 1. BH at alpha=0.05
        # rejects the first four (p <= 0.0095).
        p = [
            0.0001, 0.0004, 0.0019, 0.0095, 0.0201, 0.0278, 0.0298, 0.0344,
            0.0459, 0.3240, 0.4262, 0.5719, 0.6528, 0.7590, 1.000,
        ]
        res = benjamini_hochberg(p, alpha=0.05)
        assert res.n_rejected == 4
        assert res.rejected[:4] == [True, True, True, True]
        assert not any(res.rejected[4:])

    def test_rejects_at_least_as_many_as_bonferroni(self) -> None:
        p = [0.001, 0.009, 0.02, 0.03, 0.04, 0.2, 0.5]
        bh = benjamini_hochberg(p).n_rejected
        bf = bonferroni(p).n_rejected
        assert bh >= bf

    def test_adjusted_values_are_monotone_in_sorted_order(self) -> None:
        p = [0.04, 0.001, 0.02, 0.5, 0.009]
        res = benjamini_hochberg(p)
        order = np.argsort(p)
        sorted_adj = np.array(res.adjusted_p_values)[order]
        assert np.all(np.diff(sorted_adj) >= -1e-12)


class TestHolmBonferroni:
    def test_known_example(self) -> None:
        res = holm_bonferroni([0.01, 0.04, 0.03, 0.005], alpha=0.05)
        assert res.n_rejected == 2

    def test_dominates_bonferroni(self) -> None:
        p = [0.012, 0.013, 0.014, 0.2]
        assert holm_bonferroni(p).n_rejected >= bonferroni(p).n_rejected


@pytest.mark.parametrize("method", ALL_METHODS)
class TestCommonProperties:
    def test_all_null_rejects_nothing(self, method) -> None:
        # Every p-value above alpha -> no rejections under any method.
        res = method([0.06, 0.2, 0.5, 0.9], alpha=0.05)
        assert res.n_rejected == 0
        assert res.rejected == [False, False, False, False]

    def test_n_rejected_non_increasing_as_alpha_decreases(self, method) -> None:
        rng = np.random.default_rng(3)
        # Mix of true signals and nulls.
        p = np.concatenate([rng.uniform(0, 0.01, 10), rng.uniform(0, 1, 40)]).tolist()
        counts = [method(p, alpha=a).n_rejected for a in (0.10, 0.05, 0.01, 0.001)]
        assert all(earlier >= later for earlier, later in zip(counts, counts[1:]))

    def test_adjusted_at_least_raw(self, method) -> None:
        p = [0.001, 0.02, 0.03, 0.2]
        res = method(p)
        assert all(adj >= raw - 1e-12 for adj, raw in zip(res.adjusted_p_values, p))

    def test_invalid_pvalues_raise(self, method) -> None:
        with pytest.raises(ValueError):
            method([0.1, 1.5])
        with pytest.raises(ValueError):
            method([])

    def test_invalid_alpha_raises(self, method) -> None:
        with pytest.raises(ValueError):
            method([0.1, 0.2], alpha=1.5)
