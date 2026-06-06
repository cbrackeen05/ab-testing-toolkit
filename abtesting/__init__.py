"""ab-testing-toolkit: production-quality A/B experiment analysis.

A clean, well-tested toolkit for end-to-end experiment analysis: hypothesis
testing, power and sample-size planning, multiple-comparison corrections,
sequential testing with valid early stopping, and visualization.
"""

from .corrections import (
    CorrectionResult,
    benjamini_hochberg,
    bonferroni,
    holm_bonferroni,
)
from .experiment import Experiment, ExperimentResult
from .sequential import SPRTResult, always_valid_pvalue, sprt
from .power import (
    experiment_runtime_days,
    minimum_detectable_effect,
    minimum_sample_size,
    observed_power,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Experiment",
    "ExperimentResult",
    "minimum_sample_size",
    "minimum_detectable_effect",
    "observed_power",
    "experiment_runtime_days",
    "bonferroni",
    "holm_bonferroni",
    "benjamini_hochberg",
    "CorrectionResult",
    "sprt",
    "always_valid_pvalue",
    "SPRTResult",
]
