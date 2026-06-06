"""ab-testing-toolkit: production-quality A/B experiment analysis.

A clean, well-tested toolkit for end-to-end experiment analysis: hypothesis
testing, power and sample-size planning, multiple-comparison corrections,
sequential testing with valid early stopping, and visualization.
"""

from .experiment import Experiment, ExperimentResult
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
]
