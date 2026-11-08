"""Galaxy-HRGuard: three-state reliability measurement for wrist heart rate.

The package is deliberately small. Each module corresponds to one stage of the
measurement protocol described in the manuscript:

    raw        source audit of the licensed cohort files (clock rule, legality)
    windows    window construction and the three-state label
    signals    uniform resampling of the legal input channels
    endpoint   the window universe with reference values attached
    features   the legal feature table (PPG, ACC, watch-reported HR)
    measure    the frozen estimands, the cluster bootstrap and the criteria
    online     the online alerting protocol (episodes, budgets, lead time)
    composition, blocks, within, outage
               the exploratory analyses of Table 3
    result_build, result_verify
               assembly and independent recomputation of the headline result

Only ``core`` is imported eagerly; the stage modules pull their own
dependencies when used.
"""
from __future__ import annotations

from .core import (
    DEV,
    FEATS,
    SEAL,
    SUBJECTS,
    group_macro_risk_coverage,
    norm_spectrum,
    pilot,
    risk_coverage,
    sha256_file,
    signal_features,
)

__version__ = "0.1.0"

__all__ = [
    "DEV",
    "FEATS",
    "SEAL",
    "SUBJECTS",
    "group_macro_risk_coverage",
    "norm_spectrum",
    "pilot",
    "risk_coverage",
    "sha256_file",
    "signal_features",
    "__version__",
]
