"""
A package to deduce specific profiles from history data in order to deduce
thresholds on loads for other periods.
"""

from .hybrid_rel_std import HybridThreshold
from .relative_std import RelativeSTDThreshold
from .relative import RelativeThreshold
