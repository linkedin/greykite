# BSD 2-CLAUSE LICENSE

# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:

# Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
# #ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# original author: Sayan Patra, Reza Hosseini

from enum import Enum

import numpy as np


RESIDUAL_COL = "residual"
"""The column name representing residual values."""

DEFAULT_COVERAGE_GRID = (
        list(np.arange(1, 20)/20) +
        [0.96, 0.97, 0.98, 0.99, 0.995, 0.999])
"""Default grid of coverage values to optimize anomaly detection performance. """

DEFAULT_VOLATILITY_FEATURES_LIST = [[]]
"""Default list of volatility features to use for anomaly detection."""

# Default target anomaly fraction if not passed.
DEFAULT_TARGET_ANOMALY_FRACTION = 0.05

# Default target anomaly percent if not passed.
DEFAULT_TARGET_ANOMALY_PERCENT = 5.0

Z_SCORE_COL = "z_score"
"""The column name representing z-score values."""

Z_SCORE_CUTOFF = 20.0
"""The cut-off value used to identify outliers. Values with abs(z-score) > ``Z_SCORE_CUTOFF`` are treated as outliers."""

FIG_SHOW = False
"""Whether to show figures in tests."""

PHASE_TRAIN = "train"
"""The phase name for algorithm training."""

PHASE_PREDICT = "predict"
"""The phase name for algorithm prediction."""


class PenalizeMethod(Enum):
    """Enum used in

        `~greykite.detection.detector.reward`

    to construct penalized reward functions.
    Such functions will impose a penalty when the reward value is outside
    a pre-specified interval.

    Attributes
    ----------
    ADDITIVE : `str`
        The penalty will be added to reward
    MULTIPLICATIVE : `str`
        The penalty will be multiplied to the reward
    PENALTY_ONLY : `str`
        The penalty value will be used only (original reward value is ignored.)

    """
    ADDITIVE = "additive"
    MULTIPLICATIVE = "multiplicative"
    PENALTY_ONLY = "penalty_only"
