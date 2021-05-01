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
# original_author: Albert Chen
"""Constants used by `~greykite.framework."""

EVALUATION_PERIOD_CV_MAX_SPLITS = 3
"""Default value for EvaluationPeriodParam().cv_max_splits"""
COMPUTATION_N_JOBS = 1
"""Default value for ComputationParam.n_jobs"""
COMPUTATION_VERBOSE = 1
"""Default value for ComputationParam.verbose"""
CV_REPORT_METRICS_ALL = "ALL"
"""Set `cv_report_metrics` to this value to compute all metrics during CV"""
FRACTION_OUTSIDE_TOLERANCE_NAME = "OutsideTolerance"
"""Short name used to report the result of `FRACTION_OUTSIDE_TOLERANCE` in CV"""
CUSTOM_SCORE_FUNC_NAME = "score"
"""Short name used to report the result of custom `score_func` in CV"""

# UnivariateTimeSeries ``get_quantiles_and_overlays`` output column groups
MEAN_COL_GROUP = "mean"
"""Columns with mean."""
QUANTILE_COL_GROUP = "quantile"
"""Columns with quantile."""
OVERLAY_COL_GROUP = "overlay"
"""Columns with overlay."""

FORECAST_STEP_COL = "forecast_step"
"""The column name for forecast step in benchmarking"""
