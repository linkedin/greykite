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
# original author: Albert Chen
"""Constants used by code in `~greykite.common` or in multiple places:
`~greykite.algo`, `~greykite.sklearn`,
and/or `~greykite.framework`.
"""

from enum import Enum


# The time series data is represented in pandas dataframes
# The default column names for the series are given below
TIME_COL = "ts"
"""The default name for the column with the timestamps of the time series"""
VALUE_COL = "y"
"""The default name for the column with the values of the time series"""
ACTUAL_COL = "actual"
"""The column name representing actual (observed) values"""
PREDICTED_COL = "forecast"
"""The column name representing the predicted values"""
RESIDUAL_COL = "residual"
"""The column name representing the forecast residuals."""
PREDICTED_LOWER_COL = "forecast_lower"
"""The column name representing upper bounds of prediction interval"""
PREDICTED_UPPER_COL = "forecast_upper"
"""The column name representing lower bounds of prediction interval"""
NULL_PREDICTED_COL = "forecast_null"
"""The column name representing predicted values from null model"""
ERR_STD_COL = "err_std"
"""The column name representing the error standard deviation from models"""
QUANTILE_SUMMARY_COL = "quantile_summary"
"""The column name representing the quantile summary from models"""

# Evaluation metrics corresponding to `~greykite.common.evaluation`
R2_null_model_score = "R2_null_model_score"
"""Evaluation metric. Improvement in the specified loss function compared to the predictions of a null model."""
FRACTION_OUTSIDE_TOLERANCE = "Outside Tolerance (fraction)"
"""Evaluation metric. The fraction of predictions outside the specified tolerance level"""
PREDICTION_BAND_WIDTH = "Prediction Band Width (%)"
"""Evaluation metric. Relative size of prediction bands vs actual, as a percent"""
PREDICTION_BAND_COVERAGE = "Prediction Band Coverage (fraction)"
"""Evaluation metric. Fraction of observations within the bands"""
LOWER_BAND_COVERAGE = "Coverage: Lower Band"
"""Evaluation metric. Fraction of observations within the lower band"""
UPPER_BAND_COVERAGE = "Coverage: Upper Band"
"""Evaluation metric. Fraction of observations within the upper band"""
COVERAGE_VS_INTENDED_DIFF = "Coverage Diff: Actual_Coverage - Intended_Coverage"
"""Evaluation metric. Difference between actual and intended coverage"""

# Column names used by `~greykite.common.features.timeseries_features`
EVENT_DF_DATE_COL = "date"
"""Name of date column for the DataFrames passed to silverkite `custom_daily_event_df_dict`"""
EVENT_DF_LABEL_COL = "event_name"
"""Name of event column for the DataFrames passed to silverkite `custom_daily_event_df_dict`"""
EVENT_PREFIX = "events"
"""Prefix for naming event features."""
EVENT_DEFAULT = ""
"""Label used for days without an event."""
EVENT_INDICATOR = "event"
"""Binary indicatory for an event"""
CHANGEPOINT_COL_PREFIX = "changepoint"
"""Prefix for naming changepoint features."""
CHANGEPOINT_COL_PREFIX_SHORT = "cp"
"""Short prefix for naming changepoint features."""

# Column names used by
# `~greykite.common.features.adjust_anomalous_data.adjust_anomalous_data`
START_TIME_COL = "start_time"
"""Start timestamp column name"""
END_TIME_COL = "end_time"
"""Standard end timestamp column"""
ADJUSTMENT_DELTA_COL = "adjustment_delta"
"""Adjustment column"""
METRIC_COL = "metric"
"""Column to denote metric of interest"""
DIMENSION_COL = "dimension"
"""Dimension column"""
ANOMALY_COL = "is_anomaly"
"""The default name for the column with the anomaly labels of the time series"""


# Constants related to
# `~greykite.common.features.timeseries_features.build_time_features_df`.


class TimeFeaturesEnum(Enum):
    """Time features generated by
    `~greykite.common.features.timeseries_features.build_time_features_df`.

    The item names are lower-case letters (kept the same as the values) for easier check of existence.
    To check if a string s is in this Enum,
    use ``s in TimeFeaturesEnum.__dict__["_member_names_"]``.
    Direct check of existence ``s in TimeFeaturesEnum`` is deprecated in python 3.8.
    """
    # Absolute time features
    datetime = "datetime"
    date = "date"
    year = "year"
    year_length = "year_length"
    quarter = "quarter"
    quarter_start = "quarter_start"
    quarter_length = "quarter_length"
    month = "month"
    month_length = "month_length"
    hour = "hour"
    minute = "minute"
    second = "second"
    year_quarter = "year_quarter"
    year_month = "year_month"
    woy = "woy"
    doy = "doy"
    doq = "doq"
    dom = "dom"
    dow = "dow"
    str_dow = "str_dow"
    str_doy = "str_doy"
    is_weekend = "is_weekend"
    # Relative time features
    year_woy = "year_woy"
    month_dom = "month_dom"
    year_woy_dow = "year_woy_dow"
    woy_dow = "woy_dow"
    dow_hr = "dow_hr"
    dow_hr_min = "dow_hr_min"
    tod = "tod"
    tow = "tow"
    tom = "tom"
    toq = "toq"
    toy = "toy"
    conti_year = "conti_year"
    dow_grouped = "dow_grouped"
    # ISO time features
    year_iso = "year_iso"
    year_woy_iso = "year_woy_iso"
    year_woy_dow_iso = "year_woy_dow_iso"
    # Continuous time features
    ct1 = "ct1"
    ct2 = "ct2"
    ct3 = "ct3"
    ct_sqrt = "ct_sqrt"
    ct_root3 = "ct_root3"


class GrowthColEnum(Enum):
    """Human-readable names for the growth columns generated by
    `~greykite.common.features.timeseries_features.build_time_features_df`.

    The names are the human-readable names, and the values are the corresponding
    column names generated by `~greykite.common.features.timeseries_features.build_time_features_df`.
    """
    linear = TimeFeaturesEnum.ct1.value
    quadratic = TimeFeaturesEnum.ct2.value
    cubic = TimeFeaturesEnum.ct3.value
    sqrt = TimeFeaturesEnum.ct_sqrt.value
    cuberoot = TimeFeaturesEnum.ct_root3.value


# Column names used by
# `~greykite.common.features.timeseries_lags`
LAG_INFIX = "_lag"
"""Infix for lagged feature names"""
AGG_LAG_INFIX = "avglag"
"""Infix for aggregated lag feature names"""

# Patterns for categorizing timeseries features
TREND_REGEX = f"{CHANGEPOINT_COL_PREFIX}\\d|ct\\d|ct_|{CHANGEPOINT_COL_PREFIX_SHORT}\\d"
"""Growth terms, including changepoints."""
SEASONALITY_REGEX = "sin\\d|cos\\d"
"""Seasonality terms modeled by fourier series."""
EVENT_REGEX = f"{EVENT_PREFIX}_"
"""Event terms."""
LAG_REGEX = f"{LAG_INFIX}\\d|_{AGG_LAG_INFIX}_\\d"
"""Lag terms."""

LOGGER_NAME = "Greykite"
"""Name used by the logger."""
