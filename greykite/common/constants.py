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
"""The default name for the column with the timestamps of the time series."""
VALUE_COL = "y"
"""The default name for the column with the values of the time series."""
ACTUAL_COL = "actual"
"""The column name representing actual (observed) values."""
PREDICTED_COL = "forecast"
"""The column name representing the predicted values."""
RESIDUAL_COL = "residual"
"""The column name representing the forecast residuals."""
PREDICTED_LOWER_COL = "forecast_lower"
"""The column name representing lower bounds of prediction interval."""
PREDICTED_UPPER_COL = "forecast_upper"
"""The column name representing upper bounds of prediction interval."""
NULL_PREDICTED_COL = "forecast_null"
"""The column name representing predicted values from null model."""
ERR_STD_COL = "err_std"
"""The column name representing the error standard deviation from models."""
QUANTILE_SUMMARY_COL = "quantile_summary"
"""The column name representing the quantile summary from models."""

# Evaluation metrics corresponding to `~greykite.common.evaluation`.
R2_null_model_score = "R2_null_model_score"
"""Evaluation metric. Improvement in the specified loss function compared to the predictions of a null model."""
FRACTION_OUTSIDE_TOLERANCE = "Outside Tolerance (fraction)"
"""Evaluation metric. The fraction of predictions outside the specified tolerance level."""
PREDICTION_BAND_WIDTH = "Prediction Band Width (%)"
"""Evaluation metric. Relative size of prediction bands vs actual, as a percent."""
PREDICTION_BAND_COVERAGE = "Prediction Band Coverage (fraction)"
"""Evaluation metric. Fraction of observations within the bands."""
LOWER_BAND_COVERAGE = "Coverage: Lower Band"
"""Evaluation metric. Fraction of observations within the lower band."""
UPPER_BAND_COVERAGE = "Coverage: Upper Band"
"""Evaluation metric. Fraction of observations within the upper band."""
COVERAGE_VS_INTENDED_DIFF = "Coverage Diff: Actual_Coverage - Intended_Coverage"
"""Evaluation metric. Difference between actual and intended coverage."""

# Column names used by `~greykite.common.features.timeseries_features`.
EVENT_DF_DATE_COL = "date"
"""Name of date column for the DataFrames passed to silverkite `custom_daily_event_df_dict`."""
EVENT_DF_LABEL_COL = "event_name"
"""Name of event column for the DataFrames passed to silverkite `custom_daily_event_df_dict`."""
EVENT_PREFIX = "events"
"""Prefix for naming event features."""
EVENT_DEFAULT = ""
"""Label used for days without an event."""
EVENT_INDICATOR = "event"
"""Binary indicator for an event."""
IS_EVENT_COL = "is_event"
"""Indicator column in feature matrix, 1 if the day is an event or its neighboring days."""
IS_EVENT_ADJACENT_COL = "is_event_adjacent"
"""Indicator column in feature matrix, 1 if the day is adjacent to an event."""
IS_EVENT_EXACT_COL = "is_event_exact"
"""Indicator column in feature matrix, 1 if the day is an event but not its neighboring days."""
EVENT_SHIFTED_SUFFIX_BEFORE = "_before"
"""The suffix for neighboring events before the events added to the event names."""
EVENT_SHIFTED_SUFFIX_AFTER = "_after"
"""The suffix for neighboring events after the events added to the event names."""
CHANGEPOINT_COL_PREFIX = "changepoint"
"""Prefix for naming changepoint features."""
CHANGEPOINT_COL_PREFIX_SHORT = "cp"
"""Short prefix for naming changepoint features."""
LEVELSHIFT_COL_PREFIX_SHORT = "ctp"
"""Short prefix for naming levelshift features."""

# Column names used by
# `~greykite.common.features.adjust_anomalous_data.adjust_anomalous_data`.
START_TIME_COL = "start_time"
"""Default column name for anomaly start time in the anomaly dataframe."""
END_TIME_COL = "end_time"
"""Default column name for anomaly end time in the anomaly dataframe."""
ADJUSTMENT_DELTA_COL = "adjustment_delta"
"""Default column name for anomaly adjustment in the anomaly dataframe."""
METRIC_COL = "metric"
"""Column to denote metric of interest."""
DIMENSION_COL = "dimension"
"""Dimension column."""
ANOMALY_COL = "is_anomaly"
"""Default column name for anomaly labels (boolean) in the time series."""
PREDICTED_ANOMALY_COL = "is_anomaly_predicted"
"""Default column name for predicted anomaly labels (boolean) in the time series."""

# Column names used in anomaly dataframe during anomaly detection.
SEVERITY_SCORE_COL = "severity_score"
"""Default column name for anomaly severity score in the anomaly dataframe."""
USER_REVIEWED_COL = "is_user_reviewed"
"""Default column name for whether an anomaly is reviewed by the user (boolean) in the anomaly dataframe."""
NEW_PATTERN_ANOMALY_COL = "new_pattern_anomaly"
"""Default column name for whether an anomaly is a new pattern (boolean) in the anomaly dataframe."""


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
    us_dst = "us_dst"
    eu_dst = "eu_dst"


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
"""Infix for lagged feature names."""
AGG_LAG_INFIX = "avglag"
"""Infix for aggregated lag feature names."""

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

# Default regex dictionary for component plots
DEFAULT_COMPONENTS_REGEX_DICT = {
    "Regressors": ".*regressor.|regressor",
    "Autoregressive": ".*_lag.|.*avglag.",
    "Event": f".*{EVENT_REGEX}.",
    "Seasonality": f".*tod.|.*tow.|.*dow.|.*is_weekend.|.*tom.|.*month.|.*toq.|.*quarter.|.*toy.|.*year.|.*yearly",
    "Trend": TREND_REGEX,
}

# Detailed seasonality regex dictionary for component plots
DETAILED_SEASONALITY_COMPONENTS_REGEX_DICT = {
    "Regressors": ".*regressor.|regressor",
    "Autoregressive": ".*_lag.|.*avglag.",
    "Event": f".*{EVENT_REGEX}.",
    "Daily": f".*tod.",
    "Weekly": f".*tow.|.*dow.|.*is_weekend.",
    "Monthly": f".*tom.|.*month.",
    "Quarterly": f".*toq.|.*quarter.",
    "Yearly": f".*toy.|.*year.|.*yearly",
    "Trend": TREND_REGEX,
}
