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
# original author: Kaixu Yang
"""Automatically populates changepoint detection parameters from input data."""

from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from greykite.algo.common.seasonality_inferrer import SeasonalityInferConfig
from greykite.algo.common.seasonality_inferrer import SeasonalityInferrer
from greykite.algo.common.seasonality_inferrer import TrendAdjustMethodEnum
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL


def get_changepoint_resample_freq(
        n_points: int,
        min_increment: timedelta,
        min_num_points_after_agg: int = 100) -> Optional[str]:
    """Gets the changepoint detection resample frequency parameter.

    Parameters
    ----------
    n_points : `int`
        The number of data points in the time series.
    min_increment : `datetime.timedelta`
        The minimum increment between time series points.
    min_num_points_after_agg : `int`, default 100
        The minimum number of observations required after aggregation.

    Returns
    -------
    resample_freq : `str` or None
        The resample frequency.
        Will be one of "D", "3D" and "7D".
        If None, resample will be skipped.
    """
    # When ``min_increment`` is at least 7 days,
    # the data is considered at least weekly data.
    # In this case, we don't do aggregation.
    if min_increment >= timedelta(days=7):
        # Returning None for this case.
        return None

    # From now on the data is sub-weekly.
    # The candidates of ``resample_freq`` are "7D", "3D" and "D".
    # We use the longest aggregation frequency that has at least
    # ``min_num_points_after_agg`` points after aggregation.
    # Currently, we do not support sub-daily aggregation frequency,
    # because it is not a common case.
    # We also do not recommend based on detection result for speed purpose.
    data_length = n_points * min_increment
    if data_length >= timedelta(days=7 * min_num_points_after_agg):
        resample_freq = "7D"
    elif data_length >= timedelta(days=3 * min_num_points_after_agg):
        resample_freq = "3D"
    else:
        resample_freq = "D"

    return resample_freq


def get_yearly_seasonality_order(
        df: pd.DataFrame,
        time_col: str,
        value_col: str,
        resample_freq: str) -> int:
    """Infers the yearly seasonality order for changepoint detection.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The input timeseris.
    time_col : `str`
        The column name for timestamps in ``df``.
    value_col : `str`
        The column name for values in ``df``.
    resample_freq : `str`
        The aggregation frequency string in changepoint detection configuration.

    Returns
    -------
    yearly_seasonality_order : `int`
        The inferred yearly seasonality with the best BIC score.
    """
    model = SeasonalityInferrer()
    result = model.infer_fourier_series_order(
        df=df,
        time_col=time_col,
        value_col=value_col,
        configs=[
            SeasonalityInferConfig(
                seas_name="yearly",
                col_name="toy",
                period=1.0,
                max_order=30,
                adjust_trend_method=TrendAdjustMethodEnum.seasonal_average.name,
                adjust_trend_param=dict(trend_average_col="year"),
                fit_algorithm="ridge",
                offset=0,
                tolerance=0.0,
                aggregation_period=resample_freq,
                criterion="bic",
                plotting=False
            )
        ]
    )
    return result["best_orders"]["yearly"]


def get_potential_changepoint_n(
        n_points: int,
        total_increment: timedelta,
        resample_freq: str,
        yearly_seasonality_order: int,
        cap: int = 100) -> int:
    """Gets the number of potential changepoints for changepoint detection.

    Parameters
    ----------
    n_points : `int`
        The number of data points in the time series.
    total_increment : `datetime.timedelta`
        The total time span of the input time series.
    resample_freq : `str`
        The resample frequency used in changepoint detection.
    yearly_seasonality_order : `int`
        The yearly seasonality order used in changepoint detection.
    cap : `int`, default 100
        The maximum number of potential changepoints.

    Returns
    -------
    potential_changepoint_n : `int`
        The number of potential changepoints used in changepoint detection.
    """
    try:
        # The ``resample_freq`` is one of "D", "3D" and "7D".
        n_points_after_agg = np.floor(total_increment / to_offset(resample_freq).delta)
    except AttributeError:
        # The ``resample_freq`` is None or other freq that is at least "W".
        n_points_after_agg = n_points
    # Sets number of potential changepoints to be at most
    # aggregated data length - # seasonality features - 1 (intercept term) for estimability.
    # Here we ignore dropping potential changepoints from the end.
    # If we use the function above to infer aggregation frequency,
    # there will be enough potential changepoints.
    n_changepoints = max(0, n_points_after_agg - 2 * yearly_seasonality_order - 1)
    # Caps the number of potential changepoints for speed purpose.
    n_changepoints = min(n_changepoints, cap)
    return n_changepoints


def get_no_changepoint_distance_from_end(
        min_increment: timedelta,
        forecast_horizon: int,
        min_num_points_after_last_changepoint: int = 1) -> str:
    """Gets the distance from end of time series where no changepoints will be placed.

    Parameters
    ----------
    min_increment : `datetime.timedelta`
        The minimum increment between time series points.
    forecast_horizon : `int`
        The forecast horizon.
    min_num_points_after_last_changepoint : `int`, default 1
        The minimum number of data points after the last potential changepoint.

    Returns
    -------
    no_changepoint_distance_from_end : `str`
        A string indicating the period from the end of the time series where no
        potential changepoints will be placed.
    """
    min_increment_in_days = min_increment.days
    min_forecast_horizon_in_days = (min_increment * forecast_horizon).days
    if min_forecast_horizon_in_days < 1:
        min_forecast_horizon_in_days = 1
    # Minimum distance must be at least ``min_num_points_after_last_changepoint``.
    min_distance_days = min_increment_in_days * min_num_points_after_last_changepoint
    # We add extra constraints according to the data frequency,
    # and some special constraints for the commonly seen data frequencies.
    if min_increment_in_days >= 365:
        # At least yearly data, distance is at least 2 * forecast horizon.
        min_distance_days = max(min_distance_days, min_forecast_horizon_in_days * 2)
        if min_increment_in_days == 365:
            # Exactly yearly data, between 3 years and 20 years.
            min_distance_days = max(min_distance_days, 3 * 365)
            min_distance_days = min(min_distance_days, 20 * 366)
    elif min_increment_in_days >= 28:
        # Between monthly and yearly data, distance is between 3 * forecast horizon and 6 years.
        min_distance_days = max(min_distance_days, min_forecast_horizon_in_days * 3)
        min_distance_days = min(min_distance_days, 6 * 366)
        if min_increment_in_days == 90:
            # Exactly quarterly data, at least 4 quarters.
            min_distance_days = max(min_distance_days, 4 * 90)
        elif min_increment_in_days == 28:
            # Exactly monthly data, at least 3 months.
            min_distance_days = max(min_distance_days, 3 * 28)
    else:
        # Sub-monthly data, distance is at least 4 * forecast horizon.
        min_distance_days = max(min_distance_days, min_forecast_horizon_in_days * 4)
        if min_increment_in_days == 7:
            # Weekly data, between 8 weeks and 104 weeks (2 years).
            min_distance_days = max(min_distance_days, 8 * 7)
            min_distance_days = min(min_distance_days, 104 * 7)
        elif min_increment_in_days == 1:
            # Daily data, between 30 days and 1 year.
            min_distance_days = max(min_distance_days, 30)
            min_distance_days = min(min_distance_days, 365)
        elif min_increment_in_days < 1:
            # Sub-daily data, between 14 days and 1 year.
            min_distance_days = max(min_distance_days, 14)
            min_distance_days = min(min_distance_days, 365)

    no_changepoint_distance_from_end = f"{min_distance_days}D"
    return no_changepoint_distance_from_end


def get_actual_changepoint_min_distance(
        min_increment: timedelta) -> str:
    """Gets the minimum distance between detected changepoints.

    Parameters
    ----------
    min_increment : `datetime.timedelta`
        The minimum increment between time series points.

    Returns
    -------
    actual_changepoint_min_distance : `str`
        The minimum distance between detected changepoints.
    """
    min_distance = min_increment.days * 2  # At most every two data points.
    if min_increment < timedelta(days=1):
        min_distance = max(min_distance, 14)  # At least 14 days for sub-daily data.
    elif min_increment <= timedelta(days=7):
        min_distance = max(min_distance, 30)  # At least 30 days for daily to weekly data.

    actual_changepoint_min_distance = f"{min_distance}D"
    return actual_changepoint_min_distance


def get_regularization_strength() -> float:
    """Gets the regularization strength.

    The regularization strength typically won't affect the result too much,
    if set in a reasonable range (0.4-0.7).
    Here we explicitly set it to 0.6,
    which works well in most cases.
    Setting it to None will trigger cross-validation,
    but has risk to select too many changepoints.
    """
    return 0.6


def generate_trend_changepoint_detection_params(
        df: pd.DataFrame,
        forecast_horizon: int,
        time_col: str = TIME_COL,
        value_col: str = VALUE_COL,
        yearly_seasonality_order: Optional[int] = None) -> Optional[dict]:
    """Automatically generates trend changepoint detection parameters
    based on the input data and forecast horizon.

    Parameters
    ----------
    df : `pandas.DataFrame`
        The input time series.
    forecast_horizon : `int`
        The forecast horizon.
    time_col : `str`, default TIME_COL
        The column name for timestamps in ``df``.
    value_col : `str`, default VALUE_COL`
        The column name for time series values in ``df``.
    yearly_seasonality_order : `int` or None, default None
        The yearly seasonality Fourier order.
        If a known good order is given, it will be used.
        Otherwise, it will be inferred from the algorithm.

    Returns
    -------
    params : `dict` [`str`, any]
        The generated trend changepoint detection parameters.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    total_increment = df[time_col].max() - df[time_col].min()

    # Auto changepoints is not active if training data is too short.
    # In such cases, autoregression should be used.
    if total_increment < timedelta(days=90):
        return None

    n_points = len(df)
    min_increment = min((df[time_col] - df[time_col].shift(1)).dropna())

    # Infers ``resample_freq``.
    resample_freq = get_changepoint_resample_freq(
        n_points=n_points,
        min_increment=min_increment
    )

    # Infers ``yearly_seasonality_order``.
    if yearly_seasonality_order is None:
        yearly_seasonality_order = get_yearly_seasonality_order(
            df=df,
            time_col=time_col,
            value_col=value_col,
            resample_freq=resample_freq
        )
    elif yearly_seasonality_order < 0:
        raise ValueError(f"Yearly seasonality order must be a non-negative integer, "
                         f"found {yearly_seasonality_order}.")

    # Infers ``potential_changepoint_n``.
    potential_changepoint_n = get_potential_changepoint_n(
        n_points=n_points,
        total_increment=total_increment,
        resample_freq=resample_freq,
        yearly_seasonality_order=yearly_seasonality_order,
        cap=100
    )

    # Infers ``no_changepoint_distance_from_end``.
    no_changepoint_distance_from_end = get_no_changepoint_distance_from_end(
        min_increment=min_increment,
        forecast_horizon=forecast_horizon,
        min_num_points_after_last_changepoint=4
    )

    # Infers ``actual_changepoint_min_distance``.
    actual_changepoint_min_distance = get_actual_changepoint_min_distance(
        min_increment=min_increment
    )

    # Infers ``regularization_strength``.
    regularization_strength = get_regularization_strength()

    return dict(
        yearly_seasonality_order=yearly_seasonality_order,
        resample_freq=resample_freq,
        regularization_strength=regularization_strength,
        actual_changepoint_min_distance=actual_changepoint_min_distance,
        potential_changepoint_n=potential_changepoint_n,
        no_changepoint_distance_from_end=no_changepoint_distance_from_end
    )
