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
"""Functions that extract timeseries properties, with additional
domain logic for forecasting.
"""

import math
from datetime import timedelta

from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.enums import SimpleTimeFrequencyEnum
from greykite.common.enums import TimeEnum
from greykite.common.features.timeseries_features import get_default_origin_for_time_vars
from greykite.common.time_properties import get_canonical_data
from greykite.common.time_properties import min_gap_in_seconds


def get_default_horizon_from_period(period, num_observations=None):
    """Returns default forecast horizon based on input data period and num_observations
    :param period: float
        Period of each observation (i.e. average time between observations, in seconds)
    :param num_observations: Optional[int]
        Number of observations for training
    :return: int
        default number of periods to forecast
    """
    default_from_period = get_simple_time_frequency_from_period(period).value.default_horizon
    if num_observations is not None:
        default_from_observations = num_observations // 2  # twice as much training data as forecast horizon
        return min(default_from_period, default_from_observations)  # horizon based on limiting factor
    else:
        return default_from_period


def get_simple_time_frequency_from_period(period):
    """Returns SimpleTimeFrequencyEnum based on input data period
    :param period: float
        Period of each observation (i.e. average time between observations, in seconds)
    :return: SimpleTimeFrequencyEnum
        SimpleTimeFrequencyEnum is used to define default values for horizon, seasonality, etc.
        (but original data frequency is not modified)
    """
    freq_threshold = [
        (SimpleTimeFrequencyEnum.MINUTE, 10.05),  # <= 10 minutes is considered minute-level, buffer for abnormalities
        (SimpleTimeFrequencyEnum.HOUR, 6.05),     # <= 6 hours is considered hourly, buffer for abnormalities
        (SimpleTimeFrequencyEnum.DAY, 2.05),      # <= 2 days is considered daily, buffer for daylight savings
        (SimpleTimeFrequencyEnum.WEEK, 2.05),     # <= 2 weeks is considered weekly, buffer for daylight savings
        (SimpleTimeFrequencyEnum.MONTH, 2.05),    # <= 2 months is considered monthly, buffer for 31-day month
        (SimpleTimeFrequencyEnum.YEAR, 1.01),     # <= 1 years is considered yearly, buffer for leap year
    ]

    for simple_freq, threshold in freq_threshold:
        if period <= simple_freq.value.seconds_per_observation * threshold:
            return simple_freq
    return SimpleTimeFrequencyEnum.MULTIYEAR


def get_forecast_time_properties(
        df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        freq=None,
        date_format=None,
        regressor_cols=None,
        lagged_regressor_cols=None,
        train_end_date=None,
        forecast_horizon=None):
    """Returns the number of training points in `df`, the start year, and prediction end year

    Parameters
    ----------
    df : `pandas.DataFrame` with columns [``time_col``, ``value_col``]
        Univariate timeseries data to forecast
    time_col : `str`, default ``TIME_COL`` in constants.py
        Name of timestamp column in df
    value_col : `str`, default ``VALUE_COL`` in constants.py
        Name of value column in df (the values to forecast)
    freq : `str` or None, default None
        Frequency of input data. Used to generate future dates for prediction.
        Frequency strings can have multiples, e.g. '5H'.
        See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        for a list of frequency aliases.
        If None, inferred by pd.infer_freq.
        Provide this parameter if ``df`` has missing timepoints.
    date_format : `str` or None, default None
        strftime format to parse time column, eg ``%m/%d/%Y``.
        Note that ``%f`` will parse all the way up to nanoseconds.
        If None (recommended), inferred by `pandas.to_datetime`.
    regressor_cols : `list` [`str`] or None, optional, default None
        A list of regressor columns used in the training and prediction DataFrames.
        If None, no regressor columns are used.
        Regressor columns that are unavailable in ``df`` are dropped.
    lagged_regressor_cols : `list` [`str`] or None, optional, default None
        A list of lagged regressor columns used in the training and prediction DataFrames.
        If None, no lagged regressor columns are used.
        Lagged regressor columns that are unavailable in ``df`` are dropped.
    train_end_date : `datetime.datetime`, optional, default None
        Last date to use for fitting the model. Forecasts are generated after this date.
        If None, it is set to the last date with a non-null value in
        ``value_col`` of ``df``.
    forecast_horizon : `int` or None, default None
        Number of periods to forecast into the future. Must be > 0
        If None, default is determined from input data frequency

    Returns
    -------
    time_properties : `dict` [`str`, `any`]
        Time properties dictionary with keys:

        ``"period"`` : `int`
            Period of each observation (i.e. minimum time between observations, in seconds).
        ``"simple_freq"`` : `SimpleTimeFrequencyEnum`
            ``SimpleTimeFrequencyEnum`` member corresponding to data frequency.
        ``"num_training_points"`` : `int`
            Number of observations for training.
        ``"num_training_days"`` : `int`
            Number of days for training.
        ``"days_per_observation"``: `float`
            The time frequency in day units.
        ``"forecast_horizon"``: `int`
            The number of time intervals for which forecast is needed.
        ``"forecast_horizon_in_timedelta"``: `datetime.timedelta`
            The forecast horizon length in timedelta units.
        ``"forecast_horizon_in_days"``: `float`
            The forecast horizon length in day units.
        ``"start_year"`` : `int`
            Start year of the training period.
        ``"end_year"`` : `int`
            End year of the forecast period.
        ``"origin_for_time_vars"`` : `float`
            Continuous time representation of the first date in ``df``.
    """
    if regressor_cols is None:
        regressor_cols = []

    # Defines ``fit_df``, the data available for fitting the model
    # and its time column (in `datetime.datetime` format)
    canonical_data_dict = get_canonical_data(
        df=df,
        time_col=time_col,
        value_col=value_col,
        freq=freq,
        date_format=date_format,
        train_end_date=train_end_date,
        regressor_cols=regressor_cols,
        lagged_regressor_cols=lagged_regressor_cols)
    fit_df = canonical_data_dict["fit_df"]

    # Calculates basic time properties
    train_start = fit_df[TIME_COL].min()
    start_year = int(train_start.strftime("%Y"))
    origin_for_time_vars = get_default_origin_for_time_vars(fit_df, TIME_COL)
    period = min_gap_in_seconds(df=fit_df, time_col=TIME_COL)
    simple_freq = get_simple_time_frequency_from_period(period)
    num_training_points = fit_df.shape[0]

    # Calculates number of (fractional) days in the training set
    time_delta = fit_df[TIME_COL].max() - train_start
    num_training_days = (
            time_delta.days
            + (time_delta.seconds + period) / TimeEnum.ONE_DAY_IN_SECONDS.value)

    # Calculates forecast horizon (as a number of periods)
    if forecast_horizon is None:
        # expected to be kept in sync with default value set in ``get_default_time_parameters``
        forecast_horizon = get_default_horizon_from_period(
            period=period,
            num_observations=num_training_points)

    days_per_observation = period / TimeEnum.ONE_DAY_IN_SECONDS.value
    forecast_horizon_in_days = forecast_horizon * days_per_observation
    forecast_horizon_in_timedelta = timedelta(days=forecast_horizon_in_days)

    # Calculates forecast end year
    train_end = fit_df[TIME_COL].max()
    days_to_forecast = math.ceil(forecast_horizon * days_per_observation)
    future_end = train_end + timedelta(days=days_to_forecast)
    end_year = int(future_end.strftime("%Y"))

    return {
        "period": period,
        "simple_freq": simple_freq,
        "num_training_points": num_training_points,
        "num_training_days": num_training_days,
        "days_per_observation": days_per_observation,
        "forecast_horizon": forecast_horizon,
        "forecast_horizon_in_timedelta": forecast_horizon_in_timedelta,
        "forecast_horizon_in_days": forecast_horizon_in_days,
        "start_year": start_year,
        "end_year": end_year,
        "origin_for_time_vars": origin_for_time_vars
    }
