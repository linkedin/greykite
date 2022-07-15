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
# original author: Reza Hosseini
"""Functions that extract timeseries properties."""

import datetime
import warnings
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd

from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.features.adjust_anomalous_data import adjust_anomalous_data
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message


def describe_timeseries(df, time_col):
    """Checks if a time series consists of equal time increments
    and if it is in increasing order.

    :param df: data.frame which includes the time column in datetime format
    :param time_col: time column

    :return: a dictionary with following items

        - dataframe ("df") with added delta columns
        - "regular_increments" booleans to see if time deltas as the same
        - "increasing" a boolean to denote if the time increments are increasing
        - "min_timestamp": minimum timestamp in data
        - "max_timestamp": maximum timestamp in data
        - "mean_increment_secs": mean of increments in seconds
        - "min_increment_secs": minimum of increments in seconds
        - "median_increment_secs": median of increments in seconds
        - "mean_delta": mean of time increments
        - "min_delta": min of the time increments
        - "max_delta": max of the time increments
        - "median_delta": median of the time increments
        - "freq_in_secs": the frequency of the timeseries in seconds which is defined to be the median time-gap
        - "freq_in_days": the frequency of the timeseries in days which is defined to be the median time-gap
        - "freq_in_timedelta": the frequency of the timeseries in `datetime.timedelta` which is defined to be the median time-gap

    """
    df = df.copy(deep=True)
    if df.shape[0] < 2:
        raise Exception("dataframe needs to have at least two rows")
    df["delta"] = (
            df[time_col] - df[time_col].shift()).fillna(pd.Timedelta(seconds=0))
    df["delta_sec"] = df["delta"].values / np.timedelta64(1, "s")

    delta_sec = df["delta_sec"][1:]
    mean_increment_secs = np.mean(delta_sec)
    min_increment_secs = min(delta_sec)
    median_increment_secs = np.median(delta_sec)
    regular_increments = (max(delta_sec) == min(delta_sec))
    increasing = min(delta_sec > 0)
    min_timestamp = df[time_col].min()
    max_timestamp = df[time_col].max()

    delta = df["delta"][1:]
    min_delta = min(delta)
    max_delta = max(delta)
    mean_delta = np.mean(delta)
    median_delta = np.median(delta)

    # The frequency is defined by the median time-gap
    freq_in_secs = median_increment_secs
    freq_in_days = freq_in_secs / (24 * 3600)
    freq_in_timedelta = datetime.timedelta(days=freq_in_days)

    return {
            "df": df,
            "regular_increments": regular_increments,
            "increasing": increasing,
            "min_timestamp": min_timestamp,
            "max_timestamp": max_timestamp,
            "mean_increment_secs": mean_increment_secs,
            "min_increment_secs": min_increment_secs,
            "median_increment_secs": median_increment_secs,
            "mean_delta": mean_delta,
            "median_delta": median_delta,
            "min_delta": min_delta,
            "max_delta": max_delta,
            "freq_in_secs": freq_in_secs,
            "freq_in_days": freq_in_days,
            "freq_in_timedelta": freq_in_timedelta
            }


def min_gap_in_seconds(df, time_col):
    """Returns the smallest gap between observations in df[time_col].

    Assumes df[time_col] is sorted in ascending order without duplicates.

    :param df: pd.DataFrame
        input timeseries
    :param time_col: str
        time column name in `df`
    :return: float
        minimum gap between observations, in seconds
    """
    if df.shape[0] < 2:
        raise ValueError(f"Must provide at least two data points. Found {df.shape[0]}.")
    timestamps = pd.to_datetime(df[time_col])
    period = (timestamps - timestamps.shift()).min()
    return period.days*24*3600 + period.seconds


def find_missing_dates(timestamp_series):
    """Identifies any gaps in pandas Series containing timestamps
    Timestamps are assumed to be in sorted order
    :return: pd.DataFrame with dates of the gaps in the time series, if any
    """
    timestamp_series = timestamp_series.sort_values(ascending=True, inplace=False)  # don't modify original
    delta = timestamp_series.diff()[1:].fillna(pd.Timedelta(seconds=0))
    delta_sec = delta.values / np.timedelta64(1, "s")
    min_gap = min(delta_sec)

    # dates surrounding the gap
    gap_start_points = timestamp_series[:-1][delta_sec > min_gap].reset_index(drop=True)
    gap_end_points = timestamp_series[1:][delta_sec > min_gap].reset_index(drop=True)
    # size of the gaps, in periods
    gap_periods = pd.Series((delta_sec[delta_sec > min_gap] / min_gap) - 1)
    gaps = pd.concat([gap_start_points, gap_end_points, gap_periods], axis=1, ignore_index=True)
    gaps.columns = ["right_before_gap", "right_after_gap", "gap_size"]
    return gaps


def fill_missing_dates(df, time_col=TIME_COL, freq=None):
    """Looks for gaps in df[time_col] and returns a pandas.DataFrame
        with the missing rows added in.
        Warning: if freq doesn't match intended freq, then values may be removed.

    Parameters
    ----------
    df : `pandas.DataFrame`
        dataframe with column ``time_col``
    time_col: `str`
        time column name, default TIME_COL
    freq: `str`
        timeseries frequency,
        DateOffset alias, default None (automatically inferred)

    Returns
    -------
    full_df : `pandas.DataFrame`
        ``df`` with rows added for missing timestamps
    added_timepoints : `int`
        The number of rows added to ``df``
    dropped_timepoints : `int`
        The number of rows removed from ``df``.
        If the timestamps in ``df`` are not evenly spaced,
        irregular timestamps may be removed.
    """
    freq = freq if freq is not None else pd.infer_freq(df[time_col])
    df = df.reset_index(drop=True)
    complete_dates = pd.DataFrame({
        time_col: pd.date_range(
            start=min(df[time_col]),
            end=max(df[time_col]),
            freq=freq)
    })
    full_df = pd.merge(complete_dates, df, how="left", on=time_col)

    # counts the timestamps in one but not the other
    before = set(df[time_col].values)
    after = set(full_df[time_col].values)
    added_timepoints = len(after - before)
    dropped_timepoints = len(before - after)
    if added_timepoints > 0:
        log_message(f"Added {added_timepoints} missing dates. There were {len(before)} values originally.",
                    LoggingLevelEnum.INFO)
    if dropped_timepoints > 0:
        warnings.warn(f"Dropped {dropped_timepoints} dates when filling gaps in input data. Provide data frequency"
                      f" and make sure data points are evenly spaced.")

    return full_df, added_timepoints, dropped_timepoints


def get_canonical_data(
        df: pd.DataFrame,
        time_col: str = TIME_COL,
        value_col: str = VALUE_COL,
        freq: str = None,
        date_format: str = None,
        tz: str = None,
        train_end_date: datetime = None,
        regressor_cols: List[str] = None,
        lagged_regressor_cols: List[str] = None,
        anomaly_info: Optional[Union[Dict, List[Dict]]] = None):
    """Loads data to internal representation. Parses date column,
    sets timezone aware index.
    Checks for irregularities and raises an error if input is invalid.
    Adjusts for anomalies according to ``anomaly_info``.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Input timeseries. A data frame which includes the timestamp column
        as well as the value column.
    time_col : `str`
        The column name in ``df`` representing time for the time series data.
        The time column can be anything that can be parsed by pandas DatetimeIndex.
    value_col: `str`
        The column name which has the value of interest to be forecasted.
    freq : `str` or None, default None
        Timeseries frequency, DateOffset alias, If None automatically inferred.
    date_format : `str` or None, default None
        strftime format to parse time column, eg ``%m/%d/%Y``.
        Note that ``%f`` will parse all the way up to nanoseconds.
        If None (recommended), inferred by `pandas.to_datetime`.
    tz : `str` or pytz.timezone object or None, default None
        Passed to `pandas.tz_localize` to localize the timestamp.
    train_end_date : `datetime.datetime` or None, default None
        Last date to use for fitting the model. Forecasts are generated after this date.
        If None, it is set to the minimum of ``self.last_date_for_val`` and
        ``self.last_date_for_reg``.
    regressor_cols: `list` [`str`] or None, default None
        A list of regressor columns used in the training and prediction DataFrames.
        If None, no regressor columns are used.
        Regressor columns that are unavailable in ``df`` are dropped.git
    lagged_regressor_cols: `list` [`str`] or None, default None
        A list of additional columns needed for lagged regressors in thggede training and prediction DataFrames.
        This list can have overlap with ``regressor_cols``.
        If None, no additional columns are added to the DataFrame.
        Lagged regressor columns that are unavailable in ``df`` are dropped.
    anomaly_info : `dict` or `list` [`dict`] or None, default None
        Anomaly adjustment info. Anomalies in ``df``
        are corrected before any forecasting is done.

        If None, no adjustments are made.

        A dictionary containing the parameters to
        `~greykite.common.features.adjust_anomalous_data.adjust_anomalous_data`.
        See that function for details.
        The possible keys are:

            ``"value_col"`` : `str`
                The name of the column in ``df`` to adjust. You may adjust the value
                to forecast as well as any numeric regressors.
            ``"anomaly_df"`` : `pandas.DataFrame`
                Adjustments to correct the anomalies.
            ``"start_time_col"``: `str`, default START_TIME_COL
                Start date column in ``anomaly_df``.
            ``"end_time_col"``: `str`, default END_TIME_COL
                End date column in ``anomaly_df``.
            ``"adjustment_delta_col"``: `str` or None, default None
                Impact column in ``anomaly_df``.
            ``"filter_by_dict"``: `dict` or None, default None
                Used to filter ``anomaly_df`` to the relevant anomalies for
                the ``value_col`` in this dictionary.
                Key specifies the column name, value specifies the filter value.
            ``"filter_by_value_col""``: `str` or None, default None
                Adds ``{filter_by_value_col: value_col}`` to ``filter_by_dict``
                if not None, for the ``value_col`` in this dictionary.
            ``"adjustment_method"`` : `str` ("add" or "subtract"), default "add"
                How to make the adjustment, if ``adjustment_delta_col`` is provided.

        Accepts a list of such dictionaries to adjust multiple columns in ``df``.

    Returns
    -------
    canonical_data_dict : `dict`
        Dictionary containing the dataset in canonical form, and information such as
        train end date. Keys:

            ``"df"`` : `pandas.DataFrame`
                Data frame containing timestamp and value, with standardized column names for internal use
                (TIME_COL, VALUE_COL). Rows are sorted by time index, and missing gaps between dates are filled
                in so that dates are spaced at regular intervals. Values are adjusted for anomalies
                according to ``anomaly_info``.
                The index can be timezone aware (but TIME_COL is not).
            ``"df_before_adjustment"`` : `pandas.DataFrame` or None
                ``df`` before adjustment by ``anomaly_info``.
                If ``anomaly_info`` is None, this is None.
            ``"fit_df"`` : `pandas.DataFrame`
                A subset of the returned ``df``, with data up until ``train_end_date``.
            ``"freq"`` : `pandas.DataFrame`
                timeseries frequency, inferred if not provided
            ``"time_stats"`` : `dict`
                Information about the time column:

                    ``"gaps"``: missing_dates
                    ``"added_timepoints"``: added_timepoints
                    ``"dropped_timepoints"``: dropped_timepoints

            ``"regressor_cols"`` : `list` [`str`]
                A list of regressor columns.
             ``"lagged_regressor_cols"`` : `list` [`str`]
                A list of lagged regressor columns.
            ``"fit_cols"`` : `list` [`str`]
                Names of time column, value column, regressor columns, and lagged regressor columns.
            ``"train_end_date"`` : `datetime.datetime`
                Last date or timestamp for training. It is always less than or equal to
                minimum non-null values of ``last_date_for_val`` and ``last_date_for_reg``.
            ``"last_date_for_val"`` : `datetime.datetime`
                Date or timestamp corresponding  to last non-null value in ``df[value_col]``.
            ``"last_date_for_reg"`` : `datetime.datetime` or None
                Date or timestamp corresponding to last non-null value in ``df[regressor_cols]``.
                If ``regressor_cols`` is None, ``last_date_for_reg`` is None.
            ``"last_date_for_lag_reg"`` : `datetime.datetime` or None
                Date or timestamp corresponding to last non-null value in ``df[lagged_regressor_cols]``.
                If ``lagged_regressor_cols`` is None, ``last_date_for_lag_reg`` is None.
    """
    if time_col not in df.columns:
        raise ValueError(f"{time_col} column is not in input data")
    if value_col not in df.columns:
        raise ValueError(f"{value_col} column is not in input data")
    if df.shape[0] <= 2:
        raise ValueError(
            f"Time series has < 3 observations. More data are needed for forecasting.")

    # Standardizes the time column name.
    # `value_col` is standardized after anomalies are adjusted.
    df_standardized = df.rename({
        time_col: TIME_COL,
    }, axis=1)
    df_standardized[TIME_COL] = pd.to_datetime(
        df_standardized[TIME_COL],
        format=date_format,
        infer_datetime_format=True)
    # Drops data points from duplicate time stamps
    df_standardized.drop_duplicates(
        subset=[TIME_COL],
        keep='first',
        inplace=True)
    if df.shape[0] > df_standardized.shape[0]:
        warnings.warn(
            f"Duplicate timestamps have been removed.",
            UserWarning)
    df = df_standardized.sort_values(by=TIME_COL)
    # Infers data frequency
    inferred_freq = pd.infer_freq(df[TIME_COL])
    if freq is None:
        freq = inferred_freq
    elif inferred_freq is not None and freq != inferred_freq:
        warnings.warn(
            f"Provided frequency '{freq}' does not match inferred frequency '{inferred_freq}'."
            f" Using '{freq}'.", UserWarning)  # NB: with missing data, it's better to provide freq
    # Handles gaps in time series
    missing_dates = find_missing_dates(df[TIME_COL])
    df, added_timepoints, dropped_timepoints = fill_missing_dates(
        df,
        time_col=TIME_COL,
        freq=freq)
    time_stats = {
        "gaps": missing_dates,
        "added_timepoints": added_timepoints,
        "dropped_timepoints": dropped_timepoints
    }
    # Creates index with localized timestamp
    df.index = df[TIME_COL]
    df.index.name = None
    if tz is not None:
        df = df.tz_localize(tz)

    df_before_adjustment = None
    if anomaly_info is not None:
        # Saves values before adjustment.
        df_before_adjustment = df.copy()
        # Adjusts columns in df (e.g. `value_col`, `regressor_cols`)
        # using the anomaly info. One dictionary of parameters
        # for `adjust_anomalous_data` is provided for each column to adjust.
        if not isinstance(anomaly_info, (list, tuple)):
            anomaly_info = [anomaly_info]
        for single_anomaly_info in anomaly_info:
            adjusted_df_dict = adjust_anomalous_data(
                df=df,
                time_col=TIME_COL,
                **single_anomaly_info)
            # `self.df` with values for single_anomaly_info["value_col"] adjusted.
            df = adjusted_df_dict["adjusted_df"]

        # Standardizes `value_col` name
        df_before_adjustment.rename({
            value_col: VALUE_COL
        }, axis=1, inplace=True)
    # Standardizes `value_col` name
    df.rename({
        value_col: VALUE_COL
    }, axis=1, inplace=True)

    # Finds date of last available value
    last_date_available = df[TIME_COL].max()
    last_date_for_val = df[df[VALUE_COL].notnull()][TIME_COL].max()
    last_date_for_reg = None
    if regressor_cols:
        available_regressor_cols = [col for col in df.columns if col not in [TIME_COL, VALUE_COL]]
        cols_not_selected = set(regressor_cols) - set(available_regressor_cols)
        regressor_cols = [col for col in regressor_cols if col in available_regressor_cols]
        if cols_not_selected:
            warnings.warn(f"The following columns are not available to use as "
                          f"regressors: {sorted(cols_not_selected)}")
        last_date_for_reg = df[df[regressor_cols].notnull().any(axis=1)][TIME_COL].max()
        max_train_end_date = min(last_date_for_val, last_date_for_reg)
    else:
        regressor_cols = []
        max_train_end_date = last_date_for_val

    last_date_for_lag_reg = None
    if lagged_regressor_cols:
        available_regressor_cols = [col for col in df.columns if col not in [TIME_COL, VALUE_COL]]
        cols_not_selected = set(lagged_regressor_cols) - set(available_regressor_cols)
        lagged_regressor_cols = [col for col in lagged_regressor_cols if col in available_regressor_cols]
        if cols_not_selected:
            warnings.warn(f"The following columns are not available to use as "
                          f"lagged regressors: {sorted(cols_not_selected)}")
        last_date_for_lag_reg = df[df[lagged_regressor_cols].notnull().any(axis=1)][TIME_COL].max()
    else:
        lagged_regressor_cols = []

    # Chooses appropriate train_end_date
    if train_end_date is None:
        train_end_date = max_train_end_date
        if train_end_date < last_date_available:
            warnings.warn(
                f"{value_col} column of the provided TimeSeries contains "
                f"null values at the end. Setting 'train_end_date' to the last timestamp with a "
                f"non-null value ({train_end_date}).",
                UserWarning)
    elif train_end_date > max_train_end_date:
        warnings.warn(
            f"Input timestamp for the parameter 'train_end_date' "
            f"({train_end_date}) either exceeds the last available timestamp or"
            f"{value_col} column of the provided TimeSeries contains null "
            f"values at the end. Setting 'train_end_date' to the last timestamp with a "
            f"non-null value ({max_train_end_date}).",
            UserWarning)
        train_end_date = max_train_end_date

    extra_reg_cols = [col for col in df.columns if col not in regressor_cols and col in lagged_regressor_cols]
    fit_cols = [TIME_COL, VALUE_COL] + regressor_cols + extra_reg_cols
    fit_df = df[df[TIME_COL] <= train_end_date][fit_cols]

    return {
        "df": df,
        "df_before_adjustment": df_before_adjustment,
        "fit_df": fit_df,
        "freq": freq,
        "time_stats": time_stats,
        "regressor_cols": regressor_cols,
        "lagged_regressor_cols": lagged_regressor_cols,
        "fit_cols": fit_cols,
        "train_end_date": train_end_date,
        "last_date_for_val": last_date_for_val,
        "last_date_for_reg": last_date_for_reg,
        "last_date_for_lag_reg": last_date_for_lag_reg,
    }
