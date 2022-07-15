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
"""Preprocessing function to handle anomalies in input data
before forecasting.
"""

import warnings

import numpy as np
import pandas as pd

from greykite.common.constants import ANOMALY_COL
from greykite.common.constants import END_TIME_COL
from greykite.common.constants import START_TIME_COL


def adjust_anomalous_data(
        df,
        time_col,
        value_col,
        anomaly_df,
        start_time_col=START_TIME_COL,
        end_time_col=END_TIME_COL,
        adjustment_delta_col=None,
        filter_by_dict=None,
        filter_by_value_col=None,
        adjustment_method="add"):
    """This function takes:

        - a time series, in the form of a dataframe: ``df``
        - the anomaly information, in the form of a dataframe: ``anomaly_df``.

    It then adjusts the values of the time series based on the perceived impact
    of the anomalies given in the column ``adjustment_delta_col`` and
    assigns `np.nan` if the impact is not given.

    Note that ``anomaly_df`` can contain the anomaly information for many
    different timeseries. This is enabled by allowing multiple metrics and
    dimensions to be listed in the same anomaly dataframe. Columns can indicate
    the metric name and dimension value.

    This function first subsets the ``anomaly_df`` to the relevant rows for the
    ``value_col`` as specified by ``filter_by_dict``, then makes the specified
    adjustments to ``df``.

    Parameters
    ----------
    df : `pandas.DataFrame`
        A data frame which includes the timestamp column
        as well as the value column.
    time_col : `str`
        The column name in ``df`` representing time for the time series data.
        The time column can be anything that can be parsed by `pandas.DatetimeIndex`.
    value_col: `str`
        The column name which has the value of interest to be forecasted.
    anomaly_df : `pandas.DataFrame`
        A dataframe which includes the anomaly information for
        the input series (``df``) but potentially for multiple series and dimensions.

        This dataframe must include these two columns:

            - ``start_time_col``
            - ``end_time_col``

        and include

             - ``adjustment_delta_col`` if it is not None in the function call.

         Moreover if dimensions are requested by passing the
         ``filter_by_dict`` argument (not None), all of this dictionary
         keys must also appear in ``anomaly_df``.

        Here is an example::

            anomaly_df = pd.DataFrame({
                "start_time": ["1/1/2018", "1/4/2018", "1/8/2018", "1/10/2018"],
                "end_time": ["1/2/2018", "1/6/2018", "1/9/2018", "1/10/2018"],
                "adjustment_delta": [np.nan, 3, -5, np.nan],
                # extra columns for filtering
                "metric": ["y", "y", "z", "z"],
                "platform": ["MOBILE", "MOBILE", "DESKTOP", "DESKTOP"],
                "vertical": ["ads", "sales", "ads", "ads"],
            })

        In the above example,

            - "start_time" is the start date of the anomaly, which is provided using the argument ``start_time_col``.
            - "end_time" is the end date of the anomaly, which is provided using the argument ``end_time_col``.
            - "adjustment_delta" is the column which includes the delta if it is known. The name of this
              column is provided using the argument ``adjustment_delta_col``. Use `numpy.nan` if the
              adjustment size is not known, and the adjusted value will be set to `numpy.nan`.
            - "metric", "platform", and "vertical" are example columns for filtering. They
              contain the metric name and dimensions for which the anomaly is applicable.
              ``filter_by_dict` is used to filter on these columns to get the relevant
              anomalies for the timeseries represented by ``df[value_col]``.

    start_time_col : `str`, default ``START_TIME_COL``
        The column name in ``anomaly_df`` representing the start timestamp of
        the anomalous period, inclusive.
        The format can be anything that can be parsed by pandas DatetimeIndex.
    end_time_col : `str`, default ``END_TIME_COL``
        The column name in anomaly_df representing the start timestamp of
        the anomalous period, inclusive.
        The format can be anything that can be parsed by pandas DatetimeIndex.
    adjustment_delta_col : `str` or None, default None
        The column name in ``anomaly_df`` for the impact delta of the anomalies
        on the values of the series.

        If the value is available, it will be used to adjust the timeseries values
        in the given period by adding or subtracting this value to the raw series
        values in that period. Whether to add or subtract is specified by
        ``adjustment_method``.
        If the value for a row is "" or np.nan, the adjusted value is set to np.nan.

        If ``adjustment_delta_col`` is None, all adjusted values are set to np.nan.
    filter_by_dict : `dict` [`str`, `any`] or None, default None
        A dictionary whose keys are column names of ``anomaly_df``,
        and values are the desired value for that column (e.g. a string or int).
        If the value is an iterable (list, tuple, set), then it enumerates
        all allowed values for that column.

        This dictionary is used to filter ``anomaly_df`` to the matching anomalies.
        This helps when the ``anomaly_df`` includes the anomalies for various metrics
        and dimensions, so matching is needed to get the relevant anomalies for ``df``.

        Columns in ``anomaly_df`` can contain information on metric name,
        metric dimension (e.g. mobile/desktop), issue severity, etc. for filtering.
    filter_by_value_col: `str` or None, default None
        If provided, ``{filter_by_value_col: value_col}`` is added to ``filter_by_dict``
        for filtering. This filters ``anomaly_df`` to rows where
        ``anomaly_df[filter_by_value_col] == value_col``.

        If ``value_col`` is the metric name, this is a convenient way to find anomalies
        matching the metric name.
    adjustment_method : `str` ("add" or "subtract"), default "add"
        How the adjustment in ``anomaly_df`` should be used to adjust
        the value in ``df``.

            - If "add", the value in ``adjustment_delta_col`` is added to the original value.
            - If "subtract", it is subtracted from the original value.

    Returns
    -------
    Result : `dict`
        A dictionary with the following items (specified by key):

        - "adjusted_df": `pandas.DataFrame`
            A dataframe identical to the input dataframe ``df``, but with
            ``value_col`` updated to the adjusted values.
        - "augmented_df": `pandas.DataFrame`
            A dataframe identical to the input dataframe ``df``, with
            two extra columns

            - ANOMALY_COL: Anomaly labels for the time series.
            1 and 0 indicates anomalous and non-anomalous points, respectively.
            - ``f"adjusted_{value_col}"``: Adjusted values.

            ``value_col`` retains the original values.
            This is useful to inspect which values have changed.
    """
    df = df.copy()
    augmented_df = df.copy()
    augmented_df[ANOMALY_COL] = 0
    anomaly_df = anomaly_df.copy()
    new_value_col = f"adjusted_{value_col}"

    if new_value_col in df.columns:
        raise ValueError(
            f"`df` cannot include this column name: {new_value_col}."
            f"This is because {new_value_col} will be autogenerated to include"
            "adjusted values in `augmented_df`.")
    if adjustment_method not in ["add", "subtract"]:
        raise ValueError(f"`adjustment_method` '{adjustment_method}' is not recognized, "
                         f"must be one of ['add', 'subtract']")
    if adjustment_delta_col is not None and adjustment_method == "subtract":
        anomaly_df[adjustment_delta_col] *= -1.0

    if filter_by_value_col is not None:
        if filter_by_dict is not None:
            filter_by_dict[filter_by_value_col] = value_col
        else:
            filter_by_dict = {filter_by_value_col: value_col}
    # If `filter_by_dict` is passed, we subset the anomaly dataframe
    # to filter on additional column values as specified.
    # Allows columns in `anomaly_df` indicating the metric name and dimension value
    # to be used to filter the rows.
    if filter_by_dict is not None:
        relevant_rows = pd.Series(np.repeat(True, anomaly_df.shape[0]))
        for filter_col, allowed_value in filter_by_dict.items():
            if filter_col not in anomaly_df.columns:
                raise ValueError(
                    f"Column '{filter_col}' was requested by `filter_by_dict`, "
                    "but the corresponding column name in `anomaly_df` is not found.")
            if isinstance(allowed_value, (tuple, list, set)):
                relevant_rows &= anomaly_df[filter_col].isin(allowed_value)
            else:
                relevant_rows &= (anomaly_df[filter_col] == allowed_value)
        anomaly_df = anomaly_df[relevant_rows]

    # Adjusts the values of the series using the anomaly information given
    # in the i-th column of the ``anomaly_df`` (after all the subsetting is done).
    augmented_df[new_value_col] = augmented_df[value_col]
    try:
        time_values = pd.to_datetime(augmented_df[time_col])
        anomaly_df[start_time_col] = pd.to_datetime(anomaly_df[start_time_col])
        anomaly_df[end_time_col] = pd.to_datetime(anomaly_df[end_time_col])
    except Exception as e:
        warnings.warn(f"Dates could not be parsed by `pandas.to_datetime`, using string comparison "
                      f"for dates instead. Error message:\n{e}")
        time_values = augmented_df[time_col].astype(str)
        anomaly_df[start_time_col] = anomaly_df[start_time_col].astype(str)
        anomaly_df[end_time_col] = anomaly_df[end_time_col].astype(str)
    for i in range(anomaly_df.shape[0]):
        row = anomaly_df.iloc[i]
        t1 = row[start_time_col]
        t2 = row[end_time_col]
        if t2 < t1:
            raise ValueError(
                f"End Time: {t2} cannot be before Start Time: {t1}, in ``anomaly_df``.")
        bool_index = (time_values >= t1) & (time_values <= t2)
        augmented_df.loc[bool_index, ANOMALY_COL] = 1

        if adjustment_delta_col is not None:
            delta = row[adjustment_delta_col]
            if (delta != "") and not np.isnan(delta):
                augmented_df.loc[bool_index, new_value_col] = (
                        augmented_df.loc[bool_index, new_value_col] + delta)
            else:
                augmented_df.loc[bool_index, new_value_col] = np.nan
        else:
            augmented_df.loc[bool_index, new_value_col] = np.nan

    # Rewrites `df` values with the adjusted values
    df[value_col] = augmented_df[new_value_col]

    return {
        "adjusted_df": df,
        "augmented_df": augmented_df}
