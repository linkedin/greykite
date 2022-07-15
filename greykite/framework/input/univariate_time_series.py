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
"""Input timeseries."""

import warnings
from datetime import datetime
from functools import partial
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd

from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message
from greykite.common.time_properties import describe_timeseries
from greykite.common.time_properties import get_canonical_data
from greykite.common.viz.timeseries_plotting import add_groupby_column
from greykite.common.viz.timeseries_plotting import flexible_grouping_evaluation
from greykite.common.viz.timeseries_plotting import grouping_evaluation
from greykite.common.viz.timeseries_plotting import plot_multivariate
from greykite.common.viz.timeseries_plotting import plot_univariate
from greykite.framework.constants import MEAN_COL_GROUP
from greykite.framework.constants import OVERLAY_COL_GROUP
from greykite.framework.constants import QUANTILE_COL_GROUP


class UnivariateTimeSeries:
    """Defines univariate time series input. The dataset can include regressors,
    but only one metric is designated as the target metric to forecast.

    Loads time series into a standard format. Provides statistics, plotting
    functions, and ability to generate future dataframe for prediction.

    Attributes
    ----------
    df: `pandas.DataFrame`
        Data frame containing timestamp and value, with standardized column names for internal use
        (TIME_COL, VALUE_COL). Rows are sorted by time index, and missing gaps between dates are filled
        in so that dates are spaced at regular intervals. Values are adjusted for anomalies
        according to ``anomaly_info``.
        The index can be timezone aware (but TIME_COL is not).
    y: `pandas.Series`, dtype float64
        Value of time series to forecast.
    time_stats: `dict`
        Summary statistics about the timestamp column.
    value_stats: `dict`
        Summary statistics about the value column.
    original_time_col: `str`
        Name of time column in original input data.
    original_value_col: `str`
        Name of value column in original input data.
    regressor_cols: `list` [`str`]
        A list of regressor columns in the training and prediction DataFrames.
    lagged_regressor_cols: `list` [`str`]
        A list of additional columns needed for lagged regressors in the training and prediction DataFrames.
    last_date_for_val: `datetime.datetime` or None, default None
        Date or timestamp corresponding  to last non-null value in ``df[original_value_col]``.
    last_date_for_reg: `datetime.datetime` or None, default None
        Date or timestamp corresponding to last non-null value in ``df[regressor_cols]``.
        If ``regressor_cols`` is None, ``last_date_for_reg`` is None.
    last_date_for_lag_reg: `datetime.datetime` or None, default None
        Date or timestamp corresponding to last non-null value in ``df[lagged_regressor_cols]``.
        If ``lagged_regressor_cols`` is None, ``last_date_for_lag_reg`` is None.
    train_end_date: `datetime.datetime`
        Last date or timestamp in ``fit_df``. It is always less than or equal to
        minimum non-null values of ``last_date_for_val`` and ``last_date_for_reg``.
    fit_cols: `list` [`str`]
        A list of columns used in the training and prediction DataFrames.
    fit_df: `pandas.DataFrame`
        Data frame containing timestamp and value, with standardized column names for internal use.
        Will be used for fitting (train, cv, backtest).
    fit_y: `pandas.Series`, dtype float64
        Value of time series for fit_df.
    freq: `str`
        timeseries frequency, DateOffset alias, e.g. {'T' (minute), 'H', D', 'W', 'M' (month end), 'MS' (month start),
        'Y' (year end), 'Y' (year start)}
        See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    anomaly_info : `dict` or `list` [`dict`] or None, default None
        Anomaly adjustment info. Anomalies in ``df``
        are corrected before any forecasting is done.
        See ``self.load_data()``
    df_before_adjustment : `pandas.DataFrame` or None, default None
        ``self.df`` before adjustment by ``anomaly_info``.
        Used by ``self.plot()`` to show the adjustment.
    """
    def __init__(self) -> None:
        self.df: Optional[pd.DataFrame] = None
        self.y: Optional[pd.Series] = None
        self.time_stats: Optional[Dict] = None
        self.value_stats: Optional[Dict] = None
        self.original_time_col: Optional[str] = None
        self.original_value_col: Optional[str] = None
        self.regressor_cols: List[str] = []
        self.lagged_regressor_cols: List[str] = []
        self.last_date_for_val: Optional[datetime] = None
        self.last_date_for_reg: Optional[datetime] = None
        self.last_date_for_lag_reg: Optional[datetime] = None
        self.train_end_date: Optional[str] = None
        self.fit_cols: List[str] = []
        self.fit_df: Optional[pd.DataFrame] = None
        self.fit_y: Optional[pd.DataFrame] = None
        self.freq: Optional[str] = None
        self.anomaly_info: Optional[Union[Dict, List[Dict]]] = None
        self.df_before_adjustment: Optional[pd.DataFrame] = None

    def load_data(
            self,
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
            Regressor columns that are unavailable in ``df`` are dropped.
        lagged_regressor_cols: `list` [`str`] or None, default None
            A list of additional columns needed for lagged regressors in the training and prediction DataFrames.
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
        self : Returns self.
            Sets ``self.df`` with standard column names,
            value adjusted for anomalies, and time gaps filled in,
            sorted by time index.
        """
        self.original_time_col = time_col
        self.original_value_col = value_col
        self.anomaly_info = anomaly_info

        canonical_data_dict = get_canonical_data(
            df=df,
            time_col=time_col,
            value_col=value_col,
            freq=freq,
            date_format=date_format,
            tz=tz,
            train_end_date=train_end_date,
            regressor_cols=regressor_cols,
            lagged_regressor_cols=lagged_regressor_cols,
            anomaly_info=anomaly_info)
        self.df = canonical_data_dict["df"]
        self.df_before_adjustment = canonical_data_dict["df_before_adjustment"]
        self.fit_df = canonical_data_dict["fit_df"]
        self.freq = canonical_data_dict["freq"]
        self.time_stats = canonical_data_dict["time_stats"]
        self.regressor_cols = canonical_data_dict["regressor_cols"]
        self.lagged_regressor_cols = canonical_data_dict["lagged_regressor_cols"]
        self.fit_cols = canonical_data_dict["fit_cols"]
        self.train_end_date = canonical_data_dict["train_end_date"]
        self.last_date_for_val = canonical_data_dict["last_date_for_val"]
        self.last_date_for_reg = canonical_data_dict["last_date_for_reg"]
        self.last_date_for_lag_reg = canonical_data_dict["last_date_for_lag_reg"]

        # y (possibly with null values) after gaps have been filled in and anomalies corrected
        self.y = self.df[VALUE_COL]
        self.fit_y = self.fit_df[VALUE_COL]

        # computes statistics of processed dataset
        self.describe_time_col()
        self.describe_value_col()  # compute value statistics

        log_message(f"last date for fit: {self.train_end_date}", LoggingLevelEnum.INFO)
        log_message(f"last date for {self.original_value_col}: {self.last_date_for_val}", LoggingLevelEnum.INFO)
        log_message(f"last date with any regressor: {self.last_date_for_reg}", LoggingLevelEnum.INFO)
        log_message(f"columns available to use as regressors: {', '.join(self.regressor_cols)}", LoggingLevelEnum.INFO)
        log_message(f"columns available to use as lagged regressors: {', '.join(self.lagged_regressor_cols)}", LoggingLevelEnum.INFO)

        return self

    def describe_time_col(self):
        """Basic descriptive stats on the timeseries time column.

        Returns
        -------
        time_stats: `dict`
            Dictionary with descriptive stats on the timeseries time column.

                * data_points: int
                    number of time points
                * mean_increment_secs: float
                    mean frequency
                * min_timestamp: datetime64
                    start date
                * max_timestamp: datetime64
                    end date
        """
        if self.df is None:
            raise RuntimeError("Must load data before describing dataset")

        timeseries_info = describe_timeseries(df=self.df, time_col=TIME_COL)
        data_points = self.df.shape[0]
        mean_increment_secs = timeseries_info["mean_increment_secs"]
        min_timestamp = timeseries_info["min_timestamp"]
        max_timestamp = timeseries_info["max_timestamp"]

        log_message("Input time stats:", LoggingLevelEnum.INFO)
        log_message(f"  data points: {data_points}", LoggingLevelEnum.INFO)
        log_message(f"  avg increment (sec): {mean_increment_secs:.2f}", LoggingLevelEnum.INFO)
        log_message(f"  start date: {min_timestamp}", LoggingLevelEnum.INFO)
        log_message(f"  end date: {max_timestamp}", LoggingLevelEnum.INFO)

        time_stats = {
            "data_points": data_points,  # total number of time points, including missing ones
            "mean_increment_secs": mean_increment_secs,  # after filling in gaps
            "min_timestamp": min_timestamp,
            "max_timestamp": max_timestamp,
        }
        self.time_stats.update(time_stats)  # compute time statistics
        return time_stats

    def describe_value_col(self):
        """Basic descriptive stats on the timeseries value column.

        Returns
        -------
        value_stats : `dict` [`str`, `float`]
            Dict with keys: count, mean, std, min, 25%, 50%, 75%, max
        """
        if self.df is None:
            raise RuntimeError("Must load data before describing values")
        self.value_stats = self.df[VALUE_COL].describe()  # count is the total number of provided timepoints
        log_message("Input value stats:", LoggingLevelEnum.INFO)
        log_message(repr(self.value_stats), LoggingLevelEnum.INFO)
        return self.value_stats

    def make_future_dataframe(self, periods: int = None, include_history=True):
        """Extends the input data for prediction into the future.

        Includes the historical values (VALUE_COL) so this can be fed
        into a Pipeline that transforms input data for fitting, and for
        use in evaluation.

        Parameters
        ----------
        periods : int or None
            Number of periods to forecast.
            If there are no regressors, default is 30.
            If there are regressors, default is to predict all available dates.
        include_history : bool
            Whether to return historical dates and values with future dates.

        Returns
        -------
        future_df : `pandas.DataFrame`
            Dataframe with future timestamps for prediction.
            Contains columns for:

                * prediction dates (``TIME_COL``),
                * values (``VALUE_COL``),
                * optional regressors
        """
        if self.df is None:
            raise RuntimeError("Must load data before generating future dates.")

        # determines the number of future periods to predict
        if self.regressor_cols:
            max_regressor_periods = len(self.df[
                (self.df[TIME_COL] > self.train_end_date)
                & (self.df[TIME_COL] <= self.last_date_for_reg)
            ])
            if periods is None:
                periods = max_regressor_periods
            elif periods > max_regressor_periods:
                warnings.warn(
                    f"Provided periods '{periods}' is more than allowed ('{max_regressor_periods}') due to "
                    f"the length of regressor columns. Using '{max_regressor_periods}'.",
                    UserWarning)
                periods = max_regressor_periods
        elif periods is None:
            periods = 30

        # the future dates for prediction
        dates = pd.date_range(
            start=self.train_end_date,
            periods=periods + 1,  # an extra in case we include start
            freq=self.freq)
        dates = dates[dates > self.train_end_date]  # drops values up to train_end_date
        dates = dates[:periods]  # returns the correct number of periods

        if self.regressor_cols:
            # return TIME_COL, VALUE_COL, and regressors
            last_date_for_predict = dates.max()
            if include_history:
                valid_indices = (self.df[TIME_COL] <= last_date_for_predict)
            else:
                valid_indices = ((self.df[TIME_COL] > self.train_end_date)
                                 & (self.df[TIME_COL] <= last_date_for_predict))
            future_df = self.df[valid_indices]
        else:
            # return TIME_COL, VALUE_COL
            future_df = self.df.reindex(index=dates)
            future_df[TIME_COL] = future_df.index
            if include_history:
                future_df = pd.concat([self.fit_df, future_df], axis=0, sort=False)

        return future_df[self.fit_cols]

    def plot(
            self,
            color="rgb(32, 149, 212)",
            show_anomaly_adjustment=False,
            **kwargs):
        """Returns interactive plotly graph of the value against time.

        If anomaly info is provided, there is an option to show the anomaly adjustment.

        Parameters
        ----------
        color : `str`, default "rgb(32, 149, 212)" (light blue)
            Color of the value line (after adjustment, if applicable).
        show_anomaly_adjustment : `bool`, default False
            Whether to show the anomaly adjustment.
        kwargs : additional parameters
            Additional parameters to pass to
            `~greykite.common.viz.timeseries_plotting.plot_univariate`
            such as title and color.

        Returns
        -------
        fig : `plotly.graph_objects.Figure`
            Interactive plotly graph of the value against time.

            See `~greykite.common.viz.timeseries_plotting.plot_forecast_vs_actual`
            return value for how to plot the figure and add customization.
        """
        df = self.df.copy()
        # Plots value after anomaly adjustment
        y_col_style_dict = {
            VALUE_COL: dict(
                name=self.original_value_col,
                mode="lines",
                line=dict(
                    color=color,
                ),
                opacity=0.8
            )
        }
        if show_anomaly_adjustment:
            if self.anomaly_info is not None:
                # Adds value before adjustment to ``df``
                postfix = "_unadjusted"
                df[f"{VALUE_COL}{postfix}"] = self.df_before_adjustment[VALUE_COL]
                y_col_style_dict[f"{VALUE_COL}{postfix}"] = dict(
                    name=f"{self.original_value_col}{postfix}",
                    mode="lines",
                    line=dict(
                        color="#B3B3B3",  # light gray
                    ),
                    opacity=0.8
                )
            else:
                raise ValueError("There is no `anomaly_info` to show. `show_anomaly_adjustment` must be False.")
        return plot_multivariate(
            df,
            TIME_COL,
            y_col_style_dict,
            xlabel=self.original_time_col,
            ylabel=self.original_value_col,
            **kwargs)

    def get_grouping_evaluation(
            self,
            aggregation_func=np.nanmean,
            aggregation_func_name="mean",
            groupby_time_feature=None,
            groupby_sliding_window_size=None,
            groupby_custom_column=None):
        """Group-wise computation of aggregated timeSeries value.
        Can be used to evaluate error/ aggregated value by a time feature,
        over time, or by a user-provided column.

        Exactly one of: ``groupby_time_feature``, ``groupby_sliding_window_size``,
        ``groupby_custom_column`` must be provided.

        Parameters
        ----------
        aggregation_func : callable, optional, default ``numpy.nanmean``
            Function that aggregates an array to a number.
            Signature (y: array) -> aggregated value: float.
        aggregation_func_name : `str` or None, optional, default "mean"
            Name of grouping function, used to report results.
            If None, defaults to "aggregation".
        groupby_time_feature : `str` or None, optional
            If provided, groups by a column generated by
            `~greykite.common.features.timeseries_features.build_time_features_df`.
            See that function for valid values.
        groupby_sliding_window_size : `int` or None, optional
            If provided, sequentially partitions data into groups of size
            ``groupby_sliding_window_size``.
        groupby_custom_column : `pandas.Series` or None, optional
            If provided, groups by this column value. Should be same length as the DataFrame.

        Returns
        -------
        grouped_df : `pandas.DataFrame` with two columns:

            (1) grouping_func_name:
                evaluation metric for aggregation of timeseries.
            (2) group name:
                group name depends on the grouping method:
                ``groupby_time_feature`` for ``groupby_time_feature``
                ``cst.TIME_COL`` for ``groupby_sliding_window_size``
                ``groupby_custom_column.name`` for ``groupby_custom_column``.
        """
        df = self.df.copy()
        if aggregation_func_name:
            grouping_func_name = f"{aggregation_func_name} of {VALUE_COL}"
        else:
            grouping_func_name = f"aggregation of {VALUE_COL}"

        def grouping_func(grp):
            return aggregation_func(grp[VALUE_COL])

        result = add_groupby_column(
            df=df,
            time_col=TIME_COL,
            groupby_time_feature=groupby_time_feature,
            groupby_sliding_window_size=groupby_sliding_window_size,
            groupby_custom_column=groupby_custom_column)

        grouped_df = grouping_evaluation(
            df=result["df"],
            groupby_col=result["groupby_col"],
            grouping_func=grouping_func,
            grouping_func_name=grouping_func_name)
        return grouped_df

    def plot_grouping_evaluation(
            self,
            aggregation_func=np.nanmean,
            aggregation_func_name="mean",
            groupby_time_feature=None,
            groupby_sliding_window_size=None,
            groupby_custom_column=None,
            xlabel=None,
            ylabel=None,
            title=None):
        """Computes aggregated timeseries by group and plots the result.
        Can be used to plot aggregated timeseries by a time feature, over time,
        or by a user-provided column.

        Exactly one of: ``groupby_time_feature``, ``groupby_sliding_window_size``,
        ``groupby_custom_column`` must be provided.

        Parameters
        ----------
        aggregation_func : callable, optional, default ``numpy.nanmean``
            Function that aggregates an array to a number.
            Signature (y: array) -> aggregated value: float.
        aggregation_func_name : `str` or None, optional, default "mean"
            Name of grouping function, used to report results.
            If None, defaults to "aggregation".
        groupby_time_feature : `str` or None, optional
            If provided, groups by a column generated by
            `~greykite.common.features.timeseries_features.build_time_features_df`.
            See that function for valid values.
        groupby_sliding_window_size : `int` or None, optional
            If provided, sequentially partitions data into groups of size
            ``groupby_sliding_window_size``.
        groupby_custom_column : `pandas.Series` or None, optional
            If provided, groups by this column value. Should be same length as the DataFrame.
        xlabel : `str`, optional, default None
            X-axis label of the plot.
        ylabel : `str`, optional, default None
            Y-axis label of the plot.
        title : `str` or None, optional
            Plot title. If None, default is based on axis labels.

        Returns
        -------
        fig : `plotly.graph_objects.Figure`
            plotly graph object showing aggregated timeseries by group.
            x-axis label depends on the grouping method:
            ``groupby_time_feature`` for ``groupby_time_feature``
            ``TIME_COL`` for ``groupby_sliding_window_size``
            ``groupby_custom_column.name`` for ``groupby_custom_column``.
        """
        grouped_df = self.get_grouping_evaluation(
            aggregation_func=aggregation_func,
            aggregation_func_name=aggregation_func_name,
            groupby_time_feature=groupby_time_feature,
            groupby_sliding_window_size=groupby_sliding_window_size,
            groupby_custom_column=groupby_custom_column)

        xcol, ycol = grouped_df.columns
        fig = plot_univariate(
            df=grouped_df,
            x_col=xcol,
            y_col=ycol,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title)

        return fig

    def get_quantiles_and_overlays(
            self,
            groupby_time_feature=None,
            groupby_sliding_window_size=None,
            groupby_custom_column=None,
            show_mean=False,
            show_quantiles=False,
            show_overlays=False,
            overlay_label_time_feature=None,
            overlay_label_sliding_window_size=None,
            overlay_label_custom_column=None,
            center_values=False,
            value_col=VALUE_COL,
            mean_col_name="mean",
            quantile_col_prefix="Q",
            **overlay_pivot_table_kwargs):
        """Computes mean, quantiles, and overlays by the requested grouping dimension.

        Overlays are best explained in the plotting context. The grouping dimension goes on
        the x-axis, and one line is shown for each level of the overlay dimension. This
        function returns a column for each line to plot (e.g. mean, each quantile,
        each overlay value).

        Exactly one of: ``groupby_time_feature``, ``groupby_sliding_window_size``,
        ``groupby_custom_column`` must be provided as the grouping dimension.

        If ``show_overlays`` is True, exactly one of: ``overlay_label_time_feature``,
        ``overlay_label_sliding_window_size``, ``overlay_label_custom_column`` can be
        provided to specify the ``label_col`` (overlay dimension). Internally, the
        function calls `pandas.DataFrame.pivot_table` with ``index=groupby_col``,
        ``columns=label_col``, ``values=value_col`` to get the overlay values for plotting.
        You can pass additional parameters to `pandas.DataFrame.pivot_table` via
        ``overlay_pivot_table_kwargs``, e.g. to change the aggregation method. If an explicit
        label is not provided, the records are labeled by their position within the group.

        For example, to show yearly seasonality mean, quantiles, and overlay plots for
        each individual year, use::

            self.get_quantiles_and_overlays(
                groupby_time_feature="doy",         # Rows: a row for each day of year (1, 2, ..., 366)
                show_mean=True,                     # mean value on that day
                show_quantiles=[0.1, 0.9],          # quantiles of the observed distribution on that day
                show_overlays=True,                 # Include overlays defined by ``overlay_label_time_feature``
                overlay_label_time_feature="year")  # One column for each observed "year" (2016, 2017, 2018, ...)

        To show weekly seasonality over time, use::

            self.get_quantiles_and_overlays(
                groupby_time_feature="dow",            # Rows: a row for each day of week (1, 2, ..., 7)
                show_mean=True,                        # mean value on that day
                show_quantiles=[0.1, 0.5, 0.9],        # quantiles of the observed distribution on that day
                show_overlays=True,                    # Include overlays defined by ``overlay_label_time_feature``
                overlay_label_sliding_window_size=90,  # One column for each 90 period sliding window in the dataset,
                aggfunc="median")                      # overlay value is the median value for the dow over the period (default="mean").


        It may be difficult to assess the weekly seasonality from the previous result,
        because overlays shift up/down over time due to trend/yearly seasonality.
        Use ``center_values=True`` to adjust each overlay so its average value is centered at 0.
        Mean and quantiles are shifted by a single constant to center the mean at 0, while
        preserving their relative values::

            self.get_quantiles_and_overlays(
                groupby_time_feature="dow",
                show_mean=True,
                show_quantiles=[0.1, 0.5, 0.9],
                show_overlays=True,
                overlay_label_sliding_window_size=90,
                aggfunc="median",
                center_values=True)  # Centers the output

        Centering reduces the variability in the overlays to make it easier to isolate
        the effect by the groupby column. As a result, centered overlays have smaller
        variability than that reported by the quantiles, which operate on the original,
        uncentered data points. Similarly, if overlays are aggregates of individual values
        (i.e. ``aggfunc`` is needed in the call to `pandas.DataFrame.pivot_table`),
        the quantiles of overlays will be less extreme than those of the original data.

            - To assess variability conditioned on the groupby value, check the quantiles.
            - To assess variability conditioned on both the groupby and overlay value,
              after any necessary aggregation, check the variability of the overlay values.
              Compute quantiles of overlays from the return value if desired.

        Parameters
        ----------
        groupby_time_feature : `str` or None, default None
            If provided, groups by a column generated by
            `~greykite.common.features.timeseries_features.build_time_features_df`.
            See that function for valid values.
        groupby_sliding_window_size : `int` or None, default None
            If provided, sequentially partitions data into groups of size
            ``groupby_sliding_window_size``.
        groupby_custom_column : `pandas.Series` or None, default None
            If provided, groups by this column value. Should be same length as the DataFrame.
        show_mean : `bool`, default False
            Whether to return the mean value by the groupby column.
        show_quantiles : `bool` or `list` [`float`] or `numpy.array`, default False
            Whether to return the quantiles of the value by the groupby column.
            If False, does not return quantiles. If True, returns default
            quantiles (0.1 and 0.9). If array-like, a list of quantiles
            to compute (e.g. (0.1, 0.25, 0.75, 0.9)).
        show_overlays : `bool` or `int` or array-like [`int` or `str`] or None, default False
            Whether to return overlays of the value by the groupby column.

            If False, no overlays are shown.

            If True and ``label_col`` is defined, calls `pandas.DataFrame.pivot_table` with
            ``index=groupby_col``, ``columns=label_col``, ``values=value_col``.
            ``label_col`` is defined by one of ``overlay_label_time_feature``,
            ``overlay_label_sliding_window_size``, or ``overlay_label_custom_column``.
            Returns one column for each value of the ``label_col``.

            If True and the ``label_col`` is not defined, returns the raw values within
            each group. Values across groups are put into columns by their position in
            the group (1st element in group, 2nd, 3rd, etc.). Positional order in a group
            is not guaranteed to correspond to anything meaningful, so the items within a
            column may not have anything in common. It is better to specify one of ``overlay_*``
            to explicitly define the overlay labels.

            If an integer, the number of overlays to randomly sample. The same as True,
            then randomly samples up to `int` columns. This is useful if there are too many values.

            If a list [int], a list of column indices (int type). The same as True,
            then selects the specified columns by index.

            If a list [str], a list of column names. Column names are matched by their
            string representation to the names in this list. The same as True,
            then selects the specified columns by name.
        overlay_label_time_feature : `str` or None, default None
            If ``show_overlays`` is True, can be used to define ``label_col``,
            i.e. which dimension to show separately as overlays.

            If provided, uses a column generated by
            `~greykite.common.features.timeseries_features.build_time_features_df`.
            See that function for valid values.
        overlay_label_sliding_window_size : `int` or None, default None
            If ``show_overlays`` is True, can be used to define ``label_col``,
            i.e. which dimension to show separately as overlays.

            If provided, uses a column that sequentially partitions data into groups
            of size ``groupby_sliding_window_size``.
        overlay_label_custom_column : `pandas.Series` or None, default None
            If ``show_overlays`` is True, can be used to define ``label_col``,
            i.e. which dimension to show separately as overlays.

            If provided, uses this column value. Should be same length as the DataFrame.

        value_col : `str`, default VALUE_COL
            The column name for the value column. By default,
            shows the univariate time series value, but it can be any
            other column in ``self.df``.

        mean_col_name : `str`, default "mean"
            The name to use for the mean column in the output.
            Applies if ``show_mean=True``.

        quantile_col_prefix : `str`, default "Q"
            The prefix to use for quantile column names in the output.
            Columns are named with this prefix followed by the quantile,
            rounded to 2 decimal places.

        center_values : `bool`, default False
            Whether to center the return values.
            If True, shifts each overlay so its average value is centered at 0.
            Shifts mean and quantiles by a constant to center the mean at 0, while
            preserving their relative values.

            If False, values are not centered.

        overlay_pivot_table_kwargs : additional parameters
            Additional keyword parameters to pass to `pandas.DataFrame.pivot_table`,
            used in generating the overlays. See above description for details.

        Returns
        -------
        grouped_df : `pandas.DataFrame`
            Dataframe with mean, quantiles, and overlays by the grouping column. Overlays
            are defined by the grouping column and overlay dimension.

            ColumnIndex is a multiindex with first level as the "category", a subset of
            [MEAN_COL_GROUP, QUANTILE_COL_GROUP, OVERLAY_COL_GROUP] depending on what is requests.

                - grouped_df[MEAN_COL_GROUP] = df with single column, named ``mean_col_name``.

                - grouped_df[QUANTILE_COL_GROUP] = df with a column for each quantile, named
                  f"{quantile_col_prefix}{round(str(q))}", where ``q`` is the quantile.

                - grouped_df[OVERLAY_COL_GROUP] = df with one column per overlay value, named
                  by the overlay value.

            For example, it might look like::

                category    mean    quantile        overlay
                name        mean    Q0.1    Q0.9    2007    2008    2009
                doy
                1	        8.42	7.72    9.08	8.29	7.75	8.33
                2	        8.82	8.20    9.56	8.43	8.80	8.53
                3	        8.95	8.25    9.88	8.26	9.12	8.70
                4	        9.07	8.60    9.49	8.10	9.99	8.73
                5	        8.73	8.29    9.24	7.95	9.26	8.37
                ...         ...     ...     ...     ...     ...     ...

        """
        # Default quantiles to show if `show_quantiles` is boolean
        if isinstance(show_quantiles, bool):
            if show_quantiles:
                show_quantiles = [0.1, 0.9]
            else:
                show_quantiles = None

        # Adds grouping dimension
        result = add_groupby_column(
            df=self.df,
            time_col=TIME_COL,  # Already standardized
            groupby_time_feature=groupby_time_feature,
            groupby_sliding_window_size=groupby_sliding_window_size,
            groupby_custom_column=groupby_custom_column)
        df = result["df"]
        groupby_col = result["groupby_col"]
        grouped_df = None

        # Whether an overlay label is provided
        add_overlay_label = (overlay_label_time_feature is not None) or \
            (overlay_label_sliding_window_size is not None) or \
            (overlay_label_custom_column is not None)
        overlay_df = None

        # Defines an aggregation function to compute mean, quantiles, and overlays
        agg_kwargs = {}
        if show_mean:
            agg_kwargs.update({mean_col_name: pd.NamedAgg(column=value_col, aggfunc=np.nanmean)})
        if show_quantiles is not None:
            # Returns the quantiles of the group's `value_col` as a list
            agg_kwargs.update({quantile_col_prefix: pd.NamedAgg(
                column=value_col,
                aggfunc=lambda grp_values: partial(np.nanquantile, q=show_quantiles)(grp_values).tolist())})
        if show_overlays is not False:
            if add_overlay_label:
                # Uses DataFrame pivot_table to get overlay labels as columns, `groupby_col` as index
                label_result = add_groupby_column(
                    df=df,
                    time_col=TIME_COL,
                    groupby_time_feature=overlay_label_time_feature,
                    groupby_sliding_window_size=overlay_label_sliding_window_size,
                    groupby_custom_column=overlay_label_custom_column)
                label_col = label_result["groupby_col"]
                overlay_df = label_result["df"].pivot_table(
                    index=groupby_col,
                    columns=label_col,
                    values=value_col,
                    **overlay_pivot_table_kwargs)
            else:
                # Uses aggregation to get overlays.
                # Takes original values within each group.
                # Values across groups are put into columns by their position
                # within the group (1st element in group, 2nd, 3rd, etc.)
                agg_kwargs.update({"overlay": pd.NamedAgg(column=value_col, aggfunc=tuple)})

        # Names the quantile columns
        # Keeps to 2 decimal places to handle numerical imprecision.
        list_names_dict = {quantile_col_prefix: [
            f"{quantile_col_prefix}{str(round(x, 2))}" for x in show_quantiles]}\
            if show_quantiles is not None else {}
        if agg_kwargs:
            grouped_df = flexible_grouping_evaluation(
                result["df"],
                map_func_dict=None,
                groupby_col=result["groupby_col"],
                agg_kwargs=agg_kwargs,
                extend_col_names=False,
                unpack_list=True,
                list_names_dict=list_names_dict)
        # Adds overlays if requested and not already computed during aggregation
        if overlay_df is not None:
            overlay_df.columns = map(str, overlay_df.columns)

        # Either overlay_df or grouped_df is populated
        if grouped_df is None and overlay_df is None:
            raise ValueError("Must enable at least one of: show_mean, show_quantiles, show_overlays.")
        grouped_df = pd.concat([grouped_df, overlay_df], axis=1)

        # Creates MultiIndex for column names to categorize the column names by their type
        mean_cols = [mean_col_name] if show_mean else []
        quantile_cols = list_names_dict.get(quantile_col_prefix, [])
        overlay_cols = [col for col in list(grouped_df.columns) if col not in mean_cols + quantile_cols]

        if isinstance(show_overlays, int) and not isinstance(show_overlays, bool):
            # Samples from `overlay_cols`
            which_overlays = sorted(np.random.choice(
                range(len(overlay_cols)),
                size=min(show_overlays, len(overlay_cols)),
                replace=False))
            overlay_cols = list(np.array(overlay_cols)[which_overlays])
        elif isinstance(show_overlays, (list, tuple, np.ndarray)):
            # Selects from `overlay_cols`
            all_integers = np.issubdtype(np.array(show_overlays).dtype, np.integer)
            if all_integers:
                overlay_cols = [col for i, col in enumerate(overlay_cols) if i in show_overlays]
            else:
                overlay_cols = [col for col in overlay_cols if str(col) in show_overlays]
        cols = mean_cols + quantile_cols + overlay_cols  # Reorders columns by group
        grouped_df = grouped_df[cols]
        categories = list(np.repeat(
            [MEAN_COL_GROUP, QUANTILE_COL_GROUP, OVERLAY_COL_GROUP],  # Labels columns by category
            [len(mean_cols), len(quantile_cols), len(overlay_cols)]))
        cateory_col_index = pd.MultiIndex.from_arrays([categories, cols], names=["category", "name"])
        grouped_df.columns = cateory_col_index

        if center_values:
            # Each overlay is independently shifted to have mean 0.
            if OVERLAY_COL_GROUP in grouped_df:
                grouped_df[OVERLAY_COL_GROUP] -= grouped_df[OVERLAY_COL_GROUP].mean()
            # Mean and quantiles are shifted by the same constant, so the mean column is centered at 0.
            if MEAN_COL_GROUP in grouped_df:
                mean_shift = grouped_df[MEAN_COL_GROUP].mean()[0]
                grouped_df[MEAN_COL_GROUP] -= mean_shift
            else:
                mean_shift = self.df[value_col].mean()
            if QUANTILE_COL_GROUP in grouped_df:
                grouped_df[QUANTILE_COL_GROUP] -= mean_shift

        return grouped_df

    def plot_quantiles_and_overlays(
            self,
            groupby_time_feature=None,
            groupby_sliding_window_size=None,
            groupby_custom_column=None,
            show_mean=False,
            show_quantiles=False,
            show_overlays=False,
            overlay_label_time_feature=None,
            overlay_label_sliding_window_size=None,
            overlay_label_custom_column=None,
            center_values=False,
            value_col=VALUE_COL,
            mean_col_name="mean",
            quantile_col_prefix="Q",
            mean_style=None,
            quantile_style=None,
            overlay_style=None,
            xlabel=None,
            ylabel=None,
            title=None,
            showlegend=True,
            **overlay_pivot_table_kwargs):
        """Plots mean, quantiles, and overlays by the requested grouping dimension.

        The grouping dimension goes on the x-axis, and one line is shown for the mean,
        each quantile, and each level of the overlay dimension, as requested. By default,
        shading is applied between the quantiles.

        Exactly one of: ``groupby_time_feature``, ``groupby_sliding_window_size``,
        ``groupby_custom_column`` must be provided as the grouping dimension.

        If ``show_overlays`` is True, exactly one of: ``overlay_label_time_feature``,
        ``overlay_label_sliding_window_size``, ``overlay_label_custom_column`` can be
        provided to specify the ``label_col`` (overlay dimension). Internally, the
        function calls `pandas.DataFrame.pivot_table` with ``index=groupby_col``,
        ``columns=label_col``, ``values=value_col`` to get the overlay values for plotting.
        You can pass additional parameters to `pandas.DataFrame.pivot_table` via
        ``overlay_pivot_table_kwargs``, e.g. to change the aggregation method. If an explicit
        label is not provided, the records are labeled by their position within the group.

        For example, to show yearly seasonality mean, quantiles, and overlay plots for
        each individual year, use::

            self.plot_quantiles_and_overlays(
                groupby_time_feature="doy",         # Rows: a row for each day of year (1, 2, ..., 366)
                show_mean=True,                     # mean value on that day
                show_quantiles=[0.1, 0.9],          # quantiles of the observed distribution on that day
                show_overlays=True,                 # Include overlays defined by ``overlay_label_time_feature``
                overlay_label_time_feature="year")  # One column for each observed "year" (2016, 2017, 2018, ...)

        To show weekly seasonality over time, use::

            self.plot_quantiles_and_overlays(
                groupby_time_feature="dow",            # Rows: a row for each day of week (1, 2, ..., 7)
                show_mean=True,                        # mean value on that day
                show_quantiles=[0.1, 0.5, 0.9],        # quantiles of the observed distribution on that day
                show_overlays=True,                    # Include overlays defined by ``overlay_label_time_feature``
                overlay_label_sliding_window_size=90,  # One column for each 90 period sliding window in the dataset,
                aggfunc="median")                      # overlay value is the median value for the dow over the period (default="mean").

        It may be difficult to assess the weekly seasonality from the previous result,
        because overlays shift up/down over time due to trend/yearly seasonality.
        Use ``center_values=True`` to adjust each overlay so its average value is centered at 0.
        Mean and quantiles are shifted by a single constant to center the mean at 0, while
        preserving their relative values::

            self.plot_quantiles_and_overlays(
                groupby_time_feature="dow",
                show_mean=True,
                show_quantiles=[0.1, 0.5, 0.9],
                show_overlays=True,
                overlay_label_sliding_window_size=90,
                aggfunc="median",
                center_values=True)  # Centers the output

        Centering reduces the variability in the overlays to make it easier to isolate
        the effect by the groupby column. As a result, centered overlays have smaller
        variability than that reported by the quantiles, which operate on the original,
        uncentered data points. Similarly, if overlays are aggregates of individual values
        (i.e. ``aggfunc`` is needed in the call to `pandas.DataFrame.pivot_table`),
        the quantiles of overlays will be less extreme than those of the original data.

            - To assess variability conditioned on the groupby value, check the quantiles.
            - To assess variability conditioned on both the groupby and overlay value,
              after any necessary aggregation, check the variability of the overlay values.
              Compute quantiles of overlays from the return value if desired.

        Parameters
        ----------
        groupby_time_feature : `str` or None, default None
            If provided, groups by a column generated by
            `~greykite.common.features.timeseries_features.build_time_features_df`.
            See that function for valid values.
        groupby_sliding_window_size : `int` or None, default None
            If provided, sequentially partitions data into groups of size
            ``groupby_sliding_window_size``.
        groupby_custom_column : `pandas.Series` or None, default None
            If provided, groups by this column value. Should be same length as the DataFrame.
        show_mean : `bool`, default False
            Whether to return the mean value by the groupby column.
        show_quantiles : `bool` or `list` [`float`] or `numpy.array`, default False
            Whether to return the quantiles of the value by the groupby column.
            If False, does not return quantiles. If True, returns default
            quantiles (0.1 and 0.9). If array-like, a list of quantiles
            to compute (e.g. (0.1, 0.25, 0.75, 0.9)).
        show_overlays : `bool` or `int` or array-like [`int` or `str`], default False
            Whether to return overlays of the value by the groupby column.

            If False, no overlays are shown.

            If True and ``label_col`` is defined, calls `pandas.DataFrame.pivot_table` with
            ``index=groupby_col``, ``columns=label_col``, ``values=value_col``.
            ``label_col`` is defined by one of ``overlay_label_time_feature``,
            ``overlay_label_sliding_window_size``, or ``overlay_label_custom_column``.
            Returns one column for each value of the ``label_col``.

            If True and the ``label_col`` is not defined, returns the raw values within
            each group. Values across groups are put into columns by their position in
            the group (1st element in group, 2nd, 3rd, etc.). Positional order in a group
            is not guaranteed to correspond to anything meaningful, so the items within a
            column may not have anything in common. It is better to specify one of ``overlay_*``
            to explicitly define the overlay labels.

            If an integer, the number of overlays to randomly sample. The same as True,
            then randomly samples up to `int` columns. This is useful if there are too many values.

            If a list [int], a list of column indices (int type). The same as True,
            then selects the specified columns by index.

            If a list [str], a list of column names. Column names are matched by their
            string representation to the names in this list. The same as True,
            then selects the specified columns by name.

        overlay_label_time_feature : `str` or None, default None
            If ``show_overlays`` is True, can be used to define ``label_col``,
            i.e. which dimension to show separately as overlays.

            If provided, uses a column generated by
            `~greykite.common.features.timeseries_features.build_time_features_df`.
            See that function for valid values.
        overlay_label_sliding_window_size : `int` or None, default None
            If ``show_overlays`` is True, can be used to define ``label_col``,
            i.e. which dimension to show separately as overlays.

            If provided, uses a column that sequentially partitions data into groups
            of size ``groupby_sliding_window_size``.
        overlay_label_custom_column : `pandas.Series` or None, default None
            If ``show_overlays`` is True, can be used to define ``label_col``,
            i.e. which dimension to show separately as overlays.

            If provided, uses this column value. Should be same length as the DataFrame.

        value_col : `str`, default VALUE_COL
            The column name for the value column. By default,
            shows the univariate time series value, but it can be any
            other column in ``self.df``.

        mean_col_name : `str`, default "mean"
            The name to use for the mean column in the output.
            Applies if ``show_mean=True``.

        quantile_col_prefix : `str`, default "Q"
            The prefix to use for quantile column names in the output.
            Columns are named with this prefix followed by the quantile,
            rounded to 2 decimal places.

        center_values : `bool`, default False
            Whether to center the return values.
            If True, shifts each overlay so its average value is centered at 0.
            Shifts mean and quantiles by a constant to center the mean at 0, while
            preserving their relative values.

            If False, values are not centered.

        mean_style: `dict` or None, default None
            How to style the mean line, passed as keyword arguments to
            `plotly.graph_objects.Scatter`. If None, the default is::

                mean_style = {
                    "line": dict(
                        width=2,
                        color="#595959"),  # gray
                    "legendgroup": MEAN_COL_GROUP}

        quantile_style: `dict` or None, default None
            How to style the quantile lines, passed as keyword arguments to
            `plotly.graph_objects.Scatter`. If None, the default is::

                quantile_style = {
                    "line": dict(
                        width=2,
                        color="#1F9AFF",  # blue
                        dash="solid"),
                    "legendgroup": QUANTILE_COL_GROUP,  # show/hide them together
                    "fill": "tonexty"}

            Note that fill style is removed from to the first quantile line, to
            fill only between items in the same category.

        overlay_style: `dict` or None, default None
            How to style the overlay lines, passed as keyword arguments to
            `plotly.graph_objects.Scatter`. If None, the default is::

                overlay_style = {
                    "opacity": 0.5,  # makes it easier to see density
                    "line": dict(
                        width=1,
                        color="#B3B3B3",  # light gray
                        dash="solid"),
                    "legendgroup": OVERLAY_COL_GROUP}

        xlabel : `str`, optional, default None
            X-axis label of the plot.
        ylabel : `str`, optional, default None
            Y-axis label of the plot. If None, uses ``value_col``.
        title : `str` or None, default None
            Plot title. If None, default is based on axis labels.
        showlegend : `bool`, default True
            Whether to show the legend.
        overlay_pivot_table_kwargs : additional parameters
            Additional keyword parameters to pass to `pandas.DataFrame.pivot_table`,
            used in generating the overlays.
            See `~greykite.framework.input.univariate_time_series.UnivariateTimeSeries.get_quantiles_and_overlays`
            description for details.

        Returns
        -------
        fig : `plotly.graph_objects.Figure`
            plotly graph object showing the mean, quantiles, and overlays.

        See Also
        --------
        `~greykite.framework.input.univariate_time_series.UnivariateTimeSeries.get_quantiles_and_overlays`
            To get the mean, quantiles, and overlays as a `pandas.DataFrame` without plotting.
        """

        if ylabel is None:
            ylabel = value_col

        grouped_df = self.get_quantiles_and_overlays(
            groupby_time_feature=groupby_time_feature,
            groupby_sliding_window_size=groupby_sliding_window_size,
            groupby_custom_column=groupby_custom_column,
            show_mean=show_mean,
            show_quantiles=show_quantiles,
            show_overlays=show_overlays,
            overlay_label_time_feature=overlay_label_time_feature,
            overlay_label_sliding_window_size=overlay_label_sliding_window_size,
            overlay_label_custom_column=overlay_label_custom_column,
            center_values=center_values,
            value_col=value_col,
            mean_col_name=mean_col_name,
            quantile_col_prefix=quantile_col_prefix,
            **overlay_pivot_table_kwargs)

        if mean_style is None:
            mean_style = {
                "line": dict(
                    width=2,
                    color="#595959"),  # gray
                "legendgroup": MEAN_COL_GROUP}
        if quantile_style is None:
            quantile_style = {
                "line": dict(
                    width=2,
                    color="#1F9AFF",  # blue
                    dash="solid"),
                "legendgroup": QUANTILE_COL_GROUP,  # show/hide them together
                "fill": "tonexty"}
        if overlay_style is None:
            overlay_style = {
                "opacity": 0.5,  # makes it easier to see density
                "line": dict(
                    width=1,
                    color="#B3B3B3",  # light gray
                    dash="solid"),
                "legendgroup": OVERLAY_COL_GROUP}
        style_dict = {
            MEAN_COL_GROUP: mean_style,
            QUANTILE_COL_GROUP: quantile_style,
            OVERLAY_COL_GROUP: overlay_style}

        y_col_style_dict = {}
        # All categories in grouped_df. Reverses the order so the first category is plotted last (on top).
        categories = grouped_df.columns.get_level_values(0).unique()[::-1]
        for category in categories:
            style = style_dict.get(category, {})
            if "fill" in style:
                # If fill is part of the style, plotly fills the area between this line and
                # the previous line added to the plot.
                # Since we only want to fill between lines in the same category (e.g. between quantiles),
                # we remove the "fill" from the first line within each category. Otherwise the first
                # line in this category would fill to the last line in the previous category.
                category_style_dict = {grouped_df[category].columns[0]: {k: v for k, v in style.items() if k != "fill"}}
                category_style_dict.update({col: style for col in grouped_df[category].columns[1:]})
            else:
                category_style_dict = {col: style for col in grouped_df[category].columns}
            y_col_style_dict.update(category_style_dict)

        grouped_df.columns = list(grouped_df.columns.get_level_values(1))  # MultiIndex is not needed for plotting
        x_col = grouped_df.index.name
        grouped_df.reset_index(inplace=True)
        fig = plot_multivariate(
            grouped_df,
            x_col=x_col,
            y_col_style_dict=y_col_style_dict,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            showlegend=showlegend)
        return fig
