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
"""Forecast output."""

from functools import partial

import pandas as pd
from sklearn.metrics import mean_squared_error

from greykite.common import constants as cst
from greykite.common.evaluation import ElementwiseEvaluationMetricEnum
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.common.evaluation import add_finite_filter_to_scorer
from greykite.common.evaluation import calc_pred_coverage
from greykite.common.evaluation import calc_pred_err
from greykite.common.evaluation import fraction_outside_tolerance
from greykite.common.evaluation import r2_null_model_score
from greykite.common.python_utils import apply_func_to_columns
from greykite.common.viz.timeseries_plotting import add_groupby_column
from greykite.common.viz.timeseries_plotting import flexible_grouping_evaluation
from greykite.common.viz.timeseries_plotting import grouping_evaluation
from greykite.common.viz.timeseries_plotting import plot_forecast_vs_actual
from greykite.common.viz.timeseries_plotting import plot_multivariate
from greykite.common.viz.timeseries_plotting import plot_univariate
from greykite.framework.input.univariate_time_series import UnivariateTimeSeries


class UnivariateForecast:
    """Stores predicted and actual values.
    Provides functionality to evaluate a forecast:

        - plots true against actual with prediction bands.
        - evaluates model performance.

    Input should be one of two kinds of forecast results:

        - model fit to train data, forecast on test set (actuals available).
        - model fit to all data, forecast on future dates (actuals not available).

    The input ``df`` is a concatenation of fitted and forecasted values.

    Attributes
    ----------
    df : `pandas.DataFrame`
        Timestamp, predicted, and actual values.
    time_col : `str`
        Column in ``df`` with timestamp (default "ts").
    actual_col : `str`
        Column in ``df`` with actual values (default "y").
    predicted_col : `str`
        Column in ``df`` with predicted values (default "forecast").
    predicted_lower_col : `str` or None
        Column in ``df`` with predicted lower bound (default "forecast_lower", optional).
    predicted_upper_col : `str` or None
        Column in ``df`` with predicted upper bound (default "forecast_upper", optional).
    null_model_predicted_col : `str` or None
        Column in ``df`` with predicted value of null model (default "forecast_null", optional).
    ylabel : `str`
        Unit of measurement (default "y")
    train_end_date : `str` or `datetime` or None, default None
        End date for train period. If `None`, assumes all data were used for training.
    test_start_date : `str` or `datetime` or None, default None
        Start date of test period. If `None`, set to the ``time_col`` value immediately after
        ``train_end_date``. This assumes that all data not used in training were used for testing.
    forecast_horizon : `int` or None, default None
        Number of periods forecasted into the future. Must be > 0.
    coverage : `float` or None
        Intended coverage of the prediction bands (0.0 to 1.0).
    r2_loss_function : `function`
        Loss function to calculate ``cst.R2_null_model_score``, with signature
        ``loss_func(y_true, y_pred)`` (default mean_squared_error)
    estimator : An instance of an estimator that implements `greykite.models.base_forecast_estimator.BaseForecastEstimator`.
        The fitted estimator, the last step in the forecast pipeline.
    relative_error_tolerance : `float` or None, default None
        Threshold to compute the ``Outside Tolerance`` metric,
        defined as the fraction of forecasted values whose relative
        error is strictly greater than ``relative_error_tolerance``.
        For example, 0.05 allows for 5% relative error.
        If `None`, the metric is not computed.
    df_train : `pandas.DataFrame`
        Subset of ``df`` where ``df[time_col]`` <= ``train_end_date``.
    df_test : `pandas.DataFrame`
        Subset of ``df`` where ``df[time_col]`` > ``train_end_date``.
    train_evaluation : `dict` [`str`, `float`]
        Evaluation metrics on training set.
    test_evaluation : `dict` [`str`, `float`]
        Evaluation metrics on test set (if actual values provided after train_end_date).
    test_na_count : `int`
        Count of NA values in test data.
    """

    def __init__(
            self,
            df,
            time_col=cst.TIME_COL,
            actual_col=cst.ACTUAL_COL,
            predicted_col=cst.PREDICTED_COL,
            predicted_lower_col=cst.PREDICTED_LOWER_COL,
            predicted_upper_col=cst.PREDICTED_UPPER_COL,
            null_model_predicted_col=cst.NULL_PREDICTED_COL,
            ylabel=cst.VALUE_COL,
            train_end_date=None,
            test_start_date=None,
            forecast_horizon=None,
            coverage=0.95,
            r2_loss_function=mean_squared_error,
            estimator=None,
            relative_error_tolerance=None):
        if predicted_lower_col is not None or predicted_upper_col is not None:
            if coverage is None:
                raise ValueError("`coverage` must be provided when lower/upper bounds are set")
            elif coverage < 0.0 or coverage > 1.0:
                raise ValueError("`coverage` must be between 0.0 and 1.0")

        if train_end_date is not None and not all(pd.Series(train_end_date).isin(df[time_col])):
            raise ValueError(
                f"train_end_date {train_end_date} is not found in time column.\n"
                f"The time range in data is: from {min(df[time_col])} to {max(df[time_col])}.\n"
                f"The data size is: {df.shape[0]}.\n"
                f"Type of train_end_date is {type(train_end_date)}.\n"
                f"Type of df[time_col] is {df[time_col].dtype}.\n")

        if any([col not in df.columns
                for col in [time_col, actual_col, predicted_col, predicted_lower_col, predicted_upper_col,
                            null_model_predicted_col] if col is not None]):
            raise ValueError(f"Column not found in data frame")

        self.df = df
        self.time_col = time_col
        self.actual_col = actual_col
        self.predicted_col = predicted_col
        self.predicted_lower_col = predicted_lower_col
        self.predicted_upper_col = predicted_upper_col
        self.null_model_predicted_col = null_model_predicted_col
        self.ylabel = ylabel
        self.train_end_date = train_end_date
        self.test_start_date = test_start_date
        self.forecast_horizon = forecast_horizon
        self.coverage = coverage
        self.r2_loss_function = r2_loss_function
        self.estimator = estimator
        self.relative_error_tolerance = relative_error_tolerance

        self.df[self.time_col] = pd.to_datetime(df[time_col], infer_datetime_format=True)
        if train_end_date is None:
            self.train_end_date = df[self.time_col].max()
        else:
            self.train_end_date = pd.to_datetime(train_end_date, infer_datetime_format=True)
        self.df_train = df[df[time_col] <= self.train_end_date]

        if test_start_date is None:
            # This expects no gaps in time column
            inferred_freq = pd.infer_freq(self.df[self.time_col])
            # Uses pd.date_range because pd.Timedelta does not work for complicated frequencies e.g. "W-MON"
            self.test_start_date = pd.date_range(
                start=self.train_end_date,
                periods=2,
                freq=inferred_freq)[-1]
        else:
            self.test_start_date = pd.to_datetime(test_start_date, infer_datetime_format=True)
        self.df_test = df[df[time_col] >= self.test_start_date]

        self.test_na_count = self.df_test[actual_col].isna().sum()

        # compute evaluation metrics
        self.train_evaluation = None
        self.test_evaluation = None
        self.compute_evaluation_metrics_split()

    def __evaluation_metrics(self, data):
        """Computes various evaluation metrics for the forecast.

        :param data: pd.DataFrame
            with columns according to self.__init__ (e.g. subset of self.df)
        :return: dictionary with evaluation metrics
        """
        metrics = calc_pred_err(data[self.actual_col], data[self.predicted_col])
        if self.relative_error_tolerance is not None:
            metrics[cst.FRACTION_OUTSIDE_TOLERANCE] = partial(
                fraction_outside_tolerance,
                rtol=self.relative_error_tolerance)(
                    y_true=data[self.actual_col],
                    y_pred=data[self.predicted_col])
        else:
            metrics[cst.FRACTION_OUTSIDE_TOLERANCE] = None

        for metric in [cst.R2_null_model_score, cst.PREDICTION_BAND_WIDTH, cst.PREDICTION_BAND_COVERAGE,
                       cst.LOWER_BAND_COVERAGE, cst.UPPER_BAND_COVERAGE, cst.COVERAGE_VS_INTENDED_DIFF]:
            metrics.update({metric: None})

        # evaluates cst.R2_null_model_score if null model is available
        if self.null_model_predicted_col is not None:
            metrics.update({cst.R2_null_model_score: r2_null_model_score(data[self.actual_col],
                                                                         data[self.predicted_col],
                                                                         y_pred_null=data[self.null_model_predicted_col],
                                                                         loss_func=self.r2_loss_function)})

        # evaluates prediction bands if available
        if self.predicted_lower_col is not None and self.predicted_upper_col is not None:
            result = calc_pred_coverage(
                data[self.actual_col],
                data[self.predicted_col],
                data[self.predicted_lower_col],
                data[self.predicted_upper_col],
                self.coverage)
            metrics.update(result)
        return metrics

    def compute_evaluation_metrics_split(self):
        """Computes __evaluation_metrics for train and test set separately.

        :return: dictionary with train and test evaluation metrics
        """
        self.train_evaluation = self.__evaluation_metrics(self.df_train)
        # only test evaluation if there are actuals (e.g. not future prediction and not all missing)
        if self.df_test.shape[0] > 0 and self.test_na_count < self.df_test.shape[0]:
            self.test_evaluation = self.__evaluation_metrics(self.df_test)
        return {
            "Train": self.train_evaluation,
            "Test": self.test_evaluation
        }

    def plot(self, **kwargs):
        """Plots predicted against actual.

        Parameters
        ----------
        kwargs : additional parameters
            Additional parameters to pass to
            `~greykite.common.viz.timeseries_plotting.plot_forecast_vs_actual`
            such as title, colors, and line styling.

        Returns
        -------
        fig : `plotly.graph_objects.Figure`
            Plotly figure of forecast against actuals, with prediction
            intervals if available.

            See `~greykite.common.viz.timeseries_plotting.plot_forecast_vs_actual`
            return value for how to plot the figure and add customization.
        """
        return plot_forecast_vs_actual(
            self.df,
            time_col=self.time_col,
            actual_col=self.actual_col,
            predicted_col=self.predicted_col,
            predicted_lower_col=self.predicted_lower_col,
            predicted_upper_col=self.predicted_upper_col,
            ylabel=self.ylabel,
            train_end_date=self.train_end_date,
            **kwargs)

    def get_grouping_evaluation(
            self,
            score_func=EvaluationMetricEnum.MeanAbsolutePercentError.get_metric_func(),
            score_func_name=EvaluationMetricEnum.MeanAbsolutePercentError.get_metric_name(),
            which="train",
            groupby_time_feature=None,
            groupby_sliding_window_size=None,
            groupby_custom_column=None):
        """Group-wise computation of forecasting error.
        Can be used to evaluate error/ aggregated value by a time feature,
        over time, or by a user-provided column.

        Exactly one of: ``groupby_time_feature``, ``groupby_sliding_window_size``,
        ``groupby_custom_column`` must be provided.

        Parameters
        ----------
        score_func : callable, optional
            Function that maps two arrays to a number.
            Signature (y_true: array, y_pred: array) -> error: float
        score_func_name : `str` or None, optional
            Name of the score function used to report results.
            If None, defaults to "metric".
        which: `str`
            "train" or "test". Which dataset to evaluate.
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
                evaluation metric computing forecasting error of timeseries.
            (2) group name:
                group name depends on the grouping method:
                ``groupby_time_feature`` for ``groupby_time_feature``
                ``cst.TIME_COL`` for ``groupby_sliding_window_size``
                ``groupby_custom_column.name`` for ``groupby_custom_column``.
        """
        df = self.df_train.copy() if which.lower() == "train" else self.df_test.copy()
        score_func = add_finite_filter_to_scorer(score_func)  # in case it's not already added
        if score_func_name:
            grouping_func_name = f"{which} {score_func_name}"
        else:
            grouping_func_name = f"{which} metric"

        def grouping_func(grp):
            return score_func(grp[self.actual_col], grp[self.predicted_col])

        result = add_groupby_column(
            df=df,
            time_col=self.time_col,
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
            score_func=EvaluationMetricEnum.MeanAbsolutePercentError.get_metric_func(),
            score_func_name=EvaluationMetricEnum.MeanAbsolutePercentError.get_metric_name(),
            which="train",
            groupby_time_feature=None,
            groupby_sliding_window_size=None,
            groupby_custom_column=None,
            xlabel=None,
            ylabel=None,
            title=None):
        """Computes error by group and plots the result.
        Can be used to plot error by a time feature, over time, or by a user-provided column.

        Exactly one of: ``groupby_time_feature``, ``groupby_sliding_window_size``,
        ``groupby_custom_column`` must be provided.

        Parameters
        ----------
        score_func : callable, optional
            Function that maps two arrays to a number.
            Signature (y_true: array, y_pred: array) -> error: float
        score_func_name : `str` or None, optional
            Name of the score function used to report results.
            If None, defaults to "metric".
        which: `str`, optional, default "train"
            Which dataset to evaluate, "train" or "test".
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
            Plot title, if None this function creates a suitable title.

        Returns
        -------
        fig : `plotly.graph_objects.Figure`
            plotly graph object showing forecasting error by group.
            x-axis label depends on the grouping method:
            ``groupby_time_feature`` for ``groupby_time_feature``
            ``time_col`` for ``groupby_sliding_window_size``
            ``groupby_custom_column.name`` for ``groupby_custom_column``.
        """
        grouped_df = self.get_grouping_evaluation(
            score_func=score_func,
            score_func_name=score_func_name,
            which=which,
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

    def autocomplete_map_func_dict(self, map_func_dict):
        """Sweeps through ``map_func_dict``, converting values that are
        `~greykite.common.evaluation.ElementwiseEvaluationMetricEnum`
        member names to their corresponding row-wise evaluation function with appropriate
        column names for this UnivariateForecast instance.

        For example::

            map_func_dict = {
                "squared_error": ElementwiseEvaluationMetricEnum.SquaredError.name,
                "coverage": ElementwiseEvaluationMetricEnum.Coverage.name,
                "custom_metric": custom_function
            }

            is converted to

            map_func_dict = {
                "squared_error": lambda row: ElementwiseEvaluationMetricEnum.SquaredError.get_metric_func()(
                                            row[self.actual_col],
                                            row[self.predicted_col]),
                "coverage": lambda row: ElementwiseEvaluationMetricEnum.Coverage.get_metric_func()(
                                            row[self.actual_col],
                                            row[self.predicted_lower_col],
                                            row[self.predicted_upper_col]),
                "custom_metric": custom_function
            }

        Parameters
        ----------
        map_func_dict : `dict` or None
            Same as `~greykite.common.viz.timeseries_plotting.flexible_grouping_evaluation`,
            with one exception: values may a ElementwiseEvaluationMetricEnum member name.
            There are converted a callable for ``flexible_grouping_evaluation``.

        Returns
        -------
        map_func_dict : `dict`
            Can be passed to `~greykite.common.viz.timeseries_plotting.flexible_grouping_evaluation`.
        """
        if map_func_dict is None:
            updated_map_func_dict = None
        else:
            updated_map_func_dict = {}
            for name, func in map_func_dict.items():
                if isinstance(func, str):
                    if func in ElementwiseEvaluationMetricEnum.__members__:
                        enum = ElementwiseEvaluationMetricEnum[func]
                        row_func = enum.get_metric_func()
                        args = enum.get_metric_args()
                        arg_mapping = {  # maps canonical names to the names in self.df
                            cst.ACTUAL_COL: self.actual_col,
                            cst.PREDICTED_COL: self.predicted_col,
                            cst.PREDICTED_LOWER_COL: self.predicted_lower_col,
                            cst.PREDICTED_UPPER_COL: self.predicted_upper_col,
                            cst.NULL_PREDICTED_COL: self.null_model_predicted_col,
                            cst.TIME_COL: self.time_col,
                        }
                        try:
                            cols = [arg_mapping[arg] for arg in args]
                        except KeyError:
                            raise ValueError(f"Auto-complete for {func} is not "
                                             f"enabled, please specify the full function.")

                        # Creates a function that passes the appropriate columns
                        # to the elementwise evaluation function
                        updated_map_func_dict[name] = apply_func_to_columns(row_func, cols)
                    else:
                        valid_names = ", ".join(ElementwiseEvaluationMetricEnum.__dict__["_member_names_"])
                        raise ValueError(f"{func} is not a recognized elementwise evaluation "
                                         f"metric. Must be one of: {valid_names}")
                else:
                    # `func` is a callable
                    updated_map_func_dict[name] = func
        return updated_map_func_dict

    def get_flexible_grouping_evaluation(
            self,
            which="train",
            groupby_time_feature=None,
            groupby_sliding_window_size=None,
            groupby_custom_column=None,
            map_func_dict=None,
            agg_kwargs=None,
            extend_col_names=False):
        """Group-wise computation of evaluation metrics. Whereas ``self.get_grouping_evaluation``
        computes one metric, this allows computation of any number of custom metrics.

        For example:

            * Mean and quantiles of squared error by group.
            * Mean and quantiles of residuals by group.
            * Mean and quantiles of actual and forecast by group.
            * % of actuals outside prediction intervals by group
            * any combination of the above metrics by the same group

        First adds a groupby column by passing ``groupby_`` parameters to
        `~greykite.common.viz.timeseries_plotting.add_groupby_column`.
        Then computes grouped evaluation metrics by passing ``map_func_dict``,
        ``agg_kwargs`` and ``extend_col_names`` to
        `~greykite.common.viz.timeseries_plotting.flexible_grouping_evaluation`.

        Exactly one of: ``groupby_time_feature``, ``groupby_sliding_window_size``,
        ``groupby_custom_column`` must be provided.

        which: `str`
            "train" or "test". Which dataset to evaluate.
        groupby_time_feature : `str` or None, optional
            If provided, groups by a column generated by
            `~greykite.common.features.timeseries_features.build_time_features_df`.
            See that function for valid values.
        groupby_sliding_window_size : `int` or None, optional
            If provided, sequentially partitions data into groups of size
            ``groupby_sliding_window_size``.
        groupby_custom_column : `pandas.Series` or None, optional
            If provided, groups by this column value. Should be same length as the DataFrame.
        map_func_dict : `dict` [`str`, `callable`] or None, default None
            Row-wise transformation functions to create new columns.
            If None, no new columns are added.

                - key: new column name
                - value: row-wise function to apply to ``df`` to generate the column value.
                         Signature (row: `pandas.DataFrame`) -> transformed value: `float`.

            For example::

                map_func_dict = {
                    "residual": lambda row: row["actual"] - row["forecast"],
                    "squared_error": lambda row: (row["actual"] - row["forecast"])**2
                }

            Some predefined functions are available in
            `~greykite.common.evaluation.ElementwiseEvaluationMetricEnum`. For example::

                map_func_dict = {
                    "residual": lambda row: ElementwiseEvaluationMetricEnum.Residual.get_metric_func()(
                        row["actual"],
                        row["forecast"]),
                    "squared_error": lambda row: ElementwiseEvaluationMetricEnum.SquaredError.get_metric_func()(
                        row["actual"],
                        row["forecast"]),
                    "q90_loss": lambda row: ElementwiseEvaluationMetricEnum.Quantile90.get_metric_func()(
                        row["actual"],
                        row["forecast"]),
                    "abs_percent_error": lambda row: ElementwiseEvaluationMetricEnum.AbsolutePercentError.get_metric_func()(
                        row["actual"],
                        row["forecast"]),
                    "coverage": lambda row: ElementwiseEvaluationMetricEnum.Coverage.get_metric_func()(
                        row["actual"],
                        row["forecast_lower"],
                        row["forecast_upper"]),
                }

            As shorthand, it is sufficient to provide the enum member name.  These are
            auto-expanded into the appropriate function.
            So the following is equivalent::

                map_func_dict = {
                    "residual": ElementwiseEvaluationMetricEnum.Residual.name,
                    "squared_error": ElementwiseEvaluationMetricEnum.SquaredError.name,
                    "q90_loss": ElementwiseEvaluationMetricEnum.Quantile90.name,
                    "abs_percent_error": ElementwiseEvaluationMetricEnum.AbsolutePercentError.name,
                    "coverage": ElementwiseEvaluationMetricEnum.Coverage.name,
                }

        agg_kwargs : `dict` or None, default None
            Passed as keyword args to `pandas.core.groupby.DataFrameGroupBy.aggregate` after creating
            new columns and grouping by ``groupby_col``.

            See `pandas.core.groupby.DataFrameGroupBy.aggregate` or
            `~greykite.common.viz.timeseries_plotting.flexible_grouping_evaluation`
            for details.

        extend_col_names : `bool` or None, default False
            How to flatten index after aggregation.
            In some cases, the column index after aggregation is a multi-index.
            This parameter controls how to flatten an index with 2 levels to 1 level.

                - If None, the index is not flattened.
                - If True, column name is a composite: ``{index0}_{index1}``
                  Use this option if index1 is not unique.
                - If False, column name is simply ``{index1}``

            Ignored if the ColumnIndex after aggregation has only one level (e.g.
            if named aggregation is used in ``agg_kwargs``).

        Returns
        -------
        df_transformed : `pandas.DataFrame`
            ``df`` after transformation and optional aggregation.

            If ``groupby_col`` is None, returns ``df`` with additional columns as the keys in ``map_func_dict``.
            Otherwise, ``df`` is grouped by ``groupby_col`` and this becomes the index. Columns
            are determined by ``agg_kwargs`` and ``extend_col_names``.

        See Also
        --------
        `~greykite.common.viz.timeseries_plotting.add_groupby_column` : called by this function
        `~greykite.common.viz.timeseries_plotting.flexible_grouping_evaluation` : called by this function
        """
        df = self.df_train if which.lower() == "train" else self.df_test
        df = df.copy()
        result = add_groupby_column(
            df=df,
            time_col=self.time_col,
            groupby_time_feature=groupby_time_feature,
            groupby_sliding_window_size=groupby_sliding_window_size,
            groupby_custom_column=groupby_custom_column)
        df = result["df"]

        map_func_dict = self.autocomplete_map_func_dict(map_func_dict)
        grouped_df = flexible_grouping_evaluation(
            df,
            map_func_dict=map_func_dict,
            groupby_col=result["groupby_col"],
            agg_kwargs=agg_kwargs,
            extend_col_names=extend_col_names)

        return grouped_df

    def plot_flexible_grouping_evaluation(
            self,
            which="train",
            groupby_time_feature=None,
            groupby_sliding_window_size=None,
            groupby_custom_column=None,
            map_func_dict=None,
            agg_kwargs=None,
            extend_col_names=False,
            y_col_style_dict="auto-fill",
            default_color="rgba(0, 145, 202, 1.0)",
            xlabel=None,
            ylabel=None,
            title=None,
            showlegend=True):
        """Plots group-wise evaluation metrics. Whereas
        `~greykite.framework.output.univariate_forecast.UnivariateForecast.plot_grouping_evaluation`
        shows one metric, this can show any number of custom metrics.

        For example:

            * Mean and quantiles of squared error by group.
            * Mean and quantiles of residuals by group.
            * Mean and quantiles of actual and forecast by group.
            * % of actuals outside prediction intervals by group
            * any combination of the above metrics by the same group

        See `~greykite.framework.output.univariate_forecast.UnivariateForecast.get_flexible_grouping_evaluation`
        for details.

        which: `str`
            "train" or "test". Which dataset to evaluate.
        groupby_time_feature : `str` or None, optional
            If provided, groups by a column generated by
            `~greykite.common.features.timeseries_features.build_time_features_df`.
            See that function for valid values.
        groupby_sliding_window_size : `int` or None, optional
            If provided, sequentially partitions data into groups of size
            ``groupby_sliding_window_size``.
        groupby_custom_column : `pandas.Series` or None, optional
            If provided, groups by this column value. Should be same length as the DataFrame.
        map_func_dict : `dict` [`str`, `callable`] or None, default None
            Grouping evaluation metric specification, along with ``agg_kwargs``.
            See `~greykite.framework.output.univariate_forecast.UnivariateForecast.get_flexible_grouping_evaluation`.
        agg_kwargs : `dict` or None, default None
            Grouping evaluation metric specification, along with ``map_func_dict``.
            See `~greykite.framework.output.univariate_forecast.UnivariateForecast.get_flexible_grouping_evaluation`.
        extend_col_names : `bool` or None, default False
            How to name the grouping metrics.
            See `~greykite.framework.output.univariate_forecast.UnivariateForecast.get_flexible_grouping_evaluation`.
        y_col_style_dict: `dict` [`str`, `dict` or None] or "plotly" or "auto" or "auto-fill", default "auto-fill"
            The column(s) to plot on the y-axis, and how to style them. The names should match
            those generated by ``agg_kwargs`` and ``extend_col_names``.
            The function
            `~greykite.framework.output.univariate_forecast.UnivariateForecast.get_flexible_grouping_evaluation`
            can be used to check the column names.

            For convenience, start with "auto-fill" or "plotly", then adjust styling as needed.

            See `~greykite.common.viz.timeseries_plotting.plot_multivariate` for details.

        default_color: `str`, default "rgba(0, 145, 202, 1.0)" (blue)
            Default line color when ``y_col_style_dict`` is one of "auto", "auto-fill".
        xlabel : `str` or None, default None
            x-axis label. If None, default is ``x_col``.
        ylabel : `str` or None, default None
            y-axis label. If None, y-axis is not labeled.
        title : `str` or None, default None
            Plot title. If None and ``ylabel`` is provided, a default title is used.
        showlegend : `bool`, default True
            Whether to show the legend.

        Returns
        -------
        fig : `plotly.graph_objects.Figure`
            Interactive plotly graph showing the evaluation metrics.

            See `~greykite.common.viz.timeseries_plotting.plot_forecast_vs_actual`
            return value for how to plot the figure and add customization.

        See Also
        --------
        `~greykite.framework.output.univariate_forecast.UnivariateForecast.get_flexible_grouping_evaluation` : called by this function
        `~greykite.common.viz.timeseries_plotting.plot_multivariate` : called by this function
        """
        grouped_df = self.get_flexible_grouping_evaluation(
            which=which,
            groupby_time_feature=groupby_time_feature,
            groupby_sliding_window_size=groupby_sliding_window_size,
            groupby_custom_column=groupby_custom_column,
            map_func_dict=map_func_dict,
            agg_kwargs=agg_kwargs,
            extend_col_names=extend_col_names)

        x_col = grouped_df.index.name
        grouped_df.reset_index(inplace=True)
        fig = plot_multivariate(
            grouped_df,
            x_col=x_col,
            y_col_style_dict=y_col_style_dict,
            default_color=default_color,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            showlegend=showlegend)
        return fig

    def make_univariate_time_series(self):
        """Converts prediction into a UnivariateTimeSeries
        Useful to convert a forecast into the input regressor for a subsequent forecast.

        :return: UnivariateTimeSeries
        """
        ts = UnivariateTimeSeries()
        df = (self.df[[self.time_col, self.predicted_col]]
              .rename({self.predicted_col: self.ylabel}, axis=1))
        ts.load_data(df, self.time_col, self.ylabel)
        return ts

    def plot_components(self, **kwargs):
        """Class method to plot the components of a `UnivariateForecast` object.

        ``Silverkite`` calculates component plots based on ``fit`` dataset.
        ``Prophet`` calculates component plots based on ``predict`` dataset.

        For estimator specific component plots with advanced plotting options call
        ``self.estimator.plot_components()``.

        Returns
        -------
        fig: `plotly.graph_objects.Figure` for ``Silverkite``
             `matplotlib.figure.Figure` for ``Prophet``
             Figure plotting components against appropriate time scale.
        """
        return self.estimator.plot_components(**kwargs)
