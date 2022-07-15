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
# original author: Sayan Patra
"""Class for benchmarking model templates."""

from typing import Dict
from typing import List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import DEFAULT_PLOTLY_COLORS
from tqdm.autonotebook import tqdm

from greykite.common.constants import ACTUAL_COL
from greykite.common.constants import PREDICTED_COL
from greykite.common.constants import PREDICTED_LOWER_COL
from greykite.common.constants import PREDICTED_UPPER_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.common.evaluation import add_finite_filter_to_scorer
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message
from greykite.common.python_utils import get_pattern_cols
from greykite.common.viz.timeseries_plotting import plot_multivariate
from greykite.common.viz.timeseries_plotting import plot_multivariate_grouped
from greykite.framework.benchmark.benchmark_class_helper import forecast_pipeline_rolling_evaluation
from greykite.framework.constants import FORECAST_STEP_COL
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.forecaster import Forecaster
from greykite.sklearn.cross_validation import RollingTimeSeriesSplit


class BenchmarkForecastConfig:
    """Class for benchmarking multiple ForecastConfig on a rolling window basis.

    Attributes
    ----------
    df : `pandas.DataFrame`
        Timeseries data to forecast.
        Contains columns [`time_col`, `value_col`], and optional regressor columns.
        Regressor columns should include future values for prediction.

    configs : `Dict` [`str`,  `ForecastConfig`]
        Dictionary of model configurations.
        A model configuration is a ``ForecastConfig``.
        See :class:`~greykite.framework.templates.autogen.forecast_config.ForecastConfig` for details on
        valid ``ForecastConfig``.
        Validity of the ``configs`` for benchmarking is checked via the ``validate`` method.

    tscv : `~greykite.sklearn.cross_validation.RollingTimeSeriesSplit`
        Cross-validation object that determines the rolling window evaluation.
        See :class:`~greykite.sklearn.cross_validation.RollingTimeSeriesSplit` for details.
        The ``forecast_horizon`` and ``periods_between_train_test`` parameters of ``configs`` are
        matched against that of ``tscv``. A ValueError is raised if there is a mismatch.

    forecaster : `~greykite.framework.templates.forecaster.Forecaster`
        Forecaster used to create the forecasts.

    is_run : bool, default False
        Indicator of whether the `run` method is executed.
        After executing `run`, this indicator is set to True.
        Some class methods like ``get_forecast`` requires ``is_run`` to be True
        to be executed.

    result : `dict`
        Stores the benchmarking results. Has the same keys as ``configs``.

    forecasts : `pandas.DataFrame`, default None
        Merged DataFrame of forecasts, upper and lower confidence interval for all
        input ``configs``. Also stores train end date and forecast step for each prediction.
    """

    def __init__(
            self,
            df: pd.DataFrame,
            configs: Dict[str, ForecastConfig],
            tscv: RollingTimeSeriesSplit,
            forecaster: Forecaster = Forecaster()):
        self.df = df
        self.configs = configs
        self.tscv = tscv
        self.forecaster = forecaster

        self.is_run = False

        # output
        self.result = dict.fromkeys(configs.keys())
        self.forecasts = None

    def validate(self):
        """Validates the inputs to the class for the method ``run``.

        Raises a ValueError if there is a mismatch between the following parameters
        of ``configs`` and ``tscv``:

         - ``forecast_horizon``
         - ``periods_between_train_test``

        Raises ValueError if all the ``configs`` do not have the same ``coverage`` parameter.
        """
        coverage_list = []
        for config_name, config in self.configs.items():
            # Computes pipeline parameters
            pipeline_params = self.forecaster.apply_forecast_config(
                df=self.df,
                config=config)

            # Checks forecast_horizon
            if pipeline_params["forecast_horizon"] != self.tscv.forecast_horizon:
                raise ValueError(f"{config_name}'s 'forecast_horizon' ({config.forecast_horizon}) does "
                                 f"not match that of 'tscv' ({self.tscv.forecast_horizon}).")

            # Checks periods_between_train_test
            if pipeline_params["periods_between_train_test"] is None:
                pipeline_params["periods_between_train_test"] = 0
            if pipeline_params["periods_between_train_test"] != self.tscv.periods_between_train_test:
                raise ValueError(f"{config_name}'s 'periods_between_train_test' ({pipeline_params['periods_between_train_test']}) "
                                 f"does not match that of 'tscv' ({self.tscv.periods_between_train_test}).")

            self.result[config_name] = dict(pipeline_params=pipeline_params)
            coverage_list.append(config.coverage)

        # Checks all coverages are same
        if coverage_list[1:] != coverage_list[:-1]:
            raise ValueError("All forecast configs must have same coverage.")

    def run(self):
        """Runs every config and stores the output of the
        :func:`~greykite.framework.pipeline.pipeline.forecast_pipeline`.
        This function runs only if the ``configs`` and ``tscv`` are jointly valid.

        Returns
        -------
        self : Returns self. Stores pipeline output of every config in ``self.result``.
        """
        self.validate()
        with tqdm(self.result.items(), ncols=800, leave=True) as progress_bar:
            for (config_name, config) in progress_bar:
                # Description will be displayed on the left of progress bar
                progress_bar.set_description(f"Benchmarking '{config_name}' ")
                rolling_evaluation = forecast_pipeline_rolling_evaluation(
                    pipeline_params=config["pipeline_params"],
                    tscv=self.tscv)
                config["rolling_evaluation"] = rolling_evaluation

        self.is_run = True

    def extract_forecasts(self):
        """Extracts forecasts, upper and lower confidence interval for each individual
        config. This is saved as a ``pandas.DataFrame`` with the name
        ``rolling_forecast_df`` within the corresponding config of ``self.result``.
        e.g. if config key is "silverkite", then the forecasts are stored in
        ``self.result["silverkite"]["rolling_forecast_df"]``.

        This method also constructs a merged DataFrame of forecasts,
        upper and lower confidence interval for all input ``configs``.
        """
        if not self.is_run:
            raise ValueError("Please execute 'run' method to create forecasts.")

        merged_df = pd.DataFrame()
        for config_name, config in self.result.items():
            rolling_evaluation = config["rolling_evaluation"]
            rolling_forecast_df = pd.DataFrame()
            for num, (split_key, split_value) in enumerate(rolling_evaluation.items()):
                forecast = split_value["pipeline_result"].forecast
                # Subsets forecast_horizon rows from the end of forecast dataframe
                forecast_df = forecast.df.iloc[-forecast.forecast_horizon:]
                forecast_df.insert(0, "train_end_date", forecast.train_end_date)
                forecast_df.insert(1, FORECAST_STEP_COL, np.arange(forecast.forecast_horizon) + 1)
                forecast_df.insert(2, "split_num", num)
                rolling_forecast_df = pd.concat([rolling_forecast_df, forecast_df], axis=0)
            rolling_forecast_df = rolling_forecast_df.reset_index(drop=True)
            self.result[config_name]["rolling_forecast_df"] = rolling_forecast_df

            # Merges the forecasts of individual config
            # Augments prediction columns with config name
            pred_cols = [PREDICTED_COL]
            if PREDICTED_LOWER_COL in rolling_forecast_df.columns:
                pred_cols.append(PREDICTED_LOWER_COL)
            if PREDICTED_UPPER_COL in rolling_forecast_df.columns:
                pred_cols.append(PREDICTED_UPPER_COL)
            mapper = {
                col: f"{config_name}_{col}" for col in pred_cols
            }
            if merged_df.empty:
                temp_df = rolling_forecast_df.rename(columns=mapper)
            else:
                temp_df = rolling_forecast_df[pred_cols].rename(columns=mapper)

            merged_df = pd.concat([merged_df, temp_df], axis=1)

        self.forecasts = merged_df.reset_index(drop=True)

    def plot_forecasts_by_step(
            self,
            forecast_step: int,
            config_names: List = None,
            xlabel: str = TIME_COL,
            ylabel: str = VALUE_COL,
            title: str = None,
            showlegend: bool = True):
        """Returns a ``forecast_step`` ahead rolling forecast plot.
        The plot consists one line for each valid. ``config_names``.
        If available, the corresponding actual values are also plotted.

        For a more customizable plot, see
        :func:`~greykite.common.viz.timeseries_plotting.plot_multivariate`

        Parameters
        ----------
        forecast_step : `int`
            Which forecast step to plot. A forecast step is an integer between 1 and the
            forecast horizon, inclusive, indicating the number of periods from train end date
            to the prediction date (# steps ahead).
        config_names : `list` [`str`], default None
            Which config results to plot. A list of config names.
            If None, uses all the available config keys.
        xlabel : `str` or None, default TIME_COL
            x-axis label.
        ylabel : `str` or None, default VALUE_COL
            y-axis label.
        title : `str` or None, default None
            Plot title. If None, default is based on ``forecast_step``.
        showlegend : `bool`, default True
            Whether to show the legend.

        Returns
        -------
        fig : `plotly.graph_objects.Figure`
            Interactive plotly graph.
            Plots multiple column(s) in ``self.forecasts`` against ``TIME_COL``.

            See `~greykite.common.viz.timeseries_plotting.plot_forecast_vs_actual`
            return value for how to plot the figure and add customization.
        """
        if self.forecasts is None:
            self.extract_forecasts()

        if forecast_step > self.tscv.forecast_horizon:
            raise ValueError(f"`forecast_step` ({forecast_step}) must be less than or equal to "
                             f"forecast horizon ({self.tscv.forecast_horizon}).")

        config_names = self.get_valid_config_names(config_names)
        y_cols = [TIME_COL, ACTUAL_COL] + \
                 [f"{config_name}_{PREDICTED_COL}" for config_name in config_names]

        df = self.forecasts[self.forecasts[FORECAST_STEP_COL] == forecast_step]
        df = df[y_cols]

        if title is None:
            title = f"{forecast_step}-step ahead rolling forecasts"
        fig = plot_multivariate(
            df=df,
            x_col=TIME_COL,
            y_col_style_dict="plotly",
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            showlegend=showlegend)

        return fig

    def plot_forecasts_by_config(
            self,
            config_name: str,
            colors: List = DEFAULT_PLOTLY_COLORS,
            xlabel: str = TIME_COL,
            ylabel: str = VALUE_COL,
            title: str = None,
            showlegend: bool = True):
        """Returns a rolling plot of the forecasts by ``config_name`` against ``TIME_COL``.
        The plot consists of one line for each available split. Some lines may overlap if test
        period in corresponding splits intersect. Hence every line is given a different color.
        If available, the corresponding actual values are also plotted.

        For a more customizable plot, see
        :func:`~greykite.common.viz.timeseries_plotting.plot_multivariate_grouped`

        Parameters
        ----------
        config_name : `str`
            Which config result to plot.
            The name must match the name of one of the input ``configs``.
        colors : [`str`, `List` [`str`]], default ``DEFAULT_PLOTLY_COLORS``
            Which colors to use to build the color palette.
            This can be a list of RGB colors or a `str` from ``PLOTLY_SCALES``.
            To use a single color for all lines, pass a `List` with a single color.
        xlabel : `str` or None, default TIME_COL
            x-axis label.
        ylabel : `str` or None, default VALUE_COL
            y-axis label.
        title : `str` or None, default None
            Plot title. If None, default is based on ``config_name``.
        showlegend : `bool`, default True
            Whether to show the legend.

        Returns
        -------
        fig : `plotly.graph_objects.Figure`
            Interactive plotly graph.
            Plots multiple column(s) in ``self.forecasts`` against ``TIME_COL``.
        """
        if self.forecasts is None:
            self.extract_forecasts()

        config_name = self.get_valid_config_names([config_name])[0]

        if title is None:
            title = f"Rolling forecast for {config_name}"
        fig = plot_multivariate_grouped(
            df=self.forecasts,
            x_col=TIME_COL,
            y_col_style_dict={
                ACTUAL_COL: {
                    "line": {
                        "width": 1,
                        "dash": "solid"
                    }
                }
            },
            grouping_x_col="split_num",
            grouping_x_col_values=None,
            grouping_y_col_style_dict={
                f"{config_name}_{PREDICTED_COL}": {
                    "name": "split",
                    "line": {
                        "width": 1,
                        "dash": "solid"
                    }
                }
            },
            colors=colors,
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            showlegend=showlegend)

        return fig

    def get_evaluation_metrics(
            self,
            metric_dict: Dict,
            config_names: List = None):
        """Returns rolling train and test evaluation metric values.

        Parameters
        ----------
        metric_dict : `dict` [`str`, `callable`]
            Evaluation metrics to compute.

                - key: evaluation metric name, used to create column name in output.
                - value: metric function to apply to forecast df in each split to generate the column value.
                        Signature (y_true: `str`, y_pred: `str`) -> transformed value: `float`.

            For example::

                metric_dict = {
                    "median_residual": lambda y_true, y_pred: np.median(y_true - y_pred),
                    "mean_squared_error": lambda y_true, y_pred: np.mean((y_true - y_pred)**2)
                }

            Some predefined functions are available in
            `~greykite.common.evaluation`. For example::

                metric_dict = {
                    "correlation": lambda y_true, y_pred: correlation(y_true, y_pred),
                    "RMSE": lambda y_true, y_pred: root_mean_squared_error(y_true, y_pred),
                    "Q_95": lambda y_true, y_pred: partial(quantile_loss(y_true, y_pred, q=0.95))
                }

            As shorthand, it is sufficient to provide the corresponding ``EvaluationMetricEnum``
            member.  These are auto-expanded into the appropriate function.
            So the following is equivalent::

                metric_dict = {
                    "correlation": EvaluationMetricEnum.Correlation,
                    "RMSE": EvaluationMetricEnum.RootMeanSquaredError,
                    "Q_95": EvaluationMetricEnum.Quantile95
                }

        config_names : `list` [`str`], default None
            Which config results to plot. A list of config names.
            If None, uses all the available config keys.

        Returns
        -------
        evaluation_metrics_df : pd.DataFrame
            A DataFrame containing splitwise train and test evaluation metrics for ``metric_dict``
            and ``config_names``.

            For example. Let's assume::

                metric_dict = {
                    "RMSE": EvaluationMetricEnum.RootMeanSquaredError,
                    "Q_95": EvaluationMetricEnum.Quantile95
                }

                config_names = ["default_prophet", "custom_silverkite"]
                These are valid ``config_names`` and there are 2 splits for each.

                Then evaluation_metrics_df =

                config_name     split_num   train_RMSE  test_RMSE   train_Q_95  test_Q_95
                default_prophet      0          *           *           *           *
                default_prophet      1          *           *           *           *
                custom_silverkite    0          *           *           *           *
                custom_silverkite    1          *           *           *           *

                where * represents computed values.
        """
        if not self.is_run:
            raise ValueError("Please execute the 'run' method before computing evaluation metrics.")

        metric_dict = self.autocomplete_metric_dict(
            metric_dict=metric_dict,
            enum_class=EvaluationMetricEnum)

        config_names = self.get_valid_config_names(config_names=config_names)

        evaluation_metrics_df = pd.DataFrame()
        for config_name in config_names:
            rolling_evaluation = self.result[config_name]["rolling_evaluation"]
            for num, (split_key, split_value) in enumerate(rolling_evaluation.items()):
                forecast = split_value["pipeline_result"].forecast
                split_metrics = {
                    "config_name": config_name,
                    "split_num": num}
                # Updates train metrics
                df_train = forecast.df_train
                split_metrics.update({
                    f"train_{metric_name}": metric_func(
                        df_train[forecast.actual_col],
                        df_train[forecast.predicted_col]
                    ) for metric_name, metric_func in metric_dict.items()
                })
                # Updates test metrics
                df_test = forecast.df_test
                if df_test.shape[0] > 0 and forecast.test_na_count < df_test.shape[0]:
                    split_metrics.update({
                        f"test_{metric_name}": metric_func(
                            df_test[forecast.actual_col],
                            df_test[forecast.predicted_col]
                        ) for metric_name, metric_func in metric_dict.items()
                    })
                else:
                    split_metrics.update({
                        f"test_{metric_name}": np.nan
                        for metric_name, metric_func in metric_dict.items()
                    })

                split_metrics_df = pd.DataFrame(split_metrics, index=[num])
                evaluation_metrics_df = pd.concat([evaluation_metrics_df, split_metrics_df])
        # Resets index and fills missing values (e.g. when correlation is not defined) with np.nan
        evaluation_metrics_df = evaluation_metrics_df.reset_index(drop=True).fillna(value=np.nan)
        temp_df = evaluation_metrics_df.copy()
        # Rearranges columns so that train and test error of a config are side by side
        evaluation_metrics_df = pd.DataFrame()
        evaluation_metrics_df["config_name"] = temp_df["config_name"]
        evaluation_metrics_df["split_num"] = temp_df["split_num"]
        for metric_name in metric_dict.keys():
            evaluation_metrics_df[f"train_{metric_name}"] = temp_df[f"train_{metric_name}"]
            evaluation_metrics_df[f"test_{metric_name}"] = temp_df[f"test_{metric_name}"]

        return evaluation_metrics_df

    def plot_evaluation_metrics(
            self,
            metric_dict: Dict,
            config_names: List = None,
            xlabel: str = None,
            ylabel: str = "Metric value",
            title: str = None,
            showlegend: bool = True):
        """Returns a barplot of the train and test values of ``metric_dict`` of ``config_names``.
        Value of a metric for all ``config_names`` are plotted as a grouped bar.
        Train and test values of a metric are plot side-by-side for easy comparison.

        Parameters
        ----------
        metric_dict : `dict` [`str`, `callable`]
            Evaluation metrics to compute. Same as
            `~greykite.framework.framework.benchmark.benchmark_class.BenchmarkForecastConfig.get_evaluation_metrics`.
            To get the best visualization, keep number of metrics <= 2.
        config_names : `list` [`str`], default None
            Which config results to plot. A list of config names.
            If None, uses all the available config keys.
        xlabel : `str` or None, default None
            x-axis label.
        ylabel : `str` or None, default "Metric value"
            y-axis label.
        title : `str` or None, default None
            Plot title.
        showlegend : `bool`, default True
            Whether to show the legend.

        Returns
        -------
         fig : `plotly.graph_objects.Figure`
            Interactive plotly bar plot.
        """
        evaluation_metrics_df = self.get_evaluation_metrics(
            metric_dict=metric_dict,
            config_names=config_names)

        # This function groups by config name
        evaluation_metrics_df = (evaluation_metrics_df
                                 .drop(columns=["split_num"])
                                 .groupby("config_name")
                                 .mean()
                                 .dropna(how="all"))

        # Rearranges columns so that train and test error of a config are side by side
        plot_df = pd.DataFrame()
        for metric_name in metric_dict.keys():
            plot_df[f"train_{metric_name}"] = evaluation_metrics_df[f"train_{metric_name}"]
            plot_df[f"test_{metric_name}"] = evaluation_metrics_df[f"test_{metric_name}"]

        if title is None:
            title = "Average evaluation metric across rolling windows"
        data = []
        # Each row (index) is a config. Adds each row to the bar plot.
        for index in plot_df.index:
            data.append(
                go.Bar(
                    name=index,
                    x=plot_df.columns,
                    y=plot_df.loc[index].values
                )
            )
        layout = go.Layout(
            xaxis=dict(title=xlabel),
            yaxis=dict(title=ylabel),
            title=title,
            title_x=0.5,
            showlegend=showlegend,
            barmode="group",
        )
        fig = go.Figure(data=data, layout=layout)

        return fig

    def get_grouping_evaluation_metrics(
            self,
            metric_dict: Dict,
            config_names: List = None,
            which: str = "train",
            groupby_time_feature: str = None,
            groupby_sliding_window_size: int = None,
            groupby_custom_column: pd.Series = None):
        """Returns splitwise rolling evaluation metric values.
         These values are grouped by the grouping method chosen by ``groupby_time_feature``,
         ``groupby_sliding_window_size`` and ``groupby_custom_column``.
        See `~greykite.framework.output.univariate_forecast.UnivariateForecast.get_grouping_evaluation`
        for details on grouping method.

         Parameters
        ----------
        metric_dict : `dict` [`str`, `callable`]
            Evaluation metrics to compute. Same as
            `~greykite.framework.framework.benchmark.benchmark_class.BenchmarkForecastConfig.get_evaluation_metrics`.
        config_names : `list` [`str`], default None
            Which config results to plot. A list of config names.
            If None, uses all the available config keys.
        which: `str`
            "train" or "test". Which dataset to evaluate.
        groupby_time_feature : `str` or None, default None
            If provided, groups by a column generated by
            `~greykite.common.features.timeseries_features.build_time_features_df`.
            See that function for valid values.
        groupby_sliding_window_size : `int` or None, default None
            If provided, sequentially partitions data into groups of size
            ``groupby_sliding_window_size``.
        groupby_custom_column : `pandas.Series` or None, default None
            If provided, groups by this column value. Should be same length as the DataFrame.

        Returns
        -------
        grouped_evaluation_df : `pandas.DataFrame`
            A DataFrame containing splitwise train and test evaluation metrics for ``metric_dict``
            and ``config_names``. The evaluation metrics are grouped by the grouping method.
        """
        if not self.is_run:
            raise ValueError("Please execute the 'run' method before computing "
                             "grouped evaluation metrics.")

        metric_dict = self.autocomplete_metric_dict(
            metric_dict=metric_dict,
            enum_class=EvaluationMetricEnum)

        config_names = self.get_valid_config_names(config_names=config_names)

        grouped_evaluation_df = pd.DataFrame()
        for config_name in config_names:
            rolling_evaluation = self.result[config_name]["rolling_evaluation"]
            for num, (split_key, split_value) in enumerate(rolling_evaluation.items()):
                forecast = split_value["pipeline_result"].forecast
                split_evaluation_df = pd.DataFrame()
                for metric_name, metric_func in metric_dict.items():
                    grouped_df = forecast.get_grouping_evaluation(
                        score_func=metric_func,
                        score_func_name=metric_name,
                        which=which,
                        groupby_time_feature=groupby_time_feature,
                        groupby_sliding_window_size=groupby_sliding_window_size,
                        groupby_custom_column=groupby_custom_column)
                    # Adds grouped_df to split_evaluation_df, handling the case if split_evaluation_df is empty
                    # If the actual values are missing, grouped_df.shape[0] might be 0
                    if grouped_df.shape[0] > 0:
                        if split_evaluation_df.empty:
                            split_evaluation_df = grouped_df
                        else:
                            groupby_col = split_evaluation_df.columns[0]
                            split_evaluation_df = pd.merge(split_evaluation_df, grouped_df, on=groupby_col)
                    else:
                        # This column name is the same as that obtained from
                        # `~greykite.framework.output.univariate_forecast.UnivariateForecast.get_grouping_evaluation`
                        split_evaluation_df[f"{which} {metric_name}"] = np.nan
                split_evaluation_df.insert(0, "config_name", config_name)
                split_evaluation_df.insert(1, "split_num", num)
                grouped_evaluation_df = pd.concat([grouped_evaluation_df, split_evaluation_df])
        grouped_evaluation_df = grouped_evaluation_df.reset_index(drop=True)

        return grouped_evaluation_df

    def plot_grouping_evaluation_metrics(
            self,
            metric_dict: Dict,
            config_names: List = None,
            which: str = "train",
            groupby_time_feature: str = None,
            groupby_sliding_window_size: int = None,
            groupby_custom_column: pd.Series = None,
            xlabel=None,
            ylabel="Metric value",
            title=None,
            showlegend=True):
        """Returns a line plot of the grouped evaluation values of ``metric_dict`` of ``config_names``.
        These values are grouped by the grouping method chosen by ``groupby_time_feature``,
         ``groupby_sliding_window_size`` and ``groupby_custom_column``.
        See `~greykite.framework.output.univariate_forecast.UnivariateForecast.get_grouping_evaluation`
        for details on grouping method.

         Parameters
        ----------
        metric_dict : `dict` [`str`, `callable`]
            Evaluation metrics to compute. Same as
            `~greykite.framework.framework.benchmark.benchmark_class.BenchmarkForecastConfig.get_evaluation_metrics`.
            To get the best visualization, keep number of metrics <= 2.
        config_names : `list` [`str`], default None
            Which config results to plot. A list of config names.
            If None, uses all the available config keys.
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
        xlabel : `str` or None, default None
            x-axis label. If None, label is determined by the groupby column name.
        ylabel : `str` or None, default "Metric value"
            y-axis label.
        title : `str` or None, default None
            Plot title. If None, default is based on ``config_name``.
        showlegend : `bool`, default True
            Whether to show the legend.

        Returns
        -------
         fig : `plotly.graph_objects.Figure`
            Interactive plotly graph.
        """
        grouped_evaluation_df = self.get_grouping_evaluation_metrics(
            metric_dict=metric_dict,
            config_names=config_names,
            which=which,
            groupby_time_feature=groupby_time_feature,
            groupby_sliding_window_size=groupby_sliding_window_size,
            groupby_custom_column=groupby_custom_column)

        # Figures out groupby_col name by process of elimination
        cols = [col for col in grouped_evaluation_df.columns if col not in ["config_name", "split_num"]]
        groupby_col = get_pattern_cols(cols, pos_pattern=".*", neg_pattern=which)[0]

        plot_df = (grouped_evaluation_df
                   .drop(columns=["split_num"])            # Drops redundant column
                   .groupby(["config_name", groupby_col])  # Averages values across splits
                   .mean()
                   .dropna(how="all")                      # Drops rows with all NA values
                   .unstack(level=0)                       # Moves config_name from multiindex rows to multiindex columns
                   .sort_index(axis=1)                     # Sorts on groupby_col to plot groups in logical order
                   )

        # Flattens and renames multiindex columns
        cols = [groupby_col] + ["_".join(v) for v in plot_df.columns]
        plot_df = pd.DataFrame(plot_df.to_records())
        plot_df.columns = cols

        if xlabel is None:
            xlabel = groupby_col
        if title is None:
            title = f"{which} performance by {xlabel} across rolling windows"
        fig = plot_multivariate(
            df=plot_df,
            x_col=groupby_col,
            y_col_style_dict="plotly",
            xlabel=xlabel,
            ylabel=ylabel,
            title=title,
            showlegend=showlegend)

        return fig

    def get_runtimes(self, config_names: List = None):
        """Returns rolling average runtime in seconds for ``config_names``.

        Parameters
        ----------
        config_names : `list` [`str`], default None
            Which config results to plot. A list of config names.
            If None, uses all the available config keys.

        Returns
        -------
        runtimes_df : pd.DataFrame
            A DataFrame containing splitwise runtime in seconds for ``config_names``.

            For example. Let's assume::

                config_names = ["default_prophet", "custom_silverkite"]
                These are valid ``config_names`` and there are 2 splits for each.

                Then runtimes_df =

                config_name     split_num   runtime_sec
                default_prophet      0          *
                default_prophet      1          *
                custom_silverkite    0          *
                custom_silverkite    1          *

                where * represents computed values.
        """
        if not self.is_run:
            raise ValueError("Please execute the 'run' method to obtain runtimes.")

        config_names = self.get_valid_config_names(config_names=config_names)
        runtimes_df = pd.DataFrame()
        for config_name in config_names:
            rolling_evaluation = self.result[config_name]["rolling_evaluation"]
            for num, (split_key, split_value) in enumerate(rolling_evaluation.items()):
                split_runtime_df = pd.DataFrame({
                    "config_name": config_name,
                    "split_num": num,
                    "runtime_sec": split_value["runtime_sec"]
                }, index=[num])
                runtimes_df = pd.concat([runtimes_df, split_runtime_df])

        return runtimes_df.reset_index(drop=True)

    def plot_runtimes(
            self,
            config_names: List = None,
            xlabel: str = None,
            ylabel: str = "Mean runtime in seconds",
            title: str = "Average runtime across rolling windows",
            showlegend: bool = True):
        """Returns a barplot of the runtimes of ``config_names``.

        Parameters
        ----------
        config_names : `list` [`str`], default None
            Which config results to plot. A list of config names.
            If None, uses all the available config keys.
        xlabel : `str` or None, default None
            x-axis label.
        ylabel : `str` or None, default "Mean runtime in seconds"
            y-axis label.
        title : `str` or None, default "Average runtime across rolling windows"
            Plot title.
        showlegend : `bool`, default True
            Whether to show the legend.

        Returns
        -------
         fig : `plotly.graph_objects.Figure`
            Interactive plotly bar plot.
        """
        runtimes_df = self.get_runtimes(config_names=config_names)

        plot_df = runtimes_df.drop(columns=["split_num"]).groupby("config_name").mean()
        data = [go.Bar(x=plot_df.index, y=plot_df["runtime_sec"], name="Runtime")]
        layout = go.Layout(
            xaxis=dict(title=xlabel),
            yaxis=dict(title=ylabel),
            title=title,
            title_x=0.5,
            showlegend=showlegend,
        )
        fig = go.Figure(data=data, layout=layout)

        return fig

    def get_valid_config_names(self, config_names: List = None):
        """Validate ``config_names`` against keys of ``configs``.
        Raises a ValueError in case of a mismatch.

        Parameters
        ----------
         config_names : `list` [`str`], default None
            Which config results to plot. A list of config names.
            If None, uses all the available config keys.

        Returns
        -------
        config_names : `list`
            List of valid config names.
        """
        available_config_names = list(self.configs.keys())
        if config_names is None:
            config_names = available_config_names
        else:
            missing_config_names = set(config_names) - set(available_config_names)
            if len(missing_config_names) > 0:
                raise ValueError(f"The following config keys are missing: {missing_config_names}.")

        return config_names

    @staticmethod
    def autocomplete_metric_dict(metric_dict, enum_class):
        """Sweeps through ``metric_dict``, converting members of ``enum_class`` to
        their corresponding evaluation function.

        For example::

            metric_dict = {
                "correlation": EvaluationMetricEnum.Correlation,
                "RMSE": EvaluationMetricEnum.RootMeanSquaredError,
                "Q_95": EvaluationMetricEnum.Quantile95
                "custom_metric": custom_function
            }

            is converted to

            metric_dict = {
                "correlation": correlation(y_true, y_pred),
                "RMSE": root_mean_squared_error(y_true, y_pred),
                "Q_95": quantile_loss_q(y_true, y_pred, q=0.95),
                "custom_function": custom_function
            }

        Parameters
        ----------
        metric_dict : `dict` [`str`, `callable`]
            Evaluation metrics to compute. Same as
            `~greykite.framework.framework.benchmark.benchmark_class.BenchmarkForecastConfig.get_evaluation_metrics`.
        enum_class : Enum
            The enum class ``metric_dict`` elements might be member of.
            It must have a method ``get_metric_func``.

        Returns
        -------
        updated_metric_dict : `dict`
            Autocompleted metric dict.
        """
        updated_metric_dict = {}
        for metric_name, metric_value in metric_dict.items():
            if isinstance(metric_value, enum_class):
                updated_metric_dict[metric_name] = metric_value.get_metric_func()
            else:
                updated_metric_dict[metric_name] = add_finite_filter_to_scorer(metric_value)
                if not callable(metric_value):
                    raise ValueError(f"Value of '{metric_name}' should be a callable or a member of {enum_class}.")

        return updated_metric_dict

    def save(self):
        log_message("Benchmark save is not implemented yet.", LoggingLevelEnum.WARNING)

    def summary(self):
        log_message("Benchmark summary is not implemented yet.", LoggingLevelEnum.WARNING)
