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
"""Computation pipeline for end-to-end forecasting
using any ``BaseForecastEstimator``.
Runs cross validation, backtest, and forecast.
"""

import functools
import warnings
from dataclasses import dataclass

import pandas as pd
from sklearn import clone
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message
from greykite.common.python_utils import get_integer
from greykite.common.time_properties import min_gap_in_seconds
from greykite.framework.constants import COMPUTATION_N_JOBS
from greykite.framework.constants import CV_REPORT_METRICS_ALL
from greykite.framework.input.univariate_time_series import UnivariateTimeSeries
from greykite.framework.output.univariate_forecast import UnivariateForecast
from greykite.framework.pipeline.utils import get_basic_pipeline
from greykite.framework.pipeline.utils import get_default_time_parameters
from greykite.framework.pipeline.utils import get_forecast
from greykite.framework.pipeline.utils import get_hyperparameter_searcher
from greykite.sklearn.cross_validation import RollingTimeSeriesSplit
from greykite.sklearn.estimator.simple_silverkite_estimator import SimpleSilverkiteEstimator


@dataclass
class ForecastResult:
    """Forecast results. Contains results from cross-validation, backtest, and forecast,
    the trained model, and the original input data.
    """
    timeseries: UnivariateTimeSeries = None
    """Input time series in standard format with stats and convenient plot functions."""
    grid_search: RandomizedSearchCV = None
    """Result of cross-validation grid search on training dataset. The relevant attributes are:

        * ``cv_results_`` cross-validation scores
        * ``best_estimator_`` the model used for backtesting
        * ``best_params_`` the optimal parameters used for backtesting.

    Also see `~greykite.framework.utils.result_summary.summarize_grid_search_results`.
    We recommend using this function to extract results, rather than accessing ``cv_results_`` directly.
    """
    model: Pipeline = None
    """Model fitted on full dataset, using the best parameters selected via cross-validation.
    Has ``.fit()``, ``.predict()``, and diagnostic functions depending on the model.
    """
    backtest: UnivariateForecast = None
    """Forecast on backtest period.
    Backtest period is a holdout test set to check forecast quality against
    the most recent actual values available.
    The best model from cross validation is refit on data prior to this period.
    The timestamps in ``backtest.df`` are sorted in ascending order.
    Has a ``.plot()`` method and attributes to get
    forecast vs actuals, evaluation results.
    """
    forecast: UnivariateForecast = None
    """Forecast on future period.
    Future dates are after the train end date, following the holdout test set.
    The best model from cross validation is refit on data prior to this period.
    The timestamps in ``forecast.df`` are sorted in ascending order.
    Has a ``.plot()`` method and attributes to get
    forecast vs actuals, evaluation results.
    """


def validate_pipeline_input(pipeline_function):
    """Decorator that validates inputs to forecast_pipeline function and sets defaults"""
    @functools.wraps(pipeline_function)
    def pipeline_wrapper(
            # The arguments to this wrapper must be identical to forecast_pipeline() function.
            # We don't use **kwargs
            # because it's easier to check parameters directly.
            # input
            df: pd.DataFrame,
            time_col=TIME_COL,
            value_col=VALUE_COL,
            date_format=None,
            tz=None,
            freq=None,
            train_end_date=None,
            anomaly_info=None,
            # model
            pipeline=None,
            regressor_cols=None,
            lagged_regressor_cols=None,
            estimator=SimpleSilverkiteEstimator(),
            hyperparameter_grid=None,
            hyperparameter_budget=None,
            n_jobs=COMPUTATION_N_JOBS,
            verbose=1,
            # forecast
            forecast_horizon=None,
            coverage=0.95,
            test_horizon=None,
            periods_between_train_test=None,
            agg_periods=None,
            agg_func=None,
            # evaluation
            score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name,
            score_func_greater_is_better=False,
            cv_report_metrics=None,
            null_model_params=None,
            relative_error_tolerance=None,
            # CV
            cv_horizon=None,
            cv_min_train_periods=None,
            cv_expanding_window=False,
            cv_use_most_recent_splits=False,
            cv_periods_between_splits=None,
            cv_periods_between_train_test=0,
            cv_max_splits=3):
        if coverage is not None and (coverage < 0 or coverage > 1):
            raise ValueError(f"coverage must be between 0 and 1, found {coverage}")
        if relative_error_tolerance is not None and relative_error_tolerance < 0:
            raise ValueError(f"relative_error_tolerance must non-negative, found {relative_error_tolerance}")

        # default values for forecast horizon, test, and cross-validation parameters
        period = min_gap_in_seconds(df=df, time_col=time_col)
        num_observations = df.shape[0]
        default_time_params = get_default_time_parameters(
            period=period,
            num_observations=num_observations,
            forecast_horizon=forecast_horizon,
            test_horizon=test_horizon,
            periods_between_train_test=periods_between_train_test,
            cv_horizon=cv_horizon,
            cv_min_train_periods=cv_min_train_periods,
            cv_periods_between_train_test=cv_periods_between_train_test)
        forecast_horizon = default_time_params.get("forecast_horizon")
        test_horizon = default_time_params.get("test_horizon")
        periods_between_train_test = default_time_params.get("periods_between_train_test")
        cv_horizon = default_time_params.get("cv_horizon")
        cv_min_train_periods = default_time_params.get("cv_min_train_periods")
        cv_periods_between_train_test = default_time_params.get("cv_periods_between_train_test")

        # ensures the values are integers in the proper domain
        if hyperparameter_budget is not None:
            hyperparameter_budget = get_integer(
                hyperparameter_budget,
                "hyperparameter_budget",
                min_value=1)

        if (cv_horizon == 0 or cv_max_splits == 0) and test_horizon == 0:
            warnings.warn("Both CV and backtest are skipped! Make sure this is intended."
                          " It's important to check model performance on historical data."
                          " Set cv_horizon and cv_max_splits to nonzero values to enable CV."
                          " Set test_horizon to nonzero value to enable backtest.")

        if test_horizon == 0:
            warnings.warn("No data selected for test (test_horizon=0). "
                          "It is important to check out of sample performance")

        # checks horizon against data size
        if num_observations < forecast_horizon * 2:
            warnings.warn(f"Not enough training data to forecast the full forecast_horizon."
                          " Exercise extra caution with"
                          f" forecasted values after {num_observations // 2} periods.")

        if test_horizon > num_observations:
            raise ValueError(f"test_horizon ({test_horizon}) is too large."
                             " Must be less than the number "
                             f"of input data points: {num_observations})")

        if test_horizon > forecast_horizon:
            warnings.warn(f"test_horizon should never be larger than forecast_horizon.")

        if test_horizon > num_observations // 3:
            warnings.warn(f"test_horizon should be <= than 1/3 of the data set size to allow enough data to train"
                          f" a backtest model. Consider reducing to {num_observations // 3}. If this is smaller"
                          f" than the forecast_horizon, you will need to make a trade-off between setting"
                          f" test_horizon=forecast_horizon and having enough data left over to properly"
                          f" train a realistic backtest model.")

        log_message(f"forecast_horizon: {forecast_horizon}", LoggingLevelEnum.INFO)
        log_message(f"test_horizon: {test_horizon}", LoggingLevelEnum.INFO)
        log_message(f"cv_horizon: {cv_horizon}", LoggingLevelEnum.INFO)

        return pipeline_function(
            df,
            time_col=time_col,
            value_col=value_col,
            date_format=date_format,
            tz=tz,
            freq=freq,
            train_end_date=train_end_date,
            anomaly_info=anomaly_info,
            pipeline=pipeline,
            regressor_cols=regressor_cols,
            lagged_regressor_cols=lagged_regressor_cols,
            estimator=estimator,
            hyperparameter_grid=hyperparameter_grid,
            hyperparameter_budget=hyperparameter_budget,
            n_jobs=n_jobs,
            verbose=verbose,
            forecast_horizon=forecast_horizon,
            coverage=coverage,
            test_horizon=test_horizon,
            periods_between_train_test=periods_between_train_test,
            agg_periods=agg_periods,
            agg_func=agg_func,
            score_func=score_func,
            score_func_greater_is_better=score_func_greater_is_better,
            cv_report_metrics=cv_report_metrics,
            null_model_params=null_model_params,
            relative_error_tolerance=relative_error_tolerance,
            cv_horizon=cv_horizon,
            cv_min_train_periods=cv_min_train_periods,
            cv_expanding_window=cv_expanding_window,
            cv_use_most_recent_splits=cv_use_most_recent_splits,
            cv_periods_between_splits=cv_periods_between_splits,
            cv_periods_between_train_test=cv_periods_between_train_test,
            cv_max_splits=cv_max_splits
        )
    return pipeline_wrapper


@validate_pipeline_input
def forecast_pipeline(
        # input
        df: pd.DataFrame,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        date_format=None,
        tz=None,
        freq=None,
        train_end_date=None,
        anomaly_info=None,
        # model
        pipeline=None,
        regressor_cols=None,
        lagged_regressor_cols=None,
        estimator=SimpleSilverkiteEstimator(),
        hyperparameter_grid=None,
        hyperparameter_budget=None,
        n_jobs=COMPUTATION_N_JOBS,
        verbose=1,
        # forecast
        forecast_horizon=None,
        coverage=0.95,
        test_horizon=None,
        periods_between_train_test=None,
        agg_periods=None,
        agg_func=None,
        # evaluation
        score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name,
        score_func_greater_is_better=False,
        cv_report_metrics=CV_REPORT_METRICS_ALL,
        null_model_params=None,
        relative_error_tolerance=None,
        # CV
        cv_horizon=None,
        cv_min_train_periods=None,
        cv_expanding_window=False,
        cv_use_most_recent_splits=False,
        cv_periods_between_splits=None,
        cv_periods_between_train_test=None,
        cv_max_splits=3):
    """Computation pipeline for end-to-end forecasting.

    Trains a forecast model end-to-end:

        1. checks input data
        2. runs cross-validation to select optimal hyperparameters e.g. best model
        3. evaluates best model on test set
        4. provides forecast of best model (re-trained on all data) into the future

    Returns forecasts with methods to plot and see diagnostics.
    Also returns the fitted pipeline and CV results.

    Provides a high degree of customization over training and evaluation parameters:

        1. model
        2. cross validation
        3. evaluation
        4. forecast horizon

    See test cases for examples.

    Parameters
    ----------
    df : `pandas.DataFrame`
        Timeseries data to forecast.
        Contains columns [`time_col`, `value_col`], and optional regressor columns
        Regressor columns should include future values for prediction

    time_col : `str`, default TIME_COL in constants.py
        name of timestamp column in df

    value_col : `str`, default VALUE_COL in constants.py
        name of value column in df (the values to forecast)

    date_format : `str` or None, default None
        strftime format to parse time column, eg ``%m/%d/%Y``.
        Note that ``%f`` will parse all the way up to nanoseconds.
        If None (recommended), inferred by `pandas.to_datetime`.

    tz : `str` or None, default None
        Passed to `pandas.tz_localize` to localize the timestamp

    freq : `str` or None, default None
        Frequency of input data. Used to generate future dates for prediction.
        Frequency strings can have multiples, e.g. '5H'.
        See https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
        for a list of frequency aliases.
        If None, inferred by `pandas.infer_freq`.
        Provide this parameter if ``df`` has missing timepoints.

    train_end_date : `datetime.datetime`, optional, default None
        Last date to use for fitting the model. Forecasts are generated after this date.
        If None, it is set to the last date with a non-null value in
        ``value_col`` of ``df``.

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

    pipeline : `sklearn.pipeline.Pipeline` or None, default None
        Pipeline to fit. The final named step must be called "estimator".
        If None, will use the default Pipeline from
        `~greykite.framework.pipeline.utils.get_basic_pipeline`.

    regressor_cols : `list` [`str`] or None, default None
        A list of regressor columns used in the training and prediction DataFrames.
        It should contain only the regressors that are being used in the grid search.
        If None, no regressor columns are used.
        Regressor columns that are unavailable in ``df`` are dropped.

    lagged_regressor_cols : `list` [`str`] or None, default None
        A list of additional columns needed for lagged regressors in the training and prediction DataFrames.
        This list can have overlap with ``regressor_cols``.
        If None, no additional columns are added to the DataFrame.
        Lagged regressor columns that are unavailable in ``df`` are dropped.

    estimator : instance of an estimator that implements `greykite.algo.models.base_forecast_estimator.BaseForecastEstimator`
        Estimator to use as the final step in the pipeline.
        Ignored if ``pipeline`` is provided.

    forecast_horizon : `int` or None, default None
        Number of periods to forecast into the future. Must be > 0.
        If None, default is determined from input data frequency

    coverage : `float` or None, default=0.95
        Intended coverage of the prediction bands (0.0 to 1.0)
        If None, the upper/lower predictions are not returned
        Ignored if `pipeline` is provided. Uses coverage of the ``pipeline`` estimator instead.

    test_horizon : `int` or None, default None
        Numbers of periods held back from end of df for test.
        The rest is used for cross validation.
        If None, default is forecast_horizon. Set to 0 to skip backtest.

    periods_between_train_test : `int` or None, default None
        Number of periods for the gap between train and test data.
        If None, default is 0.

    agg_periods : `int` or None, default None
        Number of periods to aggregate before evaluation.

        Model is fit and forecasted on the dataset's original frequency.

        Before evaluation, the actual and forecasted values are aggregated,
        using rolling windows of size ``agg_periods`` and the function
        ``agg_func``. (e.g. if the dataset is hourly, use ``agg_periods=24, agg_func=np.sum``,
        to evaluate performance on the daily totals).

        If None, does not aggregate before evaluation.

        Currently, this is only used when calculating CV metrics and
        the R2_null_model_score metric in backtest/forecast. No pre-aggregation
        is applied for the other backtest/forecast evaluation metrics.

    agg_func : callable or None, default None
        Takes an array and returns a number, e.g. np.max, np.sum.

        Defines how to aggregate rolling windows of actual and predicted values
        before evaluation.

        Ignored if ``agg_periods`` is None.

        Currently, this is only used when calculating CV metrics and
        the R2_null_model_score metric in backtest/forecast. No pre-aggregation
        is applied for the other backtest/forecast evaluation metrics.

    score_func : `str` or callable, default ``EvaluationMetricEnum.MeanAbsolutePercentError.name``
        Score function used to select optimal model in CV.
        If a callable, takes arrays ``y_true``, ``y_pred`` and returns a float.
        If a string, must be either a
        `~greykite.common.evaluation.EvaluationMetricEnum` member name
        or `~greykite.common.constants.FRACTION_OUTSIDE_TOLERANCE`.

    score_func_greater_is_better : `bool`, default False
        True if ``score_func`` is a score function, meaning higher is better,
        and False if it is a loss function, meaning lower is better.
        Must be provided if ``score_func`` is a callable (custom function).
        Ignored if ``score_func`` is a string, because the direction is known.

    cv_report_metrics : `str`, or `list` [`str`], or None, default `~greykite.common.constants.CV_REPORT_METRICS_ALL`
        Additional metrics to compute during CV, besides the one specified by ``score_func``.

            - If the string constant `greykite.framework.constants.CV_REPORT_METRICS_ALL`,
              computes all metrics in ``EvaluationMetricEnum``. Also computes
              ``FRACTION_OUTSIDE_TOLERANCE`` if ``relative_error_tolerance`` is not None.
              The results are reported by the short name (``.get_metric_name()``) for ``EvaluationMetricEnum``
              members and ``FRACTION_OUTSIDE_TOLERANCE_NAME`` for ``FRACTION_OUTSIDE_TOLERANCE``.
              These names appear in the keys of ``forecast_result.grid_search.cv_results_``
              returned by this function.
            - If a list of strings, each of the listed metrics is computed. Valid strings are
              `~greykite.common.evaluation.EvaluationMetricEnum` member names
              and `~greykite.common.constants.FRACTION_OUTSIDE_TOLERANCE`.

              For example::

                ["MeanSquaredError", "MeanAbsoluteError", "MeanAbsolutePercentError", "MedianAbsolutePercentError", "FractionOutsideTolerance2"]

            - If None, no additional metrics are computed.

    null_model_params : `dict` or None, default None
        Defines baseline model to compute ``R2_null_model_score`` evaluation metric.
        ``R2_null_model_score`` is the improvement in the loss function relative
        to a null model. It can be used to evaluate model quality with respect to
        a simple baseline. For details, see
        `~greykite.common.evaluation.r2_null_model_score`.

        The null model is a `~sklearn.dummy.DummyRegressor`,
        which returns constant predictions.

        Valid keys are "strategy", "constant", "quantile".
        See `~sklearn.dummy.DummyRegressor`. For example::

            null_model_params = {
                "strategy": "mean",
            }
            null_model_params = {
                "strategy": "median",
            }
            null_model_params = {
                "strategy": "quantile",
                "quantile": 0.8,
            }
            null_model_params = {
                "strategy": "constant",
                "constant": 2.0,
            }

        If None, ``R2_null_model_score`` is not calculated.

        Note: CV model selection always optimizes ``score_func`, not
        the ``R2_null_model_score``.

    relative_error_tolerance : `float` or None, default None
        Threshold to compute the ``Outside Tolerance`` metric,
        defined as the fraction of forecasted values whose relative
        error is strictly greater than ``relative_error_tolerance``.
        For example, 0.05 allows for 5% relative error.
        If `None`, the metric is not computed.

    hyperparameter_grid : `dict`, `list` [`dict`] or None, default None
        Sets properties of the steps in the pipeline,
        and specifies combinations to search over.
        Should be valid input to `sklearn.model_selection.GridSearchCV` (param_grid)
        or `sklearn.model_selection.RandomizedSearchCV` (param_distributions).

        Prefix transform/estimator attributes by the name of the step in the pipeline.
        See details at: https://scikit-learn.org/stable/modules/compose.html#nested-parameters

        If None, uses the default pipeline parameters.

    hyperparameter_budget : `int` or None, default None
        Max number of hyperparameter sets to try within the ``hyperparameter_grid`` search space

        Runs a full grid search if ``hyperparameter_budget`` is sufficient to exhaust full
        ``hyperparameter_grid``, otherwise samples uniformly at random from the space.

        If None, uses defaults:

            * full grid search if all values are constant
            * 10 if any value is a distribution to sample from

    n_jobs : `int` or None, default `~greykite.framework.constants.COMPUTATION_N_JOBS`
        Number of jobs to run in parallel
        (the maximum number of concurrently running workers).
        ``-1`` uses all CPUs. ``-2`` uses all CPUs but one.
        ``None`` is treated as 1 unless in a `joblib.Parallel` backend context
        that specifies otherwise.

    verbose : `int`, default 1
        Verbosity level during CV.
        if > 0, prints number of fits
        if > 1, prints fit parameters, total score + fit time
        if > 2, prints train/test scores

    cv_horizon : `int` or None, default None
        Number of periods in each CV test set
        If None, default is ``forecast_horizon``.
        Set either ``cv_horizon`` or ``cv_max_splits`` to 0 to skip CV.

    cv_min_train_periods : `int` or None, default None
        Minimum number of periods for training each CV fold.
        If cv_expanding_window is False, every training period is this size
        If None, default is 2 * ``cv_horizon``

    cv_expanding_window : `bool`, default False
        If True, training window for each CV split is fixed to the first available date.
        Otherwise, train start date is sliding, determined by ``cv_min_train_periods``.

    cv_use_most_recent_splits: `bool`, default False
        If True, splits from the end of the dataset are used.
        Else a sampling strategy is applied. Check
        `~greykite.sklearn.cross_validation.RollingTimeSeriesSplit._sample_splits`
        for details.

    cv_periods_between_splits : `int` or None, default None
        Number of periods to slide the test window between CV splits
        If None, default is ``cv_horizon``

    cv_periods_between_train_test : `int` or None, default None
        Number of periods for the gap between train and test in a CV split.
        If None, default is ``periods_between_train_test``.

    cv_max_splits : `int` or None, default 3
        Maximum number of CV splits.
        Given the above configuration, samples up to max_splits train/test splits,
        preferring splits toward the end of available data. If None, uses all splits.
        Set either ``cv_horizon`` or ``cv_max_splits`` to 0 to skip CV.

    Returns
    -------
    forecast_result : :class:`~greykite.framework.pipeline.pipeline.ForecastResult`
        Forecast result. See :class:`~greykite.framework.pipeline.pipeline.ForecastResult`
        for details.

            * If ``cv_horizon=0``, ``forecast_result.grid_search.best_estimator_``
              and ``forecast_result.grid_search.best_params_`` attributes are defined
              according to the provided single set of parameters. There must be a single
              set of parameters to skip cross-validation.
            * If ``test_horizon=0``, ``forecast_result.backtest`` is None.
    """
    if hyperparameter_grid is None or hyperparameter_grid == []:
        hyperparameter_grid = {}
    # When hyperparameter_grid is a singleton list, unlist it
    if isinstance(hyperparameter_grid, list) and len(hyperparameter_grid) == 1:
        hyperparameter_grid = hyperparameter_grid[0]

    # Loads full dataset
    ts = UnivariateTimeSeries()
    ts.load_data(
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

    # Splits data into training and test sets. ts.df uses standardized column names
    if test_horizon == 0:
        train_df = ts.fit_df
        train_y = ts.fit_y
        test_df = pd.DataFrame(columns=list(df.columns))
    else:
        # Make sure to refit best_pipeline appropriately
        train_df, test_df, train_y, test_y = train_test_split(
            ts.fit_df,
            ts.fit_y,
            train_size=ts.fit_df.shape[0] - test_horizon - periods_between_train_test,
            test_size=test_horizon + periods_between_train_test,
            shuffle=False)  # this is important since this is timeseries forecasting!
    log_message(f"Train size: {train_df.shape[0]}. Test size: {test_df.shape[0]}", LoggingLevelEnum.INFO)

    # Defines default training pipeline
    if pipeline is None:
        pipeline = get_basic_pipeline(
            estimator=estimator,
            score_func=score_func,
            score_func_greater_is_better=score_func_greater_is_better,
            agg_periods=agg_periods,
            agg_func=agg_func,
            relative_error_tolerance=relative_error_tolerance,
            coverage=coverage,
            null_model_params=null_model_params,
            regressor_cols=ts.regressor_cols,
            lagged_regressor_cols=ts.lagged_regressor_cols)

    # Searches for the best parameters, and refits model with selected parameters on the entire training set
    if cv_horizon == 0 or cv_max_splits == 0:
        # No cross-validation. Only one set of hyperparameters is allowed.
        try:
            if len(ParameterGrid(hyperparameter_grid)) > 1:
                raise ValueError(
                    "CV is required to identify the best model because there are multiple options "
                    "in `hyperparameter_grid`. Either provide a single option or set `cv_horizon` and `cv_max_splits` "
                    "to nonzero values.")
        except TypeError:  # Parameter value is not iterable
            raise ValueError(
                "CV is required to identify the best model because `hyperparameter_grid` contains "
                "a distribution. Either remove the distribution or set `cv_horizon` and `cv_max_splits` "
                "to nonzero values.")

        # Fits model to entire train set. Params must be set manually since it's not done by grid search
        params = {k: v[0] for k, v in hyperparameter_grid.items()}  # unpack lists, `v` is a singleton list with the parameter value
        best_estimator = pipeline.set_params(**params).fit(train_df, train_y)

        # Wraps this model in a dummy RandomizedSearchCV object to return the backtest model
        grid_search = get_hyperparameter_searcher(
            hyperparameter_grid=hyperparameter_grid,
            model=pipeline,
            cv=None,  # no cross-validation
            hyperparameter_budget=hyperparameter_budget,
            n_jobs=n_jobs,
            verbose=verbose,
            score_func=score_func,
            score_func_greater_is_better=score_func_greater_is_better,
            cv_report_metrics=cv_report_metrics,
            agg_periods=agg_periods,
            agg_func=agg_func,
            relative_error_tolerance=relative_error_tolerance)
        # Sets relevant attributes. Others are undefined (cv_results_, best_score_, best_index_, scorer_, refit_time_)
        grid_search.best_estimator_ = best_estimator
        grid_search.best_params_ = params
        grid_search.n_splits_ = 0
    else:
        # Defines cross-validation splitter
        cv = RollingTimeSeriesSplit(
            forecast_horizon=cv_horizon,
            min_train_periods=cv_min_train_periods,
            expanding_window=cv_expanding_window,
            use_most_recent_splits=cv_use_most_recent_splits,
            periods_between_splits=cv_periods_between_splits,
            periods_between_train_test=cv_periods_between_train_test,
            max_splits=cv_max_splits)

        # Defines grid search approach for CV
        grid_search = get_hyperparameter_searcher(
            hyperparameter_grid=hyperparameter_grid,
            model=pipeline,
            cv=cv,
            hyperparameter_budget=hyperparameter_budget,
            n_jobs=n_jobs,
            verbose=verbose,
            score_func=score_func,
            score_func_greater_is_better=score_func_greater_is_better,
            cv_report_metrics=cv_report_metrics,
            agg_periods=agg_periods,
            agg_func=agg_func,
            relative_error_tolerance=relative_error_tolerance)
        grid_search.fit(train_df, train_y)
        best_estimator = grid_search.best_estimator_

    # Evaluates historical performance, fits model to all data (train+test)
    if test_horizon > 0:
        backtest_train_end_date = train_df[TIME_COL].max()
        # Uses pd.date_range because pd.Timedelta does not work for complicated frequencies e.g. "W-MON"
        backtest_test_start_date = pd.date_range(
            start=backtest_train_end_date,
            periods=periods_between_train_test + 2,  # Adds 2 as start parameter is inclusive
            freq=ts.freq)[-1]
        backtest = get_forecast(
            df=ts.fit_df,  # Backtest needs to happen on fit_df, not on the entire df
            trained_model=best_estimator,
            train_end_date=backtest_train_end_date,
            test_start_date=backtest_test_start_date,
            forecast_horizon=test_horizon,
            xlabel=time_col,
            ylabel=value_col,
            relative_error_tolerance=relative_error_tolerance)
        best_pipeline = clone(best_estimator)  # Copies optimal parameters
        best_pipeline.fit(ts.fit_df, ts.y)  # Refits this model on entire training dataset
    else:
        backtest = None  # Backtest training metrics are the same as forecast training metrics
        best_pipeline = best_estimator  # best_model is already fit to all data

    # Makes future predictions
    periods = forecast_horizon + periods_between_train_test
    future_df = ts.make_future_dataframe(
        periods=periods,
        include_history=True)

    forecast_train_end_date = ts.train_end_date
    # Uses pd.date_range because pd.Timedelta does not work for complicated frequencies e.g. "W-MON"
    forecast_test_start_date = pd.date_range(
        start=forecast_train_end_date,
        periods=periods_between_train_test + 2,  # Adds 2 as start parameter is inclusive
        freq=ts.freq)[-1]
    forecast = get_forecast(
        df=future_df,
        trained_model=best_pipeline,
        train_end_date=forecast_train_end_date,
        test_start_date=forecast_test_start_date,
        forecast_horizon=forecast_horizon,
        xlabel=time_col,
        ylabel=value_col,
        relative_error_tolerance=relative_error_tolerance)

    result = ForecastResult(
        timeseries=ts,
        grid_search=grid_search,
        model=best_pipeline,
        backtest=backtest,
        forecast=forecast
    )
    return result
