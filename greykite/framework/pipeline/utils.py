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
"""Utility functions for
`~greykite.framework.pipeline.pipeline`.
"""

import math
from functools import partial

import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from greykite.common import constants as cst
from greykite.common.constants import FRACTION_OUTSIDE_TOLERANCE
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.common.evaluation import add_finite_filter_to_scorer
from greykite.common.evaluation import add_preaggregation_to_scorer
from greykite.common.evaluation import fraction_outside_tolerance
from greykite.common.logging import LoggingLevelEnum
from greykite.common.logging import log_message
from greykite.common.python_utils import get_integer
from greykite.common.python_utils import unique_elements_in_list
from greykite.common.time_properties_forecast import get_default_horizon_from_period
from greykite.framework.constants import CUSTOM_SCORE_FUNC_NAME
from greykite.framework.constants import CV_REPORT_METRICS_ALL
from greykite.framework.constants import FRACTION_OUTSIDE_TOLERANCE_NAME
from greykite.framework.output.univariate_forecast import UnivariateForecast
from greykite.sklearn.estimator.simple_silverkite_estimator import SimpleSilverkiteEstimator
from greykite.sklearn.sklearn_scorer import make_scorer_df
from greykite.sklearn.transform.column_selector import ColumnSelector
from greykite.sklearn.transform.drop_degenerate_transformer import DropDegenerateTransformer
from greykite.sklearn.transform.dtype_column_selector import DtypeColumnSelector
from greykite.sklearn.transform.normalize_transformer import NormalizeTransformer
from greykite.sklearn.transform.null_transformer import NullTransformer
from greykite.sklearn.transform.pandas_feature_union import PandasFeatureUnion
from greykite.sklearn.transform.zscore_outlier_transformer import ZscoreOutlierTransformer


def get_best_index(results, metric="score", greater_is_better=False):
    """Suitable for use as the `refit` parameter to
    `~sklearn.model_selection.RandomizedSearchCV`, after wrapping
    with `functools.partial`.

    Callable that takes ``cv_results_`` from grid search
    and returns the best index.

    Parameters
    ----------
    results : `dict` [`str`, `numpy.array`]
        Results from CV grid search.
        See `~sklearn.model_selection.RandomizedSearchCV`
        ``cv_results_`` attribute for the format.
    metric : `str`, default "score"
        Which metric to use to select the best parameters.
        In single metric evaluation, the metric name should be "score".
        For multi-metric evaluation, the ``scoring`` parameter to
        `~sklearn.model_selection.RandomizedSearchCV` is a dictionary,
        and ``metric`` must be a key of ``scoring``.
    greater_is_better : `bool`, default False
        If True, selects the parameters with highest test values
        for ``metric``. Otherwise, selects those with the lowest
        test values for ``metric``.

    Returns
    -------
    best_index : `int`
        Best index to use for refitting the model.

    Examples
    --------
    >>> from functools import partial
    >>> from sklearn.model_selection import RandomizedSearchCV
    >>> refit = partial(get_best_index, metric="score", greater_is_better=False)
    >>> # RandomizedSearchCV(..., refit=refit)
    """
    if greater_is_better:
        # Note: in case of ties, the index corresponds to the first
        #   optimal value. But the order during CV may not match the order
        #   in fitted grid search ``.cv_results_`` attribute.
        # Note: "rank_test_{metric}" is ranked assuming greater_is_better=True,
        #   so these ranks are the opposite of the true ranks if greater_is_better=False.
        best_index = results[f"mean_test_{metric}"].argmax()
    else:
        best_index = results[f"mean_test_{metric}"].argmin()
    return best_index


def get_default_time_parameters(
        period,
        num_observations,
        forecast_horizon=None,
        test_horizon=None,
        periods_between_train_test=None,
        cv_horizon=None,
        cv_min_train_periods=None,
        cv_expanding_window=False,
        cv_periods_between_splits=None,
        cv_periods_between_train_test=None,
        cv_max_splits=3):
    """Returns default forecast horizon, backtest, and cross-validation parameters,
    given the input frequency, size, and user requested values.

    This function is called from the `~greykite.framework.pipeline.pipeline.forecast_pipeline`
    directly, to provide suitable default to users of forecast_pipeline, and because the default
    should not depend on model configuration (the template).

    Parameters
    ----------
    period: `float`
        Period of each observation (i.e. average time between observations, in seconds).
    num_observations: `int`
        Number of observations in the input data.
    forecast_horizon: `int` or None, default None
        Number of periods to forecast into the future. Must be > 0.
        If None, default is determined from input data frequency.
    test_horizon: `int` or None, default None
        Numbers of periods held back from end of df for test.
        The rest is used for cross validation.
        If None, default is ``forecast_horizon``. Set to 0 to skip backtest.
    periods_between_train_test : `int` or None, default None
        Number of periods gap between train and test in a CV split.
        If None, default is 0.
    cv_horizon: `int` or None, default None
        Number of periods in each CV test set.
        If None, default is ``forecast_horizon``. Set to 0 to skip CV.
    cv_min_train_periods: `int` or None, default None
        Minimum number of periods for training each CV fold.
        If ``cv_expanding_window`` is False, every training period is this size.
        If None, default is 2 * ``cv_horizon``.
    cv_expanding_window: `bool`, default False
        If True, training window for each CV split is fixed to the first available date.
        Otherwise, train start date is sliding, determined by ``cv_min_train_periods``.
    cv_periods_between_splits: `int` or None, default None
        Number of periods to slide the test window between CV splits
        If None, default is ``cv_horizon``.
    cv_periods_between_train_test: `int` or None, default None
        Number of periods gap between train and test in a CV split.
        If None, default is ``periods_between_train_test``.
    cv_max_splits: `int` or None, default 3
        Maximum number of CV splits. Given the above configuration, samples up to max_splits train/test splits,
        preferring splits toward the end of available data. If None, uses all splits.

    Returns
    -------
    time_params : `dict` [`str`, `int`]
        keys are parameter names, values are their default values.
    """
    if forecast_horizon is None:
        forecast_horizon = get_default_horizon_from_period(
            period=period,
            num_observations=num_observations)
    forecast_horizon = get_integer(val=forecast_horizon, name="forecast_horizon", min_value=1)

    test_horizon = get_integer(
        val=test_horizon,
        name="test_horizon",
        min_value=0,
        default_value=forecast_horizon)
    # reduces test_horizon to default 80/20 split if there is not enough data
    if test_horizon >= num_observations:
        test_horizon = math.floor(num_observations * 0.2)

    cv_horizon = get_integer(
        val=cv_horizon,
        name="cv_horizon",
        min_value=0,
        default_value=forecast_horizon)
    # RollingTimeSeriesSplit handles the case of no CV splits, not handled in detail here
    # temporary patch to avoid the case where cv_horizon==num_observations, which throws an error
    # in RollingTimeSeriesSplit
    if cv_horizon >= num_observations:
        cv_horizon = math.floor(num_observations * 0.2)

    periods_between_train_test = get_integer(
        val=periods_between_train_test,
        name="periods_between_train_test",
        min_value=0,
        default_value=0)

    cv_periods_between_train_test = get_integer(
        val=cv_periods_between_train_test,
        name="cv_periods_between_train_test",
        min_value=0,
        default_value=periods_between_train_test)

    return {
        "forecast_horizon": forecast_horizon,
        "test_horizon": test_horizon,
        "periods_between_train_test": periods_between_train_test,
        "cv_horizon": cv_horizon,
        "cv_min_train_periods": cv_min_train_periods,
        "cv_periods_between_train_test": cv_periods_between_train_test
    }


def get_basic_pipeline(
        estimator=SimpleSilverkiteEstimator(),
        score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name,
        score_func_greater_is_better=False,
        agg_periods=None,
        agg_func=None,
        relative_error_tolerance=None,
        coverage=0.95,
        null_model_params=None,
        regressor_cols=None,
        lagged_regressor_cols=None):
    """Returns a basic pipeline for univariate forecasting.
    Allows for outlier detection, normalization, null imputation,
    degenerate column removal, and forecast model fitting. By default,
    only null imputation is enabled. See source code for the pipeline steps.

    Notes
    -----
    While ``score_func`` is used to define the estimator's score function, the
    the ``scoring`` parameter of `~sklearn.model_selection.RandomizedSearchCV`
    should be provided when using this pipeline in grid search.
    Otherwise, grid search assumes higher values are better for ``score_func``.

    Parameters
    ----------
    estimator : instance of an estimator that implements `greykite.sklearn.estimator.base_forecast_estimator.BaseForecastEstimator`, default SimpleSilverkiteEstimator()  # noqa: E501
        Estimator to use as the final step in the pipeline.
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
    agg_periods : `int` or None, default None
        Number of periods to aggregate before evaluation.
        Model is fit at original frequency, and forecast is
        aggregated according to ``agg_periods``
        E.g. fit model on hourly data, and evaluate performance at daily level
        If None, does not apply aggregation
    agg_func : callable or None, default None
        Takes an array and returns a number, e.g. np.max, np.sum
        Used to aggregate data prior to evaluation (applied to actual and predicted)
        Ignored if ``agg_periods`` is None
    relative_error_tolerance : `float` or None, default None
        Threshold to compute the
        `~greykite.common.constants.FRACTION_OUTSIDE_TOLERANCE`
        metric, defined as the fraction of forecasted values whose relative
        error is strictly greater than ``relative_error_tolerance``.
        For example, 0.05 allows for 5% relative error.
        Required if ``score_func`` is
        `~greykite.common.constants.FRACTION_OUTSIDE_TOLERANCE`.
    coverage : `float` or None, default=0.95
        Intended coverage of the prediction bands (0.0 to 1.0)
        If None, the upper/lower predictions are not returned
        Ignored if `pipeline` is provided. Uses coverage of the ``pipeline`` estimator instead.
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
    regressor_cols : `list` [`str`] or None, default None
        A list of regressor columns used in the training and prediction DataFrames.
        It should contain only the regressors that are being used in the grid search.
        If None, no regressor columns are used.
        Regressor columns that are unavailable in ``df`` are dropped.
    lagged_regressor_cols: `list` [`str`] or None, default None
        A list of additional columns needed for lagged regressors in the training and prediction DataFrames.
        This list can have overlap with ``regressor_cols``.
        If None, no additional columns are added to the DataFrame.
        Lagged regressor columns that are unavailable in ``df`` are dropped.

    Returns
    -------
    pipeline : `sklearn.pipeline.Pipeline`
        sklearn Pipeline for univariate forecasting.
    """
    score_func, _, _ = get_score_func_with_aggregation(
        score_func=score_func,
        greater_is_better=score_func_greater_is_better,
        agg_periods=agg_periods,
        agg_func=agg_func,
        relative_error_tolerance=relative_error_tolerance)

    if regressor_cols is None:
        regressor_cols = []
    if lagged_regressor_cols is None:
        lagged_regressor_cols = []
    all_reg_cols = unique_elements_in_list(regressor_cols + lagged_regressor_cols)

    # A new unfitted estimator with the same parameters
    estimator_clone = clone(estimator)
    # Sets parameters common to all `BaseForecastEstimator`
    estimator_clone.set_params(
        score_func=score_func,
        coverage=coverage,
        null_model_params=null_model_params)

    # Note:
    #   Unlike typical ML, "y" (target values) is part of "X" (training data).
    #   Some forecasting models require that all historical values are available
    #   (no gaps in the timeseries). By including "y" values as part of "X", they
    #   can be transformed prior to fitting the estimator. This allows outlier removal and
    #   null imputation that respects train/test boundaries, to avoid leaking future
    #   information into the past. Evaluation is always done against original "y".
    #   Parameters for this pipeline are set via `hyperparameter_grid`.
    pipeline = Pipeline([
        ("input", PandasFeatureUnion([
            ("date", Pipeline([
                ("select_date", ColumnSelector([TIME_COL]))  # leaves time column unmodified
            ])),
            ("response", Pipeline([  # applies outlier and null transformation to value column
                ("select_val", ColumnSelector([VALUE_COL])),
                ("outlier", ZscoreOutlierTransformer(z_cutoff=None)),
                ("null", NullTransformer(impute_algorithm="interpolate"))
            ])),
            ("regressors_numeric", Pipeline([
                ("select_reg", ColumnSelector(all_reg_cols)),
                ("select_reg_numeric", DtypeColumnSelector(include="number")),
                ("outlier", ZscoreOutlierTransformer(z_cutoff=None)),
                ("normalize", NormalizeTransformer(normalize_algorithm=None)),  # no normalization by default
                ("null", NullTransformer(impute_algorithm="interpolate"))
            ])),
            ("regressors_other", Pipeline([
                ("select_reg", ColumnSelector(all_reg_cols)),
                ("select_reg_non_numeric", DtypeColumnSelector(exclude="number"))
            ]))
        ])),
        ("degenerate", DropDegenerateTransformer()),  # default `drop_degenerate=False`
        # Sets BaseForecastEstimator parameters (`score_func`, etc.).
        # Other parameters of the estimator are set by `hyperparameter_grid` later.
        ("estimator", estimator_clone)
    ])
    return pipeline


def get_score_func_with_aggregation(
        score_func,
        greater_is_better=None,
        agg_periods=None,
        agg_func=None,
        relative_error_tolerance=None):
    """Returns a score function that pre-aggregates inputs according to ``agg_func``,
    and filters out invalid true values before evaluation. This allows fitting
    the model at a granular level, yet evaluating at a coarser level.

    Also returns the proper direction and short name for the score function.

    Parameters
    ----------
    score_func : `str` or callable
        If callable, a function that maps two arrays to a number:
        ``(true, predicted) -> score``.
    greater_is_better : `bool`, default False
        True if ``score_func`` is a score function, meaning higher is better,
        and False if it is a loss function, meaning lower is better.
        Must be provided if ``score_func`` is a callable (custom function).
        Ignored if ``score_func`` is a string, because the direction is known.
    agg_periods : `int` or None, default None
        Number of periods to aggregate before evaluation.
        Model is fit at original frequency, and forecast is
        aggregated according to ``agg_periods``
        E.g. fit model on hourly data, and evaluate performance at daily level
        If None, does not apply aggregation
    agg_func : callable or None, default None
        Takes an array and returns a number, e.g. np.max, np.sum
        Used to aggregate data prior to evaluation (applied to actual and predicted)
        Ignored if ``agg_periods`` is None
    relative_error_tolerance : `float` or None, default None
        Threshold to compute the
        `~greykite.common.constants.FRACTION_OUTSIDE_TOLERANCE`
        metric, defined as the fraction of forecasted values whose relative
        error is strictly greater than ``relative_error_tolerance``.
        For example, 0.05 allows for 5% relative error.
        Required if ``score_func`` is
        `~greykite.common.constants.FRACTION_OUTSIDE_TOLERANCE`.

    Returns
    -------
    score_func : callable
        scorer with pre-aggregation function and filter,
    greater_is_better : `bool`
        Whether ``greater_is_better`` for the scorer.
        Uses the provided ``greater_is_better`` if the
        provided ``score_func`` is a callable.
        Otherwise, looks up the direction.
    short_name : `str`
        Canonical short name for the ``score_func``.
    """
    if isinstance(score_func, str):
        # If a string is provided, looks up the callable score_func and greater_is_better.
        # Otherwise, uses the args directly
        if score_func == FRACTION_OUTSIDE_TOLERANCE:
            if relative_error_tolerance is None:
                raise ValueError("Must specify `relative_error_tolerance` to request "
                                 "FRACTION_OUTSIDE_TOLERANCE as a metric.")
            score_func = partial(
                fraction_outside_tolerance,
                rtol=relative_error_tolerance)
            greater_is_better = False
            short_name = FRACTION_OUTSIDE_TOLERANCE_NAME
        else:
            try:
                enum = EvaluationMetricEnum[score_func]
                score_func = enum.get_metric_func()
                greater_is_better = enum.get_metric_greater_is_better()
                short_name = enum.get_metric_name()
            except KeyError:
                valid_names = ", ".join(EvaluationMetricEnum.__dict__["_member_names_"])
                raise NotImplementedError(f"Evaluation metric {score_func} is not available. Must be one of: {valid_names}")
    elif callable(score_func):
        # Uses the score_func and greater_is_better passed to the function
        short_name = CUSTOM_SCORE_FUNC_NAME
    else:
        raise ValueError("`score_func` must be an `EvaluationMetricEnum` member name, "
                         "FRACTION_OUTSIDE_TOLERANCE, or callable.")

    if agg_periods is not None:
        score_func = add_preaggregation_to_scorer(score_func, agg_periods, agg_func)

    # Filters out elements that can't be compared in ``y_true``
    score_func = add_finite_filter_to_scorer(score_func)

    return score_func, greater_is_better, short_name


def get_scoring_and_refit(
        score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name,
        score_func_greater_is_better=False,
        cv_report_metrics=None,
        agg_periods=None,
        agg_func=None,
        relative_error_tolerance=None):
    """Provides ``scoring`` and ``refit`` parameters for
    `~sklearn.model_selection.RandomizedSearchCV`.

    Together, ``scoring`` and ``refit`` specify how what metrics to evaluate and how
    to evaluate the predictions on the test set to identify the optimal model.

    Notes
    -----
    Sets ``greater_is_better=True`` in `scoring` for all metrics to report them with their
    original sign, and properly accounts for this in ``refit`` to extract the best index.

    Pass both `scoring` and `refit` to `~sklearn.model_selection.RandomizedSearchCV`

    Parameters
    ----------
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
    cv_report_metrics : `~greykite.common.constants.CV_REPORT_METRICS_ALL`, or `list` [`str`], or None, default None  # noqa: E501
        Additional metrics to compute during CV, besides the one specified by ``score_func``.

            - If the string constant `greykite.common.constants.CV_REPORT_METRICS_ALL`,
              computes all metrics in ``EvaluationMetricEnum``. Also computes
              ``FRACTION_OUTSIDE_TOLERANCE`` if ``relative_error_tolerance`` is not None.
              The results are reported by the short name (``.get_metric_name()``) for ``EvaluationMetricEnum``
              members and ``FRACTION_OUTSIDE_TOLERANCE_NAME`` for ``FRACTION_OUTSIDE_TOLERANCE``.
            - If a list of strings, each of the listed metrics is computed. Valid strings are
              `greykite.common.evaluation.EvaluationMetricEnum` member names
              and `~greykite.common.constants.FRACTION_OUTSIDE_TOLERANCE`.

              For example::

                ["MeanSquaredError", "MeanAbsoluteError", "MeanAbsolutePercentError", "MedianAbsolutePercentError", "FractionOutsideTolerance2"]

            - If None, no additional metrics are computed.
    agg_periods : `int` or None, default None
        Number of periods to aggregate before evaluation.
        Model is fit at original frequency, and forecast is
        aggregated according to ``agg_periods``
        E.g. fit model on hourly data, and evaluate performance at daily level
        If None, does not apply aggregation
    agg_func : callable or None, default None
        Takes an array and returns a number, e.g. np.max, np.sum
        Used to aggregate data prior to evaluation (applied to actual and predicted)
        Ignored if ``agg_periods`` is None
    relative_error_tolerance : `float` or None, default None
        Threshold to compute the
        `~greykite.common.constants.FRACTION_OUTSIDE_TOLERANCE`
        metric, defined as the fraction of forecasted values whose relative
        error is strictly greater than ``relative_error_tolerance``.
        For example, 0.05 allows for 5% relative error.
        If `None`, the metric is not computed.

    Returns
    -------
    scoring : `dict`
        A dictionary of metrics to evaluate for each CV split.
        The key is the metric name, the value is an instance
        of `~greykite.common.evaluation_PredictScorerDF` generated
        by :func:`~greykite.common.evaluation.make_scorer_df`.

        The value has a score method that takes actual and predicted values
        and returns a single number.

        There is one item in the dictionary for ``score_func``
        and an additional item for each additional element in
        ``cv_report_metrics``.

            - The key for ``score_func`` if it is a callable is
              `~greykite.common.constants.CUSTOM_SCORE_FUNC_NAME`.
            - The key for ``EvaluationMetricEnum`` member name is the short name
              from ``.get_metric_name()``.
            - The key for `~greykite.common.constants.FRACTION_OUTSIDE_TOLERANCE`
              is `~greykite.common.constants.FRACTION_OUTSIDE_TOLERANCE_NAME`.

        See `~sklearn.model_selection.RandomizedSearchCV`.

    refit : callable
        Callable that takes ``cv_results_`` from grid search
        and returns the best index.

        See `~sklearn.model_selection.RandomizedSearchCV`.
    """
    if cv_report_metrics is None:
        cv_report_metrics = []
    elif cv_report_metrics == CV_REPORT_METRICS_ALL:
        cv_report_metrics = EvaluationMetricEnum.__dict__["_member_names_"].copy()
        # Computes `FRACTION_OUTSIDE_TOLERANCE` if `relative_error_tolerance` is specified
        if relative_error_tolerance is not None:
            cv_report_metrics.append(FRACTION_OUTSIDE_TOLERANCE)
    else:
        cv_report_metrics = cv_report_metrics.copy()

    # Defines scoring metrics to evaluate on each CV split.
    # The results in .cv_results_ attribute are reported as
    # ``f"mean_test_{name}"``, ``f"mean_train_{name}"``, etc.,
    # where `name` is a key in the `scoring` dictionary.
    scoring = {}
    # Adds all recognized metrics in `cv_report_metrics` to `scoring`
    for metric_name in cv_report_metrics:
        # Adds aggregation to the metric function and gets the short name
        func, _, short_name = get_score_func_with_aggregation(
            score_func=metric_name,
            agg_periods=agg_periods,
            agg_func=agg_func,
            relative_error_tolerance=relative_error_tolerance)
        # Strings/functions typically accepted by
        #   `sklearn.model_selection.RandomizedSearchCV`
        #   are not compatible with the predictions returned by
        #   `~greykite.sklearn.estimator.base_forecast_estimator.BaseForecastEstimator`
        #   because predictions are returned as a dataframe.
        #   Thus, instead of ``"neg_median_absolute_error"``,
        #   use ``make_scorer_df(median_absolute_error, greater_is_better=False)``
        #   See `~greykite.common.evaluation.make_scorer_df` for details.
        # Uses `greater_is_better=True` to avoid flipping the metric sign in CV results.
        scoring[short_name] = make_scorer_df(
            score_func=func,
            greater_is_better=True)

    # Adds `score_func` to `scoring`
    func, greater_is_better, short_name = get_score_func_with_aggregation(
        score_func=score_func,  # string or callable
        greater_is_better=score_func_greater_is_better,
        agg_periods=agg_periods,
        agg_func=agg_func,
        relative_error_tolerance=relative_error_tolerance)
    scoring[short_name] = make_scorer_df(
        score_func=func,
        # Uses `greater_is_better=True` to avoid flipping the metric sign in CV results.
        # The `refit` parameter defined below should be passed to grid search to ensure proper
        # extraction of the optimal result, accounting for `greater_is_better` of the metric.
        greater_is_better=True)
    # Defines `refit` function with the true `greater_is_better` to pick the best result from CV.
    refit = partial(get_best_index, metric=short_name, greater_is_better=greater_is_better)
    return scoring, refit


def get_hyperparameter_searcher(
        hyperparameter_grid,
        model,
        cv=None,
        hyperparameter_budget=None,
        n_jobs=1,
        verbose=1,
        **kwargs) -> RandomizedSearchCV:
    """Returns RandomizedSearchCV object for hyperparameter tuning via cross validation

    `sklearn.model_selection.RandomizedSearchCV` runs a full grid search if
    ``hyperparameter_budget`` is sufficient to exhaust the full
    ``hyperparameter_grid``, otherwise it samples uniformly at random from the space.

    Parameters
    ----------
    hyperparameter_grid : `dict` or `list` [`dict`]
        Dictionary with parameters names (string) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        Lists of parameters are sampled uniformly.

        May also be a list of such dictionaries to avoid undesired combinations of parameters.
        Passed as ``param_distributions`` to `sklearn.model_selection.RandomizedSearchCV`,
        see docs for more info.
    model: estimator object
        A object of that type is instantiated for each grid point. This is assumed to implement
        the scikit-learn estimator interface.
    cv: `int`, cross-validation generator, iterable, or None, default None
        Determines the cross-validation splitting strategy.
        See `sklearn.model_selection.RandomizedSearchCV`.
    hyperparameter_budget: `int` or None, default None
        max number of hyperparameter sets to try within the hyperparameter_grid search space
        If None, uses defaults:

            * exhaustive grid search if all values are constant
            * 10 if any value is a distribution to sample from

    n_jobs : `int` or None, default 1
        Number of jobs to run in parallel
        (the maximum number of concurrently running workers).
        ``-1`` uses all CPUs. ``-2`` uses all CPUs but one.
        ``None`` is treated as 1 unless in a `joblib.Parallel` backend context
        that specifies otherwise.
    verbose : `int`, default 1
        Verbosity level during CV.

        * if > 0, prints number of fits
        * if > 1, prints fit parameters, total score + fit time
        * if > 2, prints train/test scores
    kwargs : additional parameters
        Keyword arguments to pass to `~greykite.framework.pipeline.utils.get_scoring_and_refit`.
        Accepts the following parameters:

            - ``"score_func"``
            - ``"score_func_greater_is_better"``
            - ``"cv_report_metrics"``
            - ``"agg_periods"``
            - ``"agg_func"``
            - ``"relative_error_tolerance"``

    Returns
    -------
    grid_search : `sklearn.model_selection.RandomizedSearchCV`
        Object that can run randomized search on hyper parameters.
    """
    if hyperparameter_budget is None:
        # sets reasonable defaults when hyperparameter_budget is not provided
        try:
            # exhaustive search if explicit values are provided
            hyperparameter_budget = len(ParameterGrid(hyperparameter_grid))
            log_message(f"Setting hyperparameter_budget to {hyperparameter_budget} for full grid search.",
                        LoggingLevelEnum.DEBUG)
        except TypeError:  # parameter value is not iterable
            # sets budget to 10 if distribution for randomized search is provided
            hyperparameter_budget = 10
            log_message(f"Setting hyperparameter_budget to {hyperparameter_budget} to sample from"
                        f" provided distributions (and lists).", LoggingLevelEnum.WARNING)

    scoring, refit = get_scoring_and_refit(**kwargs)

    # note: RandomizedSearchCV operates like GridSearchCV when hyperparameter_grid contains no distributions
    grid_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=hyperparameter_grid,  # a fixed list or distribution to sample from
        n_iter=hyperparameter_budget,             # samples uniformly, up to hyperparameter_budget
        scoring=scoring,                          # model evaluation criteria (note: if None, uses the score function of the estimator)
        n_jobs=n_jobs,                            # parallelism
        refit=refit,                              # selects the best model
        cv=cv,
        verbose=verbose,
        pre_dispatch="2*n_jobs",                  # controls memory consumption
        return_train_score=True                   # NB: could be False for speedup
    )
    return grid_search


def get_forecast(
        df,
        trained_model: Pipeline,
        train_end_date=None,
        test_start_date=None,
        forecast_horizon=None,
        xlabel=cst.TIME_COL,
        ylabel=cst.VALUE_COL,
        relative_error_tolerance=None) -> UnivariateForecast:
    """Runs model predictions on ``df`` and creates a
    `~greykite.framework.output.univariate_forecast.UnivariateForecast` object.

    Parameters
    ----------
    df: `pandas.DataFrame`
        Has columns cst.TIME_COL, cst.VALUE_COL, to forecast.
    trained_model: `sklearn.pipeline`
        A fitted Pipeline with ``estimator`` step and predict function.
    train_end_date: `datetime.datetime`, default `None`
        Train end date. Passed to
        `~greykite.framework.output.univariate_forecast.UnivariateForecast`.
    test_start_date: `datetime.datetime`, default `None`
        Test start date. Passed to
        `~greykite.framework.output.univariate_forecast.UnivariateForecast`.
    forecast_horizon : `int` or None, default None
        Number of periods forecasted into the future. Must be > 0. Passed to
        `~greykite.framework.output.univariate_forecast.UnivariateForecast`.
    xlabel: `str`
        Time column to use in representing forecast (e.g. x-axis in plots).
    ylabel: `str`
        Time column to use in representing forecast (e.g. y-axis in plots).
    relative_error_tolerance : `float` or None, default None
        Threshold to compute the ``Outside Tolerance`` metric,
        defined as the fraction of forecasted values whose relative
        error is strictly greater than ``relative_error_tolerance``.
        For example, 0.05 allows for 5% relative error.
        If `None`, the metric is not computed.

    Returns
    -------
    univariate_forecast : `~greykite.framework.output.univariate_forecast.UnivariateForecast`
        Forecasts represented as a ``UnivariateForecast`` object.
    """
    predicted_df = trained_model.predict(df)
    # This is more robust than using trained_model.named_steps["estimator"] e.g.
    # if the user calls forecast_pipeline with a custom pipeline, where the last
    # step isn't named "estimator".
    trained_estimator = trained_model.steps[-1][-1]
    coverage = trained_estimator.coverage

    # combines actual with predictions
    union_df = pd.DataFrame({
        xlabel: df[cst.TIME_COL].values,
        # .values here, since df and predicted_df have different indexes
        cst.ACTUAL_COL: df[cst.VALUE_COL].values,
        # evaluation and plots are done on the values *before* any transformations
        cst.PREDICTED_COL: predicted_df[cst.PREDICTED_COL].values
    })

    predicted_lower_col = None
    predicted_upper_col = None
    null_model_predicted_col = None

    # adds lower bound if available
    if cst.PREDICTED_LOWER_COL in predicted_df.columns:
        predicted_lower_col = cst.PREDICTED_LOWER_COL
        union_df[cst.PREDICTED_LOWER_COL] = predicted_df[cst.PREDICTED_LOWER_COL].values
        if coverage is None:
            raise Exception("coverage must be provided")

    # adds upper bound if available
    if cst.PREDICTED_UPPER_COL in predicted_df.columns:
        predicted_upper_col = cst.PREDICTED_UPPER_COL
        union_df[cst.PREDICTED_UPPER_COL] = predicted_df[cst.PREDICTED_UPPER_COL].values
        if coverage is None:
            raise Exception("coverage must be provided")

    # adds null prediction if available
    if trained_estimator.null_model is not None:
        null_model_predicted_col = cst.NULL_PREDICTED_COL
        null_predicted_df = trained_estimator.null_model.predict(df)
        union_df[cst.NULL_PREDICTED_COL] = null_predicted_df[cst.PREDICTED_COL].values

    return UnivariateForecast(
        union_df,
        time_col=xlabel,
        actual_col=cst.ACTUAL_COL,
        predicted_col=cst.PREDICTED_COL,
        predicted_lower_col=predicted_lower_col,
        predicted_upper_col=predicted_upper_col,
        null_model_predicted_col=null_model_predicted_col,
        ylabel=ylabel,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
        forecast_horizon=forecast_horizon,
        coverage=coverage,
        r2_loss_function=trained_estimator.score_func,  # this score_func includes preaggregation if requested
        estimator=trained_estimator,
        relative_error_tolerance=relative_error_tolerance
    )
