import warnings
from functools import partial
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from greykite.common.constants import ACTUAL_COL
from greykite.common.constants import PREDICTED_COL
from greykite.common.constants import PREDICTION_BAND_COVERAGE
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.common.evaluation import fraction_outside_tolerance
from greykite.common.python_utils import assert_equal
from greykite.common.testing_utils import assert_eval_function_equal
from greykite.framework.constants import FRACTION_OUTSIDE_TOLERANCE_NAME
from greykite.framework.output.univariate_forecast import UnivariateForecast
from greykite.framework.pipeline.pipeline import validate_pipeline_input
from greykite.framework.pipeline.utils import get_score_func_with_aggregation
from greykite.framework.utils.result_summary import summarize_grid_search_results
from greykite.sklearn.estimator.simple_silverkite_estimator import SimpleSilverkiteEstimator
from greykite.sklearn.sklearn_scorer import _PredictScorerDF


def assert_refit(
        refit,
        expected_metric,
        expected_greater_is_better):
    """Checks if `refit` is initialized with
        metric = expected_metric,
        and greater_is_better = expected_greater_is_better.

    Parameters
    ----------
    refit : callable
        `refit` callable to check
    expected_metric : `str`
        expected metric used to define ``refit``
    expected_greater_is_better : `bool`
        expected greater_is_better used to define ``refit``
    """
    expected_index = 1 if expected_greater_is_better else 0
    assert refit(results={f"mean_test_{expected_metric}": np.array([0.0, 1.0])}) == expected_index


def assert_scoring(
        scoring,
        expected_keys=None,
        agg_periods=None,
        agg_func=None,
        relative_error_tolerance=None):
    """Checks if `scoring` has the expected keys and score functions
    defined by the other parameters.

    Parameters
    ----------
    scoring : `dict`
        ``scoring`` dictionary to check
    expected_keys : `set` [`str`] or None
        Expected keys in `scoring` dictionary.
        If None, does not check the keys.
    agg_periods : callable or None
        What was passed to `get_scoring_and_refit`
    agg_func : `int` or None
        What was passed to `get_scoring_and_refit`
    relative_error_tolerance : `float` or None
        What was passed to `get_scoring_and_refit`
        Must provide `relative_error_tolerance` to check FRACTION_OUTSIDE_TOLERANCE_NAME.
    """
    if expected_keys is not None:
        assert scoring.keys() == expected_keys
    # a few metrics to spot check
    name_func = {
        EvaluationMetricEnum.MeanAbsolutePercentError.get_metric_name(): EvaluationMetricEnum.MeanAbsolutePercentError.get_metric_func(),
        EvaluationMetricEnum.Quantile95.get_metric_name(): EvaluationMetricEnum.Quantile95.get_metric_func(),
        FRACTION_OUTSIDE_TOLERANCE_NAME: partial(fraction_outside_tolerance, rtol=relative_error_tolerance)
    }
    for name, scorer in scoring.items():
        assert isinstance(scorer, _PredictScorerDF)
        assert scorer._sign == 1  # because greater_is_better=True
        if name in name_func:
            expected_func = get_score_func_with_aggregation(
                score_func=name_func[name],
                agg_periods=agg_periods,
                agg_func=agg_func)[0]
            assert_eval_function_equal(
                scorer._score_func,
                expected_func)


def assert_proper_grid_search(
        grid_search,
        expected_grid_size=None,
        lower_bound=None,
        upper_bound=None,
        score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name,
        greater_is_better=False,
        cv_report_metrics_names=None):
    """Checks fitted hyperparameter grid search result.

    Parameters
    ----------
    grid_search : `sklearn.model_selection.RandomizedSearchCV`
        Fitted RandomizedSearchCV object
    expected_grid_size : `int` or None, default None
        Expected number of options evaluated in grid search.
        If None, does not check the expected size.
    lower_bound : `float` or None, default None
        Lower bound on CV test set error.
        If None, does not check the test error.
    upper_bound : `float` or None, default None
        Upper bound on CV test set error.
        If None, does not check the test error.
    score_func : `str` or callable, default ``EvaluationMetricEnum.MeanAbsolutePercentError.name``
        Score function used to select optimal model in CV.
        The same as passed to ``forecast_pipeline`` and grid search.
        If a callable, takes arrays ``y_true``, ``y_pred`` and returns a float.
        If a string, must be either a
        `~greykite.common.evaluation.EvaluationMetricEnum` member name
        or `~greykite.common.constants.FRACTION_OUTSIDE_TOLERANCE`.
    greater_is_better : `bool`, default False
        Whether higher values are better.
        Must be explicitly passed for testing (not derived from ``score_func``).
    cv_report_metrics_names : `list` [`str`] or None, default None
        Additional metrics besides ``metric`` calculated during CV.
        If None, no other metrics are checked in the result.

        Unlike in ``forecast_pipeline``, these are the expected names,
        in the CV output, such as:

            - ``enum.get_metric_name()``
            - ``"CUSTOM_SCORE_FUNC_NAME"``
            - ``"FRACTION_OUTSIDE_TOLERANCE_NAME"``

    Raises
    ------
    AssertionError
        If grid search did not run as expected.
    """
    _, _, short_name = get_score_func_with_aggregation(
        score_func=score_func,  # string or callable
        greater_is_better=greater_is_better,
        # Dummy value, doesn't matter because we ignore the returned `score_func`
        relative_error_tolerance=0.01)
    # attributes are populated
    assert hasattr(grid_search, "best_estimator_")
    assert hasattr(grid_search, "cv_results_")
    if callable(grid_search.refit):
        # `grid_search.refit` is a callable if `grid_search` comes from
        # `forecast_pipeline`.
        # Checks if `best_index_` and `refit` match `metric` and `greater_is_better`.
        assert grid_search.best_index_ == grid_search.refit(grid_search.cv_results_)
        split_scores = grid_search.cv_results_[f"mean_test_{short_name}"]
        expected_best_score = max(split_scores) if greater_is_better else min(split_scores)
        assert split_scores[grid_search.best_index_] == expected_best_score
        assert split_scores[grid_search.best_index_] is not None
        assert not np.isnan(split_scores[grid_search.best_index_])
        assert_refit(
            grid_search.refit,
            expected_metric=short_name,
            expected_greater_is_better=greater_is_better)
    elif grid_search.refit is True:
        # In single metric evaluation, refit_metric is "score".
        short_name = "score"
        # `best_score_` is populated, and the optimal score is the highest
        # test set score. Metrics where `greater_is_better=False` are
        # assumed to be negated in the ``scoring`` parameter so that
        # higher values are better.
        assert hasattr(grid_search, "best_score_")
        best_score = grid_search.best_score_
        test_scores = grid_search.cv_results_[f"mean_test_{short_name}"]
        best_score2 = test_scores[grid_search.best_index_]
        assert best_score == max(test_scores)
        assert best_score2 == max(test_scores)

    if expected_grid_size is not None:
        assert len(grid_search.cv_results_[f"mean_test_{short_name}"]) == expected_grid_size
    # Parameters are populated
    assert_equal(grid_search.cv_results_["params"][grid_search.best_index_], grid_search.best_params_)

    # All metrics are computed
    if cv_report_metrics_names is None:
        cv_report_metrics_names = []
    for expected_metric in cv_report_metrics_names + [short_name]:
        assert f"mean_test_{expected_metric}" in grid_search.cv_results_.keys()
        assert f"std_test_{expected_metric}" in grid_search.cv_results_.keys()
        assert f"mean_train_{expected_metric}" in grid_search.cv_results_.keys()
        assert f"std_train_{expected_metric}" in grid_search.cv_results_.keys()

    if lower_bound is not None or upper_bound is not None:
        grid_results = summarize_grid_search_results(grid_search, score_func=score_func)
        if lower_bound is not None:
            assert all(grid_results[f"mean_test_{short_name}"] >= lower_bound)
        if upper_bound is not None:
            assert all(grid_results[f"mean_test_{short_name}"] <= upper_bound)


def check_forecast_pipeline_result(
        result,
        coverage=0.95,
        strategy=None,
        interactive=False,
        expected_grid_size=None,
        lower_bound_cv=None,
        upper_bound_cv=None,
        score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name,
        greater_is_better=False,
        cv_report_metrics_names=None,
        relative_error_tolerance=None):
    """Helper function that validates forecast_pipeline output.
    Raises an AssertionError is results do not match the expected values.

    Parameters
    ----------
    result : :class:`~greykite.framework.pipeline.pipeline.ForecastResult`
        ``forecast_pipeline`` output to check
    coverage : `float` or None, default 0.95
        The ``coverage`` passed to ``forecast_pipeline``
    strategy : `str` or None, default None
        Null model strategy.
        If None, not checked.
    interactive : `bool`, default False
        Whether to plot and print results.
    expected_grid_size : `int` or None, default None
        Expected number of options evaluated in grid search.
        If None, does not check the expected size.
    lower_bound_cv : `float` or None, default None
        Lower bound on CV test set error.
        If None, does not check the test error.
    upper_bound_cv : `float` or None, default None
        Upper bound on CV test set error.
        If None, does not check the test error.
    score_func : `str` or callable, default ``EvaluationMetricEnum.MeanAbsolutePercentError.name``
        Score function used to select optimal model in CV.
        The same as passed to ``forecast_pipeline`` and grid search.
        If a callable, takes arrays ``y_true``, ``y_pred`` and returns a float.
        If a string, must be either a
        `~greykite.common.evaluation.EvaluationMetricEnum` member name
        or `~greykite.common.constants.FRACTION_OUTSIDE_TOLERANCE`.
    greater_is_better : `bool`, default False
        Whether higher values are better.
        Must be explicitly passed for testing (not derived from ``score_func``).
    cv_report_metrics_names : `list` [`str`] or None, default None
        Additional metrics besides ``metric`` calculated during CV.
        If None, no other metrics are checked in the result.

        Unlike in ``forecast_pipeline``, these are the expected names,
        in the CV output, such as:

            - ``enum.get_metric_name()``
            - ``"CUSTOM_SCORE_FUNC_NAME"``
            - ``"FRACTION_OUTSIDE_TOLERANCE_NAME"``
    relative_error_tolerance : `float` or None
        The ``relative_error_tolerance`` passed to ``forecast_pipeline``
    """
    assert isinstance(result.grid_search, RandomizedSearchCV)
    assert isinstance(result.model, Pipeline)
    assert isinstance(result.backtest, UnivariateForecast)
    assert isinstance(result.forecast, UnivariateForecast)

    assert_proper_grid_search(
        result.grid_search,
        expected_grid_size=expected_grid_size,
        lower_bound=lower_bound_cv,
        upper_bound=upper_bound_cv,
        score_func=score_func,
        greater_is_better=greater_is_better,
        cv_report_metrics_names=cv_report_metrics_names)

    ts = result.timeseries
    assert ts.df[VALUE_COL].equals(ts.y)
    assert result.backtest.train_evaluation is not None
    assert result.backtest.test_evaluation is not None
    if coverage is None:
        assert result.forecast.coverage is None
        assert result.backtest.coverage is None
        assert result.backtest.train_evaluation[PREDICTION_BAND_COVERAGE] is None
        assert result.backtest.test_evaluation[PREDICTION_BAND_COVERAGE] is None
        expected_cols = [TIME_COL, ACTUAL_COL, PREDICTED_COL]
        assert list(result.backtest.df.columns) == expected_cols
        assert list(result.forecast.df.columns) == expected_cols
    else:
        assert round(result.forecast.coverage, 3) == round(coverage, 3)
        assert round(result.backtest.coverage, 3) == round(coverage, 3)
        assert result.backtest.train_evaluation[PREDICTION_BAND_COVERAGE] is not None
        assert result.backtest.test_evaluation[PREDICTION_BAND_COVERAGE] is not None

    assert result.forecast.train_evaluation is not None

    # Tests if null model params are set for CV
    estimator = result.model.steps[-1][-1]
    if estimator.null_model is not None and strategy is not None:
        assert estimator.null_model.strategy == strategy

    # Tests if relative_error_tolerance is set for backtest/forecast
    if relative_error_tolerance is not None:
        assert result.backtest.relative_error_tolerance == relative_error_tolerance
        assert result.forecast.relative_error_tolerance == relative_error_tolerance

    if interactive:
        print("backtest_train_evaluation", result.backtest.train_evaluation)
        print("backtest_test_evaluation", result.backtest.test_evaluation)
        print("forecast_train_evaluation", result.forecast.train_evaluation)
        print("forecast_test_evaluation", result.forecast.test_evaluation)
        print(summarize_grid_search_results(
            result.grid_search,
            score_func=score_func,
            score_func_greater_is_better=greater_is_better))
        ts.plot().write_html("test_ts_plot.html")
        result.backtest.plot().write_html("test_backtest_plot.html")
        result.forecast.plot().write_html("test_forecast_plot.html")


def assert_basic_pipeline_equal(actual: Pipeline, expected: Pipeline):
    """Asserts that the two pipelines are equal
    The Pipelines should be created by `get_basic_pipeline`.
    """
    # checks features
    actual_params = actual.get_params()
    expected_params = expected.get_params()
    check_keys = [
        'input__date__select_date__column_names',
        'input__response__select_val__column_names',
        'input__response__outlier__use_fit_baseline',
        'input__response__outlier__z_cutoff',
        'input__response__null__impute_algorithm',
        'input__response__null__impute_all',
        'input__response__null__impute_params',
        'input__response__null__max_frac',
        'input__regressors_numeric__select_reg__column_names',
        'input__regressors_numeric__select_reg_numeric__exclude',
        'input__regressors_numeric__select_reg_numeric__include',
        'input__regressors_numeric__outlier__use_fit_baseline',
        'input__regressors_numeric__outlier__z_cutoff',
        'input__regressors_numeric__normalize__normalize_algorithm',
        'input__regressors_numeric__normalize__normalize_params',
        'input__regressors_numeric__null__impute_algorithm',
        'input__regressors_numeric__null__impute_all',
        'input__regressors_numeric__null__impute_params',
        'input__regressors_numeric__null__max_frac',
        'input__regressors_other__select_reg__column_names',
        'input__regressors_other__select_reg_non_numeric__exclude',
        'input__regressors_other__select_reg_non_numeric__include',
        'degenerate__drop_degenerate'
    ]
    for key in check_keys:
        assert actual_params[key] == expected_params[key],\
            f"{key} is different, found {actual_params[key]}, expected {expected_params[key]}"

    # checks estimator
    actual_estimator = actual.steps[-1][-1]
    expected_estimator = expected.steps[-1][-1]
    assert isinstance(actual_estimator, type(expected_estimator))
    actual_estimator_dict = actual_estimator.__dict__.copy()
    expected_estimator_dict = expected_estimator.__dict__.copy()
    del actual_estimator_dict["null_model"]
    del expected_estimator_dict["null_model"]
    actual_estimator_dict.pop("silverkite", None)
    expected_estimator_dict.pop("silverkite", None)
    actual_estimator_dict.pop("silverkite_diagnostics", None)
    expected_estimator_dict.pop("silverkite_diagnostics", None)
    actual_score_func = actual_estimator_dict.pop("score_func")
    expected_score_func = expected_estimator_dict.pop("score_func")
    assert_equal(actual_estimator_dict, expected_estimator_dict)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert_eval_function_equal(actual_score_func, expected_score_func)


def assert_forecast_pipeline_result_equal(actual, expected, is_silverkite=True, rel=1e-5):
    """Raises an AssertionError if the two forecast pipeline results are not equal"""
    if is_silverkite:
        # checks training matrix, predictors, evaluation splits,
        # and fit parameters
        keys_to_check = {
            "time_col",
            "value_col",
            "origin_for_time_vars",
            "daily_event_df_dict",
            "changepoint_values",
            "continuous_time_col",
            # "growth_func",  # skip, hard to check function equality
            # "fs_func",
            # "autoreg_func",
            "max_lag_order",
            "uncertainty_dict",
            "pred_cols",
            "last_date_for_fit",
            "x_mat"  # training matrix
        }
        simple_model_dict = actual.model.steps[-1][-1].model_dict
        model_dict = expected.model.steps[-1][-1].model_dict
        assert_equal(
            {k: v for k, v in simple_model_dict.items() if k in keys_to_check},
            {k: v for k, v in model_dict.items() if k in keys_to_check},
            ignore_list_order=True
        )
        simple_fit_algorithm_dict = actual.model.steps[-1][-1].fit_algorithm_dict
        fit_algorithm_dict = expected.model.steps[-1][-1].fit_algorithm_dict
        assert simple_fit_algorithm_dict == fit_algorithm_dict

    # checks evaluation result on backtest and forecast,
    # including prediction interval and null model metrics
    if actual.backtest is None:
        assert expected.backtest is None
    else:
        assert_equal(
            actual.backtest.train_evaluation,
            expected.backtest.train_evaluation,
            rel=rel)
        assert_equal(
            actual.backtest.test_evaluation,
            expected.backtest.test_evaluation,
            rel=rel)
    assert_equal(
        actual.forecast.train_evaluation,
        expected.forecast.train_evaluation,
        rel=rel)
    assert_equal(
        actual.forecast.test_evaluation,
        expected.forecast.test_evaluation,
        rel=rel)
    assert_equal(
        actual.grid_search.best_index_,
        expected.grid_search.best_index_,
        rel=rel)
    # checks CV search parameters
    assert_equal(
        actual.grid_search.cv.__dict__,
        expected.grid_search.cv.__dict__
    )


@validate_pipeline_input
def mock_pipeline(
        # The arguments and defaults should be identical to forecast_pipeline() function
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
        n_jobs=1,
        verbose=1,
        # forecast
        forecast_horizon=None,
        coverage: Optional[float] = 0.95,
        test_horizon=None,
        periods_between_train_test=None,
        agg_periods=None,
        agg_func=None,
        # evaluation
        score_func=EvaluationMetricEnum.MeanSquaredError.name,
        score_func_greater_is_better=False,
        cv_report_metrics=None,
        null_model_params=None,
        relative_error_tolerance: Optional[float] = 0.05,
        # CV
        cv_horizon=None,
        cv_min_train_periods=None,
        cv_expanding_window=False,
        cv_use_most_recent_splits=False,
        cv_periods_between_splits=None,
        cv_periods_between_train_test=0,
        cv_max_splits=3):
    """Create and returns custom pipeline parameters"""
    return {
        "df": df,
        "time_col": time_col,
        "value_col": value_col,
        "date_format": date_format,
        "tz": tz,
        "freq": freq,
        "train_end_date": train_end_date,
        "anomaly_info": anomaly_info,
        "pipeline": pipeline,
        "regressor_cols": regressor_cols,
        "lagged_regressor_cols": lagged_regressor_cols,
        "estimator": estimator,
        "hyperparameter_grid": hyperparameter_grid,
        "hyperparameter_budget": hyperparameter_budget,
        "n_jobs": n_jobs,
        "verbose": verbose,
        "forecast_horizon": forecast_horizon,
        "coverage": coverage,
        "test_horizon": test_horizon,
        "periods_between_train_test": periods_between_train_test,
        "agg_periods": agg_periods,
        "agg_func": agg_func,
        "score_func": score_func,
        "score_func_greater_is_better": score_func_greater_is_better,
        "cv_report_metrics": cv_report_metrics,
        "null_model_params": null_model_params,
        "relative_error_tolerance": relative_error_tolerance,
        "cv_horizon": cv_horizon,
        "cv_min_train_periods": cv_min_train_periods,
        "cv_expanding_window": cv_expanding_window,
        "cv_use_most_recent_splits": cv_use_most_recent_splits,
        "cv_periods_between_splits": cv_periods_between_splits,
        "cv_periods_between_train_test": cv_periods_between_train_test,
        "cv_max_splits": cv_max_splits,
    }
