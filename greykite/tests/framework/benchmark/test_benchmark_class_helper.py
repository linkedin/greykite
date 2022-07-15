import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error

from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.common.python_utils import assert_equal
from greykite.common.testing_utils import generate_df_for_tests
from greykite.common.testing_utils import generate_df_with_reg_for_tests
from greykite.framework.benchmark.benchmark_class_helper import forecast_pipeline_rolling_evaluation
from greykite.framework.pipeline.pipeline import forecast_pipeline
from greykite.framework.utils.framework_testing_utils import mock_pipeline
from greykite.framework.utils.result_summary import summarize_grid_search_results
from greykite.sklearn.cross_validation import RollingTimeSeriesSplit
from greykite.sklearn.estimator.prophet_estimator import ProphetEstimator
from greykite.sklearn.estimator.silverkite_estimator import SilverkiteEstimator


try:
    import prophet  # noqa
except ModuleNotFoundError:
    pass


@pytest.mark.skipif("prophet" not in sys.modules,
                    reason="Module 'prophet' not installed, pytest for 'ProphetTemplate' skipped.")
def test_forecast_pipeline_rolling_evaluation_prophet():
    """Checks the output rolling evaluation with Prophet template"""
    data = generate_df_with_reg_for_tests(
        freq="D",
        periods=30,
        remove_extra_cols=True,
        mask_test_actuals=True)
    reg_cols = ["regressor1", "regressor2", "regressor3"]
    keep_cols = [TIME_COL, VALUE_COL] + reg_cols
    df = data["df"][keep_cols]

    hyperparameter_grid = {
        "estimator__weekly_seasonality": [True],
        "estimator__daily_seasonality": [True, False],
        "estimator__n_changepoints": [0],  # to speed up test case, remove for better fit
        "estimator__uncertainty_samples": [10],  # to speed up test case
        "estimator__add_regressor_dict": [{
            "regressor1": {
                "prior_scale": 10,
                "standardize": True,
                "mode": 'additive'
            },
            "regressor2": {
                "prior_scale": 15,
                "standardize": False,
                "mode": 'additive'
            },
            "regressor3": {}
        }]
    }
    pipeline_params = mock_pipeline(
        df=df,
        forecast_horizon=3,
        regressor_cols=["regressor1", "regressor2", "regressor3"],
        estimator=ProphetEstimator(),
        hyperparameter_grid=hyperparameter_grid)
    tscv = RollingTimeSeriesSplit(forecast_horizon=3, expanding_window=True, max_splits=1)
    rolling_evaluation = forecast_pipeline_rolling_evaluation(
        pipeline_params=pipeline_params,
        tscv=tscv)

    expected_splits_n = tscv.max_splits
    assert len(rolling_evaluation.keys()) == expected_splits_n
    assert set(rolling_evaluation.keys()) == {"split_0"}

    split0_output = rolling_evaluation["split_0"]
    assert round(split0_output["runtime_sec"], 3) == split0_output["runtime_sec"]

    pipeline_result = split0_output["pipeline_result"]
    # Calculates expected pipeline
    train, test = list(tscv.split(X=df))[0]
    df_train = df.loc[train]
    pipeline_params_updated = pipeline_params
    pipeline_params_updated["test_horizon"] = 0
    pipeline_params_updated["df"] = df_train
    expected_pipeline_result = forecast_pipeline(**pipeline_params_updated)

    assert pipeline_result.backtest is None
    # Checks output is identical when there is only 1 split
    pipeline_grid_search = summarize_grid_search_results(
        pipeline_result.grid_search,
        score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name
    )
    expected_grid_search = summarize_grid_search_results(
        expected_pipeline_result.grid_search,
        score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name
    )
    assert_equal(
        pipeline_grid_search["mean_test_MAPE"],
        expected_grid_search["mean_test_MAPE"]
    )
    assert_equal(
        pipeline_result.grid_search.cv.__dict__,
        expected_pipeline_result.grid_search.cv.__dict__
    )
    # Checks forecast df has the correct number of rows
    expected_rows = pipeline_result.timeseries.fit_df.shape[0] + tscv.forecast_horizon
    assert pipeline_result.forecast.df.shape[0] == expected_rows


def test_forecast_pipeline_rolling_evaluation_silverkite():
    """Checks the output rolling evaluation with Silverkite template"""
    data = generate_df_with_reg_for_tests(
        freq="1D",
        periods=20 * 7,  # short-term: 20 weeks of data
        remove_extra_cols=True,
        mask_test_actuals=True)
    regressor_cols = ["regressor1", "regressor2", "regressor_categ"]
    keep_cols = [TIME_COL, VALUE_COL] + regressor_cols
    df = data["df"][keep_cols]

    coverage = 0.1
    hyperparameter_grid = {
        "estimator__origin_for_time_vars": [None],  # inferred from training data
        "estimator__fs_components_df": [
            pd.DataFrame({
                "name": ["tow"],
                "period": [7.0],
                "order": [3],
                "seas_names": ["weekly"]})],
        "estimator__extra_pred_cols": [
            regressor_cols,
            regressor_cols + ["ct_sqrt"]
        ],  # two cases: no growth term and single growth term
        "estimator__fit_algorithm_dict": [{"fit_algorithm": "linear"}]
    }
    pipeline_params = mock_pipeline(
        df=df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        date_format=None,  # not recommended, but possible to specify
        freq=None,
        regressor_cols=regressor_cols,
        estimator=SilverkiteEstimator(),
        hyperparameter_grid=hyperparameter_grid,
        hyperparameter_budget=1,
        n_jobs=1,
        forecast_horizon=2 * 7,
        coverage=coverage,
        test_horizon=2 * 7,
        periods_between_train_test=2 * 7,
        agg_periods=7,
        agg_func=np.mean,
        score_func=mean_absolute_error,  # callable score_func
        null_model_params=None,
        cv_horizon=1 * 7,
        cv_expanding_window=True,
        cv_min_train_periods=8 * 7,
        cv_periods_between_splits=7,
        cv_periods_between_train_test=3 * 7,
        cv_max_splits=2)
    tscv = RollingTimeSeriesSplit(
        forecast_horizon=2 * 7,
        min_train_periods=10 * 7,
        expanding_window=True,
        use_most_recent_splits=True,
        periods_between_splits=2 * 7,
        periods_between_train_test=2 * 7,
        max_splits=3)
    rolling_evaluation = forecast_pipeline_rolling_evaluation(
        pipeline_params=pipeline_params,
        tscv=tscv)

    expected_splits_n = tscv.max_splits
    assert len(rolling_evaluation.keys()) == expected_splits_n
    assert set(rolling_evaluation.keys()) == {"split_0", "split_1", "split_2"}

    time_col = pipeline_params["time_col"]
    for split_num, (train, test) in enumerate(tscv.split(X=df)):
        split_output = rolling_evaluation[f"split_{split_num}"]
        assert round(split_output["runtime_sec"], 3) == split_output["runtime_sec"]

        pipeline_result = split_output["pipeline_result"]

        # Checks every split uses all the available data for training
        ts = pipeline_result.timeseries
        train_end_date = df.iloc[train[-1]][time_col]
        assert ts.train_end_date == train_end_date

        assert pipeline_result.backtest is None

        # Checks every split has forecast for train+test periods passed by tscv
        forecast = pipeline_result.forecast
        assert forecast.df.shape[0] == ts.fit_df.shape[0] + tscv.periods_between_train_test + tscv.forecast_horizon


def test_forecast_pipeline_rolling_evaluation_error():
    """Checks errors of forecast_pipeline_rolling_evaluation"""
    data = generate_df_for_tests(freq="D", periods=30)
    df = data["df"]
    tscv = RollingTimeSeriesSplit(forecast_horizon=7, periods_between_train_test=7, max_splits=1)
    # Different forecast_horizon in pipeline_params and tscv
    with pytest.raises(ValueError, match="Forecast horizon in 'pipeline_params' "
                                         "does not match that of the 'tscv'."):
        pipeline_params = mock_pipeline(df=df, forecast_horizon=15)
        forecast_pipeline_rolling_evaluation(
            pipeline_params=pipeline_params,
            tscv=tscv)

    with pytest.raises(ValueError, match="'periods_between_train_test' in 'pipeline_params' "
                                         "does not match that of the 'tscv'."):
        pipeline_params = mock_pipeline(df=df, forecast_horizon=7, periods_between_train_test=2)
        forecast_pipeline_rolling_evaluation(
            pipeline_params=pipeline_params,
            tscv=tscv)
