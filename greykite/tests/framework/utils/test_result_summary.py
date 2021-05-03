import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from greykite.common.constants import FRACTION_OUTSIDE_TOLERANCE
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.common.python_utils import assert_equal
from greykite.common.testing_utils import generate_df_for_tests
from greykite.framework.constants import CUSTOM_SCORE_FUNC_NAME
from greykite.framework.constants import CV_REPORT_METRICS_ALL
from greykite.framework.constants import FRACTION_OUTSIDE_TOLERANCE_NAME
from greykite.framework.pipeline.pipeline import forecast_pipeline
from greykite.framework.utils.result_summary import get_ranks_and_splits
from greykite.framework.utils.result_summary import summarize_grid_search_results
from greykite.sklearn.estimator.null_model import DummyEstimator
from greykite.sklearn.estimator.silverkite_estimator import SilverkiteEstimator


def test_get_ranks_and_splits(pipeline_results):
    """Tests get_ranks_and_splits"""
    # string `score_func`, combine_splits=True
    grid_search = pipeline_results["1"].grid_search
    ranks_and_splits = get_ranks_and_splits(
        grid_search,
        score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name,
        greater_is_better=None,
        combine_splits=True,
        decimals=None)

    short_name = EvaluationMetricEnum.MeanAbsolutePercentError.get_metric_name()
    assert ranks_and_splits["short_name"] == short_name
    assert ranks_and_splits["ranks"].argmin() == grid_search.cv_results_[f"mean_test_{short_name}"].argmin()
    assert ranks_and_splits["split_train"][2][1] == grid_search.cv_results_[f"split1_train_{short_name}"][2]
    assert ranks_and_splits["split_test"][3][0] == grid_search.cv_results_[f"split0_test_{short_name}"][3]

    # `decimals=2`
    ranks_and_splits = get_ranks_and_splits(
        grid_search,
        score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name,
        greater_is_better=None,
        combine_splits=True,
        decimals=2)
    short_name = EvaluationMetricEnum.MeanAbsolutePercentError.get_metric_name()
    assert ranks_and_splits["short_name"] == short_name
    assert ranks_and_splits["ranks"].argmin() == grid_search.cv_results_[f"mean_test_{short_name}"].argmin()
    assert ranks_and_splits["split_train"][2][1] == round(grid_search.cv_results_[f"split1_train_{short_name}"][2], 2)
    assert ranks_and_splits["split_test"][3][0] == round(grid_search.cv_results_[f"split0_test_{short_name}"][3], 2)

    # callable `score_func`, `combine_splits=False`
    grid_search = pipeline_results["3"].grid_search
    ranks_and_splits = get_ranks_and_splits(
        grid_search,
        score_func=mean_absolute_error,
        greater_is_better=True,
        combine_splits=False,
        decimals=None)
    short_name = CUSTOM_SCORE_FUNC_NAME
    assert ranks_and_splits["short_name"] == short_name
    assert ranks_and_splits["ranks"].argmin() == grid_search.cv_results_[f"mean_test_{short_name}"].argmax()  # NB: max because greater_is_better=True
    assert ranks_and_splits["split_train"] is None
    assert ranks_and_splits["split_test"] is None

    # metric is missing
    with pytest.warns(Warning, match="Metric 'MAPE' is not available in the CV results."):
        ranks_and_splits = get_ranks_and_splits(
            grid_search,
            score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name,
            greater_is_better=None,
            combine_splits=True,
            decimals=None)
        assert ranks_and_splits == {
            "short_name": 'MAPE',
            "ranks": None,
            "split_train": None,
            "split_test": None}

    # `warns=False`
    with pytest.warns(None):
        ranks_and_splits = get_ranks_and_splits(
            grid_search,
            score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name,
            greater_is_better=None,
            combine_splits=True,
            decimals=None,
            warn_metric=False)
        assert ranks_and_splits == {
            "short_name": 'MAPE',
            "ranks": None,
            "split_train": None,
            "split_test": None}


def test_summarize_grid_search_results(pipeline_results):
    """Tests summarize_grid_search_results"""
    # Tests EvaluationMetricEnum `score_func`, `cv_report_metrics=CV_REPORT_METRICS_ALL`
    grid_search = pipeline_results["1"].grid_search
    metric = EvaluationMetricEnum.MeanAbsolutePercentError
    cv_result = summarize_grid_search_results(
        grid_search=grid_search,
        only_changing_params=True,
        combine_splits=True,
        score_func=metric.name,
        score_func_greater_is_better=metric.get_metric_greater_is_better())
    assert cv_result.shape == (4, 60)
    # The proper scores are extracted
    short_name = metric.get_metric_name()
    expected = grid_search.cv_results_[f"mean_test_{short_name}"]
    assert_equal(np.array(cv_result[f"mean_test_{short_name}"]), expected)
    # Rank direction is correct
    assert cv_result[f"rank_test_{short_name}"].idxmin() == cv_result[f"mean_test_{short_name}"].idxmin()
    assert all(cv_result[f"mean_test_{short_name}"] > 0)
    assert ["rank_test_MAE", "rank_test_MSE", "rank_test_MedAPE", "rank_test_MAPE",
            "mean_test_MAE", "mean_test_MSE", "mean_test_MedAPE", "mean_test_MAPE",
            "split_test_MAPE", "split_test_MSE", "split_test_MAE",
            "split_test_MedAPE", "mean_train_MAE", "mean_train_MSE",
            "mean_train_MedAPE", "mean_train_MAPE", "params",
            "param_estimator__strategy", "param_estimator__quantile",
            "param_estimator__constant", "split_train_MAPE", "split_train_MSE",
            "split_train_MAE", "split_train_MedAPE", "mean_fit_time",
            "std_fit_time", "mean_score_time", "std_score_time", "split0_test_MAE",
            "split1_test_MAE", "split2_test_MAE", "std_test_MAE",
            "split0_train_MAE", "split1_train_MAE", "split2_train_MAE",
            "std_train_MAE", "split0_test_MSE", "split1_test_MSE",
            "split2_test_MSE", "std_test_MSE", "split0_train_MSE",
            "split1_train_MSE", "split2_train_MSE", "std_train_MSE",
            "split0_test_MedAPE", "split1_test_MedAPE", "split2_test_MedAPE",
            "std_test_MedAPE", "split0_train_MedAPE", "split1_train_MedAPE",
            "split2_train_MedAPE", "std_train_MedAPE", "split0_test_MAPE",
            "split1_test_MAPE", "split2_test_MAPE", "std_test_MAPE",
            "split0_train_MAPE", "split1_train_MAPE", "split2_train_MAPE",
            "std_train_MAPE"] == list(cv_result.columns)

    # `combine_splits=False`
    cv_result = summarize_grid_search_results(
        grid_search=grid_search,
        only_changing_params=True,
        combine_splits=False,
        score_func=metric.name,
        score_func_greater_is_better=metric.get_metric_greater_is_better(),
        cv_report_metrics=CV_REPORT_METRICS_ALL)
    assert cv_result.shape == (4, 52)  # no train/test split summary for 4 metrics
    assert "split_test_MedAPE" not in cv_result.columns

    # cv_report_metrics=list, different column_order
    cv_result = summarize_grid_search_results(
        grid_search=grid_search,
        only_changing_params=True,
        combine_splits=False,
        score_func=metric.name,
        score_func_greater_is_better=metric.get_metric_greater_is_better(),
        cv_report_metrics=[EvaluationMetricEnum.MeanSquaredError.name],
        column_order=["mean", "time", ".*"])
    assert cv_result.shape == (4, 30)  # only two metrics in the summary
    assert ["mean_fit_time", "mean_score_time", "mean_test_MSE", "mean_train_MSE",
            "mean_test_MAPE", "mean_train_MAPE", "std_fit_time", "std_score_time",
            "param_estimator__strategy", "param_estimator__quantile",
            "param_estimator__constant", "params", "split0_test_MSE",
            "split1_test_MSE", "split2_test_MSE", "std_test_MSE", "rank_test_MSE",
            "split0_train_MSE", "split1_train_MSE", "split2_train_MSE",
            "std_train_MSE", "split0_test_MAPE", "split1_test_MAPE",
            "split2_test_MAPE", "std_test_MAPE", "rank_test_MAPE",
            "split0_train_MAPE", "split1_train_MAPE", "split2_train_MAPE",
            "std_train_MAPE"] == list(cv_result.columns)
    # These metrics are computed but not requested in summary
    assert "rank_test_MedAPE" not in cv_result.columns
    assert "mean_test_MAE" not in cv_result.columns

    # cv_report_metrics=None, different column order
    cv_result = summarize_grid_search_results(
        grid_search=grid_search,
        only_changing_params=True,
        combine_splits=False,
        score_func=metric.name,
        score_func_greater_is_better=metric.get_metric_greater_is_better(),
        cv_report_metrics=None,
        column_order=["split", "rank", "mean", "params"])
    assert cv_result.shape == (4, 12)  # only one metric in the summary
    assert ["split0_test_MAPE", "split1_test_MAPE", "split2_test_MAPE",
            "split0_train_MAPE", "split1_train_MAPE", "split2_train_MAPE",
            "rank_test_MAPE", "mean_fit_time", "mean_score_time", "mean_test_MAPE",
            "mean_train_MAPE", "params"] == list(cv_result.columns)
    assert "rank_test_MSE" not in cv_result.columns

    # Tests FRACTION_OUTSIDE_TOLERANCE `score_func`
    grid_search = pipeline_results["2"].grid_search
    cv_result = summarize_grid_search_results(
        grid_search=grid_search,
        only_changing_params=True,
        score_func=FRACTION_OUTSIDE_TOLERANCE,
        score_func_greater_is_better=False)
    assert cv_result.shape == (4, 242)
    # The proper scores are extracted
    short_name = FRACTION_OUTSIDE_TOLERANCE_NAME
    expected = grid_search.cv_results_[f"mean_test_{short_name}"]
    assert_equal(np.array(cv_result[f"mean_test_{short_name}"]), expected)
    # Rank direction is correct
    assert cv_result[f"rank_test_{short_name}"].idxmin() == cv_result[f"mean_test_{short_name}"].idxmin()
    assert all(cv_result[f"mean_test_{short_name}"] > 0)

    # Tests callable `score_func`, greater_is_better=True, split scores
    grid_search = pipeline_results["3"].grid_search
    cv_max_splits = 2
    cv_result = summarize_grid_search_results(
        grid_search=grid_search,
        only_changing_params=True,
        score_func=mean_absolute_error,
        score_func_greater_is_better=True)
    assert cv_result.shape == (4, 20)
    # the proper scores are extracted
    short_name = CUSTOM_SCORE_FUNC_NAME
    expected = grid_search.cv_results_[f"mean_test_{short_name}"]
    assert_equal(np.array(cv_result[f"mean_test_{short_name}"]), expected)
    # Rank direction is correct
    assert cv_result[f"rank_test_{short_name}"].idxmin() == cv_result[f"mean_test_{short_name}"].idxmax()  # NB: max
    assert all(cv_result[f"mean_test_{short_name}"] > 0)
    assert len(cv_result["params"][0]) == 2  # two params have multiple options in the grid
    assert len(cv_result[f"split_test_{short_name}"][0]) == cv_max_splits
    # no rounding is applied
    assert cv_result[f"mean_test_{short_name}"][1] == pytest.approx(2.430402, rel=1e-5)
    assert cv_result[f"mean_train_{short_name}"][1] == pytest.approx(1.839883, rel=1e-5)
    assert cv_result[f"std_test_{short_name}"][1] == pytest.approx(0.16548, rel=1e-5)
    assert cv_result[f"split_test_{short_name}"][1][0] == pytest.approx(2.26492, rel=1e-5)
    assert cv_result[f"split_train_{short_name}"][1][0] == pytest.approx(1.84082, rel=1e-5)
    expected = grid_search.cv_results_
    for k, v in cv_result.items():
        if k in expected and k not in ("params", f"rank_test_{short_name}"):
            assert_equal(pd.Series(expected[k], name=k), v)

    # decimals=2, and only_changing_params=False
    cv_result = summarize_grid_search_results(
        grid_search=grid_search,
        only_changing_params=False,
        decimals=2,
        score_func=mean_absolute_error,
        score_func_greater_is_better=False)
    assert cv_result.shape == (4, 20)
    # only_changing_params=False, so all params in hyperparameter_grid are included
    assert len(cv_result["params"][0]) == 4
    # rounding is applied
    assert cv_result[f"mean_test_{short_name}"][1] == 2.43
    assert cv_result[f"mean_train_{short_name}"][1] == 1.84
    assert cv_result[f"std_test_{short_name}"][1] == 0.17
    assert cv_result[f"split_test_{short_name}"][1][0] == 2.26
    assert cv_result[f"split_train_{short_name}"][1][0] == 1.84


@pytest.fixture(scope="module")
def pipeline_results():
    """Runs forecast_pipeline three times to get
     grid search results"""
    pipeline_results = {}

    data = generate_df_for_tests(freq="1D", periods=20 * 7)
    df = data["df"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    hyperparameter_grid = [
        {
            "estimator__strategy": ["quantile"],
            "estimator__quantile": [0.9]
        },
        {
            "estimator__strategy": ["mean"]
        },
        {
            "estimator__strategy": ["constant"],
            "estimator__constant": [1.0, 2.0]
        }
    ]
    pipeline = Pipeline([("estimator", DummyEstimator())])
    # Tests MAPE `score_func`, list `cv_report_metrics`
    metric = EvaluationMetricEnum.MeanAbsolutePercentError
    pipeline_results["1"] = forecast_pipeline(
        df,
        pipeline=pipeline,
        hyperparameter_grid=hyperparameter_grid,
        n_jobs=-1,
        forecast_horizon=20,
        coverage=None,
        agg_periods=7,
        agg_func=np.sum,
        score_func=metric.name,
        score_func_greater_is_better=metric.get_metric_greater_is_better(),
        cv_report_metrics=[
            EvaluationMetricEnum.MeanAbsoluteError.name,
            EvaluationMetricEnum.MeanSquaredError.name,
            EvaluationMetricEnum.MedianAbsolutePercentError.name,
        ],
        null_model_params=None)

    # Tests FRACTION_OUTSIDE_TOLERANCE `score_func`, all `cv_report_metrics`
    pipeline = Pipeline([("estimator", DummyEstimator())])
    pipeline_results["2"] = forecast_pipeline(
        df,
        pipeline=pipeline,
        hyperparameter_grid=hyperparameter_grid,
        n_jobs=-1,
        forecast_horizon=20,
        coverage=None,
        score_func=FRACTION_OUTSIDE_TOLERANCE,
        score_func_greater_is_better=False,
        cv_report_metrics=CV_REPORT_METRICS_ALL,
        null_model_params=None,
        relative_error_tolerance=0.02)

    # Tests callable `score_func`, greater_is_better=True, no `cv_report_metrics`
    fs1 = pd.DataFrame({
        "name": ["tow", "conti_year"],
        "period": [7.0, 1.0],
        "order": [3, 3],
        "seas_names": ["weekly", None]})
    fs2 = pd.DataFrame({
        "name": ["tow"],
        "period": [7.0],
        "order": [3],
        "seas_names": ["weekly"]})
    hyperparameter_grid = {
        "estimator__origin_for_time_vars": [2018],
        "estimator__extra_pred_cols": [["ct1"], ["ct2"]],
        "estimator__fit_algorithm_dict": [{"fit_algorithm": "linear"}],
        "estimator__fs_components_df": [fs1, fs2],
    }
    cv_max_splits = 2
    pipeline_results["3"] = forecast_pipeline(
        df,
        estimator=SilverkiteEstimator(),
        hyperparameter_grid=hyperparameter_grid,
        hyperparameter_budget=4,
        n_jobs=1,
        forecast_horizon=3 * 7,
        test_horizon=2 * 7,
        score_func=mean_absolute_error,  # callable score_func
        score_func_greater_is_better=True,  # Not really True, only for the sake of testing
        null_model_params=None,
        cv_horizon=1 * 7,
        cv_expanding_window=True,
        cv_min_train_periods=7 * 7,
        cv_periods_between_splits=7,
        cv_periods_between_train_test=3 * 7,
        cv_max_splits=cv_max_splits)
    return pipeline_results
