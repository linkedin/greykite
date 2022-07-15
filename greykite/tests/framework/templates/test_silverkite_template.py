import dataclasses
import datetime
import warnings

import numpy as np
import pandas as pd
import pytest

from greykite.algo.forecast.silverkite.forecast_silverkite import SilverkiteForecast
from greykite.algo.forecast.silverkite.forecast_simple_silverkite_helper import generate_holiday_events
from greykite.algo.forecast.silverkite.forecast_simple_silverkite_helper import get_event_pred_cols
from greykite.common.constants import ADJUSTMENT_DELTA_COL
from greykite.common.constants import END_TIME_COL
from greykite.common.constants import METRIC_COL
from greykite.common.constants import PREDICTION_BAND_COVERAGE
from greykite.common.constants import START_TIME_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.common.python_utils import assert_equal
from greykite.common.testing_utils import generate_df_for_tests
from greykite.common.testing_utils import generate_df_with_reg_for_tests
from greykite.framework.constants import COMPUTATION_N_JOBS
from greykite.framework.constants import CV_REPORT_METRICS_ALL
from greykite.framework.templates.autogen.forecast_config import ComputationParam
from greykite.framework.templates.autogen.forecast_config import EvaluationMetricParam
from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.templates.silverkite_template import SilverkiteTemplate
from greykite.framework.templates.silverkite_template import apply_default_model_components
from greykite.framework.templates.silverkite_template import get_extra_pred_cols
from greykite.framework.utils.framework_testing_utils import assert_basic_pipeline_equal
from greykite.framework.utils.framework_testing_utils import check_forecast_pipeline_result
from greykite.sklearn.estimator.silverkite_diagnostics import SilverkiteDiagnostics
from greykite.sklearn.estimator.silverkite_estimator import SilverkiteEstimator


@pytest.fixture
def silverkite():
    return SilverkiteForecast()


@pytest.fixture
def silverkite_diagnostics():
    return SilverkiteDiagnostics()


@pytest.fixture
def model_components_param(silverkite, silverkite_diagnostics):
    return ModelComponentsParam(
        seasonality={
            "fs_components_df": None
        },
        events=None,
        changepoints={
            "changepoints_dict": {
                "method": "uniform",
                "n_changepoints": 20,
            }
        },
        uncertainty={
            "uncertainty_dict": {
                "uncertainty_method": "simple_conditional_residuals",
                # `quantiles` is not provided, requires `config.coverage` to be set
                # "params": {
                #     "conditional_cols": ["dow_hr"],
                #     "quantiles": [0.02, 0.98],
                #     "quantile_estimation_method": "normal_fit",
                #     "sample_size_thresh": 5,
                #     "small_sample_size_method": "std_quantiles",
                #     "small_sample_size_quantile": 0.98,
                # }
            }
        },
        custom={
            "silverkite": silverkite,
            "silverkite_diagnostics": silverkite_diagnostics,
            "extra_pred_cols": [["ct1"], ["ct2"], ["regressor1", "regressor3"]],
            "max_admissible_value": 4,
        }
    )


def test_get_extra_pred_cols():
    extra_pred_cols = get_extra_pred_cols(
        model_components=None)
    assert extra_pred_cols is None

    extra_pred_cols = get_extra_pred_cols(
        model_components=ModelComponentsParam(
            custom={})
    )
    assert extra_pred_cols is None

    extra_pred_cols = get_extra_pred_cols(
        model_components=ModelComponentsParam(
            custom={
                "extra_pred_cols": ["p1", "p2", "p3"]
            }
        )
    )
    assert set(extra_pred_cols) == {"p1", "p2", "p3"}

    extra_pred_cols = get_extra_pred_cols(
        model_components=ModelComponentsParam(
            custom={
                "extra_pred_cols": [["p1"], ["p2", "p3"], None, []]
            }
        )
    )
    assert set(extra_pred_cols) == {"p1", "p2", "p3"}


def test_apply_default_model_components(model_components_param, silverkite, silverkite_diagnostics):
    model_components = apply_default_model_components()
    assert_equal(model_components.seasonality, {
        "fs_components_df": [pd.DataFrame({
            "name": ["tod", "tow", "tom", "toq", "toy"],
            "period": [24.0, 7.0, 1.0, 1.0, 1.0],
            "order": [3, 3, 1, 1, 5],
            "seas_names": ["daily", "weekly", "monthly", "quarterly", "yearly"]})],
    })
    assert model_components.growth == {}
    assert model_components.events == {
        "daily_event_df_dict": [None],
    }
    assert model_components.changepoints == {
        "changepoints_dict": [None],
        "seasonality_changepoints_dict": [None],
    }
    assert model_components.autoregression == {
        "autoreg_dict": [None],
        "simulation_num": [10],
        "fast_simulation": [False]
    }
    assert model_components.regressors == {}
    assert model_components.uncertainty == {
        "uncertainty_dict": [None],
    }
    assert_equal(model_components.custom, {
        "silverkite": [SilverkiteForecast()],
        "silverkite_diagnostics": [SilverkiteDiagnostics()],
        "origin_for_time_vars": [None],
        "extra_pred_cols": ["ct1"],  # linear growth
        "drop_pred_cols": [None],
        "explicit_pred_cols": [None],
        "fit_algorithm_dict": [{
            "fit_algorithm": "linear",
            "fit_algorithm_params": None,
        }],
        "min_admissible_value": [None],
        "max_admissible_value": [None],
        "regression_weight_col": [None],
        "normalize_method": [None]
    }, ignore_keys={
        "silverkite": None,
        "silverkite_diagnostics": None
    })
    assert model_components.custom["silverkite"][0] != silverkite  # a different instance was created
    assert model_components.custom["silverkite_diagnostics"][0] != silverkite_diagnostics

    # overwrite some parameters
    time_properties = {
        "origin_for_time_vars": 2020
    }
    original_components = dataclasses.replace(model_components_param)  # creates a copy
    updated_components = apply_default_model_components(
        model_components=model_components_param,
        time_properties=time_properties)
    assert original_components == model_components_param  # not mutated by the function
    assert updated_components.seasonality == model_components_param.seasonality
    assert updated_components.events == {
        "daily_event_df_dict": [None],
    }
    assert updated_components.changepoints == {
        "changepoints_dict": {  # combination of defaults and provided params
            "method": "uniform",
            "n_changepoints": 20,
        },
        "seasonality_changepoints_dict": [None],
    }
    assert updated_components.autoregression == {
        "autoreg_dict": [None],
        "simulation_num": [10],
        "fast_simulation": [False]}
    assert updated_components.uncertainty == model_components_param.uncertainty
    assert updated_components.custom == {  # combination of defaults and provided params
        "silverkite": silverkite,  # the same object that was passed in (not a copy)
        "silverkite_diagnostics": silverkite_diagnostics,
        "origin_for_time_vars": [time_properties["origin_for_time_vars"]],  # from time_properties
        "extra_pred_cols": [["ct1"], ["ct2"], ["regressor1", "regressor3"]],
        "drop_pred_cols": [None],
        "explicit_pred_cols": [None],
        "max_admissible_value": 4,
        "fit_algorithm_dict": [{
            "fit_algorithm": "linear",
            "fit_algorithm_params": None,
        }],
        "min_admissible_value": [None],
        "normalize_method": [None],
        "regression_weight_col": [None],
    }

    # `time_properties` without start_year key
    updated_components = apply_default_model_components(
        model_components=model_components_param,
        time_properties={})
    assert updated_components.custom["origin_for_time_vars"] == [None]

    updated_components = apply_default_model_components(
        model_components=ModelComponentsParam(
            autoregression={
                "autoreg_dict": {
                    "lag_dict": {"orders": [7]},
                    "agg_lag_dict": {
                        "orders_list": [[7, 7*2, 7*3]],
                        "interval_list": [(7, 7*2)]},
                    "series_na_fill_func": lambda s: s.bfill().ffill()}
            })
    )

    autoreg_dict = updated_components.autoregression["autoreg_dict"]
    assert autoreg_dict["lag_dict"] == {"orders": [7]}
    assert autoreg_dict["agg_lag_dict"]["orders_list"] == [[7, 14, 21]]
    assert autoreg_dict["agg_lag_dict"]["interval_list"] == [(7, 14)]

    updated_components = apply_default_model_components(
        model_components=ModelComponentsParam(
            lagged_regressors={
                "lagged_regressor_dict": {
                    "regressor2": {
                        "lag_dict": {"orders": [5]},
                        "agg_lag_dict": {
                            "orders_list": [[7, 7 * 2, 7 * 3]],
                            "interval_list": [(8, 7 * 2)]},
                        "series_na_fill_func": lambda s: s.bfill().ffill()}
                }
            })
    )
    lagged_regressor_dict = updated_components.lagged_regressors["lagged_regressor_dict"]
    assert list(lagged_regressor_dict.keys()) == ["regressor2"]
    assert lagged_regressor_dict["regressor2"]["lag_dict"] == {"orders": [5]}
    assert lagged_regressor_dict["regressor2"]["agg_lag_dict"]["orders_list"] == [[7, 14, 21]]
    assert lagged_regressor_dict["regressor2"]["agg_lag_dict"]["interval_list"] == [(8, 14)]


def test_property():
    """Tests properties"""
    assert SilverkiteTemplate().allow_model_template_list is False
    assert SilverkiteTemplate().allow_model_components_param_list is False

    template = SilverkiteTemplate()
    assert template.DEFAULT_MODEL_TEMPLATE == "SK"
    assert isinstance(template.estimator, SilverkiteEstimator)
    assert template.estimator.coverage is None
    assert template.apply_forecast_config_defaults().model_template == "SK"

    estimator = SilverkiteEstimator(coverage=0.99)
    template = SilverkiteTemplate(estimator=estimator)
    assert template.estimator is estimator


def test_get_regressor_cols():
    template = SilverkiteTemplate()
    template.config = template.apply_forecast_config_defaults()
    template.df = pd.DataFrame(columns=["p1", "p2"])
    regressor_cols = template.get_regressor_cols()
    assert regressor_cols is None

    template.config.model_components_param = ModelComponentsParam(
        custom={}
    )
    regressor_cols = template.get_regressor_cols()
    assert regressor_cols is None

    template.config.model_components_param = ModelComponentsParam(
        custom={
            "extra_pred_cols": ["p1", "p2", "p3", template.config.metadata_param.time_col]
        }
    )
    regressor_cols = template.get_regressor_cols()
    assert set(regressor_cols) == {"p1", "p2"}

    template.config.model_components_param = ModelComponentsParam(
        custom={
            "extra_pred_cols": [["p1"], ["p2", "p3"], None, []]
        }
    )
    regressor_cols = template.get_regressor_cols()
    assert set(regressor_cols) == {"p1", "p2"}


def test_get_lagged_regressor_info():
    # Without lagged regressors
    template = SilverkiteTemplate()
    template.config = template.apply_forecast_config_defaults()
    expected_lagged_regressor_info = {
        "lagged_regressor_cols": None,
        "overall_min_lag_order": None,
        "overall_max_lag_order": None
    }
    assert template.get_lagged_regressor_info() == expected_lagged_regressor_info

    # With lagged regressors
    template.config.model_components_param = ModelComponentsParam(
        lagged_regressors={
            "lagged_regressor_dict": [{
                "regressor2": {
                    "lag_dict": {"orders": [5]},
                    "agg_lag_dict": {
                        "orders_list": [[7, 7 * 2, 7 * 3]],
                        "interval_list": [(8, 7 * 2)]},
                    "series_na_fill_func": lambda s: s.bfill().ffill()}
            }, {
                "regressor_bool": {
                    "lag_dict": {"orders": [1]},
                    "agg_lag_dict": {
                        "orders_list": [[7, 7 * 2]],
                        "interval_list": [(8, 7 * 2)]},
                    "series_na_fill_func": lambda s: s.bfill().ffill()}
            }]
        })
    lagged_regressor_info = template.get_lagged_regressor_info()
    assert set(lagged_regressor_info["lagged_regressor_cols"]) == {"regressor2", "regressor_bool"}
    assert lagged_regressor_info["overall_min_lag_order"] == 1
    assert lagged_regressor_info["overall_max_lag_order"] == 21


def test_get_silverkite_hyperparameter_grid(model_components_param, silverkite, silverkite_diagnostics):
    template = SilverkiteTemplate()
    template.config = template.apply_forecast_config_defaults()
    hyperparameter_grid = template.get_hyperparameter_grid()
    expected_grid = {
        "estimator__silverkite": [SilverkiteForecast()],
        "estimator__silverkite_diagnostics": [SilverkiteDiagnostics()],
        "estimator__origin_for_time_vars": [None],
        "estimator__extra_pred_cols": [["ct1"]],
        "estimator__drop_pred_cols": [None],
        "estimator__explicit_pred_cols": [None],
        "estimator__train_test_thresh": [None],
        "estimator__training_fraction": [None],
        "estimator__fit_algorithm_dict": [{
            "fit_algorithm": "linear",
            "fit_algorithm_params": None}],
        "estimator__daily_event_df_dict": [None],
        "estimator__fs_components_df": [pd.DataFrame({
            "name": ["tod", "tow", "tom", "toq", "toy"],
            "period": [24.0, 7.0, 1.0, 1.0, 1.0],
            "order": [3, 3, 1, 1, 5],
            "seas_names": ["daily", "weekly", "monthly", "quarterly", "yearly"]})],
        "estimator__autoreg_dict": [None],
        "estimator__simulation_num": [10],
        "estimator__fast_simulation": [False],
        "estimator__lagged_regressor_dict": [None],
        "estimator__changepoints_dict": [None],
        "estimator__seasonality_changepoints_dict": [None],
        "estimator__changepoint_detector": [None],
        "estimator__min_admissible_value": [None],
        "estimator__max_admissible_value": [None],
        "estimator__normalize_method": [None],
        "estimator__regression_weight_col": [None],
        "estimator__uncertainty_dict": [None],
    }
    assert_equal(
        hyperparameter_grid,
        expected_grid,
        ignore_keys={"estimator__silverkite": None, "estimator__silverkite_diagnostics": None})
    assert hyperparameter_grid["estimator__silverkite"][0] != silverkite
    assert hyperparameter_grid["estimator__silverkite_diagnostics"][0] != silverkite_diagnostics

    # Tests auto-list conversion
    template.config.model_components_param = model_components_param
    template.time_properties = {"origin_for_time_vars": 2020}
    hyperparameter_grid = template.get_hyperparameter_grid()
    expected_grid = {
        "estimator__silverkite": [silverkite],
        "estimator__silverkite_diagnostics": [silverkite_diagnostics],
        "estimator__origin_for_time_vars": [2020],
        "estimator__extra_pred_cols": [["ct1"], ["ct2"], ["regressor1", "regressor3"]],
        "estimator__drop_pred_cols": [None],
        "estimator__explicit_pred_cols": [None],
        "estimator__train_test_thresh": [None],
        "estimator__training_fraction": [None],
        "estimator__fit_algorithm_dict": [{
            "fit_algorithm": "linear",
            "fit_algorithm_params": None,
        }],
        "estimator__daily_event_df_dict": [None],
        "estimator__fs_components_df": [None],
        "estimator__autoreg_dict": [None],
        "estimator__simulation_num": [10],
        "estimator__fast_simulation": [False],
        "estimator__lagged_regressor_dict": [None],
        "estimator__changepoints_dict": [{
            "method": "uniform",
            "n_changepoints": 20,
        }],
        "estimator__seasonality_changepoints_dict": [None],
        "estimator__changepoint_detector": [None],
        "estimator__min_admissible_value": [None],
        "estimator__max_admissible_value": [4],
        "estimator__normalize_method": [None],
        "estimator__regression_weight_col": [None],
        "estimator__uncertainty_dict": [{
            "uncertainty_method": "simple_conditional_residuals"
        }],
    }
    assert_equal(hyperparameter_grid, expected_grid)

    # Tests hyperparameter_override
    template.config.model_components_param.hyperparameter_override = [
        {
            "input__response__null__max_frac": 0.1,
            "estimator__min_admissible_value": [2],
            "estimator__extra_pred_cols": ["override_estimator__extra_pred_cols"],
        },
        {},
        {
            "estimator__extra_pred_cols": ["val1", "val2"],
            "estimator__origin_for_time_vars": [2019],
        },
        None
    ]
    template.time_properties = {"origin_for_time_vars": 2020}
    hyperparameter_grid = template.get_hyperparameter_grid()
    expected_grid["estimator__origin_for_time_vars"] = [2020]
    updated_grid1 = expected_grid.copy()
    updated_grid1["input__response__null__max_frac"] = [0.1]
    updated_grid1["estimator__min_admissible_value"] = [2]
    updated_grid1["estimator__extra_pred_cols"] = [["override_estimator__extra_pred_cols"]]
    updated_grid2 = expected_grid.copy()
    updated_grid2["estimator__extra_pred_cols"] = [["val1", "val2"]]
    updated_grid2["estimator__origin_for_time_vars"] = [2019]
    expected_grid = [
        updated_grid1,
        expected_grid,
        updated_grid2,
        expected_grid]
    assert_equal(hyperparameter_grid, expected_grid)


def test_apply_template_decorator():
    data = generate_df_for_tests(freq="D", periods=10)
    df = data["df"]
    template = SilverkiteTemplate()
    with pytest.raises(
            ValueError,
            match="SilverkiteTemplate only supports config.model_template='SK', found 'PROPHET'"):
        template.apply_template_for_pipeline_params(
            df=df,
            config=ForecastConfig(model_template="PROPHET")
        )


def test_silverkite_template():
    """Tests test_silverkite_template with default config"""
    data = generate_df_for_tests(freq="D", periods=10)
    df = data["df"]
    template = SilverkiteTemplate()
    config = ForecastConfig(model_template="SK")
    params = template.apply_template_for_pipeline_params(
        df=df,
        config=config
    )
    assert config == ForecastConfig(model_template="SK")  # not modified
    pipeline = params.pop("pipeline", None)

    metric = EvaluationMetricEnum.MeanAbsolutePercentError
    expected_params = dict(
        df=df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        date_format=None,
        freq=None,
        train_end_date=None,
        anomaly_info=None,
        # model
        regressor_cols=None,
        lagged_regressor_cols=None,
        estimator=None,
        hyperparameter_grid=template.hyperparameter_grid,
        hyperparameter_budget=None,
        n_jobs=COMPUTATION_N_JOBS,
        verbose=1,
        # forecast
        forecast_horizon=None,
        coverage=None,
        test_horizon=None,
        periods_between_train_test=None,
        agg_periods=None,
        agg_func=None,
        # evaluation
        score_func=metric.name,
        score_func_greater_is_better=metric.get_metric_greater_is_better(),
        cv_report_metrics=CV_REPORT_METRICS_ALL,
        null_model_params=None,
        relative_error_tolerance=None,
        # CV
        cv_horizon=None,
        cv_min_train_periods=None,
        cv_expanding_window=True,
        cv_use_most_recent_splits=None,
        cv_periods_between_splits=None,
        cv_periods_between_train_test=None,
        cv_max_splits=3
    )
    assert_basic_pipeline_equal(pipeline, template.pipeline)
    assert_equal(params, expected_params)


def test_silverkite_template_custom(model_components_param):
    """"Tests simple_silverkite_template with custom parameters,
    and data that has regressors"""
    data = generate_df_with_reg_for_tests(
        freq="H",
        periods=300*24,
        remove_extra_cols=True,
        mask_test_actuals=True)
    df = data["df"]
    time_col = "some_time_col"
    value_col = "some_value_col"
    df.rename({
        TIME_COL: time_col,
        VALUE_COL: value_col
    }, axis=1, inplace=True)

    metric = EvaluationMetricEnum.MeanAbsoluteError
    # anomaly adjustment adds 10.0 to every record
    adjustment_size = 10.0
    anomaly_df = pd.DataFrame({
        START_TIME_COL: [df[time_col].min()],
        END_TIME_COL: [df[time_col].max()],
        ADJUSTMENT_DELTA_COL: [adjustment_size],
        METRIC_COL: [value_col]
    })
    anomaly_info = {
        "value_col": VALUE_COL,
        "anomaly_df": anomaly_df,
        "start_time_col": START_TIME_COL,
        "end_time_col": END_TIME_COL,
        "adjustment_delta_col": ADJUSTMENT_DELTA_COL,
        "filter_by_dict": {METRIC_COL: VALUE_COL},
        "adjustment_method": "add"
    }
    metadata = MetadataParam(
        time_col=time_col,
        value_col=value_col,
        freq="H",
        date_format="%Y-%m-%d-%H",
        train_end_date=datetime.datetime(2019, 7, 1),
        anomaly_info=anomaly_info
    )
    evaluation_metric = EvaluationMetricParam(
        cv_selection_metric=metric.name,
        cv_report_metrics=[EvaluationMetricEnum.MedianAbsolutePercentError.name],
        agg_periods=24,
        agg_func=np.max,
        null_model_params={
            "strategy": "quantile",
            "constant": None,
            "quantile": 0.8
        },
        relative_error_tolerance=0.01
    )
    evaluation_period = EvaluationPeriodParam(
        test_horizon=1,
        periods_between_train_test=2,
        cv_horizon=3,
        cv_min_train_periods=4,
        cv_expanding_window=True,
        cv_use_most_recent_splits=True,
        cv_periods_between_splits=5,
        cv_periods_between_train_test=6,
        cv_max_splits=7
    )
    computation = ComputationParam(
        hyperparameter_budget=10,
        n_jobs=None,
        verbose=1
    )
    forecast_horizon = 20
    coverage = 0.7
    template = SilverkiteTemplate()
    params = template.apply_template_for_pipeline_params(
        df=df,
        config=ForecastConfig(
            model_template=ModelTemplateEnum.SK.name,
            metadata_param=metadata,
            forecast_horizon=forecast_horizon,
            coverage=coverage,
            evaluation_metric_param=evaluation_metric,
            evaluation_period_param=evaluation_period,
            model_components_param=model_components_param,
            computation_param=computation
        )
    )
    pipeline = params.pop("pipeline", None)
    expected_params = dict(
        df=df,
        time_col=time_col,
        value_col=value_col,
        date_format=metadata.date_format,
        freq=metadata.freq,
        train_end_date=metadata.train_end_date,
        anomaly_info=metadata.anomaly_info,
        # model
        regressor_cols=template.regressor_cols,
        lagged_regressor_cols=template.lagged_regressor_cols,
        estimator=None,
        hyperparameter_grid=template.hyperparameter_grid,
        hyperparameter_budget=computation.hyperparameter_budget,
        n_jobs=computation.n_jobs,
        verbose=computation.verbose,
        # forecast
        forecast_horizon=forecast_horizon,
        coverage=coverage,
        test_horizon=evaluation_period.test_horizon,
        periods_between_train_test=evaluation_period.periods_between_train_test,
        agg_periods=evaluation_metric.agg_periods,
        agg_func=evaluation_metric.agg_func,
        relative_error_tolerance=evaluation_metric.relative_error_tolerance,
        # evaluation
        score_func=metric.name,
        score_func_greater_is_better=metric.get_metric_greater_is_better(),
        cv_report_metrics=evaluation_metric.cv_report_metrics,
        null_model_params=evaluation_metric.null_model_params,
        # CV
        cv_horizon=evaluation_period.cv_horizon,
        cv_min_train_periods=evaluation_period.cv_min_train_periods,
        cv_expanding_window=evaluation_period.cv_expanding_window,
        cv_use_most_recent_splits=evaluation_period.cv_use_most_recent_splits,
        cv_periods_between_splits=evaluation_period.cv_periods_between_splits,
        cv_periods_between_train_test=evaluation_period.cv_periods_between_train_test,
        cv_max_splits=evaluation_period.cv_max_splits
    )
    assert_basic_pipeline_equal(pipeline, template.pipeline)
    assert_equal(params, expected_params)


# The following tests run `silverkite_template` through the pipeline.
# They ensure `forecast_pipeline` and `SilverkiteEstimator` can interpret the parameters
# passed directly and through the `hyperparameter_grid`.
def test_run_template_1():
    """Runs default template"""
    data = generate_df_for_tests(
        freq="H",
        periods=700 * 24)
    df = data["train_df"]
    forecast_horizon = data["test_df"].shape[0]

    config = ForecastConfig(
        model_template=ModelTemplateEnum.SK.name,
        forecast_horizon=forecast_horizon,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = Forecaster().run_forecast_config(
            df=df,
            config=config,
        )

        rmse = EvaluationMetricEnum.RootMeanSquaredError.get_metric_name()
        q80 = EvaluationMetricEnum.Quantile80.get_metric_name()
        assert result.backtest.test_evaluation[rmse] == pytest.approx(2.037, rel=1e-2)
        assert result.backtest.test_evaluation[q80] == pytest.approx(0.836, rel=1e-2)
        assert result.forecast.train_evaluation[rmse] == pytest.approx(2.004, rel=1e-2)
        assert result.forecast.train_evaluation[q80] == pytest.approx(0.800, rel=1e-2)
        check_forecast_pipeline_result(
            result,
            coverage=None,
            strategy=None,
            score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name,
            greater_is_better=False)


def test_run_template_2():
    """Runs custom template with all options"""
    data = generate_df_with_reg_for_tests(
        freq="D",
        periods=400,
        remove_extra_cols=True,
        mask_test_actuals=True)
    reg_cols = ["regressor1", "regressor2", "regressor_categ"]
    keep_cols = [TIME_COL, VALUE_COL] + reg_cols
    df = data["df"][keep_cols]
    forecast_horizon = data["test_df"].shape[0]

    daily_event_df_dict = generate_holiday_events(
        countries=["UnitedStates"],
        holidays_to_model_separately=["New Year's Day"],
        year_start=2017,
        year_end=2022,
        pre_num=2,
        post_num=2)
    event_pred_cols = get_event_pred_cols(daily_event_df_dict)
    model_components = ModelComponentsParam(
        seasonality={
            "fs_components_df": pd.DataFrame({
                "name": ["tow", "tom", "toq", "toy"],
                "period": [7.0, 1.0, 1.0, 1.0],
                "order": [2, 1, 1, 5],
                "seas_names": ["weekly", "monthly", "quarterly", "yearly"]
            })
        },
        events={
            "daily_event_df_dict": daily_event_df_dict
        },
        changepoints={
            "changepoints_dict": {
                "method": "auto",
                "yearly_seasonality_order": 3,
                "regularization_strength": 0.5,
                "resample_freq": "14D",
                "potential_changepoint_distance": "56D",
                "no_changepoint_proportion_from_end": 0.2
            },
            "seasonality_changepoints_dict": {
                "potential_changepoint_distance": "60D",
                "regularization_strength": 0.5,
                "no_changepoint_proportion_from_end": 0.2
            },
        },
        autoregression=None,
        uncertainty={
            "uncertainty_dict": None,
        },
        custom={
            "origin_for_time_vars": None,
            "extra_pred_cols": [["ct1"] + reg_cols + event_pred_cols],  # growth, regressors, events
            "fit_algorithm_dict": {
                "fit_algorithm": "ridge",
                "fit_algorithm_params": {"cv": 2}
            },
            "min_admissible_value": min(df[VALUE_COL]) - abs(max(df[VALUE_COL])),
            "max_admissible_value": max(df[VALUE_COL]) * 2,
        }
    )
    config = ForecastConfig(
        model_template=ModelTemplateEnum.SK.name,
        forecast_horizon=forecast_horizon,
        coverage=0.9,
        model_components_param=model_components,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = Forecaster().run_forecast_config(
            df=df,
            config=config,
        )
        rmse = EvaluationMetricEnum.RootMeanSquaredError.get_metric_name()
        q80 = EvaluationMetricEnum.Quantile80.get_metric_name()
        assert result.backtest.test_evaluation[rmse] == pytest.approx(2.692, rel=1e-2)
        assert result.backtest.test_evaluation[q80] == pytest.approx(1.531, rel=1e-2)
        assert result.backtest.test_evaluation[PREDICTION_BAND_COVERAGE] == pytest.approx(0.823, rel=1e-2)
        assert result.forecast.train_evaluation[rmse] == pytest.approx(2.304, rel=1e-2)
        assert result.forecast.train_evaluation[q80] == pytest.approx(0.921, rel=1e-2)
        assert result.forecast.train_evaluation[PREDICTION_BAND_COVERAGE] == pytest.approx(0.897, rel=1e-2)
        check_forecast_pipeline_result(
            result,
            coverage=0.9,
            strategy=None,
            score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name,
            greater_is_better=False)


def test_run_template_3():
    """Runs custom template with monthly data"""
    data = generate_df_with_reg_for_tests(
        freq="MS",
        periods=48,
        remove_extra_cols=True,
        mask_test_actuals=True)
    reg_cols = ["regressor1", "regressor2", "regressor_categ"]
    keep_cols = [TIME_COL, VALUE_COL] + reg_cols
    df = data["df"][keep_cols]
    forecast_horizon = data["test_df"].shape[0]

    model_components = ModelComponentsParam(
        seasonality=None,
        events=None,
        changepoints=None,
        autoregression=None,
        uncertainty={"uncertainty_dict": None})
    config = ForecastConfig(
        model_template=ModelTemplateEnum.SK.name,
        forecast_horizon=forecast_horizon,
        coverage=0.9,
        model_components_param=model_components,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = Forecaster().run_forecast_config(
            df=df,
            config=config,
        )
        rmse = EvaluationMetricEnum.RootMeanSquaredError.get_metric_name()
        assert result.backtest.test_evaluation[rmse] == pytest.approx(4.08, rel=1e-1)
        check_forecast_pipeline_result(
            result,
            coverage=0.9,
            strategy=None,
            score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name,
            greater_is_better=False)


def test_run_template_4():
    """Runs custom template with monthly data and auto-regression"""
    data = generate_df_with_reg_for_tests(
        freq="MS",
        periods=48,
        remove_extra_cols=True,
        mask_test_actuals=True)
    reg_cols = ["regressor1", "regressor2", "regressor_categ"]
    keep_cols = [TIME_COL, VALUE_COL] + reg_cols
    df = data["df"][keep_cols]
    forecast_horizon = data["test_df"].shape[0]

    model_components = ModelComponentsParam(
        custom=dict(
            fit_algorithm_dict=dict(fit_algorithm="linear"),
            extra_pred_cols=["ct2"]),
        autoregression=dict(autoreg_dict=dict(lag_dict=dict(orders=[1]))),
        uncertainty=dict(uncertainty_dict=None))
    config = ForecastConfig(
        model_template=ModelTemplateEnum.SK.name,
        forecast_horizon=forecast_horizon,
        coverage=0.9,
        model_components_param=model_components,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = Forecaster().run_forecast_config(
            df=df,
            config=config,
        )
        rmse = EvaluationMetricEnum.RootMeanSquaredError.get_metric_name()
        assert result.backtest.test_evaluation[rmse] == pytest.approx(4.95, rel=1e-1)
        check_forecast_pipeline_result(
            result,
            coverage=0.9,
            strategy=None,
            score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name,
            greater_is_better=False)


def test_run_template_5():
    """Runs custom template with monthly data and auto-regression with fast simulation"""
    data = generate_df_with_reg_for_tests(
        freq="MS",
        periods=48,
        remove_extra_cols=True,
        mask_test_actuals=True)
    reg_cols = ["regressor1", "regressor2", "regressor_categ"]
    keep_cols = [TIME_COL, VALUE_COL] + reg_cols
    df = data["df"][keep_cols]
    forecast_horizon = data["test_df"].shape[0]

    model_components = ModelComponentsParam(
        custom=dict(
            fit_algorithm_dict=dict(fit_algorithm="linear"),
            extra_pred_cols=["ct2"]),
        autoregression=dict(
            autoreg_dict=dict(lag_dict=dict(orders=[1])),
            fast_simulation=True),
        uncertainty=dict(uncertainty_dict=None))
    config = ForecastConfig(
        model_template=ModelTemplateEnum.SK.name,
        forecast_horizon=forecast_horizon,
        coverage=0.9,
        model_components_param=model_components,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = Forecaster().run_forecast_config(
            df=df,
            config=config,
        )
        rmse = EvaluationMetricEnum.RootMeanSquaredError.get_metric_name()
        assert result.backtest.test_evaluation[rmse] == pytest.approx(4.95, rel=1e-1)
        check_forecast_pipeline_result(
            result,
            coverage=0.9,
            strategy=None,
            score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name,
            greater_is_better=False)

    assert result.model._final_estimator.model_dict["fast_simulation"] is True


def test_run_template_6():
    """Runs custom template with monthly data, auto-regression and lagged regressors"""
    data = generate_df_with_reg_for_tests(
        freq="MS",
        periods=48,
        remove_extra_cols=True,
        mask_test_actuals=True)
    reg_cols_all = ["regressor1", "regressor2", "regressor_categ"]
    reg_cols = ["regressor1"]
    keep_cols = [TIME_COL, VALUE_COL] + reg_cols_all
    df = data["df"][keep_cols]
    test_df = data["test_df"]
    forecast_horizon = test_df.shape[0]

    model_components = ModelComponentsParam(
        custom=dict(
            fit_algorithm_dict=dict(fit_algorithm="linear"),
            extra_pred_cols=reg_cols),
        autoregression=dict(autoreg_dict=dict(lag_dict=dict(orders=[1]))),
        lagged_regressors={
            "lagged_regressor_dict": [
                {"regressor2": "auto"},
                {"regressor_categ": {"lag_dict": {"orders": [5]}}}
            ]},
        uncertainty=dict(uncertainty_dict=None))
    config = ForecastConfig(
        model_template=ModelTemplateEnum.SK.name,
        forecast_horizon=forecast_horizon,
        coverage=0.9,
        model_components_param=model_components,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = Forecaster().run_forecast_config(
            df=df,
            config=config,
        )
        rmse = EvaluationMetricEnum.RootMeanSquaredError.get_metric_name()
        assert result.backtest.test_evaluation[rmse] == pytest.approx(4.46, rel=1e-1)
        check_forecast_pipeline_result(
            result,
            coverage=0.9,
            strategy=None,
            score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name,
            greater_is_better=False)
        # Checks lagged regressor columns
        actual_pred_cols = set(result.model[-1].model_dict["pred_cols"])
        actual_x_mat_cols = set(result.model[-1].model_dict["x_mat"].columns)
        expected_pred_cols = {
            "regressor1",
            "y_lag1",
            "regressor_categ_lag5"
        }
        expected_x_mat_cols = {
            "regressor1",
            "y_lag1",
            "regressor_categ_lag5[T.c2]",
            "regressor_categ_lag5[T.c2]"
        }
        assert expected_pred_cols.issubset(actual_pred_cols)
        assert expected_x_mat_cols.issubset(actual_x_mat_cols)

        trained_estimator = result.model[-1]
        forecast = trained_estimator.forecast
        forecast_x_mat = trained_estimator.forecast_x_mat
        fit_x_mat = trained_estimator.model_dict["x_mat"]
        assert len(forecast) == len(df)
        assert len(forecast_x_mat) == len(df)
        assert len(fit_x_mat) == len(df) - forecast_horizon

        # Does a new prediction and checks if the ``forecast`` and
        # ``forecast_x_mat`` are updated
        pred_df = trained_estimator.predict(test_df[:3])
        forecast_x_mat = trained_estimator.forecast_x_mat
        forecast = trained_estimator.forecast
        assert len(pred_df) == 3
        assert len(forecast) == 3
        assert len(forecast_x_mat) == 3
