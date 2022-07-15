import datetime

import numpy as np
import pandas as pd

import greykite.common.constants as cst
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.common.python_utils import assert_equal
from greykite.common.testing_utils import generate_df_for_tests
from greykite.common.testing_utils import generate_df_with_reg_for_tests
from greykite.framework.constants import COMPUTATION_N_JOBS
from greykite.framework.constants import CV_REPORT_METRICS_ALL
from greykite.framework.templates.auto_arima_template import AutoArimaTemplate
from greykite.framework.templates.autogen.forecast_config import ComputationParam
from greykite.framework.templates.autogen.forecast_config import EvaluationMetricParam
from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.utils.framework_testing_utils import assert_basic_pipeline_equal
from greykite.sklearn.estimator.auto_arima_estimator import AutoArimaEstimator


def test_property():
    """Tests properties"""
    assert AutoArimaTemplate().allow_model_template_list is False
    assert AutoArimaTemplate().allow_model_components_param_list is False
    assert AutoArimaTemplate().get_regressor_cols() is None

    template = AutoArimaTemplate()
    assert template.DEFAULT_MODEL_TEMPLATE == "AUTO_ARIMA"
    assert isinstance(template.estimator, AutoArimaEstimator)
    assert template.estimator.coverage == 0.90
    assert template.apply_forecast_config_defaults().model_template == "AUTO_ARIMA"

    estimator = AutoArimaEstimator(coverage=0.99)
    template = AutoArimaTemplate(estimator=estimator)
    assert template.estimator is estimator


def test_get_regressor_cols():
    """Tests get_regressor_names"""
    template = AutoArimaTemplate()
    # no regressors
    model_components = ModelComponentsParam()
    template.config = ForecastConfig(model_components_param=model_components)
    assert template.get_regressor_cols() is None

    model_components = ModelComponentsParam(regressors={})
    template.config = ForecastConfig(model_components_param=model_components)
    assert template.get_regressor_cols() is None


def test_auto_arima_hyperparameter_grid_default():
    """Tests get_hyperparameter_grid and apply_prophet_model_components_defaults"""
    template = AutoArimaTemplate()
    template.config = template.apply_forecast_config_defaults()
    # model_components is None
    hyperparameter_grid = template.get_hyperparameter_grid()

    expected_grid = {
        # Additional parameters
        "estimator__freq": [None],
        # pmdarima fit parameters
        "estimator__start_p": [2],
        "estimator__d": [None],
        "estimator__start_q": [2],
        "estimator__max_p": [5],
        "estimator__max_d": [2],
        "estimator__max_q": [5],
        "estimator__start_P": [1],
        "estimator__D": [None],
        "estimator__start_Q": [1],
        "estimator__max_P": [2],
        "estimator__max_D": [1],
        "estimator__max_Q": [2],
        "estimator__max_order": [5],
        "estimator__m": [1],
        "estimator__seasonal": [True],
        "estimator__stationary": [False],
        "estimator__information_criterion": ["aic"],
        "estimator__alpha": [0.05],
        "estimator__test": ["kpss"],
        "estimator__seasonal_test": ["ocsb"],
        "estimator__stepwise": [True],
        "estimator__n_jobs": [1],
        "estimator__start_params": [None],
        "estimator__trend": [None],
        "estimator__method": ["lbfgs"],
        "estimator__maxiter": [50],
        "estimator__offset_test_args": [None],
        "estimator__seasonal_test_args": [None],
        "estimator__suppress_warnings": [True],
        "estimator__error_action": ["trace"],
        "estimator__trace": [False],
        "estimator__random": [False],
        "estimator__random_state": [None],
        "estimator__n_fits": [10],
        "estimator__out_of_sample_size": [0],
        "estimator__scoring": ["mse"],
        "estimator__scoring_args": [None],
        "estimator__with_intercept": ["auto"],
        # pmdarima predict parameters
        "estimator__return_conf_int": [True],
        "estimator__dynamic": [False]
    }
    assert_equal(actual=hyperparameter_grid, expected=expected_grid)


def test_auto_arima_template_default():
    """Tests auto_arima_template with default values, for limited data"""
    num_days = 10
    data = generate_df_for_tests(freq="D", periods=num_days, train_start_date="2018-01-01")
    df = data["df"]
    template = AutoArimaTemplate()
    config = ForecastConfig(model_template="AUTO_ARIMA")
    params = template.apply_template_for_pipeline_params(
        df=df,
        config=config
    )
    # not modified
    assert config == ForecastConfig(model_template="AUTO_ARIMA")
    # checks result
    metric = EvaluationMetricEnum.MeanAbsolutePercentError
    pipeline = params.pop("pipeline", None)
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


def test_auto_arima_template_custom():
    """Tests auto arima template with custom values, with long range input"""
    # prepares input data
    data = generate_df_with_reg_for_tests(
        freq="H",
        periods=300*24,
        remove_extra_cols=True,
        mask_test_actuals=True)
    df = data["df"]
    time_col = "some_time_col"
    value_col = "some_value_col"
    df.rename({
        cst.TIME_COL: time_col,
        cst.VALUE_COL: value_col
    }, axis=1, inplace=True)
    # prepares params and calls template
    metric = EvaluationMetricEnum.MeanAbsoluteError
    # anomaly adjustment adds 10.0 to every record
    adjustment_size = 10.0
    anomaly_df = pd.DataFrame({
        cst.START_TIME_COL: [df[time_col].min()],
        cst.END_TIME_COL: [df[time_col].max()],
        cst.ADJUSTMENT_DELTA_COL: [adjustment_size],
        cst.METRIC_COL: [value_col]
    })
    anomaly_info = {
        "value_col": cst.VALUE_COL,
        "anomaly_df": anomaly_df,
        "start_time_col": cst.START_TIME_COL,
        "end_time_col": cst.END_TIME_COL,
        "adjustment_delta_col": cst.ADJUSTMENT_DELTA_COL,
        "filter_by_dict": {cst.METRIC_COL: cst.VALUE_COL},
        "adjustment_method": "add"
    }

    metadata = MetadataParam(
        time_col=time_col,
        value_col=value_col,
        freq="H",
        date_format="%Y-%m-%d-%H",
        train_end_date=datetime.datetime(2019, 7, 1),
        anomaly_info=anomaly_info,
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
    model_components = ModelComponentsParam(
        # Everything except `custom` and `hyperparameter_override` are ignored
        seasonality={
            "yearly_seasonality": [True],
            "weekly_seasonality": [False],
            "daily_seasonality": [4],
            "add_seasonality_dict": [{
                "yearly": {
                    "period": 365.25,
                    "fourier_order": 20,
                    "prior_scale": 20.0
                },
                "quarterly": {
                    "period": 365.25/4,
                    "fourier_order": 15
                },
                "weekly": {
                    "period": 7,
                    "fourier_order": 35,
                    "prior_scale": 30.0
                }
            }]
        },
        growth={
            "growth_term": "linear"
        },
        events={
            "holiday_lookup_countries": ["UnitedStates", "UnitedKingdom", "India"],
            "holiday_pre_num_days": [2],
            "holiday_post_num_days": [3],
            "holidays_prior_scale": [5.0]
        },
        regressors={
            "add_regressor_dict": [{
                "regressor1": {
                    "prior_scale": 10.0,
                    "mode": 'additive'
                },
                "regressor2": {
                    "prior_scale": 20.0,
                    "mode": 'multiplicative'
                },
            }]
        },
        changepoints={
            "changepoint_prior_scale": [0.05],
            "changepoints": [None],
            "n_changepoints": [50],
            "changepoint_range": [0.9]
        },
        uncertainty={
            "mcmc_samples": [500],
            "uncertainty_samples": [2000]
        },
        custom={
            "start_p": [1],
            "max_p": [10]
        },
        hyperparameter_override={
            "estimator__max_p": [8, 10],
            "estimator__information_criterion": ["bic"],
        }
    )
    computation = ComputationParam(
        hyperparameter_budget=10,
        n_jobs=None,
        verbose=1
    )
    forecast_horizon = 20
    coverage = 0.7
    config = ForecastConfig(
        model_template=ModelTemplateEnum.AUTO_ARIMA.name,
        metadata_param=metadata,
        forecast_horizon=forecast_horizon,
        coverage=coverage,
        evaluation_metric_param=evaluation_metric,
        evaluation_period_param=evaluation_period,
        model_components_param=model_components,
        computation_param=computation
    )
    template = AutoArimaTemplate()
    params = template.apply_template_for_pipeline_params(
        df=df,
        config=config
    )
    pipeline = params.pop("pipeline", None)
    assert_basic_pipeline_equal(pipeline, template.pipeline)

    # Adding start_year and end_year based on the input df
    model_components.events["start_year"] = df[time_col].min().year
    model_components.events["end_year"] = df[time_col].max().year
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
        # evaluation
        score_func=metric.name,
        score_func_greater_is_better=metric.get_metric_greater_is_better(),
        cv_report_metrics=evaluation_metric.cv_report_metrics,
        null_model_params=evaluation_metric.null_model_params,
        relative_error_tolerance=evaluation_metric.relative_error_tolerance,
        # CV
        cv_horizon=evaluation_period.cv_horizon,
        cv_min_train_periods=evaluation_period.cv_min_train_periods,
        cv_expanding_window=evaluation_period.cv_expanding_window,
        cv_use_most_recent_splits=evaluation_period.cv_use_most_recent_splits,
        cv_periods_between_splits=evaluation_period.cv_periods_between_splits,
        cv_periods_between_train_test=evaluation_period.cv_periods_between_train_test,
        cv_max_splits=evaluation_period.cv_max_splits
    )
    assert_equal(params, expected_params)


def test_run_auto_arima_template_custom():
    """Tests running auto arima template through the pipeline"""
    data = generate_df_with_reg_for_tests(
        freq="D",
        periods=50,
        train_frac=0.8,
        conti_year_origin=2018,
        remove_extra_cols=True,
        mask_test_actuals=True)
    # select relevant columns for testing
    relevant_cols = [cst.TIME_COL, cst.VALUE_COL, "regressor1", "regressor2", "regressor3"]
    df = data["df"][relevant_cols]
    forecast_horizon = data["fut_time_num"]

    # Model components - custom holidays; other params as defaults
    model_components = ModelComponentsParam(
        # Everything except `custom` and `hyperparameter_override` are ignored
        seasonality={
            "seasonality_mode": ["additive"],
            "yearly_seasonality": ["auto"],
            "weekly_seasonality": [True],
            "daily_seasonality": ["auto"],
        },
        growth={
            "growth_term": ["linear"]
        },
        events={
            "holiday_pre_num_days": [1],
            "holiday_post_num_days": [1],
            "holidays_prior_scale": [1.0]
        },
        changepoints={
            "changepoint_prior_scale": [0.05],
            "n_changepoints": [1],
            "changepoint_range": [0.5],
        },
        regressors={
            "add_regressor_dict": [{
                "regressor1": {
                    "prior_scale": 10,
                    "standardize": True,
                    "mode": "additive"
                },
                "regressor2": {
                    "prior_scale": 15,
                    "standardize": False,
                    "mode": "additive"
                },
                "regressor3": {}
            }]
        },
        uncertainty={
            "uncertainty_samples": [10]
        },
        custom={
            "max_order": [10],
            "information_criterion": ["bic"]
        }
    )

    metadata = MetadataParam(
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        freq="D",
    )
    evaluation_period = EvaluationPeriodParam(
        test_horizon=5,  # speeds up test case
        periods_between_train_test=5,
        cv_horizon=0,  # speeds up test case
    )
    config = ForecastConfig(
        model_template=ModelTemplateEnum.AUTO_ARIMA.name,
        metadata_param=metadata,
        forecast_horizon=forecast_horizon,
        coverage=0.95,
        model_components_param=model_components,
        evaluation_period_param=evaluation_period,
    )
    result = Forecaster().run_forecast_config(
        df=df,
        config=config,
    )

    forecast_df = result.forecast.df_test.reset_index(drop=True)
    expected_cols = ["ts", "actual", "forecast", "forecast_lower", "forecast_upper"]
    assert list(forecast_df.columns) == expected_cols
    assert result.backtest.coverage == 0.95, "coverage is not correct"
    # NB: coverage is poor because of very small dataset size and low uncertainty_samples
    assert result.backtest.train_evaluation[cst.PREDICTION_BAND_COVERAGE] is not None
    assert result.backtest.test_evaluation[cst.PREDICTION_BAND_COVERAGE] is not None
    assert result.backtest.train_evaluation["MSE"] is not None
    assert result.backtest.test_evaluation["MSE"] is not None
    assert result.forecast.train_evaluation[cst.PREDICTION_BAND_COVERAGE] is not None
    assert result.forecast.train_evaluation["MSE"] is not None


def test_run_auto_arima_template_default():
    """Tests running default auto arima template through the pipeline"""
    df = generate_df_for_tests(
        freq="MS",
        periods=95,
        conti_year_origin=2018
    )["df"]
    coverage = 0.95
    forecast_horizon = 12
    metadata = MetadataParam(
        time_col=TIME_COL,
        value_col=VALUE_COL,
        freq="MS"
    )
    # Creates `AUTO_ARIMA` config
    model_template = ModelTemplateEnum.AUTO_ARIMA.name
    model_components = ModelComponentsParam()
    arima = ForecastConfig(
        metadata_param=metadata,
        forecast_horizon=forecast_horizon,
        coverage=coverage,
        model_template=model_template,
        model_components_param=model_components
    )
    forecaster = Forecaster()
    result = forecaster.run_forecast_config(
        df=df,
        config=arima
    )
    forecast_df = result.forecast.df_test.reset_index(drop=True)
    expected_cols = ["ts", "actual", "forecast", "forecast_lower", "forecast_upper"]
    assert list(forecast_df.columns) == expected_cols
    assert result.backtest.coverage == 0.95, "coverage is not correct"
    # NB: coverage is poor because of very small dataset size and low uncertainty_samples
    assert result.backtest.train_evaluation[cst.PREDICTION_BAND_COVERAGE] is not None
    assert result.backtest.test_evaluation[cst.PREDICTION_BAND_COVERAGE] is not None
