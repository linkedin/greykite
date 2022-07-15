import json
from typing import Optional

import pytest
from pytest import fail

from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.framework.constants import COMPUTATION_N_JOBS
from greykite.framework.constants import COMPUTATION_VERBOSE
from greykite.framework.constants import CV_REPORT_METRICS_ALL
from greykite.framework.constants import EVALUATION_PERIOD_CV_MAX_SPLITS
from greykite.framework.templates.autogen.forecast_config import ComputationParam
from greykite.framework.templates.autogen.forecast_config import EvaluationMetricParam
from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.autogen.forecast_config import forecast_config_from_dict
from greykite.framework.templates.autogen.forecast_config import from_list_dict
from greykite.framework.templates.autogen.forecast_config import from_list_dict_or_none
from greykite.framework.templates.forecast_config_defaults import ForecastConfigDefaults
from greykite.framework.templates.model_templates import ModelTemplateEnum


def test_from_list_dict():
    x = [{"k1": "v1"}, {"k2": "v2"}]
    assert from_list_dict(lambda x: x, x) == x
    assert from_list_dict(lambda x: x + "x", x) == [{"k1": "v1x"}, {"k2": "v2x"}]
    with pytest.raises(AssertionError):
        from_list_dict(lambda x: x, x[0])
    with pytest.raises(AssertionError):
        from_list_dict(lambda x: x, ["k", "v"])


def test_from_list_dict_or_none():
    x = [{"k1": "v1"}, {"k2": "v2"}, None, {}]
    assert from_list_dict_or_none(lambda x: x, x) == x
    assert from_list_dict_or_none(lambda x: x + "x", x) == [{"k1": "v1x"}, {"k2": "v2x"}, None, {}]
    with pytest.raises(AssertionError):
        from_list_dict_or_none(lambda x: x, x[0])
    with pytest.raises(AssertionError):
        from_list_dict_or_none(lambda x: x, None)


def assert_default_forecast_config(config: Optional[ForecastConfig] = None):
    """Asserts for the default ForecastConfig values"""
    try:
        config = ForecastConfigDefaults().apply_forecast_config_defaults(config)
        assert config.model_template == ModelTemplateEnum.AUTO.name
        assert config.metadata_param.time_col == TIME_COL
        assert config.metadata_param.value_col == VALUE_COL
        assert config.evaluation_period_param.periods_between_train_test is None
        assert config.evaluation_period_param.cv_max_splits == EVALUATION_PERIOD_CV_MAX_SPLITS
        assert config.evaluation_metric_param.cv_selection_metric == EvaluationMetricEnum.MeanAbsolutePercentError.name
        assert config.evaluation_metric_param.cv_report_metrics == CV_REPORT_METRICS_ALL
        assert config.computation_param.n_jobs == COMPUTATION_N_JOBS
        assert config.computation_param.verbose == COMPUTATION_VERBOSE
        assert config.to_dict()  # runs without error
    except Exception:
        fail("Config should not raise Exception")


def test_empty_forecast_config():
    """Tests an empty ForecastConfig dataclass"""
    assert_default_forecast_config()


def test_none_forecast_config():
    """Tests a None ForecastConfig dataclass"""
    assert_default_forecast_config(None)


def test_default_forecast_config():
    """Tests an Empty ForecastConfig dataclass"""
    assert_default_forecast_config(ForecastConfig())


def test_default_forecast_json():
    """Tests an Empty forecast json config"""
    json_str: str = "{}"
    forecast_dict = json.loads(json_str)
    config = forecast_config_from_dict(forecast_dict)
    assert_default_forecast_config(config)


def assert_forecast_config(config: Optional[ForecastConfig] = None):
    """Asserts the forecast config values. This function expects a particular config and is not generic"""
    config = ForecastConfigDefaults().apply_forecast_config_defaults(config)
    assert config.model_template == ModelTemplateEnum.SILVERKITE.name
    assert config.metadata_param.time_col == "custom_time_col"
    assert config.metadata_param.value_col == VALUE_COL
    assert config.metadata_param.freq is None
    assert config.metadata_param.date_format is None
    assert config.metadata_param.train_end_date is None
    assert config.metadata_param.anomaly_info == [{"key": "value"}, {"key2": "value2"}]
    assert config.evaluation_period_param.test_horizon == 10
    assert config.evaluation_period_param.periods_between_train_test == 5
    assert config.evaluation_period_param.cv_horizon is None
    assert config.evaluation_period_param.cv_min_train_periods == 20
    assert config.evaluation_period_param.cv_expanding_window is True
    assert config.evaluation_period_param.cv_use_most_recent_splits is None
    assert config.evaluation_period_param.cv_periods_between_splits is None
    assert config.evaluation_period_param.cv_periods_between_train_test == config.evaluation_period_param.periods_between_train_test
    assert config.evaluation_period_param.cv_max_splits == EVALUATION_PERIOD_CV_MAX_SPLITS
    assert config.evaluation_metric_param.cv_selection_metric == EvaluationMetricEnum.MeanSquaredError.name
    assert config.evaluation_metric_param.cv_report_metrics == [
        EvaluationMetricEnum.MeanAbsoluteError.name,
        EvaluationMetricEnum.MeanAbsolutePercentError.name]
    assert config.evaluation_metric_param.agg_periods is None
    assert config.evaluation_metric_param.agg_func is None
    assert config.evaluation_metric_param.null_model_params is None
    assert config.evaluation_metric_param.relative_error_tolerance == 0.02
    assert config.model_components_param.autoregression == {"autoreg_dict": {"autoreg_param": 0}}
    assert config.model_components_param.changepoints is None
    assert config.model_components_param.custom == {"custom_param": 1}
    assert config.model_components_param.growth == {"growth_param": 2}
    assert config.model_components_param.events == {"events_param": 3}
    assert (config.model_components_param.hyperparameter_override == [{"h1": 4}, {"h2": 5}, None]
            or config.model_components_param.hyperparameter_override == [{"h1": 4}, {"h2": 5}, {}])
    assert config.model_components_param.regressors == {"names": ["regressor1", "regressor2"]}
    assert config.model_components_param.lagged_regressors == {"lagged_regressor_dict": {"lag_reg_param": 0}}
    assert config.model_components_param.seasonality == {"seas_param": 6}
    assert config.model_components_param.uncertainty == {"uncertainty_param": 7}
    assert config.computation_param.hyperparameter_budget is None
    assert config.computation_param.n_jobs == COMPUTATION_N_JOBS
    assert config.computation_param.verbose == COMPUTATION_VERBOSE
    assert config.to_dict()  # runs without error


def test_forecast_config():
    """Tests ForecastConfig dataclass"""
    config = ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        metadata_param=MetadataParam(
            time_col="custom_time_col",
            anomaly_info=[{"key": "value"}, {"key2": "value2"}]),
        evaluation_period_param=EvaluationPeriodParam(
            test_horizon=10,
            periods_between_train_test=5,
            cv_min_train_periods=20),
        evaluation_metric_param=EvaluationMetricParam(
            cv_selection_metric=EvaluationMetricEnum.MeanSquaredError.name,
            cv_report_metrics=[EvaluationMetricEnum.MeanAbsoluteError.name,
                               EvaluationMetricEnum.MeanAbsolutePercentError.name],
            relative_error_tolerance=0.02),
        model_components_param=ModelComponentsParam(
            autoregression={"autoreg_dict": {"autoreg_param": 0}},
            changepoints=None,
            custom={"custom_param": 1},
            growth={"growth_param": 2},
            events={"events_param": 3},
            hyperparameter_override=[{"h1": 4}, {"h2": 5}, None],
            regressors={"names": ["regressor1", "regressor2"]},
            lagged_regressors={"lagged_regressor_dict": {"lag_reg_param": 0}},
            seasonality={"seas_param": 6},
            uncertainty={"uncertainty_param": 7}),
        computation_param=ComputationParam(n_jobs=None)
    )
    assert_forecast_config(config)

    # Tests a string passed to `cv_report_metrics`
    assert ForecastConfig(
        evaluation_metric_param=EvaluationMetricParam(
            cv_report_metrics=CV_REPORT_METRICS_ALL),
    ).to_dict()


def assert_forecast_config_json(config: Optional[ForecastConfig] = None):
    """Asserts the forecast config values. This function expects a particular config and is not generic"""
    config = ForecastConfigDefaults().apply_forecast_config_defaults(config)
    assert config.model_template == ModelTemplateEnum.SILVERKITE.name
    assert config.metadata_param.time_col == "custom_time_col"
    assert config.metadata_param.value_col == VALUE_COL
    assert config.metadata_param.freq is None
    assert config.metadata_param.date_format is None
    assert config.metadata_param.train_end_date is None
    assert config.metadata_param.anomaly_info == [{"key": "value"}, {"key2": "value2"}]
    assert config.evaluation_period_param.test_horizon == 10
    assert config.evaluation_period_param.periods_between_train_test == 5
    assert config.evaluation_period_param.cv_horizon is None
    assert config.evaluation_period_param.cv_min_train_periods == 20
    assert config.evaluation_period_param.cv_expanding_window is True
    assert config.evaluation_period_param.cv_use_most_recent_splits is None
    assert config.evaluation_period_param.cv_periods_between_splits is None
    assert config.evaluation_period_param.cv_periods_between_train_test == config.evaluation_period_param.periods_between_train_test
    assert config.evaluation_period_param.cv_max_splits == EVALUATION_PERIOD_CV_MAX_SPLITS
    assert config.evaluation_metric_param.cv_selection_metric == EvaluationMetricEnum.MeanSquaredError.name
    assert config.evaluation_metric_param.cv_report_metrics == [
        EvaluationMetricEnum.MeanAbsoluteError.name,
        EvaluationMetricEnum.MeanAbsolutePercentError.name]
    assert config.evaluation_metric_param.agg_periods is None
    assert config.evaluation_metric_param.agg_func is None
    assert config.evaluation_metric_param.null_model_params is None
    assert config.evaluation_metric_param.relative_error_tolerance == 0.02
    assert config.model_components_param.autoregression == {"autoreg_dict": {"autoreg_param": 0}}
    assert config.model_components_param.changepoints is None
    assert config.model_components_param.custom == {"custom_param": 1}
    assert config.model_components_param.growth == {"growth_param": 2}
    assert config.model_components_param.events == {"events_param": 3}
    assert (config.model_components_param.hyperparameter_override == [{"h1": 4}, {"h2": 5}, None]
            or config.model_components_param.hyperparameter_override == [{"h1": 4}, {"h2": 5}, {}])
    assert config.model_components_param.regressors == {"names": ["regressor1", "regressor2"]}
    assert config.model_components_param.lagged_regressors == {"lagged_regressor_dict": {"lag_reg_param": 0}}
    assert config.model_components_param.seasonality == {"seas_param": 6}
    assert config.model_components_param.uncertainty == {"uncertainty_param": 7}
    assert config.computation_param.hyperparameter_budget is None
    assert config.computation_param.n_jobs == COMPUTATION_N_JOBS
    assert config.computation_param.verbose == COMPUTATION_VERBOSE
    assert config.to_dict()  # runs without error


def test_forecast_config_json():
    """Tests ForecastConfig json"""
    json_str = """{
        "model_template": "SILVERKITE",
        "metadata_param": {
            "time_col": "custom_time_col",
            "anomaly_info": [{"key": "value"}, {"key2": "value2"}]
        },
        "evaluation_period_param": {
            "test_horizon": 10,
            "periods_between_train_test": 5,
            "cv_min_train_periods": 20,
            "cv_max_splits": 3
        },
        "evaluation_metric_param": {
            "cv_selection_metric": "MeanSquaredError",
            "cv_report_metrics": ["MeanAbsoluteError", "MeanAbsolutePercentError"],
            "relative_error_tolerance": 0.02
        },
        "model_components_param": {
            "autoregression": {
                "autoreg_dict": {"autoreg_param": 0}
            },
            "custom": {
                "custom_param": 1
            },
            "growth": {
                "growth_param": 2
            },
            "events": {
                "events_param": 3
            },
            "hyperparameter_override": [
                {"h1": 4},
                {"h2": 5},
                {}
            ],
            "regressors": {
                "names": ["regressor1", "regressor2"]
            },
            "lagged_regressors": {
                "lagged_regressor_dict": {"lag_reg_param": 0}
            },
            "seasonality": {
                "seas_param": 6
            },
            "uncertainty": {
                "uncertainty_param": 7
            }
        },
        "computation_param": {
            "verbose": 1
        }
    }"""

    forecast_dict = json.loads(json_str)
    config = forecast_config_from_dict(forecast_dict)
    assert_forecast_config_json(config)

    # Tests a string passed to `cv_report_metrics`
    json_str = """{
        "evaluation_metric_param": {
            "cv_report_metrics": "ALL"
        }
    }"""
    forecast_dict = json.loads(json_str)
    config = forecast_config_from_dict(forecast_dict)
    assert config.evaluation_metric_param.cv_report_metrics == "ALL"
    assert config.to_dict()


def assert_forecast_config_json_multiple_model_componments_parameter(config: Optional[ForecastConfig] = None):
    """Asserts the forecast config values. This function expects a particular config and is not generic"""
    config = ForecastConfigDefaults().apply_forecast_config_defaults(config)
    assert config.model_template == [ModelTemplateEnum.SILVERKITE.name,
                                     ModelTemplateEnum.SILVERKITE_DAILY_90.name,
                                     ModelTemplateEnum.SILVERKITE_WEEKLY.name]
    assert config.evaluation_metric_param.relative_error_tolerance == 0.02
    # First model_components_param
    model_components_param_1 = config.model_components_param[0]
    assert model_components_param_1.autoregression is None
    assert model_components_param_1.changepoints is None
    assert model_components_param_1.custom is None
    assert model_components_param_1.growth == {"growth_param": 0}
    assert model_components_param_1.events == {"events_param": 1}
    assert (model_components_param_1.hyperparameter_override is None
            or model_components_param_1.hyperparameter_override is None)
    assert model_components_param_1.regressors == {"names": ["regressor1", "regressor2"]}
    assert model_components_param_1.lagged_regressors is None
    assert model_components_param_1.seasonality == {"seas_param": 2}
    assert model_components_param_1.uncertainty == {"uncertainty_param": 3}
    # Second model_components_param
    model_components_param_2 = config.model_components_param[1]
    assert model_components_param_2.autoregression == {"autoreg_dict": {"autoreg_param": 0}}
    assert model_components_param_2.changepoints is None
    assert model_components_param_2.custom == {"custom_param": 1}
    assert model_components_param_2.growth == {"growth_param": 2}
    assert model_components_param_2.events == {"events_param": 3}
    assert (model_components_param_2.hyperparameter_override == [{"h1": 4}, {"h2": 5}, None]
            or model_components_param_2.hyperparameter_override == [{"h1": 4}, {"h2": 5}, {}])
    assert model_components_param_2.regressors == {"names": ["regressor1", "regressor2"]}
    assert model_components_param_2.lagged_regressors == {"lagged_regressor_dict": {"lag_reg_param": 0}}
    assert model_components_param_2.seasonality == {"seas_param": 6}
    assert model_components_param_2.uncertainty == {"uncertainty_param": 7}
    assert config.to_dict()  # runs without error


def test_forecast_config_json_multiple_model_componments_parameter():
    """Tests ForecastConfig json with a list of model_template and model_components_param parameters"""
    json_str = """{
        "model_template": ["SILVERKITE", "SILVERKITE_DAILY_90", "SILVERKITE_WEEKLY"],
        "metadata_param": {
            "time_col": "custom_time_col",
            "anomaly_info": [{"key": "value"}, {"key2": "value2"}]
        },
        "evaluation_period_param": {
            "test_horizon": 10,
            "periods_between_train_test": 5,
            "cv_min_train_periods": 20,
            "cv_max_splits": 3
        },
        "evaluation_metric_param": {
            "cv_selection_metric": "MeanSquaredError",
            "cv_report_metrics": ["MeanAbsoluteError", "MeanAbsolutePercentError"],
            "relative_error_tolerance": 0.02
        },
        "model_components_param": [{
            "growth": {
                "growth_param": 0
            },
            "events": {
                "events_param": 1
            },
            "regressors": {
                "names": ["regressor1", "regressor2"]
            },
            "seasonality": {
                "seas_param": 2
            },
            "uncertainty": {
                "uncertainty_param": 3
            }
        },
        {
            "autoregression": {
                "autoreg_dict": {"autoreg_param": 0}
            },
            "custom": {
                "custom_param": 1
            },
            "growth": {
                "growth_param": 2
            },
            "events": {
                "events_param": 3
            },
            "hyperparameter_override": [
                {"h1": 4},
                {"h2": 5},
                {}
            ],
            "regressors": {
                "names": ["regressor1", "regressor2"]
            },
            "lagged_regressors": {
                "lagged_regressor_dict": {"lag_reg_param": 0}
            },
            "seasonality": {
                "seas_param": 6
            },
            "uncertainty": {
                "uncertainty_param": 7
            }
        }],
        "computation_param": {
            "verbose": 1
        }
    }"""
    forecast_dict = json.loads(json_str)
    config = forecast_config_from_dict(forecast_dict)
    assert_forecast_config_json_multiple_model_componments_parameter(config)

    # Null values inside `model_template` and `model_components_param`
    json_str = """{
        "model_template": ["SILVERKITE", null, "SILVERKITE_WEEKLY"],
        "model_components_param": [null,
        {
            "autoregression": {
                "autoreg_dict": {"autoreg_param": 0}
            },
            "custom": {
                "custom_param": 1
            },
            "growth": {
                "growth_param": 2
            },
            "events": {
                "events_param": 3
            },
            "hyperparameter_override": [
                {"h1": 4},
                {"h2": 5},
                {}
            ],
            "regressors": {
                "names": ["regressor1", "regressor2"]
            },
            "lagged_regressors": {
                "lagged_regressor_dict": {"lag_reg_param": 0}
            },
            "seasonality": {
                "seas_param": 6
            },
            "uncertainty": {
                "uncertainty_param": 7
            }
        }]
    }"""
    forecast_dict = json.loads(json_str)
    config2 = forecast_config_from_dict(forecast_dict)
    assert config2.model_template == ["SILVERKITE", None, "SILVERKITE_WEEKLY"]
    assert config2.model_components_param[0] is None
    assert config2.model_components_param[1:] == config.model_components_param[1:]

    # Tests a string passed to `cv_report_metrics`
    json_str = """{
        "evaluation_metric_param": {
            "cv_report_metrics": "ALL"
        }
    }"""
    forecast_dict = json.loads(json_str)
    config = forecast_config_from_dict(forecast_dict)
    assert config.evaluation_metric_param.cv_report_metrics == "ALL"
    assert config.to_dict()


def test_forecast_one_by_one():
    # None
    config = ForecastConfig(forecast_one_by_one=None)
    assert config.to_dict()["forecast_one_by_one"] is None
    config = ForecastConfig().from_dict({"forecast_one_by_one": None})
    assert config.forecast_one_by_one is None
    # int
    config = ForecastConfig(forecast_one_by_one=1)
    assert config.to_dict()["forecast_one_by_one"] == 1
    config = ForecastConfig().from_dict({"forecast_one_by_one": 1})
    assert config.forecast_one_by_one == 1
    # bool
    config = ForecastConfig(forecast_one_by_one=True)
    assert config.to_dict()["forecast_one_by_one"] is True
    config = ForecastConfig().from_dict({"forecast_one_by_one": False})
    assert config.forecast_one_by_one is False
    # List of int
    config = ForecastConfig(forecast_one_by_one=[1, 2, 3])
    assert config.to_dict()["forecast_one_by_one"] == [1, 2, 3]
    config = ForecastConfig().from_dict({"forecast_one_by_one": [1, 2, 3]})
    assert config.forecast_one_by_one == [1, 2, 3]
