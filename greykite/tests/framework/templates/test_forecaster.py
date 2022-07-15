import sys
import warnings
from enum import Enum

import numpy as np
import pandas as pd
import pytest
from testfixtures import LogCapture

from greykite.common.constants import LOGGER_NAME
from greykite.common.constants import PREDICTED_LOWER_COL
from greykite.common.constants import PREDICTED_UPPER_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.data_loader import DataLoader
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.common.python_utils import assert_equal
from greykite.common.testing_utils import generate_df_for_tests
from greykite.common.testing_utils import generate_df_with_reg_for_tests
from greykite.framework.templates.autogen.forecast_config import ComputationParam
from greykite.framework.templates.autogen.forecast_config import EvaluationMetricParam
from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplate
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.templates.prophet_template import ProphetTemplate
from greykite.framework.templates.silverkite_template import SilverkiteTemplate
from greykite.framework.templates.simple_silverkite_template import SimpleSilverkiteTemplate
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_COMPONENT_KEYWORDS
from greykite.framework.templates.simple_silverkite_template_config import SimpleSilverkiteTemplateOptions
from greykite.framework.utils.framework_testing_utils import assert_basic_pipeline_equal
from greykite.framework.utils.framework_testing_utils import assert_forecast_pipeline_result_equal
from greykite.framework.utils.framework_testing_utils import check_forecast_pipeline_result
from greykite.framework.utils.result_summary import summarize_grid_search_results


try:
    import prophet  # noqa
except ModuleNotFoundError:
    pass


@pytest.fixture
def df_config():
    data = generate_df_with_reg_for_tests(
        freq="W-MON",
        periods=140,
        remove_extra_cols=True,
        mask_test_actuals=True)
    reg_cols = ["regressor1", "regressor2", "regressor_categ"]
    keep_cols = [TIME_COL, VALUE_COL] + reg_cols
    df = data["df"][keep_cols]

    model_template = "SILVERKITE"
    evaluation_metric = EvaluationMetricParam(
        cv_selection_metric=EvaluationMetricEnum.MeanAbsoluteError.name,
        agg_periods=7,
        agg_func=np.max,
        null_model_params={
            "strategy": "quantile",
            "constant": None,
            "quantile": 0.5
        }
    )
    evaluation_period = EvaluationPeriodParam(
        test_horizon=10,
        periods_between_train_test=5,
        cv_horizon=4,
        cv_min_train_periods=80,
        cv_expanding_window=False,
        cv_periods_between_splits=20,
        cv_periods_between_train_test=3,
        cv_max_splits=3
    )
    model_components = ModelComponentsParam(
        regressors={
            "regressor_cols": reg_cols
        },
        custom={
            "fit_algorithm_dict": {
                "fit_algorithm": "ridge",
                "fit_algorithm_params": {"cv": 2}
            }
        }
    )
    computation = ComputationParam(
        verbose=2
    )
    forecast_horizon = 27
    coverage = 0.90
    config = ForecastConfig(
        model_template=model_template,
        computation_param=computation,
        coverage=coverage,
        evaluation_metric_param=evaluation_metric,
        evaluation_period_param=evaluation_period,
        forecast_horizon=forecast_horizon,
        model_components_param=model_components
    )
    return {
        "df": df,
        "config": config,
        "model_template": model_template,
        "reg_cols": reg_cols,
    }


class MySimpleSilverkiteTemplate(SimpleSilverkiteTemplate):
    """Same as `SimpleSilverkiteTemplate`, but with different
    default model template.
    """
    DEFAULT_MODEL_TEMPLATE = "SILVERKITE_WEEKLY"


class MyModelTemplateEnum(Enum):
    """Custom version of TemplateEnum for test cases"""
    MYSILVERKITE = ModelTemplate(
        template_class=MySimpleSilverkiteTemplate,
        description="My own version of Silverkite.")
    SILVERKITE = ModelTemplate(
        template_class=SimpleSilverkiteTemplate,
        description="My own version of Silverkite.")


class MissingSimpleSilverkiteTemplateEnum(Enum):
    """Custom version of TemplateEnum for test cases.
    SimpleSilverkiteTemplate is not included.
    """
    SK = ModelTemplate(
        template_class=SilverkiteTemplate,
        description="Silverkite template.")
    PROPHET = ModelTemplate(
        template_class=ProphetTemplate,
        description="Prophet template.")


def test_init():
    """Tests constructor"""
    forecaster = Forecaster()
    assert forecaster.model_template_enum == ModelTemplateEnum
    assert forecaster.default_model_template_name == "AUTO"
    forecaster = Forecaster(
        model_template_enum=MyModelTemplateEnum,
        default_model_template_name="MYSILVERKITE"
    )
    assert forecaster.model_template_enum == MyModelTemplateEnum
    assert forecaster.default_model_template_name == "MYSILVERKITE"


def test_get_config_with_default_model_template_and_components():
    """Tests `__get_config_with_default_model_template_and_components`"""
    forecaster = Forecaster()
    config = forecaster._Forecaster__get_config_with_default_model_template_and_components()
    assert config == ForecastConfig(
        model_template=ModelTemplateEnum.AUTO.name,
        model_components_param=ModelComponentsParam()
    )

    # Overrides `default_model_template_name`, unnests `model_components_param`.
    forecaster = Forecaster(default_model_template_name="SK")
    config = ForecastConfig(
        model_components_param=[ModelComponentsParam()]
    )
    config = forecaster._Forecaster__get_config_with_default_model_template_and_components(config)
    assert config == ForecastConfig(
        model_template=ModelTemplateEnum.SK.name,
        model_components_param=ModelComponentsParam()
    )

    # Overrides `model_template_enum` and `default_model_template_name`
    forecaster = Forecaster(
        model_template_enum=MyModelTemplateEnum,
        default_model_template_name="MYSILVERKITE"
    )
    config = forecaster._Forecaster__get_config_with_default_model_template_and_components()
    assert config == ForecastConfig(
        model_template=MyModelTemplateEnum.MYSILVERKITE.name,
        model_components_param=ModelComponentsParam()
    )


def test_get_template_class():
    """Tests `__get_template_class`"""
    forecaster = Forecaster()
    assert forecaster._Forecaster__get_template_class() == SimpleSilverkiteTemplate
    assert forecaster._Forecaster__get_template_class(
        config=ForecastConfig(model_template=ModelTemplateEnum.SILVERKITE_WEEKLY.name)) == SimpleSilverkiteTemplate
    if "prophet" in sys.modules:
        assert forecaster._Forecaster__get_template_class(
            config=ForecastConfig(model_template=ModelTemplateEnum.PROPHET.name)) == ProphetTemplate
    assert forecaster._Forecaster__get_template_class(
        config=ForecastConfig(model_template=ModelTemplateEnum.SK.name)) == SilverkiteTemplate

    # list `model_template`
    model_template = [
        ModelTemplateEnum.SILVERKITE.name,
        ModelTemplateEnum.SILVERKITE_DAILY_90.name,
        SimpleSilverkiteTemplateOptions()]
    forecaster = Forecaster()
    assert forecaster._Forecaster__get_template_class(config=ForecastConfig(model_template=model_template)) == SimpleSilverkiteTemplate

    # `model_template` name is wrong
    model_template = "SOME_TEMPLATE"
    with pytest.raises(ValueError, match=f"Model Template '{model_template}' is not recognized! "
                                         f"Must be one of: SILVERKITE, "
                                         f"SILVERKITE_DAILY_1_CONFIG_1, SILVERKITE_DAILY_1_CONFIG_2, SILVERKITE_DAILY_1_CONFIG_3, "
                                         f"SILVERKITE_DAILY_1, SILVERKITE_DAILY_90, "
                                         f"SILVERKITE_WEEKLY, SILVERKITE_MONTHLY, SILVERKITE_HOURLY_1, SILVERKITE_HOURLY_24, "
                                         f"SILVERKITE_HOURLY_168, SILVERKITE_HOURLY_336, SILVERKITE_EMPTY"):
        forecaster = Forecaster()
        forecaster._Forecaster__get_template_class(
            config=ForecastConfig(model_template=model_template))

    # List of `model_template` that include names not compatible with `SimpleSilverkiteTemplate`.
    model_template = [
        ModelTemplateEnum.SK.name,
        ModelTemplateEnum.SILVERKITE.name,
        ModelTemplateEnum.SILVERKITE_DAILY_90.name,
        SimpleSilverkiteTemplateOptions()]
    with pytest.raises(ValueError, match="All model templates must use the same template class"):
        forecaster = Forecaster()
        forecaster._Forecaster__get_template_class(config=ForecastConfig(model_template=model_template))

    # list of `model_template` not supported by template class
    model_template = [ModelTemplateEnum.SK.name, ModelTemplateEnum.SK.name]
    with pytest.raises(ValueError, match="The template class <class "
                                         "'greykite.framework.templates.silverkite_template.SilverkiteTemplate'> "
                                         "does not allow `model_template` to be a list"):
        forecaster = Forecaster()
        forecaster._Forecaster__get_template_class(config=ForecastConfig(model_template=model_template))

    # List of `model_components_param` not compatible with `model_template`.
    model_template = ModelTemplateEnum.SK.name
    config = ForecastConfig(
        model_template=model_template,
        model_components_param=[ModelComponentsParam(), ModelComponentsParam()]
    )
    with pytest.raises(ValueError, match=f"Model template {model_template} does not support a list of `ModelComponentsParam`."):
        forecaster = Forecaster()
        forecaster._Forecaster__get_template_class(config=config)

    # List of a single `model_components_param` is acceptable for a model template
    # that does not accept multiple `model_components_param`.
    forecaster = Forecaster()
    config = ForecastConfig(
        model_template=model_template,
        model_components_param=[ModelComponentsParam()]
    )
    forecaster._Forecaster__get_template_class(config=config)
    # List of multiple `model_components_param` is accepted by SILVERKITE
    config = ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        model_components_param=[ModelComponentsParam(), ModelComponentsParam()]
    )
    forecaster._Forecaster__get_template_class(config=config)

    # Error for unrecognized model template when there is no simple silverkite template
    model_template = "UNKNOWN"
    with pytest.raises(ValueError, match=rf"Model Template '{model_template}' is not recognized! "
                                         rf"Must be one of: SK, PROPHET\."):
        forecaster = Forecaster(
            model_template_enum=MissingSimpleSilverkiteTemplateEnum,
            default_model_template_name="SK",
        )
        forecaster._Forecaster__get_template_class(config=ForecastConfig(model_template=model_template))

    # Custom `model_template_enum`
    forecaster = Forecaster(
        model_template_enum=MyModelTemplateEnum,
        default_model_template_name="MYSILVERKITE",
    )
    assert forecaster._Forecaster__get_template_class() == MySimpleSilverkiteTemplate

    if "prophet" in sys.modules:
        model_template = ModelTemplateEnum.PROPHET.name  # `model_template` name is wrong
        with pytest.raises(ValueError, match=f"Model Template '{model_template}' is not recognized! "
                                             f"Must be one of: MYSILVERKITE, SILVERKITE or satisfy the `SimpleSilverkiteTemplate` rules."):
            forecaster._Forecaster__get_template_class(config=ForecastConfig(model_template=model_template))

    model_template = SimpleSilverkiteTemplateOptions()  # dataclass
    with LogCapture(LOGGER_NAME) as log_capture:
        forecaster._Forecaster__get_template_class(config=ForecastConfig(model_template=model_template))
        log_capture.check(
            (LOGGER_NAME,
             'DEBUG',
             'Model template SimpleSilverkiteTemplateOptions(freq=<SILVERKITE_FREQ.DAILY: '
             "'DAILY'>, seas=<SILVERKITE_SEAS.LT: 'LT'>, gr=<SILVERKITE_GR.LINEAR: "
             "'LINEAR'>, cp=<SILVERKITE_CP.NONE: 'NONE'>, hol=<SILVERKITE_HOL.NONE: "
             "'NONE'>, feaset=<SILVERKITE_FEASET.OFF: 'OFF'>, "
             "algo=<SILVERKITE_ALGO.LINEAR: 'LINEAR'>, ar=<SILVERKITE_AR.OFF: 'OFF'>, "
             "dsi=<SILVERKITE_DSI.AUTO: 'AUTO'>, wsi=<SILVERKITE_WSI.AUTO: 'AUTO'>) is "
             'not found in the template enum. Checking if model template is suitable for '
             '`SimpleSilverkiteTemplate`.'),
            (LOGGER_NAME,
             'DEBUG',
             'Multiple template classes could be used for the model template '
             "SimpleSilverkiteTemplateOptions(freq=<SILVERKITE_FREQ.DAILY: 'DAILY'>, "
             "seas=<SILVERKITE_SEAS.LT: 'LT'>, gr=<SILVERKITE_GR.LINEAR: 'LINEAR'>, "
             "cp=<SILVERKITE_CP.NONE: 'NONE'>, hol=<SILVERKITE_HOL.NONE: 'NONE'>, "
             "feaset=<SILVERKITE_FEASET.OFF: 'OFF'>, algo=<SILVERKITE_ALGO.LINEAR: "
             "'LINEAR'>, ar=<SILVERKITE_AR.OFF: 'OFF'>, dsi=<SILVERKITE_DSI.AUTO: "
             "'AUTO'>, wsi=<SILVERKITE_WSI.AUTO: 'AUTO'>): [<class "
             "'test_forecaster.MySimpleSilverkiteTemplate'>, <class "
             "'greykite.framework.templates.simple_silverkite_template.SimpleSilverkiteTemplate'>]"),
            (LOGGER_NAME,
             'DEBUG',
             "Using template class <class 'test_forecaster.MySimpleSilverkiteTemplate'> "
             'for the model template '
             "SimpleSilverkiteTemplateOptions(freq=<SILVERKITE_FREQ.DAILY: 'DAILY'>, "
             "seas=<SILVERKITE_SEAS.LT: 'LT'>, gr=<SILVERKITE_GR.LINEAR: 'LINEAR'>, "
             "cp=<SILVERKITE_CP.NONE: 'NONE'>, hol=<SILVERKITE_HOL.NONE: 'NONE'>, "
             "feaset=<SILVERKITE_FEASET.OFF: 'OFF'>, algo=<SILVERKITE_ALGO.LINEAR: "
             "'LINEAR'>, ar=<SILVERKITE_AR.OFF: 'OFF'>, dsi=<SILVERKITE_DSI.AUTO: "
             "'AUTO'>, wsi=<SILVERKITE_WSI.AUTO: 'AUTO'>)"))


def test_apply_forecast_config(df_config):
    """Tests `apply_forecast_config`"""
    df = df_config["df"]
    config = df_config["config"]
    model_template = df_config["model_template"]
    reg_cols = df_config["reg_cols"]

    # The same class can be re-used. `df` and `config` are taken from the function call
    #   to `apply_forecast_config`. Only `model_template_enum` and
    #   `default_model_template_name` are persistent in the state.
    forecaster = Forecaster(default_model_template_name=ModelTemplateEnum.SILVERKITE.name)

    # no config
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline_params = forecaster.apply_forecast_config(
            df=df)

        template_class = SimpleSilverkiteTemplate  # based on `default_model_template_name`
        expected_pipeline_params = template_class().apply_template_for_pipeline_params(
            df=df)
        assert_basic_pipeline_equal(pipeline_params.pop("pipeline"), expected_pipeline_params.pop("pipeline"))
        assert_equal(pipeline_params, expected_pipeline_params)
        assert forecaster.config is not None
        assert forecaster.template_class == template_class
        assert isinstance(forecaster.template, forecaster.template_class)
        assert forecaster.pipeline_params is not None

    # custom config
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipeline_params = forecaster.apply_forecast_config(
            df=df,
            config=config)

        template_class = ModelTemplateEnum[model_template].value.template_class  # SimpleSilverkiteTemplate
        expected_pipeline_params = template_class().apply_template_for_pipeline_params(
            df,
            config)
        expected_pipeline = expected_pipeline_params.pop("pipeline")
        assert_basic_pipeline_equal(pipeline_params.pop("pipeline"), expected_pipeline)
        assert_equal(pipeline_params, expected_pipeline_params)

        # Custom `model_template_enum`. Same result, because
        #   `MySimpleSilverkiteTemplate` has the same apply_template_for_pipeline_params
        #   as `SimpleSilverkiteTemplate`.
        forecaster = Forecaster(model_template_enum=MyModelTemplateEnum)
        pipeline_params = forecaster.apply_forecast_config(df=df, config=config)
        assert_basic_pipeline_equal(pipeline_params.pop("pipeline"), expected_pipeline)
        assert_equal(pipeline_params, expected_pipeline_params)

    if "prophet" in sys.modules:
        # `model_component` of config is incompatible with model_template
        forecaster = Forecaster()
        config = ForecastConfig(
            model_template=ModelTemplateEnum.PROPHET.name,
            model_components_param=ModelComponentsParam(
                regressors={
                    "regressor_cols": reg_cols
                }
            )
        )
        with pytest.raises(ValueError) as record:
            forecaster.apply_forecast_config(df=df, config=config)
            assert "Unexpected key(s) found: {\'regressor_cols\'}. The valid keys are: " \
                   "dict_keys([\'add_regressor_dict\'])" in str(record)

        # metadata of config is incompatible with df
        df = df.rename(columns={TIME_COL: "some_time_col", VALUE_COL: "some_value_col"})
        with pytest.raises(ValueError, match="ts column is not in input data"):
            forecaster.apply_forecast_config(df=df, config=config)


def test_run_forecast_config():
    """Tests `run_forecast_config`"""
    data = generate_df_for_tests(freq="H", periods=14*24)
    df = data["df"]

    # Checks if exception is raised
    with pytest.raises(ValueError, match="is not recognized"):
        forecaster = Forecaster()
        forecaster.run_forecast_config(df=df, config=ForecastConfig(model_template="unknown_template"))
    with pytest.raises(ValueError, match="is not recognized"):
        forecaster = Forecaster()
        forecaster.run_forecast_json(df=df, json_str="""{ "model_template": "unknown_template" }""")

    # All run_forecast_config* functions return the same result for the default config,
    # call forecast_pipeline, and return a result with the proper format.
    np.random.seed(123)
    forecaster = Forecaster()
    default_result = forecaster.run_forecast_config(df=df)
    score_func = EvaluationMetricEnum.MeanAbsolutePercentError.name
    check_forecast_pipeline_result(
        default_result,
        coverage=None,
        strategy=None,
        score_func=score_func,
        greater_is_better=False)
    assert_equal(forecaster.forecast_result, default_result)

    np.random.seed(123)
    forecaster = Forecaster()
    json_result = forecaster.run_forecast_json(df=df)
    check_forecast_pipeline_result(
        json_result,
        coverage=None,
        strategy=None,
        score_func=score_func,
        greater_is_better=False)
    assert_forecast_pipeline_result_equal(json_result, default_result, rel=0.02)


def test_run_forecast_config_custom():
    """Tests `run_forecast_config` on weekly data with custom config:

     - numeric and categorical regressors
     - coverage
     - null model
    """
    data = generate_df_with_reg_for_tests(
        freq="W-MON",
        periods=140,
        remove_extra_cols=True,
        mask_test_actuals=True)
    reg_cols = ["regressor1", "regressor2", "regressor_categ"]
    keep_cols = [TIME_COL, VALUE_COL] + reg_cols
    df = data["df"][keep_cols]

    metric = EvaluationMetricEnum.MeanAbsoluteError
    evaluation_metric = EvaluationMetricParam(
        cv_selection_metric=metric.name,
        agg_periods=7,
        agg_func=np.max,
        null_model_params={
            "strategy": "quantile",
            "constant": None,
            "quantile": 0.5
        }
    )

    evaluation_period = EvaluationPeriodParam(
        test_horizon=10,
        periods_between_train_test=5,
        cv_horizon=4,
        cv_min_train_periods=80,
        cv_expanding_window=False,
        cv_periods_between_splits=20,
        cv_periods_between_train_test=3,
        cv_max_splits=3
    )

    model_components = ModelComponentsParam(
        regressors={
            "regressor_cols": reg_cols
        },
        custom={
            "fit_algorithm_dict": {
                "fit_algorithm": "ridge",
                "fit_algorithm_params": {"cv": 2}
            }
        }
    )
    computation = ComputationParam(
        verbose=2
    )
    forecast_horizon = 27
    coverage = 0.90

    forecast_config = ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        computation_param=computation,
        coverage=coverage,
        evaluation_metric_param=evaluation_metric,
        evaluation_period_param=evaluation_period,
        forecast_horizon=forecast_horizon,
        model_components_param=model_components
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecaster = Forecaster()
        result = forecaster.run_forecast_config(
            df=df,
            config=forecast_config)

        mse = EvaluationMetricEnum.RootMeanSquaredError.get_metric_name()
        q80 = EvaluationMetricEnum.Quantile80.get_metric_name()
        assert result.backtest.test_evaluation[mse] == pytest.approx(3.299, rel=1e-2)
        assert result.backtest.test_evaluation[q80] == pytest.approx(1.236, rel=1e-2)
        assert result.forecast.train_evaluation[mse] == pytest.approx(1.782, rel=1e-2)
        assert result.forecast.train_evaluation[q80] == pytest.approx(0.746, rel=1e-2)
        check_forecast_pipeline_result(
            result,
            coverage=coverage,
            strategy=None,
            score_func=metric.name,
            greater_is_better=False)

    # Note that for newer scikit-learn version, needs to add a check for ValueError, matching "model is misconfigured"
    with pytest.raises((ValueError, KeyError)) as exception_info:
        model_components = ModelComponentsParam(
            regressors={
                "regressor_cols": ["missing_regressor"]
            }
        )
        forecaster = Forecaster()
        result = forecaster.run_forecast_config(
            df=df,
            config=ForecastConfig(
                model_template=ModelTemplateEnum.SILVERKITE.name,
                model_components_param=model_components
            )
        )
        check_forecast_pipeline_result(
            result,
            coverage=None,
            strategy=None,
            score_func=metric.get_metric_func(),
            greater_is_better=False)
    info_str = str(exception_info.value)
    assert "missing_regressor" in info_str or "model is misconfigured" in info_str


def test_run_forecast_json():
    """Tests:
     - no coverage
     - hourly data (2+ years)
     - default `hyperparameter_grid` (all interaction terms enabled)
    """
    # sets random state for consistent comparison
    data = generate_df_for_tests(
        freq="H",
        periods=700*24)
    df = data["train_df"]

    json_str = """{
        "model_template": "SILVERKITE",
        "forecast_horizon": 3359,
        "model_components_param": {
            "custom": {
                "fit_algorithm_dict": {
                    "fit_algorithm": "linear"
                }
            }
        }
    }"""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        forecaster = Forecaster()
        result = forecaster.run_forecast_json(
            df=df,
            json_str=json_str)

        mse = EvaluationMetricEnum.RootMeanSquaredError.get_metric_name()
        q80 = EvaluationMetricEnum.Quantile80.get_metric_name()
        assert result.backtest.test_evaluation[mse] == pytest.approx(2.120, rel=0.03)
        assert result.backtest.test_evaluation[q80] == pytest.approx(0.863, rel=0.02)
        assert result.forecast.train_evaluation[mse] == pytest.approx(1.975, rel=0.02)
        assert result.forecast.train_evaluation[q80] == pytest.approx(0.786, rel=1e-2)
        check_forecast_pipeline_result(
            result,
            coverage=None,
            strategy=None,
            score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name,
            greater_is_better=False)


def test_run_forecast_config_with_single_simple_silverkite_template():
    # The generic name of single simple silverkite templates are not added to `ModelTemplateEnum`,
    # therefore we test if these are recognized.
    data = generate_df_for_tests(freq="D", periods=365)
    df = data["df"]
    metric = EvaluationMetricEnum.MeanAbsoluteError
    evaluation_metric = EvaluationMetricParam(
        cv_selection_metric=metric.name,
        agg_periods=7,
        agg_func=np.max,
        null_model_params={
            "strategy": "quantile",
            "constant": None,
            "quantile": 0.5
        }
    )

    evaluation_period = EvaluationPeriodParam(
        test_horizon=10,
        periods_between_train_test=5,
        cv_horizon=4,
        cv_min_train_periods=80,
        cv_expanding_window=False,
        cv_periods_between_splits=20,
        cv_periods_between_train_test=3,
        cv_max_splits=2
    )

    model_components = ModelComponentsParam(
        hyperparameter_override=[
            {"estimator__yearly_seasonality": 1},
            {"estimator__yearly_seasonality": 2}
        ]
    )
    computation = ComputationParam(
        verbose=2
    )
    forecast_horizon = 27
    coverage = 0.90

    single_template_class = SimpleSilverkiteTemplateOptions(
        freq=SILVERKITE_COMPONENT_KEYWORDS.FREQ.value.DAILY,
        seas=SILVERKITE_COMPONENT_KEYWORDS.SEAS.value.NONE
    )

    forecast_config = ForecastConfig(
        model_template=[single_template_class, "DAILY_ALGO_SGD", "SILVERKITE_DAILY_90"],
        computation_param=computation,
        coverage=coverage,
        evaluation_metric_param=evaluation_metric,
        evaluation_period_param=evaluation_period,
        forecast_horizon=forecast_horizon,
        model_components_param=model_components
    )

    forecaster = Forecaster()
    result = forecaster.run_forecast_config(
        df=df,
        config=forecast_config)

    summary = summarize_grid_search_results(result.grid_search)
    # single_template_class is 1 template,
    # "DAILY_ALGO_SGD" is 1 template and "SILVERKITE_DAILY_90" has 4 templates.
    # With 2 items in `hyperparameter_override, there should be a total of 12 cases.
    assert summary.shape[0] == 12

    # Tests functionality for single template class only.
    forecast_config = ForecastConfig(
        model_template=single_template_class,
        computation_param=computation,
        coverage=coverage,
        evaluation_metric_param=evaluation_metric,
        evaluation_period_param=evaluation_period,
        forecast_horizon=forecast_horizon
    )

    forecaster = Forecaster()
    pipeline_parameters = forecaster.apply_forecast_config(
        df=df,
        config=forecast_config
    )
    assert_equal(
        actual=pipeline_parameters["hyperparameter_grid"],
        expected={
            "estimator__time_properties": [None],
            "estimator__origin_for_time_vars": [None],
            "estimator__train_test_thresh": [None],
            "estimator__training_fraction": [None],
            "estimator__fit_algorithm_dict": [{"fit_algorithm": "linear", "fit_algorithm_params": None}],
            "estimator__auto_holiday": [False],
            "estimator__holidays_to_model_separately": [[]],
            "estimator__holiday_lookup_countries": [[]],
            "estimator__holiday_pre_num_days": [0],
            "estimator__holiday_post_num_days": [0],
            "estimator__holiday_pre_post_num_dict": [None],
            "estimator__daily_event_df_dict": [None],
            "estimator__auto_growth": [False],
            "estimator__changepoints_dict": [None],
            "estimator__seasonality_changepoints_dict": [None],
            "estimator__auto_seasonality": [False],
            "estimator__yearly_seasonality": [0],
            "estimator__quarterly_seasonality": [0],
            "estimator__monthly_seasonality": [0],
            "estimator__weekly_seasonality": [0],
            "estimator__daily_seasonality": [0],
            "estimator__max_daily_seas_interaction_order": [0],
            "estimator__max_weekly_seas_interaction_order": [2],
            "estimator__autoreg_dict": [None],
            "estimator__simulation_num": [10],
            "estimator__fast_simulation": [False],
            "estimator__lagged_regressor_dict": [None],
            "estimator__min_admissible_value": [None],
            "estimator__max_admissible_value": [None],
            "estimator__normalize_method": ["zero_to_one"],
            "estimator__uncertainty_dict": [None],
            "estimator__growth_term": ["linear"],
            "estimator__regressor_cols": [[]],
            "estimator__feature_sets_enabled": [False],
            "estimator__extra_pred_cols": [[]],
            "estimator__drop_pred_cols": [None],
            "estimator__explicit_pred_cols": [None],
            "estimator__regression_weight_col": [None],
        },
        ignore_keys={"estimator__time_properties": None}
    )


def test_estimator_plot_components_from_forecaster():
    """Tests estimator's plot_components function after the Forecaster has set everything up at the top most level"""
    # Test with real data (Female-births) via model template
    dl = DataLoader()
    data_path = dl.get_data_home(data_sub_dir="daily")
    df = dl.get_df(data_path=data_path, data_name="daily_female_births")
    metadata = MetadataParam(time_col="Date", value_col="Births", freq="D")
    model_components = ModelComponentsParam(
        seasonality={
            "yearly_seasonality": True,
            "quarterly_seasonality": True,
            "weekly_seasonality": True,
            "daily_seasonality": False
        }
    )
    result = Forecaster().run_forecast_config(
        df=df,
        config=ForecastConfig(
            model_template=ModelTemplateEnum.SILVERKITE.name,
            forecast_horizon=30,  # forecast 1 month
            coverage=0.95,  # 95% prediction intervals
            metadata_param=metadata,
            model_components_param=model_components
        )
    )
    estimator = result.model.steps[-1][-1]
    assert estimator.plot_components()


def test_estimator_get_coef_summary_from_forecaster():
    """Tests model summary for silverkite model with missing values in value_col after everything is setup by Forecaster"""
    dl = DataLoader()
    df_pt = dl.load_peyton_manning()
    config = ForecastConfig().from_dict(dict(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        forecast_horizon=10,
        metadata_param=dict(
            time_col="ts",
            value_col="y",
            freq="D"
        ),
        model_components_param=dict(
            custom={
                "fit_algorithm_dict": {"fit_algorithm": "linear"}
            }
        )
    ))
    result = Forecaster().run_forecast_config(
        df=df_pt[:365],  # shortens df to speed up
        config=config
    )
    summary = result.model[-1].summary()
    x = summary.get_coef_summary(
        is_intercept=True,
        return_df=True)
    assert x.shape[0] == 1
    summary.get_coef_summary(is_time_feature=True)
    summary.get_coef_summary(is_event=True)
    summary.get_coef_summary(is_trend=True)
    summary.get_coef_summary(is_interaction=True)
    x = summary.get_coef_summary(is_lag=True)
    assert x is None
    x = summary.get_coef_summary(
        is_trend=True,
        is_seasonality=False,
        is_interaction=False,
        return_df=True)
    assert all([":" not in col for col in x["Pred_col"].tolist()])
    assert "ct1" in x["Pred_col"].tolist()
    assert "sin1_ct1_yearly" not in x["Pred_col"].tolist()
    x = summary.get_coef_summary(return_df=True)
    assert x.shape[0] == summary.info_dict["coef_summary_df"].shape[0]


def test_auto_model_template():
    df = pd.DataFrame({
        "ts": pd.date_range("2020-01-01", freq="D", periods=100),
        "y": range(100)
    })
    config = ForecastConfig(
        model_template=ModelTemplateEnum.AUTO.name,
        forecast_horizon=1,
        metadata_param=MetadataParam(
            time_col="ts",
            value_col="y",
            freq="D"
        ),
        evaluation_period_param=EvaluationPeriodParam(
            cv_max_splits=1,
            test_horizon=0
        )
    )
    forecaster = Forecaster()
    forecaster.apply_forecast_config(
        df=df,
        config=config
    )
    assert forecaster.config.model_template == ModelTemplateEnum.SILVERKITE_DAILY_1_CONFIG_1.name

    # Not able to infer frequency, so the default is SILVERKITE
    df = df.drop([1])  # drops the second row
    config.metadata_param.freq = None
    forecaster = Forecaster()
    assert forecaster._Forecaster__get_model_template(df, config) == ModelTemplateEnum.SILVERKITE.name
    forecaster.apply_forecast_config(
        df=df,
        config=config
    )
    assert forecaster.config.model_template == ModelTemplateEnum.SILVERKITE.name


def test_quantile_regression_uncertainty_model():
    """Tests the quantile regression based uncertainty model."""
    df = DataLoader().load_peyton_manning().iloc[-365:].reset_index(drop=True)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    config = ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        forecast_horizon=7,
        coverage=0.95,
        metadata_param=MetadataParam(
            time_col="ts",
            value_col="y",
            freq="D"
        ),
        evaluation_period_param=EvaluationPeriodParam(
            cv_max_splits=1,
            test_horizon=1
        ),
        model_components_param=ModelComponentsParam(
            uncertainty=dict(
                uncertainty_dict=dict(
                    uncertainty_method="quantile_regression",
                    params=dict(
                        is_residual_based=False
                    )
                )
            )
        )
    )
    forecaster = Forecaster()
    forecast_result = forecaster.run_forecast_config(
        df=df,
        config=config
    )
    assert PREDICTED_LOWER_COL in forecast_result.forecast.df_test.columns
    assert PREDICTED_UPPER_COL in forecast_result.forecast.df_test.columns
    assert forecast_result.forecast.df_test[PREDICTED_LOWER_COL].isna().sum() == 0
    assert forecast_result.forecast.df_test[PREDICTED_UPPER_COL].isna().sum() == 0
    assert all(
        forecast_result.forecast.df_test[PREDICTED_LOWER_COL] <= forecast_result.forecast.df_test[PREDICTED_UPPER_COL])
