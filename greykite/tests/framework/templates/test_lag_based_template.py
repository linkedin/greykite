import pandas as pd
import pytest
from testfixtures import LogCapture

from greykite.common.aggregation_function_enum import AggregationFunctionEnum
from greykite.common.constants import LOGGER_NAME
from greykite.common.constants import PREDICTED_COL
from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.lag_based_template import LagBasedTemplate
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.utils.result_summary import summarize_grid_search_results
from greykite.sklearn.estimator.lag_based_estimator import LagBasedEstimator
from greykite.sklearn.estimator.lag_based_estimator import LagUnitEnum


@pytest.fixture
def df_daily():
    df = pd.DataFrame({
        "ts": pd.date_range("2020-01-01", freq="D", periods=1000),
        "y": list(range(1000))
    })
    return df


@pytest.fixture
def df_hourly():
    df = pd.DataFrame({
        "ts": pd.date_range("2020-01-01", freq="H", periods=24 * 365),
        "y": list(range(24 * 365))
    })
    return df


def test_properties():
    """Tests class properties."""
    model = LagBasedTemplate()
    assert model.DEFAULT_MODEL_TEMPLATE == ModelTemplateEnum.LAG_BASED.name
    assert model.allow_model_template_list is False
    assert model.allow_model_components_param_list is False
    assert isinstance(model.estimator, LagBasedEstimator)


def test_regressors():
    """Tests getting regressor and lagged regressor info."""
    model = LagBasedTemplate()
    assert model.get_regressor_cols() is None
    assert model.get_lagged_regressor_info() == {
        "lagged_regressor_cols": None,
        "overall_min_lag_order": None,
        "overall_max_lag_order": None
    }


def test_apply_default_params():
    """Tests applying default parameters."""
    # No model components given.
    model = LagBasedTemplate()
    forecast_config = ForecastConfig()
    model.config = forecast_config
    model_components = ModelComponentsParam()
    model_components = model.apply_lag_based_model_components_defaults(model_components=model_components)
    assert model_components.custom == dict(
        freq=None,
        lag_unit=LagUnitEnum.week.name,
        lags=[1],
        agg_func=AggregationFunctionEnum.mean.name,
        agg_func_params=None,
        past_df=None,
        series_na_fill_func=None
    )
    assert model_components.uncertainty == dict(
        uncertainty_dict=None
    )

    # Model components given.
    model = LagBasedTemplate()
    metadata_param = MetadataParam(
        time_col="ts",
        value_col="y",
        freq="W"
    )
    forecast_config = ForecastConfig(metadata_param=metadata_param)
    model.config = forecast_config
    model_components = ModelComponentsParam(
        seasonality=dict(yearly_seasonality=5),
        custom=dict(
            lag_unit=LagUnitEnum.day.name
        ),
        uncertainty=dict(
            uncertainty_dict=dict(
                uncertainty_method="simple_conditional_residuals"
            )
        )
    )
    model_components = model.apply_lag_based_model_components_defaults(model_components=model_components)
    assert model_components.custom == dict(
        freq="W",  # ``freq`` is fetched from ``metadata_param``.
        lag_unit=LagUnitEnum.day.name,  # ``lag_unit`` is given by user
        lags=[1],
        agg_func=AggregationFunctionEnum.mean.name,
        agg_func_params=None,
        past_df=None,
        series_na_fill_func=None
    )
    assert model_components.uncertainty == dict(
        uncertainty_dict=dict(
            uncertainty_method="simple_conditional_residuals"
        )  # ``uncertainty_dict`` is given by user
    )


def test_get_hyperparameter_grid():
    """Tests getting hyperparameter grid."""
    model = LagBasedTemplate()
    metadata_param = MetadataParam(
        time_col="ts",
        value_col="y",
        freq="H"
    )
    model_components = ModelComponentsParam(
        seasonality=dict(yearly_seasonality=5),
        custom=dict(
            lag_unit=LagUnitEnum.day.name
        ),
        uncertainty=dict(
            uncertainty_dict=dict(
                uncertainty_method="simple_conditional_residuals"
            )
        ),
        hyperparameter_override=dict(
            estimator__lags=[[1], [1, 2, 3]]
        )
    )
    forecast_config = ForecastConfig(
        metadata_param=metadata_param,
        model_components_param=model_components
    )
    model.config = forecast_config
    hyperparameter_grid = model.get_hyperparameter_grid()
    assert hyperparameter_grid == {
        "estimator__freq": ["H"],
        "estimator__lag_unit": [LagUnitEnum.day.name],
        "estimator__lags": [[1], [1, 2, 3]],
        "estimator__agg_func": [AggregationFunctionEnum.mean.name],
        "estimator__agg_func_params": [None],
        "estimator__uncertainty_dict": [
            dict(
                uncertainty_method="simple_conditional_residuals"
            )
        ],
        "estimator__past_df": [None],
        "estimator__series_na_fill_func": [None]
    }


def test_run_template_daily_default(df_daily):
    """Tests running model template with daily data."""
    forecaster = Forecaster()
    metadata_param = MetadataParam(
        time_col="ts",
        value_col="y",
        freq="D"
    )
    evaluation = EvaluationPeriodParam(
        cv_max_splits=0,
        test_horizon=0
    )
    forecast_config = ForecastConfig(
        forecast_horizon=1,
        model_template=ModelTemplateEnum.LAG_BASED.name,
        metadata_param=metadata_param,
        evaluation_period_param=evaluation
    )
    result = forecaster.run_forecast_config(
        df=df_daily,
        config=forecast_config
    )
    # Checks forecast values are correct.
    # Default is week over week.
    assert result.forecast.df_test[PREDICTED_COL].iloc[-1] == df_daily["y"].iloc[-7]


def test_run_template_hourly_default(df_hourly):
    """Tests running model template with hourly data."""
    forecaster = Forecaster()
    metadata_param = MetadataParam(
        time_col="ts",
        value_col="y",
        freq="D"
    )
    evaluation = EvaluationPeriodParam(
        cv_max_splits=0,
        test_horizon=0
    )
    forecast_config = ForecastConfig(
        forecast_horizon=1,
        model_template=ModelTemplateEnum.LAG_BASED.name,
        metadata_param=metadata_param,
        evaluation_period_param=evaluation
    )
    result = forecaster.run_forecast_config(
        df=df_hourly,
        config=forecast_config
    )
    # Checks forecast values are correct.
    # Default is week over week.
    assert result.forecast.df_test[PREDICTED_COL].iloc[-1] == df_hourly["y"].iloc[-7 * 24]


def test_run_template_daily(df_daily):
    """Tests running model template with daily data."""
    forecaster = Forecaster()
    metadata_param = MetadataParam(
        time_col="ts",
        value_col="y",
        freq="D"
    )
    model_components = ModelComponentsParam(
        custom=dict(
            lag_unit=LagUnitEnum.day.name,
            lags=[[1], [1, 2, 3]]
        )
    )
    evaluation = EvaluationPeriodParam(
        cv_max_splits=1,
        test_horizon=0
    )
    forecast_config = ForecastConfig(
        forecast_horizon=1,
        model_template=ModelTemplateEnum.LAG_BASED.name,
        metadata_param=metadata_param,
        model_components_param=model_components,
        evaluation_period_param=evaluation
    )
    result = forecaster.run_forecast_config(
        df=df_daily,
        config=forecast_config
    )
    # 2 different sets of parameters in grid search.
    cv_result = summarize_grid_search_results(result.grid_search)
    assert len(cv_result) == 2
    # Checks forecast values are correct.
    assert result.forecast.df_test[PREDICTED_COL].iloc[-1] == df_daily["y"].iloc[-1]


def test_run_template_hourly_single(df_hourly):
    """Tests running model template with hourly data."""
    forecaster = Forecaster()
    metadata_param = MetadataParam(
        time_col="ts",
        value_col="y",
        freq="H"
    )
    model_components = ModelComponentsParam(
        custom=dict(
            lag_unit=LagUnitEnum.week.name,
            lags=[1, 2, 3]
        )
    )
    evaluation = EvaluationPeriodParam(
        cv_max_splits=1,
        test_horizon=1
    )
    forecast_config = ForecastConfig(
        forecast_horizon=24,
        model_template=ModelTemplateEnum.LAG_BASED.name,
        metadata_param=metadata_param,
        model_components_param=model_components,
        evaluation_period_param=evaluation
    )
    result = forecaster.run_forecast_config(
        df=df_hourly,
        config=forecast_config
    )
    # Checks forecast values are correct.
    assert result.forecast.df_test[PREDICTED_COL].iloc[-24:].reset_index(drop=True).astype(float).equals(
        df_hourly["y"].iloc[-(24 * 7 * 2): -(24 * 7 * 2 - 24)].reset_index(drop=True).astype(float)
    )


def test_run_template_hourly(df_hourly):
    """Tests running model template with hourly data."""
    forecaster = Forecaster()
    metadata_param = MetadataParam(
        time_col="ts",
        value_col="y",
        freq="H"
    )
    model_components = ModelComponentsParam(
        custom=dict(
            lag_unit=LagUnitEnum.week.name,
            lags=[[1], [1, 2, 3]]
        )
    )
    evaluation = EvaluationPeriodParam(
        cv_max_splits=1,
        test_horizon=0
    )
    forecast_config = ForecastConfig(
        forecast_horizon=24,
        model_template=ModelTemplateEnum.LAG_BASED.name,
        metadata_param=metadata_param,
        model_components_param=model_components,
        evaluation_period_param=evaluation
    )
    result = forecaster.run_forecast_config(
        df=df_hourly,
        config=forecast_config
    )
    # 2 different sets of parameters in grid search.
    cv_result = summarize_grid_search_results(result.grid_search)
    assert len(cv_result) == 2
    # Checks forecast values are correct.
    assert result.forecast.df_test[PREDICTED_COL].iloc[-24:].reset_index(drop=True).astype(float).equals(
        df_hourly["y"].iloc[-(24 * 7): -(24 * 7 - 24)].reset_index(drop=True).astype(float)
    )


def test_unknown_keys(df_daily):
    """Tests unknown keys in ``model_components.custom``."""
    with LogCapture(LOGGER_NAME) as log_capture:
        forecaster = Forecaster()
        metadata_param = MetadataParam(
            time_col="ts",
            value_col="y",
            freq="D"
        )
        model_components = ModelComponentsParam(
            custom=dict(
                lag_unit=LagUnitEnum.week.name,
                unknown_key="some_value"
            )
        )
        evaluation = EvaluationPeriodParam(
            cv_max_splits=0,
            test_horizon=0
        )
        forecast_config = ForecastConfig(
            forecast_horizon=24,
            model_template=ModelTemplateEnum.LAG_BASED.name,
            metadata_param=metadata_param,
            model_components_param=model_components,
            evaluation_period_param=evaluation
        )
        forecaster.run_forecast_config(
            df=df_daily,
            config=forecast_config
        )
        log_capture.check_present((
            LOGGER_NAME,
            "WARNING",
            "The following keys are not recognized and ignored for `LagBasedTemplate`: ['unknown_key']"
        ))


def test_monthly_data():
    """Tests monthly data with default template. This should raise an error."""
    df = pd.DataFrame({
        "ts": pd.date_range("2020-01-01", freq="MS", periods=100),
        "y": 1
    })
    with pytest.raises(
            ValueError,
            match="The lag unit 'week' must be at least equal to the data frequency 'MS'."):
        forecaster = Forecaster()
        metadata_param = MetadataParam(
            time_col="ts",
            value_col="y",
            freq="MS"
        )
        evaluation = EvaluationPeriodParam(
            cv_max_splits=0,
            test_horizon=0
        )
        forecast_config = ForecastConfig(
            forecast_horizon=24,
            model_template=ModelTemplateEnum.LAG_BASED.name,
            metadata_param=metadata_param,
            evaluation_period_param=evaluation
        )
        forecaster.run_forecast_config(
            df=df,
            config=forecast_config
        )
