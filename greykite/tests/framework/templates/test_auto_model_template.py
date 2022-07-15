import pandas as pd
import pytest
from testfixtures import LogCapture

from greykite.common.logging import LOGGER_NAME
from greykite.framework.templates.auto_model_template import get_auto_silverkite_model_template
from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_DAILY_90
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_HOURLY_1
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_HOURLY_24
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_HOURLY_168
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_HOURLY_336
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_WEEKLY


DEFAULT_MODEL_TEMPLATE = ModelTemplateEnum.SILVERKITE.name


@pytest.fixture
def hourly_data():
    df = pd.DataFrame({
        "ts": pd.date_range("2020-01-01", periods=3000, freq="H"),
        "y": range(3000)
    })
    return df


@pytest.fixture
def daily_data():
    df = pd.DataFrame({
        "ts": pd.date_range("2020-01-01", periods=1000, freq="D"),
        "y": range(1000)
    })
    return df


@pytest.fixture
def weekly_data():
    df = pd.DataFrame({
        "ts": pd.date_range("2020-01-01", periods=100, freq="W"),
        "y": range(100)
    })
    return df


@pytest.fixture
def monthly_data():
    df = pd.DataFrame({
        "ts": pd.date_range("2020-01-01", periods=100, freq="MS"),
        "y": range(100)
    })
    return df


def test_exception(hourly_data):
    """Exception."""
    with pytest.raises(ValueError, match="The `default_model_template_name` in "
                                         "`get_auto_silverkite_model_template` cannot be 'AUTO'."):
        get_auto_silverkite_model_template(
            df=hourly_data,
            default_model_template_name=ModelTemplateEnum.AUTO.name,
            config=ForecastConfig(
                forecast_horizon=10
            )
        )


@pytest.mark.parametrize("forecast_horizon,cv_max_splits,expected_template", [
    (1, 1, SILVERKITE_HOURLY_1[0]),
    (24, 1, SILVERKITE_HOURLY_24[0]),
    (168, 1, SILVERKITE_HOURLY_168[0]),
    (336, 1, SILVERKITE_HOURLY_336[0]),
    (1, 30, ModelTemplateEnum.SILVERKITE_HOURLY_1.name),
    (24, 5, ModelTemplateEnum.SILVERKITE_HOURLY_24.name),
    (168, 5, ModelTemplateEnum.SILVERKITE_HOURLY_168.name),
    (336, 5, ModelTemplateEnum.SILVERKITE_HOURLY_336.name)])
def test_get_model_template_hourly(forecast_horizon, cv_max_splits, expected_template, hourly_data):
    """Hourly model template."""
    df = hourly_data
    config = ForecastConfig(
        forecast_horizon=forecast_horizon,
        evaluation_period_param=EvaluationPeriodParam(
            cv_max_splits=cv_max_splits
        )
    )
    with LogCapture(LOGGER_NAME) as log_capture:
        model_template = get_auto_silverkite_model_template(
            df=df,
            default_model_template_name=DEFAULT_MODEL_TEMPLATE,
            config=config
        )
        assert model_template == expected_template
        log_capture.check_present((
            LOGGER_NAME,
            "INFO",
            f"Model template was set to 'auto'. "
            f"Automatically found most appropriate model template '{model_template}'."
        ))


@pytest.mark.parametrize("forecast_horizon,cv_max_splits,expected_template", [
    (3, 1, ModelTemplateEnum.SILVERKITE_DAILY_1_CONFIG_1.name),
    (90, 1, SILVERKITE_DAILY_90[0]),
    (3, 10, ModelTemplateEnum.SILVERKITE_DAILY_1.name),
    (90, 5, ModelTemplateEnum.SILVERKITE_DAILY_90.name),
    (30, 1, ModelTemplateEnum.SILVERKITE.name)])
def test_get_model_template_daily(forecast_horizon, cv_max_splits, expected_template, daily_data):
    """Daily model template."""
    df = daily_data
    config = ForecastConfig(
        forecast_horizon=forecast_horizon,
        evaluation_period_param=EvaluationPeriodParam(
            cv_max_splits=cv_max_splits
        )
    )
    with LogCapture(LOGGER_NAME) as log_capture:
        model_template = get_auto_silverkite_model_template(
            df=df,
            default_model_template_name=DEFAULT_MODEL_TEMPLATE,
            config=config
        )
        assert model_template == expected_template
        log_capture.check_present((
            LOGGER_NAME,
            "INFO",
            f"Model template was set to 'auto'. "
            f"Automatically found most appropriate model template '{model_template}'."
        ))


@pytest.mark.parametrize("forecast_horizon,cv_max_splits,expected_template", [
    (3, 1, SILVERKITE_WEEKLY[0]),
    (3, 10, ModelTemplateEnum.SILVERKITE_WEEKLY.name)])
def test_get_model_template_weekly(forecast_horizon, cv_max_splits, expected_template, weekly_data):
    """Weekly model template."""
    df = weekly_data
    config = ForecastConfig(
        forecast_horizon=forecast_horizon,
        evaluation_period_param=EvaluationPeriodParam(
            cv_max_splits=cv_max_splits
        )
    )
    with LogCapture(LOGGER_NAME) as log_capture:
        model_template = get_auto_silverkite_model_template(
            df=df,
            default_model_template_name=DEFAULT_MODEL_TEMPLATE,
            config=config
        )
        assert model_template == expected_template
        log_capture.check_present((
            LOGGER_NAME,
            "INFO",
            f"Model template was set to 'auto'. "
            f"Automatically found most appropriate model template '{model_template}'."
        ))


def test_get_model_template_monthly(monthly_data):
    """Monthly model template."""
    df = monthly_data
    config = ForecastConfig(
        forecast_horizon=1,
        evaluation_period_param=EvaluationPeriodParam(
            cv_max_splits=1
        )
    )
    with LogCapture(LOGGER_NAME) as log_capture:
        model_template = get_auto_silverkite_model_template(
            df=df,
            default_model_template_name=DEFAULT_MODEL_TEMPLATE,
            config=config
        )
        assert model_template == ModelTemplateEnum.SILVERKITE_MONTHLY.name
        log_capture.check_present((
            LOGGER_NAME,
            "INFO",
            f"Model template was set to 'auto'. "
            f"Automatically found most appropriate model template '{model_template}'."
        ))


def test_no_frequency(daily_data):
    """Frequency not given and not inferrable, using default template."""
    df = daily_data.iloc[[1, 2, 3, 5]].reset_index(drop=True)
    config = ForecastConfig(
        forecast_horizon=1,
        evaluation_period_param=EvaluationPeriodParam(
            cv_max_splits=1
        )
    )
    with LogCapture(LOGGER_NAME) as log_capture:
        model_template = get_auto_silverkite_model_template(
            df=df,
            default_model_template_name=DEFAULT_MODEL_TEMPLATE,
            config=config
        )
        assert model_template == ModelTemplateEnum.SILVERKITE.name
        log_capture.check_present((
            LOGGER_NAME,
            "INFO",
            f"Model template was set to 'auto', however, the data frequency "
            f"is not given and can not be inferred. "
            f"Using default model template '{model_template}'."
        ))


def test_no_horizon(daily_data):
    """Forecast horizon not given, using default template."""
    df = daily_data
    config = ForecastConfig(
        evaluation_period_param=EvaluationPeriodParam(
            cv_max_splits=1
        )
    )
    # For daily data, when forecast horizon is not given, the default is 30 days.
    # The corresponding model template is "SILVERKITE".
    model_template = get_auto_silverkite_model_template(
        df=df,
        default_model_template_name=DEFAULT_MODEL_TEMPLATE,
        config=config
    )
    assert model_template == ModelTemplateEnum.SILVERKITE.name


def test_cv_not_enough(daily_data):
    """``cv_max_split`` is 5 but data length is too short to have 5 splits.
    Using single model template.
    """
    df = daily_data
    config = ForecastConfig(
        forecast_horizon=1,
        evaluation_period_param=EvaluationPeriodParam(
            cv_max_splits=5,
            test_horizon=len(df) - 5
        )
    )
    model_template = get_auto_silverkite_model_template(
        df=df,
        default_model_template_name=DEFAULT_MODEL_TEMPLATE,
        config=config
    )
    assert model_template == ModelTemplateEnum.SILVERKITE_DAILY_1_CONFIG_1.name
