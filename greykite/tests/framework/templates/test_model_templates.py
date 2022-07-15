from greykite.framework.templates.auto_arima_template import AutoArimaTemplate
from greykite.framework.templates.lag_based_template import LagBasedTemplate
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.templates.multistage_forecast_template import MultistageForecastTemplate
from greykite.framework.templates.prophet_template import ProphetTemplate
from greykite.framework.templates.silverkite_template import SilverkiteTemplate
from greykite.framework.templates.simple_silverkite_template import SimpleSilverkiteTemplate


def test_model_template_enum():
    """Tests ModelTemplateEnum accessors"""
    assert ModelTemplateEnum.SILVERKITE.value.template_class == SimpleSilverkiteTemplate
    assert "Silverkite model with automatic growth, seasonality, holidays," in ModelTemplateEnum.SILVERKITE.value.description

    assert ModelTemplateEnum.SILVERKITE_DAILY_1_CONFIG_1.value.template_class == SimpleSilverkiteTemplate
    assert "Config 1 in template ``SILVERKITE_DAILY_1``." in ModelTemplateEnum.SILVERKITE_DAILY_1_CONFIG_1.value.description

    assert ModelTemplateEnum.SILVERKITE_DAILY_1_CONFIG_2.value.template_class == SimpleSilverkiteTemplate
    assert "Config 2 in template ``SILVERKITE_DAILY_1``." in ModelTemplateEnum.SILVERKITE_DAILY_1_CONFIG_2.value.description

    assert ModelTemplateEnum.SILVERKITE_DAILY_1_CONFIG_3.value.template_class == SimpleSilverkiteTemplate
    assert "Config 3 in template ``SILVERKITE_DAILY_1``." in ModelTemplateEnum.SILVERKITE_DAILY_1_CONFIG_3.value.description

    assert ModelTemplateEnum.SILVERKITE_DAILY_1.value.template_class == SimpleSilverkiteTemplate
    assert "Silverkite model specifically tuned for daily data and 1-day forecast" in ModelTemplateEnum.SILVERKITE_DAILY_1.value.description

    assert ModelTemplateEnum.SILVERKITE_DAILY_90.value.template_class == SimpleSilverkiteTemplate
    assert "Silverkite model specifically tuned for daily" in ModelTemplateEnum.SILVERKITE_DAILY_90.value.description

    assert ModelTemplateEnum.SILVERKITE_WEEKLY.value.template_class == SimpleSilverkiteTemplate
    assert "Silverkite model specifically tuned for weekly" in ModelTemplateEnum.SILVERKITE_WEEKLY.value.description

    assert ModelTemplateEnum.SILVERKITE_MONTHLY.value.template_class == SimpleSilverkiteTemplate
    assert "Silverkite model specifically tuned for monthly" in ModelTemplateEnum.SILVERKITE_MONTHLY.value.description

    assert ModelTemplateEnum.SILVERKITE_HOURLY_1.value.template_class == SimpleSilverkiteTemplate
    assert "Silverkite model specifically tuned for hourly data with 1 hour" in ModelTemplateEnum.SILVERKITE_HOURLY_1.value.description

    assert ModelTemplateEnum.SILVERKITE_HOURLY_24.value.template_class == SimpleSilverkiteTemplate
    assert "Silverkite model specifically tuned for hourly data with 24 hours" in ModelTemplateEnum.SILVERKITE_HOURLY_24.value.description

    assert ModelTemplateEnum.SILVERKITE_HOURLY_168.value.template_class == SimpleSilverkiteTemplate
    assert "Silverkite model specifically tuned for hourly data with 168 hours" in ModelTemplateEnum.SILVERKITE_HOURLY_168.value.description

    assert ModelTemplateEnum.SILVERKITE_HOURLY_336.value.template_class == SimpleSilverkiteTemplate
    assert "Silverkite model specifically tuned for hourly data with 336 hours" in ModelTemplateEnum.SILVERKITE_HOURLY_336.value.description

    assert ModelTemplateEnum.SILVERKITE_EMPTY.value.template_class == SimpleSilverkiteTemplate
    assert "Silverkite model with no component included by default" in ModelTemplateEnum.SILVERKITE_EMPTY.value.description

    assert ModelTemplateEnum.SK.value.template_class == SilverkiteTemplate
    assert "Silverkite model with low-level interface" in ModelTemplateEnum.SK.value.description

    assert ModelTemplateEnum.PROPHET.value.template_class == ProphetTemplate
    assert "Prophet model" in ModelTemplateEnum.PROPHET.value.description

    assert ModelTemplateEnum.AUTO_ARIMA.value.template_class == AutoArimaTemplate
    assert "Auto ARIMA model" in ModelTemplateEnum.AUTO_ARIMA.value.description

    assert ModelTemplateEnum.SILVERKITE_TWO_STAGE.value.template_class == MultistageForecastTemplate
    assert "MultistageForecastTemplate's default model template." in ModelTemplateEnum.SILVERKITE_TWO_STAGE.value.description

    assert ModelTemplateEnum.MULTISTAGE_EMPTY.value.template_class == MultistageForecastTemplate
    assert "Empty configuration for Multistage Forecast." in ModelTemplateEnum.MULTISTAGE_EMPTY.value.description

    assert ModelTemplateEnum.AUTO.value.template_class == SimpleSilverkiteTemplate
    assert "Automatically selects the SimpleSilverkite model template" in ModelTemplateEnum.AUTO.value.description

    assert ModelTemplateEnum.LAG_BASED.value.template_class == LagBasedTemplate
    assert "Uses aggregated past observations as predictions. " in ModelTemplateEnum.LAG_BASED.value.description

    assert ModelTemplateEnum.SILVERKITE_WOW.value.template_class == MultistageForecastTemplate
    assert "The Silverkite+WOW model uses Silverkite to model" in ModelTemplateEnum.SILVERKITE_WOW.value.description
