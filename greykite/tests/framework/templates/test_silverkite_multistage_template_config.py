from greykite.framework.templates.auto_arima_template import AutoArimaTemplate
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.prophet_template import ProphetTemplate
from greykite.framework.templates.silverkite_multistage_template_config import SILVERKITE_MULTISTAGE_EMPTY
from greykite.framework.templates.silverkite_multistage_template_config import SILVERKITE_TWO_STAGE
from greykite.framework.templates.silverkite_multistage_template_config import SilverkiteMultistageModelTemplateEnum
from greykite.framework.templates.silverkite_multistage_template_config import SilverkiteMultistageTemplateConfig
from greykite.framework.templates.silverkite_multistage_template_config import SilverkiteMultistageTemplateConstants
from greykite.framework.templates.silverkite_template import SilverkiteTemplate
from greykite.framework.templates.simple_silverkite_template import SimpleSilverkiteTemplate


def test_silvekite_multistage_template_constants():
    """Tests `silverkite_muiltistage_template_constants`"""
    constants = SilverkiteMultistageTemplateConstants()

    assert constants.SILVERKITE_TWO_STAGE == SILVERKITE_TWO_STAGE
    assert constants.SILVERKITE_MULTISTAGE_EMPTY == SILVERKITE_MULTISTAGE_EMPTY
    assert constants.SilverkiteMultistageModelTemplateEnum == SilverkiteMultistageModelTemplateEnum


def test_silverkite_multistage_template_config():
    """Tests the `SilverkiteMultistageTemplateConfig` data class."""
    assert SilverkiteMultistageTemplateConfig.train_length == f"{7 * 56}D"
    assert SilverkiteMultistageTemplateConfig.fit_length is None
    assert SilverkiteMultistageTemplateConfig.agg_freq is None
    assert SilverkiteMultistageTemplateConfig.agg_func == "nanmean"
    assert SilverkiteMultistageTemplateConfig.model_template == "SILVERKITE"
    assert SilverkiteMultistageTemplateConfig.model_components is None


def test_silverkite_multistage():
    """Tests the SILVERKITE_TWO_STAGE template. To alert any changes to the template."""
    assert len(SILVERKITE_TWO_STAGE.custom["silverkite_multistage_configs"]) == 2

    assert SILVERKITE_TWO_STAGE.custom["silverkite_multistage_configs"][0].train_length == f"{7 * 56}D"
    assert SILVERKITE_TWO_STAGE.custom["silverkite_multistage_configs"][0].fit_length is None
    assert SILVERKITE_TWO_STAGE.custom["silverkite_multistage_configs"][0].agg_func == "nanmean"
    assert SILVERKITE_TWO_STAGE.custom["silverkite_multistage_configs"][0].agg_freq == "D"
    assert SILVERKITE_TWO_STAGE.custom["silverkite_multistage_configs"][0].model_template == "SILVERKITE"
    assert SILVERKITE_TWO_STAGE.custom["silverkite_multistage_configs"][0].model_components == ModelComponentsParam(
        seasonality={
            "yearly_seasonality": 12,
            "quarterly_seasonality": 5,
            "monthly_seasonality": 5,
            "weekly_seasonality": 4,
            "daily_seasonality": 0,
        },
        growth={
            "growth_term": "linear"
        },
        events={
            "holidays_to_model_separately": "auto",
            "holiday_lookup_countries": "auto",
            "holiday_pre_num_days": 1,
            "holiday_post_num_days": 1,
            "holiday_pre_post_num_dict": None,
            "daily_event_df_dict": None,
        },
        changepoints={
            "changepoints_dict": {
                "method": "auto",
                "resample_freq": "D",
                "regularization_strength": 0.5,
                "potential_changepoint_distance": "15D",
                "no_changepoint_distance_from_end": "30D",
                "yearly_seasonality_order": 15,
                "yearly_seasonality_change_freq": "365D"
            },
            "seasonality_changepoints_dict": None
        },
        autoregression={
            "autoreg_dict": "auto"
        },
        regressors={
            "regressor_cols": []
        },
        lagged_regressors={
            "lagged_regressor_dict": None
        },
        uncertainty={
            "uncertainty_dict": None
        },
        custom={
            "fit_algorithm_dict": {
                "fit_algorithm": "ridge",
                "fit_algorithm_params": None,
            },
            "feature_sets_enabled": "auto",  # "auto" based on data freq and size
            "max_daily_seas_interaction_order": 0,
            "max_weekly_seas_interaction_order": 2,
            "extra_pred_cols": [],
            "min_admissible_value": None,
            "max_admissible_value": None,
        }
    )

    assert SILVERKITE_TWO_STAGE.custom["silverkite_multistage_configs"][1].train_length == f"{7 * 4}D"
    assert SILVERKITE_TWO_STAGE.custom["silverkite_multistage_configs"][1].fit_length is None
    assert SILVERKITE_TWO_STAGE.custom["silverkite_multistage_configs"][1].agg_func == "nanmean"
    assert SILVERKITE_TWO_STAGE.custom["silverkite_multistage_configs"][1].agg_freq is None
    assert SILVERKITE_TWO_STAGE.custom["silverkite_multistage_configs"][1].model_template == "SILVERKITE"
    assert SILVERKITE_TWO_STAGE.custom["silverkite_multistage_configs"][1].model_components == ModelComponentsParam(
        seasonality={
            "yearly_seasonality": 0,
            "quarterly_seasonality": 0,
            "monthly_seasonality": 0,
            "weekly_seasonality": 0,
            "daily_seasonality": 12,
        },
        growth={
            "growth_term": None
        },
        events={
            "holidays_to_model_separately": [],
            "holiday_lookup_countries": [],
            "holiday_pre_num_days": 0,
            "holiday_post_num_days": 0,
            "holiday_pre_post_num_dict": None,
            "daily_event_df_dict": None,
        },
        changepoints={
            "changepoints_dict": None,
            "seasonality_changepoints_dict": None
        },
        autoregression={
            "autoreg_dict": "auto"
        },
        regressors={
            "regressor_cols": []
        },
        lagged_regressors={
            "lagged_regressor_dict": None
        },
        uncertainty={
            "uncertainty_dict": None
        },
        custom={
            "fit_algorithm_dict": {
                "fit_algorithm": "ridge",
                "fit_algorithm_params": None,
            },
            "feature_sets_enabled": "auto",  # "auto" based on data freq and size
            "max_daily_seas_interaction_order": 5,
            "max_weekly_seas_interaction_order": 2,
            "extra_pred_cols": [],
            "min_admissible_value": None,
            "max_admissible_value": None,
        }
    )


def test_silverkite_multistage_model_template_enum():
    """Tests the members of `SilverkiteMultistageModelTemplateEnum`."""
    assert SilverkiteMultistageModelTemplateEnum.SILVERKITE.value == SimpleSilverkiteTemplate
    assert SilverkiteMultistageModelTemplateEnum.SILVERKITE_EMPTY.value == SimpleSilverkiteTemplate
    assert SilverkiteMultistageModelTemplateEnum.SILVERKITE_WITH_AR.value == SimpleSilverkiteTemplate
    assert SilverkiteMultistageModelTemplateEnum.PROPHET.value == ProphetTemplate
    assert SilverkiteMultistageModelTemplateEnum.SK.value == SilverkiteTemplate
    assert SilverkiteMultistageModelTemplateEnum.AUTO_ARIMA.value == AutoArimaTemplate
