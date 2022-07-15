from greykite.framework.templates.auto_arima_template import AutoArimaTemplate
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.multistage_forecast_template_config import MULTISTAGE_EMPTY
from greykite.framework.templates.multistage_forecast_template_config import SILVERKITE_TWO_STAGE
from greykite.framework.templates.multistage_forecast_template_config import SILVERKITE_WOW
from greykite.framework.templates.multistage_forecast_template_config import MultistageForecastModelTemplateEnum
from greykite.framework.templates.multistage_forecast_template_config import MultistageForecastTemplateConfig
from greykite.framework.templates.multistage_forecast_template_config import MultistageForecastTemplateConstants
from greykite.framework.templates.prophet_template import ProphetTemplate
from greykite.framework.templates.silverkite_template import SilverkiteTemplate
from greykite.framework.templates.simple_silverkite_template import SimpleSilverkiteTemplate


def test_multistage_forecast_template_constants():
    """Tests `muiltistage_forecast_template_constants`"""
    constants = MultistageForecastTemplateConstants()

    assert constants.SILVERKITE_TWO_STAGE == SILVERKITE_TWO_STAGE
    assert constants.MULTISTAGE_EMPTY == MULTISTAGE_EMPTY
    assert constants.MultistageForecastModelTemplateEnum == MultistageForecastModelTemplateEnum
    assert constants.SILVERKITE_WOW == SILVERKITE_WOW


def test_multistage_forecast_template_config():
    """Tests the `MultistageForecastTemplateConfig` data class."""
    assert MultistageForecastTemplateConfig.train_length == f"{7 * 56}D"
    assert MultistageForecastTemplateConfig.fit_length is None
    assert MultistageForecastTemplateConfig.agg_freq is None
    assert MultistageForecastTemplateConfig.agg_func == "nanmean"
    assert MultistageForecastTemplateConfig.model_template == "SILVERKITE"
    assert MultistageForecastTemplateConfig.model_components is None


def test_multistage_forecast():
    """Tests the SILVERKITE_TWO_STAGE template. To alert any changes to the template."""
    assert len(SILVERKITE_TWO_STAGE.custom["multistage_forecast_configs"]) == 2

    assert SILVERKITE_TWO_STAGE.custom["multistage_forecast_configs"][0].train_length == f"{7 * 56}D"
    assert SILVERKITE_TWO_STAGE.custom["multistage_forecast_configs"][0].fit_length is None
    assert SILVERKITE_TWO_STAGE.custom["multistage_forecast_configs"][0].agg_func == "nanmean"
    assert SILVERKITE_TWO_STAGE.custom["multistage_forecast_configs"][0].agg_freq == "D"
    assert SILVERKITE_TWO_STAGE.custom["multistage_forecast_configs"][0].model_template == "SILVERKITE"
    assert SILVERKITE_TWO_STAGE.custom["multistage_forecast_configs"][0].model_components == ModelComponentsParam(
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
            "drop_pred_cols": None,
            "explicit_pred_cols": None,
            "regression_weight_col": None,
            "normalize_method": "zero_to_one"
        }
    )

    assert SILVERKITE_TWO_STAGE.custom["multistage_forecast_configs"][1].train_length == f"{7 * 4}D"
    assert SILVERKITE_TWO_STAGE.custom["multistage_forecast_configs"][1].fit_length is None
    assert SILVERKITE_TWO_STAGE.custom["multistage_forecast_configs"][1].agg_func == "nanmean"
    assert SILVERKITE_TWO_STAGE.custom["multistage_forecast_configs"][1].agg_freq is None
    assert SILVERKITE_TWO_STAGE.custom["multistage_forecast_configs"][1].model_template == "SILVERKITE"
    assert SILVERKITE_TWO_STAGE.custom["multistage_forecast_configs"][1].model_components == ModelComponentsParam(
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
            "drop_pred_cols": None,
            "explicit_pred_cols": None,
            "regression_weight_col": None,
            "normalize_method": "zero_to_one"
        }
    )


def test_multistage_forecast_silverkite_wow():
    """Tests the SILVERKITE_WOW template. To alert any changes to the template."""
    assert len(SILVERKITE_WOW.custom["multistage_forecast_configs"]) == 2

    assert SILVERKITE_WOW.custom["multistage_forecast_configs"][0].train_length == f"1096D"
    assert SILVERKITE_WOW.custom["multistage_forecast_configs"][0].fit_length is None
    assert SILVERKITE_WOW.custom["multistage_forecast_configs"][0].agg_func == "nanmean"
    assert SILVERKITE_WOW.custom["multistage_forecast_configs"][0].agg_freq == "D"
    assert SILVERKITE_WOW.custom["multistage_forecast_configs"][0].model_template == "SILVERKITE_EMPTY"
    assert SILVERKITE_WOW.custom["multistage_forecast_configs"][0].model_components == ModelComponentsParam(
        seasonality={
            "auto_seasonality": True,
            "yearly_seasonality": True,
            "quarterly_seasonality": True,
            "monthly_seasonality": True,
            "weekly_seasonality": False,
            "daily_seasonality": False,
        },
        growth={
            "growth_term": "linear"
        },
        events={
            "auto_holiday": True,
            "holidays_to_model_separately": None,
            "holiday_lookup_countries": ("US",),
            "holiday_pre_num_days": 0,
            "holiday_post_num_days": 0,
            "holiday_pre_post_num_dict": None,
            "daily_event_df_dict": None,
        },
        changepoints={
            "auto_growth": True,
            "changepoints_dict": None,
            "seasonality_changepoints_dict": None
        },
        autoregression={
            "autoreg_dict": None
        },
        regressors={
            "regressor_cols": None
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
            "feature_sets_enabled": None,
            "max_daily_seas_interaction_order": 0,
            "max_weekly_seas_interaction_order": 0,
            "extra_pred_cols": [],
            "min_admissible_value": None,
            "max_admissible_value": None,
            "drop_pred_cols": None,
            "explicit_pred_cols": None,
            "regression_weight_col": None,
            "normalize_method": "zero_to_one"
        }
    )

    assert SILVERKITE_WOW.custom["multistage_forecast_configs"][1].train_length == f"28D"
    assert SILVERKITE_WOW.custom["multistage_forecast_configs"][1].fit_length is None
    assert SILVERKITE_WOW.custom["multistage_forecast_configs"][1].agg_func == "nanmean"
    assert SILVERKITE_WOW.custom["multistage_forecast_configs"][1].agg_freq is None
    assert SILVERKITE_WOW.custom["multistage_forecast_configs"][1].model_template == "LAG_BASED"
    assert SILVERKITE_WOW.custom["multistage_forecast_configs"][1].model_components == ModelComponentsParam(
        custom={
            "freq": None,
            "lag_unit": "week",
            "lags": [1],
            "agg_func": "mean",
            "agg_func_params": None,
            "past_df": None,
            "series_na_fill_func": None
        },
        uncertainty={
            "uncertainty_dict": None
        }
    )


def test_multistage_forecast_model_template_enum():
    """Tests the members of `MultistageForecastModelTemplateEnum`."""
    assert MultistageForecastModelTemplateEnum.SILVERKITE.value == SimpleSilverkiteTemplate
    assert MultistageForecastModelTemplateEnum.SILVERKITE_EMPTY.value == SimpleSilverkiteTemplate
    assert MultistageForecastModelTemplateEnum.PROPHET.value == ProphetTemplate
    assert MultistageForecastModelTemplateEnum.SK.value == SilverkiteTemplate
    assert MultistageForecastModelTemplateEnum.AUTO_ARIMA.value == AutoArimaTemplate
