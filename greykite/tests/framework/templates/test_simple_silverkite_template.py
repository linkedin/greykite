import dataclasses
import datetime
import warnings
from typing import Type

import numpy as np
import pandas as pd
import pytest
from testfixtures import LogCapture

from greykite.algo.forecast.silverkite.constants.silverkite_constant import SilverkiteConstant
from greykite.algo.forecast.silverkite.constants.silverkite_holiday import SilverkiteHoliday
from greykite.algo.forecast.silverkite.forecast_simple_silverkite import SimpleSilverkiteForecast
from greykite.common.constants import ADJUSTMENT_DELTA_COL
from greykite.common.constants import END_TIME_COL
from greykite.common.constants import METRIC_COL
from greykite.common.constants import PREDICTED_COL
from greykite.common.constants import START_TIME_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.data_loader import DataLoader
from greykite.common.evaluation import EvaluationMetricEnum
from greykite.common.logging import LOGGER_NAME
from greykite.common.python_utils import assert_equal
from greykite.common.testing_utils import generate_df_for_tests
from greykite.common.testing_utils import generate_df_with_reg_for_tests
from greykite.common.time_properties_forecast import get_forecast_time_properties
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
from greykite.framework.templates.simple_silverkite_template import SimpleSilverkiteTemplate
from greykite.framework.templates.simple_silverkite_template_config import SILVERKITE_COMPONENT_KEYWORDS
from greykite.framework.templates.simple_silverkite_template_config import SimpleSilverkiteTemplateOptions
from greykite.framework.utils.framework_testing_utils import assert_basic_pipeline_equal
from greykite.framework.utils.framework_testing_utils import check_forecast_pipeline_result
from greykite.framework.utils.result_summary import summarize_grid_search_results
from greykite.sklearn.estimator.silverkite_diagnostics import SilverkiteDiagnostics
from greykite.sklearn.estimator.simple_silverkite_estimator import SimpleSilverkiteEstimator


@pytest.fixture
def silverkite():
    return SimpleSilverkiteForecast()


@pytest.fixture
def silverkite_diagnostics():
    return SilverkiteDiagnostics()


class MySilverkiteHoliday(SilverkiteHoliday):
    """Custom SilverkiteHoliday constants"""
    HOLIDAY_LOOKUP_COUNTRIES_AUTO = ("UnitedStates")
    ALL_HOLIDAYS_IN_COUNTRIES = "ALL_HOLIDAYS_CONSTANTS"


class MySilverkiteHolidayMixin:
    """Custom mixin for the SilverkiteHoliday constants"""
    def get_silverkite_holiday(self) -> Type[SilverkiteHoliday]:
        """Return the SilverkiteHoliday constants"""
        return MySilverkiteHoliday


class MySilverkiteConstant(
    MySilverkiteHolidayMixin,  # custom value, overrides defaults below
    SilverkiteConstant,
):
    """Custom constants that will be used by Silverkite."""
    pass


def test_template_name_from_dataclass():
    sst = SimpleSilverkiteTemplate()
    name = SimpleSilverkiteTemplateOptions()
    name = sst._SimpleSilverkiteTemplate__template_name_from_dataclass(name)
    assert name == "DAILY_SEAS_LT_GR_LINEAR_CP_NONE_HOL_NONE_FEASET_OFF_ALGO_LINEAR_AR_OFF_DSI_AUTO_WSI_AUTO"
    name = SimpleSilverkiteTemplateOptions(
        freq=SILVERKITE_COMPONENT_KEYWORDS.FREQ.value.HOURLY,
        cp=SILVERKITE_COMPONENT_KEYWORDS.CP.value.LT
    )
    name = sst._SimpleSilverkiteTemplate__template_name_from_dataclass(name)
    assert name == "HOURLY_SEAS_LT_GR_LINEAR_CP_LT_HOL_NONE_FEASET_OFF_ALGO_LINEAR_AR_OFF_DSI_AUTO_WSI_AUTO"


def test_decode_single_template():
    sst = SimpleSilverkiteTemplate()
    # Does not change "SILVERKITE" or "SILVERKITE_DAILY_1_CONFIG_{i}", i = 1, 2, 3.
    assert sst._SimpleSilverkiteTemplate__decode_single_template("SILVERKITE") == "SILVERKITE"
    assert sst._SimpleSilverkiteTemplate__decode_single_template("SILVERKITE_DAILY_1_CONFIG_1") == "SILVERKITE_DAILY_1_CONFIG_1"
    assert sst._SimpleSilverkiteTemplate__decode_single_template("SILVERKITE_DAILY_1_CONFIG_2") == "SILVERKITE_DAILY_1_CONFIG_2"
    assert sst._SimpleSilverkiteTemplate__decode_single_template("SILVERKITE_DAILY_1_CONFIG_3") == "SILVERKITE_DAILY_1_CONFIG_3"
    # Puts components in the correct order.
    assert (sst._SimpleSilverkiteTemplate__decode_single_template("HOURLY_AR_AUTO_CP_LT_FEASET_AUTO_HOL_NONE_GR_LINEAR_ALGO_RIDGE_SEAS_NM") ==
            "HOURLY_SEAS_NM_GR_LINEAR_CP_LT_HOL_NONE_FEASET_AUTO_ALGO_RIDGE_AR_AUTO_DSI_AUTO_WSI_AUTO")
    # Recognizes all keywords 1.
    assert (sst._SimpleSilverkiteTemplate__decode_single_template("HOURLY_SEAS_NM_GR_LINEAR_CP_LT_HOL_NONE_FEASET_AUTO_ALGO_RIDGE_AR_AUTO_DSI_AUTO_WSI_AUTO") ==
            "HOURLY_SEAS_NM_GR_LINEAR_CP_LT_HOL_NONE_FEASET_AUTO_ALGO_RIDGE_AR_AUTO_DSI_AUTO_WSI_AUTO")
    # Recognizes all keywords 2.
    assert (sst._SimpleSilverkiteTemplate__decode_single_template("HOURLY_SEAS_NM_GR_LINEAR_CP_LT_HOL_NONE_FEASET_AUTO_ALGO_RIDGE_AR_AUTO_DSI_OFF_WSI_OFF") ==
            "HOURLY_SEAS_NM_GR_LINEAR_CP_LT_HOL_NONE_FEASET_AUTO_ALGO_RIDGE_AR_AUTO_DSI_OFF_WSI_OFF")
    # Automatically fills missing components.
    assert (sst._SimpleSilverkiteTemplate__decode_single_template("DAILY_AR_AUTO_CP_LT_FEASET_AUTO_HOL_NONE_GR_LINEAR") ==
            "DAILY_SEAS_LT_GR_LINEAR_CP_LT_HOL_NONE_FEASET_AUTO_ALGO_LINEAR_AR_AUTO_DSI_AUTO_WSI_AUTO")
    # Provides frequency only.
    assert (sst._SimpleSilverkiteTemplate__decode_single_template("DAILY") ==
            "DAILY_SEAS_LT_GR_LINEAR_CP_NONE_HOL_NONE_FEASET_OFF_ALGO_LINEAR_AR_OFF_DSI_AUTO_WSI_AUTO")
    # Error: not recognized as a single template.
    with pytest.raises(
            ValueError,
            match=f"Template not recognized. Please make sure the frequency belongs to \\['HOURLY', 'DAILY', 'WEEKLY'\\]."):
        sst._SimpleSilverkiteTemplate__decode_single_template("SILVERKITE_DAILY_1")
    with pytest.raises(
            ValueError,
            match=f"Template not recognized. Please make sure the frequency belongs to \\['HOURLY', 'DAILY', 'WEEKLY'\\]."):
        sst._SimpleSilverkiteTemplate__decode_single_template("SILVERKITE_DAILY_90")
    # ERROR: no keyword.
    with pytest.raises(
            ValueError,
            match=f"No keyword found in the template name. Please make sure your template name consists of keyword \\+ value pairs."):
        sst._SimpleSilverkiteTemplate__decode_single_template("WEEKLY_LT")
    # Error: wrong value without keyword.
    with pytest.raises(
            ValueError,
            match="ONN is not recognized for AR. The valid values are \\['OFF', 'AUTO', 'DEFAULT'\\]."):
        sst._SimpleSilverkiteTemplate__decode_single_template("WEEKLY_AR_ONN")
    # Error: wrong format1.
    with pytest.raises(
            ValueError,
            match="CP is not recognized for SEAS. The valid values are \\['LT', 'NM', 'HV', 'LTQM', 'NMQM', 'HVQM', 'NONE', 'DEFAULT'\\]."):
        sst._SimpleSilverkiteTemplate__decode_single_template("WEEKLY_SEAS_CP")
    # Error: wrong format2.
    with pytest.raises(
            ValueError,
            match="Template name is not valid. It must be either keyword \\+ value or purely values."):
        sst._SimpleSilverkiteTemplate__decode_single_template("WEEKLY_SEAS")
    # Unused components warning.
    with pytest.warns(UserWarning) as record:
        sst._SimpleSilverkiteTemplate__decode_single_template("HOURLY_AR_AUTO_CP_LT_FEASET_AUTO_HOL_NONE_GR_LINEAR_ALGO_RIDGE_SEAS_NM_AAA")
        assert ("The following words are not used because they are neither a component keyword"
                f" nor following a component keyword, or because you have duplicate keywords. {['AAA']}" in record[0].message.args[0])
    # Default fill warning.
    with pytest.warns(UserWarning) as record:
        sst._SimpleSilverkiteTemplate__decode_single_template("HOURLY_AR_AUTO_CP_LT_FEASET_AUTO_HOL_NONE_GR_LINEAR_ALGO_RIDGE")
        assert ("The following component keywords are not found in the template name, "
                "thus the default values will be used. For default values, please check the doc. "
                f"{['SEAS']}" in record[0].message.args[0])
    # Tests input a dataclass.
    name = SimpleSilverkiteTemplateOptions(
            freq=SILVERKITE_COMPONENT_KEYWORDS.FREQ.value.HOURLY,
            cp=SILVERKITE_COMPONENT_KEYWORDS.CP.value.LT
        )
    name = sst._SimpleSilverkiteTemplate__decode_single_template(name)
    assert name == "HOURLY_SEAS_LT_GR_LINEAR_CP_LT_HOL_NONE_FEASET_OFF_ALGO_LINEAR_AR_OFF_DSI_AUTO_WSI_AUTO"


def test_check_template_type():
    sst = SimpleSilverkiteTemplate()
    assert sst.check_template_type("SILVERKITE") == "single"
    assert sst.check_template_type("SILVERKITE_DAILY_1_CONFIG_1") == "single"
    assert sst.check_template_type("SILVERKITE_DAILY_1_CONFIG_2") == "single"
    assert sst.check_template_type("SILVERKITE_DAILY_1_CONFIG_3") == "single"
    assert sst.check_template_type("WEEKLY") == "single"
    assert sst.check_template_type("WEEKLY_CP_LT") == "single"
    assert sst.check_template_type("WEEKLY_LT") == "single"
    assert sst.check_template_type("SILVERKITE_DAILY_1") == "multi"
    assert sst.check_template_type("SILVERKITE_DAILY_90") == "multi"
    assert sst.check_template_type("SILVERKITE_WEEKLY") == "multi"
    assert sst.check_template_type(SimpleSilverkiteTemplateOptions()) == "single"
    with pytest.raises(
            ValueError,
            match=f"The template name SILVERKITE_WEEKLY_100 is not recognized. It must be 'SILVERKITE', "
                  f"'SILVERKITE_DAILY_1_CONFIG_1', 'SILVERKITE_DAILY_1_CONFIG_2', 'SILVERKITE_DAILY_1_CONFIG_3', 'SILVERKITE_EMPTY', "
                  f"a `SimpleSilverkiteTemplateOptions` data class, of the type"
                  " '\\{FREQ\\}_SEAS_\\{VAL\\}_GR_\\{VAL\\}_CP_\\{VAL\\}_HOL_\\{VAL\\}_FEASET_\\{VAL\\}_ALGO_\\{VAL\\}_AR_\\{VAL\\}' or"
                  f" belong to \\['SILVERKITE_DAILY_1', 'SILVERKITE_DAILY_90', 'SILVERKITE_WEEKLY', 'SILVERKITE_HOURLY_1', 'SILVERKITE_HOURLY_24', "
                  f"'SILVERKITE_HOURLY_168', 'SILVERKITE_HOURLY_336'\\]."):
        sst.check_template_type("SILVERKITE_WEEKLY_100")


def test_get_name_string_from_model_template():
    sst = SimpleSilverkiteTemplate()
    name_string = sst._SimpleSilverkiteTemplate__get_name_string_from_model_template("SILVERKITE")
    assert name_string == ["SILVERKITE"]
    name_string = sst._SimpleSilverkiteTemplate__get_name_string_from_model_template("SILVERKITE_DAILY_1_CONFIG_1")
    assert name_string == ["SILVERKITE_DAILY_1_CONFIG_1"]
    name_string = sst._SimpleSilverkiteTemplate__get_name_string_from_model_template("SILVERKITE_DAILY_1_CONFIG_2")
    assert name_string == ["SILVERKITE_DAILY_1_CONFIG_2"]
    name_string = sst._SimpleSilverkiteTemplate__get_name_string_from_model_template("SILVERKITE_DAILY_1_CONFIG_3")
    assert name_string == ["SILVERKITE_DAILY_1_CONFIG_3"]
    name_string = sst._SimpleSilverkiteTemplate__get_name_string_from_model_template("SILVERKITE_EMPTY")
    assert name_string == ["DAILY_SEAS_NONE_GR_NONE_CP_NONE_HOL_NONE_FEASET_OFF_ALGO_LINEAR_AR_OFF_DSI_OFF_WSI_OFF"]
    name_string = sst._SimpleSilverkiteTemplate__get_name_string_from_model_template("DAILY_SEAS_HV_CP_HV")
    assert name_string == ["DAILY_SEAS_HV_GR_LINEAR_CP_HV_HOL_NONE_FEASET_OFF_ALGO_LINEAR_AR_OFF_DSI_AUTO_WSI_AUTO"]
    name_string = sst._SimpleSilverkiteTemplate__get_name_string_from_model_template(
        SimpleSilverkiteTemplateOptions(
            freq=SILVERKITE_COMPONENT_KEYWORDS.FREQ.value.DAILY,
            seas=SILVERKITE_COMPONENT_KEYWORDS.SEAS.value.HV,
            cp=SILVERKITE_COMPONENT_KEYWORDS.CP.value.HV
        )
    )
    assert name_string == ["DAILY_SEAS_HV_GR_LINEAR_CP_HV_HOL_NONE_FEASET_OFF_ALGO_LINEAR_AR_OFF_DSI_AUTO_WSI_AUTO"]
    name_strings = sst._SimpleSilverkiteTemplate__get_name_string_from_model_template(
        [
            "SILVERKITE",
            "SILVERKITE_DAILY_1_CONFIG_1",
            "SILVERKITE_DAILY_1_CONFIG_2",
            "SILVERKITE_DAILY_1_CONFIG_3",
            "SILVERKITE_EMPTY",
            "SILVERKITE_WEEKLY",
            "DAILY_SEAS_NONE_GR_NONE_CP_NONE_HOL_NONE_FEASET_OFF_ALGO_LINEAR_AR_OFF_DSI_OFF_WSI_OFF",
            SimpleSilverkiteTemplateOptions(
                freq=SILVERKITE_COMPONENT_KEYWORDS.FREQ.value.DAILY,
                seas=SILVERKITE_COMPONENT_KEYWORDS.SEAS.value.HV,
                cp=SILVERKITE_COMPONENT_KEYWORDS.CP.value.HV)]
    )
    assert name_strings == [
        "SILVERKITE",
        "SILVERKITE_DAILY_1_CONFIG_1",
        "SILVERKITE_DAILY_1_CONFIG_2",
        "SILVERKITE_DAILY_1_CONFIG_3",
        "DAILY_SEAS_NONE_GR_NONE_CP_NONE_HOL_NONE_FEASET_OFF_ALGO_LINEAR_AR_OFF_DSI_OFF_WSI_OFF",
        "WEEKLY_SEAS_NM_GR_LINEAR_CP_NONE_HOL_NONE_FEASET_OFF_ALGO_LINEAR_AR_OFF_DSI_AUTO_WSI_AUTO",
        "WEEKLY_SEAS_NM_GR_LINEAR_CP_LT_HOL_NONE_FEASET_OFF_ALGO_LINEAR_AR_OFF_DSI_AUTO_WSI_AUTO",
        "WEEKLY_SEAS_HV_GR_LINEAR_CP_NM_HOL_NONE_FEASET_OFF_ALGO_RIDGE_AR_OFF_DSI_AUTO_WSI_AUTO",
        "WEEKLY_SEAS_HV_GR_LINEAR_CP_LT_HOL_NONE_FEASET_OFF_ALGO_RIDGE_AR_OFF_DSI_AUTO_WSI_AUTO",
        "DAILY_SEAS_NONE_GR_NONE_CP_NONE_HOL_NONE_FEASET_OFF_ALGO_LINEAR_AR_OFF_DSI_OFF_WSI_OFF",
        "DAILY_SEAS_HV_GR_LINEAR_CP_HV_HOL_NONE_FEASET_OFF_ALGO_LINEAR_AR_OFF_DSI_AUTO_WSI_AUTO"
    ]


def test_get_single_model_components_param_from_template():
    sst = SimpleSilverkiteTemplate()
    model_components = sst._SimpleSilverkiteTemplate__get_single_model_components_param_from_template("SILVERKITE")
    assert model_components == ModelComponentsParam(
        seasonality={
            "auto_seasonality": False,
            "yearly_seasonality": "auto",
            "quarterly_seasonality": "auto",
            "monthly_seasonality": "auto",
            "weekly_seasonality": "auto",
            "daily_seasonality": "auto",
        },
        growth={
            "growth_term": "linear"
        },
        events={
            "auto_holiday": False,
            "holidays_to_model_separately": "auto",
            "holiday_lookup_countries": "auto",
            "holiday_pre_num_days": 2,
            "holiday_post_num_days": 2,
            "holiday_pre_post_num_dict": None,
            "daily_event_df_dict": None,
        },
        changepoints={
            "auto_growth": False,
            "changepoints_dict": {
                "method": "auto",
                "yearly_seasonality_order": 15,
                "resample_freq": "3D",
                "regularization_strength": 0.6,
                "actual_changepoint_min_distance": "30D",
                "potential_changepoint_distance": "15D",
                "no_changepoint_distance_from_end": "90D"
            },
            "seasonality_changepoints_dict": None
        },
        autoregression={
            "autoreg_dict": "auto",
            "simulation_num": 10,
            "fast_simulation": False,
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
            "drop_pred_cols": None,
            "explicit_pred_cols": None,
            "regression_weight_col": None,
            "min_admissible_value": None,
            "max_admissible_value": None,
            "normalize_method": "zero_to_one"
        }
    )

    model_components = sst._SimpleSilverkiteTemplate__get_single_model_components_param_from_template("SILVERKITE_DAILY_1_CONFIG_1")
    assert model_components == ModelComponentsParam(
        seasonality={
            "auto_seasonality": False,
            "yearly_seasonality": 8,
            "quarterly_seasonality": 0,
            "monthly_seasonality": 7,
            "weekly_seasonality": 1,
            "daily_seasonality": 0,
        },
        growth={
            "growth_term": "linear"
        },
        events={
            "auto_holiday": False,
            "holidays_to_model_separately": SilverkiteHoliday.HOLIDAYS_TO_MODEL_SEPARATELY_AUTO,
            "holiday_lookup_countries": SilverkiteHoliday.HOLIDAY_LOOKUP_COUNTRIES_AUTO,
            "holiday_pre_num_days": 2,
            "holiday_post_num_days": 2,
            "holiday_pre_post_num_dict": None,
            "daily_event_df_dict": None,
        },
        changepoints={
            "auto_growth": False,
            "changepoints_dict": {
                "method": "auto",
                "resample_freq": "7D",
                "regularization_strength": 0.809,
                "potential_changepoint_distance": "7D",
                "no_changepoint_distance_from_end": "7D",
                "yearly_seasonality_order": 8,
                "yearly_seasonality_change_freq": None,
            },
            "seasonality_changepoints_dict": None
        },
        autoregression={
            "autoreg_dict": "auto",
            "simulation_num": 10,
            "fast_simulation": False
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
            "drop_pred_cols": None,
            "explicit_pred_cols": None,
            "regression_weight_col": None,
            "min_admissible_value": None,
            "max_admissible_value": None,
            "normalize_method": "zero_to_one"
        }
    )
    model_components = sst._SimpleSilverkiteTemplate__get_single_model_components_param_from_template("SILVERKITE_DAILY_1_CONFIG_2")
    assert model_components == ModelComponentsParam(
        seasonality={
            "auto_seasonality": False,
            "yearly_seasonality": 1,
            "quarterly_seasonality": 0,
            "monthly_seasonality": 4,
            "weekly_seasonality": 6,
            "daily_seasonality": 0,
        },
        growth={
            "growth_term": "linear"
        },
        events={
            "auto_holiday": False,
            "holidays_to_model_separately": SilverkiteHoliday.HOLIDAYS_TO_MODEL_SEPARATELY_AUTO,
            "holiday_lookup_countries": SilverkiteHoliday.HOLIDAY_LOOKUP_COUNTRIES_AUTO,
            "holiday_pre_num_days": 2,
            "holiday_post_num_days": 2,
            "holiday_pre_post_num_dict": None,
            "daily_event_df_dict": None,
        },
        changepoints={
            "auto_growth": False,
            "changepoints_dict": {
                "method": "auto",
                "resample_freq": "7D",
                "regularization_strength": 0.624,
                "potential_changepoint_distance": "7D",
                "no_changepoint_distance_from_end": "17D",
                "yearly_seasonality_order": 1,
                "yearly_seasonality_change_freq": None,
            },
            "seasonality_changepoints_dict": None
        },
        autoregression={
            "autoreg_dict": "auto",
            "simulation_num": 10,
            "fast_simulation": False
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
            "drop_pred_cols": None,
            "explicit_pred_cols": None,
            "regression_weight_col": None,
            "min_admissible_value": None,
            "max_admissible_value": None,
            "normalize_method": "zero_to_one"
        }
    )
    model_components = sst._SimpleSilverkiteTemplate__get_single_model_components_param_from_template("SILVERKITE_DAILY_1_CONFIG_3")
    assert model_components == ModelComponentsParam(
        seasonality={
            "auto_seasonality": False,
            "yearly_seasonality": 40,
            "quarterly_seasonality": 0,
            "monthly_seasonality": 0,
            "weekly_seasonality": 2,
            "daily_seasonality": 0,
        },
        growth={
            "growth_term": "linear"
        },
        events={
            "auto_holiday": False,
            "holidays_to_model_separately": SilverkiteHoliday.HOLIDAYS_TO_MODEL_SEPARATELY_AUTO,
            "holiday_lookup_countries": SilverkiteHoliday.HOLIDAY_LOOKUP_COUNTRIES_AUTO,
            "holiday_pre_num_days": 2,
            "holiday_post_num_days": 2,
            "holiday_pre_post_num_dict": None,
            "daily_event_df_dict": None,
        },
        changepoints={
            "auto_growth": False,
            "changepoints_dict": {
                "method": "auto",
                "resample_freq": "7D",
                "regularization_strength": 0.590,
                "potential_changepoint_distance": "7D",
                "no_changepoint_distance_from_end": "8D",
                "yearly_seasonality_order": 40,
                "yearly_seasonality_change_freq": None,
            },
            "seasonality_changepoints_dict": None
        },
        autoregression={
            "autoreg_dict": "auto",
            "simulation_num": 10,
            "fast_simulation": False
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
            "drop_pred_cols": None,
            "explicit_pred_cols": None,
            "regression_weight_col": None,
            "min_admissible_value": None,
            "max_admissible_value": None,
            "normalize_method": "zero_to_one"
        }
    )
    model_components = sst._SimpleSilverkiteTemplate__get_single_model_components_param_from_template(
        "DAILY_SEAS_NONE_GR_NONE_CP_NONE_HOL_NONE_FEASET_OFF_ALGO_LINEAR_AR_OFF_DSI_OFF_WSI_OFF")
    assert model_components == ModelComponentsParam(
        seasonality={
            "auto_seasonality": False,
            "yearly_seasonality": 0,
            "quarterly_seasonality": 0,
            "monthly_seasonality": 0,
            "weekly_seasonality": 0,
            "daily_seasonality": 0,
        },
        growth={
            "growth_term": None
        },
        events={
            "auto_holiday": False,
            "holidays_to_model_separately": [],
            "holiday_lookup_countries": [],
            "holiday_pre_num_days": 0,
            "holiday_post_num_days": 0,
            "holiday_pre_post_num_dict": None,
            "daily_event_df_dict": None,
        },
        changepoints={
            "auto_growth": False,
            "changepoints_dict": None,
            "seasonality_changepoints_dict": None
        },
        autoregression={
            "autoreg_dict": None,
            "simulation_num": 10,
            "fast_simulation": False
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
                "fit_algorithm": "linear",
                "fit_algorithm_params": None,
            },
            "feature_sets_enabled": False,
            "max_daily_seas_interaction_order": 0,
            "max_weekly_seas_interaction_order": 0,
            "extra_pred_cols": [],
            "drop_pred_cols": None,
            "explicit_pred_cols": None,
            "regression_weight_col": None,
            "min_admissible_value": None,
            "max_admissible_value": None,
            "normalize_method": "zero_to_one"
        }
    )


def test_get_model_components_from_model_template(silverkite, silverkite_diagnostics):
    """Tests get_model_components_from_model_template and get_model_components_and_override_from_model_template."""
    sst = SimpleSilverkiteTemplate()
    model_components = sst.get_model_components_from_model_template("SILVERKITE")[0]
    assert model_components.seasonality == {
        "auto_seasonality": False,
        "yearly_seasonality": "auto",
        "quarterly_seasonality": "auto",
        "monthly_seasonality": "auto",
        "weekly_seasonality": "auto",
        "daily_seasonality": "auto",
    }
    assert model_components.growth == {
        "growth_term": "linear"
    }
    assert model_components.events == {
        "auto_holiday": False,
        "holidays_to_model_separately": "auto",
        "holiday_lookup_countries": "auto",
        "holiday_pre_num_days": 2,
        "holiday_post_num_days": 2,
        "holiday_pre_post_num_dict": None,
        "daily_event_df_dict": None,
    }
    assert model_components.changepoints == {
        "auto_growth": False,
        "changepoints_dict": {
            "method": "auto",
            "yearly_seasonality_order": 15,
            "resample_freq": "3D",
            "regularization_strength": 0.6,
            "actual_changepoint_min_distance": "30D",
            "potential_changepoint_distance": "15D",
            "no_changepoint_distance_from_end": "90D"
        },
        "seasonality_changepoints_dict": None
    }
    assert model_components.autoregression == {
        "autoreg_dict": "auto",
        "simulation_num": 10,
        "fast_simulation": False
    }
    assert model_components.regressors == {
        "regressor_cols": []
    }
    assert model_components.lagged_regressors == {
       "lagged_regressor_dict": None
    }
    assert model_components.uncertainty == {
        "uncertainty_dict": None
    }
    assert model_components.custom == {
        "fit_algorithm_dict": {
            "fit_algorithm": "ridge",
            "fit_algorithm_params": None
        },
        "feature_sets_enabled": "auto",
        "max_daily_seas_interaction_order": 5,
        "max_weekly_seas_interaction_order": 2,
        "extra_pred_cols": [],
        "drop_pred_cols": None,
        "explicit_pred_cols": None,
        "regression_weight_col": None,
        "min_admissible_value": None,
        "max_admissible_value": None,
        "normalize_method": "zero_to_one"
    }
    assert model_components.hyperparameter_override is None

    # overwrite some parameters
    daily_event_df_dict = silverkite._SimpleSilverkiteForecast__get_silverkite_holidays(
        holiday_lookup_countries=["India"],
        holidays_to_model_separately=["Easter Sunday", "Republic Day"],
        start_year=2017,
        end_year=2025,
        pre_num=2,
        post_num=2)
    model_components = ModelComponentsParam(
        seasonality={
            "auto_seasonality": False,
            "yearly_seasonality": True,
            "weekly_seasonality": False
        },
        growth={
            "growth_term": "quadratic"
        },
        events={
            "auto_holiday": False,
            "holidays_to_model_separately": SilverkiteHoliday.ALL_HOLIDAYS_IN_COUNTRIES,
            "holiday_lookup_countries": ["UnitedStates"],
            "holiday_pre_num_days": 3,
            "holiday_pre_post_num_dict": {"New Year's Day": (7, 3)},
            "daily_event_df_dict": daily_event_df_dict
        },
        changepoints={
            "auto_growth": False,
            "changepoints_dict": {
                "method": "uniform",
                "n_changepoints": 20,
            },
            "seasonality_changepoints_dict": {
                "regularization_strength": 0.5
            }
        },
        autoregression={
            "autoreg_dict": {
                "dummy_key": "test_value"
            },
            "simulation_num": 10,
            "fast_simulation": False
        },
        regressors={
            "regressor_cols": ["r1", "r2"]
        },
        lagged_regressors={
            "lagged_regressor_dict": {
                "dummy_key": "test_value"
            }
        },
        uncertainty={
            "uncertainty_dict": "auto",
        },
        custom={
            "fit_algorithm_dict": {
                "fit_algorithm": "ridge",
                "fit_algorithm_params": {"normalize": True},
            },
            "feature_sets_enabled": False
        },
        hyperparameter_override={
            "input__response__null__max_frac": 0.1,
            "estimator__silverkite": silverkite,
            "estimator__silverkite_diagnostics": silverkite_diagnostics,
        }
    )
    original_components = dataclasses.replace(model_components)  # creates a copy
    updated_components = sst._SimpleSilverkiteTemplate__get_model_components_and_override_from_model_template(
        template="SILVERKITE",
        model_components=model_components)[0]
    assert original_components == model_components  # not mutated by the function
    assert updated_components.growth == {
        "growth_term": "quadratic"
    }
    assert updated_components.seasonality == {
        "auto_seasonality": False,
        "yearly_seasonality": True,
        "quarterly_seasonality": "auto",
        "weekly_seasonality": False,
        "monthly_seasonality": "auto",
        "daily_seasonality": "auto",
    }
    assert_equal(updated_components.events, {
        "auto_holiday": False,
        "holidays_to_model_separately": SilverkiteHoliday.ALL_HOLIDAYS_IN_COUNTRIES,
        "holiday_lookup_countries": ["UnitedStates"],
        "holiday_pre_num_days": 3,
        "holiday_post_num_days": 2,
        "holiday_pre_post_num_dict": {"New Year's Day": (7, 3)},
        "daily_event_df_dict": daily_event_df_dict
    })
    assert updated_components.changepoints == {
        "auto_growth": False,
        "changepoints_dict": {
            "method": "uniform",
            "n_changepoints": 20,
        },
        "seasonality_changepoints_dict": {
            "regularization_strength": 0.5
        }
    }
    assert updated_components.autoregression == {
        "autoreg_dict": {
            "dummy_key": "test_value"
        },
        "simulation_num": 10,
        "fast_simulation": False
    }
    assert updated_components.regressors == {
        "regressor_cols": ["r1", "r2"]
    }
    assert updated_components.lagged_regressors == {
        "lagged_regressor_dict": {
            "dummy_key": "test_value"
        }
    }
    assert updated_components.uncertainty == {
        "uncertainty_dict": "auto"
    }
    assert updated_components.custom == {
        "fit_algorithm_dict": {
            "fit_algorithm": "ridge",
            "fit_algorithm_params": {"normalize": True},
        },
        "feature_sets_enabled": False,
        "max_daily_seas_interaction_order": 5,
        "max_weekly_seas_interaction_order": 2,
        "extra_pred_cols": [],
        "drop_pred_cols": None,
        "explicit_pred_cols": None,
        "regression_weight_col": None,
        "min_admissible_value": None,
        "max_admissible_value": None,
        "normalize_method": "zero_to_one"
    }
    assert updated_components.hyperparameter_override == {
        "input__response__null__max_frac": 0.1,
        "estimator__silverkite": silverkite,
        "estimator__silverkite_diagnostics": silverkite_diagnostics
    }
    # test change point features
    model_components = ModelComponentsParam(
        changepoints={
            "changepoints_dict": {
                "method": "auto",
                "yearly_seasonality_order": 10,
                "no_changepoint_proportion_from_end": 0.2
            }
        }
    )
    updated_components = sst._SimpleSilverkiteTemplate__get_model_components_and_override_from_model_template(
        template="SILVERKITE",
        model_components=model_components)[0]
    assert updated_components.changepoints == {
        "auto_growth": False,
        "changepoints_dict": {
            "method": "auto",
            "yearly_seasonality_order": 10,
            "no_changepoint_proportion_from_end": 0.2
        },
        "seasonality_changepoints_dict": None
    }


def test_override_model_components(silverkite, silverkite_diagnostics):
    sst = SimpleSilverkiteTemplate()
    default_model_components = ModelComponentsParam(
        seasonality={
            "auto_seasonality": False,
            "yearly_seasonality": 8,
            "quarterly_seasonality": 3,
            "monthly_seasonality": 2,
            "weekly_seasonality": 3,
            "daily_seasonality": 0
        },
        growth={
            "growth_term": "linear"
        },
        changepoints={
            "auto_growth": False,
            "changepoints_dict": {
                "method": "auto",
                "resample_freq": "7D",
                "regularization_strength": 0.6,
                "potential_changepoint_distance": "15D",
                "no_changepoint_distance_from_end": "90D",
                "yearly_seasonality_order": 15
            },
            "seasonality_changepoints_dict": None
        },
        events={
            "auto_holiday": False,
            "holidays_to_model_separately": [],
            "holiday_lookup_countries": [],
            "holiday_pre_num_days": 0,
            "holiday_post_num_days": 0,
            "holiday_pre_post_num_dict": None,
            "daily_event_df_dict": None,
        },
        custom={
            "feature_sets_enabled": None,
            "fit_algorithm_dict": {
                "fit_algorithm": "linear",
                "fit_algorithm_params": None
            },
            "max_daily_seas_interaction_order": 5,
            "max_weekly_seas_interaction_order": 2,
            "extra_pred_cols": [],
            "min_admissible_value": None,
            "max_admissible_value": None
        },
        autoregression={
            "autoreg_dict": None,
            "simulation_num": 10,
            "fast_simulation": False
        },
        regressors={
            "regressor_cols": None
        },
        lagged_regressors={
            "lagged_regressor_dict": None
        },
        uncertainty={
            "uncertainty_dict": None
        })
    model_components = ModelComponentsParam(
        seasonality={
            "yearly_seasonality": 15,
        },
        custom={
            "fit_algorithm_dict": {
                "fit_algorithm": "ridge",
                "fit_algorithm_params": None
            },
        },
        hyperparameter_override={
            "estimator__silverkite": silverkite,
            "estimator__silverkite_diagnostics": silverkite_diagnostics,
            "estimator__daily_seasonality": 10
        })
    new_model_components = sst._SimpleSilverkiteTemplate__override_model_components(
        default_model_components=default_model_components,
        model_components=model_components)
    assert new_model_components == ModelComponentsParam(
        seasonality={
            "auto_seasonality": False,
            "yearly_seasonality": 15,
            "quarterly_seasonality": 3,
            "monthly_seasonality": 2,
            "weekly_seasonality": 3,
            "daily_seasonality": 0
        },
        growth={
            "growth_term": "linear"
        },
        changepoints={
            "auto_growth": False,
            "changepoints_dict": {
                "method": "auto",
                "resample_freq": "7D",
                "regularization_strength": 0.6,
                "potential_changepoint_distance": "15D",
                "no_changepoint_distance_from_end": "90D",
                "yearly_seasonality_order": 15
            },
            "seasonality_changepoints_dict": None
        },
        events={
            "auto_holiday": False,
            "holidays_to_model_separately": [],
            "holiday_lookup_countries": [],
            "holiday_pre_num_days": 0,
            "holiday_post_num_days": 0,
            "holiday_pre_post_num_dict": None,
            "daily_event_df_dict": None,
        },
        custom={
            "feature_sets_enabled": None,
            "fit_algorithm_dict": {
                "fit_algorithm": "ridge",
                "fit_algorithm_params": None
            },
            "max_daily_seas_interaction_order": 5,
            "max_weekly_seas_interaction_order": 2,
            "extra_pred_cols": [],
            "min_admissible_value": None,
            "max_admissible_value": None
        },
        autoregression={
            "autoreg_dict": None,
            "simulation_num": 10,
            "fast_simulation": False,
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
        hyperparameter_override={
            "estimator__silverkite": silverkite,
            "estimator__silverkite_diagnostics": silverkite_diagnostics,
            "estimator__daily_seasonality": 10
        })
    # Test None model component.
    new_model_components = sst._SimpleSilverkiteTemplate__override_model_components(default_model_components)
    assert new_model_components == ModelComponentsParam(
        seasonality={
            "auto_seasonality": False,
            "yearly_seasonality": 8,
            "quarterly_seasonality": 3,
            "monthly_seasonality": 2,
            "weekly_seasonality": 3,
            "daily_seasonality": 0
        },
        growth={
            "growth_term": "linear"
        },
        changepoints={
            "auto_growth": False,
            "changepoints_dict": {
                "method": "auto",
                "resample_freq": "7D",
                "regularization_strength": 0.6,
                "potential_changepoint_distance": "15D",
                "no_changepoint_distance_from_end": "90D",
                "yearly_seasonality_order": 15
            },
            "seasonality_changepoints_dict": None
        },
        events={
            "auto_holiday": False,
            "holidays_to_model_separately": [],
            "holiday_lookup_countries": [],
            "holiday_pre_num_days": 0,
            "holiday_post_num_days": 0,
            "holiday_pre_post_num_dict": None,
            "daily_event_df_dict": None,
        },
        custom={
            "feature_sets_enabled": None,
            "fit_algorithm_dict": {
                "fit_algorithm": "linear",
                "fit_algorithm_params": None
            },
            "max_daily_seas_interaction_order": 5,
            "max_weekly_seas_interaction_order": 2,
            "extra_pred_cols": [],
            "min_admissible_value": None,
            "max_admissible_value": None
        },
        autoregression={
            "autoreg_dict": None,
            "simulation_num": 10,
            "fast_simulation": False
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
        hyperparameter_override={})


def test_get_model_components_and_override_from_model_template_single():
    """Tests `get_model_components_and_override_from_model_template` for single model template"""
    sst = SimpleSilverkiteTemplate()
    model_components = sst._SimpleSilverkiteTemplate__get_model_components_and_override_from_model_template(
        template="DAILY_CP_LT_FEASET_AUTO_AR_OFF",
        model_components=ModelComponentsParam(
            seasonality={"daily_seasonality": 12},
            regressors={"regressor_cols": ["x"]}
        )
    )
    # Checks it pulls the correct model template and overrides the parameters.
    assert model_components[0] == ModelComponentsParam(
        seasonality={
            "auto_seasonality": False,
            "yearly_seasonality": 8,
            "quarterly_seasonality": 0,
            "monthly_seasonality": 0,
            "weekly_seasonality": 3,
            "daily_seasonality": 12
        },
        growth={
            "growth_term": "linear"
        },
        changepoints={
            "auto_growth": False,
            "changepoints_dict": {
                "method": "auto",
                "resample_freq": "7D",
                "regularization_strength": 0.6,
                "potential_changepoint_distance": "15D",
                "no_changepoint_distance_from_end": "90D",
                "yearly_seasonality_order": 15,
                "yearly_seasonality_change_freq": None
            },
            "seasonality_changepoints_dict": None
        },
        events={
            "auto_holiday": False,
            "holidays_to_model_separately": [],
            "holiday_lookup_countries": [],
            "holiday_pre_num_days": 0,
            "holiday_post_num_days": 0,
            "holiday_pre_post_num_dict": None,
            "daily_event_df_dict": None,
        },
        custom={
            "feature_sets_enabled": "auto",
            "fit_algorithm_dict": {
                "fit_algorithm": "linear",
                "fit_algorithm_params": None
            },
            "max_daily_seas_interaction_order": 0,
            "max_weekly_seas_interaction_order": 2,
            "extra_pred_cols": [],
            "drop_pred_cols": None,
            "explicit_pred_cols": None,
            "regression_weight_col": None,
            "min_admissible_value": None,
            "max_admissible_value": None,
            "normalize_method": "zero_to_one"
        },
        autoregression={
            "autoreg_dict": None,
            "simulation_num": 10,
            "fast_simulation": False,
        },
        regressors={
            "regressor_cols": ["x"]
        },
        lagged_regressors={
            "lagged_regressor_dict": None
        },
        uncertainty={
            "uncertainty_dict": None
        },
        hyperparameter_override={})


def test_get_model_components_and_override_from_model_template_multi():
    """Tests `get_model_components_and_override_from_model_template` for multi model template"""
    sst = SimpleSilverkiteTemplate()
    model_components = sst._SimpleSilverkiteTemplate__get_model_components_and_override_from_model_template(
        template="SILVERKITE_DAILY_90",
        model_components=ModelComponentsParam(
            seasonality={"daily_seasonality": 12},
            regressors={"regressor_cols": ["x"]}
        )
    )
    # The model_components is used to override all single templates in `SILVERKITE_DAILY_90`.
    assert all([
        single_model_components.seasonality["daily_seasonality"] == 12
        for single_model_components in model_components])
    assert all([
        single_model_components.regressors == {"regressor_cols": ["x"]}
        for single_model_components in model_components])

    # List of template names.
    model_components = sst._SimpleSilverkiteTemplate__get_model_components_and_override_from_model_template(
        template=["WEEKLY_CP_LT", "SILVERKITE_DAILY_90"],
        model_components=ModelComponentsParam(
            seasonality={"daily_seasonality": 12},
            regressors={"regressor_cols": ["x"]}
        )
    )
    assert all([
        single_model_components.seasonality["daily_seasonality"] == 12
        for single_model_components in model_components])
    assert all([
        single_model_components.regressors == {"regressor_cols": ["x"]}
        for single_model_components in model_components])
    assert len(model_components) == 5


# The subsequent tests are for `SimpleSilverkiteTemplate` class
def test_property():
    """Tests properties"""
    assert SimpleSilverkiteTemplate().allow_model_template_list is True
    assert SimpleSilverkiteTemplate().allow_model_components_param_list is True

    template = SimpleSilverkiteTemplate()
    assert template.DEFAULT_MODEL_TEMPLATE == "SILVERKITE"
    assert isinstance(template.estimator, SimpleSilverkiteEstimator)
    assert template.estimator.coverage is None
    assert template.apply_forecast_config_defaults().model_template == "SILVERKITE"

    estimator = SimpleSilverkiteEstimator(coverage=0.99)
    template = SimpleSilverkiteTemplate(estimator=estimator)
    assert template.estimator is estimator

    silverkite_constants = MySilverkiteConstant()
    silverkite = SimpleSilverkiteForecast(constants=silverkite_constants)
    estimator = SimpleSilverkiteEstimator(silverkite=silverkite)
    template = SimpleSilverkiteTemplate(estimator=estimator)
    assert template._silverkite_holiday.ALL_HOLIDAYS_IN_COUNTRIES == MySilverkiteHoliday.ALL_HOLIDAYS_IN_COUNTRIES


def test_get_regressor_cols():
    """Tests get_regressor_cols"""
    template = SimpleSilverkiteTemplate()
    template.config = template.apply_forecast_config_defaults()
    regressor_cols = template.get_regressor_cols()
    assert regressor_cols is None

    # single list
    model_components = ModelComponentsParam(
        regressors={"regressor_cols": None})
    template.config.model_components_param = model_components
    regressor_cols = template.get_regressor_cols()
    assert regressor_cols is None

    # single list
    model_components = ModelComponentsParam(
        regressors={"regressor_cols": ["c1", "c2", "c3"]})
    template.config.model_components_param = model_components
    regressor_cols = template.get_regressor_cols()
    assert set(regressor_cols) == {"c1", "c2", "c3"}

    # list of lists
    model_components = ModelComponentsParam(
        regressors={
            "regressor_cols": [
                ["c1", "c2", "c3"],
                ["c1", "c4"],
                ["c5"],
            ]})
    template.config.model_components_param = model_components
    regressor_cols = template.get_regressor_cols()
    assert set(regressor_cols) == {"c1", "c2", "c3", "c4", "c5"}

    # list of lists, including None and []
    model_components = ModelComponentsParam(
        regressors={
            "regressor_cols": [
                ["c1", "c2", "c3"],
                None,
                [],
                ["c5"],
            ]})
    template.config.model_components_param = model_components
    regressor_cols = template.get_regressor_cols()
    assert set(regressor_cols) == {"c1", "c2", "c3", "c5"}

    # list of model components
    model_components = [
        ModelComponentsParam(
            regressors={
                "regressor_cols": [
                    ["c1", "c2", "c3"],
                    None,
                    [],
                    ["c5"],
                ]}),
        ModelComponentsParam(
            regressors={
                "regressor_cols": [
                    ["c6", "c2", "c3"],
                    None,
                    [],
                    ["c7"],
                ]})
    ]
    template.config.model_components_param = model_components
    regressor_cols = template.get_regressor_cols()
    assert set(regressor_cols) == {"c1", "c2", "c3", "c5", "c6", "c7"}


def test_get_lagged_regressor_info():
    # Without lagged regressors
    template = SimpleSilverkiteTemplate()
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


def test_apply_default_model_components_daily_1():
    template = SimpleSilverkiteTemplate()
    template.config = template.apply_forecast_config_defaults()
    template.config.model_template = "SILVERKITE_DAILY_1"
    hyperparameter_grid = template.get_hyperparameter_grid()
    assert hyperparameter_grid == [
        # Config 1
        dict(
            # Seasonality orders
            estimator__auto_seasonality=[False],
            estimator__yearly_seasonality=[8],
            estimator__quarterly_seasonality=[0],
            estimator__monthly_seasonality=[7],
            estimator__weekly_seasonality=[1],
            estimator__daily_seasonality=[0],
            # Growth and changepoints
            estimator__auto_growth=[False],
            estimator__growth_term=["linear"],
            estimator__changepoints_dict=[{
                "method": "auto",
                "resample_freq": "7D",
                "regularization_strength": 0.809,
                "potential_changepoint_distance": "7D",
                "no_changepoint_distance_from_end": "7D",
                "yearly_seasonality_order": 8,
                "yearly_seasonality_change_freq": None
            }],
            estimator__seasonality_changepoints_dict=[None],
            # Holidays
            estimator__auto_holiday=[False],
            estimator__holidays_to_model_separately=[SilverkiteHoliday.HOLIDAYS_TO_MODEL_SEPARATELY_AUTO],
            estimator__holiday_lookup_countries=[SilverkiteHoliday.HOLIDAY_LOOKUP_COUNTRIES_AUTO],
            estimator__holiday_pre_num_days=[2],
            estimator__holiday_post_num_days=[2],
            estimator__holiday_pre_post_num_dict=[None],
            estimator__daily_event_df_dict=[None],
            # Feature sets
            estimator__feature_sets_enabled=["auto"],
            # Fit algorithm
            estimator__fit_algorithm_dict=[{
                "fit_algorithm": "ridge",
                "fit_algorithm_params": None
            }],
            # Other parameters
            estimator__max_daily_seas_interaction_order=[5],
            estimator__max_weekly_seas_interaction_order=[2],
            estimator__extra_pred_cols=[[]],
            estimator__drop_pred_cols=[None],
            estimator__explicit_pred_cols=[None],
            estimator__regression_weight_col=[None],
            estimator__min_admissible_value=[None],
            estimator__max_admissible_value=[None],
            estimator__normalize_method=["zero_to_one"],
            estimator__autoreg_dict=["auto"],
            estimator__simulation_num=[10],
            estimator__fast_simulation=[False],
            estimator__regressor_cols=[[]],
            estimator__lagged_regressor_dict=[None],
            estimator__uncertainty_dict=[None],
            estimator__time_properties=[None],
            estimator__origin_for_time_vars=[None],
            estimator__train_test_thresh=[None],
            estimator__training_fraction=[None]
        ),
        # Config 2
        dict(
            # Seasonality orders
            estimator__auto_seasonality=[False],
            estimator__yearly_seasonality=[1],
            estimator__quarterly_seasonality=[0],
            estimator__monthly_seasonality=[4],
            estimator__weekly_seasonality=[6],
            estimator__daily_seasonality=[0],
            # Growth and changepoints
            estimator__auto_growth=[False],
            estimator__growth_term=["linear"],
            estimator__changepoints_dict=[{
                "method": "auto",
                "resample_freq": "7D",
                "regularization_strength": 0.624,
                "potential_changepoint_distance": "7D",
                "no_changepoint_distance_from_end": "17D",
                "yearly_seasonality_order": 1,
                "yearly_seasonality_change_freq": None
            }],
            estimator__seasonality_changepoints_dict=[None],
            # Holidays
            estimator__auto_holiday=[False],
            estimator__holidays_to_model_separately=[SilverkiteHoliday.HOLIDAYS_TO_MODEL_SEPARATELY_AUTO],
            estimator__holiday_lookup_countries=[SilverkiteHoliday.HOLIDAY_LOOKUP_COUNTRIES_AUTO],
            estimator__holiday_pre_num_days=[2],
            estimator__holiday_post_num_days=[2],
            estimator__holiday_pre_post_num_dict=[None],
            estimator__daily_event_df_dict=[None],
            # Feature sets
            estimator__feature_sets_enabled=["auto"],
            # Fit algorithm
            estimator__fit_algorithm_dict=[{
                "fit_algorithm": "ridge",
                "fit_algorithm_params": None
            }],
            # Other parameters
            estimator__max_daily_seas_interaction_order=[5],
            estimator__max_weekly_seas_interaction_order=[2],
            estimator__extra_pred_cols=[[]],
            estimator__drop_pred_cols=[None],
            estimator__explicit_pred_cols=[None],
            estimator__regression_weight_col=[None],
            estimator__min_admissible_value=[None],
            estimator__max_admissible_value=[None],
            estimator__normalize_method=["zero_to_one"],
            estimator__autoreg_dict=["auto"],
            estimator__simulation_num=[10],
            estimator__fast_simulation=[False],
            estimator__regressor_cols=[[]],
            estimator__lagged_regressor_dict=[None],
            estimator__uncertainty_dict=[None],
            estimator__time_properties=[None],
            estimator__origin_for_time_vars=[None],
            estimator__train_test_thresh=[None],
            estimator__training_fraction=[None]
        ),
        # Config 3
        dict(
            # Seasonality orders
            estimator__auto_seasonality=[False],
            estimator__yearly_seasonality=[40],
            estimator__quarterly_seasonality=[0],
            estimator__monthly_seasonality=[0],
            estimator__weekly_seasonality=[2],
            estimator__daily_seasonality=[0],
            # Growth and changepoints
            estimator__auto_growth=[False],
            estimator__growth_term=["linear"],
            estimator__changepoints_dict=[{
                "method": "auto",
                "resample_freq": "7D",
                "regularization_strength": 0.590,
                "potential_changepoint_distance": "7D",
                "no_changepoint_distance_from_end": "8D",
                "yearly_seasonality_order": 40,
                "yearly_seasonality_change_freq": None
            }],
            estimator__seasonality_changepoints_dict=[None],
            # Holidays
            estimator__auto_holiday=[False],
            estimator__holidays_to_model_separately=[SilverkiteHoliday.HOLIDAYS_TO_MODEL_SEPARATELY_AUTO],
            estimator__holiday_lookup_countries=[SilverkiteHoliday.HOLIDAY_LOOKUP_COUNTRIES_AUTO],
            estimator__holiday_pre_num_days=[2],
            estimator__holiday_post_num_days=[2],
            estimator__holiday_pre_post_num_dict=[None],
            estimator__daily_event_df_dict=[None],
            # Feature sets
            estimator__feature_sets_enabled=["auto"],
            # Fit algorithm
            estimator__fit_algorithm_dict=[{
                "fit_algorithm": "ridge",
                "fit_algorithm_params": None
            }],
            # Other parameters
            estimator__max_daily_seas_interaction_order=[5],
            estimator__max_weekly_seas_interaction_order=[2],
            estimator__extra_pred_cols=[[]],
            estimator__drop_pred_cols=[None],
            estimator__explicit_pred_cols=[None],
            estimator__regression_weight_col=[None],
            estimator__min_admissible_value=[None],
            estimator__max_admissible_value=[None],
            estimator__normalize_method=["zero_to_one"],
            estimator__autoreg_dict=["auto"],
            estimator__simulation_num=[10],
            estimator__fast_simulation=[False],
            estimator__regressor_cols=[[]],
            estimator__lagged_regressor_dict=[None],
            estimator__uncertainty_dict=[None],
            estimator__time_properties=[None],
            estimator__origin_for_time_vars=[None],
            estimator__train_test_thresh=[None],
            estimator__training_fraction=[None]
        )
    ]


def test_apply_default_model_components_daily_90():
    template = SimpleSilverkiteTemplate()
    template.config = template.apply_forecast_config_defaults()
    template.config.model_template = "SILVERKITE_DAILY_90"
    hyperparameter_grid = template.get_hyperparameter_grid()
    assert hyperparameter_grid == [
        # Config 1
        # Light seasonality, light trend changepoints, separate +- 2 days holidays.
        # Auto feature sets, linear regression.
        dict(
            estimator__time_properties=[None],
            estimator__origin_for_time_vars=[None],
            estimator__train_test_thresh=[None],
            estimator__training_fraction=[None],
            # Seasonality orders
            estimator__auto_seasonality=[False],
            estimator__yearly_seasonality=[8],
            estimator__quarterly_seasonality=[3],
            estimator__monthly_seasonality=[2],
            estimator__weekly_seasonality=[3],
            estimator__daily_seasonality=[0],
            # Growth and changepoints
            estimator__auto_growth=[False],
            estimator__growth_term=["linear"],
            estimator__changepoints_dict=[{
                "method": "auto",
                "resample_freq": "7D",
                "regularization_strength": 0.6,
                "potential_changepoint_distance": "15D",
                "no_changepoint_distance_from_end": "90D",
                "yearly_seasonality_order": 15,
                "yearly_seasonality_change_freq": None
            }],
            estimator__seasonality_changepoints_dict=[None],
            # Holidays
            estimator__auto_holiday=[False],
            estimator__holidays_to_model_separately=["auto"],
            estimator__holiday_lookup_countries=["auto"],
            estimator__holiday_pre_num_days=[2],
            estimator__holiday_post_num_days=[2],
            estimator__holiday_pre_post_num_dict=[None],
            estimator__daily_event_df_dict=[None],
            # Feature sets
            estimator__feature_sets_enabled=["auto"],
            # Fit algorithm
            estimator__fit_algorithm_dict=[{
                "fit_algorithm": "linear",
                "fit_algorithm_params": None
            }],
            # Other parameters
            estimator__max_daily_seas_interaction_order=[0],
            estimator__max_weekly_seas_interaction_order=[2],
            estimator__extra_pred_cols=[[]],
            estimator__drop_pred_cols=[None],
            estimator__explicit_pred_cols=[None],
            estimator__regression_weight_col=[None],
            estimator__min_admissible_value=[None],
            estimator__max_admissible_value=[None],
            estimator__normalize_method=["zero_to_one"],
            estimator__autoreg_dict=[None],
            estimator__simulation_num=[10],
            estimator__fast_simulation=[False],
            estimator__regressor_cols=[[]],
            estimator__lagged_regressor_dict=[None],
            estimator__uncertainty_dict=[None]
        ),
        # Config 2
        # Light seasonality, no trend changepoints, separate +- 2 days holidays.
        # Auto feature sets, linear regression.
        dict(
            estimator__time_properties=[None],
            estimator__origin_for_time_vars=[None],
            estimator__train_test_thresh=[None],
            estimator__training_fraction=[None],
            # Seasonality orders
            estimator__auto_seasonality=[False],
            estimator__yearly_seasonality=[8],
            estimator__quarterly_seasonality=[3],
            estimator__monthly_seasonality=[2],
            estimator__weekly_seasonality=[3],
            estimator__daily_seasonality=[0],
            # Growth and changepoints
            estimator__auto_growth=[False],
            estimator__growth_term=["linear"],
            estimator__changepoints_dict=[None],
            estimator__seasonality_changepoints_dict=[None],
            # Holidays
            estimator__auto_holiday=[False],
            estimator__holidays_to_model_separately=["auto"],
            estimator__holiday_lookup_countries=["auto"],
            estimator__holiday_pre_num_days=[2],
            estimator__holiday_post_num_days=[2],
            estimator__holiday_pre_post_num_dict=[None],
            estimator__daily_event_df_dict=[None],
            # Feature sets
            estimator__feature_sets_enabled=["auto"],
            # Fit algorithm
            estimator__fit_algorithm_dict=[{
                "fit_algorithm": "linear",
                "fit_algorithm_params": None
            }],
            # Other parameters
            estimator__max_daily_seas_interaction_order=[0],
            estimator__max_weekly_seas_interaction_order=[2],
            estimator__extra_pred_cols=[[]],
            estimator__drop_pred_cols=[None],
            estimator__explicit_pred_cols=[None],
            estimator__regression_weight_col=[None],
            estimator__min_admissible_value=[None],
            estimator__max_admissible_value=[None],
            estimator__normalize_method=["zero_to_one"],
            estimator__autoreg_dict=[None],
            estimator__simulation_num=[10],
            estimator__fast_simulation=[False],
            estimator__regressor_cols=[[]],
            estimator__lagged_regressor_dict=[None],
            estimator__uncertainty_dict=[None]
        ),
        # Config 3
        # Light seasonality, light trend changepoints, separate +- 2 days holidays.
        # Auto feature sets, ridge regression.
        dict(
            estimator__time_properties=[None],
            estimator__origin_for_time_vars=[None],
            estimator__train_test_thresh=[None],
            estimator__training_fraction=[None],
            # Seasonality orders
            estimator__auto_seasonality=[False],
            estimator__yearly_seasonality=[8],
            estimator__quarterly_seasonality=[3],
            estimator__monthly_seasonality=[2],
            estimator__weekly_seasonality=[3],
            estimator__daily_seasonality=[0],
            # Growth and changepoints
            estimator__auto_growth=[False],
            estimator__growth_term=["linear"],
            estimator__changepoints_dict=[{
                "method": "auto",
                "resample_freq": "7D",
                "regularization_strength": 0.6,
                "potential_changepoint_distance": "15D",
                "no_changepoint_distance_from_end": "90D",
                "yearly_seasonality_order": 15,
                "yearly_seasonality_change_freq": None
            }],
            estimator__seasonality_changepoints_dict=[None],
            # Holidays
            estimator__auto_holiday=[False],
            estimator__holidays_to_model_separately=["auto"],
            estimator__holiday_lookup_countries=["auto"],
            estimator__holiday_pre_num_days=[2],
            estimator__holiday_post_num_days=[2],
            estimator__holiday_pre_post_num_dict=[None],
            estimator__daily_event_df_dict=[None],
            # Feature sets
            estimator__feature_sets_enabled=["auto"],
            # Fit algorithm
            estimator__fit_algorithm_dict=[{
                "fit_algorithm": "ridge",
                "fit_algorithm_params": None
            }],
            # Other parameters
            estimator__max_daily_seas_interaction_order=[0],
            estimator__max_weekly_seas_interaction_order=[2],
            estimator__extra_pred_cols=[[]],
            estimator__drop_pred_cols=[None],
            estimator__explicit_pred_cols=[None],
            estimator__regression_weight_col=[None],
            estimator__min_admissible_value=[None],
            estimator__max_admissible_value=[None],
            estimator__normalize_method=["zero_to_one"],
            estimator__autoreg_dict=[None],
            estimator__simulation_num=[10],
            estimator__fast_simulation=[False],
            estimator__regressor_cols=[[]],
            estimator__lagged_regressor_dict=[None],
            estimator__uncertainty_dict=[None]
        ),
        # Config 4
        # Year/week seasonality, light trend changepoints, separate +- 4 days holidays.
        # Auto feature sets, ridge regression.
        dict(
            estimator__time_properties=[None],
            estimator__origin_for_time_vars=[None],
            estimator__train_test_thresh=[None],
            estimator__training_fraction=[None],
            # Seasonality orders
            estimator__auto_seasonality=[False],
            estimator__yearly_seasonality=[15],
            estimator__quarterly_seasonality=[0],
            estimator__monthly_seasonality=[0],
            estimator__weekly_seasonality=[3],
            estimator__daily_seasonality=[0],
            # Growth and changepoints
            estimator__auto_growth=[False],
            estimator__growth_term=["linear"],
            estimator__changepoints_dict=[{
                "method": "auto",
                "resample_freq": "7D",
                "regularization_strength": 0.6,
                "potential_changepoint_distance": "15D",
                "no_changepoint_distance_from_end": "90D",
                "yearly_seasonality_order": 15,
                "yearly_seasonality_change_freq": None
            }],
            estimator__seasonality_changepoints_dict=[None],
            # Holidays
            estimator__auto_holiday=[False],
            estimator__holidays_to_model_separately=["auto"],
            estimator__holiday_lookup_countries=["auto"],
            estimator__holiday_pre_num_days=[4],
            estimator__holiday_post_num_days=[4],
            estimator__holiday_pre_post_num_dict=[None],
            estimator__daily_event_df_dict=[None],
            # Feature sets
            estimator__feature_sets_enabled=["auto"],
            # Fit algorithm
            estimator__fit_algorithm_dict=[{
                "fit_algorithm": "ridge",
                "fit_algorithm_params": None
            }],
            # Other parameters
            estimator__max_daily_seas_interaction_order=[0],
            estimator__max_weekly_seas_interaction_order=[2],
            estimator__extra_pred_cols=[[]],
            estimator__drop_pred_cols=[None],
            estimator__explicit_pred_cols=[None],
            estimator__regression_weight_col=[None],
            estimator__min_admissible_value=[None],
            estimator__max_admissible_value=[None],
            estimator__normalize_method=["zero_to_one"],
            estimator__autoreg_dict=[None],
            estimator__simulation_num=[10],
            estimator__fast_simulation=[False],
            estimator__regressor_cols=[[]],
            estimator__lagged_regressor_dict=[None],
            estimator__uncertainty_dict=[None]
        )
    ]


def test_apply_default_model_components_weekly():
    template = SimpleSilverkiteTemplate()
    template.config = template.apply_forecast_config_defaults()
    template.config.model_template = "SILVERKITE_WEEKLY"
    hyperparameter_grid = template.get_hyperparameter_grid()
    assert hyperparameter_grid == [
        # Config 1
        # For weekly data, normal seasonality up to yearly, no trend changepoints,
        # no holiday, no feature sets and linear fit algorithm.
        dict(
            estimator__time_properties=[None],
            estimator__origin_for_time_vars=[None],
            estimator__train_test_thresh=[None],
            estimator__training_fraction=[None],
            # Seasonality orders
            estimator__auto_seasonality=[False],
            estimator__yearly_seasonality=[15],
            estimator__quarterly_seasonality=[0],
            estimator__monthly_seasonality=[0],
            estimator__weekly_seasonality=[0],
            estimator__daily_seasonality=[0],
            # Growth and changepoints
            estimator__auto_growth=[False],
            estimator__growth_term=["linear"],
            estimator__changepoints_dict=[None],
            estimator__seasonality_changepoints_dict=[None],
            # Holidays
            estimator__auto_holiday=[False],
            estimator__holidays_to_model_separately=[[]],
            estimator__holiday_lookup_countries=[[]],
            estimator__holiday_pre_num_days=[0],
            estimator__holiday_post_num_days=[0],
            estimator__holiday_pre_post_num_dict=[None],
            estimator__daily_event_df_dict=[None],
            # Feature sets
            estimator__feature_sets_enabled=[False],
            # Fit algorithm
            estimator__fit_algorithm_dict=[{
                "fit_algorithm": "linear",
                "fit_algorithm_params": None
            }],
            # Other parameters
            estimator__max_daily_seas_interaction_order=[0],
            estimator__max_weekly_seas_interaction_order=[0],
            estimator__extra_pred_cols=[[]],
            estimator__drop_pred_cols=[None],
            estimator__explicit_pred_cols=[None],
            estimator__regression_weight_col=[None],
            estimator__min_admissible_value=[None],
            estimator__max_admissible_value=[None],
            estimator__normalize_method=["zero_to_one"],
            estimator__autoreg_dict=[None],
            estimator__simulation_num=[10],
            estimator__fast_simulation=[False],
            estimator__regressor_cols=[[]],
            estimator__lagged_regressor_dict=[None],
            estimator__uncertainty_dict=[None]
        ),
        # Config 2
        # For weekly data, normal seasonality up to yearly, light trend changepoints,
        # no holiday, no feature sets and linear fit algorithm.
        dict(
            estimator__time_properties=[None],
            estimator__origin_for_time_vars=[None],
            estimator__train_test_thresh=[None],
            estimator__training_fraction=[None],
            # Seasonality orders
            estimator__auto_seasonality=[False],
            estimator__yearly_seasonality=[15],
            estimator__quarterly_seasonality=[0],
            estimator__monthly_seasonality=[0],
            estimator__weekly_seasonality=[0],
            estimator__daily_seasonality=[0],
            # Growth and changepoints
            estimator__auto_growth=[False],
            estimator__growth_term=["linear"],
            estimator__changepoints_dict=[dict(
                method="auto",
                resample_freq="7D",
                regularization_strength=0.6,
                potential_changepoint_distance="14D",
                no_changepoint_distance_from_end="180D",
                yearly_seasonality_order=15,
                yearly_seasonality_change_freq=None
            )],
            estimator__seasonality_changepoints_dict=[None],
            # Holidays
            estimator__auto_holiday=[False],
            estimator__holidays_to_model_separately=[[]],
            estimator__holiday_lookup_countries=[[]],
            estimator__holiday_pre_num_days=[0],
            estimator__holiday_post_num_days=[0],
            estimator__holiday_pre_post_num_dict=[None],
            estimator__daily_event_df_dict=[None],
            # Feature sets
            estimator__feature_sets_enabled=[False],
            # Fit algorithm
            estimator__fit_algorithm_dict=[{
                "fit_algorithm": "linear",
                "fit_algorithm_params": None
            }],
            # Other parameters
            estimator__max_daily_seas_interaction_order=[0],
            estimator__max_weekly_seas_interaction_order=[0],
            estimator__extra_pred_cols=[[]],
            estimator__drop_pred_cols=[None],
            estimator__explicit_pred_cols=[None],
            estimator__regression_weight_col=[None],
            estimator__min_admissible_value=[None],
            estimator__max_admissible_value=[None],
            estimator__normalize_method=["zero_to_one"],
            estimator__autoreg_dict=[None],
            estimator__simulation_num=[10],
            estimator__fast_simulation=[False],
            estimator__regressor_cols=[[]],
            estimator__lagged_regressor_dict=[None],
            estimator__uncertainty_dict=[None]
        ),
        # Config 3
        # For weekly data, heavy seasonality up to yearly, normal trend changepoints,
        # no holiday, no feature sets and ridge fit algorithm.
        dict(
            estimator__time_properties=[None],
            estimator__origin_for_time_vars=[None],
            estimator__train_test_thresh=[None],
            estimator__training_fraction=[None],
            # Seasonality orders
            estimator__auto_seasonality=[False],
            estimator__yearly_seasonality=[25],
            estimator__quarterly_seasonality=[0],
            estimator__monthly_seasonality=[0],
            estimator__weekly_seasonality=[0],
            estimator__daily_seasonality=[0],
            # Growth and changepoints
            estimator__auto_growth=[False],
            estimator__growth_term=["linear"],
            estimator__changepoints_dict=[{
                "method": "auto",
                "resample_freq": "7D",
                "regularization_strength": 0.5,
                "potential_changepoint_distance": "14D",
                "no_changepoint_distance_from_end": "180D",
                "yearly_seasonality_order": 15,
                "yearly_seasonality_change_freq": "365D"
            }],
            estimator__seasonality_changepoints_dict=[None],
            # Holidays
            estimator__auto_holiday=[False],
            estimator__holidays_to_model_separately=[[]],
            estimator__holiday_lookup_countries=[[]],
            estimator__holiday_pre_num_days=[0],
            estimator__holiday_post_num_days=[0],
            estimator__holiday_pre_post_num_dict=[None],
            estimator__daily_event_df_dict=[None],
            # Feature sets
            estimator__feature_sets_enabled=[False],
            # Fit algorithm
            estimator__fit_algorithm_dict=[{
                "fit_algorithm": "ridge",
                "fit_algorithm_params": None
            }],
            # Other parameters
            estimator__max_daily_seas_interaction_order=[0],
            estimator__max_weekly_seas_interaction_order=[0],
            estimator__extra_pred_cols=[[]],
            estimator__drop_pred_cols=[None],
            estimator__explicit_pred_cols=[None],
            estimator__regression_weight_col=[None],
            estimator__min_admissible_value=[None],
            estimator__max_admissible_value=[None],
            estimator__normalize_method=["zero_to_one"],
            estimator__autoreg_dict=[None],
            estimator__simulation_num=[10],
            estimator__fast_simulation=[False],
            estimator__regressor_cols=[[]],
            estimator__lagged_regressor_dict=[None],
            estimator__uncertainty_dict=[None]
        ),
        # Config 4
        # For weekly data, heavy seasonality up to yearly, light trend changepoints,
        # no holiday, no feature sets and ridge fit algorithm.
        dict(
            estimator__time_properties=[None],
            estimator__origin_for_time_vars=[None],
            estimator__train_test_thresh=[None],
            estimator__training_fraction=[None],
            # Seasonality orders
            estimator__auto_seasonality=[False],
            estimator__yearly_seasonality=[25],
            estimator__quarterly_seasonality=[0],
            estimator__monthly_seasonality=[0],
            estimator__weekly_seasonality=[0],
            estimator__daily_seasonality=[0],
            # Growth and changepoints
            estimator__auto_growth=[False],
            estimator__growth_term=["linear"],
            estimator__changepoints_dict=[dict(
                method="auto",
                resample_freq="7D",
                regularization_strength=0.6,
                potential_changepoint_distance="14D",
                no_changepoint_distance_from_end="180D",
                yearly_seasonality_order=15,
                yearly_seasonality_change_freq=None
            )],
            estimator__seasonality_changepoints_dict=[None],
            # Holidays
            estimator__auto_holiday=[False],
            estimator__holidays_to_model_separately=[[]],
            estimator__holiday_lookup_countries=[[]],
            estimator__holiday_pre_num_days=[0],
            estimator__holiday_post_num_days=[0],
            estimator__holiday_pre_post_num_dict=[None],
            estimator__daily_event_df_dict=[None],
            # Feature sets
            estimator__feature_sets_enabled=[False],
            # Fit algorithm
            estimator__fit_algorithm_dict=[{
                "fit_algorithm": "ridge",
                "fit_algorithm_params": None
            }],
            # Other parameters
            estimator__max_daily_seas_interaction_order=[0],
            estimator__max_weekly_seas_interaction_order=[0],
            estimator__extra_pred_cols=[[]],
            estimator__drop_pred_cols=[None],
            estimator__explicit_pred_cols=[None],
            estimator__regression_weight_col=[None],
            estimator__min_admissible_value=[None],
            estimator__max_admissible_value=[None],
            estimator__normalize_method=["zero_to_one"],
            estimator__autoreg_dict=[None],
            estimator__simulation_num=[10],
            estimator__fast_simulation=[False],
            estimator__regressor_cols=[[]],
            estimator__lagged_regressor_dict=[None],
            estimator__uncertainty_dict=[None]
        )
    ]


def test_apply_default_model_template_hourly():
    template = SimpleSilverkiteTemplate()
    template.config = template.apply_forecast_config_defaults()
    template.config.model_template = "SILVERKITE_HOURLY_1"
    template.get_hyperparameter_grid()
    template.config.model_template = "SILVERKITE_HOURLY_24"
    template.get_hyperparameter_grid()
    template.config.model_template = "SILVERKITE_HOURLY_168"
    template.get_hyperparameter_grid()
    template.config.model_template = "SILVERKITE_HOURLY_336"
    template.get_hyperparameter_grid()


def test_get_simple_silverkite_hyperparameter_grid(silverkite, silverkite_diagnostics):
    """Tests get_silverkite_hyperparameter_grid"""
    # tests default values, unpacking, conversion to list
    silverkite = SimpleSilverkiteForecast()
    silverkite_diagnostics = SilverkiteDiagnostics()
    template = SimpleSilverkiteTemplate()
    template.config = template.apply_forecast_config_defaults()
    hyperparameter_grid = template.get_hyperparameter_grid()
    expected_grid = {
        "estimator__time_properties": [None],
        "estimator__origin_for_time_vars": [None],
        "estimator__train_test_thresh": [None],
        "estimator__training_fraction": [None],
        "estimator__fit_algorithm_dict": [{"fit_algorithm": "ridge", "fit_algorithm_params": None}],
        "estimator__auto_holiday": [False],
        "estimator__holidays_to_model_separately": ["auto"],
        "estimator__holiday_lookup_countries": ["auto"],
        "estimator__holiday_pre_num_days": [2],
        "estimator__holiday_post_num_days": [2],
        "estimator__holiday_pre_post_num_dict": [None],
        "estimator__daily_event_df_dict": [None],
        "estimator__auto_growth": [False],
        "estimator__changepoints_dict": [{
            "method": "auto",
            "yearly_seasonality_order": 15,
            "resample_freq": "3D",
            "regularization_strength": 0.6,
            "actual_changepoint_min_distance": "30D",
            "potential_changepoint_distance": "15D",
            "no_changepoint_distance_from_end": "90D"
        }],
        "estimator__seasonality_changepoints_dict": [None],
        "estimator__auto_seasonality": [False],
        "estimator__yearly_seasonality": ["auto"],
        "estimator__quarterly_seasonality": ["auto"],
        "estimator__monthly_seasonality": ["auto"],
        "estimator__weekly_seasonality": ["auto"],
        "estimator__daily_seasonality": ["auto"],
        "estimator__max_daily_seas_interaction_order": [5],
        "estimator__max_weekly_seas_interaction_order": [2],
        "estimator__autoreg_dict": ["auto"],
        "estimator__simulation_num": [10],
        "estimator__fast_simulation": [False],
        "estimator__min_admissible_value": [None],
        "estimator__max_admissible_value": [None],
        "estimator__normalize_method": ["zero_to_one"],
        "estimator__uncertainty_dict": [None],
        "estimator__growth_term": ["linear"],
        "estimator__regressor_cols": [[]],
        "estimator__lagged_regressor_dict": [None],
        "estimator__feature_sets_enabled": ["auto"],
        "estimator__extra_pred_cols": [[]],
        "estimator__drop_pred_cols": [None],
        "estimator__explicit_pred_cols": [None],
        "estimator__regression_weight_col": [None]
    }
    assert_equal(hyperparameter_grid, expected_grid)

    # able to set parameters and use hyperparameter_override to override
    daily_event_df_dict = silverkite._SimpleSilverkiteForecast__get_silverkite_holidays(
        holiday_lookup_countries=["India"],
        holidays_to_model_separately=["Easter Sunday", "Republic Day"],
        start_year=2017,
        end_year=2025,
        pre_num=2,
        post_num=2)
    model_components = ModelComponentsParam(
        events={
            "holiday_lookup_countries": ["UnitedStates"],
            "holidays_to_model_separately": SilverkiteHoliday.ALL_HOLIDAYS_IN_COUNTRIES,
            "holiday_pre_post_num_dict": {"New Year's Day": (7, 3)},
            "daily_event_df_dict": daily_event_df_dict
        },
        seasonality={
            "yearly_seasonality": True,
            "weekly_seasonality": False,
        },
        regressors={
            "regressor_cols": ["reg1", "reg2"]
        },
        custom={
            "extra_pred_cols": ["some_column"]
        },
        hyperparameter_override={
            "input__response__null__max_frac": 0.1,
            "estimator__silverkite": silverkite,
            "estimator__silverkite_diagnostics": silverkite_diagnostics,
            "estimator__growth_term": ["override_estimator__growth_term"],
            "estimator__extra_pred_cols": ["override_estimator__extra_pred_cols"]
        }
    )
    # Checks the automatic list conversion via `dictionaries_values_to_lists`,
    # and the `time_properties` parameter
    data = generate_df_for_tests(freq="D", periods=10)
    df = data["df"]
    time_properties = get_forecast_time_properties(df=df)
    template = SimpleSilverkiteTemplate()
    template.config = template.apply_forecast_config_defaults()
    template.config.model_components_param = model_components
    template.time_properties = time_properties
    hyperparameter_grid = template.get_hyperparameter_grid()
    assert time_properties["origin_for_time_vars"] is not None
    updated_grid = expected_grid.copy()
    updated_grid["estimator__time_properties"] = [time_properties]
    updated_grid["estimator__holiday_lookup_countries"] = [["UnitedStates"]]
    updated_grid["estimator__holidays_to_model_separately"] = [SilverkiteHoliday.ALL_HOLIDAYS_IN_COUNTRIES]
    updated_grid["estimator__holiday_pre_post_num_dict"] = [{"New Year's Day": (7, 3)}]
    updated_grid["estimator__daily_event_df_dict"] = [daily_event_df_dict]
    updated_grid["estimator__yearly_seasonality"] = [True]
    updated_grid["estimator__weekly_seasonality"] = [False]
    updated_grid["input__response__null__max_frac"] = [0.1]
    updated_grid["estimator__silverkite"] = [silverkite]
    updated_grid["estimator__silverkite_diagnostics"] = [silverkite_diagnostics]
    updated_grid["estimator__growth_term"] = ["override_estimator__growth_term"]
    updated_grid["estimator__regressor_cols"] = [["reg1", "reg2"]]
    updated_grid["estimator__extra_pred_cols"] = [["override_estimator__extra_pred_cols"]]
    assert_equal(hyperparameter_grid, updated_grid)

    # hyperparameter_override can be a list of dictionaries/None
    model_components = ModelComponentsParam(
        hyperparameter_override=[
            {
                "input__response__null__max_frac": 0.1,
                "estimator__growth_term": ["override_estimator__growth_term"],
                "estimator__extra_pred_cols": ["override_estimator__extra_pred_cols"]
            },
            {},
            {
                "estimator__extra_pred_cols": ["val1", "val2"]
            },
            None
        ]
    )
    template = SimpleSilverkiteTemplate()
    template.config = template.apply_forecast_config_defaults()
    template.config.model_components_param = model_components
    hyperparameter_grid = template.get_hyperparameter_grid()
    updated_grid1 = expected_grid.copy()
    updated_grid1["input__response__null__max_frac"] = [0.1]
    updated_grid1["estimator__growth_term"] = ["override_estimator__growth_term"]
    updated_grid1["estimator__extra_pred_cols"] = [["override_estimator__extra_pred_cols"]]
    updated_grid2 = expected_grid.copy()
    updated_grid2["estimator__extra_pred_cols"] = [["val1", "val2"]]
    expected_grid = [
        updated_grid1,
        expected_grid,
        updated_grid2]
    assert_equal(hyperparameter_grid, expected_grid)

    # Tests list of ``ModelComponentsParam``.
    model_components = [
        ModelComponentsParam(
            seasonality={
                "yearly_seasonality": 5
            },
            hyperparameter_override=[
                {
                    "estimator__daily_seasonality": 2
                },
                {
                    "estimator__daily_seasonality": 3
                }
            ]
        ),
        ModelComponentsParam(
            seasonality={
                "yearly_seasonality": 6
            }
        )
    ]
    template = SimpleSilverkiteTemplate()
    template.config = template.apply_forecast_config_defaults()
    template.config.model_template = ["SILVERKITE_EMPTY", "DAILY"]
    template.config.model_components_param = model_components
    hyperparameter_grid = template.get_hyperparameter_grid()
    # There should be 6 dictionaries.
    assert len(hyperparameter_grid) == 6
    assert hyperparameter_grid[0]["estimator__yearly_seasonality"] == [5]
    assert hyperparameter_grid[0]["estimator__daily_seasonality"] == [2]
    assert hyperparameter_grid[1]["estimator__yearly_seasonality"] == [5]
    assert hyperparameter_grid[1]["estimator__daily_seasonality"] == [3]
    assert hyperparameter_grid[2]["estimator__yearly_seasonality"] == [5]
    assert hyperparameter_grid[2]["estimator__daily_seasonality"] == [2]
    assert hyperparameter_grid[3]["estimator__yearly_seasonality"] == [5]
    assert hyperparameter_grid[3]["estimator__daily_seasonality"] == [3]
    assert hyperparameter_grid[4]["estimator__yearly_seasonality"] == [6]
    assert hyperparameter_grid[5]["estimator__yearly_seasonality"] == [6]


def test_simple_silverkite_template():
    """"Tests simple_silverkite_template"""
    data = generate_df_for_tests(freq="D", periods=10)
    df = data["df"]
    template = SimpleSilverkiteTemplate()
    config = ForecastConfig(model_template="SILVERKITE")
    params = template.apply_template_for_pipeline_params(
        df=df,
        config=config
    )
    assert config == ForecastConfig(model_template="SILVERKITE")  # not modified
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
    actual_time_properties = params["hyperparameter_grid"]["estimator__time_properties"][0]
    assert_equal(actual_time_properties, template.time_properties)
    assert actual_time_properties["origin_for_time_vars"] is not None


def test_simple_silverkite_template_custom():
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
        train_end_date=datetime.datetime(2019, 2, 1),  # last date with value is 2019/2/26
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

    model_components = ModelComponentsParam(
        seasonality={
            "yearly_seasonality": True,
            "weekly_seasonality": False
        },
        growth={
            "growth_term": "quadratic"
        },
        events={
            "holidays_to_model_separately": MySilverkiteHoliday.ALL_HOLIDAYS_IN_COUNTRIES,  # custom value
            "holiday_lookup_countries": ["UnitedStates"],
            "holiday_pre_num_days": 3,
        },
        changepoints={
            "changepoints_dict": {
                "method": "uniform",
                "n_changepoints": 20,
            }
        },
        autoregression={
            "autoreg_dict": {
                "dummy_key": "test_value"
            }
        },
        regressors={
            "regressor_cols": [
                "regressor1",
                "regressor2",
                "regressor3",
                "regressor_bool",
                "regressor_categ"]
        },
        lagged_regressors={
            "lagged_regressor_dict": {
                "dummy_key": "test_value"
            }
        },
        uncertainty={
            "uncertainty_dict": "auto",
        },
        hyperparameter_override={
            "input__response__null__max_frac": 0.1
        },
        custom={
            "fit_algorithm_dict": {
                "fit_algorithm": "ridge",
                "fit_algorithm_params": {"normalize": True},
            },
            "feature_sets_enabled": False
        }
    )
    computation = ComputationParam(
        hyperparameter_budget=10,
        n_jobs=None,
        verbose=1
    )
    forecast_horizon = 20
    coverage = 0.7
    silverkite_constants = MySilverkiteConstant()
    silverkite = SimpleSilverkiteForecast(constants=silverkite_constants)
    estimator = SimpleSilverkiteEstimator(silverkite=silverkite)
    template = SimpleSilverkiteTemplate(estimator=estimator)
    params = template.apply_template_for_pipeline_params(
        df=df,
        config=ForecastConfig(
            model_template=ModelTemplateEnum.SILVERKITE.name,
            metadata_param=metadata,
            forecast_horizon=forecast_horizon,
            coverage=coverage,
            evaluation_metric_param=evaluation_metric,
            evaluation_period_param=evaluation_period,
            model_components_param=model_components,
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
    assert_basic_pipeline_equal(pipeline, template.pipeline)
    assert pipeline.steps[-1][-1].silverkite is not silverkite  # a copy is created by get_basic_pipeline clone
    assert pipeline.steps[-1][-1].silverkite._silverkite_holiday == silverkite_constants.get_silverkite_holiday()
    assert pipeline.steps[-1][-1].silverkite._silverkite_holiday.HOLIDAY_LOOKUP_COUNTRIES_AUTO == ("UnitedStates")  # custom estimator is used
    assert_equal(params, expected_params)
    actual_time_properties = params["hyperparameter_grid"]["estimator__time_properties"][0]
    assert_equal(actual_time_properties, template.time_properties)
    assert actual_time_properties["origin_for_time_vars"] is not None


# The following tests run `SimpleSilverkiteTemplate` through the pipeline.
# They ensure `forecast_pipeline` and `SimpleSilverkiteEstimator` can interpret the parameters
# passed directly and through the `hyperparameter_grid`.
def test_run_template_1():
    """Tests:
     - no coverage
     - hourly data (2+ years)
     - default `hyperparameter_grid` (all interaction terms enabled)
    """
    model_components = ModelComponentsParam(
        custom={
            "fit_algorithm_dict": {
                "fit_algorithm": "linear",
            }
        }
    )
    data = generate_df_for_tests(
        freq="H",
        periods=700 * 24)
    df = data["train_df"]
    forecast_horizon = data["test_df"].shape[0]

    config = ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        forecast_horizon=forecast_horizon,
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
        assert result.backtest.test_evaluation[rmse] == pytest.approx(2.120, rel=0.02)
        assert result.backtest.test_evaluation[q80] == pytest.approx(0.863, rel=0.02)
        assert result.forecast.train_evaluation[rmse] == pytest.approx(1.975, rel=1e-2)
        assert result.forecast.train_evaluation[q80] == pytest.approx(0.786, rel=1e-2)
        check_forecast_pipeline_result(
            result,
            coverage=None,
            strategy=None,
            score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name,
            greater_is_better=False)


def test_run_template_2():
    """Tests:
     - coverage
     - daily data
     - default `hyperparameter_grid` (all interaction terms enabled)
    """
    # sets random state for consistent comparison
    model_components = ModelComponentsParam(
        custom={
            "fit_algorithm_dict": {
                "fit_algorithm": "sgd",
                "fit_algorithm_params": {"random_state": 1234}
            }
        }
    )

    data = generate_df_for_tests(
        freq="D",
        periods=90)
    df = data["train_df"]
    forecast_horizon = data["test_df"].shape[0]
    coverage = 0.90

    config = ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        forecast_horizon=forecast_horizon,
        coverage=coverage,
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
        assert result.backtest.test_evaluation[rmse] == pytest.approx(1.968, rel=1e-2)
        assert result.backtest.test_evaluation[q80] == pytest.approx(0.573, rel=1e-2)
        assert result.forecast.train_evaluation[rmse] == pytest.approx(1.953, rel=1e-2)
        assert result.forecast.train_evaluation[q80] == pytest.approx(0.784, rel=1e-2)
        check_forecast_pipeline_result(
            result,
            coverage=coverage,
            strategy=None,
            score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name,
            greater_is_better=False)


def test_run_template_3():
    """Tests:
     - numeric and categorical regressors
     - coverage
     - custom parameters
     - null model
     - weekly data
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

    config = ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        forecast_horizon=forecast_horizon,
        coverage=coverage,
        evaluation_metric_param=evaluation_metric,
        evaluation_period_param=evaluation_period,
        model_components_param=model_components,
        computation_param=computation,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = Forecaster().run_forecast_config(
            df=df,
            config=config)
        rmse = EvaluationMetricEnum.RootMeanSquaredError.get_metric_name()
        q80 = EvaluationMetricEnum.Quantile80.get_metric_name()
        assert result.backtest.test_evaluation[rmse] == pytest.approx(3.299, rel=1e-2)
        assert result.backtest.test_evaluation[q80] == pytest.approx(1.236, rel=1e-2)
        assert result.forecast.train_evaluation[rmse] == pytest.approx(1.782, rel=1e-2)
        assert result.forecast.train_evaluation[q80] == pytest.approx(0.746, rel=1e-2)
        check_forecast_pipeline_result(
            result,
            coverage=coverage,
            strategy=None,
            score_func=metric.name,
            greater_is_better=False)

    # Note that for newer scikit-learn version, needs to add a check for ValueError, matching "model is misconfigured"
    with pytest.raises((ValueError, KeyError)) as exception_info, pytest.warns(
            UserWarning,
            match="Removing the columns from the input list of 'regressor_cols'"
                  " that are unavailable in the input DataFrame"):
        model_components = ModelComponentsParam(
            regressors={
                "regressor_cols": ["missing_regressor"]
            }
        )
        Forecaster().run_forecast_config(
            df=df,
            config=ForecastConfig(
                model_template=ModelTemplateEnum.SILVERKITE.name,
                model_components_param=model_components,
            ))
    info_str = str(exception_info.value)
    assert "missing_regressor" in info_str or "model is misconfigured" in info_str


def test_run_template_4():
    """Tests:
     - coverage set via uncertainty
     - hyperparameter_grid list
     - monthly data.
    """
    data = generate_df_for_tests(
        freq="MS",
        periods=90)
    df = data["df"]
    forecast_horizon = 30
    # coverage is None but uncertainty is provided
    model_components = ModelComponentsParam(
        uncertainty={
            "uncertainty_dict": {
                "uncertainty_method": "simple_conditional_residuals",
                "params": {
                    "conditional_cols": ["dow_hr"],
                    "quantiles": [0.005, 0.995],  # inferred coverage = 0.99
                    "quantile_estimation_method": "normal_fit",
                    "sample_size_thresh": 5,
                    "small_sample_size_method": "std_quantiles",
                    "small_sample_size_quantile": 0.98}}
        },
        custom={
            "fit_algorithm_dict": {
                "fit_algorithm": "ridge"
            }
        },
        # a list for hyperparameter_grid search
        hyperparameter_override=[
            {"estimator__growth_term": ["linear", "quadratic"]},
            {"estimator__yearly_seasonality": [2]}
        ]
    )
    config = ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        forecast_horizon=forecast_horizon,
        model_components_param=model_components,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = Forecaster().run_forecast_config(
            df=df,
            config=config)

    rmse = EvaluationMetricEnum.RootMeanSquaredError.get_metric_name()
    q80 = EvaluationMetricEnum.Quantile80.get_metric_name()

    # 3 options evaluated
    score_func = EvaluationMetricEnum.MeanAbsolutePercentError.name
    grid_results = summarize_grid_search_results(
        result.grid_search,
        score_func=score_func,
        score_func_greater_is_better=False)
    assert grid_results.shape[0] == 3
    expected_params = [
        [('estimator__yearly_seasonality', 'auto'), ('estimator__growth_term', 'linear')],
        [('estimator__yearly_seasonality', 'auto'), ('estimator__growth_term', 'quadratic')],
        [('estimator__yearly_seasonality', 2), ('estimator__growth_term', 'linear')],
    ]
    assert all(param in list(grid_results["params"]) for param in expected_params)
    assert result.grid_search.best_index_ == 2
    assert result.backtest.test_evaluation[rmse] == pytest.approx(5.425, rel=1e-2)
    assert result.backtest.test_evaluation[q80] == pytest.approx(1.048, rel=1e-2)
    assert result.forecast.train_evaluation[rmse] == pytest.approx(2.526, rel=1e-2)
    assert result.forecast.train_evaluation[q80] == pytest.approx(0.991, rel=1e-2)
    check_forecast_pipeline_result(
        result,
        coverage=0.99,
        strategy=None,
        score_func=score_func,
        greater_is_better=False)


def test_run_template_5():
    """Tests configuration of the time origin passed to forecast_silverkite.
     - origin_for_time_vars by default is set based on the the entire dataset train start date
     - it's possible to override this by adjusting ``estimator__time_properties``.
    """
    # Synthetic dataset with cubic growth so that
    # accuracy depends on having the right time origin.
    data = generate_df_for_tests(
        freq="W",
        periods=230,
        growth_coef=4.0,
        growth_pow=3.0,
        intercept=100.0
    )
    df = data["train_df"]  # training dataset with 185 points
    forecast_horizon = 30
    # Configures 3 CV splits with train start indices: 0, 30, 60.
    evaluation_period = EvaluationPeriodParam(
        cv_expanding_window=False,  # rolling start date
        cv_horizon=30,
        cv_min_train_periods=55,
        cv_periods_between_train_test=10,
        test_horizon=30
    )
    coverage = None
    # Parameters that match synthetic dataset. Should fit well
    # because origin for growth term is the start of the dataset.
    model_components = ModelComponentsParam(
        growth={
            "growth_term": "cubic"
        },
        events={
            "holidays_to_model_separately": None,
            "holiday_lookup_countries": None,
        },
        seasonality={
            "yearly_seasonality": True,
            "quarterly_seasonality": False,
            "monthly_seasonality": False,
            "weekly_seasonality": False,
            "daily_seasonality": False,
        },
        custom={
            "fit_algorithm_dict": {
                "fit_algorithm": "linear"
            }
        }
    )
    # Modifies parameters so that `origin_for_time_vars` is None
    # in the function call to `forecast_silverkite`.
    # The time origin for each CV split is determined by the CV split train
    # start date (rather than entire dataset's train start date).
    time_properties = get_forecast_time_properties(
        df=df,
        forecast_horizon=forecast_horizon)
    time_properties["origin_for_time_vars"] = None
    model_components_dynamic_origin = dataclasses.replace(
        # returns a copy with the ``hyperparameter_override`` field replaced
        model_components,
        hyperparameter_override={
            "estimator__origin_for_time_vars": None,  # default is None, included here for completeness
            "estimator__time_properties": time_properties,
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # The model with fixed `origin_for_time_vars` should have good CV test error
        # because the growth term has the true origin.
        result = Forecaster().run_forecast_config(
            df=df,
            config=ForecastConfig(
                model_template=ModelTemplateEnum.SILVERKITE.name,
                forecast_horizon=forecast_horizon,
                coverage=coverage,
                model_components_param=model_components,
                evaluation_period_param=evaluation_period,
            ))
        metric_name = EvaluationMetricEnum.MeanAbsolutePercentError.get_metric_name()
        cv_results = result.grid_search.cv_results_
        assert cv_results[f"mean_train_{metric_name}"][0] == pytest.approx(1.221, rel=1e-2)
        assert cv_results[f"mean_test_{metric_name}"][0] == pytest.approx(38.81, rel=1e-2)

        # The model with `origin_for_time_vars=None` should have poor CV test error
        # because the growth term has the wrong origin.
        result_dynamic_origin = Forecaster().run_forecast_config(
            df=df,
            config=ForecastConfig(
                model_template=ModelTemplateEnum.SILVERKITE.name,
                forecast_horizon=forecast_horizon,
                coverage=coverage,
                model_components_param=model_components_dynamic_origin,
                evaluation_period_param=evaluation_period,
            ))
        cv_results = result_dynamic_origin.grid_search.cv_results_
        assert cv_results[f"mean_train_{metric_name}"][0] == pytest.approx(1.226, rel=1e-2)
        assert cv_results[f"mean_test_{metric_name}"][0] == pytest.approx(9.320, rel=1e-2)


def test_run_template_6():
    """Tests automatic change point detection feature
    """
    dl = DataLoader()
    data = dl.load_peyton_manning().iloc[-730:]
    model_components = ModelComponentsParam(
        seasonality={
           "weekly_seasonality": 0  # No weekly seasonality
        },
        custom={
            "fit_algorithm_dict": {
                "fit_algorithm": "linear"
            }
        },
        changepoints={
            "changepoints_dict": {
                "method": "auto",
                "yearly_seasonality_order": 6,
                "resample_freq": "2D",
                "actual_changepoint_min_distance": "100D",
                "potential_changepoint_distance": "50D",
                "no_changepoint_proportion_from_end": 0.3
            },
            "seasonality_changepoints_dict": {
                "no_changepoint_proportion_from_end": 0.3,
                "regularization_strength": 0.1
            }
        }
    )
    evaluation_period = EvaluationPeriodParam(
        cv_expanding_window=False,  # rolling start date
        cv_horizon=0,
        cv_min_train_periods=55,
        cv_periods_between_train_test=10,
        test_horizon=30
    )
    config = ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        forecast_horizon=30,
        coverage=None,
        model_components_param=model_components,
        evaluation_period_param=evaluation_period,
    )
    result = Forecaster().run_forecast_config(
        df=data,
        config=config,
    )
    # checks changepoints_dict
    assert result.backtest.estimator.changepoints_dict == {
        "method": "auto",
        "yearly_seasonality_order": 6,
        "resample_freq": "2D",
        "actual_changepoint_min_distance": "100D",
        "potential_changepoint_distance": "50D",
        "no_changepoint_proportion_from_end": 0.3
    }
    # checks there are change points
    assert len(result.model.steps[-1][-1].model_dict["changepoint_values"]) > 0
    seasonality_changepoints = result.model.steps[-1][-1].model_dict["seasonality_changepoint_dates"]
    assert max([len(value) for value in seasonality_changepoints.values()]) > 0
    # checks weekly seasonality changepoint detected but not include because weekly seasonality has order 0
    assert len(seasonality_changepoints["weekly"]) > 0
    assert all(["weekly" not in col for col in result.model[-1].model_dict["pred_cols"]])


def test_run_template_7():
    """Tests custom events
    """
    dl = DataLoader()
    data = dl.load_peyton_manning().iloc[-1000:]
    model_components = ModelComponentsParam(
        custom={
            "fit_algorithm_dict": {
                "fit_algorithm": "linear"
            }
        },
        events={
            "holiday_lookup_countries": [],
            "daily_event_df_dict": {
                "superbowl": pd.DataFrame({
                    "date": ["2008-02-03", "2009-02-01", "2010-02-07", "2011-02-06",
                             "2012-02-05", "2013-02-03", "2014-02-02", "2015-02-01", "2016-02-07"],  # dates
                    "event_name": ["event"] * 9  # labels
                })
            }
        },
        changepoints={
            "changepoints_dict": None
        }
    )
    evaluation_period = EvaluationPeriodParam(
        cv_expanding_window=False,  # rolling start date
        cv_horizon=30,
        cv_min_train_periods=55,
        cv_periods_between_train_test=10,
        test_horizon=30,
        cv_max_splits=1
    )
    config = ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        forecast_horizon=30,
        coverage=None,
        model_components_param=model_components,
        evaluation_period_param=evaluation_period,
    )
    Forecaster().run_forecast_config(
        df=data,
        config=config,
    )


def test_run_template_8():
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
        growth=dict(growth_term=None),
        seasonality=dict(
            yearly_seasonality=[False],
            quarterly_seasonality=[False],
            monthly_seasonality=[False],
            weekly_seasonality=[False],
            daily_seasonality=[False]),
        custom=dict(
            fit_algorithm_dict=dict(fit_algorithm="linear"),
            extra_pred_cols=["ct2"]),
        regressors=dict(regressor_cols=None),
        autoregression=dict(autoreg_dict=dict(lag_dict=dict(orders=[1]))),
        uncertainty=dict(uncertainty_dict=None))
    config = ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
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
        assert result.backtest.test_evaluation[rmse] == pytest.approx(6.691, rel=1e-1)
        check_forecast_pipeline_result(
            result,
            coverage=0.9,
            strategy=None,
            score_func=EvaluationMetricEnum.MeanAbsolutePercentError.name,
            greater_is_better=False)


def test_run_template_9():
    """Tests:
     - regressor and lagged regressors
     - coverage
     - custom parameters
     - null model
     - weekly data
    """
    data = generate_df_with_reg_for_tests(
        freq="W-MON",
        periods=140,
        remove_extra_cols=True,
        mask_test_actuals=True)
    reg_cols_all = ["regressor1", "regressor2", "regressor_bool", "regressor_categ"]
    reg_cols = ["regressor1"]
    keep_cols = [TIME_COL, VALUE_COL] + reg_cols_all
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
        lagged_regressors={
            "lagged_regressor_dict": {
                "regressor2": {
                    "lag_dict": {"orders": [5]},
                    "agg_lag_dict": {
                        "orders_list": [[7, 7 * 2, 7 * 3]],
                        "interval_list": [(8, 7 * 2)]}
                },
                "regressor_bool": "auto",
                "regressor_categ": {
                    "lag_dict": {"orders": [5]}
                }
            }
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
        model_template=ModelTemplateEnum.SILVERKITE.name,
        forecast_horizon=forecast_horizon,
        coverage=coverage,
        evaluation_metric_param=evaluation_metric,
        evaluation_period_param=evaluation_period,
        model_components_param=model_components,
        computation_param=computation,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = Forecaster().run_forecast_config(
            df=df,
            config=config)
        rmse = EvaluationMetricEnum.RootMeanSquaredError.get_metric_name()
        q80 = EvaluationMetricEnum.Quantile80.get_metric_name()
        assert result.backtest.test_evaluation[rmse] == pytest.approx(3.360, rel=1e-2)
        assert result.backtest.test_evaluation[q80] == pytest.approx(1.139, rel=1e-2)
        assert result.forecast.train_evaluation[rmse] == pytest.approx(2.069, rel=1e-2)
        assert result.forecast.train_evaluation[q80] == pytest.approx(0.771, rel=1e-2)
        check_forecast_pipeline_result(
            result,
            coverage=coverage,
            strategy=None,
            score_func=metric.name,
            greater_is_better=False)
        # Checks lagged regressor columns
        actual_pred_cols = set(result.model[-1].model_dict["pred_cols"])
        actual_x_mat_cols = set(result.model[-1].model_dict["x_mat"].columns)
        expected_pred_cols = {
            'regressor1',
            'ct1',
            'regressor2_lag5',
            'regressor2_avglag_7_14_21',
            'regressor2_avglag_8_to_14',
            'regressor_categ_lag5'
        }
        expected_x_mat_cols = {
            'regressor1',
            'ct1',
            'regressor2_lag5',
            'regressor2_avglag_7_14_21',
            'regressor2_avglag_8_to_14',
            'regressor_categ_lag5[T.c2]',
            'regressor_categ_lag5[T.c2]'
        }
        assert expected_pred_cols.issubset(actual_pred_cols)
        assert expected_x_mat_cols.issubset(actual_x_mat_cols)

    # Note that for newer scikit-learn version, needs to add a check for ValueError, matching "model is misconfigured"
    with pytest.raises((ValueError, KeyError)) as exception_info:
        model_components = ModelComponentsParam(
            regressors={
                "regressor_cols": ["missing_regressor"]
            }
        )
        Forecaster().run_forecast_config(
            df=df,
            config=ForecastConfig(
                model_template=ModelTemplateEnum.SILVERKITE.name,
                model_components_param=model_components,
            ))
    info_str = str(exception_info.value)
    assert "missing_regressor" in info_str or "model is misconfigured" in info_str

    # Note that for newer scikit-learn version, needs to add a check for ValueError, matching "model is misconfigured"
    with pytest.raises((ValueError, KeyError)) as exception_info:
        model_components = ModelComponentsParam(
            lagged_regressors={
                "lagged_regressor_dict": {
                    "missing_lagged_regressor": {"lag_dict": {"orders": [5]}}
                }
            }
        )
        Forecaster().run_forecast_config(
            df=df,
            config=ForecastConfig(
                model_template=ModelTemplateEnum.SILVERKITE.name,
                model_components_param=model_components,
            ))
    info_str = str(exception_info.value)
    assert "missing_lagged_regressor" in info_str or "model is misconfigured" in info_str

    # Note that for newer scikit-learn version, needs to add a check for ValueError, matching "model is misconfigured"
    with pytest.raises((ValueError, KeyError)) as exception_info:
        model_components = ModelComponentsParam(
            lagged_regressors={
                "lagged_regressor_dict": {
                    "missing_lagged_regressor": {"lag_dict": {"orders": [5]}},
                    "regressor_bool": {"lag_dict": {"orders": [5]}}
                }
            }
        )
        Forecaster().run_forecast_config(
            df=df,
            config=ForecastConfig(
                model_template=ModelTemplateEnum.SILVERKITE.name,
                model_components_param=model_components,
            ))
    info_str = str(exception_info.value)
    assert "missing_lagged_regressor" in info_str or "model is misconfigured" in info_str


def test_run_template_daily_1():
    dl = DataLoader()
    data = dl.load_peyton_manning()
    evaluation_period = EvaluationPeriodParam(
        cv_expanding_window=False,  # rolling start date
        cv_horizon=10,
        cv_min_train_periods=55,
        cv_periods_between_train_test=0,
        test_horizon=30,
        cv_max_splits=1
    )
    config = ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE_DAILY_1.name,
        forecast_horizon=10,
        evaluation_period_param=evaluation_period,
    )
    result = Forecaster().run_forecast_config(
        df=data.iloc[:100],
        config=config,
    )
    check_forecast_pipeline_result(
        result=result,
        coverage=None,
        expected_grid_size=3
    )


def test_run_template_daily_90():
    dl = DataLoader()
    data = dl.load_peyton_manning()
    evaluation_period = EvaluationPeriodParam(
        cv_expanding_window=False,  # rolling start date
        cv_horizon=10,
        cv_min_train_periods=55,
        cv_periods_between_train_test=0,
        test_horizon=30,
        cv_max_splits=1
    )
    config = ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE_DAILY_90.name,
        forecast_horizon=10,
        evaluation_period_param=evaluation_period,
    )
    result = Forecaster().run_forecast_config(
        df=data.iloc[:100],
        config=config,
    )
    check_forecast_pipeline_result(
        result=result,
        coverage=None,
        expected_grid_size=4
    )


def test_run_template_weekly():
    dl = DataLoader()
    data = dl.load_peyton_manning()
    evaluation_period = EvaluationPeriodParam(
        cv_expanding_window=False,  # rolling start date
        cv_horizon=10,
        cv_min_train_periods=52,
        cv_periods_between_train_test=0,
        test_horizon=10,
        cv_max_splits=1
    )
    config = ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE_WEEKLY.name,
        forecast_horizon=10,
        evaluation_period_param=evaluation_period,
    )
    data["ts"] = pd.to_datetime(data["ts"])
    new_df = data.iloc[:730].resample(rule="W-TUE", on="ts").mean()
    new_df["ts"] = new_df.index
    new_df = new_df.reset_index(drop=True)
    result = Forecaster().run_forecast_config(
        df=new_df,
        config=config,
    )
    check_forecast_pipeline_result(
        result=result,
        coverage=None,
        expected_grid_size=4
    )


def test_run_template_forecast_one_by_one():
    dl = DataLoader()
    data = dl.load_peyton_manning()
    evaluation_period = EvaluationPeriodParam(
        cv_expanding_window=False,  # rolling start date
        cv_horizon=7,
        cv_min_train_periods=365,
        cv_periods_between_train_test=0,
        test_horizon=7,
        cv_max_splits=1
    )
    config = ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE_WEEKLY.name,
        forecast_horizon=7,
        evaluation_period_param=evaluation_period,
        model_components_param=ModelComponentsParam(
            autoregression=dict(autoreg_dict="auto")),
        forecast_one_by_one=[1, 2, 4]
    )
    data["ts"] = pd.to_datetime(data["ts"])
    new_df = data.iloc[:730]
    result = Forecaster().run_forecast_config(
        df=new_df,
        config=config,
    )
    assert len(result.model[-1].estimators) == 3  # 3 models were used.
    # We specified 1, 2, 4 in config, so the forecast horizons should be 1, 3, 7
    assert result.model[-1].estimators[0].forecast_horizon == 1
    assert result.model[-1].estimators[1].forecast_horizon == 3
    assert result.model[-1].estimators[2].forecast_horizon == 7
    # We have the right forecasts from each model.
    future_df = result.forecast.df_test[[TIME_COL]]
    assert_equal(
        result.model[-1].estimators[0].predict(future_df.iloc[:1])[[PREDICTED_COL]].values,
        result.forecast.df_test.iloc[:1][[PREDICTED_COL]].values)
    assert_equal(
        result.model[-1].estimators[1].predict(future_df.iloc[1:3])[[PREDICTED_COL]].values,
        result.forecast.df_test.iloc[1:3][[PREDICTED_COL]].values)
    assert_equal(
        result.model[-1].estimators[2].predict(future_df.iloc[3:7])[[PREDICTED_COL]].values,
        result.forecast.df_test.iloc[3:7][[PREDICTED_COL]].values)


def test_silverkite_empty():
    dl = DataLoader()
    data = dl.load_peyton_manning()
    evaluation_period = EvaluationPeriodParam(
        cv_expanding_window=False,  # rolling start date
        cv_horizon=10,
        cv_min_train_periods=52,
        cv_periods_between_train_test=0,
        test_horizon=10,
        cv_max_splits=1
    )
    config = ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE_EMPTY.name,
        forecast_horizon=10,
        evaluation_period_param=evaluation_period,
    )
    # Resamples to weekly, faster.
    data["ts"] = pd.to_datetime(data["ts"])
    new_df = data.iloc[:730].resample(rule="W-TUE", on="ts").mean()
    new_df["ts"] = new_df.index
    new_df = new_df.reset_index(drop=True)
    result = Forecaster().run_forecast_config(
        df=new_df,
        config=config,
    )
    # Only the intercept term is in the design matrix.
    assert result.model[-1].model_dict["x_mat"].shape[1] == 1


def test_silverkite_ar_lag():
    dl = DataLoader()
    data = dl.load_peyton_manning()
    evaluation_period = EvaluationPeriodParam(
        cv_expanding_window=False,  # rolling start date
        cv_horizon=1,
        cv_min_train_periods=52,
        cv_periods_between_train_test=0,
        test_horizon=1,
        cv_max_splits=1
    )
    config = ForecastConfig(
        model_template="DAILY_AR_AUTO",
        forecast_horizon=1,
        evaluation_period_param=evaluation_period,
    )
    data["ts"] = pd.to_datetime(data["ts"])
    new_df = data.iloc[:365]
    result = Forecaster().run_forecast_config(
        df=new_df,
        config=config,
    )
    assert "y_lag1" in result.model[-1].model_dict["pred_cols"]


def test_silverkite_simulation_num():
    dl = DataLoader()
    data = dl.load_peyton_manning()
    evaluation_period = EvaluationPeriodParam(
        cv_expanding_window=False,  # rolling start date
        cv_horizon=0,
        cv_min_train_periods=52,
        cv_periods_between_train_test=0,
        test_horizon=1,
        cv_max_splits=1
    )
    config = ForecastConfig(
        model_template="SILVERKITE_EMPTY",
        forecast_horizon=2,
        evaluation_period_param=evaluation_period,
        model_components_param=ModelComponentsParam(
            autoregression=dict(
                autoreg_dict={
                    "lag_dict": {"orders": [1]}},
                simulation_num=2
            )
        )
    )
    data["ts"] = pd.to_datetime(data["ts"])
    new_df = data.iloc[:365]
    result = Forecaster().run_forecast_config(
        df=new_df,
        config=config,
    )
    assert result.model[-1].model_dict["simulation_num"] == 2
    assert result.model[-1].model_dict["fast_simulation"] is False

    # Tests fast simulation
    config = ForecastConfig(
        model_template="SILVERKITE_EMPTY",
        forecast_horizon=2,
        evaluation_period_param=evaluation_period,
        model_components_param=ModelComponentsParam(
            autoregression=dict(
                autoreg_dict={
                    "lag_dict": {"orders": [1]}},
                simulation_num=2,
                fast_simulation=True
            )
        )
    )
    result = Forecaster().run_forecast_config(
        df=new_df,
        config=config,
    )
    assert result.model[-1].model_dict["simulation_num"] == 2
    assert result.model[-1].model_dict["fast_simulation"] is True


def test_silverkite_float32():
    dl = DataLoader()
    data = dl.load_peyton_manning()
    data[TIME_COL] = pd.to_datetime(data[TIME_COL])
    data[VALUE_COL] = data[VALUE_COL].astype("float32")
    evaluation_period = EvaluationPeriodParam(
        cv_expanding_window=False,  # rolling start date
        cv_horizon=0,
        cv_min_train_periods=52,
        cv_periods_between_train_test=0,
        test_horizon=1,
        cv_max_splits=1
    )
    config = ForecastConfig(
        model_template="SILVERKITE",
        forecast_horizon=1,
        evaluation_period_param=evaluation_period,
        model_components_param=ModelComponentsParam()
    )
    new_df = data.iloc[:365]
    Forecaster().run_forecast_config(
        df=new_df,
        config=config,
    )


def test_silverkite_monthly_template():
    dl = DataLoader()
    data = dl.load_peyton_manning()
    data[TIME_COL] = pd.to_datetime(data[TIME_COL])
    data = data.resample("MS", on=TIME_COL).sum().reset_index(drop=False)
    evaluation_period = EvaluationPeriodParam(
        test_horizon=1,
        cv_max_splits=1
    )
    config = ForecastConfig(
        model_template="SILVERKITE_MONTHLY",
        forecast_horizon=1,
        evaluation_period_param=evaluation_period,
        model_components_param=ModelComponentsParam()
    )
    Forecaster().run_forecast_config(
        df=data,
        config=config,
    )


def test_silverkite_monthly_template_potential_changepoint_cap():
    df = pd.DataFrame({
        TIME_COL: pd.date_range("2000-01-01", freq="MS", periods=12 * 200),
        VALUE_COL: 1
    })
    with LogCapture(LOGGER_NAME) as log_capture:
        evaluation_period = EvaluationPeriodParam(
            test_horizon=0,
            cv_max_splits=0
        )
        config = ForecastConfig(
            model_template="SILVERKITE_MONTHLY",
            forecast_horizon=1,
            evaluation_period_param=evaluation_period,
        )
        Forecaster().run_forecast_config(
            df=df,
            config=config,
        )
        log_capture.check_present((
            LOGGER_NAME,
            "INFO",
            f"Number of potential changepoints is capped by 'potential_changepoint_n_max' "
            f"as 100. The 'potential_changepoint_distance' "
            f"180D is ignored. The original number of changepoints was 405."
        ))


def test_silverkite_auto_config():
    """Tests automatic seasonality, growth and holidays."""
    dl = DataLoader()
    data = dl.load_peyton_manning()
    data[TIME_COL] = pd.to_datetime(data[TIME_COL])
    evaluation_period = EvaluationPeriodParam(
        test_horizon=1,
        cv_max_splits=1
    )
    config = ForecastConfig(
        model_template="SILVERKITE",
        forecast_horizon=1,
        evaluation_period_param=evaluation_period,
        model_components_param=ModelComponentsParam(
            growth=dict(growth_term="quadratic"),  # will be overriden by auto growth
            seasonality=dict(auto_seasonality=True),
            events=dict(auto_holiday=True),
            changepoints=dict(auto_growth=True)
        )
    )
    result = Forecaster().run_forecast_config(
        df=data,
        config=config,
    )
    assert result.model[-1].model_dict["fs_components_df"][["name", "period", "order", "seas_names"]].equals(pd.DataFrame({
        "name": ["tow", "tom", "toq", "ct1"],
        "period": [7.0, 1.0, 1.0, 1.0],
        "order": [3, 1, 1, 6],
        "seas_names": ["weekly", "monthly", "quarterly", "yearly"]
    }))
    assert len(result.model[-1].model_dict["daily_event_df_dict"]) == 194
    assert "ct1" in result.model[-1].model_dict["x_mat"].columns
