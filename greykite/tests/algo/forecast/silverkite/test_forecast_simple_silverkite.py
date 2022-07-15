import datetime
import math
import sys
import warnings
from datetime import timedelta

import pandas as pd
import pytest
from testfixtures import LogCapture

import greykite.common.constants as cst
from greykite.algo.changepoint.adalasso.changepoint_detector import ChangepointDetector
from greykite.algo.forecast.silverkite.constants.silverkite_column import SilverkiteColumn
from greykite.algo.forecast.silverkite.constants.silverkite_holiday import SilverkiteHoliday
from greykite.algo.forecast.silverkite.constants.silverkite_seasonality import SilverkiteSeasonality
from greykite.algo.forecast.silverkite.constants.silverkite_seasonality import SilverkiteSeasonalityEnum
from greykite.algo.forecast.silverkite.forecast_simple_silverkite import SimpleSilverkiteForecast
from greykite.algo.forecast.silverkite.forecast_simple_silverkite_helper import generate_holiday_events
from greykite.common.constants import CHANGEPOINT_COL_PREFIX
from greykite.common.constants import EVENT_DF_DATE_COL
from greykite.common.constants import EVENT_DF_LABEL_COL
from greykite.common.constants import LOGGER_NAME
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.data_loader import DataLoader
from greykite.common.enums import SimpleTimeFrequencyEnum
from greykite.common.enums import TimeEnum
from greykite.common.features.timeseries_features import get_available_holidays_across_countries
from greykite.common.features.timeseries_features import get_holidays
from greykite.common.python_utils import assert_equal
from greykite.common.python_utils import update_dictionary
from greykite.common.testing_utils import generate_df_for_tests
from greykite.common.testing_utils import generate_df_with_reg_for_tests


@pytest.fixture
def daily_data_reg():
    """Generates daily data with regressors for tests.
    Returns
    -------
    df : pd.DataFrame
        with columns of type: datetime, number, number, boolean, object, category
    """
    df = generate_df_with_reg_for_tests(
        freq="D",
        periods=50,
        train_start_date=datetime.datetime(2018, 11, 30),
        remove_extra_cols=False)["df"]
    df["dow_categorical"] = df["str_dow"].astype("category")
    df = df[[cst.TIME_COL, cst.VALUE_COL, "regressor1", "regressor2", "regressor_bool", "str_dow", "dow_categorical"]]
    df = df.rename({
        cst.TIME_COL: "custom_time_col",
        cst.VALUE_COL: "custom_value_col"
    }, axis=1)
    return df


@pytest.fixture
def weekly_data():
    """Generates 20 weeks of weekly data for tests"""
    return generate_df_for_tests(
        freq="W",
        periods=20,
        train_start_date=datetime.datetime(2018, 11, 30))["df"]


@pytest.fixture
def hourly_data():
    """Generates 500 days of hourly data for tests"""
    return generate_df_for_tests(
        freq="H",
        periods=24*500,
        train_start_date=datetime.datetime(2018, 7, 1))["df"]


def test_origin_for_time_vars(hourly_data):
    """Tests ``convert_params`` origin_for_time_vars output"""
    time_properties = {
        "ts": None,
        "period": TimeEnum.ONE_HOUR_IN_SECONDS.value,
        "simple_freq": SimpleTimeFrequencyEnum.HOUR,
        "num_training_points": hourly_data.shape[0],
        "num_training_days": math.floor(hourly_data.shape[0] / 24),
        "start_year": 2018,
        "end_year": 2019,
        "origin_for_time_vars": 2018.12
    }

    # ``origin_for_time_vars`` takes precedence
    silverkite = SimpleSilverkiteForecast()
    parameters = silverkite.convert_params(
        df=hourly_data,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        origin_for_time_vars=2017.12,
        time_properties=time_properties
    )
    assert parameters["origin_for_time_vars"] == 2017.12

    # then default to ``time_properties``
    parameters = silverkite.convert_params(
        df=hourly_data,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        time_properties=time_properties)
    assert parameters["origin_for_time_vars"] == 2018.12

    # then default to the first point in ``df``
    parameters = silverkite.convert_params(
        df=hourly_data,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)
    assert parameters["origin_for_time_vars"] == 2018.495890410959


def test_daily_event_df_dict(hourly_data):
    """Tests ``convert_params`` daily_event_df_dict output"""
    silverkite = SimpleSilverkiteForecast()
    time_properties = {
        "ts": None,
        "period": TimeEnum.ONE_HOUR_IN_SECONDS.value,
        "simple_freq": SimpleTimeFrequencyEnum.HOUR,
        "num_training_points": hourly_data.shape[0],
        "num_training_days": math.floor(hourly_data.shape[0] / 24),
        "start_year": 2018,
        "end_year": 2019,
        "origin_for_time_vars": 2018.12
    }
    holiday_lookup_countries = ["UnitedStates", "China"]
    holidays_to_model_separately = SilverkiteHoliday.ALL_HOLIDAYS_IN_COUNTRIES
    holiday_pre_num_days = 1
    holiday_post_num_days = 2
    holiday_pre_post_num_dict = {
        "New Year's Day": (7, 3),
        "Christmas Day": (3, 3)
    }
    expected = silverkite._SimpleSilverkiteForecast__get_silverkite_holidays(
        holiday_lookup_countries=holiday_lookup_countries,
        holidays_to_model_separately=holidays_to_model_separately,
        start_year=time_properties["start_year"],
        end_year=time_properties["end_year"],
        pre_num=holiday_pre_num_days,
        post_num=holiday_post_num_days,
        pre_post_num_dict=holiday_pre_post_num_dict)

    # uses provided ``time_properties``
    # (checks that ``time_properties`` is properly set)
    parameters = silverkite.convert_params(
        df=hourly_data,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        time_properties=time_properties,
        holiday_lookup_countries=holiday_lookup_countries,
        holidays_to_model_separately=holidays_to_model_separately,
        holiday_pre_num_days=holiday_pre_num_days,
        holiday_post_num_days=holiday_post_num_days,
        holiday_pre_post_num_dict=holiday_pre_post_num_dict)
    assert_equal(parameters["daily_event_df_dict"], expected)

    # uses time properties calculated from the data
    parameters = silverkite.convert_params(
        df=hourly_data,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        holiday_lookup_countries=holiday_lookup_countries,
        holidays_to_model_separately=holidays_to_model_separately,
        holiday_pre_num_days=holiday_pre_num_days,
        holiday_post_num_days=holiday_post_num_days,
        holiday_pre_post_num_dict=holiday_pre_post_num_dict)
    assert_equal(parameters["daily_event_df_dict"], expected)

    # With `daily_event_df_dict` and holidays
    countries = ["US", "India", "UnitedKingdom"]
    event_df_dict = get_holidays(
        countries,
        year_start=2015,
        year_end=2030)
    parameters = silverkite.convert_params(
        df=hourly_data,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        holiday_lookup_countries=holiday_lookup_countries,
        holidays_to_model_separately=holidays_to_model_separately,
        holiday_pre_num_days=holiday_pre_num_days,
        holiday_post_num_days=holiday_post_num_days,
        holiday_pre_post_num_dict=holiday_pre_post_num_dict,
        daily_event_df_dict=event_df_dict)
    expected = update_dictionary(expected, overwrite_dict=event_df_dict)
    assert_equal(parameters["daily_event_df_dict"], expected)

    # With `daily_event_df_dict` and no holidays
    parameters = silverkite.convert_params(
        df=hourly_data,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        holiday_lookup_countries=None,
        holidays_to_model_separately=None,
        daily_event_df_dict=event_df_dict)
    assert_equal(parameters["daily_event_df_dict"], event_df_dict)

    # With no `daily_event_df_dict` and no holidays
    parameters = silverkite.convert_params(
        df=hourly_data,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        holiday_lookup_countries=None,
        holidays_to_model_separately=None,
        daily_event_df_dict=None)
    assert_equal(parameters["daily_event_df_dict"], None)


def test_fs_components_df(hourly_data):
    """Tests ``convert_simple_silverkite_params`` fs_components_df output"""
    silverkite = SimpleSilverkiteForecast()
    time_properties = {
        "ts": None,
        "period": TimeEnum.ONE_HOUR_IN_SECONDS.value,
        "simple_freq": SimpleTimeFrequencyEnum.HOUR,
        "num_training_points": hourly_data.shape[0],
        "num_training_days": math.floor(hourly_data.shape[0] / 24),
        "start_year": 2018,
        "end_year": 2019,
        "origin_for_time_vars": 2018.12
    }

    seasonality = {
        "yearly_seasonality": "auto",
        "quarterly_seasonality": "auto",
        "monthly_seasonality": "auto",
        "weekly_seasonality": "auto",
        "daily_seasonality": "auto",
    }
    expected = silverkite._SimpleSilverkiteForecast__get_silverkite_seasonality(
        simple_freq=time_properties["simple_freq"].name,
        num_days=time_properties["num_training_days"],
        seasonality=seasonality
    )

    silverkite = SimpleSilverkiteForecast()
    parameters = silverkite.convert_params(
        df=hourly_data,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        time_properties=time_properties,
        yearly_seasonality=seasonality["yearly_seasonality"],
        quarterly_seasonality=seasonality["quarterly_seasonality"],
        monthly_seasonality=seasonality["monthly_seasonality"],
        weekly_seasonality=seasonality["weekly_seasonality"],
        daily_seasonality=seasonality["daily_seasonality"],
    )
    assert_equal(parameters["fs_components_df"], expected)

    seasonality = {
        "yearly_seasonality": True,
        "quarterly_seasonality": False,
        "monthly_seasonality": 10,
        "weekly_seasonality": 2,
        "daily_seasonality": False,
    }
    expected = silverkite._SimpleSilverkiteForecast__get_silverkite_seasonality(
        simple_freq=time_properties["simple_freq"].name,
        num_days=time_properties["num_training_days"],
        seasonality=seasonality
    )
    parameters = silverkite.convert_params(
        df=hourly_data,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        time_properties=time_properties,
        yearly_seasonality=seasonality["yearly_seasonality"],
        quarterly_seasonality=seasonality["quarterly_seasonality"],
        monthly_seasonality=seasonality["monthly_seasonality"],
        weekly_seasonality=seasonality["weekly_seasonality"],
        daily_seasonality=seasonality["daily_seasonality"],
    )
    assert_equal(parameters["fs_components_df"], expected)


def test_passthrough(daily_data_reg):
    """Tests ``convert_simple_silverkite_params`` on parameters that are passed through."""

    # non-default values for the passthrough parameters
    # ``df``, ``time_col``, ``value_col``, ``changepoints_dict`` are used
    # within the function and need to be real values for this test
    passthrough_params = dict(
        df=daily_data_reg,
        time_col="custom_time_col",
        value_col="custom_value_col",
        train_test_thresh=datetime.datetime(2019, 1, 1),
        training_fraction=0.3,
        fit_algorithm="lasso",
        fit_algorithm_params={"fit_algorithm_param": "value"},
        changepoints_dict={
            "method": "uniform",
            "n_changepoints": 20},
        autoreg_dict={"autoreg_param": "value"},
        lagged_regressor_dict={"lagged_regressor_param": "value"},
        min_admissible_value=1,
        max_admissible_value=10,
        uncertainty_dict={"uncertainty_param": "value"})

    silverkite = SimpleSilverkiteForecast()
    parameters = silverkite.convert_params(**passthrough_params)
    # only check the passthrough parameters
    result_params = {k: v for k, v in parameters.items() if k in passthrough_params.keys()}
    assert_equal(result_params, passthrough_params)


def test_extra_pred_cols(hourly_data, daily_data_reg, weekly_data):
    """Tests ``convert_simple_silverkite_params`` extra_pred_cols output
    It should contain:

        - growth
        - regressors
        - holidays
        - feature sets

    And not have any duplicates.
    """
    extra_pred_cols = ["epc1", "epc2"]
    growth_term = "linear"

    # some columns that belong to each feature set
    # assumes the default holidays are used
    # assumes `growth_term="linear"`
    example_cols_hour_of_week = ["C(Q('dow_hr'), levels=['1_00', '1_01', '1_02', '1_03', '1_04', '1_05', '1_06', '1_07', '1_08', '1_09', '1_10', '1_11', '1_12', '1_13', '1_14', '1_15', '1_16', '1_17', '1_18', '1_19', '1_20', '1_21', '1_22', '1_23', '2_00', '2_01', '2_02', '2_03', '2_04', '2_05', '2_06', '2_07', '2_08', '2_09', '2_10', '2_11', '2_12', '2_13', '2_14', '2_15', '2_16', '2_17', '2_18', '2_19', '2_20', '2_21', '2_22', '2_23', '3_00', '3_01', '3_02', '3_03', '3_04', '3_05', '3_06', '3_07', '3_08', '3_09', '3_10', '3_11', '3_12', '3_13', '3_14', '3_15', '3_16', '3_17', '3_18', '3_19', '3_20', '3_21', '3_22', '3_23', '4_00', '4_01', '4_02', '4_03', '4_04', '4_05', '4_06', '4_07', '4_08', '4_09', '4_10', '4_11', '4_12', '4_13', '4_14', '4_15', '4_16', '4_17', '4_18', '4_19', '4_20', '4_21', '4_22', '4_23', '5_00', '5_01', '5_02', '5_03', '5_04', '5_05', '5_06', '5_07', '5_08', '5_09', '5_10', '5_11', '5_12', '5_13', '5_14', '5_15', '5_16', '5_17', '5_18', '5_19', '5_20', '5_21', '5_22', '5_23', '6_00', '6_01', '6_02', '6_03', '6_04', '6_05', '6_06', '6_07', '6_08', '6_09', '6_10', '6_11', '6_12', '6_13', '6_14', '6_15', '6_16', '6_17', '6_18', '6_19', '6_20', '6_21', '6_22', '6_23', '7_00', '7_01', '7_02', '7_03', '7_04', '7_05', '7_06', '7_07', '7_08', '7_09', '7_10', '7_11', '7_12', '7_13', '7_14', '7_15', '7_16', '7_17', '7_18', '7_19', '7_20', '7_21', '7_22', '7_23'])"]  # noqa: E501
    example_cols_weekend_seas = ["is_weekend:sin12_tod_daily", "is_weekend:cos12_tod_daily"]
    example_cols_day_of_week_seas = ["C(Q('str_dow'), levels=['1-Mon', '2-Tue', '3-Wed', '4-Thu', '5-Fri', '6-Sat', '7-Sun']):sin12_tod_daily", "C(Q('str_dow'), levels=['1-Mon', '2-Tue', '3-Wed', '4-Thu', '5-Fri', '6-Sat', '7-Sun']):cos12_tod_daily"]  # noqa: E501
    example_cols_trend_daily_seas = ["is_weekend:ct1:sin12_tod_daily", "is_weekend:ct1:cos12_tod_daily"]
    example_cols_event_seas = ["C(Q('events_Independence Day'), levels=['', 'event']):sin12_tod_daily", "C(Q('events_New Years Day_plus_1'), levels=['', 'event']):cos12_tod_daily"]  # noqa: E501
    example_cols_event_weekend_seas = ["is_weekend:C(Q('events_Independence Day'), levels=['', 'event']):sin12_tod_daily", "is_weekend:C(Q('events_New Years Day_plus_1'), levels=['', 'event']):cos12_tod_daily"]  # noqa: E501
    example_cols_day_of_week = ["C(Q('str_dow'), levels=['1-Mon', '2-Tue', '3-Wed', '4-Thu', '5-Fri', '6-Sat', '7-Sun'])"]  # noqa: E501
    example_cols_trend_weekend = ["is_weekend:ct1"]
    example_cols_trend_day_of_week = ["C(Q('str_dow'), levels=['1-Mon', '2-Tue', '3-Wed', '4-Thu', '5-Fri', '6-Sat', '7-Sun']):ct1"]  # noqa: E501
    example_cols_trend_weekly_seas = ["ct1:sin4_tow_weekly", "ct1:cos4_tow_weekly"]
    example_cols = {
        SilverkiteColumn.COLS_HOUR_OF_WEEK: example_cols_hour_of_week,
        SilverkiteColumn.COLS_WEEKEND_SEAS: example_cols_weekend_seas,
        SilverkiteColumn.COLS_DAY_OF_WEEK_SEAS: example_cols_day_of_week_seas,
        SilverkiteColumn.COLS_TREND_DAILY_SEAS: example_cols_trend_daily_seas,
        SilverkiteColumn.COLS_EVENT_SEAS: example_cols_event_seas,
        SilverkiteColumn.COLS_EVENT_WEEKEND_SEAS: example_cols_event_weekend_seas,
        SilverkiteColumn.COLS_DAY_OF_WEEK: example_cols_day_of_week,
        SilverkiteColumn.COLS_TREND_WEEKEND: example_cols_trend_weekend,
        SilverkiteColumn.COLS_TREND_DAY_OF_WEEK: example_cols_trend_day_of_week,
        SilverkiteColumn.COLS_TREND_WEEKLY_SEAS: example_cols_trend_weekly_seas,
    }
    # example holiday columns expected in the output
    example_holiday_cols = [
        "C(Q('events_Chinese New Year'), levels=['', 'event'])",
        "C(Q('events_Chinese New Year_minus_1'), levels=['', 'event'])",
        "C(Q('events_Chinese New Year_minus_2'), levels=['', 'event'])",
        "C(Q('events_Chinese New Year_plus_1'), levels=['', 'event'])",
        "C(Q('events_Chinese New Year_plus_2'), levels=['', 'event'])",
        "C(Q('events_Independence Day'), levels=['', 'event'])"
    ]
    excluded_holiday_cols = [  # should not appear
        "C(Q('events_Chinese New Year_minus_3'), levels=['', 'event'])"
        "C(Q('events_Independence Day_plus_3'), levels=['', 'event'])"
    ]

    # TEST 1 -- specified feature sets, on hourly data
    feature_sets_enabled = {
        SilverkiteColumn.COLS_HOUR_OF_WEEK: False,
        SilverkiteColumn.COLS_WEEKEND_SEAS: True,
        SilverkiteColumn.COLS_DAY_OF_WEEK_SEAS: False,
        SilverkiteColumn.COLS_TREND_DAILY_SEAS: True,
        SilverkiteColumn.COLS_EVENT_SEAS: False,
        SilverkiteColumn.COLS_EVENT_WEEKEND_SEAS: True,
        SilverkiteColumn.COLS_DAY_OF_WEEK: False,
        SilverkiteColumn.COLS_TREND_WEEKEND: True,
        SilverkiteColumn.COLS_TREND_DAY_OF_WEEK: False,
        SilverkiteColumn.COLS_TREND_WEEKLY_SEAS: True
    }

    silverkite = SimpleSilverkiteForecast()
    parameters = silverkite.convert_params(
        df=hourly_data,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        growth_term=growth_term,
        feature_sets_enabled=feature_sets_enabled,
        extra_pred_cols=extra_pred_cols,
    )
    result_cols = parameters["extra_pred_cols"]

    feature_set_cols = []   # feature set columns that should appear
    excluded_cols = []     # columns that should not appear
    excluded_cols += excluded_holiday_cols
    for feature_set_name, use_set in feature_sets_enabled.items():
        if use_set:
            feature_set_cols += example_cols[feature_set_name]
        else:
            excluded_cols += example_cols[feature_set_name]
    expected_cols = set(
        [cst.GrowthColEnum[growth_term].value] +
        example_holiday_cols +
        feature_set_cols +
        extra_pred_cols
    )
    assert len(result_cols) == 427                          # desired length
    assert len(result_cols) == len(set(result_cols))        # no duplicates
    assert set(expected_cols).issubset(set(result_cols))    # contains expected cols
    assert len(set(excluded_cols).intersection(set(result_cols))) == 0  # does not contain excluded cols

    # No feature sets
    parameters = silverkite.convert_params(
        df=hourly_data,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        growth_term=growth_term,
        feature_sets_enabled=None,  # equivalent to False
        extra_pred_cols=extra_pred_cols,
    )
    result_cols = parameters["extra_pred_cols"]
    assert len(result_cols) == 58
    assert not any(col in result_cols for _, cols in example_cols.items() for col in cols)

    # TEST 2 -- specified feature sets, on daily data with regressors
    # NB: some feature sets don't make sense for daily data, used here for unit test
    regressor_cols = ["regressor1", "regressor2"]
    feature_sets_enabled = {
        SilverkiteColumn.COLS_HOUR_OF_WEEK: True,
        SilverkiteColumn.COLS_WEEKEND_SEAS: True,
        SilverkiteColumn.COLS_DAY_OF_WEEK_SEAS: True,
        SilverkiteColumn.COLS_TREND_DAILY_SEAS: True,
        SilverkiteColumn.COLS_EVENT_SEAS: True,
        SilverkiteColumn.COLS_EVENT_WEEKEND_SEAS: True,
        SilverkiteColumn.COLS_DAY_OF_WEEK: True,
        SilverkiteColumn.COLS_TREND_WEEKEND: False,
        SilverkiteColumn.COLS_TREND_DAY_OF_WEEK: True,
        SilverkiteColumn.COLS_TREND_WEEKLY_SEAS: False
    }
    # Adds some extra holidays and checks if they are included
    event_df_dict = get_holidays(
        ["UnitedStates", "China"],
        year_start=2015,
        year_end=2030)
    more_example_holiday_cols = [
        'C(Q(\'events_China\'), levels=[\'\', "New Year\'s Day", \'Chinese New Year\', \'Tomb-Sweeping Day\', '
        '\'Labor Day\', \'Dragon Boat Festival\', \'Mid-Autumn Festival\', \'National Day\', \'National Day, '
        'Mid-Autumn Festival\'])',
        'C(Q(\'events_UnitedStates\'), levels=[\'\', "New Year\'s Day", \'Martin Luther King Jr. Day\', '
        '"Washington\'s Birthday", \'Memorial Day\', \'Independence Day\', \'Independence Day (Observed)\', '
        '\'Labor Day\', \'Columbus Day\', \'Veterans Day\', \'Thanksgiving\', \'Christmas Day\', '
        '\'Christmas Day (Observed)\', "New Year\'s Day (Observed)", \'Veterans Day (Observed)\', '
        '\'Juneteenth National Independence Day\', \'Juneteenth National Independence Day (Observed)\'])']

    parameters = silverkite.convert_params(
        df=daily_data_reg,
        time_col="custom_time_col",
        value_col="custom_value_col",
        daily_event_df_dict=event_df_dict,
        growth_term=growth_term,
        regressor_cols=regressor_cols,
        feature_sets_enabled=feature_sets_enabled,
        extra_pred_cols=extra_pred_cols,
    )
    result_cols = parameters["extra_pred_cols"]

    feature_set_cols = []   # feature set columns that should appear
    excluded_cols = []     # columns that should not appear
    excluded_cols += excluded_holiday_cols
    # no daily seasonality for daily data, so these aren't used
    feature_sets_used = feature_sets_enabled.copy()
    feature_sets_used[SilverkiteColumn.COLS_WEEKEND_SEAS] = False
    feature_sets_used[SilverkiteColumn.COLS_DAY_OF_WEEK_SEAS] = False
    feature_sets_used[SilverkiteColumn.COLS_TREND_DAILY_SEAS] = False
    feature_sets_used[SilverkiteColumn.COLS_EVENT_SEAS] = False
    feature_sets_used[SilverkiteColumn.COLS_EVENT_WEEKEND_SEAS] = False
    for feature_set_name, use_set in feature_sets_used.items():
        if use_set:
            feature_set_cols += example_cols[feature_set_name]
        else:
            excluded_cols += example_cols[feature_set_name]
    expected_cols = set(
        [cst.GrowthColEnum[growth_term].value] +
        regressor_cols +
        example_holiday_cols +
        more_example_holiday_cols +
        feature_set_cols +
        extra_pred_cols
    )
    assert len(result_cols) == 65                           # desired length
    assert len(result_cols) == len(set(result_cols))        # no duplicates
    assert set(expected_cols).issubset(set(result_cols))    # contains expected cols
    assert len(set(excluded_cols).intersection(set(result_cols))) == 0  # does not contain excluded cols

    # TEST 3 -- default feature sets, on weekly data
    parameters = silverkite.convert_params(
        df=weekly_data,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        growth_term=growth_term,
        extra_pred_cols=extra_pred_cols,
    )
    result_cols = parameters["extra_pred_cols"]

    feature_set_cols = []   # feature set columns that should appear
    excluded_cols = []     # columns that should not appear
    excluded_cols += excluded_holiday_cols
    # no feature sets apply for weekly data
    feature_sets_used = {
        SilverkiteColumn.COLS_HOUR_OF_WEEK: False,
        SilverkiteColumn.COLS_WEEKEND_SEAS: False,
        SilverkiteColumn.COLS_DAY_OF_WEEK_SEAS: False,
        SilverkiteColumn.COLS_TREND_DAILY_SEAS: False,
        SilverkiteColumn.COLS_EVENT_SEAS: False,
        SilverkiteColumn.COLS_EVENT_WEEKEND_SEAS: False,
        SilverkiteColumn.COLS_DAY_OF_WEEK: False,
        SilverkiteColumn.COLS_TREND_WEEKEND: False,
        SilverkiteColumn.COLS_TREND_DAY_OF_WEEK: False,
        SilverkiteColumn.COLS_TREND_WEEKLY_SEAS: False
    }
    for feature_set_name, use_set in feature_sets_used.items():
        if use_set:
            feature_set_cols += example_cols[feature_set_name]
        else:
            excluded_cols += example_cols[feature_set_name]
    expected_cols = set(
        [cst.GrowthColEnum[growth_term].value] +
        example_holiday_cols +
        feature_set_cols +
        extra_pred_cols
    )
    assert len(result_cols) == 58                           # desired length
    assert len(result_cols) == len(set(result_cols))        # no duplicates
    assert set(expected_cols).issubset(set(result_cols))    # contains expected cols
    assert len(set(excluded_cols).intersection(set(result_cols))) == 0  # does not contain excluded cols


def test_convert_simple_silverkite_params_hourly(hourly_data):
    """Tests ``convert_simple_silverkite_params`` on hourly data.
    A supplementary test to the parameter-specific tests above.
    """
    silverkite = SimpleSilverkiteForecast()
    parameters = silverkite.convert_params(
        df=hourly_data,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL)
    extra_pred_cols = parameters.pop("extra_pred_cols")

    # expected parameters
    origin_for_time_vars = 2018.495890410959
    expected_event = silverkite._SimpleSilverkiteForecast__get_silverkite_holidays(
        holiday_lookup_countries="auto",
        holidays_to_model_separately="auto",
        start_year=2018,
        end_year=2019,  # 500 days of hourly, and short forecast horizon
        pre_num=2,
        post_num=2)
    expected_fs = silverkite._SimpleSilverkiteForecast__get_silverkite_seasonality(
        simple_freq=SimpleTimeFrequencyEnum.HOUR.name,
        num_days=500,
        seasonality={
            "yearly_seasonality": "auto",
            "quarterly_seasonality": "auto",
            "monthly_seasonality": "auto",
            "weekly_seasonality": "auto",
            "daily_seasonality": "auto",
        }
    )
    expected = dict(
        df=hourly_data,
        time_col=cst.TIME_COL,
        value_col=cst.VALUE_COL,
        origin_for_time_vars=origin_for_time_vars,
        # extra_pred_cols=extra_pred_cols,  # checked separately
        drop_pred_cols=None,
        explicit_pred_cols=None,
        train_test_thresh=None,
        training_fraction=0.9,
        fit_algorithm="ridge",
        fit_algorithm_params=None,
        daily_event_df_dict=expected_event,
        changepoints_dict=None,
        fs_components_df=expected_fs,
        autoreg_dict=None,
        past_df=None,
        lagged_regressor_dict=None,
        seasonality_changepoints_dict=None,
        min_admissible_value=None,
        max_admissible_value=None,
        uncertainty_dict=None,
        normalize_method=None,
        changepoint_detector=None,
        regression_weight_col=None,
        # ``forecast_horizon`` not given, but automatically populated from
        # ``time_properties``, which is the default 24 for hourly data.
        forecast_horizon=24,
        simulation_based=False,
        simulation_num=10,
        fast_simulation=False
    )
    assert_equal(parameters, expected)

    assert len(parameters['daily_event_df_dict'].keys()) == 55
    assert len(extra_pred_cols) == 452
    assert {
        "ct1",
        "C(Q('events_Chinese New Year'), levels=['', 'event'])",
        "C(Q('events_Chinese New Year_minus_1'), levels=['', 'event'])",
        "C(Q('str_dow'), levels=['1-Mon', '2-Tue', '3-Wed', '4-Thu', '5-Fri', '6-Sat', '7-Sun'])",
        "C(Q('events_Christmas Day'), levels=['', 'event']):sin4_tod_daily",
        "ct1:cos4_tow_weekly"
    }.issubset(set(extra_pred_cols))


def test_forecast_simple_silverkite_freq():
    """Tests if ``forecast_silverkite`` can run with the parameters
    from ``convert_simple_silverkite_params`` on any frequency.

    Main purpose is to check that the constructed ``extra_pred_cols``
    are available for forecasting. The values of passthrough parameters
    are tested above, and test cases of ``forecast_silverkite`` show that
    they work.
    """
    # Sets recursion limit to make the test case pass for pytest 5.3.5 and above.
    # The final test_setting below has 494 terms in the model formula.
    # Patsy uses recursion to generate the model formula, so the number
    # of terms is limited by recursion depth (see https://github.com/pydata/patsy/issues/18)
    # Within pytest, the default recursion limit is 3000.
    # Setting it explicitly here makes the test pass.
    sys.setrecursionlimit(3000)

    test_settings = [
        # freq, period, forecast_horizon
        ("Y", 10, 1),           # short-range train/predict
        ("MS", 20, 2),
        ("W-TUE", 100, 20),
        ("D", 10, 1),
        ("D", 400, 30),
        ("H", 20*24, 2 * 365 * 24),  # long-range prediction to test unobserved categorical levels
        ("H", 400*24, 100*24),  # long-range training to make sure all features are present
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # suppresses test case warnings that can be ignored
        #   - The granularity of data is larger than daily. Ensure the daily events data match the timestamps
        #   - The default of the `iid` parameter will change from True to False...
        #   - R^2 score is not well-defined with less than two samples.
        #   - y_true is constant. Correlation is not defined.
        for freq, periods, forecast_horizon in test_settings:
            iteration = f"freq={freq}, periods={periods}, horizon={forecast_horizon}"
            data = generate_df_for_tests(freq=freq, periods=periods)["df"]

            # trains a model
            silverkite = SimpleSilverkiteForecast()
            trained_model = silverkite.forecast_simple(
                df=data,
                time_col=cst.TIME_COL,
                value_col=cst.VALUE_COL,
                forecast_horizon=forecast_horizon)

            # checks the parameters
            parameters = silverkite.convert_params(
                df=data,
                time_col=cst.TIME_COL,
                value_col=cst.VALUE_COL,
                forecast_horizon=forecast_horizon)
            keys_to_check = [
                "time_col",
                "value_col",
                "origin_for_time_vars",
                "uncertainty_dict"
            ]
            actual = {k: v for k, v in trained_model.items() if k in keys_to_check}
            expected = {k: v for k, v in parameters.items() if k in keys_to_check}
            assert_equal(actual, expected)
            # all the requests prediction columns are included
            expected_pred_cols = set(parameters["extra_pred_cols"])
            actual_pred_cols = set(trained_model["pred_cols"])
            assert expected_pred_cols.issubset(actual_pred_cols), f"Missing prediction cols for {iteration}"

            # able to predict into the future
            result = silverkite.predict_n_no_sim(
                fut_time_num=forecast_horizon,
                trained_model=trained_model,
                freq=freq)["fut_df"]
            assert result.shape[0] == forecast_horizon, f"Wrong size for {iteration}"

            expected_cols = {"ts", "y"}
            assert expected_cols.issubset(set(result.columns)), f"Missing columns for {iteration}"


def test_forecast_simple_silverkite_changepoints():
    dl = DataLoader()
    df = dl.load_peyton_manning()
    silverkite = SimpleSilverkiteForecast()
    trained_model = silverkite.forecast_simple(
        df=df,
        time_col="ts",
        value_col="y",
        changepoints_dict={
            "method": "auto",
            "yearly_seasonality_order": 6,
            "resample_freq": "2D",
            "actual_changepoint_min_distance": "100D",
            "potential_changepoint_distance": "50D",
            "no_changepoint_proportion_from_end": 0.3
        }
    )
    changepoint_values = trained_model["changepoint_values"]
    df_length = trained_model["x_mat"]["ct1"].iloc[-1]
    cp_distance = timedelta(days=100) / (pd.to_datetime(df["ts"].iloc[-1]) - pd.to_datetime(df["ts"].iloc[0]))
    # has change points
    assert len(changepoint_values) >= 0
    # checks no change points at the end
    assert changepoint_values[-1] <= df_length * 0.7
    # checks change point distance is at least "100D"
    min_cp_dist = min([changepoint_values[i] - changepoint_values[i - 1] for i in range(1, len(changepoint_values))])
    assert min_cp_dist >= df_length * cp_distance
    # checks the ChangepointDetector class is passed to forecast_silverkite
    assert trained_model["changepoint_detector"].trend_changepoints is not None
    # checks the number of change points is consistent with the change points detected by ChangepointDetector
    cd = ChangepointDetector()
    res = cd.find_trend_changepoints(
        df=df,
        time_col="ts",
        value_col="y",
        yearly_seasonality_order=6,
        resample_freq="2D",
        actual_changepoint_min_distance="100D",
        potential_changepoint_distance="50D",
        no_changepoint_proportion_from_end=0.3
    )
    changepoint_dates = res["trend_changepoints"]
    assert len(changepoint_values) == len(changepoint_dates)
    # test ``changepoints_dict`` is None
    trained_model = silverkite.forecast_simple(
        df=df,
        time_col="ts",
        value_col="y",
        changepoints_dict=None
    )
    changepoint_values = trained_model["changepoint_values"]
    assert changepoint_values is None


def test_forecast_simple_silverkite_seasonality_changepoints():
    dl = DataLoader()
    df = dl.load_peyton_manning()
    no_change_prop = 0.3
    silverkite = SimpleSilverkiteForecast()
    trained_model = silverkite.forecast_simple(
        df=df,
        time_col="ts",
        value_col="y",
        changepoints_dict={
            "method": "auto",
            "no_changepoint_proportion_from_end": no_change_prop
        },
        seasonality_changepoints_dict={
            "no_changepoint_proportion_from_end": no_change_prop,
            "regularization_strength": 0.4
        }
    )
    changepoints = trained_model["seasonality_changepoint_dates"]
    df_length = df.shape[0]
    # has change points
    assert max([len(value) for value in changepoints.values()]) > 0
    # checks no change points at the end
    assert max([max(value) for value in changepoints.values()]) <= pd.to_datetime(df["ts"].iloc[int(df_length * (1 - no_change_prop))])


def test_get_requested_seasonality_order():
    """Tests get_requested_seasonality_order"""
    silverkite = SimpleSilverkiteForecast()
    assert silverkite._SimpleSilverkiteForecast__get_requested_seasonality_order(
        requested_seasonality="auto",
        default_order=5,
        is_enabled_auto=True) == 5
    assert silverkite._SimpleSilverkiteForecast__get_requested_seasonality_order(
        requested_seasonality="auto",
        default_order=5,
        is_enabled_auto=False) == 0

    assert silverkite._SimpleSilverkiteForecast__get_requested_seasonality_order(
        requested_seasonality=True,
        default_order=6,
        is_enabled_auto=True) == 6
    assert silverkite._SimpleSilverkiteForecast__get_requested_seasonality_order(
        requested_seasonality=True,
        default_order=6,
        is_enabled_auto=False) == 6

    assert silverkite._SimpleSilverkiteForecast__get_requested_seasonality_order(
        requested_seasonality=False,
        default_order=5,
        is_enabled_auto=True) == 0
    assert silverkite._SimpleSilverkiteForecast__get_requested_seasonality_order(
        requested_seasonality=False,
        default_order=5,
        is_enabled_auto=False) == 0

    assert silverkite._SimpleSilverkiteForecast__get_requested_seasonality_order(
        requested_seasonality=1,
        default_order=5,
        is_enabled_auto=True) == 1
    assert silverkite._SimpleSilverkiteForecast__get_requested_seasonality_order(
        requested_seasonality=2,
        default_order=5,
        is_enabled_auto=False) == 2

    with pytest.raises(ValueError, match="invalid literal for int\(\) with base 10\: \'invalid\'"):  # noqa: W605
        silverkite._SimpleSilverkiteForecast__get_requested_seasonality_order(requested_seasonality="invalid")


def test_get_silverkite_seasonality():
    """Tests get_silverkite_seasonality"""
    silverkite = SimpleSilverkiteForecast()
    fs = silverkite._SimpleSilverkiteForecast__get_silverkite_seasonality(
        simple_freq=SimpleTimeFrequencyEnum.DAY.name,
        num_days=10)
    assert fs is None

    fs = silverkite._SimpleSilverkiteForecast__get_silverkite_seasonality(
        simple_freq=SimpleTimeFrequencyEnum.DAY.name,
        num_days=100)
    cols = ["name", "period", "order", "seas_names"]
    expected = pd.DataFrame([
        ['tow', 7.0, 4, 'weekly']
    ], columns=cols)
    assert fs.equals(expected)

    fs = silverkite._SimpleSilverkiteForecast__get_silverkite_seasonality(
        simple_freq=SimpleTimeFrequencyEnum.DAY.name,
        num_days=365)
    expected = pd.DataFrame([
        ['tow', 7.0, 4, 'weekly'],
        ['toq', 1.0, 5, 'quarterly']
    ], columns=cols)
    assert fs.equals(expected)

    fs = silverkite._SimpleSilverkiteForecast__get_silverkite_seasonality(
        simple_freq=SimpleTimeFrequencyEnum.DAY.name,
        num_days=SilverkiteSeasonalityEnum.YEARLY_SEASONALITY.value.default_min_days)
    expected = pd.DataFrame([
        ['tow', 7.0, 4, 'weekly'],
        ['toq', 1.0, 5, 'quarterly'],
        ['ct1', 1.0, 15, 'yearly']
    ], columns=cols)
    assert fs.equals(expected)

    fs = silverkite._SimpleSilverkiteForecast__get_silverkite_seasonality(
        simple_freq=SimpleTimeFrequencyEnum.DAY.name,
        num_days=SilverkiteSeasonalityEnum.YEARLY_SEASONALITY.value.default_min_days,
        seasonality={"monthly_seasonality": True, "yearly_seasonality": False}
    )
    expected = pd.DataFrame([
        ['tow', 7.0, 4, 'weekly'],
        ['tom', 1.0, 2, 'monthly'],
        ['toq', 1.0, 5, 'quarterly'],
    ], columns=cols)
    assert fs.equals(expected)

    fs = silverkite._SimpleSilverkiteForecast__get_silverkite_seasonality(
        simple_freq=SimpleTimeFrequencyEnum.HOUR.name,
        num_days=SilverkiteSeasonalityEnum.YEARLY_SEASONALITY.value.default_min_days)
    expected = pd.DataFrame([
        ['tod', 24.0, 12, 'daily'],
        ['tow', 7.0, 4, 'weekly'],
        ['toq', 1.0, 5, 'quarterly'],
        ['ct1', 1.0, 15, 'yearly']
    ], columns=cols)
    assert fs.equals(expected)

    fs = silverkite._SimpleSilverkiteForecast__get_silverkite_seasonality(
        simple_freq=SimpleTimeFrequencyEnum.HOUR.name,
        num_days=SilverkiteSeasonalityEnum.YEARLY_SEASONALITY.value.default_min_days,
        seasonality={"weekly_seasonality": 6, "quarterly_seasonality": False})
    expected = pd.DataFrame([
        ['tod', 24.0, 12, 'daily'],
        ['tow', 7.0, 6, 'weekly'],
        ['ct1', 1.0, 15, 'yearly']
    ], columns=cols)
    assert fs.equals(expected)

    fs = silverkite._SimpleSilverkiteForecast__get_silverkite_seasonality(
        simple_freq=SimpleTimeFrequencyEnum.WEEK.name,
        num_days=TimeEnum.ONE_YEAR_IN_DAYS.value)
    expected = pd.DataFrame([
        ['tom', 1.0, 2, 'monthly'],
        ['toq', 1.0, 5, 'quarterly']
    ], columns=cols)
    assert fs.equals(expected)

    fs = silverkite._SimpleSilverkiteForecast__get_silverkite_seasonality(
        simple_freq=SimpleTimeFrequencyEnum.MONTH.name,
        num_days=TimeEnum.ONE_YEAR_IN_DAYS.value)
    assert fs is None

    fs = silverkite._SimpleSilverkiteForecast__get_silverkite_seasonality(
        simple_freq=SimpleTimeFrequencyEnum.QUARTER.name,
        num_days=TimeEnum.ONE_YEAR_IN_DAYS.value)
    assert fs is None

    fs = silverkite._SimpleSilverkiteForecast__get_silverkite_seasonality(
        simple_freq=SimpleTimeFrequencyEnum.YEAR.name,
        num_days=TimeEnum.ONE_YEAR_IN_DAYS.value)
    assert fs is None

    fs = silverkite._SimpleSilverkiteForecast__get_silverkite_seasonality(
        simple_freq=SimpleTimeFrequencyEnum.MULTIYEAR.name,
        num_days=TimeEnum.ONE_YEAR_IN_DAYS.value)
    assert fs is None

    with pytest.raises(
            ValueError,
            match="unknown must be one of dict_keys\(\[\'daily_seasonality\',"  # noqa: W605
                  " \'weekly_seasonality\', \'monthly_seasonality\', \'quarterly_seasonality\',"  # noqa: W605
                  " \'yearly_seasonality\'\]\)"):  # noqa: W605
        silverkite._SimpleSilverkiteForecast__get_silverkite_seasonality(
            simple_freq=SimpleTimeFrequencyEnum.WEEK.name,
            num_days=SilverkiteSeasonalityEnum.YEARLY_SEASONALITY.value.default_min_days,
            seasonality={"unknown": "auto"})

    with LogCapture(LOGGER_NAME) as log_capture:
        silverkite._SimpleSilverkiteForecast__get_silverkite_seasonality(
            simple_freq=SimpleTimeFrequencyEnum.MONTH.name,
            num_days=TimeEnum.ONE_YEAR_IN_DAYS.value,
            seasonality={
                "weekly_seasonality": 3
            }
        )
        log_capture.check(
            (LOGGER_NAME,
             "WARNING",
             "'weekly_seasonality' is typically not valid for data with 'MONTH' "
             "frequency. Each seasonality period should cover multiple observations in the "
             "data. To remove these seasonality terms from the model, remove "
             "weekly_seasonality=3 or set it to 'auto' or 0.")
        )


def test_get_seasonality_order_from_dataframe():
    """Tests get_seasonality_order_from_dataframe"""
    silverkite = SimpleSilverkiteForecast()
    fs = silverkite._SimpleSilverkiteForecast__get_silverkite_seasonality(
        simple_freq=SimpleTimeFrequencyEnum.DAY.name,
        num_days=SilverkiteSeasonalityEnum.YEARLY_SEASONALITY.value.default_min_days)

    # properly fetches seasonality order
    assert silverkite._SimpleSilverkiteForecast__get_seasonality_order_from_dataframe(
        seasonality=SilverkiteSeasonalityEnum.WEEKLY_SEASONALITY.value,
        fs=fs) == SilverkiteSeasonalityEnum.WEEKLY_SEASONALITY.value.order
    assert silverkite._SimpleSilverkiteForecast__get_seasonality_order_from_dataframe(
        seasonality=SilverkiteSeasonalityEnum.QUARTERLY_SEASONALITY.value,
        fs=fs) == SilverkiteSeasonalityEnum.QUARTERLY_SEASONALITY.value.order
    assert silverkite._SimpleSilverkiteForecast__get_seasonality_order_from_dataframe(
        seasonality=SilverkiteSeasonalityEnum.YEARLY_SEASONALITY.value,
        fs=fs) == SilverkiteSeasonalityEnum.YEARLY_SEASONALITY.value.order
    # max_order
    assert silverkite._SimpleSilverkiteForecast__get_seasonality_order_from_dataframe(
        seasonality=SilverkiteSeasonalityEnum.WEEKLY_SEASONALITY.value,
        fs=fs,
        max_order=1) == 1

    # returns 0 if seasonality is not found in the dataframe
    assert silverkite._SimpleSilverkiteForecast__get_seasonality_order_from_dataframe(
        seasonality=SilverkiteSeasonalityEnum.DAILY_SEASONALITY.value,
        fs=fs) == 0

    seas = SilverkiteSeasonality(
        name="ct1",
        period=1.0,
        order=2,
        seas_names="another_one",
        default_min_days=10)
    assert silverkite._SimpleSilverkiteForecast__get_seasonality_order_from_dataframe(
        seasonality=seas,
        fs=fs) == 0

    # returns 0 if there is no dataframe
    assert silverkite._SimpleSilverkiteForecast__get_seasonality_order_from_dataframe(
        seasonality=SilverkiteSeasonalityEnum.WEEKLY_SEASONALITY.value,
        fs=None) == 0


def test_get_feature_sets_enabled():
    """Tests get_feature_sets_enabled"""
    silverkite = SimpleSilverkiteForecast()

    # daily data, 1+ years
    feature_sets_enabled = silverkite._SimpleSilverkiteForecast__get_feature_sets_enabled(
        simple_freq=SimpleTimeFrequencyEnum.DAY.name,
        num_days=1000,
        feature_sets_enabled="auto")
    expected = {
        SilverkiteColumn.COLS_HOUR_OF_WEEK: False,
        SilverkiteColumn.COLS_WEEKEND_SEAS: False,
        SilverkiteColumn.COLS_DAY_OF_WEEK_SEAS: False,
        SilverkiteColumn.COLS_TREND_DAILY_SEAS: False,
        SilverkiteColumn.COLS_EVENT_SEAS: False,
        SilverkiteColumn.COLS_EVENT_WEEKEND_SEAS: False,
        SilverkiteColumn.COLS_DAY_OF_WEEK: True,
        SilverkiteColumn.COLS_TREND_WEEKEND: True,
        SilverkiteColumn.COLS_TREND_DAY_OF_WEEK: True,
        SilverkiteColumn.COLS_TREND_WEEKLY_SEAS: True,
    }
    assert feature_sets_enabled == expected

    # daily data, less than 1 quarter
    feature_sets_enabled = silverkite._SimpleSilverkiteForecast__get_feature_sets_enabled(
        simple_freq=SimpleTimeFrequencyEnum.DAY.name,
        num_days=40,
        feature_sets_enabled="auto")
    expected = {
        SilverkiteColumn.COLS_HOUR_OF_WEEK: False,
        SilverkiteColumn.COLS_WEEKEND_SEAS: False,
        SilverkiteColumn.COLS_DAY_OF_WEEK_SEAS: False,
        SilverkiteColumn.COLS_TREND_DAILY_SEAS: False,
        SilverkiteColumn.COLS_EVENT_SEAS: False,
        SilverkiteColumn.COLS_EVENT_WEEKEND_SEAS: False,
        SilverkiteColumn.COLS_DAY_OF_WEEK: True,
        SilverkiteColumn.COLS_TREND_WEEKEND: True,
        SilverkiteColumn.COLS_TREND_DAY_OF_WEEK: False,
        SilverkiteColumn.COLS_TREND_WEEKLY_SEAS: False,
    }
    assert feature_sets_enabled == expected

    # hourly data, 3+ years
    feature_sets_enabled = silverkite._SimpleSilverkiteForecast__get_feature_sets_enabled(
        simple_freq=SimpleTimeFrequencyEnum.HOUR.name,
        num_days=2000,
        feature_sets_enabled="auto")
    expected = {
        SilverkiteColumn.COLS_HOUR_OF_WEEK: True,
        SilverkiteColumn.COLS_WEEKEND_SEAS: True,
        SilverkiteColumn.COLS_DAY_OF_WEEK_SEAS: True,
        SilverkiteColumn.COLS_TREND_DAILY_SEAS: True,
        SilverkiteColumn.COLS_EVENT_SEAS: False,
        SilverkiteColumn.COLS_EVENT_WEEKEND_SEAS: True,
        SilverkiteColumn.COLS_DAY_OF_WEEK: True,
        SilverkiteColumn.COLS_TREND_WEEKEND: True,
        SilverkiteColumn.COLS_TREND_DAY_OF_WEEK: True,
        SilverkiteColumn.COLS_TREND_WEEKLY_SEAS: True,
    }
    assert feature_sets_enabled == expected

    # hourly data, less than 1 year
    feature_sets_enabled = silverkite._SimpleSilverkiteForecast__get_feature_sets_enabled(
        simple_freq=SimpleTimeFrequencyEnum.HOUR.name,
        num_days=101,
        feature_sets_enabled="auto")
    expected = {
        SilverkiteColumn.COLS_HOUR_OF_WEEK: True,
        SilverkiteColumn.COLS_WEEKEND_SEAS: True,
        SilverkiteColumn.COLS_DAY_OF_WEEK_SEAS: True,
        SilverkiteColumn.COLS_TREND_DAILY_SEAS: True,
        SilverkiteColumn.COLS_EVENT_SEAS: True,
        SilverkiteColumn.COLS_EVENT_WEEKEND_SEAS: False,
        SilverkiteColumn.COLS_DAY_OF_WEEK: True,
        SilverkiteColumn.COLS_TREND_WEEKEND: True,
        SilverkiteColumn.COLS_TREND_DAY_OF_WEEK: True,
        SilverkiteColumn.COLS_TREND_WEEKLY_SEAS: False,
    }
    assert feature_sets_enabled == expected

    # checks if `feature_sets_enabled` can be used to override settings
    feature_sets_enabled = silverkite._SimpleSilverkiteForecast__get_feature_sets_enabled(
        simple_freq=SimpleTimeFrequencyEnum.HOUR.name,
        num_days=101,
        feature_sets_enabled={
            SilverkiteColumn.COLS_DAY_OF_WEEK_SEAS: "auto",
            SilverkiteColumn.COLS_EVENT_SEAS: False,
            SilverkiteColumn.COLS_EVENT_WEEKEND_SEAS: True,
            SilverkiteColumn.COLS_TREND_WEEKEND: None,
            SilverkiteColumn.COLS_TREND_WEEKLY_SEAS: "auto",
        })
    expected = {
        SilverkiteColumn.COLS_HOUR_OF_WEEK: True,
        SilverkiteColumn.COLS_WEEKEND_SEAS: True,
        SilverkiteColumn.COLS_DAY_OF_WEEK_SEAS: True,
        SilverkiteColumn.COLS_TREND_DAILY_SEAS: True,
        SilverkiteColumn.COLS_EVENT_SEAS: False,  # updated
        SilverkiteColumn.COLS_EVENT_WEEKEND_SEAS: True,  # updated
        SilverkiteColumn.COLS_DAY_OF_WEEK: True,
        SilverkiteColumn.COLS_TREND_WEEKEND: False,  # updated
        SilverkiteColumn.COLS_TREND_DAY_OF_WEEK: True,
        SilverkiteColumn.COLS_TREND_WEEKLY_SEAS: False,
    }
    assert feature_sets_enabled == expected

    # weekly data
    feature_sets_enabled = silverkite._SimpleSilverkiteForecast__get_feature_sets_enabled(
        simple_freq=SimpleTimeFrequencyEnum.WEEK.name,
        num_days=60,
        feature_sets_enabled="auto")
    expected = {
        SilverkiteColumn.COLS_HOUR_OF_WEEK: False,
        SilverkiteColumn.COLS_WEEKEND_SEAS: False,
        SilverkiteColumn.COLS_DAY_OF_WEEK_SEAS: False,
        SilverkiteColumn.COLS_TREND_DAILY_SEAS: False,
        SilverkiteColumn.COLS_EVENT_SEAS: False,
        SilverkiteColumn.COLS_EVENT_WEEKEND_SEAS: False,
        SilverkiteColumn.COLS_DAY_OF_WEEK: False,
        SilverkiteColumn.COLS_TREND_WEEKEND: False,
        SilverkiteColumn.COLS_TREND_DAY_OF_WEEK: False,
        SilverkiteColumn.COLS_TREND_WEEKLY_SEAS: False,
    }
    assert feature_sets_enabled == expected

    # feature_sets_enabled=False
    feature_sets_enabled = silverkite._SimpleSilverkiteForecast__get_feature_sets_enabled(
        simple_freq=SimpleTimeFrequencyEnum.HOUR.name,
        num_days=101,
        feature_sets_enabled=False)
    assert feature_sets_enabled == expected
    # feature_sets_enabled=None
    feature_sets_enabled = silverkite._SimpleSilverkiteForecast__get_feature_sets_enabled(
        simple_freq=SimpleTimeFrequencyEnum.HOUR.name,
        num_days=101,
        feature_sets_enabled=None)
    assert feature_sets_enabled == expected
    # feature_sets_enabled=True
    feature_sets_enabled = silverkite._SimpleSilverkiteForecast__get_feature_sets_enabled(
        simple_freq=SimpleTimeFrequencyEnum.HOUR.name,
        num_days=101,
        feature_sets_enabled=True)
    assert feature_sets_enabled == {
        SilverkiteColumn.COLS_HOUR_OF_WEEK: True,
        SilverkiteColumn.COLS_WEEKEND_SEAS: True,
        SilverkiteColumn.COLS_DAY_OF_WEEK_SEAS: True,
        SilverkiteColumn.COLS_TREND_DAILY_SEAS: True,
        SilverkiteColumn.COLS_EVENT_SEAS: True,
        SilverkiteColumn.COLS_EVENT_WEEKEND_SEAS: True,
        SilverkiteColumn.COLS_DAY_OF_WEEK: True,
        SilverkiteColumn.COLS_TREND_WEEKEND: True,
        SilverkiteColumn.COLS_TREND_DAY_OF_WEEK: True,
        SilverkiteColumn.COLS_TREND_WEEKLY_SEAS: True,
    }

    # Checks exceptions
    with pytest.raises(ValueError, match="Unrecognized feature set: 'unknown_feature_set'"):
        silverkite._SimpleSilverkiteForecast__get_feature_sets_enabled(
            simple_freq=SimpleTimeFrequencyEnum.HOUR.name,
            num_days=101,
            feature_sets_enabled={
                "unknown_feature_set": True
            })
    with pytest.raises(ValueError, match=f"Unrecognized `feature_sets_enabled` dictionary value for key {SilverkiteColumn.COLS_TREND_WEEKLY_SEAS}: "
                                         f"expected bool or 'auto' or None. Found: off"):
        silverkite._SimpleSilverkiteForecast__get_feature_sets_enabled(
            simple_freq=SimpleTimeFrequencyEnum.HOUR.name,
            num_days=101,
            feature_sets_enabled={
                SilverkiteColumn.COLS_TREND_WEEKLY_SEAS: "off"
            })
    with pytest.raises(ValueError, match="Unrecognized type for `feature_sets_enabled`: expected bool, dict, 'auto', or None. Found: on"):
        silverkite._SimpleSilverkiteForecast__get_feature_sets_enabled(
            simple_freq=SimpleTimeFrequencyEnum.HOUR.name,
            num_days=101,
            feature_sets_enabled="on")


def test_get_feature_sets_terms():
    """Tests get_feature_sets_terms"""
    silverkite = SimpleSilverkiteForecast()
    daily_seas_interaction_order = 4
    weekly_seas_interaction_order = 3
    growth_term = "ct2"
    changepoint_cols = [f"{CHANGEPOINT_COL_PREFIX}0", f"{CHANGEPOINT_COL_PREFIX}1"]
    num_trend_terms = 1 + len(changepoint_cols)  # growth plus changepoints
    holidays = silverkite._SimpleSilverkiteForecast__get_silverkite_holidays(
        holiday_lookup_countries="auto",
        holidays_to_model_separately="auto",
        start_year=2017,
        end_year=2025,
        pre_num=2,
        post_num=2)
    feature_sets_terms = silverkite._SimpleSilverkiteForecast__get_feature_sets_terms(
        daily_event_df_dict=holidays,
        daily_seas_interaction_order=daily_seas_interaction_order,
        weekly_seas_interaction_order=weekly_seas_interaction_order,
        growth_term=growth_term,
        changepoint_cols=changepoint_cols)
    num_interaction_holidays = len([k for k in holidays.keys() if k in SilverkiteHoliday.HOLIDAYS_TO_INTERACT])

    expected = {
        # number of terms, substring of first term
        SilverkiteColumn.COLS_HOUR_OF_WEEK: (1, "C(Q('dow_hr'), levels=['1_00',"),
        SilverkiteColumn.COLS_WEEKEND_SEAS: (2 * daily_seas_interaction_order, "is_weekend:sin1_tod_daily"),
        SilverkiteColumn.COLS_DAY_OF_WEEK_SEAS: (2 * daily_seas_interaction_order, "C(Q('str_dow'), levels=['1-Mon"),
        SilverkiteColumn.COLS_TREND_DAILY_SEAS: (2 * daily_seas_interaction_order * num_trend_terms, f"is_weekend:{growth_term}:sin1_tod_daily"),
        SilverkiteColumn.COLS_EVENT_SEAS: (2 * daily_seas_interaction_order * num_interaction_holidays, "C(Q('events_Christmas Day'), l"),
        SilverkiteColumn.COLS_EVENT_WEEKEND_SEAS: (2 * daily_seas_interaction_order * num_interaction_holidays, "is_weekend:C(Q('events_Christm"),
        SilverkiteColumn.COLS_DAY_OF_WEEK: (1, "C(Q('str_dow'), levels=['1-Mon"),
        SilverkiteColumn.COLS_TREND_WEEKEND: (num_trend_terms, f"is_weekend:{growth_term}"),
        SilverkiteColumn.COLS_TREND_DAY_OF_WEEK: (num_trend_terms, "C(Q('str_dow'), levels=['1-Mon"),
        SilverkiteColumn.COLS_TREND_WEEKLY_SEAS: (2 * weekly_seas_interaction_order * num_trend_terms, "ct2:sin1_tow_weekly"),
    }
    assert sorted(list(feature_sets_terms.keys())) == sorted(list(expected.keys()))
    for feature_set, (expected_len, expected_term) in expected.items():
        assert len(feature_sets_terms[feature_set]) == expected_len
        if expected_len > 0:
            assert feature_sets_terms[feature_set][0][:30] == expected_term

    # if there are events and seasonality but no growth
    feature_sets_terms = silverkite._SimpleSilverkiteForecast__get_feature_sets_terms(
        daily_event_df_dict=holidays,
        daily_seas_interaction_order=daily_seas_interaction_order,
        weekly_seas_interaction_order=weekly_seas_interaction_order,
        growth_term=None,
        changepoint_cols=None)
    expected = {
        # number of terms, substring of first term
        SilverkiteColumn.COLS_HOUR_OF_WEEK: (1, "C(Q('dow_hr'), levels=['1_00',"),
        SilverkiteColumn.COLS_WEEKEND_SEAS: (2 * daily_seas_interaction_order, "is_weekend:sin1_tod_daily"),
        SilverkiteColumn.COLS_DAY_OF_WEEK_SEAS: (2 * daily_seas_interaction_order, "C(Q('str_dow'), levels=['1-Mon"),
        SilverkiteColumn.COLS_TREND_DAILY_SEAS: (0, None),
        SilverkiteColumn.COLS_EVENT_SEAS: (
            2 * daily_seas_interaction_order * num_interaction_holidays, "C(Q('events_Christmas Day'), l"),
        SilverkiteColumn.COLS_EVENT_WEEKEND_SEAS: (
            2 * daily_seas_interaction_order * num_interaction_holidays, "is_weekend:C(Q('events_Christm"),
        SilverkiteColumn.COLS_DAY_OF_WEEK: (1, "C(Q('str_dow'), levels=['1-Mon"),
        SilverkiteColumn.COLS_TREND_WEEKEND: (0, None),
        SilverkiteColumn.COLS_TREND_DAY_OF_WEEK: (0, None),
        SilverkiteColumn.COLS_TREND_WEEKLY_SEAS: (0, None),
    }
    assert sorted(list(feature_sets_terms.keys())) == sorted(list(expected.keys()))
    for feature_set, (expected_len, expected_term) in expected.items():
        assert len(feature_sets_terms[feature_set]) == expected_len
        if expected_len > 0:
            assert feature_sets_terms[feature_set][0][:30] == expected_term

    # if there are no events, seasonality, or growth
    feature_sets_terms = silverkite._SimpleSilverkiteForecast__get_feature_sets_terms(
        daily_event_df_dict=None,
        daily_seas_interaction_order=0,
        weekly_seas_interaction_order=0,
        growth_term=None,
        changepoint_cols=None)
    expected = {
        # number of terms, substring of first term
        SilverkiteColumn.COLS_HOUR_OF_WEEK: (1, "C(Q('dow_hr'), levels=['1_00',"),
        SilverkiteColumn.COLS_WEEKEND_SEAS: (0, None),
        SilverkiteColumn.COLS_DAY_OF_WEEK_SEAS: (0, None),
        SilverkiteColumn.COLS_TREND_DAILY_SEAS: (0, None),
        SilverkiteColumn.COLS_EVENT_SEAS: (0, None),
        SilverkiteColumn.COLS_EVENT_WEEKEND_SEAS: (0, None),
        SilverkiteColumn.COLS_DAY_OF_WEEK: (1, "C(Q('str_dow'), levels=['1-Mon"),
        SilverkiteColumn.COLS_TREND_WEEKEND: (0, None),
        SilverkiteColumn.COLS_TREND_DAY_OF_WEEK: (0, None),
        SilverkiteColumn.COLS_TREND_WEEKLY_SEAS: (0, None),
    }
    assert sorted(list(feature_sets_terms.keys())) == sorted(list(expected.keys()))
    for feature_set, (expected_len, expected_term) in expected.items():
        assert len(feature_sets_terms[feature_set]) == expected_len
        if expected_len > 0:
            assert feature_sets_terms[feature_set][0][:30] == expected_term


def test_get_silverkite_holidays():
    """Tests get_silverkite_holidays"""
    silverkite = SimpleSilverkiteForecast()
    # tests default values
    holidays = silverkite._SimpleSilverkiteForecast__get_silverkite_holidays(
        holiday_lookup_countries="auto",
        holidays_to_model_separately="auto",
        start_year=2017,
        end_year=2025,
        pre_num=2,
        post_num=2)
    expected = generate_holiday_events(
        countries=SilverkiteHoliday.HOLIDAY_LOOKUP_COUNTRIES_AUTO,
        holidays_to_model_separately=SilverkiteHoliday.HOLIDAYS_TO_MODEL_SEPARATELY_AUTO,
        year_start=2016,
        year_end=2026,
        pre_num=2,
        post_num=2)
    assert_equal(holidays, expected)

    # tests ALL_HOLIDAYS_IN_COUNTRIES option
    holiday_lookup_countries = ["UK", "CN", "IN"]
    holidays = silverkite._SimpleSilverkiteForecast__get_silverkite_holidays(
        holiday_lookup_countries=holiday_lookup_countries,
        holidays_to_model_separately=SilverkiteHoliday.ALL_HOLIDAYS_IN_COUNTRIES,
        start_year=2017,
        end_year=2025,
        pre_num=1,
        post_num=3)
    holidays_to_model_separately = get_available_holidays_across_countries(
        countries=holiday_lookup_countries,
        year_start=2016,
        year_end=2026)
    expected = generate_holiday_events(
        countries=holiday_lookup_countries,
        holidays_to_model_separately=holidays_to_model_separately,
        year_start=2016,
        year_end=2026,
        pre_num=1,
        post_num=3)
    assert_equal(holidays, expected)

    # custom values
    holiday_lookup_countries = ["US"]
    holidays_to_model_separately = ["New Year's Day", "Christmas Day"]
    pre_post_num_dict = {
        "New Year's Day": (7, 3),
        "Christmas Day": (3, 3)
    }
    holidays = silverkite._SimpleSilverkiteForecast__get_silverkite_holidays(
        holiday_lookup_countries=holiday_lookup_countries,
        holidays_to_model_separately=holidays_to_model_separately,
        start_year=2021,
        end_year=2022,
        pre_num=2,
        post_num=2,
        pre_post_num_dict=pre_post_num_dict)
    expected = generate_holiday_events(
        countries=holiday_lookup_countries,
        holidays_to_model_separately=holidays_to_model_separately,
        year_start=2020,
        year_end=2023,
        pre_num=2,
        post_num=2,
        pre_post_num_dict=pre_post_num_dict)
    assert_equal(holidays, expected)

    # singleton string
    holiday_lookup_countries = "India"
    holidays_to_model_separately = "New Year's Day"
    with pytest.raises(
            ValueError,
            match="`holiday_lookup_countries` should be a list, found India"):
        silverkite._SimpleSilverkiteForecast__get_silverkite_holidays(
            holiday_lookup_countries=holiday_lookup_countries,
            holidays_to_model_separately="auto",
            start_year=2021,
            end_year=2022,
            pre_num=2,
            post_num=2)
    with pytest.raises(
            ValueError,
            match="`holidays_to_model_separately` should be a list, found New Year's Day"):
        silverkite._SimpleSilverkiteForecast__get_silverkite_holidays(
            holiday_lookup_countries="auto",
            holidays_to_model_separately=holidays_to_model_separately,
            start_year=2021,
            end_year=2022,
            pre_num=2,
            post_num=2)

    # empty list or None
    holiday_lookup_countries = None
    holidays_to_model_separately = None
    holidays = silverkite._SimpleSilverkiteForecast__get_silverkite_holidays(
        holiday_lookup_countries=holiday_lookup_countries,
        holidays_to_model_separately=holidays_to_model_separately,
        start_year=2021,
        end_year=2022,
        pre_num=2,
        post_num=2)
    assert holidays is None

    holiday_lookup_countries = []
    holidays_to_model_separately = "auto"
    holidays = silverkite._SimpleSilverkiteForecast__get_silverkite_holidays(
        holiday_lookup_countries=holiday_lookup_countries,
        holidays_to_model_separately=holidays_to_model_separately,
        start_year=2021,
        end_year=2022,
        pre_num=2,
        post_num=2)
    assert holidays is None

    holiday_lookup_countries = "auto"
    holidays_to_model_separately = []
    holidays = silverkite._SimpleSilverkiteForecast__get_silverkite_holidays(
        holiday_lookup_countries=holiday_lookup_countries,
        holidays_to_model_separately=holidays_to_model_separately,
        start_year=2021,
        end_year=2022,
        pre_num=0,
        post_num=0)
    assert list(holidays.keys()) == ["Other"]


def test_auto_config_params(daily_data_reg):
    """Tests the auto options:

        - auto_growth
        - auto_holiday
        - auto_seasonality

    """
    dl = DataLoader()
    df = dl.load_peyton_manning()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    silverkite = SimpleSilverkiteForecast()
    params = silverkite.convert_params(
        df=df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        forecast_horizon=7,
        auto_holiday=True,
        holidays_to_model_separately="auto",
        holiday_lookup_countries="auto",
        holiday_pre_num_days=2,
        holiday_post_num_days=2,
        daily_event_df_dict=dict(
            custom_event=pd.DataFrame({
                EVENT_DF_DATE_COL: pd.to_datetime(["2010-03-03", "2011-03-03", "2012-03-03"]),
                EVENT_DF_LABEL_COL: "threethree"
            })
        ),
        auto_growth=True,
        growth_term="quadratic",
        changepoints_dict=dict(
            method="uniform",
            n_changepoints=2
        ),
        auto_seasonality=True,
        yearly_seasonality=0,
        quarterly_seasonality="auto",
        monthly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=5
    )

    # Seasonality is overridden by auto seasonality.
    # Monthly is forced to be 0 because the value if `False`.
    assert params["fs_components_df"].equals(pd.DataFrame({
        "name": ["tow", "toq", "ct1"],
        "period": [7.0, 1.0, 1.0],
        "order": [3, 1, 6],
        "seas_names": ["weekly", "quarterly", "yearly"]
    }))
    # Growth is overridden by auto growth.
    assert "ct1" in params["extra_pred_cols"]
    assert params["changepoints_dict"]["method"] == "custom"
    # Holidays is overridden by auto seasonality.
    assert len(params["daily_event_df_dict"]) == 198
    assert "custom_event" in params["daily_event_df_dict"]
    assert "China_Chinese New Year" in params["daily_event_df_dict"]


def test_auto_config_run(daily_data_reg):
    """Tests the auto options:

        - auto_growth
        - auto_holiday
        - auto_seasonality

    """
    dl = DataLoader()
    df = dl.load_peyton_manning()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    silverkite = SimpleSilverkiteForecast()
    silverkite.forecast_simple(
        df=df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        forecast_horizon=7,
        auto_holiday=True,
        holidays_to_model_separately="auto",
        holiday_lookup_countries="auto",
        holiday_pre_num_days=2,
        holiday_post_num_days=2,
        daily_event_df_dict=dict(
            custom_event=pd.DataFrame({
                EVENT_DF_DATE_COL: pd.to_datetime(["2010-03-03", "2011-03-03", "2012-03-03"]),
                EVENT_DF_LABEL_COL: "event"
            })
        ),
        auto_growth=True,
        growth_term="quadratic",
        changepoints_dict=dict(
            method="uniform",
            n_changepoints=2
        ),
        auto_seasonality=True,
        yearly_seasonality=0,
        quarterly_seasonality="auto",
        monthly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=5
    )


def test_quantile_regression():
    """Tests quantile regression fit algorithm."""
    dl = DataLoader()
    df = dl.load_peyton_manning()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    silverkite = SimpleSilverkiteForecast()
    trained_model = silverkite.forecast_simple(
        df=df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        forecast_horizon=7,
        holidays_to_model_separately="auto",
        holiday_lookup_countries="auto",
        holiday_pre_num_days=2,
        holiday_post_num_days=2,
        growth_term="linear",
        changepoints_dict=dict(
            method="uniform",
            n_changepoints=2
        ),
        yearly_seasonality=10,
        quarterly_seasonality=False,
        monthly_seasonality=False,
        weekly_seasonality=4,
        daily_seasonality=False,
        fit_algorithm="quantile_regression",
        fit_algorithm_params={
            "quantile": 0.9,
            "alpha": 0
        }
    )
    pred = silverkite.predict(df, trained_model=trained_model)["fut_df"]
    assert round(sum(pred[VALUE_COL] > df[VALUE_COL]) / len(pred), 1) == 0.9
