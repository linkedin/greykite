import numpy as np
import pandas as pd
import pytest

from greykite.algo.common.holiday_grouper import HolidayGrouper
from greykite.algo.forecast.silverkite.auto_config import get_auto_growth
from greykite.algo.forecast.silverkite.auto_config import get_auto_holidays
from greykite.algo.forecast.silverkite.auto_config import get_auto_seasonality
from greykite.common.constants import EVENT_DF_DATE_COL
from greykite.common.constants import EVENT_DF_LABEL_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.data_loader import DataLoader
from greykite.common.features.timeseries_features import get_holidays
from greykite.common.python_utils import assert_equal


@pytest.fixture
def df_daily():
    dl = DataLoader()
    data = dl.load_peyton_manning()
    data[TIME_COL] = pd.to_datetime(data[TIME_COL])
    return data


@pytest.fixture
def df_hourly():
    dl = DataLoader()
    data = dl.load_bikesharing().rename(columns={"count": VALUE_COL})[[TIME_COL, VALUE_COL]]
    data[TIME_COL] = pd.to_datetime(data[TIME_COL])
    return data


def test_get_auto_seasonality_daily(df_daily):
    """Tests get automatic seasonality."""
    seasonality = get_auto_seasonality(
        df=df_daily,
        time_col=TIME_COL,
        value_col=VALUE_COL
    )
    assert seasonality["yearly_seasonality"] == 6
    assert seasonality["quarterly_seasonality"] == 1
    assert seasonality["monthly_seasonality"] == 1
    assert seasonality["weekly_seasonality"] == 3
    assert seasonality["daily_seasonality"] == 0


def test_get_auto_seasonality_hourly(df_hourly):
    """Tests get automatic seasonality."""
    seasonality = get_auto_seasonality(
        df=df_hourly,
        time_col=TIME_COL,
        value_col=VALUE_COL
    )
    assert seasonality["yearly_seasonality"] == 2
    assert seasonality["quarterly_seasonality"] == 1
    assert seasonality["monthly_seasonality"] == 1
    assert seasonality["weekly_seasonality"] == 1
    assert seasonality["daily_seasonality"] == 10


def test_get_auto_seasonality_override(df_daily):
    """Tests get automatic seasonality with override."""
    seasonality = get_auto_seasonality(
        df=df_daily,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        yearly_seasonality=False
    )
    assert seasonality["yearly_seasonality"] == 0
    assert seasonality["quarterly_seasonality"] == 1
    assert seasonality["monthly_seasonality"] == 1
    assert seasonality["weekly_seasonality"] == 3
    assert seasonality["daily_seasonality"] == 0


def test_get_auto_holiday(df_daily):
    """Tests automatic holidays if the return is the same as calling `HolidayGrouper`."""
    # Initializes inputs that will be used in both cases.
    start_year = 2007
    end_year = 2016
    pre_num = 2
    post_num = 2
    pre_post_num_dict = {"New Year's Day": (1, 3)}
    holiday_lookup_countries = ["US"]

    # Constructs `daily_event_df_dict` through directly calling `HolidayGrouper`.
    # Constructs `holiday_df`.
    holiday_df_dict = get_holidays(
        countries=holiday_lookup_countries,
        year_start=start_year - 1,
        year_end=end_year + 1)

    holiday_df_list = [holidays for _, holidays in holiday_df_dict.items()]
    holiday_df = pd.concat(holiday_df_list)
    # Removes the observed holidays and only keep the original holidays.
    holiday_df = holiday_df[~holiday_df[EVENT_DF_LABEL_COL].str.contains("Observed")]

    # Calls `HolidayGrouper`.
    hg = HolidayGrouper(
        df=df_daily,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        holiday_df=holiday_df,
        holiday_date_col=EVENT_DF_DATE_COL,
        holiday_name_col=EVENT_DF_LABEL_COL,
        holiday_impact_pre_num_days=pre_num,
        holiday_impact_post_num_days=post_num,
        holiday_impact_dict=pre_post_num_dict
    )

    hg.group_holidays()
    daily_event_df_dict_constructed = hg.result_dict["daily_event_df_dict"]

    # Constructs `daily_event_df_dict` through `get_auto_holidays` with country list and asserts the result is the same.
    daily_event_df_dict_from_auto_country = get_auto_holidays(
        df=df_daily,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        start_year=start_year,
        end_year=end_year,
        pre_num=pre_num,
        post_num=post_num,
        pre_post_num_dict=pre_post_num_dict,
        holiday_lookup_countries=holiday_lookup_countries,
        holidays_to_model_separately=None,
        daily_event_df_dict=None,
        auto_holiday_params=None
    )
    assert_equal(daily_event_df_dict_constructed, daily_event_df_dict_from_auto_country)

    # Constructs `daily_event_df_dict` through `get_auto_holidays` with external `daily_event_df_dict` and asserts the
    # result is the same.

    # Constructs `daily_event_df_dict_input` for input based on `holiday_df`.
    daily_event_df_dict_input = {key: value for key, value in holiday_df.groupby(EVENT_DF_LABEL_COL)}
    daily_event_df_dict_from_auto_input = get_auto_holidays(
        df=df_daily,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        start_year=start_year,
        end_year=end_year,
        pre_num=pre_num,
        post_num=post_num,
        pre_post_num_dict=pre_post_num_dict,
        holiday_lookup_countries=[],
        holidays_to_model_separately=None,
        daily_event_df_dict=daily_event_df_dict_input,
        auto_holiday_params=None
    )
    assert_equal(daily_event_df_dict_constructed, daily_event_df_dict_from_auto_input)

    # Uses 'auto_holiday_params["df"]' to input time series on which holidays are referred and asserts the
    # result is the same.
    daily_event_df_dict_from_auto_input = get_auto_holidays(
        df=None,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        start_year=start_year,
        end_year=end_year,
        pre_num=pre_num,
        post_num=post_num,
        pre_post_num_dict=pre_post_num_dict,
        holiday_lookup_countries=[],
        holidays_to_model_separately=None,
        daily_event_df_dict=daily_event_df_dict_input,
        auto_holiday_params=dict(df=df_daily)
    )
    assert_equal(daily_event_df_dict_constructed, daily_event_df_dict_from_auto_input)

    # Inputs `holiday_df` directly through `auto_holiday_params` and asserts the result is the same.
    # The `holiday_lookup_countries` should be ignored in this case.
    daily_event_df_dict_from_input_holiday_df = get_auto_holidays(
        df=df_daily,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        start_year=start_year,
        end_year=end_year,
        pre_num=pre_num,
        post_num=post_num,
        pre_post_num_dict=pre_post_num_dict,
        holiday_lookup_countries=holiday_lookup_countries,
        holidays_to_model_separately=None,
        daily_event_df_dict=None,
        auto_holiday_params=dict(
            holiday_df=holiday_df
        )
    )
    assert_equal(daily_event_df_dict_constructed, daily_event_df_dict_from_input_holiday_df)

    # When holidays are not passed in through `holiday_lookup_countries` or `daily_event_df_dict`, a `ValueError`
    # should be raised.
    with pytest.raises(ValueError, match="Holiday list needs to be specified"):
        _ = get_auto_holidays(
            df=df_daily,
            time_col=TIME_COL,
            value_col=VALUE_COL,
            start_year=start_year,
            end_year=end_year,
            pre_num=pre_num,
            post_num=post_num,
            pre_post_num_dict=pre_post_num_dict,
            holiday_lookup_countries=[],
            holidays_to_model_separately=None,
            daily_event_df_dict=None,
            auto_holiday_params=None
        )

    # When no `df` is passed in through `df` or `auto_holiday_params`, a `ValueError`
    # should be raised.
    with pytest.raises(ValueError, match="Dataframe cannot be `None` or empty"):
        _ = get_auto_holidays(
            df=None,
            time_col=TIME_COL,
            value_col=VALUE_COL,
            start_year=start_year,
            end_year=end_year,
            pre_num=pre_num,
            post_num=post_num,
            pre_post_num_dict=pre_post_num_dict,
            holiday_lookup_countries=holiday_lookup_countries,
            holidays_to_model_separately=None,
            daily_event_df_dict=None,
            auto_holiday_params=None
        )


def test_get_auto_holiday_with_holidays_to_model_separately(df_daily):
    """Tests automatic holidays with its functionality of `holidays_to_model_separately`."""

    # Initializes inputs that will be used in both cases.
    start_year = 2007
    end_year = 2016
    pre_num = 2
    post_num = 2
    holiday_lookup_countries = ["US"]
    pre_post_num_dict = {"New Year's Day": (1, 3)}
    # Uses minimum thresholds to make sure holidays in holiday groupers are all preserved.
    # e.g. holidays will not be dropped in holiday grouper due to lack of similar days/different
    # impact across years etc. We can then check if the same holiday dates are preserved
    # even when we model some of them separately.
    auto_holiday_params = dict(
        min_abs_avg_score=0,
        min_same_sign_ratio=0,
        get_suffix_func=None,
        min_n_days=1
    )

    # Constructs `daily_event_df_dict` through `get_auto_holidays` with country list.
    daily_event_df_dict_no_separate = get_auto_holidays(
        df=df_daily,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        start_year=start_year,
        end_year=end_year,
        pre_num=pre_num,
        post_num=post_num,
        pre_post_num_dict=pre_post_num_dict,
        holiday_lookup_countries=holiday_lookup_countries,
        holidays_to_model_separately=None,
        daily_event_df_dict=None,
        auto_holiday_params=auto_holiday_params
    )

    # Constructs `daily_event_df_dict` through `get_auto_holidays`, here we model "New Year's Day"
    # Separately. Notice that we also specify its neighboring days though `pre_post_num_dict`.
    # We also expect its neighboring days to be modeled separately in the final `daily_event_df_dict_with_separate`.
    daily_event_df_dict_with_separate = get_auto_holidays(
        df=df_daily,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        start_year=start_year,
        end_year=end_year,
        pre_num=pre_num,
        post_num=post_num,
        pre_post_num_dict=pre_post_num_dict,
        holiday_lookup_countries=holiday_lookup_countries,
        holidays_to_model_separately=["New Year's Day"],
        daily_event_df_dict=None,
        auto_holiday_params=auto_holiday_params
    )

    # Asserts the holiday dates are the same.
    unique_dates_no_separate = pd.concat([df for df in daily_event_df_dict_no_separate.values()])["date"]
    unique_dates_no_separate = set(unique_dates_no_separate)
    unique_dates_with_separate = pd.concat([df for df in daily_event_df_dict_with_separate.values()])["date"]
    unique_dates_with_separate = set(unique_dates_with_separate)
    assert unique_dates_no_separate == unique_dates_with_separate

    # Asserts all keys for `daily_event_df_dict_no_separate` for holiday groups are also included in
    # `daily_event_df_dict_with_separate`.
    assert daily_event_df_dict_no_separate.keys() - daily_event_df_dict_with_separate.keys() == set()

    # Checks that "New Year's Day" and its neighboring days have its own groups in `daily_event_df_dict_with_separate`.
    # The number of neighboring days depend on `pre_post_num_dict`.
    assert "New Years Day" in daily_event_df_dict_with_separate.keys()
    assert "New Years Day_minus_1" in daily_event_df_dict_with_separate.keys()
    assert "New Years Day_minus_2" not in daily_event_df_dict_with_separate.keys()
    assert "New Years Day_plus_3" in daily_event_df_dict_with_separate.keys()


def test_get_auto_holiday_non_daily_df(df_hourly):
    """Tests automatic holidays for data frequency different from daily."""

    # Initializes inputs that will be used in all cases.
    start_year = 2010
    end_year = 2019
    pre_num = 2
    post_num = 2
    holiday_lookup_countries = ["US"]
    pre_post_num_dict = {}

    # Constructs `daily_event_df_dict` through `get_auto_holidays` with country list with hourly data.
    daily_event_df_dict_hourly = get_auto_holidays(
        df=df_hourly,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        start_year=start_year,
        end_year=end_year,
        pre_num=pre_num,
        post_num=post_num,
        pre_post_num_dict=pre_post_num_dict,
        holiday_lookup_countries=holiday_lookup_countries,
        holidays_to_model_separately=None,
        daily_event_df_dict=None,
        auto_holiday_params=None
    )

    # Aggregates hourly data to daily and checks if the resulted `daily_event_df_dict` is the same
    df_tmp = df_hourly.resample("D", on=TIME_COL).agg({VALUE_COL: np.nanmean})
    df_daily_reconstructed = (df_tmp.drop(columns=TIME_COL).reset_index() if TIME_COL in df_tmp.columns
                              else df_tmp.reset_index())

    daily_event_df_dict_daily_reconstructed = get_auto_holidays(
        df=df_daily_reconstructed,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        start_year=start_year,
        end_year=end_year,
        pre_num=pre_num,
        post_num=post_num,
        pre_post_num_dict=pre_post_num_dict,
        holiday_lookup_countries=holiday_lookup_countries,
        holidays_to_model_separately=None,
        daily_event_df_dict=None,
        auto_holiday_params=None
    )

    assert_equal(daily_event_df_dict_hourly, daily_event_df_dict_daily_reconstructed)

    # Aggregates hourly data to weekly and check that the correct error will be raised.
    df_tmp = df_hourly.resample("W", on=TIME_COL).agg({VALUE_COL: np.nanmean})
    df_weekly_reconstructed = (df_tmp.drop(columns=TIME_COL).reset_index() if TIME_COL in df_tmp.columns
                               else df_tmp.reset_index())

    with pytest.raises(ValueError, match="frequency less than daily"):
        get_auto_holidays(
            df=df_weekly_reconstructed,
            time_col=TIME_COL,
            value_col=VALUE_COL,
            start_year=start_year,
            end_year=end_year,
            pre_num=pre_num,
            post_num=post_num,
            pre_post_num_dict=pre_post_num_dict,
            holiday_lookup_countries=holiday_lookup_countries,
            holidays_to_model_separately=None,
            daily_event_df_dict=None,
            auto_holiday_params=None
        )


def test_get_auto_growth_daily(df_daily):
    """Tests automatic growth."""
    growth = get_auto_growth(
        df=df_daily,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        forecast_horizon=7,
        changepoints_dict_override=dict(
            method="auto",
            dates=["2010-01-01"],
            combine_changepoint_min_distance="D",
            keep_detected=True
        )
    )
    growth_term = growth["growth_term"]
    changepoints_dict = growth["changepoints_dict"]
    assert growth_term == "linear"
    assert changepoints_dict == dict(
        method="auto",
        yearly_seasonality_order=6,
        resample_freq="7D",
        regularization_strength=0.6,
        actual_changepoint_min_distance="30D",
        potential_changepoint_n=100,
        no_changepoint_distance_from_end="30D",
        dates=["2010-01-01"],
        combine_changepoint_min_distance="D",
        keep_detected=True
    )


def test_get_auto_growth_hourly(df_hourly):
    """Tests automatic growth."""
    growth = get_auto_growth(
        df=df_hourly,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        forecast_horizon=24,
        changepoints_dict_override=dict(
            method="auto",
            dates=["2010-01-01"],
            combine_changepoint_min_distance="D",
            keep_detected=True
        )
    )
    growth_term = growth["growth_term"]
    changepoints_dict = growth["changepoints_dict"]
    print(changepoints_dict)
    assert growth_term == "linear"
    assert changepoints_dict == dict(
        method="auto",
        yearly_seasonality_order=2,
        resample_freq="7D",
        regularization_strength=0.6,
        actual_changepoint_min_distance="14D",
        potential_changepoint_n=100,
        no_changepoint_distance_from_end="14D",
        dates=["2010-01-01"],
        combine_changepoint_min_distance="D",
        keep_detected=True
    )
