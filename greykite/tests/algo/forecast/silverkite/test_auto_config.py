import pandas as pd
import pytest

from greykite.algo.forecast.silverkite.auto_config import get_auto_growth
from greykite.algo.forecast.silverkite.auto_config import get_auto_holidays
from greykite.algo.forecast.silverkite.auto_config import get_auto_seasonality
from greykite.common.constants import EVENT_DF_DATE_COL
from greykite.common.constants import EVENT_DF_LABEL_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.data_loader import DataLoader


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
    """Tests automatic holidays."""
    custom_event = pd.DataFrame({
        EVENT_DF_DATE_COL: pd.to_datetime(["2015-03-03", "2016-03-03", "2017-03-03"]),
        EVENT_DF_LABEL_COL: "threethree"
    })
    holidays = get_auto_holidays(
        df=df_daily,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        countries=["UnitedStates"],
        daily_event_dict_override=dict(
            custom_event=custom_event
        )
    )
    assert len(holidays) == 31  # Only United States is used.
    assert holidays["custom_event"].equals(custom_event)
    assert "Holiday_positive_group" in holidays
    assert "Holiday_negative_group" in holidays
    assert "UnitedKingdom_Christmas Day_minus_1" not in holidays
    assert "UnitedStates_Labor Day" in holidays


def test_get_auto_holiday_super_daily(df_daily):
    """Tests automatic holidays for super daily data."""
    custom_event = pd.DataFrame({
        EVENT_DF_DATE_COL: pd.to_datetime(["2015-03-03", "2016-03-03", "2017-03-03"]),
        EVENT_DF_LABEL_COL: "threethree"
    })
    df = df_daily.resample("7D", on=TIME_COL).mean().reset_index(drop=False)
    # With custom event, result only has custom event.
    holidays = get_auto_holidays(
        df=df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        daily_event_dict_override=dict(
            custom_event=custom_event
        )
    )
    assert holidays == dict(
        custom_event=custom_event
    )
    # Without custom event, result is empty.
    holidays = get_auto_holidays(
        df=df,
        time_col=TIME_COL,
        value_col=VALUE_COL
    )
    assert holidays == {}


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
