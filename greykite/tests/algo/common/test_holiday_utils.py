import pandas as pd
import pytest

from greykite.algo.common.holiday_utils import add_shifted_events
from greykite.algo.common.holiday_utils import get_autoreg_holiday_interactions
from greykite.algo.common.holiday_utils import get_dow_grouped_suffix
from greykite.algo.common.holiday_utils import get_weekday_weekend_suffix
from greykite.common.constants import EVENT_DF_DATE_COL
from greykite.common.constants import EVENT_DF_LABEL_COL


def test_get_dow_grouped_suffix():
    """Tests `get_dow_grouped_suffix` function."""
    date = pd.to_datetime("2023-01-01")
    assert get_dow_grouped_suffix(date) == "_Sun"

    date = pd.to_datetime("2023-01-02")
    assert get_dow_grouped_suffix(date) == "_Mon/Fri"

    date = pd.to_datetime("2023-01-03")
    assert get_dow_grouped_suffix(date) == "_WD"

    date = pd.to_datetime("2023-01-04")
    assert get_dow_grouped_suffix(date) == "_WD"

    date = pd.to_datetime("2023-01-05")
    assert get_dow_grouped_suffix(date) == "_WD"

    date = pd.to_datetime("2023-01-06")
    assert get_dow_grouped_suffix(date) == "_Mon/Fri"

    date = pd.to_datetime("2023-01-07")
    assert get_dow_grouped_suffix(date) == "_Sat"


def test_get_weekday_weekend_suffix():
    """Tests `get_weekday_weekend_suffix` function."""
    date = pd.to_datetime("2023-01-01")
    assert get_weekday_weekend_suffix(date) == "_WE"

    date = pd.to_datetime("2023-01-02")
    assert get_weekday_weekend_suffix(date) == "_WD"

    date = pd.to_datetime("2023-01-03")
    assert get_weekday_weekend_suffix(date) == "_WD"

    date = pd.to_datetime("2023-01-04")
    assert get_weekday_weekend_suffix(date) == "_WD"

    date = pd.to_datetime("2023-01-05")
    assert get_weekday_weekend_suffix(date) == "_WD"

    date = pd.to_datetime("2023-01-06")
    assert get_weekday_weekend_suffix(date) == "_WD"

    date = pd.to_datetime("2023-01-07")
    assert get_weekday_weekend_suffix(date) == "_WE"


@pytest.fixture
def daily_event_df_dict():
    """A sample holiday configuration."""
    daily_event_df_dict = {
        "New Years Day": pd.DataFrame({
            EVENT_DF_DATE_COL: pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"]),
            EVENT_DF_LABEL_COL: "event"
        }),
        "Christmas Day": pd.DataFrame({
            EVENT_DF_DATE_COL: pd.to_datetime(["2020-12-25", "2021-12-25", "2022-12-25"]),
            EVENT_DF_LABEL_COL: "event"
        })
    }
    return daily_event_df_dict


def test_get_autoreg_holiday_interactions(daily_event_df_dict):
    """Tests `get_autoreg_holiday_interactions` function."""
    interaction_terms = get_autoreg_holiday_interactions(
        daily_event_df_dict=daily_event_df_dict,
        lag_names=["y_lag1", "y_avglag_7_14_21"]
    )
    assert interaction_terms == [
        "C(Q('events_Christmas Day'), levels=['', 'event']):y_lag1",
        "C(Q('events_New Years Day'), levels=['', 'event']):y_lag1",
        "C(Q('events_Christmas Day'), levels=['', 'event']):y_avglag_7_14_21",
        "C(Q('events_New Years Day'), levels=['', 'event']):y_avglag_7_14_21"
    ]


def test_add_shifted_events(daily_event_df_dict):
    """Tests `expand_holidays_with_lags` function."""
    shifted_events_dict = add_shifted_events(daily_event_df_dict=daily_event_df_dict, shifted_effect_lags=["1D", "-1D"])
    new_daily_event_df_dict = shifted_events_dict["new_daily_event_df_dict"]
    shifted_events_cols = shifted_events_dict["shifted_events_cols"]
    drop_pred_cols = shifted_events_dict["drop_pred_cols"]
    assert sorted(new_daily_event_df_dict.keys()) == [
        "Christmas Day",
        "Christmas Day_1D_after",
        "Christmas Day_1D_before",
        "New Years Day",
        "New Years Day_1D_after",
        "New Years Day_1D_before"
    ]
    assert sorted(shifted_events_cols) == [
        "events_Christmas Day_1D_after",
        "events_Christmas Day_1D_before",
        "events_New Years Day_1D_after",
        "events_New Years Day_1D_before"
    ]
    assert sorted(drop_pred_cols) == [
        "C(Q('events_Christmas Day_1D_after'), levels=['', 'Christmas Day_1D_after'])",
        "C(Q('events_Christmas Day_1D_before'), levels=['', 'Christmas Day_1D_before'])",
        "C(Q('events_New Years Day_1D_after'), levels=['', 'New Years Day_1D_after'])",
        "C(Q('events_New Years Day_1D_before'), levels=['', 'New Years Day_1D_before'])"
    ]
