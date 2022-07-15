import numpy as np
import pandas as pd
import pytest
from holidays_ext.get_holidays import get_holiday_df
from testfixtures import LogCapture

from greykite.algo.common.holiday_inferrer import HolidayInferrer
from greykite.common.constants import EVENT_DF_DATE_COL
from greykite.common.constants import EVENT_DF_LABEL_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.data_loader import DataLoader
from greykite.common.logging import LOGGER_NAME


@pytest.fixture
def daily_df():
    df = DataLoader().load_peyton_manning()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    return df


def test_init():
    """Tests instantiation."""
    hi = HolidayInferrer()
    assert hi.baseline_offsets is None
    assert hi.post_search_days is None
    assert hi.pre_search_days is None
    assert hi.independent_holiday_thres is None
    assert hi.together_holiday_thres is None
    assert hi.df is None
    assert hi.time_col is None
    assert hi.value_col is None
    assert hi.year_start is None
    assert hi.year_end is None
    assert hi.ts is None
    assert hi.country_holiday_df is None
    assert hi.holidays is None
    assert hi.score_result is None
    assert hi.score_result_avg is None
    assert hi.result is None


def test_inferrer_super_daily_data():
    """Tests holiday infer for super daily data."""
    df = pd.DataFrame({
        TIME_COL: pd.date_range("2020-01-01", freq="W", periods=100),
        VALUE_COL: list(range(100))
    })
    with LogCapture(LOGGER_NAME) as log_capture:
        hi = HolidayInferrer()
        result = hi.infer_holidays(
            df=df
        )
        # Parameters are set as default.
        assert hi.baseline_offsets == (-7, 7)
        assert hi.post_search_days == 2
        assert hi.pre_search_days == 2
        assert hi.independent_holiday_thres == 0.8
        assert hi.together_holiday_thres == 0.99
        # Super daily data do not have holidays.
        log_capture.check_present((
            LOGGER_NAME,
            "INFO",
            "Data frequency is greater than daily, "
            "holiday inferring is skipped."
        ))
        assert result is None


def test_infer_daily_data(daily_df):
    """Tests on daily data."""
    hi = HolidayInferrer()
    result = hi.infer_holidays(
        df=daily_df,
        plot=True
    )
    # Checks result.
    assert len(result["scores"]) == 50
    assert sorted(result["independent_holidays"]) == sorted([
        ('US', 'Labor Day_+0'),
        ('US', 'Labor Day_-1'),
        ('US', 'Christmas Day_+0'),
        ('US', 'Martin Luther King Jr. Day_+0'),
        ('US', "Washington's Birthday_-1"),
        ('US', 'Thanksgiving_-2'),
        ('US', "Washington's Birthday_+1"),
        ('US', 'Veterans Day_-2'),
        ('US', "Washington's Birthday_+2"),
        ('US', 'Christmas Day_+1'),
        ('US', "New Year's Day_+1"),
        ('US', 'Memorial Day_+0'),
        ('US', 'Veterans Day_+0'),
        ('US', 'Veterans Day_-1'),
        ('US', "Washington's Birthday_-2"),
        ('US', 'Thanksgiving_-1'),
        ('US', 'Labor Day_-2'),
        ('US', 'Columbus Day_+0'),
        ('US', 'Memorial Day_+1'),
        ('US', 'Labor Day_+1'),
        ('US', "New Year's Day_+0"),
        ('US', 'Martin Luther King Jr. Day_-1'),
        ('US', 'Independence Day_-2'),
        ('US', 'Christmas Day_-1'),
        ('US', 'Independence Day_-1'),
    ])

    assert sorted(result["together_holidays_positive"]) == sorted([
        ('US', 'Martin Luther King Jr. Day_+1'),
        ('US', 'Martin Luther King Jr. Day_+2'),
        ('US', 'Labor Day_+2'),
        ('US', 'Thanksgiving_+1'),
        ('US', 'Memorial Day_+2'),
        ('US', "New Year's Day_-1"),
        ('US', "New Year's Day_-2"),
        ('US', "New Year's Day_+2")
    ])

    assert sorted(result["together_holidays_negative"]) == sorted([
        ('US', 'Independence Day_+1'),
        ('US', 'Independence Day_+0'),
        ('US', 'Columbus Day_+1'),
        ('US', 'Martin Luther King Jr. Day_-2'),
        ('US', 'Veterans Day_+2'),
        ('US', 'Memorial Day_-2'),
        ('US', 'Memorial Day_-1'),
        ('US', 'Columbus Day_-2'),
        ('US', 'Columbus Day_+2'),
        ('US', 'Christmas Day_-2'),
        ('US', 'Christmas Day_+2'),
        ('US', 'Independence Day_+2'),
    ])

    assert len(result["fig"].data) == 6
    # Checks attributes.
    assert hi.df is not None
    assert hi.time_col == TIME_COL
    assert hi.value_col == VALUE_COL
    assert hi.year_start == 2007
    assert hi.year_end == 2016
    assert len(hi.ts) == len(hi.df)
    assert hi.country_holiday_df is not None
    assert len(hi.holidays) == 10
    assert len(hi.score_result) == 50
    assert len(hi.score_result_avg) == 50
    assert hi.result == result


def test_daily_data_diff_params(daily_df):
    """Tests daily data with different parameters."""
    hi = HolidayInferrer()
    result = hi.infer_holidays(
        df=daily_df,
        plot=True,
        countries=["US", "UK", "India"],
        pre_search_days=3,
        post_search_days=0,
        baseline_offsets=[-14, -7, 7, 14],
        independent_holiday_thres=0.6,
        together_holiday_thres=0.8
    )
    # Checks result.
    assert len(result["scores"]) == 196
    assert "US_Christmas Day_+0" in result["scores"]
    assert "US_New Year's Day_-3" in result["scores"]
    assert "India_All Saints Day_-1" in result["scores"]
    assert len(result["fig"].data) == 5
    assert len(result["independent_holidays"]) == 104
    assert len(result["together_holidays_positive"]) == 10
    assert len(result["together_holidays_negative"]) == 9
    # Checks attributes.
    assert hi.df is not None
    assert hi.time_col == TIME_COL
    assert hi.value_col == VALUE_COL
    assert hi.year_start == 2007
    assert hi.year_end == 2016
    assert len(hi.ts) == len(hi.df)
    assert hi.country_holiday_df is not None
    assert len(hi.holidays) == 49
    assert len(hi.score_result) == 196
    assert len(hi.score_result_avg) == 196
    assert hi.result == result


def test_sub_daily_data():
    """Tests on sub daily data."""
    np.random.seed(123)
    df = pd.DataFrame({
        TIME_COL: pd.date_range("2020-01-01", freq="H", periods=24 * 365),
        VALUE_COL: list(range(24 * 365)) + np.random.randn(24 * 365)
    })
    hi = HolidayInferrer()
    result = hi.infer_holidays(
        df=df,
        plot=True
    )
    # Checks result.
    assert len(result["scores"]) == 55
    assert len(result["fig"].data) == 6
    assert len(result["independent_holidays"]) == 7
    assert len(result["together_holidays_negative"]) == 0
    assert len(result["together_holidays_positive"]) == 1
    # Checks attributes.
    assert hi.df is not None
    assert hi.time_col == TIME_COL
    assert hi.value_col == VALUE_COL
    assert hi.year_start == 2020
    assert hi.year_end == 2020
    assert len(hi.ts) == len(hi.df)
    assert hi.country_holiday_df is not None
    assert len(hi.holidays) == 11
    assert len(hi.score_result) == 55
    assert len(hi.score_result_avg) == 55
    assert hi.result == result


def test_errors(daily_df):
    """Tests errors."""
    hi = HolidayInferrer()
    with pytest.raises(
            ValueError,
            match="Both 'post_search_days' and 'pre_search_days' must be non-negative integers."):
        hi.infer_holidays(
            df=daily_df,
            post_search_days=-1
        )
    with pytest.raises(
            ValueError,
            match="Both 'independent_holiday_thres' and 'together_holiday_thres' must be between "
                  "0 and 1 \\(inclusive\\)."):
        hi.infer_holidays(
            df=daily_df,
            independent_holiday_thres=80
        )


def test_transform_holiday_country():
    """Tests the transformations between country-holiday strings and country-holiday tuples."""
    hi = HolidayInferrer()
    holiday_country_strs = ["US_Christmas Day", "UK_New Year's Day", "France_Thanksgiving"]
    holiday_country_tuples = [
        ("US", "Christmas Day"),
        ("UK", "New Year's Day"),
        ("France", "Thanksgiving")
    ]
    assert hi._transform_country_holidays(holiday_country_strs) == holiday_country_tuples
    assert hi._transform_country_holidays(holiday_country_tuples) == holiday_country_strs


def test_get_scores():
    """Tests the scoring functions."""
    hi = HolidayInferrer()
    # Different values on christmas/New year.
    df = pd.DataFrame({
        "ts": pd.date_range("2020-12-01", freq="D", periods=40),
        "y": [1] * 22 + [2, 3, 4, 3, 2, 1, 1, 2, 3, 4, 3, 2] + [1] * 6
    })
    result = hi.infer_holidays(df=df)
    assert result["scores"]["US_Christmas Day_-2"] == [0.5]
    assert result["scores"]["US_Christmas Day_-1"] == [1.0]
    assert result["scores"]["US_Christmas Day_+0"] == [1.5]
    assert result["scores"]["US_Christmas Day_+1"] == [1.0]
    assert result["scores"]["US_Christmas Day_+2"] == [0.5]
    assert result["scores"]["US_New Year's Day_-2"] == [0.5]
    assert result["scores"]["US_New Year's Day_-1"] == [1.0]
    assert result["scores"]["US_New Year's Day_+0"] == [1.5]
    assert result["scores"]["US_New Year's Day_+1"] == [1.0]
    assert result["scores"]["US_New Year's Day_+2"] == [0.0]


def test_remove_observed():
    """Tests the observed holidays are accurately renamed
    and the corresponding original holidays are removed."""
    df = pd.DataFrame({
        "ts": pd.date_range("2020-01-01", freq="D", periods=731),
        "y": 1
    })
    # 5 observed holidays are removed.
    country_holiday_df = get_holiday_df(country_list=["US"], years=[2020, 2021, 2022])
    assert len(country_holiday_df) == 39
    hi = HolidayInferrer()
    hi.infer_holidays(df=df)
    assert len(hi.country_holiday_df) == 32
    # Tests the correctness of observed holiday removal.
    # 2020-07-03 is the observed Independence Day for 2022-07-04.
    assert hi.country_holiday_df[
               hi.country_holiday_df["ts"] == pd.Timestamp("2020-07-03")]["holiday"].values == "Independence Day"
    assert len(hi.country_holiday_df[hi.country_holiday_df["ts"] == pd.Timestamp("2020-07-04")]["holiday"]) == 0
    # 2021-07-05 is the observed Independence Day for 2022-07-04.
    assert hi.country_holiday_df[
               hi.country_holiday_df["ts"] == pd.Timestamp("2021-07-05")]["holiday"].values == "Independence Day"
    assert len(hi.country_holiday_df[hi.country_holiday_df["ts"] == pd.Timestamp("2021-07-04")]["holiday"]) == 0
    # 2021-12-24 is the observed Christmas Day for 2021-12-25.
    assert hi.country_holiday_df[
               hi.country_holiday_df["ts"] == pd.Timestamp("2021-12-24")]["holiday"].values == "Christmas Day"
    assert len(hi.country_holiday_df[hi.country_holiday_df["ts"] == pd.Timestamp("2021-12-25")]["holiday"]) == 0
    # 2021-12-31 is the observed New Year's Day for 2022-01-01.
    assert hi.country_holiday_df[
               hi.country_holiday_df["ts"] == pd.Timestamp("2021-12-31")]["holiday"].values == "New Year's Day"
    assert len(hi.country_holiday_df[hi.country_holiday_df["ts"] == pd.Timestamp("2022-01-01")]["holiday"]) == 0
    # 2022-12-26 is the observed Christmas Day for 2022-12-25.
    assert hi.country_holiday_df[
               hi.country_holiday_df["ts"] == pd.Timestamp("2022-12-26")]["holiday"].values == "Christmas Day"
    assert len(hi.country_holiday_df[hi.country_holiday_df["ts"] == pd.Timestamp("2022-12-25")]["holiday"]) == 0


def test_cum_effect(daily_df):
    """Tests the cum effect satisfies the thresholds."""
    hi = HolidayInferrer()
    result = hi.infer_holidays(df=daily_df)
    cum_effects = np.nansum(np.abs(list(hi.score_result_avg.values())))
    # Independent holidays.
    # Before adding the last one, the effect should be less than the threshold.
    # After adding the last one, the effect should exceed the threshold.
    ind_effects = [hi.score_result_avg[hi._transform_country_holidays([h])[0]] for h in result["independent_holidays"]]
    independent_holiday_effects = np.nansum(np.abs(ind_effects))
    independent_holiday_effects_minus_1 = independent_holiday_effects - min([abs(x) for x in ind_effects])
    assert independent_holiday_effects_minus_1 / cum_effects < hi.independent_holiday_thres
    assert independent_holiday_effects / cum_effects >= hi.independent_holiday_thres
    # Together holidays.
    # Before adding the last one, the effect should be less than the threshold.
    # After adding the last one, the effect should exceed the threshold.
    tog_effects = [hi.score_result_avg[hi._transform_country_holidays([h])[0]]
                   for h in result["together_holidays_positive"] + result["together_holidays_negative"]]
    together_holiday_effects = independent_holiday_effects + np.nansum(np.abs(tog_effects))
    together_holiday_effects_minus_1 = together_holiday_effects - min([abs(x) for x in tog_effects])
    assert together_holiday_effects_minus_1 / cum_effects < hi.together_holiday_thres
    assert together_holiday_effects / cum_effects >= hi.together_holiday_thres


def test_get_single_event_df():
    """Tests get a single event df from a single holiday."""
    holiday1 = ("US", "Christmas Day_-2")
    holiday2 = ("US", "Christmas Day_+0")
    holiday3 = ("US", "Christmas Day_+1")
    hi = HolidayInferrer()
    hi.infer_holidays(
        df=pd.DataFrame({
            TIME_COL: pd.date_range("2020-01-01", freq="D", periods=731),
            VALUE_COL: 1
        })
    )
    df1 = hi._get_event_df_for_single_event(
        holiday=holiday1,
        country_holiday_df=hi.country_holiday_df
    )
    df2 = hi._get_event_df_for_single_event(
        holiday=holiday2,
        country_holiday_df=hi.country_holiday_df
    )
    df3 = hi._get_event_df_for_single_event(
        holiday=holiday3,
        country_holiday_df=hi.country_holiday_df
    )
    assert df1.equals(pd.DataFrame({
        EVENT_DF_DATE_COL: pd.to_datetime(["2020-12-23", "2021-12-22", "2022-12-24"]),
        EVENT_DF_LABEL_COL: "US_Christmas Day_minus_2"
    }))
    assert df2.equals(pd.DataFrame({
        EVENT_DF_DATE_COL: pd.to_datetime(["2020-12-25", "2021-12-24", "2022-12-26"]),
        EVENT_DF_LABEL_COL: "US_Christmas Day"
    }))
    assert df3.equals(pd.DataFrame({
        EVENT_DF_DATE_COL: pd.to_datetime(["2020-12-26", "2021-12-25", "2022-12-27"]),
        EVENT_DF_LABEL_COL: "US_Christmas Day_plus_1"
    }))


def test_get_daily_event_dict(daily_df):
    """Tests get the daily event dict for all holidays."""
    hi = HolidayInferrer()

    # Can not call without inferring holidays or passing parameters manually.
    with pytest.raises(
            ValueError,
            match="Both 'country_holiday_df' and 'holidays' must be given. "
                  "Alternatively, you can run 'infer_holidays' first and "
                  "they will be pulled automatically."):
        hi.generate_daily_event_dict()

    # Infers holidays and call.
    hi.infer_holidays(df=daily_df)
    daily_event_dict = hi.generate_daily_event_dict()
    assert len(daily_event_dict) == 27
    assert "Holiday_positive_group" in daily_event_dict
    assert "Holiday_negative_group" in daily_event_dict
    # Every single holiday should cover 11 years.
    for holiday, df in daily_event_dict.items():
        if holiday not in ["Holiday_positive_group", "Holiday_negative_group"]:
            assert df.shape[0] == 11

    # With customized input.
    country_holiday_df = get_holiday_df(
        country_list=["US"],
        years=[2015, 2016]
    )
    holidays = {
        "independent_holidays": [
            ("US", "Christmas Day_+0"),
            ("US", "New Year's Day_-1")
        ],
        "together_holidays_negative": [
            ("US", "Memorial Day_+2"),
            ("US", "Labor Day_+0")
        ],
        "together_holidays_positive": []
    }
    daily_event_dict = hi.generate_daily_event_dict(
        country_holiday_df=country_holiday_df,
        holiday_result=holidays
    )
    assert len(daily_event_dict) == 3
    assert "US_Christmas Day" in daily_event_dict
    assert "US_New Years Day_minus_1" in daily_event_dict
    assert "Holiday_negative_group" in daily_event_dict
    assert "Holiday_positive_group" not in daily_event_dict
    assert daily_event_dict["US_Christmas Day"].equals(
        pd.DataFrame({
            EVENT_DF_DATE_COL: pd.to_datetime(["2016-12-25", "2015-12-25"]),
            EVENT_DF_LABEL_COL: "US_Christmas Day"
        })
    )
    assert daily_event_dict["US_New Years Day_minus_1"].equals(
        pd.DataFrame({
            EVENT_DF_DATE_COL: pd.to_datetime(["2015-12-31", "2014-12-31"]),
            EVENT_DF_LABEL_COL: "US_New Years Day_minus_1"
        })
    )
    assert daily_event_dict["Holiday_negative_group"].equals(
        pd.DataFrame({
            EVENT_DF_DATE_COL: pd.to_datetime(["2016-06-01", "2015-05-27", "2016-09-05", "2015-09-07"]),
            EVENT_DF_LABEL_COL: "event"
        })
    )
