import datetime

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from greykite.algo.forecast.silverkite.silverkite_diagnostics import SilverkiteDiagnostics
from greykite.common import constants as cst
from greykite.common.features.timeseries_features import build_time_features_df


def test_get_silverkite_components():
    """Tests get_silverkite_components function"""
    silverkite_diagnostics: SilverkiteDiagnostics = SilverkiteDiagnostics()

    # Dataframe with trend, seasonality and events
    time_col = "ts"
    # value_col name is chosen such that it contains keywords "ct" and "sin"
    # so that we can test patterns specified for each component work correctly
    value_col = "basin_impact"
    df = pd.DataFrame({
        time_col: [
            datetime.datetime(2018, 1, 1),
            datetime.datetime(2018, 1, 2),
            datetime.datetime(2018, 1, 3),
            datetime.datetime(2018, 1, 4),
            datetime.datetime(2018, 1, 5)],
        value_col: [1, 2, 3, 4, 5],
        "dummy_col": [0, 0, 0, 0, 0],
    })
    feature_df = pd.DataFrame({
        # Trend columns: growth, changepoints and interactions (total 5 columns)
        "ct1": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "ct1:tod": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "ct_sqrt": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "changepoint0_2018_01_02_00": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "changepoint1_2018_01_04_00": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        # Daily seasonality with interaction (total 4 columns)
        "sin1_tow_weekly": np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
        "cos1_tow_weekly": np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
        "is_weekend[T.True]:sin1_tow_weekly": np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
        "is_weekend[T.True]:cos1_tow_weekly": np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
        # Yearly seasonality (total 6 columns)
        "sin1_ct1_yearly": np.array([3.0, 3.0, 3.0, 3.0, 3.0]),
        "cos1_ct1_yearly": np.array([3.0, 3.0, 3.0, 3.0, 3.0]),
        "sin2_ct1_yearly": np.array([3.0, 3.0, 3.0, 3.0, 3.0]),
        "cos2_ct1_yearly": np.array([3.0, 3.0, 3.0, 3.0, 3.0]),
        "sin3_ct1_yearly": np.array([3.0, 3.0, 3.0, 3.0, 3.0]),
        "cos3_ct1_yearly": np.array([3.0, 3.0, 3.0, 3.0, 3.0]),
        # Holiday with pre and post effect (1 at the where the date and event match)
        # e.g. New Years Day is 1 at 1st January, 0 rest of the days
        "Q('events_New Years Day')[T.event]": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        "Q('events_New Years Day_minus_1')[T.event]": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "Q('events_New Years Day_minus_2')[T.event]": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "Q('events_New Years Day_plus_1')[T.event]": np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
        "Q('events_New Years Day_plus_2')[T.event]": np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
    })
    components = silverkite_diagnostics.get_silverkite_components(df, time_col, value_col, feature_df)
    expected_df = pd.DataFrame({
        time_col: df[time_col],
        value_col: df[value_col],
        "trend": 5 * np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "WEEKLY_SEASONALITY": 4 * np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
        "YEARLY_SEASONALITY": 6 * np.array([3.0, 3.0, 3.0, 3.0, 3.0]),
        cst.EVENT_PREFIX: np.array([1.0, 1.0, 1.0, 0.0, 0.0]),
        "trend_changepoints": np.array([0, 1, 0, 1, 0])})
    assert_frame_equal(components, expected_df)

    # Test error messages
    with pytest.raises(ValueError, match="feature_df must be non-empty"):
        silverkite_diagnostics.get_silverkite_components(df, time_col, value_col, feature_df=pd.DataFrame())

    with pytest.raises(ValueError, match="df and feature_df must have same number of rows."):
        silverkite_diagnostics.get_silverkite_components(df, time_col, value_col, feature_df=pd.DataFrame({"ts": [1, 2, 3]}))


def test_group_silverkite_seas_components():
    """Tests group_silverkite_seas_components"""
    silverkite_diagnostics: SilverkiteDiagnostics = SilverkiteDiagnostics()
    time_col = "ts"
    # Daily
    date_list = pd.date_range(start="2018-01-01", end="2018-01-07", freq="H").tolist()
    time_df = build_time_features_df(date_list, conti_year_origin=2018)
    df = pd.DataFrame({
        time_col: time_df["datetime"],
        "DAILY_SEASONALITY": time_df["hour"]
    })
    res = silverkite_diagnostics.group_silverkite_seas_components(df)
    expected_df = pd.DataFrame({
        "Hour of day": np.arange(24.0),
        "daily": np.arange(24.0),
    })
    assert_frame_equal(res, expected_df)

    # Weekly
    date_list = pd.date_range(start="2018-01-01", end="2018-01-20", freq="D").tolist()
    time_df = build_time_features_df(date_list, conti_year_origin=2018)
    df = pd.DataFrame({
        time_col: time_df["datetime"],
        "WEEKLY_SEASONALITY": time_df["tow"]
    })
    res = silverkite_diagnostics.group_silverkite_seas_components(df)
    expected_df = pd.DataFrame({
        "Day of week": np.arange(7.0),
        "weekly": np.arange(7.0),
    })
    assert_frame_equal(res, expected_df)

    # Monthly
    date_list = pd.date_range(start="2018-01-01", end="2018-01-31", freq="D").tolist()
    time_df = build_time_features_df(date_list, conti_year_origin=2018)
    df = pd.DataFrame({
        time_col: time_df["datetime"],
        "MONTHLY_SEASONALITY": time_df["dom"]
    })
    res = silverkite_diagnostics.group_silverkite_seas_components(df)
    expected_df = pd.DataFrame({
        "Time of month": np.arange(31.0)/31,
        "monthly": np.arange(1.0, 32.0),
    })
    assert_frame_equal(res, expected_df)

    # Quarterly (92 day quarters)
    date_list = pd.date_range(start="2018-07-01", end="2018-12-31", freq="D").tolist()
    time_df = build_time_features_df(date_list, conti_year_origin=2018)
    df = pd.DataFrame({
        time_col: time_df["datetime"],
        "QUARTERLY_SEASONALITY": time_df["toq"]
    })
    res = silverkite_diagnostics.group_silverkite_seas_components(df)
    expected_df = pd.DataFrame({
        "Time of quarter": np.arange(92.0)/92,
        "quarterly": np.arange(92.0)/92,
    })
    assert_frame_equal(res, expected_df)

    # Quarterly (90 day quarter)
    date_list = pd.date_range(start="2018-01-01", end="2018-03-31", freq="D").tolist()
    time_df = build_time_features_df(date_list, conti_year_origin=2018)
    df = pd.DataFrame({
        time_col: time_df["datetime"],
        "QUARTERLY_SEASONALITY": time_df["toq"]
    })
    res = silverkite_diagnostics.group_silverkite_seas_components(df)
    expected_df = pd.DataFrame({
        "Time of quarter": np.arange(90.0)/90,
        "quarterly": np.arange(90.0)/90,
    })
    assert_frame_equal(res, expected_df)

    # Yearly (non-leap years)
    date_list = pd.date_range(start="2018-01-01", end="2019-12-31", freq="D").tolist()
    time_df = build_time_features_df(date_list, conti_year_origin=2018)
    df = pd.DataFrame({
        time_col: time_df["datetime"],
        "YEARLY_SEASONALITY": time_df["toy"]
    })
    res = silverkite_diagnostics.group_silverkite_seas_components(df)
    expected_df = pd.DataFrame({
        "Time of year": np.arange(365.0)/365,
        "yearly": np.arange(365.0)/365,
    })
    assert_frame_equal(res, expected_df)


def test_plot_silverkite_components():
    """Tests plot_silverkite_components function"""
    silverkite_diagnostics: SilverkiteDiagnostics = SilverkiteDiagnostics()
    # Dataframe with trend, seasonality and events
    time_col = "ts"
    # value_col name is chosen such that it contains keywords "ct" and "sin"
    # so that we can test patterns specified for each component work correctly
    value_col = "basin_impact"
    df = pd.DataFrame({
        time_col: [
            datetime.datetime(2018, 1, 1),
            datetime.datetime(2018, 1, 2),
            datetime.datetime(2018, 1, 3),
            datetime.datetime(2018, 1, 4),
            datetime.datetime(2018, 1, 5)],
        value_col: [1, 2, 3, 4, 5],
    })
    feature_df = pd.DataFrame({
        # Trend columns: growth, changepoints and interactions (total 5 columns)
        "ct1": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "ct1:tod": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "ct_sqrt": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "changepoint0_2018_01_02_00": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        "changepoint1_2018_01_04_00": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        # Daily seasonality with interaction (total 4 columns)
        "sin1_tow_weekly": np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
        "cos1_tow_weekly": np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
        "is_weekend[T.True]:sin1_tow_weekly": np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
        "is_weekend[T.True]:cos1_tow_weekly": np.array([2.0, 2.0, 2.0, 2.0, 2.0]),
        # Yearly seasonality (total 6 columns)
        "sin1_ct1_yearly": np.array([3.0, 3.0, 3.0, 3.0, 3.0]),
        "cos1_ct1_yearly": np.array([3.0, 3.0, 3.0, 3.0, 3.0]),
        "sin2_ct1_yearly": np.array([3.0, 3.0, 3.0, 3.0, 3.0]),
        "cos2_ct1_yearly": np.array([3.0, 3.0, 3.0, 3.0, 3.0]),
        "sin3_ct1_yearly": np.array([3.0, 3.0, 3.0, 3.0, 3.0]),
        "cos3_ct1_yearly": np.array([3.0, 3.0, 3.0, 3.0, 3.0]),
        # Holiday with pre and post effect (1 at the where the date and event match)
        # e.g. New Years Day is 1 at 1st January, 0 rest of the days
        "Q('events_New Years Day')[T.event]": np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        "Q('events_New Years Day_minus_1')[T.event]": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "Q('events_New Years Day_minus_2')[T.event]": np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        "Q('events_New Years Day_plus_1')[T.event]": np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
        "Q('events_New Years Day_plus_2')[T.event]": np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
    })
    components = silverkite_diagnostics.get_silverkite_components(df, time_col, value_col, feature_df)

    # Check plot_silverkite_components with defaults
    fig = silverkite_diagnostics.plot_silverkite_components(components)
    assert len(fig.data) == 5 + 2  # 2 changepoints
    assert [fig.data[i].name for i in range(len(fig.data))] == list(components.columns)[1: -1] + ["trend change point"] * 2

    assert fig.layout.height == (len(fig.data) - 2) * 350  # changepoints do not create separate subplots
    assert fig.layout.showlegend is True  # legend for changepoints
    assert fig.layout.title["text"] == "Component plots"

    assert fig.layout.xaxis.title["text"] == time_col
    assert fig.layout.xaxis2.title["text"] == time_col
    assert fig.layout.xaxis3.title["text"] == "Day of week"
    assert fig.layout.xaxis4.title["text"] == "Time of year"
    assert fig.layout.xaxis5.title["text"] == time_col

    assert fig.layout.yaxis.title["text"] == value_col
    assert fig.layout.yaxis2.title["text"] == "trend"
    assert fig.layout.yaxis3.title["text"] == "weekly"
    assert fig.layout.yaxis4.title["text"] == "yearly"
    assert fig.layout.yaxis5.title["text"] == "events"

    # Check plot_silverkite_components with provided component list and warnings
    with pytest.warns(Warning) as record:
        names = ["YEARLY_SEASONALITY", value_col, "DUMMY"]
        title = "Component plot without trend and weekly seasonality"
        fig = silverkite_diagnostics.plot_silverkite_components(components, names=names, title=title)

        expected_length = 2
        assert len(fig.data) == expected_length
        assert [fig.data[i].name for i in range(len(fig.data))] == [value_col, "YEARLY_SEASONALITY"]

        assert fig.layout.height == expected_length*350
        assert fig.layout.showlegend is True
        assert fig.layout.title["text"] == title

        assert fig.layout.xaxis.title["text"] == time_col
        assert fig.layout.xaxis2.title["text"] == "Time of year"

        assert fig.layout.yaxis.title["text"] == value_col
        assert fig.layout.yaxis2.title["text"] == "yearly"
        assert f"The following components have not been specified in the model: " \
               f"{{'DUMMY'}}, plotting the rest." in record[0].message.args[0]

    # Check plot_silverkite_components with exception
    with pytest.raises(ValueError, match="None of the provided components have been specified in the model."):
        names = ["DUMMY"]
        silverkite_diagnostics.plot_silverkite_components(components, names=names)
