import datetime
from datetime import datetime as dt

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from greykite.common.constants import ADJUSTMENT_DELTA_COL
from greykite.common.constants import END_TIME_COL
from greykite.common.constants import METRIC_COL
from greykite.common.constants import START_TIME_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.python_utils import assert_equal
from greykite.common.testing_utils import generate_df_for_tests
from greykite.common.testing_utils import generate_df_with_reg_for_tests
from greykite.common.time_properties import get_canonical_data
from greykite.common.viz.timeseries_plotting import add_groupby_column
from greykite.framework.benchmark.data_loader_ts import DataLoaderTS
from greykite.framework.constants import MEAN_COL_GROUP
from greykite.framework.constants import OVERLAY_COL_GROUP
from greykite.framework.constants import QUANTILE_COL_GROUP
from greykite.framework.input.univariate_time_series import UnivariateTimeSeries


def test_check_time_series1():
    """Checks if regular data can be properly loaded. Checks time column stats"""
    ts = UnivariateTimeSeries()
    df = pd.DataFrame({
        "time": [dt(2018, 1, 1, 0, 0, 1),
                 dt(2018, 1, 1, 0, 0, 2),
                 dt(2018, 1, 1, 0, 0, 3)],
        "val": [1, 2, 3]
    })
    ts.load_data(df, "time", "val")
    assert ts.original_time_col == "time"
    assert ts.original_value_col == "val"
    assert ts.time_stats["data_points"] == 3
    assert ts.time_stats["mean_increment_secs"] == 1.0
    assert ts.time_stats["min_timestamp"] == df.min()[0]
    assert ts.time_stats["max_timestamp"] == df.max()[0]
    assert ts.time_stats["added_timepoints"] == 0
    assert ts.time_stats["dropped_timepoints"] == 0
    assert ts.freq == "S"
    assert ts.df[VALUE_COL].equals(ts.y)
    assert ts.df.index.name is None


def test_check_time_series2():
    """Checks value column stats"""
    ts = UnivariateTimeSeries()
    df = pd.DataFrame({
        TIME_COL: [dt(2018, 1, 1, 0, 0, 1),
                   dt(2018, 1, 1, 0, 0, 2),
                   dt(2018, 1, 1, 0, 0, 3)],
        VALUE_COL: [1, 2, 3]
    })
    ts.load_data(df, TIME_COL, VALUE_COL)
    assert ts.value_stats["mean"] == 2.0
    assert ts.value_stats["std"] == 1.0
    assert ts.value_stats["min"] == 1.0
    assert ts.value_stats["25%"] == 1.5
    assert ts.value_stats["50%"] == 2.0
    assert ts.value_stats["75%"] == 2.5
    assert ts.value_stats["max"] == 3.0
    assert ts.df[VALUE_COL].equals(ts.y)


def test_check_time_series_tz_local():
    """Checks date parsing with localization"""
    expected = pd.Series([dt(2018, 1, 1, 0, 0, 0),
                          dt(2018, 1, 5, 0, 0, 0),
                          dt(2018, 1, 9, 0, 0, 0)])
    expected.index = expected
    expected = expected.tz_localize("US/Pacific")
    ts = UnivariateTimeSeries()
    df = pd.DataFrame({
        TIME_COL: ["2018-01-01", "2018-01-05", "2018-01-09"],
        VALUE_COL: [1, 2, 3]
    })
    ts.load_data(df, TIME_COL, VALUE_COL, tz="US/Pacific", freq="4D")
    assert ts.time_stats["added_timepoints"] == 0
    assert ts.time_stats["dropped_timepoints"] == 0
    assert ts.df[TIME_COL].equals(expected)
    assert ts.df[VALUE_COL].equals(ts.y)


def test_check_time_series_gaps():
    """Checks gaps filled for non-regular input"""
    ts = UnivariateTimeSeries()
    df = pd.DataFrame({
        TIME_COL: [dt(2018, 1, 1, 0, 0, 1),
                   dt(2018, 1, 1, 0, 0, 2),
                   dt(2018, 1, 1, 0, 0, 10),  # intentionally out of order
                   dt(2018, 1, 1, 0, 0, 4)],
        VALUE_COL: [1, 2, 3, 4]
    })

    expected = pd.Series([dt(2018, 1, 1, 0, 0, 1),
                          dt(2018, 1, 1, 0, 0, 2),
                          dt(2018, 1, 1, 0, 0, 3),
                          dt(2018, 1, 1, 0, 0, 4),
                          dt(2018, 1, 1, 0, 0, 5),
                          dt(2018, 1, 1, 0, 0, 6),
                          dt(2018, 1, 1, 0, 0, 7),
                          dt(2018, 1, 1, 0, 0, 8),
                          dt(2018, 1, 1, 0, 0, 9),
                          dt(2018, 1, 1, 0, 0, 10)])
    expected.index = expected
    ts.load_data(df, TIME_COL, VALUE_COL, freq="S")  # the frequency should be provided when there are gaps
    assert ts.df[TIME_COL].equals(expected)
    assert ts.time_stats["data_points"] == 10  # after filling in gaps
    assert ts.value_stats["count"] == 4  # before filling in gaps
    assert ts.time_stats["added_timepoints"] == 6
    assert ts.time_stats["dropped_timepoints"] == 0
    assert ts.df[VALUE_COL].equals(ts.y)

    expected_gaps = pd.DataFrame({
        "right_before_gap": pd.Series([dt(2018, 1, 1, 0, 0, 2), dt(2018, 1, 1, 0, 0, 4)]),
        "right_after_gap": pd.Series([dt(2018, 1, 1, 0, 0, 4), dt(2018, 1, 1, 0, 0, 10)]),
        "gap_size": [1.0, 5.0]
    })
    assert ts.time_stats["gaps"].equals(expected_gaps)


def test_check_time_series_err():
    """Checks exceptions"""
    ts = UnivariateTimeSeries()
    with pytest.raises(RuntimeError, match="Must load data"):
        ts.describe_time_col()

    with pytest.raises(RuntimeError, match="Must load data"):
        ts.describe_value_col()

    with pytest.raises(RuntimeError, match="Must load data"):
        ts.make_future_dataframe()


def test_load_data_anomaly():
    """Checks anomaly_info parameter"""
    dl = DataLoaderTS()
    df = dl.load_beijing_pm()
    value_col = "pm"

    # no anomaly adjustment
    ts = UnivariateTimeSeries()
    ts.load_data(df=df, value_col=value_col)
    assert ts.df_before_adjustment is None

    # adjusts two columns
    dim_one = "one"
    dim_two = "two"
    anomaly_df = pd.DataFrame({
        START_TIME_COL: ["2011-04-04-10", "2011-10-10-00", "2012-12-20-10"],
        END_TIME_COL: ["2011-04-05-20", "2011-10-11-23", "2012-12-20-13"],
        ADJUSTMENT_DELTA_COL: [np.nan, 100.0, -100.0],
        METRIC_COL: [dim_one, dim_one, dim_two]
    })
    anomaly_info = [
        {
            "value_col": value_col,
            "anomaly_df": anomaly_df,
            "start_time_col": START_TIME_COL,
            "end_time_col": END_TIME_COL,
            "adjustment_delta_col": ADJUSTMENT_DELTA_COL,
            "filter_by_dict": {METRIC_COL: dim_one},
            "adjustment_method": "add"
        },
        {
            "value_col": "pres",
            "anomaly_df": anomaly_df,
            "start_time_col": START_TIME_COL,
            "end_time_col": END_TIME_COL,
            "adjustment_delta_col": ADJUSTMENT_DELTA_COL,
            "filter_by_dict": {METRIC_COL: dim_two},
            "adjustment_method": "subtract"
        }
    ]
    ts = UnivariateTimeSeries()
    ts.load_data(df=df, value_col=value_col, anomaly_info=anomaly_info)
    canonical_data_dict = get_canonical_data(df=df, value_col=value_col, anomaly_info=anomaly_info)
    assert_equal(ts.df, canonical_data_dict["df"])
    assert_equal(ts.df_before_adjustment, canonical_data_dict["df_before_adjustment"])


def test_make_future_dataframe():
    """Checks future dataframe creation"""
    ts = UnivariateTimeSeries()
    df = pd.DataFrame({
        TIME_COL: [dt(2018, 1, 1, 3, 0, 0),
                   dt(2018, 1, 1, 4, 0, 0),
                   dt(2018, 1, 1, 5, 0, 0),
                   dt(2018, 1, 1, 6, 0, 0),
                   dt(2018, 1, 1, 7, 0, 0)],
        VALUE_COL: [1, None, 3, None, None],
    })
    with pytest.warns(UserWarning) as record:
        ts.load_data(df, TIME_COL, VALUE_COL, regressor_cols=None)
        assert f"{ts.original_value_col} column of the provided TimeSeries contains null " \
               f"values at the end. Setting 'train_end_date' to the last timestamp with a " \
               f"non-null value ({ts.train_end_date})." in record[0].message.args[0]

    # test regressor_cols from load_data
    assert ts.regressor_cols == []

    # tests last_date_for_val from load_data
    assert ts.last_date_for_val == dt(2018, 1, 1, 5, 0, 0)
    assert ts.train_end_date == dt(2018, 1, 1, 5, 0, 0)

    # tests last_date_for_reg from load_data
    assert ts.last_date_for_reg is None

    # tests fit_df from load_data
    result_fit_df = ts.fit_df.reset_index(drop=True)  # fit_df's index is time_col
    expected_fit_df = df[df[TIME_COL] <= dt(2018, 1, 1, 5, 0, 0)]
    assert_frame_equal(result_fit_df, expected_fit_df)

    # tests fit_y
    result_fit_y = ts.fit_y.reset_index(drop=True)  # fit_y's index is time_col
    expected_fit_y = expected_fit_df[VALUE_COL]
    assert result_fit_y.equals(expected_fit_y)

    # with history, default value for periods
    result = ts.make_future_dataframe(
        periods=None,
        include_history=True)
    expected = pd.DataFrame({
        TIME_COL: pd.date_range(
            start=dt(2018, 1, 1, 3, 0, 0),
            periods=33,
            freq="H"),
        VALUE_COL: np.concatenate((result_fit_y, np.repeat(np.nan, 30)))
    })
    expected.index = expected[TIME_COL]
    expected.index.name = None
    assert_frame_equal(result, expected)

    # without history
    result = ts.make_future_dataframe(
        periods=10,
        include_history=False)
    expected = pd.DataFrame({
        TIME_COL: pd.date_range(
            start=dt(2018, 1, 1, 6, 0, 0),
            periods=10,
            freq="H"),
        VALUE_COL: np.repeat(np.nan, 10)
    })
    expected.index = expected[TIME_COL]
    expected.index.name = None
    expected.index.freq = "H"
    assert_frame_equal(result, expected)


def test_make_future_dataframe_with_regressor():
    """Checks future dataframe creation"""
    ts = UnivariateTimeSeries()
    df = pd.DataFrame({
        TIME_COL: [dt(2018, 1, 1, 3, 0, 0),
                   dt(2018, 1, 1, 4, 0, 0),
                   dt(2018, 1, 1, 5, 0, 0),
                   dt(2018, 1, 1, 6, 0, 0),
                   dt(2018, 1, 1, 7, 0, 0),
                   dt(2018, 1, 1, 8, 0, 0)],
        VALUE_COL: [1, None, 3, None, None, None],
        "regressor1": [0.01, None, 0.014, 0.016, 0.017, None],
        "regressor2": [0.11, 0.112, 0.114, 0.116, None, None],
        "regressor3": [0.21, 0.212, 0.214, 0.216, 0.217, None]
    })
    regressor_cols = [col for col in df.columns if col not in [TIME_COL, VALUE_COL]]

    with pytest.warns(Warning) as record:
        ts.load_data(df, TIME_COL, VALUE_COL, regressor_cols=regressor_cols)
        assert "y column of the provided TimeSeries contains null " \
               "values at the end" in record[0].message.args[0]

    # test regressor_cols from load_data
    assert ts.regressor_cols == ["regressor1", "regressor2", "regressor3"]

    # tests last_date_for_fit from load_data (same as without regressor)
    assert ts.train_end_date == dt(2018, 1, 1, 5, 0, 0)

    # tests last_date_for_reg from load_data
    assert ts.last_date_for_reg == dt(2018, 1, 1, 7, 0, 0)

    # tests fit_df from load_data
    result_fit_df = ts.fit_df.reset_index(drop=True)  # fit_df's index is x_col
    expected_fit_df = df[df[TIME_COL] <= dt(2018, 1, 1, 5, 0, 0)]
    assert_frame_equal(result_fit_df, expected_fit_df)

    # tests fit_y
    result_fit_y = ts.fit_y.reset_index(drop=True)  # fit_y's index is x_col
    expected_fit_y = expected_fit_df[VALUE_COL]
    assert result_fit_y.equals(expected_fit_y)

    # with history
    result = ts.make_future_dataframe(
        periods=2,
        include_history=True).reset_index(drop=True)
    expected = pd.DataFrame({
        TIME_COL: pd.date_range(start=dt(2018, 1, 1, 3, 0, 0), periods=5, freq="H"),
        VALUE_COL: [1, None, 3, None, None],
        "regressor1": [0.01, None, 0.014, 0.016, 0.017],
        "regressor2": [0.11, 0.112, 0.114, 0.116, None],
        "regressor3": [0.21, 0.212, 0.214, 0.216, 0.217]
    })
    assert_frame_equal(result, expected)

    # without history
    result = ts.make_future_dataframe(
        periods=2,
        include_history=False).reset_index(drop=True)
    expected = pd.DataFrame({
        TIME_COL: pd.date_range(start=dt(2018, 1, 1, 6, 0, 0), periods=2, freq="H"),
        VALUE_COL: np.repeat(np.nan, 2),
        "regressor1": [0.016, 0.017],
        "regressor2": [0.116, None],
        "regressor3": [0.216, 0.217]
    })
    assert result.equals(expected)

    # user doesn't request any future periods
    result = ts.make_future_dataframe(periods=None, include_history=False).reset_index(drop=True)
    expected = pd.DataFrame({
        TIME_COL: pd.date_range(start=dt(2018, 1, 1, 6, 0, 0), periods=2, freq="H"),
        VALUE_COL: np.repeat(np.nan, 2),
        "regressor1": [0.016, 0.017],
        "regressor2": [0.116, None],
        "regressor3": [0.216, 0.217]
    })
    assert result.equals(expected)

    # user requests fewer than the available periods
    result = ts.make_future_dataframe(
        periods=1,
        include_history=False).reset_index(drop=True)
    expected = pd.DataFrame({
        TIME_COL: pd.date_range(start=dt(2018, 1, 1, 6, 0, 0), periods=1, freq="H"),
        VALUE_COL: np.repeat(np.nan, 1),
        "regressor1": [0.016],
        "regressor2": [0.116],
        "regressor3": [0.216]
    })
    assert result.equals(expected)

    # user requests more than 2 periods
    with pytest.warns(Warning) as record:
        result = ts.make_future_dataframe(periods=4, include_history=False).reset_index(drop=True)
        expected = pd.DataFrame({
            TIME_COL: pd.date_range(start=dt(2018, 1, 1, 6, 0, 0), periods=2, freq="H"),
            VALUE_COL: np.repeat(np.nan, 2),
            "regressor1": [0.016, 0.017],
            "regressor2": [0.116, None],
            "regressor3": [0.216, 0.217]
        })
        assert result.equals(expected)
        assert "Provided periods '4' is more than allowed ('2') due to the length of regressor columns. " \
               "Using '2'." in record[0].message.args[0]


def test_train_end_date_without_regressors():
    """Tests make_future_dataframe and train_end_date without regressors"""
    ts = UnivariateTimeSeries()
    df = pd.DataFrame({
        TIME_COL: [dt(2018, 1, 1, 3, 0, 0),
                   dt(2018, 1, 1, 4, 0, 0),
                   dt(2018, 1, 1, 5, 0, 0),
                   dt(2018, 1, 1, 6, 0, 0),
                   dt(2018, 1, 1, 7, 0, 0)],
        VALUE_COL: [1, None, 3, None, None],
    })

    # train_end_date later than last date in df
    with pytest.warns(UserWarning) as record:
        train_end_date = dt(2018, 1, 1, 8, 0, 0)
        ts.load_data(df, TIME_COL, VALUE_COL, train_end_date=train_end_date)
        assert f"Input timestamp for the parameter 'train_end_date' " \
               f"({train_end_date}) either exceeds the last available timestamp or" \
               f"{VALUE_COL} column of the provided TimeSeries contains null " \
               f"values at the end. Setting 'train_end_date' to the last timestamp with a " \
               f"non-null value ({ts.train_end_date})." in record[0].message.args[0]
        assert ts.train_end_date == dt(2018, 1, 1, 5, 0, 0)
        result = ts.make_future_dataframe(
            periods=10,
            include_history=True)
        expected = pd.DataFrame({
            TIME_COL: pd.date_range(
                start=dt(2018, 1, 1, 3, 0, 0),
                periods=13,
                freq="H"),
            VALUE_COL: np.concatenate((ts.fit_y, np.repeat(np.nan, 10)))
        })
        expected.index = expected[TIME_COL]
        expected.index.name = None
        assert_frame_equal(result, expected)


def test_train_end_date_with_regressors():
    """Tests make_future_dataframe and train_end_date with regressors"""
    data = generate_df_with_reg_for_tests(
        freq="D",
        periods=30,
        train_start_date=datetime.datetime(2018, 1, 1),
        remove_extra_cols=True,
        mask_test_actuals=True)
    regressor_cols = ["regressor1", "regressor2", "regressor_categ"]
    keep_cols = [TIME_COL, VALUE_COL] + regressor_cols
    df = data["df"][keep_cols].copy()
    # Setting NaN values at the end
    df.loc[df.tail(2).index, "regressor1"] = np.nan
    df.loc[df.tail(4).index, "regressor2"] = np.nan
    df.loc[df.tail(6).index, "regressor_categ"] = np.nan
    df.loc[df.tail(8).index, VALUE_COL] = np.nan

    # default train_end_date, default regressor_cols
    with pytest.warns(UserWarning) as record:
        ts = UnivariateTimeSeries()
        ts.load_data(df, TIME_COL, VALUE_COL, train_end_date=None, regressor_cols=None)
        assert f"{ts.original_value_col} column of the provided TimeSeries contains null " \
               f"values at the end. Setting 'train_end_date' to the last timestamp with a " \
               f"non-null value ({ts.train_end_date})." in record[0].message.args[0]
        assert ts.train_end_date == dt(2018, 1, 22)
        assert ts.fit_df.shape == (22, 2)
        assert ts.last_date_for_val == df[df[VALUE_COL].notnull()][TIME_COL].max()
        assert ts.last_date_for_reg is None
        result = ts.make_future_dataframe(
            periods=10,
            include_history=True)
        expected = pd.DataFrame({
            TIME_COL: pd.date_range(
                start=dt(2018, 1, 1),
                periods=32,
                freq="D"),
            VALUE_COL: np.concatenate([ts.fit_y, np.repeat(np.nan, 10)])
        })
        expected.index = expected[TIME_COL]
        expected.index.name = None
        assert_frame_equal(result, expected)

    # train_end_date later than last date in df, all available regressor_cols
    with pytest.warns(UserWarning) as record:
        ts = UnivariateTimeSeries()
        train_end_date = dt(2018, 2, 10)
        ts.load_data(df, TIME_COL, VALUE_COL, train_end_date=train_end_date, regressor_cols=regressor_cols)
        assert f"Input timestamp for the parameter 'train_end_date' " \
               f"({train_end_date}) either exceeds the last available timestamp or" \
               f"{VALUE_COL} column of the provided TimeSeries contains null " \
               f"values at the end. Setting 'train_end_date' to the last timestamp with a " \
               f"non-null value ({ts.train_end_date})." in record[0].message.args[0]
        assert ts.last_date_for_val == dt(2018, 1, 22)
        assert ts.last_date_for_reg == dt(2018, 1, 28)
        result = ts.make_future_dataframe(
            periods=10,
            include_history=False)
        expected = df.copy()[22:28]
        expected.loc[expected.tail(6).index, VALUE_COL] = np.nan
        expected.index = expected[TIME_COL]
        expected.index.name = None
        assert_frame_equal(result, expected)

    # train_end_date in between last date in df and last date before null
    # user passes no regressor_cols
    with pytest.warns(UserWarning) as record:
        ts = UnivariateTimeSeries()
        train_end_date = dt(2018, 1, 25)
        regressor_cols = []
        ts.load_data(df, TIME_COL, VALUE_COL, train_end_date=train_end_date, regressor_cols=regressor_cols)
        assert f"Input timestamp for the parameter 'train_end_date' " \
               f"({train_end_date}) either exceeds the last available timestamp or" \
               f"{VALUE_COL} column of the provided TimeSeries contains null " \
               f"values at the end. Setting 'train_end_date' to the last timestamp with a " \
               f"non-null value ({ts.train_end_date})." in record[0].message.args[0]
        assert ts.train_end_date == dt(2018, 1, 22)
        assert ts.last_date_for_reg is None
        result = ts.make_future_dataframe(
            periods=10,
            include_history=True)
        expected = pd.DataFrame({
            TIME_COL: pd.date_range(
                start=dt(2018, 1, 1),
                periods=32,
                freq="D"),
            VALUE_COL: np.concatenate([ts.fit_y, np.repeat(np.nan, 10)])
        })
        expected.index = expected[TIME_COL]
        expected.index.name = None
        assert_frame_equal(result, expected)

    # train end date equal to last date before null
    # user requests a subset of the regressor_cols
    with pytest.warns(UserWarning) as record:
        ts = UnivariateTimeSeries()
        train_end_date = dt(2018, 1, 22)
        regressor_cols = ["regressor2"]
        ts.load_data(df, TIME_COL, VALUE_COL, train_end_date=train_end_date, regressor_cols=regressor_cols)
        assert ts.train_end_date == dt(2018, 1, 22)
        assert ts.last_date_for_reg == dt(2018, 1, 26)
        result = ts.make_future_dataframe(
            periods=10,
            include_history=True)
        # gathers all warning messages
        all_warnings = ""
        for i in range(len(record)):
            all_warnings += record[i].message.args[0]
        assert "Provided periods '10' is more than allowed ('4') due to the length of " \
               "regressor columns. Using '4'." in all_warnings
        expected = ts.df.copy()[[TIME_COL, VALUE_COL, "regressor2"]]
        expected = expected[expected.index <= ts.last_date_for_reg]
        assert_frame_equal(result, expected)

    # train_end_date smaller than last date before null
    # user requests regressor_cols that does not exist in df
    with pytest.warns(UserWarning) as record:
        ts = UnivariateTimeSeries()
        train_end_date = dt(2018, 1, 20)
        regressor_cols = ["regressor1", "regressor4", "regressor5"]
        ts.load_data(df, TIME_COL, VALUE_COL, train_end_date=train_end_date, regressor_cols=regressor_cols)
        assert ts.train_end_date == dt(2018, 1, 20)
        assert ts.last_date_for_reg == dt(2018, 1, 28)
        # gathers all warning messages
        all_warnings = ""
        for i in range(len(record)):
            all_warnings += record[i].message.args[0]
        assert (f"The following columns are not available to use as "
                f"regressors: ['regressor4', 'regressor5']") in all_warnings
        result = ts.make_future_dataframe(
            periods=10,
            include_history=True)
        expected = ts.df.copy()[[TIME_COL, VALUE_COL, "regressor1"]]
        expected = expected[expected.index <= ts.last_date_for_reg]
        assert_frame_equal(result, expected)


def test_plot():
    """Checks plot function"""
    # Plots with `color`
    ts = UnivariateTimeSeries()
    df = pd.DataFrame({
        TIME_COL: [dt(2018, 1, 1, 3, 0, 0),
                   dt(2018, 1, 1, 4, 0, 0),
                   dt(2018, 1, 1, 5, 0, 0)],
        VALUE_COL: [1, 2, 3]
    })
    ts.load_data(df, TIME_COL, VALUE_COL)
    fig = ts.plot(color="green")
    assert len(fig.data) == 1
    assert fig.data[0].line.color == "green"
    with pytest.raises(ValueError, match="There is no `anomaly_info` to show. `show_anomaly_adjustment` must be False."):
        ts.plot(show_anomaly_adjustment=True)

    # Plots with `show_anomaly_adjustment`
    dl = DataLoaderTS()
    df = dl.load_beijing_pm()
    value_col = "pm"
    # Masks up to 2011-02-04-03, and adds 100.0 to the rest
    anomaly_df = pd.DataFrame({
        START_TIME_COL: ["2010-01-01-00", "2011-02-04-03"],
        END_TIME_COL: ["2011-02-04-03", "2014-12-31-23"],
        ADJUSTMENT_DELTA_COL: [np.nan, 100.0],
        METRIC_COL: [value_col, value_col]
    })
    anomaly_info = {
        "value_col": value_col,
        "anomaly_df": anomaly_df,
        "start_time_col": START_TIME_COL,
        "end_time_col": END_TIME_COL,
        "adjustment_delta_col": ADJUSTMENT_DELTA_COL,
        "filter_by_dict": {METRIC_COL: value_col},
        "adjustment_method": "add"
    }
    ts = UnivariateTimeSeries()
    ts.load_data(df=df, value_col="pm", anomaly_info=anomaly_info)
    fig = ts.plot(show_anomaly_adjustment=True)
    assert len(fig.data) == 2
    assert fig.data[0].name == value_col
    assert fig.data[1].name == f"{value_col}_unadjusted"
    assert fig.layout.xaxis.title.text == ts.original_time_col
    assert fig.layout.yaxis.title.text == ts.original_value_col
    assert fig.data[0].y.shape[0] == df.shape[0]
    assert fig.data[1].y.shape[0] == df.shape[0]
    # adjusted data has more NaNs, since anomalies are replaced with NaN
    assert sum(np.isnan(fig.data[0].y)) == 10906
    assert sum(np.isnan(fig.data[1].y)) == 2067


def test_get_grouping_evaluation():
    """Tests get_grouping_evaluation function"""
    df = pd.DataFrame({
        "custom_time_column": [
            datetime.datetime(2018, 1, 1),
            datetime.datetime(2018, 1, 2),
            datetime.datetime(2018, 1, 3),
            datetime.datetime(2018, 1, 4),
            datetime.datetime(2018, 1, 5)],
        "custom_value_column": [1.0, 2.0, 3.0, 4.0, 5.0],
    })

    ts = UnivariateTimeSeries()
    ts.load_data(df, time_col="custom_time_column", value_col="custom_value_column")

    # mean, groupby_time_feature
    grouped_df = ts.get_grouping_evaluation(
        aggregation_func=np.mean,
        aggregation_func_name="mean",
        groupby_time_feature="dow")
    expected = pd.DataFrame({
        "dow": [1, 2, 3, 4, 5],  # Monday, Tuesday, etc. Time feature is used as column name
        f"mean of {VALUE_COL}": [1.0, 2.0, 3.0, 4.0, 5.0]
    })
    assert_equal(grouped_df, expected)

    # max, groupby_sliding_window_size
    grouped_df = ts.get_grouping_evaluation(
        aggregation_func=np.max,
        aggregation_func_name="max",
        groupby_sliding_window_size=2)
    expected = pd.DataFrame({
        f"{TIME_COL}_downsample": [
            datetime.datetime(2018, 1, 1),
            datetime.datetime(2018, 1, 3),
            datetime.datetime(2018, 1, 5)],
        f"max of {VALUE_COL}": [1.0, 3.0, 5.0]
    })
    assert_equal(grouped_df, expected)

    # min, groupby_custom_column
    grouped_df = ts.get_grouping_evaluation(
        aggregation_func=np.min,
        aggregation_func_name=None,
        groupby_custom_column=pd.Series(["g1", "g2", "g1", "g3", "g2"], name="custom_groups"))
    expected = pd.DataFrame({
        "custom_groups": ["g1", "g2", "g3"],
        f"aggregation of {VALUE_COL}": [1.0, 2.0, 4.0]
    })
    assert_equal(grouped_df, expected)


def test_plot_grouping_evaluation():
    """Tests plot_grouping_evaluation function"""
    df = generate_df_for_tests(freq="D", periods=20)["df"]
    df.rename(
        columns={TIME_COL: "custom_time_column", VALUE_COL: "custom_value_column"},
        inplace=True)

    ts = UnivariateTimeSeries()
    ts.load_data(df, time_col="custom_time_column", value_col="custom_value_column")

    # groupby_time_feature
    fig = ts.plot_grouping_evaluation(
        aggregation_func=np.mean,
        aggregation_func_name="mean",
        groupby_time_feature="dow")

    assert fig.data[0].name == f"mean of {VALUE_COL}"
    assert fig.layout.xaxis.title.text == "dow"
    assert fig.layout.yaxis.title.text == f"mean of {VALUE_COL}"
    assert fig.layout.title.text == f"mean of {VALUE_COL} vs dow"
    assert fig.data[0].x.shape[0] == 7

    # groupby_sliding_window_size
    fig = ts.plot_grouping_evaluation(
        aggregation_func=np.max,
        aggregation_func_name="max",
        groupby_sliding_window_size=7)  # there are 20 training points, so this creates groups of size (6, 7, 7)
    assert fig.data[0].name == f"max of {VALUE_COL}"
    assert fig.layout.xaxis.title.text == f"{TIME_COL}_downsample"
    assert fig.layout.yaxis.title.text == f"max of {VALUE_COL}"
    assert fig.layout.title.text == f"max of {VALUE_COL} vs {TIME_COL}_downsample"
    assert fig.data[0].x.shape[0] == 3

    # groupby_custom_column
    custom_groups = pd.Series(["g1", "g2", "g3", "g4", "g5"], name="custom_groups").repeat(4)
    fig = ts.plot_grouping_evaluation(
        aggregation_func=np.min,
        aggregation_func_name="min",
        groupby_custom_column=custom_groups)
    assert fig.data[0].name == f"min of {VALUE_COL}"
    assert fig.layout.xaxis.title.text == "custom_groups"
    assert fig.layout.yaxis.title.text == f"min of {VALUE_COL}"
    assert fig.layout.title.text == f"min of {VALUE_COL} vs custom_groups"
    assert fig.data[0].x.shape[0] == 5

    # custom xlabel, ylabel and title
    fig = ts.plot_grouping_evaluation(
        aggregation_func=np.mean,
        aggregation_func_name="mean",
        groupby_time_feature="dow",
        xlabel="Day of Week",
        ylabel="Average of y",
        title="Average of y by Day of week")
    assert fig.layout.xaxis.title.text == "Day of Week"
    assert fig.layout.yaxis.title.text == "Average of y"
    assert fig.layout.title.text == "Average of y by Day of week"


def test_get_quantiles_and_overlays():
    """Tests get_quantiles_and_overlays"""
    dl = DataLoaderTS()
    peyton_manning_ts = dl.load_peyton_manning_ts()

    # no columns are requested
    with pytest.raises(ValueError, match="Must enable at least one of: show_mean, show_quantiles, show_overlays."):
        peyton_manning_ts.get_quantiles_and_overlays(
            groupby_time_feature="doy")

    # show_mean only
    grouped_df = peyton_manning_ts.get_quantiles_and_overlays(
        groupby_time_feature="dow",
        show_mean=True,
        mean_col_name="custom_name")
    assert_equal(grouped_df.columns, pd.MultiIndex.from_arrays(
        [[MEAN_COL_GROUP], ["custom_name"]], names=["category", "name"]))
    assert grouped_df.index.name == "dow"
    assert grouped_df.shape == (7, 1)
    assert grouped_df.index[0] == 1

    # show_quantiles only (bool)
    grouped_df = peyton_manning_ts.get_quantiles_and_overlays(
        groupby_sliding_window_size=180,
        show_quantiles=True)
    assert_equal(grouped_df.columns, pd.MultiIndex.from_arrays(
        [[QUANTILE_COL_GROUP, QUANTILE_COL_GROUP],
         ["Q0.1", "Q0.9"]],
        names=["category", "name"]))
    assert grouped_df.index.name == "ts_downsample"
    assert grouped_df.shape == (17, 2)
    assert grouped_df.index[0] == pd.Timestamp(2007, 12, 10)

    # show_quantiles only (list)
    custom_col = pd.Series(np.random.choice(list("abcd"), size=peyton_manning_ts.df.shape[0]))
    grouped_df = peyton_manning_ts.get_quantiles_and_overlays(
        groupby_custom_column=custom_col,
        show_quantiles=[0, 0.25, 0.5, 0.75, 1],
        quantile_col_prefix="prefix")
    assert_equal(grouped_df.columns, pd.MultiIndex.from_arrays(
        [[QUANTILE_COL_GROUP]*5,
         ["prefix0", "prefix0.25", "prefix0.5", "prefix0.75", "prefix1"]],
        names=["category", "name"]))
    assert grouped_df.index.name == "groups"
    assert grouped_df.shape == (4, 5)
    assert grouped_df.index[0] == "a"
    # checks quantile computation
    df = peyton_manning_ts.df.copy()
    df["custom_col"] = custom_col.values
    quantile_df = df.groupby("custom_col")[VALUE_COL].agg([np.nanmin, np.nanmedian, np.nanmax])
    assert_equal(grouped_df["quantile"]["prefix0"], quantile_df["nanmin"], check_names=False)
    assert_equal(grouped_df["quantile"]["prefix0.5"], quantile_df["nanmedian"], check_names=False)
    assert_equal(grouped_df["quantile"]["prefix1"], quantile_df["nanmax"], check_names=False)

    # show_overlays only (bool), no overlay label
    grouped_df = peyton_manning_ts.get_quantiles_and_overlays(
        groupby_time_feature="doy",
        show_overlays=True)
    assert_equal(grouped_df.columns, pd.MultiIndex.from_arrays(
        [[OVERLAY_COL_GROUP]*9,
         [f"overlay{i}" for i in range(9)]],
        names=["category", "name"]))
    assert grouped_df.index.name == "doy"
    assert grouped_df.shape == (366, 9)
    assert grouped_df.index[0] == 1

    # show_overlays only (int below the available number), time feature overlay label
    np.random.seed(123)
    grouped_df = peyton_manning_ts.get_quantiles_and_overlays(
        groupby_time_feature="doy",
        show_overlays=4,
        overlay_label_time_feature="year")
    assert_equal(grouped_df.columns, pd.MultiIndex.from_arrays(
        [[OVERLAY_COL_GROUP]*4,
         ["2007", "2011", "2012", "2014"]],
        names=["category", "name"]))
    assert grouped_df.index.name == "doy"
    assert grouped_df.shape == (366, 4)
    assert grouped_df.index[0] == 1

    # show_overlays only (int above the available number), custom overlay label
    grouped_df = peyton_manning_ts.get_quantiles_and_overlays(
        groupby_time_feature="dom",
        show_overlays=200,
        overlay_label_custom_column=custom_col)
    assert_equal(grouped_df.columns, pd.MultiIndex.from_arrays(
        [[OVERLAY_COL_GROUP]*4,
         ["a", "b", "c", "d"]],
        names=["category", "name"]))
    assert grouped_df.index.name == "dom"
    assert grouped_df.shape == (31, 4)
    assert grouped_df.index[0] == 1

    # show_overlays only (list of indices), sliding window overlay label
    grouped_df = peyton_manning_ts.get_quantiles_and_overlays(
        groupby_time_feature="dom",
        show_overlays=[0, 4],
        overlay_label_sliding_window_size=365*2)
    assert_equal(grouped_df.columns, pd.MultiIndex.from_arrays(
        [[OVERLAY_COL_GROUP]*2,
         ["2007-12-10 00:00:00", "2015-12-08 00:00:00"]],
        names=["category", "name"]))
    assert grouped_df.index.name == "dom"
    assert grouped_df.shape == (31, 2)
    assert grouped_df.index[0] == 1

    # show_overlays only (np.ndarray), sliding window overlay label
    grouped_df = peyton_manning_ts.get_quantiles_and_overlays(
        groupby_time_feature="dom",
        show_overlays=np.arange(0, 6, 2),
        overlay_label_sliding_window_size=365*2)
    assert_equal(grouped_df.columns, pd.MultiIndex.from_arrays(
        [[OVERLAY_COL_GROUP]*3,
         ["2007-12-10 00:00:00", "2011-12-09 00:00:00", "2015-12-08 00:00:00"]],
        names=["category", "name"]))
    assert grouped_df.index.name == "dom"
    assert grouped_df.shape == (31, 3)
    assert grouped_df.index[0] == 1

    # show_overlays only (list of column names), sliding window overlay label
    grouped_df = peyton_manning_ts.get_quantiles_and_overlays(
        groupby_time_feature="dom",
        show_overlays=["2007-12-10 00:00:00", "2015-12-08 00:00:00"],
        overlay_label_sliding_window_size=365*2)
    assert_equal(grouped_df.columns, pd.MultiIndex.from_arrays(
        [[OVERLAY_COL_GROUP]*2,
         ["2007-12-10 00:00:00", "2015-12-08 00:00:00"]],
        names=["category", "name"]))
    assert grouped_df.index.name == "dom"
    assert grouped_df.shape == (31, 2)
    assert grouped_df.index[0] == 1

    # Show all 3 (no overlay label)
    grouped_df = peyton_manning_ts.get_quantiles_and_overlays(
        groupby_sliding_window_size=50,    # 50 per group (50 overlays)
        show_mean=True,
        show_quantiles=[0.05, 0.5, 0.95],  # 3 quantiles
        show_overlays=True)
    assert_equal(grouped_df.columns, pd.MultiIndex.from_arrays(
        [[MEAN_COL_GROUP] + [QUANTILE_COL_GROUP]*3 + [OVERLAY_COL_GROUP]*50,
         ["mean", "Q0.05", "Q0.5", "Q0.95"] + [f"overlay{i}" for i in range(50)]],
        names=["category", "name"]))
    assert grouped_df.index.name == "ts_downsample"
    assert grouped_df.shape == (60, 54)
    assert grouped_df.index[-1] == pd.Timestamp(2016, 1, 7)

    # Show all 3 (with overlay label).
    # Pass overlay_pivot_table_kwargs.
    grouped_df = peyton_manning_ts.get_quantiles_and_overlays(
        groupby_sliding_window_size=180,
        show_mean=True,
        show_quantiles=[0.05, 0.5, 0.95],  # 3 quantiles
        show_overlays=True,
        overlay_label_time_feature="dow",  # 7 possible values
        aggfunc="median")
    assert_equal(grouped_df.columns, pd.MultiIndex.from_arrays(
        [[MEAN_COL_GROUP] + [QUANTILE_COL_GROUP]*3 + [OVERLAY_COL_GROUP]*7,
         ["mean", "Q0.05", "Q0.5", "Q0.95", "1", "2", "3", "4", "5", "6", "7"]],
        names=["category", "name"]))
    assert grouped_df.index.name == "ts_downsample"
    assert grouped_df.shape == (17, 11)
    assert grouped_df.index[-1] == pd.Timestamp(2015, 10, 29)
    assert np.linalg.norm(grouped_df[OVERLAY_COL_GROUP].mean()) > 1.0  # not centered

    with pytest.raises(TypeError, match="pivot_table\\(\\) got an unexpected keyword argument 'aggfc'"):
        peyton_manning_ts.get_quantiles_and_overlays(
            groupby_sliding_window_size=180,
            show_mean=True,
            show_quantiles=[0.05, 0.5, 0.95],
            show_overlays=True,
            overlay_label_time_feature="dow",
            aggfc=np.nanmedian)  # unrecognized parameter

    # center_values with show_mean=True
    centered_df = peyton_manning_ts.get_quantiles_and_overlays(
        groupby_sliding_window_size=180,
        show_mean=True,
        show_quantiles=[0.05, 0.5, 0.95],
        show_overlays=True,
        overlay_label_time_feature="dow",
        aggfunc="median",
        center_values=True)
    assert np.linalg.norm(centered_df[[MEAN_COL_GROUP, OVERLAY_COL_GROUP]].mean()) < 1e-8  # centered at 0
    assert_equal(centered_df[QUANTILE_COL_GROUP], grouped_df[QUANTILE_COL_GROUP] - grouped_df[MEAN_COL_GROUP].mean()[0])

    # center_values with show_mean=False
    centered_df = peyton_manning_ts.get_quantiles_and_overlays(
        groupby_sliding_window_size=180,
        show_mean=False,
        show_quantiles=[0.05, 0.5, 0.95],
        show_overlays=True,
        overlay_label_time_feature="dow",
        aggfunc="median",
        center_values=True)
    assert np.linalg.norm(centered_df[[OVERLAY_COL_GROUP]].mean()) < 1e-8  # centered at 0
    overall_mean = peyton_manning_ts.df[VALUE_COL].mean()
    assert_equal(centered_df[QUANTILE_COL_GROUP], grouped_df[QUANTILE_COL_GROUP] - overall_mean)

    # new value_col
    df = generate_df_with_reg_for_tests(freq="D", periods=700)["df"]
    ts = UnivariateTimeSeries()
    ts.load_data(df=df)
    grouped_df = ts.get_quantiles_and_overlays(
        groupby_time_feature="dow",
        show_mean=True,
        show_quantiles=True,
        show_overlays=True,
        overlay_label_time_feature="woy",
        value_col="regressor1")

    df_dow = add_groupby_column(
        df=ts.df,
        time_col=TIME_COL,
        groupby_time_feature="dow")
    dow_mean = df_dow["df"].groupby("dow").agg(mean=pd.NamedAgg(column="regressor1", aggfunc=np.nanmean))
    assert_equal(grouped_df["mean"], dow_mean, check_names=False)


def test_plot_quantiles_and_overlays():
    """Tests plot_quantiles_and_overlays"""
    dl = DataLoaderTS()
    peyton_manning_ts = dl.load_peyton_manning_ts()
    # plots one at a time, with different axis options
    fig = peyton_manning_ts.plot_quantiles_and_overlays(
        groupby_time_feature="doy",
        show_mean=True)
    assert fig.layout.showlegend
    assert fig.layout.title.text == "y vs doy"
    assert fig.layout.xaxis.title.text == "doy"
    assert fig.layout.yaxis.title.text == "y"
    assert len(fig.data) == 1
    assert fig.data[0].mode == "lines"
    assert fig.data[0].legendgroup == MEAN_COL_GROUP
    assert fig.data[0].line.color == "#595959"
    assert fig.data[0].line.width == 2
    assert fig.data[0].name == "mean"

    fig = peyton_manning_ts.plot_quantiles_and_overlays(
        groupby_time_feature="doy",
        show_quantiles=True,
        ylabel="log(pageviews)",
        xlabel="day of year")
    assert fig.layout.showlegend
    assert fig.layout.title.text == "log(pageviews) vs day of year"
    assert fig.layout.xaxis.title.text == "day of year"
    assert fig.layout.yaxis.title.text == "log(pageviews)"
    assert len(fig.data) == 2
    assert fig.data[0].mode == "lines"
    assert fig.data[0].legendgroup == QUANTILE_COL_GROUP
    assert fig.data[0].line.color == "#1F9AFF"
    assert fig.data[0].line.width == 2
    assert fig.data[0].fill is None  # no fill from first line
    assert fig.data[1].fill == "tonexty"
    assert fig.data[0].name == "Q0.1"
    assert fig.data[1].name == "Q0.9"

    fig = peyton_manning_ts.plot_quantiles_and_overlays(
        groupby_time_feature="doy",
        show_overlays=True,
        overlay_label_time_feature="year",
        ylabel="log(pageviews)",
        xlabel="day of year",
        title="Yearly seasonality patterns",
        showlegend=False)
    assert not fig.layout.showlegend
    assert fig.layout.title.text == "Yearly seasonality patterns"
    assert fig.layout.xaxis.title.text == "day of year"
    assert fig.layout.yaxis.title.text == "log(pageviews)"
    assert len(fig.data) == 10
    assert fig.data[0].mode == "lines"
    assert fig.data[0].legendgroup == OVERLAY_COL_GROUP
    assert fig.data[0].line.color == "#B3B3B3"
    assert fig.data[0].line.width == 1
    assert fig.data[0].line.dash == "solid"
    assert fig.data[0].name == "2007"
    assert fig.data[-1].name == "2016"

    # plots all at once
    fig = peyton_manning_ts.plot_quantiles_and_overlays(
        groupby_time_feature="doy",
        show_mean=True,
        show_quantiles=True,
        show_overlays=True,
        overlay_label_time_feature="year",
        ylabel="log(pageviews)",
        xlabel="day of year",
        title="Yearly seasonality patterns",
        showlegend=True)
    assert fig.layout.showlegend
    assert fig.layout.title.text == "Yearly seasonality patterns"
    assert fig.layout.xaxis.title.text == "day of year"
    assert fig.layout.yaxis.title.text == "log(pageviews)"
    assert len(fig.data) == 13  # 1 (mean) + 2 (quantiles) + 10 (one per year)
    assert fig.data[0].mode == "lines"
    assert fig.data[0].legendgroup == OVERLAY_COL_GROUP
    assert fig.data[0].line.color == "#B3B3B3"
    assert fig.data[0].line.width == 1
    assert fig.data[0].line.dash == "solid"
    assert fig.data[0].opacity == 0.5
    assert fig.data[0].name == "2007"
    assert fig.data[9].name == "2016"
    assert fig.data[10].name == "Q0.1"
    assert fig.data[11].name == "Q0.9"
    assert fig.data[12].name == "mean"

    # plots all with custom style
    beijing_pm_ts = dl.load_beijing_pm_ts()
    mean_style = {
        "line": dict(
            width=2,
            color="#757575"),  # gray
        "legendgroup": MEAN_COL_GROUP}
    quantile_style = {
        "line": dict(
            width=2,
            color="#A3A3A3"),  # light gray
        "legendgroup": QUANTILE_COL_GROUP,
        "fill": "tonexty"}
    overlay_style = {  # Different color for each line (unspecified)
        "line": dict(  # No legendgroup to allow individual toggling of lines.
            width=1,
            dash="dot")}
    fig = beijing_pm_ts.plot_quantiles_and_overlays(
        groupby_time_feature="hour",
        show_mean=True,
        show_quantiles=[0.2, 0.8],
        show_overlays=True,
        overlay_label_time_feature="month",
        center_values=True,
        mean_col_name="avg",
        mean_style=mean_style,
        quantile_style=quantile_style,
        overlay_style=overlay_style,
        xlabel="hour of day",
        value_col="pres",
        title="Daily seasonality pattern: pres")
    assert fig.layout.showlegend
    assert fig.layout.title.text == "Daily seasonality pattern: pres"
    assert fig.layout.xaxis.title.text == "hour of day"
    assert fig.layout.yaxis.title.text == "pres"
    assert len(fig.data) == 15
    # The first timeseries is the 12 overlays (one per month), then Q0.1, Q0.9, then the mean
    assert fig.data[0].name == "1"
    assert fig.data[11].name == "12"
    assert fig.data[12].name == "Q0.2"
    assert fig.data[13].name == "Q0.8"
    assert fig.data[14].name == "avg"
    assert fig.data[0].mode == "lines"
    assert fig.data[0].legendgroup is None
    assert fig.data[0].line.color is None
    assert fig.data[12].line.color == "#A3A3A3"
    assert fig.data[14].line.color == "#757575"
