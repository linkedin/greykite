import datetime

import numpy as np
import pandas as pd
import pytest

from greykite.common.constants import ADJUSTMENT_DELTA_COL
from greykite.common.constants import END_TIME_COL
from greykite.common.constants import METRIC_COL
from greykite.common.constants import START_TIME_COL
from greykite.common.constants import TIME_COL
from greykite.common.constants import VALUE_COL
from greykite.common.data_loader import DataLoader
from greykite.common.enums import TimeEnum
from greykite.common.python_utils import assert_equal
from greykite.common.testing_utils import generate_df_with_reg_for_tests
from greykite.common.time_properties import describe_timeseries
from greykite.common.time_properties import fill_missing_dates
from greykite.common.time_properties import find_missing_dates
from greykite.common.time_properties import get_canonical_data
from greykite.common.time_properties import min_gap_in_seconds


def test_describe_timeseries():
    # testing the describe timeseries function
    x = [
            datetime.datetime(2018, 1, 1, 0, 0, 1),
            datetime.datetime(2018, 1, 1, 0, 0, 2),
            datetime.datetime(2018, 1, 1, 0, 0, 3)]

    time_stats = describe_timeseries(df=pd.DataFrame({"x": x}), time_col="x")
    assert time_stats["regular_increments"] is True
    assert time_stats["min_timestamp"] == datetime.datetime(2018, 1, 1, 0, 0, 1)
    assert time_stats["max_timestamp"] == datetime.datetime(2018, 1, 1, 0, 0, 3)
    assert time_stats["mean_increment_secs"] == 1.0
    assert time_stats["min_increment_secs"] == 1.0
    assert time_stats["median_increment_secs"] == 1.0
    assert time_stats["freq_in_secs"] == 1.0
    assert time_stats["freq_in_days"] == 1/(24*3600)
    assert time_stats["freq_in_timedelta"] == datetime.timedelta(days=1/(24*3600))

    x = [
            datetime.datetime(2018, 1, 1, 0, 0, 1),
            datetime.datetime(2018, 1, 1, 0, 0, 2),
            datetime.datetime(2018, 1, 1, 0, 0, 4)]

    time_stats = describe_timeseries(df=pd.DataFrame({"x": x}), time_col="x")
    assert time_stats["regular_increments"] is False
    assert time_stats["min_timestamp"] == datetime.datetime(2018, 1, 1, 0, 0, 1)
    assert time_stats["max_timestamp"] == datetime.datetime(2018, 1, 1, 0, 0, 4)
    assert time_stats["mean_increment_secs"] == 1.5
    assert time_stats["min_increment_secs"] == 1.0
    assert time_stats["median_increment_secs"] == 1.5
    assert time_stats["freq_in_secs"] == 1.5
    assert time_stats["freq_in_days"] == 1.5/(24*3600)
    assert time_stats["freq_in_timedelta"] == datetime.timedelta(days=1.5/(24*3600))

    x = [
            datetime.datetime(2018, 1, 1, 0, 0, 3),
            datetime.datetime(2018, 1, 1, 0, 0, 2),
            datetime.datetime(2018, 1, 1, 0, 0, 1)]

    time_stats = describe_timeseries(df=pd.DataFrame({"x": x}), time_col="x")
    assert time_stats["increasing"] is False
    assert time_stats["regular_increments"] is True


def test_min_gap_in_seconds():
    """Tests min_gap_in_seconds"""
    date_list = pd.date_range(
        start=datetime.datetime(2019, 1, 1),
        periods=20,
        freq="H").tolist()
    df = pd.DataFrame({"ts": date_list})
    assert min_gap_in_seconds(df=df, time_col="ts") == TimeEnum.ONE_HOUR_IN_SECONDS.value

    date_list = pd.date_range(
        start=datetime.datetime(2019, 1, 1),
        periods=20,
        freq="6H").tolist()
    df = pd.DataFrame({"ts": date_list})
    assert min_gap_in_seconds(df=df, time_col="ts") == 6 * TimeEnum.ONE_HOUR_IN_SECONDS.value

    date_list = pd.date_range(
        start=datetime.datetime(2019, 1, 1),
        periods=20,
        freq="MIN").tolist()
    df = pd.DataFrame({"ts": date_list})
    assert min_gap_in_seconds(df=df, time_col="ts") == TimeEnum.ONE_MINUTE_IN_SECONDS.value

    date_list = pd.date_range(
        start=datetime.datetime(2019, 1, 1),
        periods=20,
        freq="5S").tolist()
    df = pd.DataFrame({"ts": date_list})
    assert min_gap_in_seconds(df=df, time_col="ts") == 5

    df = pd.DataFrame({"ts": ["2019-01-03", "2019-01-05", "2019-01-08", "2019-01-10"]})
    assert min_gap_in_seconds(df=df, time_col="ts") == 2 * TimeEnum.ONE_DAY_IN_SECONDS.value

    df = pd.DataFrame({"ts": ["2018-01-01", "2019-01-01", "2020-01-01", "2021-01-01"]})
    assert min_gap_in_seconds(df=df, time_col="ts") == TimeEnum.ONE_YEAR_IN_SECONDS.value

    df = pd.DataFrame({"ts": ["2018-01-01"]})
    with pytest.raises(ValueError, match="Must provide at least two data points. Found 1."):
        min_gap_in_seconds(df=df, time_col="ts")


def test_find_missing_dates():
    """Tests find_missing_dates function"""
    timestamp_series = pd.Series([datetime.datetime(2018, 1, 1, 0, 0, 1),
                                  datetime.datetime(2018, 1, 1, 0, 0, 2),
                                  datetime.datetime(2018, 1, 1, 0, 0, 10),
                                  datetime.datetime(2018, 1, 1, 0, 0, 4)])

    gaps = find_missing_dates(timestamp_series)
    expected_gaps = pd.DataFrame({
        "right_before_gap": pd.Series([datetime.datetime(2018, 1, 1, 0, 0, 2), datetime.datetime(2018, 1, 1, 0, 0, 4)]),
        "right_after_gap": pd.Series([datetime.datetime(2018, 1, 1, 0, 0, 4), datetime.datetime(2018, 1, 1, 0, 0, 10)]),
        "gap_size": [1.0, 5.0]
    })
    assert gaps.equals(expected_gaps)


def test_fill_missing_dates():
    """Tests fill_missing_dates function"""
    df = pd.DataFrame({
        "time": [datetime.datetime(2018, 1, 1, 0, 0, 1),
                 datetime.datetime(2018, 1, 1, 0, 0, 2),
                 datetime.datetime(2018, 1, 1, 0, 0, 10),  # intentionally out of order
                 datetime.datetime(2018, 1, 1, 0, 0, 4)],
        VALUE_COL: [1, 2, 3, 4]
    })

    expected = pd.DataFrame({
        "time": [datetime.datetime(2018, 1, 1, 0, 0, 1),
                 datetime.datetime(2018, 1, 1, 0, 0, 2),
                 datetime.datetime(2018, 1, 1, 0, 0, 3),
                 datetime.datetime(2018, 1, 1, 0, 0, 4),
                 datetime.datetime(2018, 1, 1, 0, 0, 5),
                 datetime.datetime(2018, 1, 1, 0, 0, 6),
                 datetime.datetime(2018, 1, 1, 0, 0, 7),
                 datetime.datetime(2018, 1, 1, 0, 0, 8),
                 datetime.datetime(2018, 1, 1, 0, 0, 9),
                 datetime.datetime(2018, 1, 1, 0, 0, 10)],
        VALUE_COL: [1, 2, np.nan, 4, np.nan, np.nan, np.nan, np.nan, np.nan, 3]
    })
    df_filled, added_timepoints, dropped_timepoints = fill_missing_dates(df, time_col="time", freq="S")
    assert df_filled.equals(expected)
    assert added_timepoints == 6
    assert dropped_timepoints == 0

    # warns when wrong frequency provided, data points dropped
    with pytest.warns(UserWarning) as record:
        df_filled, added_timepoints, dropped_timepoints = fill_missing_dates(df, time_col="time", freq="MS")
        assert added_timepoints == 0  # keeps the first record
        assert dropped_timepoints == 3
        assert f"Dropped {dropped_timepoints} dates when filling gaps in input data" in record[0].message.args[0]


# The following tests are all for get_canonical_data()
def test_gcd_err():
    df = pd.DataFrame({
        TIME_COL: [datetime.datetime(2018, 1, 1, 0, 0, 3),
                   datetime.datetime(2018, 1, 1, 0, 0, 2),
                   datetime.datetime(2018, 1, 1, 0, 0, 1)],
        VALUE_COL: [1, 2, 3]
    })
    with pytest.raises(ValueError, match="Time series has < 3 observations"):
        get_canonical_data(df=df.iloc[:2, ])

    with pytest.raises(ValueError, match="column is not in input data"):
        get_canonical_data(df=df, time_col="time")

    with pytest.raises(ValueError, match="column is not in input data"):
        get_canonical_data(df=df, value_col="value")


def test_gcd_dates():
    """Checks if regular data can be properly loaded. Checks time column stats"""
    df = pd.DataFrame({
        "time": [datetime.datetime(2018, 1, 1, 0, 0, 1),
                 datetime.datetime(2018, 1, 1, 0, 0, 2),
                 datetime.datetime(2018, 1, 1, 0, 0, 3)],
        "val": [1, 2, 3]
    })
    canonical_data_dict = get_canonical_data(
        df=df,
        time_col="time",
        value_col="val")
    assert_equal(canonical_data_dict["time_stats"]["gaps"], find_missing_dates(df["time"]))
    assert canonical_data_dict["time_stats"]["added_timepoints"] == 0
    assert canonical_data_dict["time_stats"]["dropped_timepoints"] == 0
    assert canonical_data_dict["freq"] == "S"
    assert_equal(canonical_data_dict["df"][VALUE_COL].values, df["val"].values)
    assert canonical_data_dict["df"].index.name is None

    # string with date format
    date_format = "%Y-%m-%d"
    df = pd.DataFrame({
        TIME_COL: ["2018-01-01", "2018-01-05", "2018-01-09"],
        VALUE_COL: [1, 2, 3]
    })
    canonical_data_dict = get_canonical_data(
        df=df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        date_format=date_format,
        freq="4D")
    expected = pd.DataFrame({
        TIME_COL: pd.Series([datetime.datetime(2018, 1, 1, 0, 0, 0),
                             datetime.datetime(2018, 1, 5, 0, 0, 0),
                             datetime.datetime(2018, 1, 9, 0, 0, 0)],
                            name=TIME_COL),
        VALUE_COL: df[VALUE_COL]
    })
    expected.index = expected[TIME_COL]
    expected.index.name = None
    assert canonical_data_dict["time_stats"]["added_timepoints"] == 0
    assert canonical_data_dict["time_stats"]["dropped_timepoints"] == 0
    assert canonical_data_dict["freq"] == "4D"
    assert_equal(canonical_data_dict["df"], expected)

    # string with inferred date format
    canonical_data_dict = get_canonical_data(
        df=df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        freq="4D")
    assert canonical_data_dict["time_stats"]["added_timepoints"] == 0
    assert canonical_data_dict["time_stats"]["dropped_timepoints"] == 0
    assert_equal(canonical_data_dict["df"], expected)

    # time zone
    canonical_data_dict = get_canonical_data(
        df=df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        freq="4D",
        tz="US/Pacific")
    expected = expected.tz_localize("US/Pacific")
    assert canonical_data_dict["time_stats"]["added_timepoints"] == 0
    assert canonical_data_dict["time_stats"]["dropped_timepoints"] == 0
    assert_equal(canonical_data_dict["df"], expected)


def test_gcd_freq():
    # Checks warning for frequency not matching inferred frequency
    df = pd.DataFrame({
        TIME_COL: [datetime.datetime(2018, 1, 1, 0, 0, 0),
                   datetime.datetime(2018, 1, 2, 0, 0, 0),
                   datetime.datetime(2018, 1, 3, 0, 0, 0)],
        VALUE_COL: [1, 2, 3]
    })
    with pytest.warns(UserWarning) as record:
        canonical_data_dict = get_canonical_data(
            df=df,
            time_col=TIME_COL,
            value_col=VALUE_COL,
            freq="H")
        assert "does not match inferred frequency" in record[0].message.args[0]
        assert canonical_data_dict["time_stats"]["added_timepoints"] == 24*2-2
        assert canonical_data_dict["time_stats"]["dropped_timepoints"] == 0


def test_gcd_irregular():
    """Checks sort and fill missing dates"""
    # gaps in unsorted, irregular input
    df = pd.DataFrame({
        TIME_COL: [datetime.datetime(2018, 1, 1, 0, 0, 1),
                   datetime.datetime(2018, 1, 1, 0, 0, 2),
                   datetime.datetime(2018, 1, 1, 0, 0, 10),  # intentionally out of order
                   datetime.datetime(2018, 1, 1, 0, 0, 4)],
        VALUE_COL: [1, 2, 3, 4]
    })
    expected = pd.DataFrame({
        # in sorted order
        TIME_COL: pd.date_range(
            start=datetime.datetime(2018, 1, 1, 0, 0, 1),
            end=datetime.datetime(2018, 1, 1, 0, 0, 10),
            freq="S"),
        VALUE_COL: [1, 2, np.nan, 4, np.nan, np.nan, np.nan, np.nan, np.nan, 3]
    })
    expected.index = expected[TIME_COL]
    expected.index.name = None

    canonical_data_dict = get_canonical_data(
        df=df,
        time_col=TIME_COL,
        value_col=VALUE_COL,
        freq="S")  # the frequency should be provided when there are gaps
    assert canonical_data_dict["time_stats"]["added_timepoints"] == 6
    assert canonical_data_dict["time_stats"]["dropped_timepoints"] == 0
    assert_equal(canonical_data_dict["df"], expected)
    assert_equal(canonical_data_dict["time_stats"]["gaps"], find_missing_dates(df[TIME_COL]))


def test_gcd_load_data_anomaly():
    """Checks anomaly_info parameter"""
    dl = DataLoader()
    df = dl.load_beijing_pm()
    value_col = "pm"

    # no anomaly adjustment
    canonical_data_dict = get_canonical_data(
        df=df,
        time_col=TIME_COL,
        value_col=value_col)
    assert canonical_data_dict["df_before_adjustment"] is None

    dim_one = "one"
    dim_two = "two"
    anomaly_df = pd.DataFrame({
        START_TIME_COL: ["2011-04-04-10", "2011-10-10-00", "2012-12-20-10"],
        END_TIME_COL: ["2011-04-05-20", "2011-10-11-23", "2012-12-20-13"],
        ADJUSTMENT_DELTA_COL: [np.nan, 100.0, -100.0],
        METRIC_COL: [dim_one, dim_one, dim_two]  # used to filter rows in this df
    })
    # Adjusts one column (value_col)
    anomaly_info = {
        "value_col": value_col,
        "anomaly_df": anomaly_df,
        "start_time_col": START_TIME_COL,
        "end_time_col": END_TIME_COL,
        "adjustment_delta_col": ADJUSTMENT_DELTA_COL,
        "filter_by_dict": {METRIC_COL: dim_one},
        "adjustment_method": "add"
    }
    canonical_data_dict2 = get_canonical_data(
        df=df,
        time_col=TIME_COL,
        value_col=value_col,
        anomaly_info=anomaly_info)
    assert_equal(canonical_data_dict2["df_before_adjustment"], canonical_data_dict["df"])
    expected_df = canonical_data_dict["df"].copy()
    # first anomaly
    idx = ((expected_df[TIME_COL] >= anomaly_df[START_TIME_COL][0])
           & (expected_df[TIME_COL] <= anomaly_df[END_TIME_COL][0]))
    expected_df.loc[idx, VALUE_COL] = np.nan
    # second anomaly
    idx = ((expected_df[TIME_COL] >= anomaly_df[START_TIME_COL][1])
           & (expected_df[TIME_COL] <= anomaly_df[END_TIME_COL][1]))
    expected_df.loc[idx, VALUE_COL] += 100.0
    assert_equal(canonical_data_dict2["df"], expected_df)

    # Adjusts two columns
    value_col_two = "pres"  # second column to adjust
    anomaly_info = [anomaly_info, {
        "value_col": value_col_two,
        "anomaly_df": anomaly_df,
        "start_time_col": START_TIME_COL,
        "end_time_col": END_TIME_COL,
        "adjustment_delta_col": ADJUSTMENT_DELTA_COL,
        "filter_by_dict": {METRIC_COL: dim_two},
        "adjustment_method": "subtract"
    }]
    canonical_data_dict3 = get_canonical_data(
        df=df,
        time_col=TIME_COL,
        value_col=value_col,
        anomaly_info=anomaly_info)
    # third anomaly. The value is subtracted, according to `adjustment_method`.
    idx = ((expected_df[TIME_COL] >= anomaly_df[START_TIME_COL][2])
           & (expected_df[TIME_COL] <= anomaly_df[END_TIME_COL][2]))
    expected_df.loc[idx, value_col_two] -= -100.0
    assert_equal(canonical_data_dict3["df_before_adjustment"], canonical_data_dict["df"])
    assert_equal(canonical_data_dict3["df"], expected_df)


def test_gcd_train_end_date():
    """Tests train_end_date for data without regressors"""
    df = pd.DataFrame({
        TIME_COL: [datetime.datetime(2018, 1, 1, 3, 0, 0),
                   datetime.datetime(2018, 1, 1, 4, 0, 0),
                   datetime.datetime(2018, 1, 1, 5, 0, 0),
                   datetime.datetime(2018, 1, 1, 6, 0, 0),
                   datetime.datetime(2018, 1, 1, 7, 0, 0)],
        VALUE_COL: [1, np.nan, 3, np.nan, np.nan],
    })

    # no train_end_date
    with pytest.warns(UserWarning) as record:
        train_end_date = datetime.datetime(2018, 1, 1, 5, 0, 0)
        get_canonical_data(df=df)
        assert f"{VALUE_COL} column of the provided TimeSeries contains null " \
               f"values at the end. Setting 'train_end_date' to the last timestamp with a " \
               f"non-null value ({train_end_date})." in record[0].message.args[0]

    # train_end_date later than last date in df
    with pytest.warns(UserWarning) as record:
        train_end_date = datetime.datetime(2018, 1, 1, 8, 0, 0)
        result_train_end_date = datetime.datetime(2018, 1, 1, 5, 0, 0)
        get_canonical_data(df, train_end_date=train_end_date)
        assert f"Input timestamp for the parameter 'train_end_date' " \
               f"({train_end_date}) either exceeds the last available timestamp or" \
               f"{VALUE_COL} column of the provided TimeSeries contains null " \
               f"values at the end. Setting 'train_end_date' to the last timestamp with a " \
               f"non-null value ({result_train_end_date})." in record[0].message.args[0]

    # train_end_date in between last date in df and last date before
    with pytest.warns(UserWarning) as record:
        train_end_date = datetime.datetime(2018, 1, 1, 6, 0, 0)
        result_train_end_date = datetime.datetime(2018, 1, 1, 5, 0, 0)
        get_canonical_data(df, train_end_date=train_end_date)
        assert f"Input timestamp for the parameter 'train_end_date' " \
               f"({train_end_date}) either exceeds the last available timestamp or" \
               f"{VALUE_COL} column of the provided TimeSeries contains null " \
               f"values at the end. Setting 'train_end_date' to the last timestamp with a " \
               f"non-null value ({result_train_end_date})." in record[0].message.args[0]

    # train end date equal to last date before null
    canonical_data_dict = get_canonical_data(df, train_end_date=datetime.datetime(2018, 1, 1, 5, 0, 0))
    assert canonical_data_dict["train_end_date"] == datetime.datetime(2018, 1, 1, 5, 0, 0)

    # train_end_date smaller than last date before null
    canonical_data_dict = get_canonical_data(df, train_end_date=datetime.datetime(2018, 1, 1, 4, 0, 0))
    assert_equal(canonical_data_dict["fit_df"], canonical_data_dict["df"].iloc[:2])
    assert canonical_data_dict["regressor_cols"] == []
    assert canonical_data_dict["fit_cols"] == [TIME_COL, VALUE_COL]
    assert canonical_data_dict["train_end_date"] == datetime.datetime(2018, 1, 1, 4, 0, 0)
    assert canonical_data_dict["last_date_for_val"] == datetime.datetime(2018, 1, 1, 5, 0, 0)
    assert canonical_data_dict["last_date_for_reg"] is None


def test_gcd_train_end_date_regressor():
    """Tests train_end_date for data with regressors"""
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

    # last date with a value
    result_train_end_date = datetime.datetime(2018, 1, 22)

    # default train_end_date, default regressor_cols
    with pytest.warns(UserWarning) as record:
        canonical_data_dict = get_canonical_data(
            df=df,
            train_end_date=None,
            regressor_cols=None)
        assert f"{VALUE_COL} column of the provided TimeSeries contains null " \
               f"values at the end. Setting 'train_end_date' to the last timestamp with a " \
               f"non-null value ({result_train_end_date})." in record[0].message.args[0]
        assert canonical_data_dict["df"].shape == df.shape
        assert canonical_data_dict["fit_df"].shape == (22, 2)
        assert canonical_data_dict["regressor_cols"] == []
        assert canonical_data_dict["fit_cols"] == [TIME_COL, VALUE_COL]
        assert canonical_data_dict["train_end_date"] == result_train_end_date
        assert canonical_data_dict["last_date_for_val"] == result_train_end_date
        assert canonical_data_dict["last_date_for_reg"] is None

    # train_end_date later than last date in df, all available regressor_cols
    with pytest.warns(UserWarning) as record:
        train_end_date = datetime.datetime(2018, 2, 10)
        canonical_data_dict = get_canonical_data(
            df=df,
            train_end_date=train_end_date,
            regressor_cols=regressor_cols)
        assert f"Input timestamp for the parameter 'train_end_date' " \
               f"({train_end_date}) either exceeds the last available timestamp or" \
               f"{VALUE_COL} column of the provided TimeSeries contains null " \
               f"values at the end. Setting 'train_end_date' to the last timestamp with a " \
               f"non-null value ({result_train_end_date})." in record[0].message.args[0]
        assert canonical_data_dict["fit_df"].shape == (22, 5)
        assert canonical_data_dict["regressor_cols"] == regressor_cols
        assert canonical_data_dict["fit_cols"] == [TIME_COL, VALUE_COL] + regressor_cols
        assert canonical_data_dict["train_end_date"] == result_train_end_date
        assert canonical_data_dict["last_date_for_val"] == datetime.datetime(2018, 1, 22)
        assert canonical_data_dict["last_date_for_reg"] == datetime.datetime(2018, 1, 28)

    # train_end_date in between last date in df and last date before null
    # user passes no regressor_cols
    with pytest.warns(UserWarning) as record:
        train_end_date = datetime.datetime(2018, 1, 25)
        canonical_data_dict = get_canonical_data(
            df=df,
            train_end_date=train_end_date,
            regressor_cols=None)
        assert f"Input timestamp for the parameter 'train_end_date' " \
               f"({train_end_date}) either exceeds the last available timestamp or" \
               f"{VALUE_COL} column of the provided TimeSeries contains null " \
               f"values at the end. Setting 'train_end_date' to the last timestamp with a " \
               f"non-null value ({result_train_end_date})." in record[0].message.args[0]
        assert canonical_data_dict["fit_df"].shape == (22, 2)
        assert canonical_data_dict["regressor_cols"] == []
        assert canonical_data_dict["fit_cols"] == [TIME_COL, VALUE_COL]
        assert canonical_data_dict["train_end_date"] == datetime.datetime(2018, 1, 22)
        assert canonical_data_dict["last_date_for_val"] == datetime.datetime(2018, 1, 22)
        assert canonical_data_dict["last_date_for_reg"] is None

    # train end date equal to last date before null
    # user requests a subset of the regressor_cols
    train_end_date = datetime.datetime(2018, 1, 22)
    regressor_cols = ["regressor2"]
    canonical_data_dict = get_canonical_data(
        df=df,
        train_end_date=train_end_date,
        regressor_cols=regressor_cols)
    assert canonical_data_dict["fit_df"].shape == (22, 3)
    assert canonical_data_dict["regressor_cols"] == regressor_cols
    assert canonical_data_dict["fit_cols"] == [TIME_COL, VALUE_COL] + regressor_cols
    assert canonical_data_dict["train_end_date"] == datetime.datetime(2018, 1, 22)
    assert canonical_data_dict["last_date_for_val"] == datetime.datetime(2018, 1, 22)
    assert canonical_data_dict["last_date_for_reg"] == datetime.datetime(2018, 1, 26)

    # train_end_date smaller than last date before null
    # user requests regressor_cols that does not exist in df
    with pytest.warns(UserWarning) as record:
        train_end_date = datetime.datetime(2018, 1, 20)
        regressor_cols = ["regressor1", "regressor4", "regressor5"]
        canonical_data_dict = get_canonical_data(
            df=df,
            train_end_date=train_end_date,
            regressor_cols=regressor_cols)
        assert canonical_data_dict["fit_df"].shape == (20, 3)
        assert canonical_data_dict["regressor_cols"] == ["regressor1"]
        assert canonical_data_dict["fit_cols"] == [TIME_COL, VALUE_COL, "regressor1"]
        assert canonical_data_dict["train_end_date"] == datetime.datetime(2018, 1, 20)
        assert canonical_data_dict["last_date_for_val"] == datetime.datetime(2018, 1, 22)
        assert canonical_data_dict["last_date_for_reg"] == datetime.datetime(2018, 1, 28)
        assert (f"The following columns are not available to use as "
                f"regressors: ['regressor4', 'regressor5']") in record[0].message.args[0]
