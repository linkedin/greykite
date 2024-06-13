import numpy as np
import pandas as pd
import pytest

from greykite.common.constants import ADJUSTMENT_DELTA_COL
from greykite.common.constants import END_TIME_COL
from greykite.common.constants import METRIC_COL
from greykite.common.constants import START_TIME_COL
from greykite.common.constants import TIME_COL
from greykite.common.features.adjust_anomalous_data import adjust_anomalous_data
from greykite.common.features.adjust_anomalous_data import label_anomalies_multi_metric
from greykite.common.testing_utils import generate_anomalous_data
from greykite.common.testing_utils import generic_test_adjust_anomalous_data


@pytest.fixture
def data():
    return generate_anomalous_data()


def test_adjust_anomalous_data(data):
    """ Tests various scenarios for ``adjust_anomalous_data``."""
    df_raw = data["df"]
    anomaly_df = data["anomaly_df"]

    # Adjusts for `"y"` for `"level_1"` dimension.
    value_col = "y"
    adj_df_info = adjust_anomalous_data(
        df=df_raw,
        time_col=TIME_COL,
        value_col=value_col,
        anomaly_df=anomaly_df,
        start_time_col=START_TIME_COL,
        end_time_col=END_TIME_COL,
        adjustment_delta_col=ADJUSTMENT_DELTA_COL,
        filter_by_dict={
            METRIC_COL: value_col,
            "dimension1": "level_1"},
        filter_by_value_col=None,
        adjustment_method="add")
    adj_values = pd.Series([np.nan, np.nan, 2., 6., 7., 8., 6., 7., 8., 9.])
    generic_test_adjust_anomalous_data(
        value_col=value_col,
        adj_df_info=adj_df_info,
        adj_values=adj_values)

    # Adjusts for `"y"` for `"level_1"` dimension and vertical `"level_2"` or `"level_1"`,
    # adjustment_method = `"subtract"`
    value_col = "y"
    adj_df_info = adjust_anomalous_data(
        df=df_raw,
        time_col=TIME_COL,
        value_col=value_col,
        anomaly_df=anomaly_df,
        start_time_col=START_TIME_COL,
        end_time_col=END_TIME_COL,
        adjustment_delta_col=ADJUSTMENT_DELTA_COL,
        filter_by_dict={
            "dimension1": "level_1",
            "dimension2": ["level_2", "level_1"]},
        filter_by_value_col=METRIC_COL,
        adjustment_method="subtract")

    adj_values = pd.Series([np.nan, np.nan, 2., 0., 1., 2., 6., 7., 8., 9.])
    generic_test_adjust_anomalous_data(
        value_col=value_col,
        adj_df_info=adj_df_info,
        adj_values=adj_values)

    # Adjusts for `"y"` for `"level_1"` dimension and vertical `"level_2"`.
    value_col = "y"
    adj_df_info = adjust_anomalous_data(
        df=df_raw,
        time_col=TIME_COL,
        value_col=value_col,
        anomaly_df=anomaly_df,
        start_time_col=START_TIME_COL,
        end_time_col=END_TIME_COL,
        adjustment_delta_col=ADJUSTMENT_DELTA_COL,
        filter_by_dict={
            METRIC_COL: value_col,
            "dimension1": "level_1",
            "dimension2": "level_2"},
        filter_by_value_col=None)

    adj_values = pd.Series([0., 1., 2., 6., 7., 8., 6., 7., 8., 9.])
    generic_test_adjust_anomalous_data(
        value_col=value_col,
        adj_df_info=adj_df_info,
        adj_values=adj_values)

    # Adjusts for `"z"`.
    # Since for `"z"` only Desktop is impacted, no change is expected.
    value_col = "z"
    adj_df_info = adjust_anomalous_data(
        df=df_raw,
        time_col=TIME_COL,
        value_col="z",
        anomaly_df=anomaly_df,
        start_time_col=START_TIME_COL,
        end_time_col=END_TIME_COL,
        adjustment_delta_col=ADJUSTMENT_DELTA_COL,
        filter_by_dict={
            METRIC_COL: value_col,
            "dimension1": "level_1"},
        filter_by_value_col=None)

    adj_values = pd.Series([20., 21., 22., 23., 24., 25., 26., 27., 28., 29.])
    generic_test_adjust_anomalous_data(
        value_col=value_col,
        adj_df_info=adj_df_info,
        adj_values=adj_values)

    # Adjusts for `"z"` with level_2 dimension passed.
    # Since for `"z"` Desktop is impacted, changes are expected.
    value_col = "z"
    adj_df_info = adjust_anomalous_data(
        df=df_raw,
        time_col=TIME_COL,
        value_col="z",
        anomaly_df=anomaly_df,
        start_time_col=START_TIME_COL,
        end_time_col=END_TIME_COL,
        adjustment_delta_col=ADJUSTMENT_DELTA_COL,
        filter_by_dict={
            METRIC_COL: value_col,
            "dimension1": "level_2"},
        filter_by_value_col=None)

    adj_values = pd.Series([20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 22.0, 23.0, np.nan])
    generic_test_adjust_anomalous_data(
        value_col=value_col,
        adj_df_info=adj_df_info,
        adj_values=adj_values)

    # Adjusts for `"y"` with no additional dimensions filter.
    value_col = "y"
    adj_df_info = adjust_anomalous_data(
        df=df_raw,
        time_col=TIME_COL,
        value_col="y",
        anomaly_df=anomaly_df,
        start_time_col=START_TIME_COL,
        end_time_col=END_TIME_COL,
        adjustment_delta_col=ADJUSTMENT_DELTA_COL,
        filter_by_dict={METRIC_COL: value_col},
        filter_by_value_col=None)

    adj_values = pd.Series([np.nan, np.nan, 2.0, 6.0, 7.0, 8.0, 6.0, 7.0, 8.0, 9.0])
    generic_test_adjust_anomalous_data(
        value_col=value_col,
        adj_df_info=adj_df_info,
        adj_values=adj_values)

    # Same as above, using `filter_by_value_col`
    value_col = "y"
    adj_df_info = adjust_anomalous_data(
        df=df_raw,
        time_col=TIME_COL,
        value_col="y",
        anomaly_df=anomaly_df,
        start_time_col=START_TIME_COL,
        end_time_col=END_TIME_COL,
        adjustment_delta_col=ADJUSTMENT_DELTA_COL,
        filter_by_dict=None,
        filter_by_value_col=METRIC_COL)

    adj_values = pd.Series([np.nan, np.nan, 2.0, 6.0, 7.0, 8.0, 6.0, 7.0, 8.0, 9.0])
    generic_test_adjust_anomalous_data(
        value_col=value_col,
        adj_df_info=adj_df_info,
        adj_values=adj_values)

    # This does not use the metric column to match or dimensions.
    # However it does use the adjustment delta column.
    value_col = "y"
    adj_df_info = adjust_anomalous_data(
        df=df_raw,
        time_col=TIME_COL,
        value_col=value_col,
        anomaly_df=anomaly_df,
        start_time_col=START_TIME_COL,
        end_time_col=END_TIME_COL,
        adjustment_delta_col=ADJUSTMENT_DELTA_COL,
        filter_by_dict=None,
        filter_by_value_col=None)

    adj_values = pd.Series([np.nan, np.nan, 2.0, 6.0, 7.0, 8.0, 6.0, 2.0, 3.0, np.nan])
    generic_test_adjust_anomalous_data(
        value_col=value_col,
        adj_df_info=adj_df_info,
        adj_values=adj_values)

    # This does not match on metric column or dimensions.
    # Therefore all the anomalies will be used to adjust `"y"`
    # Also does not use adjustment delta, therefore every anomaly is mapped to `np.nan`
    value_col = "y"
    adj_df_info = adjust_anomalous_data(
        df=df_raw,
        time_col=TIME_COL,
        value_col=value_col,
        anomaly_df=anomaly_df,
        start_time_col=START_TIME_COL,
        end_time_col=END_TIME_COL,
        adjustment_delta_col=None,
        filter_by_dict=None,
        filter_by_value_col=None)

    adj_values = pd.Series([np.nan, np.nan, 2.0, np.nan, np.nan, np.nan, 6.0,
                            np.nan, np.nan, np.nan])
    generic_test_adjust_anomalous_data(
        value_col=value_col,
        adj_df_info=adj_df_info,
        adj_values=adj_values)

    # Same as above, filter criteria are always met
    value_col = "y"
    adj_df_info = adjust_anomalous_data(
        df=df_raw,
        time_col=TIME_COL,
        value_col=value_col,
        anomaly_df=anomaly_df,
        start_time_col=START_TIME_COL,
        end_time_col=END_TIME_COL,
        adjustment_delta_col=None,
        filter_by_dict={
            METRIC_COL: ("y", "z"),
            "dimension1": {"level_1", "level_2"},
            "dimension2": ["level_1", "level_2"],
        },
        filter_by_value_col=None)

    adj_values = pd.Series([np.nan, np.nan, 2.0, np.nan, np.nan, np.nan, 6.0,
                            np.nan, np.nan, np.nan])
    generic_test_adjust_anomalous_data(
        value_col=value_col,
        adj_df_info=adj_df_info,
        adj_values=adj_values)

    # Successful date conversion for comparison.
    # The provided dates are strings in `anomaly_df_format`, datetime in `df_raw`.
    anomaly_df_format = anomaly_df.copy()
    anomaly_df_format[START_TIME_COL] = ["1/1/2018", "1/4/2018", "1/8/2018", "1/10/2018", "1/1/2099"]
    anomaly_df_format[END_TIME_COL] = ["1/2/2018", "1/6/2018", "1/9/2018", "1/10/2018", "1/1/2099"]
    value_col = "y"
    adj_df_info = adjust_anomalous_data(
        df=df_raw,
        time_col=TIME_COL,
        value_col=value_col,
        anomaly_df=anomaly_df_format,
        start_time_col=START_TIME_COL,
        end_time_col=END_TIME_COL,
        adjustment_delta_col=None,
        filter_by_dict={
            METRIC_COL: ("y", "z"),
            "dimension1": {"level_1", "level_2"},
            "dimension2": ["level_1", "level_2"],
        },
        filter_by_value_col=None)

    adj_values = pd.Series([np.nan, np.nan, 2.0, np.nan, np.nan, np.nan, 6.0,
                            np.nan, np.nan, np.nan])
    generic_test_adjust_anomalous_data(
        value_col=value_col,
        adj_df_info=adj_df_info,
        adj_values=adj_values)

    # Checks failure to convert date
    anomaly_df_format[START_TIME_COL] = ["999/999/2018", "1/4/2018", "1/8/2018", "1/10/2018", "1/1/2099"]
    anomaly_df_format[END_TIME_COL] = ["999/999/2019", "1/6/2018", "1/9/2018", "1/10/2018", "1/1/2099"]
    with pytest.warns(
            UserWarning,
            match=r"Dates could not be parsed by `pandas.to_datetime`, using string comparison "
                  r"for dates instead. Error message:\nUnknown datetime string format, unable to parse: 999/999/2018"):
        value_col = "y"
        adjust_anomalous_data(
            df=df_raw,
            time_col=TIME_COL,
            value_col=value_col,
            anomaly_df=anomaly_df_format,
            start_time_col=START_TIME_COL,
            end_time_col=END_TIME_COL,
            adjustment_delta_col=None,
            filter_by_dict={
                METRIC_COL: ("y", "z"),
                "dimension1": {"level_1", "level_2"},
                "dimension2": ["level_1", "level_2"],
            },
            filter_by_value_col=None)

    # Checks the bad column name exception
    expected_match = "`df` cannot include this column name"
    df_bad_col_name = df_raw.copy()
    df_bad_col_name["adjusted_y"] = np.nan
    with pytest.raises(ValueError, match=expected_match):
        value_col = "y"
        adjust_anomalous_data(
            df=df_bad_col_name,
            time_col=TIME_COL,
            value_col=value_col,
            anomaly_df=anomaly_df,
            start_time_col=START_TIME_COL,
            end_time_col=END_TIME_COL,
            adjustment_delta_col=None,
            filter_by_dict=None,
            filter_by_value_col=None)

    # Checks the bad timestamps exception where a start date is after end date
    # Test is done by inputting the column names for
    # ``start_time_col`` and ``end_time_col`` in reverse order.
    expected_match = "End Time:"
    with pytest.raises(ValueError, match=expected_match):
        value_col = "y"
        adjust_anomalous_data(
            df=df_raw,
            time_col=TIME_COL,
            value_col=value_col,
            anomaly_df=anomaly_df,
            start_time_col=END_TIME_COL,
            end_time_col=START_TIME_COL,
            adjustment_delta_col=None,
            filter_by_dict=None,
            filter_by_value_col=None)

    expected_match = "Column 'device' was requested"
    with pytest.raises(ValueError, match=expected_match):
        value_col = "y"
        adjust_anomalous_data(
            df=df_raw,
            time_col=TIME_COL,
            value_col=value_col,
            anomaly_df=anomaly_df,
            start_time_col=START_TIME_COL,
            end_time_col=END_TIME_COL,
            adjustment_delta_col=None,
            filter_by_dict={"device": "iPhone"},
            filter_by_value_col=None)


def test_label_anomalies_multi_metric():
    """Tests ``label_anomalies_multi_metric``"""
    anomaly_df = pd.DataFrame({
        "start_time": ["2020-01-01", "2020-02-01", "2020-01-02", "2020-02-02", "2020-02-05"],
        "end_time": ["2020-01-03", "2020-02-04", "2020-01-05", "2020-02-06", "2020-02-08"],
        "metric": ["impressions", "impressions", "clicks", "clicks", "bookings"]
        })

    ts = pd.date_range(start="2019-12-01", end="2020-03-01", freq="D")

    np.random.seed(1317)
    df = pd.DataFrame({"ts": ts})
    size = len(df)
    value_cols = ["impressions", "clicks", "bookings"]
    df["impressions"] = np.random.normal(loc=0.0, scale=1.0, size=size)
    df["clicks"] = np.random.normal(loc=1.0, scale=1.0, size=size)
    df["bookings"] = np.random.normal(loc=2.0, scale=1.0, size=size)

    res = label_anomalies_multi_metric(
        df=df,
        time_col="ts",
        value_cols=value_cols,
        anomaly_df=anomaly_df,
        anomaly_df_grouping_col="metric",
        start_time_col="start_time",
        end_time_col="end_time")

    augmented_df = res["augmented_df"]
    is_anomaly_cols = res["is_anomaly_cols"]
    anomaly_value_cols = res["anomaly_value_cols"]
    normal_value_cols = res["normal_value_cols"]

    assert len(augmented_df) == len(df)

    assert is_anomaly_cols == [
        "impressions_is_anomaly",
        "clicks_is_anomaly",
        "bookings_is_anomaly"]

    assert anomaly_value_cols == [
        "impressions_anomaly_value",
        "clicks_anomaly_value",
        "bookings_anomaly_value"]

    assert normal_value_cols == [
        "impressions_normal_value",
        "clicks_normal_value",
        "bookings_normal_value"]

    assert set(augmented_df.columns) == set(
        ["ts"] +
        value_cols +
        is_anomaly_cols +
        anomaly_value_cols +
        normal_value_cols)

    # From the above data its clear this is the range for bookings anomalies and non-anomalies
    bookings_anomaly_time_ind = (df["ts"] >= "2020-02-05") & (df["ts"] <= "2020-02-08")
    bookings_normal_time_ind = (df["ts"] < "2020-02-05") | (df["ts"] > "2020-02-08")

    # We expect this to only consist of missing values
    x = augmented_df.loc[bookings_anomaly_time_ind]["bookings_normal_value"]
    assert x.isnull().all()

    # We expect this to only consist of missing values
    x = augmented_df.loc[bookings_normal_time_ind, "bookings_anomaly_value"]
    assert x.isnull().all()

    # Checks for anomaly values
    x = augmented_df.loc[bookings_anomaly_time_ind, "bookings_anomaly_value"]
    y = df.loc[bookings_anomaly_time_ind, "bookings"]
    assert (x == y).all()

    # Checks for normal values
    x = augmented_df.loc[bookings_normal_time_ind, "bookings_normal_value"]
    y = df.loc[bookings_normal_time_ind, "bookings"]
    assert (x == y).all()

    # Tests for raising ``ValueError`` due to non existing columns
    expected_match = "time_col:"
    with pytest.raises(ValueError, match=expected_match):
        label_anomalies_multi_metric(
            df=df,
            time_col="timestamp",  # non-existing column name
            value_cols=value_cols,
            anomaly_df=anomaly_df,
            anomaly_df_grouping_col="metric",
            start_time_col="start_time",
            end_time_col="end_time")

    expected_match = "value_col:"
    with pytest.raises(ValueError, match=expected_match):
        label_anomalies_multi_metric(
            df=df,
            time_col="ts",
            value_cols=["abandons", "impressions"],  # non-existing column name
            anomaly_df=anomaly_df,
            anomaly_df_grouping_col="metric",
            start_time_col="start_time",
            end_time_col="end_time")

    expected_match = "start_time_col:"
    with pytest.raises(ValueError, match=expected_match):
        label_anomalies_multi_metric(
            df=df,
            time_col="ts",
            value_cols=["impressions"],
            anomaly_df=anomaly_df,
            anomaly_df_grouping_col="metric",
            start_time_col="start_ts",  # non-existing column name
            end_time_col="end_time")

    expected_match = "end_time_col:"
    with pytest.raises(ValueError, match=expected_match):
        label_anomalies_multi_metric(
            df=df,
            time_col="ts",
            value_cols=["impressions"],
            anomaly_df=anomaly_df,
            anomaly_df_grouping_col="metric",
            start_time_col="start_time",
            end_time_col="end_ts")  # non-existing column name
