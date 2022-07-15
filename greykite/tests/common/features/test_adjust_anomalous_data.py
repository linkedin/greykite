import numpy as np
import pandas as pd
import pytest

from greykite.common.constants import ADJUSTMENT_DELTA_COL
from greykite.common.constants import END_TIME_COL
from greykite.common.constants import METRIC_COL
from greykite.common.constants import START_TIME_COL
from greykite.common.constants import TIME_COL
from greykite.common.features.adjust_anomalous_data import adjust_anomalous_data
from greykite.common.testing_utils import generate_anomalous_data
from greykite.common.testing_utils import generic_test_adjust_anomalous_data


@pytest.fixture
def data():
    return generate_anomalous_data()


def test_adjust_anomalous_data(data):
    """ Tests various scenarios for ``adjust_anomalous_data``."""
    df_raw = data["df"]
    anomaly_df = data["anomaly_df"]

    # Adjusts for `"y"` for `"MOBILE"` dimension.
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
            "platform": "MOBILE"},
        filter_by_value_col=None,
        adjustment_method="add")
    adj_values = pd.Series([np.nan, np.nan, 2., 6., 7., 8., 6., 7., 8., 9.])
    generic_test_adjust_anomalous_data(
        value_col=value_col,
        adj_df_info=adj_df_info,
        adj_values=adj_values)

    # Adjusts for `"y"` for `"MOBILE"` dimension and vertical `"sales"` or `"ads"`,
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
            "platform": "MOBILE",
            "vertical": ["sales", "ads"]},
        filter_by_value_col=METRIC_COL,
        adjustment_method="subtract")

    adj_values = pd.Series([np.nan, np.nan, 2., 0., 1., 2., 6., 7., 8., 9.])
    generic_test_adjust_anomalous_data(
        value_col=value_col,
        adj_df_info=adj_df_info,
        adj_values=adj_values)

    # Adjusts for `"y"` for `"MOBILE"` dimension and vertical `"sales"`.
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
            "platform": "MOBILE",
            "vertical": "sales"},
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
            "platform": "MOBILE"},
        filter_by_value_col=None)

    adj_values = pd.Series([20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
    generic_test_adjust_anomalous_data(
        value_col=value_col,
        adj_df_info=adj_df_info,
        adj_values=adj_values)

    # Adjusts for `"z"` with DESKTOP dimension passed.
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
            "platform": "DESKTOP"},
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
            "platform": {"MOBILE", "DESKTOP"},
            "vertical": ["ads", "sales"],
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
    anomaly_df_format[START_TIME_COL] = ["1/1/2018", "1/4/2018", "1/8/2018", "1/10/2018"]
    anomaly_df_format[END_TIME_COL] = ["1/2/2018", "1/6/2018", "1/9/2018", "1/10/2018"]
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
            "platform": {"MOBILE", "DESKTOP"},
            "vertical": ["ads", "sales"],
        },
        filter_by_value_col=None)

    adj_values = pd.Series([np.nan, np.nan, 2.0, np.nan, np.nan, np.nan, 6.0,
                            np.nan, np.nan, np.nan])
    generic_test_adjust_anomalous_data(
        value_col=value_col,
        adj_df_info=adj_df_info,
        adj_values=adj_values)

    # Checks failure to convert date
    anomaly_df_format[START_TIME_COL] = ["999/999/2018", "1/4/2018", "1/8/2018", "1/10/2018"]
    anomaly_df_format[END_TIME_COL] = ["999/999/2019", "1/6/2018", "1/9/2018", "1/10/2018"]
    with pytest.warns(
            UserWarning,
            match=r"Dates could not be parsed by `pandas.to_datetime`, using string comparison "
                  r"for dates instead. Error message:\nUnknown string format: 999/999/2018"):
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
                "platform": {"MOBILE", "DESKTOP"},
                "vertical": ["ads", "sales"],
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
