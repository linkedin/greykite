import numpy as np
import pandas as pd
import pytest

from greykite.common.features.timeseries_lags import build_agg_lag_df
from greykite.common.features.timeseries_lags import build_autoreg_df
from greykite.common.features.timeseries_lags import build_autoreg_df_multi
from greykite.common.features.timeseries_lags import build_lag_df
from greykite.common.features.timeseries_lags import min_max_lag_order
from greykite.common.python_utils import assert_equal


def test_build_lag_df():
    """Testing build_lag_df."""
    df = pd.DataFrame({"x": range(10), "y": range(100, 110)})
    lag_info = build_lag_df(df=df, value_col="x", max_order=3, orders=None)
    lag_df = lag_info["lag_df"]

    assert list(lag_df.columns) == ["x_lag1", "x_lag2", "x_lag3"], \
        "The expected column names were not found in lags dataframe (lag_df)"
    assert lag_df["x_lag1"].values[2].round(0) == 1.0, (
        "lag value is not correct")
    assert lag_df["x_lag2"].values[7].round(0) == 5.0, (
        "lag value is not correct")

    # example with orders provided
    lag_info = build_lag_df(
        value_col="x",
        df=df,
        max_order=None,
        orders=[1, 2, 5])

    lag_df = lag_info["lag_df"]
    assert list(lag_df.columns) == ["x_lag1", "x_lag2", "x_lag5"], \
        "The expected column names were not found in lags dataframe (lag_df)"
    assert lag_df["x_lag1"].values[2].round(0) == 1.0, (
        "lag value is not correct")
    assert lag_df["x_lag2"].values[7].round(0) == 5.0, (
        "lag value is not correct")
    assert lag_df["x_lag5"].values[8].round(0) == 3.0, (
        "lag value is not correct")


def test_build_lag_df_exception():
    df = pd.DataFrame({"x": range(10), "y": range(100, 110)})
    with pytest.raises(
            ValueError,
            match="at least one of 'max_order' or 'orders' must be provided"):
        build_lag_df(
            value_col="x",
            df=df,
            max_order=None,
            orders=None)


def test_build_lag_df_col_names_only():
    """Testing for the case where no df is passed and only col_names are generated."""
    lag_info = build_lag_df(
        value_col="x",
        df=None,
        max_order=3,
        orders=None)

    col_names = lag_info["col_names"]

    assert col_names == ["x_lag1", "x_lag2", "x_lag3"], (
        "The expected column names were not found in lags dataframe (lag_df)")

    assert lag_info["lag_df"] is None, "returned lag_df must be None"


def test_min_max_lag_order():
    """Testing max_lag_order for various cases."""
    # case with no lags
    agg_lag_dict = {
        "orders_list": [],
        "interval_list": []}
    lag_dict = {
        "orders": None,
        "max_order": None
    }

    min_max_order = min_max_lag_order(
        lag_dict=lag_dict,
        agg_lag_dict=agg_lag_dict)
    min_order = min_max_order["min_order"]
    max_order = min_max_order["max_order"]

    assert max_order == 0, (
        "max_order is not calculated correctly")

    assert min_order == np.inf, (
        "min_order is not calculated correctly")

    # case with lag_dict only including lags
    agg_lag_dict = {
        "orders_list": [],
        "interval_list": []}
    lag_dict = {
        "orders": [2, 3, 13],
        "max_order": None
    }

    min_max_order = min_max_lag_order(
        lag_dict=lag_dict,
        agg_lag_dict=agg_lag_dict)
    min_order = min_max_order["min_order"]
    max_order = min_max_order["max_order"]

    assert max_order == 13, (
        "max_order is not calculated correctly")
    assert min_order == 2, (
        "max_order is not calculated correctly")

    # `max_order` below is expected to be ignored
    # since `orders` is provided
    lag_dict = {
        "orders": [2, 3, 13],
        "max_order": 20
    }
    agg_lag_dict = None

    min_max_order = min_max_lag_order(
        lag_dict=lag_dict,
        agg_lag_dict=agg_lag_dict)
    min_order = min_max_order["min_order"]
    max_order = min_max_order["max_order"]

    assert max_order == 13, (
        "max_order is not calculated correctly")
    assert min_order == 2, (
        "max_order is not calculated correctly")

    # case with agg_lag_dict inclduing lags only
    agg_lag_dict = {
        "orders_list": [[1, 2, 3, 16]],
        "interval_list": [[1, 2]]}
    lag_dict = {
        "orders": None,
        "max_order": None
    }

    min_max_order = min_max_lag_order(
        lag_dict=lag_dict,
        agg_lag_dict=agg_lag_dict)
    min_order = min_max_order["min_order"]
    max_order = min_max_order["max_order"]

    assert max_order == 16, (
        "max_order is not calculated correctly")
    assert min_order == 1, (
        "max_order is not calculated correctly")

    # case with both agg_lag_dict and lag_dict prescribing lags
    agg_lag_dict = {
        "orders_list": [[1, 2, 3]],
        "interval_list": [(2, 3), (2, 5)]}
    lag_dict = {
        "orders": [2, 3, 8],
        "max_order": None
    }

    min_max_order = min_max_lag_order(
        lag_dict=lag_dict,
        agg_lag_dict=agg_lag_dict)
    min_order = min_max_order["min_order"]
    max_order = min_max_order["max_order"]

    assert max_order == 8, (
        "max_order is not calculated correctly")

    assert min_order == 1, (
        "min_order is not calculated correctly")

    # case with max_order appearing in lag_dict["max_order"]
    agg_lag_dict = {
        "orders_list": [[2, 3]],
        "interval_list": [(3, 6), (3, 10)]}
    lag_dict = {
        "orders": None,
        "max_order": 18
    }

    min_max_order = min_max_lag_order(
        lag_dict=lag_dict,
        agg_lag_dict=agg_lag_dict)
    min_order = min_max_order["min_order"]
    max_order = min_max_order["max_order"]

    assert max_order == 18, (
        "max_order is not calculated correctly")

    assert min_order == 1, (
        "min_order is not calculated correctly")


def test_build_agg_lag_df():
    """Testing build_agg_lag_df."""
    df = pd.DataFrame({
        "x": [1, 5, 6, 7, 8, -1, -10, -19, -20, 10],
        "y": range(10)})

    agg_lag_info = build_agg_lag_df(
        value_col="x",
        df=df,
        orders_list=[[1, 2, 5], [1, 3, 8], [2, 3, 4]],
        interval_list=[(1, 5), (1, 8)],
        agg_func=np.mean,
        agg_name="avglag")

    agg_lag_df = agg_lag_info["agg_lag_df"]

    assert list(agg_lag_df.columns) == [
        "x_avglag_1_2_5",
        "x_avglag_1_3_8",
        "x_avglag_2_3_4",
        "x_avglag_1_to_5",
        "x_avglag_1_to_8"], \
        "aggregated lag df does not have the correct names"
    assert agg_lag_df["x_avglag_1_2_5"].values[2].round(0) == 3.0, (
        "aggregated lags are not correct")
    assert agg_lag_df["x_avglag_1_to_8"].values[7].round(1) == 2.3, (
        "aggregated lags are not correct")

    # agg_func "mean" produces the same result
    assert_equal(agg_lag_df, build_agg_lag_df(
        value_col="x",
        df=df,
        orders_list=[[1, 2, 5], [1, 3, 8], [2, 3, 4]],
        interval_list=[(1, 5), (1, 8)],
        agg_func="mean",
        agg_name="avglag")["agg_lag_df"])

    # check for Exception being raised for repeated orders
    with pytest.raises(
            Exception,
            match="a list of orders in orders_list contains a duplicate element"):
        build_agg_lag_df(
            df=df,
            value_col="x",
            orders_list=[[1, 2, 2], [1, 3, 8], [2, 3, 4]],
            interval_list=[(1, 5), (1, 8)],
            agg_func=np.mean,
            agg_name="avglag")

    # check for Exception being raised for interval not being on length 2
    with pytest.raises(
            Exception,
            match="interval must be a tuple of length 2"):
        build_agg_lag_df(
            df=df,
            value_col="x",
            orders_list=[[1, 2, 3], [1, 3, 8], [2, 3, 4]],
            interval_list=[(1, 5), (1, 8, 9)],
            agg_func=np.mean,
            agg_name="avglag")

    # check for Exception being raised for interval[0] <= interval[1]
    # for each interval in interval_list
    with pytest.raises(
            Exception,
            match=r"we must have interval\[0\] <= interval\[1\], for each interval in interval_list"):
        build_agg_lag_df(
            df=df,
            value_col="x",
            orders_list=[[1, 2, 3], [1, 3, 8], [2, 3, 4]],
            interval_list=[(1, 5), (8, 1)],
            agg_func=np.mean,
            agg_name="avglag")


def test_build_agg_lag_df_col_names_only():
    """Testing build_agg_lag_df for the case where input df is not passed and
        only col_names are generated"""

    agg_lag_info = build_agg_lag_df(
        value_col="x",
        df=None,
        orders_list=[[1, 2, 5], [1, 3, 8], [2, 3, 4]],
        interval_list=[(1, 5), (1, 8)],
        agg_func=np.mean,
        agg_name="avglag")

    col_names = agg_lag_info["col_names"]

    assert col_names == [
        "x_avglag_1_2_5",
        "x_avglag_1_3_8",
        "x_avglag_2_3_4",
        "x_avglag_1_to_5",
        "x_avglag_1_to_8"], \
        "aggregated lag df does not have the correct names"

    assert agg_lag_info["agg_lag_df"] is None, (
        "returned agg_lag_df must be None")


def test_build_agg_lag_df_exception():
    df = pd.DataFrame({"x": range(10), "y": range(100, 110)})
    with pytest.raises(
            ValueError,
            match="at least one of 'orders_list' or 'interval_list' must be provided"):
        build_agg_lag_df(
            value_col="x",
            df=df,
            orders_list=None,
            interval_list=None)


def test_build_autoreg_df():
    """Testing build_autoreg_df generic use case with no data filling."""
    df = pd.DataFrame({
        "x": [1, 5, 6, 7, 8, -1, -10, -19, -20, 10],
        "y": range(10)})

    autoreg_info = build_autoreg_df(
        value_col="x",
        lag_dict={"orders": [1, 2, 5]},
        agg_lag_dict={
            "orders_list": [[1, 2, 5], [1, 3, 8], [2, 3, 4]],
            "interval_list": [(1, 5), (1, 8)]},
        series_na_fill_func=None)  # no filling of NAs

    build_lags_func = autoreg_info["build_lags_func"]
    lag_col_names = autoreg_info["lag_col_names"]
    agg_lag_col_names = autoreg_info["agg_lag_col_names"]
    max_order = autoreg_info["max_order"]
    min_order = autoreg_info["min_order"]

    lag_df_info = build_lags_func(df)
    lag_df = lag_df_info["lag_df"]
    agg_lag_df = lag_df_info["agg_lag_df"]

    assert max_order == 8, (
        "returned max_order should be 8 for the given input")

    assert min_order == 1, (
        "returned min_order should be 8 for the given input")

    assert list(lag_df.columns) == lag_col_names

    assert lag_col_names == ["x_lag1", "x_lag2", "x_lag5"], \
        "The expected column names were not found in lags dataframe (lag_df)"

    assert pd.isnull(lag_df).iloc[0, 0], (
        "lag value is not correct")
    assert pd.isnull(lag_df).iloc[0, 1], (
        "lag value is not correct")
    assert pd.isnull(lag_df).iloc[1, 1], (
        "lag value is not correct")
    assert lag_df["x_lag1"].values[2].round(0) == 5.0, (
        "lag value is not correct")
    assert lag_df["x_lag2"].values[7].round(0) == -1.0, (
        "lag value is not correct")
    assert lag_df["x_lag5"].values[8].round(0) == 7.0, (
        "lag value is not correct")

    assert list(agg_lag_df.columns) == agg_lag_col_names

    assert agg_lag_col_names == [
        "x_avglag_1_2_5",
        "x_avglag_1_3_8",
        "x_avglag_2_3_4",
        "x_avglag_1_to_5",
        "x_avglag_1_to_8"], \
        "aggregated lag df does not have the correct names"
    assert agg_lag_df["x_avglag_1_2_5"].values[2].round(0) == 3.0, (
        "aggregated lags are not correct")
    assert agg_lag_df["x_avglag_1_to_8"].values[7].round(1) == 2.3, (
        "aggregated lags are not correct")


def test_build_autoreg_df_with_filling_na():
    """Testing build_autoreg_df use case with filling missing data."""
    df = pd.DataFrame({
        "x": [1, 5, 6, 7, 8, -1, -10, -19, -20, 10],
        "y": range(10)})

    autoreg_info = build_autoreg_df(
        value_col="x",
        lag_dict={"orders": [1, 2, 5]},
        agg_lag_dict={
            "orders_list": [[1, 2, 5], [1, 3, 8], [2, 3, 4]],
            "interval_list": [(1, 5), (1, 8)]},
        series_na_fill_func=lambda s: s.bfill().ffill())  # filling NULLs with simle backward and then forward method

    build_lags_func = autoreg_info["build_lags_func"]
    lag_col_names = autoreg_info["lag_col_names"]
    agg_lag_col_names = autoreg_info["agg_lag_col_names"]
    max_order = autoreg_info["max_order"]
    min_order = autoreg_info["min_order"]

    lag_df_info = build_lags_func(df)
    lag_df = lag_df_info["lag_df"]
    agg_lag_df = lag_df_info["agg_lag_df"]

    assert max_order == 8, (
        "returned max_order should be 8 for the given input")

    assert min_order == 1, (
        "returned min_order should be 8 for the given input")

    assert list(lag_df.columns) == lag_col_names

    assert lag_col_names == ["x_lag1", "x_lag2", "x_lag5"], \
        "The expected column names were not found in lags dataframe (lag_df)"

    assert lag_df["x_lag1"].values[0].round(1) == 1.0, (
        "lag value is not correct")
    assert lag_df["x_lag1"].values[1].round(1) == 1.0, (
        "lag value is not correct")
    assert lag_df["x_lag1"].values[2].round(0) == 5.0, (
        "lag value is not correct")
    assert lag_df["x_lag2"].values[7].round(0) == -1.0, (
        "lag value is not correct")
    assert lag_df["x_lag5"].values[8].round(0) == 7.0, (
        "lag value is not correct")

    assert list(agg_lag_df.columns) == agg_lag_col_names

    assert agg_lag_col_names == [
        "x_avglag_1_2_5",
        "x_avglag_1_3_8",
        "x_avglag_2_3_4",
        "x_avglag_1_to_5",
        "x_avglag_1_to_8"], \
        "aggregated lag df does not have the correct names"
    assert agg_lag_df["x_avglag_1_2_5"].values[2].round(1) == 2.3, (
        "aggregated lags are not correct")
    assert agg_lag_df["x_avglag_1_to_8"].values[7].round(1) == 2.1, (
        "aggregated lags are not correct")


def test_build_autoreg_df_pass_past_df():
    """Testing build_autoreg_df with
        past_df being passed at predict time
    """
    df = pd.DataFrame({
        "x": [1.0, 5.0, 6.0, 7.0, 8.0, -1.0, -10.0, -19.0, -20.0, 10.0],
        "y": range(10)})

    past_df = pd.DataFrame({
        "x": [4.0, 5.0, 6.0, 7.0, 8.0, -1.0, -10.0, -11.0, -10.0, 15.0],
        "y": range(10)})

    autoreg_info = build_autoreg_df(
        value_col="x",
        lag_dict={"orders": [1, 2, 5]},
        agg_lag_dict={
            "orders_list": [[1, 2, 5], [1, 3, 8], [2, 3, 4]],
            "interval_list": [(1, 5), (1, 8)]},
        series_na_fill_func=None)  # no filling of NAs

    build_lags_func = autoreg_info["build_lags_func"]

    # the following line should run without issues (no exceptions)
    lag_df_info = build_lags_func(df=df, past_df=past_df)
    lag_df = lag_df_info["lag_df"]
    agg_lag_df = lag_df_info["agg_lag_df"]

    assert list(lag_df.columns) == ["x_lag1", "x_lag2", "x_lag5"], \
        "The expected column names were not found in lags dataframe (lag_df)"
    assert list(agg_lag_df.columns) == [
        "x_avglag_1_2_5",
        "x_avglag_1_3_8",
        "x_avglag_2_3_4",
        "x_avglag_1_to_5",
        "x_avglag_1_to_8"], \
        "aggregated lag df does not have the correct names"
    assert lag_df["x_lag1"].values[0].round(0) == 15.0, (
        "lag value is not correct")
    assert agg_lag_df["x_avglag_1_to_8"].values[7].round(1) == 3.9, (
        "aggregated lags are not correct")


def test_build_autoreg_df_exception():
    """Testing build_autoreg_df with
        past_df not having the correct value_col
        we expect exception due the missing value_col"""
    df = pd.DataFrame({
        "x": [1.0, 5.0, 6.0, 7.0, 8.0, -1.0, -10.0, -19.0, -20.0, 10.0],
        "y": range(10)})

    autoreg_info = build_autoreg_df(
        value_col="x",
        lag_dict={"orders": [1, 2, 5]},
        agg_lag_dict={
            "orders_list": [[1, 2, 5], [1, 3, 8], [2, 3, 4]],
            "interval_list": [(1, 5), (1, 8)]},
        series_na_fill_func=None)  # no filling of NAs

    build_lags_func = autoreg_info["build_lags_func"]

    # past_df with correct column names
    past_df = df.copy()
    # changing column name in past_df to something ("z")
    # other than the expected ("x")
    past_df.rename(columns={"x": "z"}, inplace=True)
    with pytest.raises(
            ValueError,
            match="x must appear in past_df if past_df is not None"):
        build_lags_func(df=df, past_df=past_df)


def test_build_autoreg_df_multi():
    """Testing build_autoreg_df_multi"""
    df = pd.DataFrame({
        "y": range(10),
        "x1": range(100, 110),
        "x2": [True, False, False, False, True, True, False, True, False, True]})

    value_col = "y"
    lagged_regressor_value_cols = ["x1", "x2"]
    value_lag_info_dict = {}

    value_lag_info_dict[value_col] = {
        "lag_dict": {"orders": [1, 2, 7]},
        "agg_lag_dict": {
            "orders_list": [[7, 7*2, 7*3]],
            "interval_list": [(1, 7), (8, 7*2)]},
        "series_na_fill_func": lambda s: s.bfill().ffill()}

    # Assigns the same lag structure to all
    for col in lagged_regressor_value_cols:
        value_lag_info_dict[col] = {
            "lag_dict": {"orders": [1, 2, 7]},
            "agg_lag_dict": {
                "orders_list": [[7, 7*2, 7*3]],
                "interval_list": [(1, 7)]},
            "series_na_fill_func": lambda s: s.bfill().ffill()}

    autoreg_info = build_autoreg_df_multi(
        value_lag_info_dict=value_lag_info_dict,
        series_na_fill_func=lambda s: s.bfill().ffill())

    autoreg_orig_col_names = autoreg_info["autoreg_orig_col_names"]
    autoreg_col_names = autoreg_info["autoreg_col_names"]
    autoreg_func = autoreg_info["autoreg_func"]
    autoreg_df = autoreg_func(df=df)

    expected_autoreg_orig_col_names = ["y", "x1", "x2"]
    expected_col_names = [
        "y_lag1",
        "y_lag2",
        "y_lag7",
        "y_avglag_7_14_21",
        "y_avglag_1_to_7",
        "y_avglag_8_to_14",
        "x1_lag1",
        "x1_lag2",
        "x1_lag7",
        "x1_avglag_7_14_21",
        "x1_avglag_1_to_7",
        "x2_lag1",
        "x2_lag2",
        "x2_lag7",
        "x2_avglag_7_14_21",
        "x2_avglag_1_to_7"]

    # Checking data types
    expected_dtypes = ['float'] * 11 + ['bool'] * 3 + ['float'] * 2
    assert (autoreg_df.dtypes != expected_dtypes).sum() == 0

    # Checking min and max lag order
    assert autoreg_info["min_order"] == 1
    assert autoreg_info["max_order"] == 21

    # Checking column names
    assert autoreg_orig_col_names == expected_autoreg_orig_col_names
    assert autoreg_col_names == expected_col_names
    assert list(autoreg_df.columns) == expected_col_names

    # Checking values for `y_avglag_1_to_7`
    # Note: y is: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    # Therefore for example the last value for `y_avglag_1_to_7` will be
    # (8 + 7 + 6 + 5 + 4 + 3 + 2) / 7 = 5.0 (as reflected in the `assert` below)
    # The first two values are expected to be zero as past data is imputed
    # with zeros
    # The third value is expectd to be (0 + 0 + 0 + 0 + 0 + 0 + 1) = 0.14
    # which will be rounded to 0.1 as reflected below
    expected_values1 = [0., 0., 0.1, 0.4, 0.9, 1.4, 2.1, 3., 4., 5.]
    assert np.allclose(
        autoreg_df["y_avglag_1_to_7"].round(1).values,
        expected_values1)

    # Checking values for `x2_avglag_1_to_7`
    # Note: x2 is: True, False, False, False, True, True, False, True, False, True
    # Therefore for example the last value for `x2_avglag_1_to_7` will be
    # 3 / 7 = 0.43, which is rounded to 0.4 as reflected below
    # The first two values are expected to be 1 as past data is imputed with
    # the first element in x2, which is True
    # The third value is expectd to be (1 + 1 + 1 + 1 + 1 + 1 + 0) / 7 = 0.85
    # which will be rounded to 0.9 as reflected below
    expected_values2 = [1.0, 1.0, 0.9, 0.7, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4]
    assert np.allclose(
        round(autoreg_df["x2_avglag_1_to_7"], 1),
        expected_values2)

    # Checks values for `x2_lag2`
    expected_values3 = [True, True, True, False, False, False, True, True, False, True]
    assert np.allclose(
        autoreg_df["x2_lag2"],
        expected_values3)

    # Tests with a given `past_df`
    past_df = pd.DataFrame({
        "y": [-10]*10,
        "x1": [-100]*10,
        "x2": [False]*10})

    autoreg_df = autoreg_func(
        df=df,
        past_df=past_df)

    assert np.allclose(
        autoreg_df["y_lag1"],
        [-10] + list(range(0, 9)))

    assert np.allclose(
        autoreg_df["x2_lag7"],
        [False]*7 + [True, False, False])
