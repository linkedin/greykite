import numpy as np
import pandas as pd
from pandas.util.testing import assert_equal

from greykite.common.features.timeseries_impute import impute_with_lags
from greykite.common.features.timeseries_impute import impute_with_lags_multi


def test_impute_with_lags():
    """Testing `impute_with_lags`"""
    # below we construct dataframes which we envision as daily data
    # therefore lags of multiples of 7 is appropriate
    df = pd.DataFrame({"y": list(range(70)) + [np.nan]*3})

    # impute with the same day of week of past week: `orders = [7]`
    impute_info = impute_with_lags(
        df=df,
        value_col="y",
        orders=[7],
        agg_func="mean",
        iter_num=1)

    assert impute_info["initial_missing_num"] == 3
    assert impute_info["final_missing_num"] == 0
    imputed_df = impute_info["df"]

    assert list(imputed_df["y"].values) == (
        list(range(70)) + [63, 64, 65])

    # impute with three recent similar values: `orders = [7, 14, 21]`
    df = pd.DataFrame({"y": list(range(70)) + [np.nan, np.nan, np.nan]})
    impute_info = impute_with_lags(
        df=df,
        value_col="y",
        orders=[7, 14, 21],
        agg_func="mean",
        iter_num=1)

    assert impute_info["initial_missing_num"] == 3
    assert impute_info["final_missing_num"] == 0
    imputed_df = impute_info["df"]

    assert list(imputed_df["y"].values) == (
        list(range(70)) + [56, 57, 58])

    # same test with larger ``iter_num``
    df = pd.DataFrame({"y": list(range(70)) + [np.nan, np.nan, np.nan]})
    impute_info = impute_with_lags(
        df=df,
        value_col="y",
        orders=[7, 14, 21],
        agg_func="mean",
        iter_num=10)  # large ``iter_num``

    assert impute_info["initial_missing_num"] == 3
    assert impute_info["final_missing_num"] == 0
    imputed_df = impute_info["df"]

    assert list(imputed_df["y"].values) == (
        list(range(70)) + [56, 57, 58])

    # NAs are ignored when a string `agg_func` is passed
    df = pd.DataFrame({"y": [1, 2, 3, np.nan, 5, 6, np.nan]})
    impute_info = impute_with_lags(
        df=df,
        value_col="y",
        orders=[2, 3, 4],
        agg_func="mean",
        iter_num=1)
    assert impute_info["initial_missing_num"] == 2
    assert impute_info["final_missing_num"] == 0
    imputed_df = impute_info["df"]
    # since the lag orders are ``[2, 3, 4]``
    # we expect the last value of the series (which is missing)
    # to be imputed to the average of ``[3, np.nan, 5]``
    # after removing the ``np.nan``. That is average of ``[5, 3]`` which is 4.
    assert list(imputed_df["y"].values) == (
        [1, 2, 3, 1.5, 5, 6, 4])

    # NAs are ignored when a function `agg_func` is passed
    df = pd.DataFrame({"y": [1, 2, 3, np.nan, 5, 6, np.nan]})
    impute_info = impute_with_lags(
        df=df,
        value_col="y",
        orders=[2, 3, 4],
        agg_func=np.mean,
        iter_num=1)
    assert impute_info["initial_missing_num"] == 2
    assert impute_info["final_missing_num"] == 0
    imputed_df = impute_info["df"]
    assert list(imputed_df["y"].values) == (
        [1, 2, 3, 1.5, 5, 6, 4])

    # The case where one iteration will not impute all
    # but two iterations will
    df = pd.DataFrame({"y": list(range(63)) + [np.nan]*10})

    # One iteration
    impute_info = impute_with_lags(
        df=df,
        value_col="y",
        orders=[7],
        agg_func=np.mean,
        iter_num=1)

    assert impute_info["initial_missing_num"] == 10
    assert impute_info["final_missing_num"] == 3
    imputed_df = impute_info["df"]

    # Three values are still not imputed
    obtained_array = np.array(imputed_df["y"])
    expected_array = np.array(
        list(range(63)) + list(range(56, 63)) + [np.nan]*3)

    np.testing.assert_array_equal(obtained_array, expected_array)

    # Two iterations
    impute_info = impute_with_lags(
        df=df,
        value_col="y",
        orders=[7],
        agg_func=np.mean,
        iter_num=2)

    assert impute_info["initial_missing_num"] == 10
    assert impute_info["final_missing_num"] == 0
    imputed_df = impute_info["df"]

    obtained_array = np.array(imputed_df["y"])
    # All values must be imputed
    expected_array = np.array(
        list(range(63)) + list(range(56, 63)) + [56, 57, 58])

    np.testing.assert_array_equal(obtained_array, expected_array)


def test_impute_with_lags_multi():
    """Tests `impute_with_lags_multi`"""
    df = pd.DataFrame({
        "a": (0.0, np.nan, -1.0, 1.0),
        "b": (np.nan, 2.0, np.nan, np.nan),
        "c": (2.0, 3.0, np.nan, 9.0),
        "d": (np.nan, 4.0, -4.0, 16.0)
    })

    # tests `cols = None`
    impute_info = impute_with_lags_multi(
        df=df,
        orders=[1],
        agg_func=np.mean,
        iter_num=1,
        cols=None
    )
    expected = pd.DataFrame({
        "a": (0.0, 0.0, -1.0, 1.0),
        "b": (np.nan, 2.0, 2.0, np.nan),
        "c": (2.0, 3.0, 3.0, 9.0),
        "d": (np.nan, 4.0, -4.0, 16.0)
    })
    assert_equal(impute_info["df"], expected)
    assert impute_info["missing_info"] == {
        "a": {"initial_missing_num": 1, "final_missing_num": 0},
        "b": {"initial_missing_num": 3, "final_missing_num": 2},
        "c": {"initial_missing_num": 1, "final_missing_num": 0},
        "d": {"initial_missing_num": 1, "final_missing_num": 1},
    }

    # tests `cols = list`
    impute_info = impute_with_lags_multi(
        df=df,
        orders=[1],
        agg_func=np.mean,
        iter_num=2,
        cols=["b", "c", "d"]
    )
    expected = pd.DataFrame({
        "a": (0.0, np.nan, -1.0, 1.0),  # not imputed
        "b": (np.nan, 2.0, 2.0, 2.0),   # imputed in second iteration
        "c": (2.0, 3.0, 3.0, 9.0),
        "d": (np.nan, 4.0, -4.0, 16.0)
    })
    assert_equal(impute_info["df"], expected)
    assert impute_info["missing_info"] == {
        "b": {"initial_missing_num": 3, "final_missing_num": 1},
        "c": {"initial_missing_num": 1, "final_missing_num": 0},
        "d": {"initial_missing_num": 1, "final_missing_num": 1},
    }
