"""
Test for null_transformer.py
"""
import numpy as np
import pandas as pd
import pytest
from pandas.util.testing import assert_equal
from sklearn.exceptions import NotFittedError
from testfixtures import LogCapture

from greykite.common.constants import LOGGER_NAME
from greykite.sklearn.transform.null_transformer import NullTransformer


@pytest.fixture
def data():
    """Generates test dataframe for null imputation"""
    return pd.DataFrame({
        "a": (0.0, np.nan, -1.0, 1.0),
        "b": (np.nan, 2.0, np.nan, np.nan),
        "c": (2.0, 3.0, np.nan, 9.0),
        "d": (np.nan, 4.0, -4.0, 16.0)
    })


def test_null_transformer1(data):
    """Checks impute_algorithm='interpolate'"""
    with pytest.raises(ValueError, match="method 'ffill' is not allowed"):
        NullTransformer(
            impute_algorithm="interpolate",
            impute_params=dict(method="ffill"),
            impute_all=False)

    # impute_all=False leaves NaNs
    impute_params = dict(method="linear", limit_direction="forward")
    null_transform = NullTransformer(
        max_frac=0.8,
        impute_algorithm="interpolate",
        impute_params=impute_params,
        impute_all=False)
    # init does not modify parameters
    assert null_transform.max_frac == 0.8
    assert null_transform.impute_algorithm == "interpolate"
    assert null_transform.impute_params == impute_params
    assert null_transform.impute_all is False
    null_transform.fit(data)
    assert null_transform.impute_params == {"axis": 0, **impute_params}
    result = null_transform.transform(data)
    expected = pd.DataFrame({
        "a": (0.0, -0.5, -1.0, 1.0),
        "b": (np.nan, 2.0, 2.0, 2.0),
        "c": (2.0, 3.0, 6.0, 9.0),
        "d": (np.nan, 4.0, -4.0, 16.0),
    })
    assert_equal(result, expected)
    assert null_transform.missing_info is None

    # impute_all=True fills NaNs
    null_transform = NullTransformer(
        impute_algorithm="interpolate",
        impute_params=dict(method="linear", limit_direction="forward"),
        impute_all=True)
    result = null_transform.fit_transform(data)
    expected = pd.DataFrame({
        "a": (0.0, -0.5, -1.0, 1.0),
        "b": (2.0, 2.0, 2.0, 2.0),
        "c": (2.0, 3.0, 6.0, 9.0),
        "d": (4.0, 4.0, -4.0, 16.0),
    })
    assert_equal(result, expected)


def test_null_transformer2(data):
    """Checks impute_algorithm='ts_interpolate'"""
    null_transform = NullTransformer(
        impute_algorithm="ts_interpolate",
        impute_all=False)
    null_transform.fit(data)
    assert null_transform.impute_params == dict(
        orders=[7, 14, 21],
        agg_func=np.mean,
        iter_num=5)
    result = null_transform.transform(data)
    # `orders` is too large for this dataset, nothing is imputed
    assert_equal(result, data)

    # two iterations
    null_transform = NullTransformer(
        impute_algorithm="ts_interpolate",
        impute_params=dict(orders=[1], agg_func=np.nanmean, iter_num=2),
        impute_all=False)
    result = null_transform.fit_transform(data)
    expected = pd.DataFrame({
        "a": (0.0, 0.0, -1.0, 1.0),
        "b": (np.nan, 2.0, 2.0, 2.0),
        "c": (2.0, 3.0, 3.0, 9.0),
        "d": (np.nan, 4.0, -4.0, 16.0),
    })
    assert_equal(result, expected)
    assert null_transform.missing_info == {
        "a": {"initial_missing_num": 1, "final_missing_num": 0},
        "b": {"initial_missing_num": 3, "final_missing_num": 1},
        "c": {"initial_missing_num": 1, "final_missing_num": 0},
        "d": {"initial_missing_num": 1, "final_missing_num": 1},
    }

    # impute_all=True
    null_transform = NullTransformer(
        impute_algorithm="ts_interpolate",
        impute_params=dict(orders=[1], agg_func=np.nanmean, iter_num=2),
        impute_all=True)
    result = null_transform.fit_transform(data)
    expected = pd.DataFrame({
        "a": (0.0, 0.0, -1.0, 1.0),
        "b": (2.0, 2.0, 2.0, 2.0),
        "c": (2.0, 3.0, 3.0, 9.0),
        "d": (4.0, 4.0, -4.0, 16.0),
    })
    assert_equal(result, expected)
    # `final_missing_num` are filled in by the second pass.
    # The counts reflect the first pass.
    assert null_transform.missing_info == {
        "a": {"initial_missing_num": 1, "final_missing_num": 0},
        "b": {"initial_missing_num": 3, "final_missing_num": 1},
        "c": {"initial_missing_num": 1, "final_missing_num": 0},
        "d": {"initial_missing_num": 1, "final_missing_num": 1},
    }


def test_null_transformer3(data):
    """Checks impute_algorithm=None"""
    # impute_algorithm=None is the default
    null_transform = NullTransformer(
        impute_all=False)
    null_transform.fit(data)
    assert null_transform.impute_params is None
    result = null_transform.transform(data)
    assert_equal(result, data)

    # impute_all=True has no effect
    null_transform = NullTransformer(
        impute_algorithm=None,
        impute_all=True)
    null_transform.fit(data)
    assert null_transform.impute_params is None
    result = null_transform.transform(data)
    assert_equal(result, data)


def test_null_transformer_error(data):
    """Checks warnings and exceptions"""
    with pytest.warns(RuntimeWarning) as record:
        null_transform = NullTransformer(max_frac=0.10)
        null_transform.fit_transform(data)
        assert "Input data has many null values" in record[0].message.args[0]

    null_transform = NullTransformer(max_frac=0.70)
    with LogCapture(LOGGER_NAME) as log_capture:
        null_transform.fit_transform(data)
        log_capture.check(
            (LOGGER_NAME, "INFO", "Missing data detected: 37.50% of all input values are null. "
                                  "(If future external regressor(s) are used, some missing values in "
                                  "`value_col` are expected.)"
             )
        )

    null_transform = NullTransformer()
    with pytest.raises(NotFittedError, match="This instance is not fitted yet"):
        null_transform.transform(data)

    null_transform = NullTransformer(impute_algorithm="unknown")
    with pytest.raises(ValueError, match="`impute_algorithm` 'unknown' is not recognized"):
        null_transform.fit_transform(data)
