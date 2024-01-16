"""
Test for difference_based_outlier_transformer.py
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from testfixtures import LogCapture

from greykite.common.constants import LOGGER_NAME
from greykite.common.python_utils import assert_equal
from greykite.sklearn.transform.difference_based_outlier_transformer import DifferenceBasedOutlierTransformer


@pytest.fixture
def data():
    """Generates test dataframe for outlier detection"""
    np.random.seed(100)
    df = pd.DataFrame({
        "a": np.repeat(1.0, 100),
        "b": np.arange(0.0, 100.0, 1.0) + np.random.normal(size=100, loc=0, scale=0.1),
        "c": np.tile([1.0, 2.0, 3.0, 4.0], 25),
        "d": np.repeat(1.0, 100),
    })
    df.loc[2, "b"] = 100.0
    df.loc[6, "d"] = 100.0
    return df


def test_difference_based_outlier_transformer(data):
    """Checks if outliers are properly replaced"""
    transformer = DifferenceBasedOutlierTransformer(
        method="z_score",
        score_type="difference",
        params=dict(
            agg_func=np.nanmean,
            lag_orders=[-1, 1],
            z_cutoff=3.5,
            max_outlier_percent=5.0
        )
    )
    # init does not modify parameters.
    assert transformer.method == "z_score"
    assert transformer.score_type == "difference"
    assert transformer.score is None
    transformer.fit(data)
    assert transformer.agg_func == np.nanmean
    assert transformer.lag_orders == [-1, 1]
    assert transformer.z_cutoff == 3.5
    assert transformer.max_outlier_percent == 5.0
    assert transformer.score is not None
    # `transform` removes outliers based on `transformer.scores`.
    with LogCapture(LOGGER_NAME) as log_capture:
        result = transformer.transform(data)
        expected = data.copy()
        expected.loc[[1, 2, 3], "b"] = np.nan
        expected.loc[[5, 6, 7], "d"] = np.nan
        assert_equal(result, expected)
        log_capture.check(
            (LOGGER_NAME, "INFO", "Detected 6 outlier(s)."))

    transformer = DifferenceBasedOutlierTransformer(
        method="tukey",
        score_type="ratio",
        params=dict(
            agg_func=np.nanmean,
            lag_orders=[-1, 1],
            tukey_cutoff=3.5,
            max_outlier_percent=5.0
        )
    )
    # init does not modify parameters.
    assert transformer.method == "tukey"
    assert transformer.score_type == "ratio"
    assert transformer.score is None
    transformer.fit(data)
    assert transformer.agg_func == np.nanmean
    assert transformer.lag_orders == [-1, 1]
    assert transformer.tukey_cutoff == 3.5
    assert transformer.max_outlier_percent == 5.0
    assert transformer.score is not None
    # `transform` removes outliers based on `transformer.scores`.
    with LogCapture(LOGGER_NAME) as log_capture:
        result = transformer.transform(data)
        expected = data.copy()
        expected.loc[[0, 1, 2, 3, 4], "b"] = np.nan
        assert_equal(result, expected)
        log_capture.check(
            (LOGGER_NAME, "INFO", "Detected 5 outlier(s)."))

    transformer = DifferenceBasedOutlierTransformer(
        method="neither_z_score_nor_tukey")
    with pytest.raises(NotImplementedError, match="is an invalid 'method'"):
        transformer.fit(data)

    transformer = DifferenceBasedOutlierTransformer(
        method="z_score",
        score_type="neither_difference_nor_ratio",
        params=dict(
            agg_func=np.nanmean,
            lag_orders=[-1, 1],
            z_cutoff=3.0,
            max_outlier_percent=5.0
        )
    )
    with pytest.raises(NotImplementedError, match="is an invalid 'score_type'"):
        transformer.fit(data)

    transformer = DifferenceBasedOutlierTransformer(
        method="z_score",
        score_type="difference",
        params=dict(
            agg_func=np.nanmean,
            lag_orders=[-1, 1],
            z_cutoff=3.0,
            max_outlier_percent=5.0
        )
    )
    with pytest.raises(NotFittedError, match="This instance is not fitted yet."):
        transformer.transform(data)
