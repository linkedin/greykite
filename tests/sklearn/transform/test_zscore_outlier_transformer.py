"""
Test for zscore_outlier_transformer.py
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from testfixtures import LogCapture

from greykite.common.constants import LOGGER_NAME
from greykite.common.python_utils import assert_equal
from greykite.sklearn.transform.zscore_outlier_transformer import ZscoreOutlierTransformer


@pytest.fixture
def data():
    """Generates test dataframe for outlier detection"""
    df = pd.DataFrame({
        "a": np.repeat(1.0, 100),
        "b": np.arange(0.0, 100.0, 1.0),
        "c": np.tile([1.0, 2.0, 3.0, 4.0], 25),
        "d": np.repeat(1.0, 100),
    })
    df.loc[2, "b"] = 100.0  # z-score 1.74
    df.loc[4, "c"] = 6.0  # z-score 2.95
    df.loc[6, "d"] = 100.0  # z-score 9.90
    return df


def test_zscore_outlier_transformer1(data):
    """Checks if outliers are properly replaced"""
    # z_cutoff=None (default)
    zscore_transform = ZscoreOutlierTransformer(
        use_fit_baseline=True)
    # init does not modify parameters
    assert zscore_transform.z_cutoff is None
    assert zscore_transform.use_fit_baseline is True
    assert zscore_transform.mean is None
    assert zscore_transform.std is None
    assert zscore_transform._is_fitted is None
    # doesn't need to be fit
    result = zscore_transform.transform(data)
    assert_equal(result, data)
    result = zscore_transform.fit_transform(data)
    assert_equal(result, data)
    assert zscore_transform.mean is None
    assert zscore_transform.std is None

    # z_cutoff=3.0, doesn't need to be fit
    with LogCapture(LOGGER_NAME) as log_capture:
        zscore_transform = ZscoreOutlierTransformer(z_cutoff=3.0)
        assert zscore_transform.z_cutoff == 3.0
        result = zscore_transform.transform(data)
        assert zscore_transform.mean is None
        assert zscore_transform.std is None
        expected = data.copy()
        expected.loc[6, "d"] = np.nan
        assert_equal(result, expected)
        log_capture.check(
            (LOGGER_NAME, "INFO", "Detected 1 outlier(s)."))
        zscore_transform.fit_transform(data)
        assert zscore_transform.mean is None
        assert zscore_transform.std is None

    # z_cutoff=2.0, requires fit
    zscore_transform = ZscoreOutlierTransformer(
        z_cutoff=2.0,
        use_fit_baseline=True)
    with pytest.raises(NotFittedError, match="This instance is not fitted yet"):
        zscore_transform.transform(data)
    with LogCapture(LOGGER_NAME) as log_capture:
        result = zscore_transform.fit_transform(data)
        expected = data.copy()
        expected.loc[4, "c"] = np.nan
        expected.loc[6, "d"] = np.nan
        assert_equal(result, expected)
        log_capture.check(
            (LOGGER_NAME, "INFO", "Detected 2 outlier(s)."))
        # uses fitted mean and std to calculate z-scores
        test_data = data + 1e5  # all values are outliers
        result = zscore_transform.transform(test_data)
        assert result.isna().all().all()

    # use_fit_baseline=False
    zscore_transform = ZscoreOutlierTransformer(
        z_cutoff=2.0,
        use_fit_baseline=False)
    result = zscore_transform.transform(test_data)
    expected = test_data.copy()
    expected.loc[4, "c"] = np.nan
    expected.loc[6, "d"] = np.nan
    assert_equal(result, expected)
